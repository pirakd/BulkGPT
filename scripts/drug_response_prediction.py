"""Drug response prediction replicating the BulkFormer paper experiment.

Cell-line representations come from a frozen NanoBulkFM (max-pool over gene embeddings).
Drug representations come from pretrained KPGT embeddings.
Concatenated features are fed to a two-layer MLP that predicts LN_IC50.
Performance is evaluated with Pearson (PCC) and Spearman (SCC) correlation.

Usage:
    # Run all 10 folds
    uv run python scripts/drug_response_prediction.py

    # Run a single fold (0-indexed) for fast iteration
    uv run python scripts/drug_response_prediction.py --fold 0

    # Use a different checkpoint
    uv run python scripts/drug_response_prediction.py --checkpoint out/train_nano_bulk_fm/.../ckpt.pt

    # Load drug-response data from the Hugging Face dataset instead of local files
    uv run python scripts/drug_response_prediction.py --data-source hf
"""
import argparse
import sys
from pathlib import Path

import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from model import NanoBulkFM, NanoBulkFMConfig
from utils import create_output_folder


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


DEFAULT_CKPT = REPO / "out/train_nano_bulk_fm/20260704_150450/ckpt.pt"
EXPR_CSV = REPO / "data/drug_response/drug_response_expr_data.csv"
IC50_CSV = REPO / "data/drug_response/drug_response_prediction_IC50.csv"
DRUG_EMB_NPZ = REPO / "data/drug_response/drug_embeddings.npz"
ESM_EMBEDDINGS_PATH = REPO / "data/genes/build_gene_protein_embeddings/20260609_114706/protein_coding_gene_esm2_embeddings.parquet"

HF_REPO_ID = "dpirak/nanobulkFM"
HF_EXPR_FILENAME = "drug_response/drug_response_expr_data.csv"
HF_IC50_FILENAME = "drug_response/drug_response_prediction_IC50.csv"
HF_DRUG_EMB_FILENAME = "drug_response/drug_embeddings.npz"
HF_ESM_FILENAME = "protein_coding_gene_esm2_embeddings.parquet"

N_FOLDS = 5
BATCH_SIZE = 128
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
DROPOUT = 0.1


class PairTensors:
    """All (cell-line embedding, drug embedding, IC50) pairs, pre-stacked and
    resident on `device` as a single block.

    The model is tiny (a 2-layer MLP) relative to the dataset (~212k pairs),
    so per-sample Python-level dict lookups + a DataLoader (even with
    pre-pooled [D] cell vectors) leave the GPU mostly idle waiting on CPU-side
    batch assembly and host->device transfers. Since the whole dataset
    (~212k x (128 + 2304) floats ≈ 2 GB) comfortably fits in GPU memory,
    stacking it once into contiguous tensors and moving it to `device` a
    single time lets every epoch's shuffling/batching happen as pure GPU
    tensor indexing, with zero host<->device traffic and zero Python
    per-sample overhead during training.
    """

    def __init__(self, cell_emb_map: dict, drug_emb_map: dict, ic50_df: pd.DataFrame, device: str):
        cell_feats = np.stack([cell_emb_map[c] for c in ic50_df["ModelID"]])   # [N, D]
        drug_feats = np.stack([drug_emb_map[s] for s in ic50_df["smiles"]])    # [N, drug_dim]
        y = ic50_df["IC50"].to_numpy(dtype=np.float32)                        # [N]

        self.cell_feats = torch.from_numpy(cell_feats).to(device)
        self.drug_feats = torch.from_numpy(drug_feats).to(device)
        self.y = torch.from_numpy(y).to(device)

    def __len__(self):
        return self.y.shape[0]

    def batches(self, idx: torch.Tensor, batch_size: int, shuffle: bool):
        """Yield (cell_feats, drug_feats, y) batches for the given GPU-resident indices."""
        if shuffle:
            idx = idx[torch.randperm(len(idx), device=idx.device)]
        for i in range(0, len(idx), batch_size):
            b = idx[i:i + batch_size]
            yield self.cell_feats[b], self.drug_feats[b], self.y[b]


class DrugResponseMLP(nn.Module):
    def __init__(self, gene_dim: int, drug_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gene_dim + drug_dim, 256),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(256, 1),
        )

    def forward(self, cell_emb: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        # cell_emb: [B, D] (pre-pooled), drug_emb: [B, drug_dim]
        return self.net(torch.cat([cell_emb, drug_emb], dim=-1)).squeeze(-1)


def resolve_path(data_source: str, local_path: Path, hf_filename: str,
                  hf_repo_id: str = HF_REPO_ID, hf_revision: str | None = None,
                  hf_cache_dir: str | None = None) -> Path:
    if data_source == "local":
        return local_path
    if data_source != "hf":
        raise ValueError(f"Unknown data_source: {data_source}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub to load data from Hugging Face.") from exc

    return Path(hf_hub_download(
        repo_id=hf_repo_id,
        filename=hf_filename,
        repo_type="dataset",
        revision=hf_revision,
        cache_dir=hf_cache_dir,
    ))


def load_fm(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = NanoBulkFMConfig(**{k: v for k, v in ckpt["model_cfg"].items()
                               if k in NanoBulkFMConfig.__dataclass_fields__})
    # NanoBulkFM requires a correctly-shaped esm_gene_embeddings tensor to
    # register the buffer; the actual values are overwritten by load_state_dict.
    esm_gene_embeddings = None
    if cfg.use_esm_embeddings:
        esm_gene_embeddings = torch.zeros(cfg.n_genes, cfg.esm_embedding_dim)
    model = NanoBulkFM(cfg, device=device, esm_gene_embeddings=esm_gene_embeddings).to(device)
    # Strip the "_orig_mod." prefix left by torch.compile() during training.
    state_dict = {k.removeprefix("_orig_mod."): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state_dict)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    gene_ids = ckpt["gene_ids"]  # Ensembl gene IDs, in the model's input order
    return model, cfg, gene_ids


def load_gene_id_to_symbol(esm_embeddings_path: Path) -> dict:
    """Map Ensembl gene ID -> HGNC symbol, sourced from the same ESM embeddings
    table used to train NanoBulkFM (see configs/train_nano_bulk_fm.yaml)."""
    df = pd.read_parquet(esm_embeddings_path, columns=["ensembl_gene_id", "hgnc_symbol"])
    return dict(zip(df["ensembl_gene_id"], df["hgnc_symbol"]))


@torch.no_grad()
def extract_cell_embeddings(fm: NanoBulkFM, expr: np.ndarray, device: str, batch_size: int = 64) -> np.ndarray:
    """Extract per-gene embeddings from the final transformer layer → [N, G, D]."""
    embs = []
    x = torch.from_numpy(expr).float()
    for i in range(0, len(x), batch_size):
        batch = x[i:i + batch_size].to(device)
        _, h = fm(batch)          # h: [B, G, D]
        embs.append(h.cpu().numpy())
    return np.concatenate(embs, axis=0)  # [N, G, D]


def load_data(ckpt_path: Path, device: str, data_source: str = "local"):
    expr_csv_path = resolve_path(data_source, EXPR_CSV, HF_EXPR_FILENAME)
    ic50_csv_path = resolve_path(data_source, IC50_CSV, HF_IC50_FILENAME)
    drug_emb_npz_path = resolve_path(data_source, DRUG_EMB_NPZ, HF_DRUG_EMB_FILENAME)
    esm_path = resolve_path(data_source, ESM_EMBEDDINGS_PATH, HF_ESM_FILENAME)

    print(f"Loading NanoBulkFM checkpoint from {ckpt_path}...")
    fm, _, gene_ids = load_fm(ckpt_path, device)  # gene_ids come from the checkpoint itself

    id_to_symbol = load_gene_id_to_symbol(esm_path)
    missing = [g for g in gene_ids if g not in id_to_symbol]
    if missing:
        raise ValueError(f"{len(missing)} model gene IDs have no HGNC symbol in {esm_path}, e.g. {missing[:5]}")
    gene_symbols = [id_to_symbol[g] for g in gene_ids]

    meta_cols = {"cell_line_display_name", "lineage_1", "lineage_2", "lineage_3", "lineage_4", "lineage_6"}
    expr_df = pd.read_csv(expr_csv_path, index_col=0, low_memory=False)
    expr_df = expr_df.drop(columns=[c for c in meta_cols if c in expr_df.columns])
    expr_matrix = expr_df[gene_symbols].to_numpy(dtype=np.float32)  # [700, 1000]
    cell_ids = expr_df.index.tolist()

    print(f"Extracting cell-line embeddings for {len(cell_ids)} cell lines...")
    cell_embs = extract_cell_embeddings(fm, expr_matrix, device)   # [700, G, D]
    # Max-pool over genes once per unique cell line (not per pair): the pooled
    # result only depends on the cell line, so pooling here instead of inside
    # the training loop avoids redoing the same [G, D] -> [D] reduction for
    # every one of the ~300 drug pairs that share each cell line.
    cell_embs = cell_embs.max(axis=1)                                # [700, D]
    cell_emb_map = {cid: cell_embs[i] for i, cid in enumerate(cell_ids)}

    drug_data = np.load(drug_emb_npz_path, allow_pickle=True)
    drug_emb_map = {str(s): drug_data["embeddings"][i].astype(np.float32)
                    for i, s in enumerate(drug_data["smiles"])}

    ic50_df = pd.read_csv(ic50_csv_path, usecols=["ModelID", "smiles", "IC50"])
    ic50_df = ic50_df[ic50_df["ModelID"].isin(cell_emb_map) & ic50_df["smiles"].isin(drug_emb_map)]
    ic50_df = ic50_df.reset_index(drop=True)

    gene_dim = next(iter(cell_emb_map.values())).shape[-1]   # D (n_embd)
    drug_dim = next(iter(drug_emb_map.values())).shape[0]

    dataset = PairTensors(cell_emb_map, drug_emb_map, ic50_df, device)
    print(f"Dataset: {len(dataset)} pairs | gene_dim={gene_dim} | drug_dim={drug_dim}")
    return dataset, gene_dim, drug_dim


@torch.no_grad()
def _predict(model, dataset: PairTensors, idx: torch.Tensor, batch_size: int):
    preds = []
    for xc, xd, _ in dataset.batches(idx, batch_size, shuffle=False):
        preds.append(model(xc, xd).cpu().numpy())
    return np.concatenate(preds)


def run_fold(fold_idx: int, train_idx, test_idx, dataset: PairTensors, gene_dim: int, drug_dim: int, device: str):
    n_val = max(1, len(train_idx) // 10)
    val_idx, tr_idx = train_idx[:n_val], train_idx[n_val:]

    # Indices live on `device` too so batching (shuffle + slice + gather) is
    # pure GPU tensor indexing with no host<->device copies per step.
    tr_idx_t = torch.from_numpy(tr_idx).long().to(device)
    val_idx_t = torch.from_numpy(val_idx).long().to(device)
    test_idx_t = torch.from_numpy(test_idx).long().to(device)

    model = DrugResponseMLP(gene_dim, drug_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    for epoch in tqdm.tqdm(range(EPOCHS), desc="Training"):
        model.train()
        for xc, xd, yb in dataset.batches(tr_idx_t, BATCH_SIZE, shuffle=True):
            optimizer.zero_grad()
            criterion(model(xc, xd), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = np.mean([
                criterion(model(xc, xd), yb).item()
                for xc, xd, yb in dataset.batches(val_idx_t, BATCH_SIZE, shuffle=False)
            ])

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()

    preds = _predict(model, dataset, test_idx_t, BATCH_SIZE)
    y_test = dataset.y[test_idx_t].cpu().numpy()

    pcc, _ = pearsonr(y_test, preds)
    scc, _ = spearmanr(y_test, preds)
    return pcc, scc, epoch + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold (0-indexed). Omit to run all folds.")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--data-source",
        choices=("local", "hf"),
        default="local",
        help="Load drug-response/ESM data from local files or the Hugging Face dataset.",
    )
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = create_output_folder("out")
    log_file = open(out_dir / "run.log", "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    dataset, gene_dim, drug_dim = load_data(args.checkpoint, device, data_source=args.data_source)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(kf.split(range(len(dataset))))

    folds_to_run = [args.fold] if args.fold is not None else list(range(args.n_folds))
    results = []

    for k in folds_to_run:
        train_idx, test_idx = splits[k]
        print(f"\nFold {k}/{args.n_folds - 1} | train={len(train_idx)} test={len(test_idx)}")
        pcc, scc, stopped_epoch = run_fold(k, train_idx, test_idx, dataset, gene_dim, drug_dim, device)
        print(f"  PCC={pcc:.4f}  SCC={scc:.4f}  (stopped at epoch {stopped_epoch})")
        results.append({"fold": k, "pcc": pcc, "scc": scc, "epochs": stopped_epoch})

    results_df = pd.DataFrame(results)
    results_df.to_csv(out_dir / "results.csv", index=False)

    if len(results) > 1:
        print(f"\n{'='*40}")
        print(f"Mean PCC: {results_df['pcc'].mean():.4f} ± {results_df['pcc'].std():.4f}")
        print(f"Mean SCC: {results_df['scc'].mean():.4f} ± {results_df['scc'].std():.4f}")
        print(f"(BulkFormer target: PCC=0.910, SCC=0.879)")

    print(f"\nResults saved to {out_dir}/results.csv")


if __name__ == "__main__":
    main()
