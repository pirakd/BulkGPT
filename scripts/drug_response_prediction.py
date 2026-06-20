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
"""
import argparse
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset

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


DEFAULT_CKPT = REPO / "out/train_nano_bulk_fm/20260620_223833/ckpt.pt"
EXPR_CSV = REPO / "data/drug_response/drug_response_expr_data.csv"
IC50_CSV = REPO / "data/drug_response/drug_response_prediction_IC50.csv"
DRUG_EMB_NPZ = REPO / "data/drug_response/drug_embeddings.npz"
ARCHS4_H5AD = REPO / "data/archs4/preprocessed_full.h5ad"

N_FOLDS = 1
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 50
PATIENCE = 5
DROPOUT = 0.1


class DrugResponseMLP(nn.Module):
    def __init__(self, gene_dim: int, drug_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gene_dim + drug_dim, 512),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(512, 1),
        )

    def forward(self, gene_embs: torch.Tensor, drug_emb: torch.Tensor) -> torch.Tensor:
        # gene_embs: [B, G, D], drug_emb: [B, drug_dim]
        cell_emb = gene_embs.max(dim=1).values          # [B, D]
        return self.net(torch.cat([cell_emb, drug_emb], dim=-1)).squeeze(-1)


def load_fm(ckpt_path: Path, device: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = NanoBulkFMConfig(**{k: v for k, v in ckpt["model_cfg"].items()
                               if k in NanoBulkFMConfig.__dataclass_fields__})
    model = NanoBulkFM(cfg, device=device).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, cfg


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


def load_data(ckpt_path: Path, device: str):
    # Gene space: map model's Ensembl IDs → HGNC symbols
    adata = ad.read_h5ad(ARCHS4_H5AD)
    gene_symbols = adata.var["gene_symbols"].tolist()   # ordered as model expects

    # Cell-line expression: select the 1000 model genes
    meta_cols = {"cell_line_display_name", "lineage_1", "lineage_2", "lineage_3", "lineage_4", "lineage_6"}
    expr_df = pd.read_csv(EXPR_CSV, index_col=0, low_memory=False)
    expr_df = expr_df.drop(columns=[c for c in meta_cols if c in expr_df.columns])
    expr_matrix = expr_df[gene_symbols].to_numpy(dtype=np.float32)  # [700, 1000]
    cell_ids = expr_df.index.tolist()

    # Cell-line embeddings via frozen FM
    print(f"Extracting cell-line embeddings for {len(cell_ids)} cell lines...")
    fm, _ = load_fm(ckpt_path, device)
    cell_embs = extract_cell_embeddings(fm, expr_matrix, device)   # [700, n_embd]
    cell_emb_map = {cid: cell_embs[i] for i, cid in enumerate(cell_ids)}

    # Drug embeddings
    drug_data = np.load(DRUG_EMB_NPZ, allow_pickle=True)
    drug_emb_map = {str(s): drug_data["embeddings"][i].astype(np.float32)
                    for i, s in enumerate(drug_data["smiles"])}

    # IC50 labels: join pairs
    ic50_df = pd.read_csv(IC50_CSV, usecols=["ModelID", "smiles", "IC50"])
    ic50_df = ic50_df[ic50_df["ModelID"].isin(cell_emb_map) & ic50_df["smiles"].isin(drug_emb_map)]
    ic50_df = ic50_df.reset_index(drop=True)

    cell_emb_dim = next(iter(cell_emb_map.values())).shape[-1]  # D (n_embd)
    drug_emb_dim = next(iter(drug_emb_map.values())).shape[0]

    X_cell = np.stack([cell_emb_map[r.ModelID] for r in ic50_df.itertuples()])
    X_drug = np.stack([drug_emb_map[str(r.smiles)] for r in ic50_df.itertuples()])
    y = ic50_df["IC50"].to_numpy(dtype=np.float32)

    print(f"Dataset: {len(y)} pairs | cell_emb={cell_emb_dim} | drug_emb={drug_emb_dim}")
    return X_cell, X_drug, y, cell_emb_dim, drug_emb_dim


def run_fold(fold_idx: int, train_idx, test_idx, X_cell, X_drug, y, gene_dim: int, drug_dim: int, device: str):
    # X_cell: [N, G, D], X_drug: [N, drug_dim]
    Xc_train = torch.from_numpy(X_cell[train_idx])
    Xd_train = torch.from_numpy(X_drug[train_idx])
    y_train = torch.from_numpy(y[train_idx])
    Xc_test = torch.from_numpy(X_cell[test_idx])
    Xd_test = torch.from_numpy(X_drug[test_idx])
    y_test = y[test_idx]

    model = DrugResponseMLP(gene_dim, drug_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = None

    # Simple 90/10 split of training data for early stopping
    n_val = max(1, len(train_idx) // 10)
    Xc_tr, Xc_val = Xc_train[n_val:], Xc_train[:n_val]
    Xd_tr, Xd_val = Xd_train[n_val:], Xd_train[:n_val]
    y_tr, y_val = y_train[n_val:], y_train[:n_val]
    tr_loader = DataLoader(TensorDataset(Xc_tr, Xd_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)

    for epoch in range(EPOCHS):
        model.train()
        for xc, xd, yb in tr_loader:
            xc, xd, yb = xc.to(device), xd.to(device), yb.to(device)
            optimizer.zero_grad()
            criterion(model(xc, xd), yb).backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = criterion(model(Xc_val.to(device), Xd_val.to(device)), y_val.to(device)).item()

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
    with torch.no_grad():
        preds = model(Xc_test.to(device), Xd_test.to(device)).cpu().numpy()

    pcc, _ = pearsonr(y_test, preds)
    scc, _ = spearmanr(y_test, preds)
    return pcc, scc, epoch + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--fold", type=int, default=None,
                        help="Run only this fold (0-indexed). Omit to run all 10.")
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = create_output_folder("out")
    log_file = open(out_dir / "run.log", "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)

    print(f"Device: {device}")
    print(f"Output: {out_dir}")

    X_cell, X_drug, y, cell_emb_dim, drug_emb_dim = load_data(args.checkpoint, device)

    kf = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    splits = list(kf.split(y))

    folds_to_run = [args.fold] if args.fold is not None else list(range(args.n_folds))
    results = []

    for k in folds_to_run:
        train_idx, test_idx = splits[k]
        print(f"\nFold {k}/{args.n_folds - 1} | train={len(train_idx)} test={len(test_idx)}")
        pcc, scc, stopped_epoch = run_fold(k, train_idx, test_idx, X_cell, X_drug, y, cell_emb_dim, drug_emb_dim, device)
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
