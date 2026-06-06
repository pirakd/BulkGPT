"""Train NanoBulkFM with masked expression modeling on preprocessed ARCHS4."""
import argparse
import math
import sys
from dataclasses import dataclass

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from model import NanoBulkFM, NanoBulkFMConfig
from utils import create_output_folder

DATA_PATH = "data/archs4/preprocessed_full.h5ad"
HF_REPO_ID = "dpirak/ARCHS4_selected_genes"
HF_FILENAME = "preprocessed_full.h5ad"
OUT_ROOT = "out"

DEBUG = True


class _Tee:
    """Duplicate writes to several streams (e.g. console + log file)."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()  # flush per write so the log survives a crash

    def flush(self):
        for s in self.streams:
            s.flush()


@dataclass
class TrainConfig:
    data_source: str = "local"
    data_path: str = DATA_PATH
    hf_repo_id: str = HF_REPO_ID
    hf_filename: str = HF_FILENAME
    hf_revision: str | None = None
    hf_cache_dir: str | None = None

    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = True

    mask_ratio: float = 0.15
    batch_size: int = 32
    val_frac: float = 0.05
    max_iters: int = 20000
    eval_interval: int = 500
    eval_iters: int = 50
    log_interval: int = 50

    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 500
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    grad_clip: float = 1.0

    n_samples_subset: int | None = None

    seed: int = 42
    device: str = "mps" if torch.backends.mps.is_available() else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@dataclass
class DebugConfig(TrainConfig):
    n_layer: int = 3
    n_head: int = 4
    n_embd: int = 128
    dropout: float = 0.0
    batch_size: int = 16
    max_iters: int = 1000000
    eval_interval: int = 200
    eval_iters: int = 10
    log_interval: int = 20
    warmup_iters: int = 5
    n_samples_subset: int | None = 200000


class ExpressionDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


def make_mask(shape, ratio, device):
    return torch.rand(shape, device=device) < ratio


def get_lr(it, cfg: TrainConfig):
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / cfg.warmup_iters
    progress = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.min_lr + coeff * (cfg.lr - cfg.min_lr)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-source",
        choices=("local", "hf"),
        default=None,
        help="Load data from a local h5ad file or a Hugging Face dataset.",
    )
    parser.add_argument(
        "--data-path",
        default=None,
        help="Local h5ad path used when --data-source=local.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=None,
        help="Hugging Face dataset repo id used when --data-source=hf.",
    )
    parser.add_argument(
        "--hf-filename",
        default=None,
        help="File path inside the Hugging Face dataset repo.",
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help="Optional Hugging Face dataset revision, branch, or commit.",
    )
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for downloaded data.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use DebugConfig defaults. Defaults to the DEBUG constant.",
    )
    return parser.parse_args()


def build_config(args) -> TrainConfig:
    use_debug = DEBUG if args.debug is None else args.debug
    cfg = DebugConfig() if use_debug else TrainConfig()
    for field in (
        "data_source",
        "data_path",
        "hf_repo_id",
        "hf_filename",
        "hf_revision",
        "hf_cache_dir",
    ):
        value = getattr(args, field)
        if value is not None:
            setattr(cfg, field, value)
    return cfg


def resolve_data_path(cfg: TrainConfig) -> str:
    if cfg.data_source == "local":
        return cfg.data_path
    if cfg.data_source != "hf":
        raise ValueError(f"Unknown data_source: {cfg.data_source}")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub to load data from Hugging Face.") from exc

    return hf_hub_download(
        repo_id=cfg.hf_repo_id,
        filename=cfg.hf_filename,
        repo_type="dataset",
        revision=cfg.hf_revision,
        cache_dir=cfg.hf_cache_dir,
    )


@torch.no_grad()
def evaluate(model, loader, cfg: TrainConfig, gene_mean: torch.Tensor):
    model.eval()
    losses = []
    preds_all, trues_all, means_all = [], [], []
    sample_pred = sample_true = None
    for i, x in enumerate(loader):
        if i >= cfg.eval_iters:
            break
        x = x.to(cfg.device)
        mask = make_mask(x.shape, cfg.mask_ratio, cfg.device)
        if not mask.any():
            mask[:, 0] = True
        pred, _, loss = model(x, mask=mask)
        losses.append(loss.item())
        baseline = gene_mean.unsqueeze(0).expand_as(x)
        preds_all.append(pred[mask].cpu())
        trues_all.append(x[mask].cpu())
        means_all.append(baseline[mask].cpu())
        if sample_pred is None:
            b0 = 0
            masked_idx = mask[b0].nonzero(as_tuple=True)[0]
            if masked_idx.numel() > 0:
                k = min(10, masked_idx.numel())
                perm = torch.randperm(masked_idx.numel(), device=masked_idx.device)[:k]
                pick = masked_idx[perm]
                sample_pred = pred[b0, pick].cpu().numpy()
                sample_true = x[b0, pick].cpu().numpy()
                sample_base = gene_mean[pick].cpu().numpy()
                sample_gene_idx = pick.cpu().numpy()
    if sample_pred is not None:
        print("  recovered vs true (masked genes):")
        print(f"    {'gene_idx':>10} {'pred':>10} {'true':>10} {'mean':>10} {'err':>10}")
        for g, p, t, m in zip(sample_gene_idx, sample_pred, sample_true, sample_base):
            print(f"    {int(g):>10d} {p:>10.4f} {t:>10.4f} {m:>10.4f} {p - t:>+10.4f}")
    model.train()
    val = float(np.mean(losses)) if losses else float("nan")
    p = torch.cat(preds_all).numpy()
    t = torch.cat(trues_all).numpy()
    m = torch.cat(means_all).numpy()
    ss_res = float(((t - p) ** 2).sum())
    ss_tot = float(((t - m) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pear = float(np.corrcoef(p, t)[0, 1])
    return val, r2, pear


def main():
    cfg = build_config(parse_args())
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out_dir = create_output_folder(OUT_ROOT)
    log_file = open(out_dir / "train.log", "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"Output dir: {out_dir}")

    data_path = resolve_data_path(cfg)
    print(f"Loading {data_path}...")
    adata = ad.read_h5ad(data_path)
    X = adata.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    X = X.astype(np.float32)
    if cfg.n_samples_subset is not None and cfg.n_samples_subset < X.shape[0]:
        rng = np.random.default_rng(cfg.seed)
        idx = rng.choice(X.shape[0], size=cfg.n_samples_subset, replace=False)
        X = X[idx]
    n_samples, n_genes = X.shape
    print(f"Loaded {n_samples} samples × {n_genes} genes")

    dataset = ExpressionDataset(X)
    n_val = max(1, int(len(dataset) * cfg.val_frac))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(cfg.seed),
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    train_X = dataset.X[train_ds.indices]
    gene_mean = train_X.mean(dim=0).to(cfg.device)

    model_cfg = NanoBulkFMConfig(
        n_genes=n_genes,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
    )
    model = NanoBulkFM(model_cfg, device=cfg.device).to(cfg.device)
    print(f"Model params: {model.get_num_params():,} | device: {cfg.device} | n_genes: {model_cfg.n_genes} | n_layer: {model_cfg.n_layer} | n_embd: {model_cfg.n_embd}")

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr, betas=cfg.betas,
    )

    best_val = float("inf")
    it = 0
    steps_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    model.train()

    while it < cfg.max_iters:
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x = next(train_iter)

        lr = get_lr(it, cfg)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x = x.to(cfg.device)
        mask = make_mask(x.shape, cfg.mask_ratio, cfg.device)
        if not mask.any():
            mask[:, 0] = True

        _, _, loss = model(x, mask=mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        epoch = it / steps_per_epoch
        if it % cfg.log_interval == 0:
            print(f"iter {it:6d} | epoch {epoch:.2f} | loss {loss.item():.4f} | lr {lr:.2e}")

        if it > 0 and it % cfg.eval_interval == 0:
            val_loss, r2, pear = evaluate(model, val_loader, cfg, gene_mean)
            print(f"iter {it:6d} | epoch {epoch:.2f} | val loss {val_loss:.4f} | R²(vs mean) {r2:+.3f} | pearson_r {pear:+.3f}")
            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_cfg": model_cfg.__dict__,
                    "iter": it,
                    "val_loss": val_loss,
                }
                torch.save(ckpt, out_dir / "ckpt.pt")
                print(f"  saved checkpoint (val {val_loss:.4f})")

        it += 1

    val_loss, r2, pear = evaluate(model, val_loader, cfg, gene_mean)
    print(f"final iter {it:6d} | epoch {it / steps_per_epoch:.2f} | val {val_loss:.4f} | R²(vs mean) {r2:+.3f} | r {pear:+.3f} (best {best_val:.4f})")


if __name__ == "__main__":
    main()
