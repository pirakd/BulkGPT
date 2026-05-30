"""Train NanoBulkFM with masked expression modeling on preprocessed ARCHS4."""
import math
import os
from dataclasses import dataclass

import anndata as ad
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from model import NanoBulkFM, NanoBulkFMConfig

DATA_PATH = "data/archs4/preprocessed_full.h5ad"
OUT_DIR = "out/nano_bulk_fm"

DEBUG = True


@dataclass
class TrainConfig:
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
    cfg = DebugConfig() if DEBUG else TrainConfig()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(OUT_DIR, exist_ok=True)

    print(f"Loading {DATA_PATH}...")
    adata = ad.read_h5ad(DATA_PATH)
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
                torch.save(ckpt, os.path.join(OUT_DIR, "ckpt.pt"))
                print(f"  saved checkpoint (val {val_loss:.4f})")

        it += 1

    val_loss, r2, pear = evaluate(model, val_loader, cfg, gene_mean)
    print(f"final iter {it:6d} | epoch {it / steps_per_epoch:.2f} | val {val_loss:.4f} | R²(vs mean) {r2:+.3f} | r {pear:+.3f} (best {best_val:.4f})")


if __name__ == "__main__":
    main()
