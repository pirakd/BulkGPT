"""Train NanoBulkFM with masked expression modeling on preprocessed ARCHS4."""
import argparse
import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader, Dataset, random_split

from lr_schedule import LRScheduleConfig, ReduceLROnPlateauLR
from model import NanoBulkFM, NanoBulkFMConfig
from utils import create_output_folder

OUT_ROOT = "out"
DEFAULT_CONFIG_PATH = "configs/train_nano_bulk_fm.yaml"


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


class ExpressionDataset(Dataset):
    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx]


def make_mask(shape, ratio, device):
    return torch.rand(shape, device=device) < ratio


def load_esm_gene_embeddings(var_names, embeddings_path):
    """Load per-gene ESM2 embeddings aligned to var_names (Ensembl gene IDs)."""
    df = pd.read_parquet(embeddings_path).set_index("ensembl_gene_id")
    dim = len(df["embedding"].iloc[0])
    aligned = np.zeros((len(var_names), dim), dtype=np.float32)
    missing = 0
    for i, gene_id in enumerate(var_names):
        if gene_id in df.index:
            aligned[i] = df.loc[gene_id, "embedding"]
        else:
            missing += 1
    if missing:
        print(f"Warning: missing ESM embeddings for {missing}/{len(var_names)} genes; using zeros.")
    return torch.from_numpy(aligned)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to a YAML config file with all run parameters. CLI flags override it.",
    )
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
        "--esm-data-source",
        choices=("local", "hf"),
        default=None,
        help="Load ESM gene embeddings from a local parquet file or a Hugging Face dataset.",
    )
    parser.add_argument(
        "--esm-embeddings-path",
        default=None,
        help="Local parquet path used when --esm-data-source=local.",
    )
    parser.add_argument(
        "--esm-hf-repo-id",
        default=None,
        help="Hugging Face dataset repo id used when --esm-data-source=hf.",
    )
    parser.add_argument(
        "--esm-hf-filename",
        default=None,
        help="File path inside the Hugging Face dataset repo for ESM embeddings.",
    )
    parser.add_argument(
        "--esm-hf-revision",
        default=None,
        help="Optional Hugging Face dataset revision, branch, or commit for ESM embeddings.",
    )
    parser.add_argument(
        "--esm-hf-cache-dir",
        default=None,
        help="Optional Hugging Face cache directory for downloaded ESM embeddings.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue training from an existing run. Requires --resume-dir.",
    )
    parser.add_argument(
        "--resume-dir",
        default=None,
        help=(
            "Path to an existing run's output folder (containing ckpt.pt and "
            "config.yaml) to resume training from. Used with --resume."
        ),
    )
    return parser.parse_args()


def build_config(args) -> SimpleNamespace:
    # --resume-dir passed on the CLI takes priority over one set in a config
    # file for deciding *which* config file to load in the first place.
    resuming = args.resume or bool(args.resume_dir)
    if resuming:
        if not args.resume_dir:
            raise ValueError("--resume requires --resume-dir to point at an existing run folder.")
        # Default to the resumed run's own resolved config unless the caller
        # explicitly passed a different --config.
        if args.config == DEFAULT_CONFIG_PATH:
            config_path = Path(args.resume_dir) / "config.yaml"
        else:
            config_path = Path(args.config)
    else:
        config_path = Path(args.config)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        data = yaml.safe_load(f) or {}

    if data.get("betas") is not None:
        data["betas"] = tuple(data["betas"])
    if not data.get("device"):
        data["device"] = "mps" if torch.backends.mps.is_available() else (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    if not data.get("lr_schedule"):
        raise ValueError(
            "Config is missing an `lr_schedule:` section (see lr_schedule.py "
            "and configs/train_nano_bulk_fm.yaml for the expected fields)."
        )
    lr_schedule = LRScheduleConfig(**data.pop("lr_schedule"))

    cfg = SimpleNamespace(**data)
    cfg.lr_schedule = lr_schedule
    for field in (
        "data_source",
        "data_path",
        "hf_repo_id",
        "hf_filename",
        "hf_revision",
        "hf_cache_dir",
        "esm_data_source",
        "esm_embeddings_path",
        "esm_hf_repo_id",
        "esm_hf_filename",
        "esm_hf_revision",
        "esm_hf_cache_dir",
    ):
        value = getattr(args, field)
        if value is not None:
            setattr(cfg, field, value)

    # `resume`/`resume_dir` may come from the config file (data.get) or the
    # CLI (args); the CLI flag can only turn resuming *on*, never off, since
    # argparse's store_true has no way to represent "explicitly False".
    cfg.resume = bool(args.resume or data.get("resume", False))
    cfg.resume_dir = args.resume_dir if args.resume_dir is not None else data.get("resume_dir")
    if cfg.resume and not cfg.resume_dir:
        raise ValueError("resume requires resume_dir to be set (via --resume-dir or the config file).")
    return cfg


def resolve_path(cfg: SimpleNamespace, source_attr: str, local_attr: str, hf_prefix: str) -> str:
    source = getattr(cfg, source_attr)
    if source == "local":
        return getattr(cfg, local_attr)
    if source != "hf":
        raise ValueError(f"Unknown {source_attr}: {source}")

    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        repo_id=getattr(cfg, f"{hf_prefix}repo_id"),
        filename=getattr(cfg, f"{hf_prefix}filename"),
        repo_type="dataset",
        revision=getattr(cfg, f"{hf_prefix}revision"),
        cache_dir=getattr(cfg, f"{hf_prefix}cache_dir"),
    )


@torch.no_grad()
def evaluate(model, loader, cfg: SimpleNamespace, gene_mean: torch.Tensor):
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
    args = parse_args()
    cfg = build_config(args)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    if cfg.resume:
        resume_from_dir = Path(cfg.resume_dir)
        if not resume_from_dir.is_dir():
            raise FileNotFoundError(f"resume_dir does not exist: {resume_from_dir}")
    out_dir = create_output_folder(OUT_ROOT)
    log_file = open(out_dir / "train.log", "w")
    sys.stdout = _Tee(sys.__stdout__, log_file)
    sys.stderr = _Tee(sys.__stderr__, log_file)
    print(f"Output dir: {out_dir}" + (f" (resuming from {cfg.resume_dir})" if cfg.resume else ""))

    cfg_dict = vars(cfg).copy()
    cfg_dict["betas"] = list(cfg_dict["betas"])
    cfg_dict["lr_schedule"] = dataclasses.asdict(cfg.lr_schedule)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)
    print(f"Saved resolved run config to {out_dir / 'config.yaml'}")

    data_path = resolve_path(cfg, "data_source", "data_path", "hf_")
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

    esm_gene_embeddings = None
    if cfg.use_esm_embeddings:
        esm_path = resolve_path(cfg, "esm_data_source", "esm_embeddings_path", "esm_hf_")
        print(f"Loading ESM embeddings from {esm_path}...")
        esm_gene_embeddings = load_esm_gene_embeddings(adata.var_names, esm_path)

    model_cfg = NanoBulkFMConfig(
        n_genes=n_genes,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        bias=cfg.bias,
        use_esm_embeddings=cfg.use_esm_embeddings,
        esm_embedding_dim=esm_gene_embeddings.shape[1] if esm_gene_embeddings is not None else None,
    )
    model = NanoBulkFM(model_cfg, device=cfg.device, esm_gene_embeddings=esm_gene_embeddings).to(cfg.device)
    if cfg.device.startswith("cuda"):
        model = torch.compile(model)
    print(f"Model params: {model.get_num_params():,} | device: {cfg.device} | n_genes: {model_cfg.n_genes} | n_layer: {model_cfg.n_layer} | n_embd: {model_cfg.n_embd}")

    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    nodecay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": cfg.weight_decay},
            {"params": nodecay_params, "weight_decay": 0.0},
        ],
        lr=cfg.lr_schedule.base_lr, betas=cfg.betas,
    )
    lr_scheduler = ReduceLROnPlateauLR(cfg.lr_schedule)

    use_amp = cfg.device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    best_val = float("inf")
    it = 0
    if cfg.resume:
        ckpt_path = resume_from_dir / "ckpt.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"No checkpoint found to resume from: {ckpt_path}")
        print(f"Loading checkpoint {ckpt_path} to resume training...")
        ckpt = torch.load(ckpt_path, map_location=cfg.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        best_val = ckpt.get("val_loss", float("inf"))
        it = ckpt.get("iter", -1) + 1
        if "lr_schedule_state" in ckpt:
            lr_scheduler.load_state_dict(ckpt["lr_schedule_state"])
        print(f"  resuming from iter {it} (best val so far {best_val:.4f})")

    steps_per_epoch = len(train_loader)
    train_iter = iter(train_loader)
    model.train()

    while it < cfg.max_iters:
        try:
            x = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x = next(train_iter)

        lr = lr_scheduler.step(it)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x = x.to(cfg.device)
        mask = make_mask(x.shape, cfg.mask_ratio, cfg.device)
        if not mask.any():
            mask[:, 0] = True

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda", dtype=torch.float16):
                _, _, loss = model(x, mask=mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, _, loss = model(x, mask=mask)
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
            reduced = lr_scheduler.on_eval(val_loss)
            if reduced:
                print(f"  val loss plateaued -> reducing LR to {lr_scheduler._lr:.2e} at iter {it}")
            if val_loss < best_val:
                best_val = val_loss
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_cfg": model_cfg.__dict__,
                    "gene_ids": list(adata.var_names),
                    "iter": it,
                    "val_loss": val_loss,
                    "lr_schedule_state": lr_scheduler.state_dict(),
                }
                torch.save(ckpt, out_dir / "ckpt.pt")
                print(f"  saved checkpoint (val {val_loss:.4f})")

        it += 1

    val_loss, r2, pear = evaluate(model, val_loader, cfg, gene_mean)
    print(f"final iter {it:6d} | epoch {it / steps_per_epoch:.2f} | val {val_loss:.4f} | R²(vs mean) {r2:+.3f} | r {pear:+.3f} (best {best_val:.4f})")


if __name__ == "__main__":
    main()
