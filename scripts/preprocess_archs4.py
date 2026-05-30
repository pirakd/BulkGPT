"""Preprocess ARCHS4 expression: QC -> TPM -> log1p -> slice to panel -> h5ad."""
import sys
from pathlib import Path

import anndata as ad
import archs4py as a4
import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import (
    aggregate_duplicate_genes_lightweight,
    build_symbol_to_ensembl_hgnc,
    load_samples,
    qc_samples,
)

ARCHS4_PATH = "data/archs4/human_gene_v2.latest.h5"
GENE_LENGTHS_CSV = "data/genes/gene_lengths.csv"
SELECTED_GENES_CSV = "data/genes/selected_genes.csv"
OUTPUT_H5AD = "data/archs4/preprocessed_full.h5ad"
BATCH_SIZE = 1000
N_SUBSET_SAMPLES = 2000  # if None, take all QC-passing samples


def compute_tpm(counts: pd.DataFrame, id_to_len: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (filtered counts on genes with lengths, TPM matrix as ndarray)."""
    # TPM needs gene lengths, so discard genes where length is unavailable.
    common = counts.index.intersection(id_to_len.index)
    counts = counts.loc[common]

    # TPM is normalized per sample, so this is equivalent whether run in batches or all at once.
    lengths_kb = id_to_len.loc[common].to_numpy(dtype=np.float32) / 1000.0
    rate = counts.to_numpy(dtype=np.float32) / lengths_kb[:, None]
    library_size = rate.sum(axis=0, keepdims=True)
    tpm = np.divide(
        rate,
        library_size,
        out=np.zeros_like(rate, dtype=np.float32),
        where=library_size != 0,
    ) * 1e6
    return counts, tpm


def remap_counts_to_ensembl(counts: pd.DataFrame, sym_to_id: dict[str, str]) -> pd.DataFrame:
    """Aggregate duplicate symbols and remap ARCHS4 symbols to Ensembl IDs."""
    counts = aggregate_duplicate_genes_lightweight(counts)
    counts.index = counts.index.map(sym_to_id)
    counts = counts[counts.index.notna()]
    return aggregate_duplicate_genes_lightweight(counts)


def compute_panel_log_tpm(
    counts: pd.DataFrame,
    id_to_len: pd.Series,
    panel: list[str],
) -> tuple[np.ndarray, int]:
    """Compute log1p(TPM) for the selected panel from one batch of raw counts."""
    counts, tpm = compute_tpm(counts, id_to_len)
    panel_index = pd.Index(panel)
    present_panel = panel_index.intersection(counts.index)

    x = np.zeros((counts.shape[1], len(panel)), dtype=np.float32)
    if len(present_panel) == 0:
        return x, counts.shape[0]

    gene_idx = counts.index.get_indexer(present_panel)
    panel_idx = panel_index.get_indexer(present_panel)
    x[:, panel_idx] = np.log1p(tpm[gene_idx, :]).T.astype(np.float32)
    return x, counts.shape[0]


def main() -> None:
    # Load ARCHS4 sample metadata, then keep only bulk RNA-seq samples passing QC.
    meta = load_samples(ARCHS4_PATH)
    meta = qc_samples(meta).set_index("geo_accession")
    if N_SUBSET_SAMPLES is not None:
        # Fixed seed makes the subset reproducible across runs.
        meta = meta.sample(min(N_SUBSET_SAMPLES, len(meta)), random_state=42)
    print(f"QC-passing samples: {len(meta)}")

    gene_lengths = pd.read_csv(GENE_LENGTHS_CSV, index_col="ensembl_id")
    id_to_len = gene_lengths["transcript_length_bp"]

    # ARCHS4 uses gene symbols; remap to Ensembl IDs using HGNC aliases too.
    print("Fetching symbol->ensembl_id mapping from HGNC complete set...")
    sym_to_id = build_symbol_to_ensembl_hgnc()
    print(f"HGNC mapping covers {len(sym_to_id):,} symbols")

    # Slice expression to the selected gene panel used by the model.
    selected = pd.read_csv(SELECTED_GENES_CSV)
    panel = [g for g in selected["ensembl_id"] if g in id_to_len.index]
    print(f"Panel genes with lengths: {len(panel)}/{len(selected)}")

    output_path = Path(OUTPUT_H5AD)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_x_path = output_path.with_suffix(".X.float32.memmap")
    X = np.memmap(tmp_x_path, mode="w+", dtype=np.float32, shape=(len(meta), len(panel)))
    filled = np.zeros(len(meta), dtype=bool)

    sample_ids = meta.index.tolist()
    total_batches = (len(sample_ids) + BATCH_SIZE - 1) // BATCH_SIZE
    print(
        f"Loading expression for {len(sample_ids)} samples in {total_batches} "
        f"batches of up to {BATCH_SIZE:,}..."
    )
    for batch_num, start in enumerate(range(0, len(sample_ids), BATCH_SIZE), start=1):
        end = min(start + BATCH_SIZE, len(sample_ids))
        batch_ids = sample_ids[start:end]
        print(f"[{batch_num}/{total_batches}] Loading samples {start:,}:{end:,}", flush=True)

        counts = a4.data.samples(ARCHS4_PATH, batch_ids, silent=False)
        counts.columns = counts.columns.astype(str)
        before = len(counts)
        counts = remap_counts_to_ensembl(counts, sym_to_id)
        x_batch, genes_with_lengths = compute_panel_log_tpm(counts, id_to_len, panel)

        row_idx = meta.index.get_indexer(counts.columns)
        if (row_idx < 0).any():
            missing = counts.columns[row_idx < 0].tolist()
            raise RuntimeError(f"ARCHS4 returned samples not found in metadata: {missing[:5]}")

        X[row_idx, :] = x_batch
        filled[row_idx] = True
        X.flush()
        print(
            f"[{batch_num}/{total_batches}] Wrote {len(row_idx):,} samples; "
            f"genes mapped {len(counts):,}/{before:,}; genes with lengths {genes_with_lengths:,}",
            flush=True,
        )

    if not filled.all():
        missing = meta.index[~filled].tolist()
        raise RuntimeError(f"Missing expression for {len(missing):,} samples, e.g. {missing[:5]}")

    # AnnData stores sample metadata in obs, gene metadata in var, expression in X.
    var = selected.set_index("ensembl_id").loc[panel]
    adata = ad.AnnData(X=np.asarray(X), obs=meta, var=var)
    print(adata)

    adata.write_h5ad(output_path)
    tmp_x_path.unlink()
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    main()
