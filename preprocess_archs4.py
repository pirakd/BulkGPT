"""Preprocess ARCHS4 expression: QC → TPM → log1p → slice to panel → h5ad."""
import os

import anndata as ad
import archs4py as a4
import numpy as np
import pandas as pd

from utils import build_symbol_to_ensembl_hgnc, load_samples, qc_samples

ARCHS4_PATH = "data/archs4/human_gene_v2.latest.h5"
GENE_LENGTHS_CSV = "data/genes/gene_lengths.csv"
SELECTED_GENES_CSV = "data/genes/selected_genes.csv"
OUTPUT_H5AD = "data/archs4/preprocessed.h5ad"
N_SUBSET_SAMPLES = 10000 # if none take all sample


def compute_tpm(counts: pd.DataFrame, id_to_len: pd.Series) -> tuple[pd.DataFrame, np.ndarray]:
    """Return (filtered counts on genes with lengths, TPM matrix as ndarray)."""
    common = counts.index.intersection(id_to_len.index)
    counts = counts.loc[common]
    lengths_kb = id_to_len.loc[common].to_numpy(dtype=np.float32) / 1000.0
    rate = counts.to_numpy(dtype=np.float32) / lengths_kb[:, None]
    tpm = rate / rate.sum(axis=0, keepdims=True) * 1e6
    return counts, tpm


def main():
    meta = load_samples(ARCHS4_PATH)
    meta = qc_samples(meta).set_index("geo_accession")
    if N_SUBSET_SAMPLES is not None:
        meta = meta.sample(min(N_SUBSET_SAMPLES, len(meta)), random_state=42)
    print(f"QC-passing samples: {len(meta)}")

    print(f"Loading expression for {len(meta)} samples...")
    counts = a4.data.samples(ARCHS4_PATH, meta.index.tolist(), silent=False)
    counts = a4.utils.aggregate_duplicate_genes(counts)
    obs = meta.loc[counts.columns]

    gene_lengths = pd.read_csv(GENE_LENGTHS_CSV, index_col="ensembl_id")
    id_to_len = gene_lengths["transcript_length_bp"]

    # ARCHS4 uses gene symbols — remap to ensembl_id using HGNC (all aliases)
    print("Fetching symbol→ensembl_id mapping from HGNC complete set...")
    sym_to_id = build_symbol_to_ensembl_hgnc()
    print(f"HGNC mapping covers {len(sym_to_id):,} symbols")
    before = len(counts)
    counts.index = counts.index.map(sym_to_id)
    counts = counts[counts.index.notna() & ~counts.index.duplicated(keep="first")]
    print(f"Genes mapped: {len(counts)}/{before}")

    counts, tpm = compute_tpm(counts, id_to_len)
    print(f"Genes with lengths: {counts.shape[0]}")
    log_tpm = np.log1p(tpm)

    selected = pd.read_csv(SELECTED_GENES_CSV)
    panel = [g for g in selected["ensembl_id"] if g in counts.index]
    print(f"Panel genes present after TPM: {len(panel)}/{len(selected)}")
    gene_idx = counts.index.get_indexer(panel)
    X = log_tpm[gene_idx, :].T.astype(np.float32)

    var = selected.set_index("ensembl_id").loc[panel]
    adata = ad.AnnData(X=X, obs=obs, var=var)
    print(adata)

    os.makedirs(os.path.dirname(OUTPUT_H5AD), exist_ok=True)
    adata.write_h5ad(OUTPUT_H5AD)
    print(f"Saved to {OUTPUT_H5AD}")


if __name__ == "__main__":
    main()
