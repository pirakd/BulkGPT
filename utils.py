import h5py
import numpy as np
import pandas as pd

HGNC_COMPLETE_SET_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)

def load_samples(archs4_path: str = "data/archs4/human_gene_v2.latest.h5") -> pd.DataFrame:
    """Load sample-level metadata from an ARCHS4 HDF5 file.

    Args:
        archs4_path: Path to the ARCHS4 .h5 file.

    Returns:
        DataFrame with columns: geo_accession, library_strategy,
        alignedreads, singlecellprobability, platform_id.
    """
    with h5py.File(archs4_path, "r") as f:
        return pd.DataFrame({
            "geo_accession": f["meta/samples/geo_accession"][:].astype(str),
            "library_strategy": f["meta/samples/library_strategy"][:].astype(str),
            "alignedreads": f["meta/samples/alignedreads"][:],
            "singlecellprobability": f["meta/samples/singlecellprobability"][:],
            "platform_id": f["meta/samples/platform_id"][:].astype(str),
        })


def qc_samples(
    meta: pd.DataFrame,
    min_aligned_reads: int = 100_000,
    max_sc_probability: float = 0.5,
) -> pd.DataFrame:
    """Filter samples to bulk RNA-seq with sufficient depth and low single-cell probability."""
    n = len(meta)
    meta = meta[meta["library_strategy"] == "RNA-Seq"]
    print(f"RNA-Seq filter: {len(meta)}/{n}")

    n = len(meta)
    meta = meta[meta["singlecellprobability"] < max_sc_probability]
    print(f"SC filter (prob<{max_sc_probability}): {len(meta)}/{n}")

    n = len(meta)
    meta = meta[meta["alignedreads"] >= min_aligned_reads]
    print(f"Aligned reads filter (>={min_aligned_reads:,}): {len(meta)}/{n}")

    return meta.reset_index(drop=True)


def aggregate_duplicate_genes_lightweight(exp: pd.DataFrame) -> pd.DataFrame:
    """Aggregate duplicate gene rows without pandas groupby over the full matrix.

    pandas groupby materializes large intermediate blocks for wide expression
    matrices. This keeps the first copy of each gene row and only sums rows for
    gene symbols that are actually duplicated.
    """
    index = pd.Index(exp.index)
    if index.is_unique:
        return exp

    values = exp.to_numpy(copy=False)
    duplicate_labels = index[index.duplicated(keep=False)].unique()
    keep_mask = ~index.duplicated(keep="first")

    for label in duplicate_labels:
        row_positions = np.flatnonzero(index == label)
        first = row_positions[0]
        if len(row_positions) > 1:
            values[first, :] = values[row_positions, :].sum(axis=0)

    return exp.loc[keep_mask]


def build_symbol_to_ensembl_hgnc() -> dict[str, str]:
    """Build a comprehensive gene symbol → Ensembl ID mapping from the HGNC complete set.

    Covers current symbols, previous symbols, and alias symbols so that legacy
    or alternative names used by tools like ARCHS4 are resolved.
    Priority (highest overwrites lower): current symbol > prev_symbol > alias_symbol.

    Returns:
        Dict mapping any known gene symbol (case-preserved) to ensembl_id.
    """
    hgnc = pd.read_csv(HGNC_COMPLETE_SET_URL, sep="\t", low_memory=False)
    hgnc = hgnc.dropna(subset=["ensembl_gene_id"])
    hgnc = hgnc[hgnc["ensembl_gene_id"].str.startswith("ENSG", na=False)]

    sym_to_id: dict[str, str] = {}

    # Pipe-separated multi-value columns — expand each entry
    for col in ["alias_symbol", "prev_symbol"]:
        sub = hgnc[["ensembl_gene_id", col]].dropna()
        for eid, cell in zip(sub["ensembl_gene_id"], sub[col]):
            for sym in str(cell).split("|"):
                sym = sym.strip()
                if sym:
                    sym_to_id[sym] = eid

    # Current symbol has highest priority — overwrites aliases/prev
    for sym, eid in zip(hgnc["symbol"], hgnc["ensembl_gene_id"]):
        sym = str(sym).strip()
        if sym:
            sym_to_id[sym] = eid

    return sym_to_id


def build_symbol_to_ensembl(hgnc: pd.DataFrame) -> dict[str, str]:
    """Build a gene symbol → Ensembl gene ID mapping from the HGNC complete set.

    One-to-one for most genes; last entry wins for the rare duplicates.

    Args:
        hgnc: HGNC complete set DataFrame with 'symbol' and 'ensembl_gene_id' columns.

    Returns:
        Dict mapping HGNC symbol to Ensembl gene ID.
    """
    return (
        hgnc[["symbol", "ensembl_gene_id"]]
        .dropna()
        .set_index("symbol")["ensembl_gene_id"]
        .to_dict()
    )


def build_ensembl_to_symbols(hgnc: pd.DataFrame) -> dict[str, list[str]]:
    """Build an Ensembl gene ID → HGNC symbol list mapping from the HGNC complete set.

    One-to-many to capture duplicate and retired symbol entries.

    Args:
        hgnc: HGNC complete set DataFrame with 'symbol' and 'ensembl_gene_id' columns.

    Returns:
        Dict mapping Ensembl gene ID to list of associated HGNC symbols.
    """
    return (
        hgnc[["symbol", "ensembl_gene_id"]]
        .dropna()
        .groupby("ensembl_gene_id")["symbol"]
        .apply(list)
        .to_dict()
    )
