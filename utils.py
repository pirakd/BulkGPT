import pandas as pd

HGNC_COMPLETE_SET_URL = (
    "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt"
)


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
