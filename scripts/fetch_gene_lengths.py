import os
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils import HGNC_COMPLETE_SET_URL, build_ensembl_to_symbols

ENSEMBL_RELEASE = "current"
BIOMART_URL = "https://www.ensembl.org/biomart/martservice"
OUT_PATH = "data/genes/gene_lengths.csv"

BIOMART_XML = """<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE Query>
<Query virtualSchemaName="default" formatter="TSV" header="1" uniqueRows="1" count="" datasetConfigVersion="0.6">
    <Dataset name="hsapiens_gene_ensembl" interface="default">
        <Attribute name="ensembl_gene_id" />
        <Attribute name="hgnc_symbol" />
        <Attribute name="gene_biotype" />
        <Attribute name="transcript_length" />
    </Dataset>
</Query>"""


def fetch_gene_lengths() -> pd.DataFrame:
    """Fetch median transcript length per Ensembl gene ID from Ensembl BioMart.

    Queries all human protein-coding transcripts from the pinned Ensembl release,
    then takes the median length per gene. Median transcript length is used as the
    effective-length proxy for TPM normalization.

    Returns:
        DataFrame indexed by ensembl_id with columns:
        hgnc_symbol, gene_biotype, transcript_length_bp (median, float).
    """
    r = requests.get(BIOMART_URL, params={"query": BIOMART_XML}, timeout=300)
    r.raise_for_status()

    df = pd.read_csv(StringIO(r.text), sep="\t")
    df.columns = ["ensembl_id", "hgnc_symbol", "gene_biotype", "transcript_length"]

    df["transcript_length"] = pd.to_numeric(df["transcript_length"], errors="coerce")
    df = df.dropna(subset=["ensembl_id", "transcript_length"])
    df = df[df["ensembl_id"].str.strip() != ""]
    df = df[df["gene_biotype"] == "protein_coding"]

    lengths = df.groupby("ensembl_id").agg(
        gene_biotype=("gene_biotype", "first"),
        transcript_length_bp=("transcript_length", "median"),
    )

    # Enrich hgnc_symbol from HGNC complete set (more reliable than BioMart's annotation)
    hgnc = pd.read_csv(HGNC_COMPLETE_SET_URL, sep="\t", low_memory=False)
    ensembl_to_syms = build_ensembl_to_symbols(hgnc)
    lengths["hgnc_symbol"] = lengths.index.map(
        lambda eid: ensembl_to_syms.get(eid, [""])[0]
    )
    return lengths[["hgnc_symbol", "gene_biotype", "transcript_length_bp"]]


if __name__ == "__main__":
    print(f"Querying Ensembl {ENSEMBL_RELEASE} BioMart for transcript lengths...")
    gene_lengths = fetch_gene_lengths()
    print(f"Fetched lengths for {len(gene_lengths):,} protein-coding genes")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    gene_lengths.to_csv(OUT_PATH)
    print(f"Saved to {OUT_PATH}")
