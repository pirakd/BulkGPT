"""Build gene-level protein embeddings from Ensembl canonical proteins.

tי
1. retrieve each human protein-coding gene's primary protein product from Ensembl;
2. run an ESM protein language model on the amino-acid sequence;
3. mean-pool residue embeddings into one embedding per gene;
4. write a parquet file keyed by Ensembl gene/protein identifiers.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1]))

from prepare import create_output_folder


DEFAULT_MODEL_NAME = "facebook/esm2_t48_15B_UR50D"
DEFAULT_OUTPUT_DIR = "data/genes"
OUTPUT_FILENAME = "protein_coding_gene_esm2_embeddings.parquet"
DEFAULT_SEQUENCE_CACHE = "data/genes/ensembl_canonical_protein_sequences.parquet"


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch Ensembl canonical protein products for human protein-coding "
            "genes and write mean-pooled ESM embeddings to parquet."
        )
    )
    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Output parquet path. If omitted, writes under a timestamped "
            f"{DEFAULT_OUTPUT_DIR}/build_gene_protein_embeddings/ run folder."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Base directory passed to utils.create_output_folder when --output is omitted.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
        help=(
            "Hugging Face model id. Default is the largest ESM2 checkpoint; "
            "smaller ESM2 options include facebook/esm2_t33_650M_UR50D."
        ),
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device: auto, cuda, cuda:0, mps, or cpu.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model dtype. auto uses bf16/fp16 on CUDA and fp32 elsewhere.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of protein windows per model forward pass.",
    )
    parser.add_argument(
        "--max-residues-per-window",
        type=int,
        default=1022,
        help=(
            "Maximum amino-acid residues per ESM2 window. ESM2 supports 1022 "
            "residues plus special tokens."
        ),
    )
    parser.add_argument(
        "--parquet-row-group-size",
        type=int,
        default=256,
        help="Number of genes buffered before each parquet row group write.",
    )
    parser.add_argument(
        "--sequence-cache",
        default=DEFAULT_SEQUENCE_CACHE,
        help="Parquet cache for Ensembl canonical protein metadata and sequences.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cached Ensembl metadata/sequences and download again.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of genes to embed, useful for smoke tests.",
    )
    parser.add_argument(
        "--compression",
        default="zstd",
        help="Parquet compression codec, e.g. zstd, snappy, or none.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True to Hugging Face model/tokenizer loading.",
    )
    return parser.parse_args()


def fetch_or_load_sequences(cache_path: Path, refresh: bool) -> pd.DataFrame:
    """Fetch canonical protein metadata and peptide sequences with pybiomart."""
    if cache_path.exists() and not refresh:
        log(f"Loading cached Ensembl protein sequences from {cache_path}")
        return pd.read_parquet(cache_path)

    try:
        from pybiomart import Dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: pybiomart. Install it with `uv add pybiomart` "
            "or `pip install pybiomart` in this environment."
        ) from exc

    log("Fetching canonical protein sequences from Ensembl BioMart with pybiomart...")
    dataset = Dataset(name="hsapiens_gene_ensembl", host="www.ensembl.org")
    df = dataset.query(
        attributes=[
            "ensembl_gene_id",
            "hgnc_symbol",
            "gene_biotype",
            "ensembl_transcript_id",
            "ensembl_peptide_id",
            "transcript_is_canonical",
            "peptide",
        ],
        filters={
            "biotype": "protein_coding",
            "transcript_biotype": "protein_coding",
            "transcript_is_canonical": "only",
        },
        use_attr_names=True,
    )
    log(f"BioMart returned {len(df):,} transcript/protein rows")

    df = df.rename(columns={"peptide": "protein_sequence", "Peptide": "protein_sequence"})
    required = {
        "ensembl_gene_id",
        "hgnc_symbol",
        "gene_biotype",
        "ensembl_transcript_id",
        "ensembl_peptide_id",
        "transcript_is_canonical",
        "protein_sequence",
    }
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"pybiomart response missing expected columns: {sorted(missing)}")

    df = df[
        (df["gene_biotype"] == "protein_coding")
        & df["ensembl_gene_id"].notna()
        & df["ensembl_peptide_id"].notna()
        & df["protein_sequence"].notna()
        & (df["ensembl_peptide_id"].astype(str).str.strip() != "")
        & (df["protein_sequence"].astype(str).str.strip() != "")
    ].copy()
    df["protein_sequence"] = df["protein_sequence"].astype(str).str.replace(r"\s+", "", regex=True)
    df["protein_sequence_length"] = df["protein_sequence"].str.len().astype(int)
    df = df.sort_values(["ensembl_gene_id", "ensembl_transcript_id"]).drop_duplicates(
        "ensembl_gene_id",
        keep="first",
    )
    df = df[
        [
            "ensembl_gene_id",
            "hgnc_symbol",
            "gene_biotype",
            "ensembl_transcript_id",
            "ensembl_peptide_id",
            "transcript_is_canonical",
            "protein_sequence",
            "protein_sequence_length",
        ]
    ].reset_index(drop=True)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    log(f"Cached {len(df):,} canonical protein sequences to {cache_path}")
    return df


def resolve_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(dtype_arg: str, device: torch.device) -> torch.dtype:
    if dtype_arg == "float32" or (dtype_arg == "auto" and device.type != "cuda"):
        return torch.float32
    if dtype_arg == "bfloat16" or (
        dtype_arg == "auto" and device.type == "cuda" and torch.cuda.is_bf16_supported()
    ):
        return torch.bfloat16
    return torch.float16


def load_hf_model(model_name: str, device: torch.device, dtype: torch.dtype, trust_remote_code: bool):
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: transformers. Install it with `uv add transformers` "
            "or `pip install transformers` in this environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.to(device)
    model.eval()
    return tokenizer, model


def split_sequence(sequence: str, max_residues: int) -> Iterator[str]:
    for start in range(0, len(sequence), max_residues):
        yield sequence[start : start + max_residues]


def mean_pool_windows(
    sequences: Sequence[str],
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
) -> np.ndarray:
    tokenized = tokenizer(
        list(sequences),
        add_special_tokens=True,
        padding=True,
        return_tensors="pt",
        return_special_tokens_mask=True,
    )
    special_tokens_mask = tokenized.pop("special_tokens_mask").bool().to(device)
    tokenized = {key: value.to(device) for key, value in tokenized.items()}

    with torch.inference_mode():
        outputs = model(**tokenized)
        hidden = outputs.last_hidden_state

    residue_mask = tokenized["attention_mask"].bool() & ~special_tokens_mask
    lengths = residue_mask.sum(dim=1).clamp(min=1).unsqueeze(1)
    pooled = (hidden * residue_mask.unsqueeze(-1)).sum(dim=1) / lengths
    return pooled.detach().cpu().float().numpy()


def clean_optional_string(value: Any) -> str | None:
    if pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def embedding_row(metadata: pd.Series, embedding: np.ndarray, model_name: str) -> dict[str, Any]:
    embedding = embedding.astype(np.float32, copy=False)
    return {
        "ensembl_gene_id": metadata["ensembl_gene_id"],
        "hgnc_symbol": clean_optional_string(metadata.get("hgnc_symbol")),
        "gene_biotype": metadata["gene_biotype"],
        "ensembl_transcript_id": metadata["ensembl_transcript_id"],
        "ensembl_peptide_id": metadata["ensembl_peptide_id"],
        "protein_sequence_length": int(metadata["protein_sequence_length"]),
        "model_name": model_name,
        "embedding_dim": int(embedding.shape[0]),
        "embedding": embedding.tolist(),
    }


def iter_gene_embeddings(
    proteins: pd.DataFrame,
    tokenizer: Any,
    model: torch.nn.Module,
    device: torch.device,
    model_name: str,
    batch_size: int,
    max_residues_per_window: int,
) -> Iterator[dict[str, Any]]:
    states: dict[int, dict[str, Any]] = {}
    pending: list[tuple[int, str, int]] = []

    def process_pending() -> list[dict[str, Any]]:
        batch = pending.copy()
        pending.clear()
        pooled = mean_pool_windows([item[1] for item in batch], tokenizer, model, device)
        completed: list[dict[str, Any]] = []

        for (row_idx, _window, window_length), window_embedding in zip(batch, pooled):
            state = states[row_idx]
            if state["weighted_sum"] is None:
                state["weighted_sum"] = np.zeros_like(window_embedding, dtype=np.float32)
            state["weighted_sum"] += window_embedding.astype(np.float32) * window_length
            state["completed_windows"] += 1

            if state["completed_windows"] == state["expected_windows"]:
                gene_embedding = state["weighted_sum"] / state["total_residues"]
                completed.append(embedding_row(state["metadata"], gene_embedding, model_name))
                del states[row_idx]

        return completed

    for row_idx, row in tqdm(proteins.iterrows(), total=len(proteins), desc="Embedding genes"):
        sequence = row["protein_sequence"]
        windows = list(split_sequence(sequence, max_residues_per_window))
        states[row_idx] = {
            "metadata": row,
            "weighted_sum": None,
            "completed_windows": 0,
            "expected_windows": len(windows),
            "total_residues": len(sequence),
        }
        for window in windows:
            pending.append((row_idx, window, len(window)))
            if len(pending) >= batch_size:
                yield from process_pending()

    if pending:
        yield from process_pending()


def parquet_schema():
    import pyarrow as pa

    return pa.schema(
        [
            ("ensembl_gene_id", pa.string()),
            ("hgnc_symbol", pa.string()),
            ("gene_biotype", pa.string()),
            ("ensembl_transcript_id", pa.string()),
            ("ensembl_peptide_id", pa.string()),
            ("protein_sequence_length", pa.int32()),
            ("model_name", pa.string()),
            ("embedding_dim", pa.int32()),
            ("embedding", pa.list_(pa.float32())),
        ]
    )


def write_embeddings_parquet(
    rows: Iterator[dict[str, Any]],
    output_path: Path,
    row_group_size: int,
    compression: str,
) -> int:
    import pyarrow as pa
    import pyarrow.parquet as pq

    schema = parquet_schema()
    writer = pq.ParquetWriter(
        output_path,
        schema=schema,
        compression=None if compression.lower() == "none" else compression,
    )
    buffer: list[dict[str, Any]] = []
    written = 0
    try:
        for row in rows:
            buffer.append(row)
            if len(buffer) >= row_group_size:
                writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
                written += len(buffer)
                buffer.clear()
        if buffer:
            writer.write_table(pa.Table.from_pylist(buffer, schema=schema))
            written += len(buffer)
    finally:
        writer.close()
    return written


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.max_residues_per_window < 1:
        raise ValueError("--max-residues-per-window must be >= 1")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = create_output_folder(args.output_dir) / OUTPUT_FILENAME

    log(f"Output parquet: {output_path}")

    proteins = fetch_or_load_sequences(Path(args.sequence_cache), args.refresh_cache)
    log(f"Canonical protein-coding genes with sequences from Ensembl: {len(proteins):,}")
    if args.limit is not None:
        proteins = proteins.head(args.limit).copy()
        log(f"Limiting run to first {len(proteins):,} genes")

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    log(f"Loading {args.model_name} on {device} with {dtype}...")
    tokenizer, model = load_hf_model(args.model_name, device, dtype, args.trust_remote_code)

    rows = iter_gene_embeddings(
        proteins,
        tokenizer,
        model,
        device,
        args.model_name,
        args.batch_size,
        args.max_residues_per_window,
    )
    written = write_embeddings_parquet(
        rows,
        output_path,
        args.parquet_row_group_size,
        args.compression,
    )
    log(f"Saved {written:,} gene embeddings to {output_path}")


if __name__ == "__main__":
    main()
