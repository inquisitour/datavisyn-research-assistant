"""
data_loader.py - Load and preprocess the gene CSV dataset.

Handles semicolon-delimited CSV, normalizes column names, and cleans
categorical values so all downstream tools operate on consistent data.
"""

import re
import pandas as pd
from pathlib import Path


# Column mapping: raw CSV header -> internal snake_case name
COLUMN_MAP: dict[str, str] = {
    "ensembl": "ensembl_id",
    "gene symbol": "gene_symbol",
    "name": "name",
    "biotype": "biotype",
    "chromosome": "chromosome",
    "seq region start": "start",
    "seq region end": "end",
}


def _normalize_header(raw: str) -> str:
    """Lower-case a raw header and collapse whitespace → single space."""
    return raw.strip().lower()


def _to_snake_case(text: str) -> str:
    """
    Convert a human-readable string to snake_case.

    Examples
    --------
    'Protein Coding'   → 'protein_coding'
    'Linc R N A'       → 'linc_rna'
    'Processed Pseudogene' → 'processed_pseudogene'
    """
    # lower-case, collapse runs of whitespace to single underscore
    normalized = re.sub(r"\s+", "_", text.strip().lower())
    # collapse multiple underscores
    normalized = re.sub(r"_+", "_", normalized)
    return normalized


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Load the gene CSV (semicolon-delimited) and return a clean DataFrame.

    Steps
    -----
    1. Read CSV with sep=';', keep raw column names.
    2. Normalize column headers to snake_case via COLUMN_MAP.
    3. Fill missing gene_symbol / name with empty string.
    4. Normalize 'biotype' to snake_case (e.g. 'Protein Coding' → 'protein_coding').
    5. Cast chromosome to str; start / end to nullable Int64.

    Parameters
    ----------
    csv_path : str | Path
        Path to the semicolon-delimited gene CSV file.

    Returns
    -------
    pd.DataFrame
        Cleaned, typed DataFrame ready for tool queries.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at: {path.resolve()}")

    # --- 1. Read raw ---
    raw = pd.read_csv(path, sep=";", dtype=str)

    # --- 2. Normalize headers ---
    raw.columns = [_normalize_header(c) for c in raw.columns]

    # Map known headers; drop anything unrecognised
    rename_map: dict[str, str] = {}
    for raw_col in raw.columns:
        if raw_col in COLUMN_MAP:
            rename_map[raw_col] = COLUMN_MAP[raw_col]

    df = raw.rename(columns=rename_map)

    # Keep only the columns we care about (ignore extras)
    expected = list(COLUMN_MAP.values())
    df = df[[c for c in expected if c in df.columns]].copy()

    # --- 3. Missing values ---
    for col in ("gene_symbol", "name"):
        if col in df.columns:
            df[col] = df[col].fillna("").str.strip()

    # Remove any trailing '[Source:...]' annotations that are not part of the gene name
    df["name"] = df["name"].str.replace(r'\s*\[Source:[^\]]*\]', '', regex=True).str.strip()

    # --- 4. Normalise biotype ---
    if "biotype" in df.columns:
        df["biotype"] = (
            df["biotype"]
            .fillna("unknown")
            .apply(lambda v: _to_snake_case(v) if isinstance(v, str) else "unknown")
        )

    # --- 5. Types ---
    if "chromosome" in df.columns:
        df["chromosome"] = df["chromosome"].fillna("").str.strip()

    for col in ("start", "end"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    return df


# Singleton: loaded once at import time if module is used directly.
_DATASET: pd.DataFrame | None = None


def get_dataset(csv_path: str | Path = "genes.csv") -> pd.DataFrame:
    """
    Return the cached dataset, loading it on first call.

    Parameters
    ----------
    csv_path : str | Path
        Path used only on the first call; subsequent calls return the cache.
    """
    global _DATASET
    if _DATASET is None:
        _DATASET = load_dataset(csv_path)
    return _DATASET
