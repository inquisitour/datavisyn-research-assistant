"""
tools.py - Data query functions called by the LLM agent as tools.

Each function operates exclusively on the cleaned pandas DataFrame from
data_loader.py.  Raw CSV is NEVER loaded into the LLM prompt.

All functions return plain Python dicts / lists so they serialise easily
as tool call results.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from data_loader import get_dataset


# Internal helpers
def _df() -> pd.DataFrame:
    """Shorthand: return the cached, cleaned DataFrame."""
    return get_dataset()


MAX_RESULTS = 50  # cap tool output to prevent context overflow


def _rows_to_dicts(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a subset DataFrame to a list of plain dicts (JSON-safe), capped at MAX_RESULTS."""
    records = []
    for _, row in df.head(MAX_RESULTS).iterrows():
        records.append(
            {
                "ensembl_id": row.get("ensembl_id", ""),
                "gene_symbol": row.get("gene_symbol", ""),
                "name": row.get("name", ""),
                "biotype": row.get("biotype", ""),
                "chromosome": str(row.get("chromosome", "")),
                "start": int(row["start"]) if pd.notna(row.get("start")) else None,
                "end": int(row["end"]) if pd.notna(row.get("end")) else None,
            }
        )
    return records


# Public tool functions
def get_genes_by_chromosome(chromosome: str) -> dict[str, Any]:
    """
    Return all genes located on the specified chromosome.

    Parameters
    ----------
    chromosome : str
        Chromosome identifier, e.g. "17", "X", "Y".

    Returns
    -------
    dict with keys 'genes' (list) and 'total' (int).
    """
    chrom = str(chromosome).strip().upper().lstrip("CHR")  # normalise 'chr17' -> '17'
    df = _df()

    mask = df["chromosome"].str.upper().str.lstrip("CHR") == chrom
    filtered = df[mask]

    return {
        "genes": _rows_to_dicts(filtered),
        "total": len(filtered),
        "chromosome": chromosome,
    }


def filter_by_biotype(biotype: str) -> dict[str, Any]:
    """
    Return genes whose biotype matches the given value.

    Matching is case-insensitive and tolerates both raw forms
    ('Protein Coding') and snake_case forms ('protein_coding').

    Parameters
    ----------
    biotype : str
        E.g. "protein_coding", "linc_rna", "processed_pseudogene".

    Returns
    -------
    dict with keys 'genes' (list), 'total' (int), 'biotype' (str).
    """
    # Normalise query the same way the loader normalises the data
    query = re.sub(r"\s+", "_", biotype.strip().lower())
    query = re.sub(r"_+", "_", query)

    df = _df()
    # Partial match to catch subtypes (e.g. "pseudogene" -> all pseudogene variants)
    mask = df["biotype"].str.lower().str.contains(query, na=False)
    filtered = df[mask]

    return {
        "genes": _rows_to_dicts(filtered),
        "total": len(filtered),
        "biotype": query,
    }
    

def filter_genes(chromosome: str | None = None, biotype: str | None = None, name: str | None = None) -> dict[str, Any]:
    """
    Filter genes by any combination of chromosome, biotype, and/or name keyword.

    Parameters
    ----------
    chromosome : str, optional
        Chromosome identifier, e.g. "17", "X". Handles "chr17" prefix automatically.
    biotype : str, optional
        Biotype in snake_case or human form, e.g. "protein_coding", "Protein Coding".
    name : str, optional
        Keyword to search in gene names and symbols (case-insensitive substring match).

    Returns
    -------
    dict with keys 'genes' (list, capped at MAX_RESULTS), 'total' (int), 'filters' (dict).
    """

    df = _df()
    
    if chromosome:
        chrom = str(chromosome).strip().upper().lstrip("CHR")
        df = df[df["chromosome"].str.upper().str.lstrip("CHR") == chrom]
    
    if biotype:
        query = re.sub(r"\s+", "_", biotype.strip().lower())
        # Direct match first
        df_match = df[df["biotype"].str.contains(query, na=False)]
        if len(df_match) == 0:
            # Strip underscores and compare (linc_rna -> lincrna matches linc_r_n_a -> lincrna)
            query_stripped = query.replace("_", "")
            df_match = df[df["biotype"].str.replace("_", "", regex=False).str.contains(query_stripped, na=False)]
        df = df_match
    
    if name:
        mask = (df["name"].str.lower().str.contains(name.lower(), na=False, regex=False) |
                df["gene_symbol"].str.lower().str.contains(name.lower(), na=False, regex=False))
        df = df[mask]
    
    return {"genes": _rows_to_dicts(df), "total": len(df), "filters": {"chromosome": chromosome, "biotype": biotype, "name": name}}


def search_gene_name(name: str) -> dict[str, Any]:
    """
    Full-text search in gene 'name' and 'gene_symbol' columns.

    Case-insensitive substring match.  Searches both columns and returns
    the union (deduplicated by ensembl_id).

    Parameters
    ----------
    name : str
        Search term, e.g. "G protein-coupled receptor".

    Returns
    -------
    dict with keys 'genes' (list), 'total' (int), 'query' (str).
    """
    query = name.strip().lower()
    df = _df()

    name_mask = df["name"].str.lower().str.contains(query, na=False, regex=False)
    sym_mask = df["gene_symbol"].str.lower().str.contains(query, na=False, regex=False)

    filtered = df[name_mask | sym_mask].drop_duplicates(subset="ensembl_id")

    return {
        "genes": _rows_to_dicts(filtered),
        "total": len(filtered),
        "query": name,
    }


def aggregate_gene_counts(field: str) -> dict[str, Any]:
    """
    Count genes grouped by a categorical field.

    Parameters
    ----------
    field : str
        One of: 'chromosome', 'biotype'.

    Returns
    -------
    dict with keys 'field', 'counts' (dict value→count), 'total_genes'.

    Raises
    ------
    ValueError if field is not supported.
    """
    allowed = {"chromosome", "biotype"}
    clean_field = field.strip().lower()
    if clean_field not in allowed:
        raise ValueError(
            f"Field '{field}' is not supported for aggregation. "
            f"Choose from: {sorted(allowed)}"
        )

    df = _df()
    counts = df[clean_field].value_counts().to_dict()

    # Sort by count descending for readability
    counts = dict(sorted(counts.items(), key=lambda x: -x[1]))

    return {
        "field": clean_field,
        "counts": counts,
        "total_genes": len(df),
    }
