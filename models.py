"""
models.py - Pydantic schemas for request/response validation.
"""

from pydantic import BaseModel, Field


# API request / response schemas

class QueryRequest(BaseModel):
    """Incoming user query payload."""

    question: str = Field(
        ...,
        min_length=3,
        max_length=1000,
        description="Natural language question about the gene dataset.",
        examples=["Which genes on Chromosome 17 are protein coding?"],
    )


class QueryResponse(BaseModel):
    """Non-streaming response (used in tests / evals)."""

    answer: str = Field(..., description="LLM-generated answer grounded in tool data.")


# Tool output schemas (returned by tools.py, forwarded to LLM as tool results)

class GeneRecord(BaseModel):
    """A single gene row from the dataset."""

    ensembl_id: str
    gene_symbol: str
    name: str
    biotype: str
    chromosome: str
    start: int | None = None
    end: int | None = None


class GeneList(BaseModel):
    """Structured list of gene records."""

    genes: list[GeneRecord]
    total: int


class AggregationResult(BaseModel):
    """Result of aggregate_gene_counts."""

    field: str
    counts: dict[str, int]  # value --> count
    total_genes: int
