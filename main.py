"""
main.py - FastAPI application entry point.

Exposes:
  POST /query  -> streaming natural language query over gene dataset
  GET  /health -> liveness check
  GET  /stats  -> dataset summary statistics
"""

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from agent import stream_agent
from data_loader import get_dataset
from models import QueryRequest

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Dataset path (override with DATA_PATH env var)
DATA_PATH = os.getenv("DATA_PATH", "genes.csv")


# Lifespan: pre-load dataset once at startup
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading gene dataset from %s …", DATA_PATH)
    df = get_dataset(DATA_PATH)
    logger.info("Dataset loaded: %d genes.", len(df))
    yield
    logger.info("Shutting down.")


# FastAPI app
app = FastAPI(
    title="Gene Research Assistant API",
    description=(
        "Natural language query interface for the human gene dataset. "
        "Uses tool-based reasoning to prevent hallucination."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Endpoints

@app.get("/health", tags=["System"])
async def health() -> dict[str, str]:
    """Liveness check."""
    return {"status": "ok"}


@app.get("/stats", tags=["System"])
async def stats() -> dict:
    """Return summary statistics about the loaded dataset."""
    df = get_dataset(DATA_PATH)
    return {
        "total_genes": len(df),
        "chromosomes": sorted(df["chromosome"].unique().tolist()),
        "biotypes": df["biotype"].value_counts().to_dict(),
        "genes_with_symbol": int((df["gene_symbol"] != "").sum()),
        "genes_with_name": int((df["name"] != "").sum()),
    }


@app.post("/query", tags=["Query"])
async def query(request: QueryRequest) -> StreamingResponse:
    """
    Accept a natural language question and return a streamed answer.

    The agent:
    1. Interprets the question.
    2. Calls appropriate dataset tools (never loads raw CSV into the prompt).
    3. Streams the final answer token by token.

    Example
    -------
    ```json
    { "question": "Which genes on Chromosome 17 are protein coding?" }
    ```
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question must not be empty.")

    async def event_stream():
        try:
            async for chunk in stream_agent(request.question):
                # Server-Sent Events format so clients can consume incrementally
                yield f"data: {chunk}\n\n"
        except Exception as exc:  # noqa: BLE001
            logger.exception("Error during streaming agent call")
            yield f"data: [ERROR] {exc}\n\n"
        finally:
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
