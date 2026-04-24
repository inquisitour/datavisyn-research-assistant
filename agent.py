"""
agent.py - LLM agent with tool-calling for gene dataset queries.

Uses the Anthropic Messages API directly with tool_use support.
The agent decides which tool(s) to call, executes them against the
clean pandas DataFrame, and synthesises a final streamed answer.

Design
------
* Tools are registered as Anthropic tool schemas.
* The agent loop runs until the model stops calling tools.
* Streaming is implemented via the /v1/messages endpoint with
  stream=True (server-sent events).
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncGenerator
from typing import Any

import anthropic

import tools as gene_tools

logger = logging.getLogger(__name__)

# Anthropic client 
_client = anthropic.Anthropic()
_async_client = anthropic.AsyncAnthropic()

MODEL = "claude-sonnet-4-20250514"

SYSTEM_PROMPT = """You are a bioinformatics research assistant with access to a human gene database. You MUST answer questions ONLY using data returned by the provided tools.

Rules:
1. Use tools to retrieve all factual information before answering.
2. Prefer using filter_genes for multi-criteria queries (e.g., chromosome + biotype).
3. Select the most appropriate tool based on the query.
4. If tools return no results, say: "I don't know. No matching genes were found."
5. Never fabricate gene names, Ensembl IDs, chromosomes, or counts.
6. Always report the true total from the tool's 'total' field, even if only 50 genes are shown. State "showing X of Y total" when truncated.
7. Be concise and accurate. Cite counts from tool output.
8. Do not include reasoning preamble. Start directly with the answer.

Formatting rules:
- Start with a plain one-line summary (no bold formatting).
- Each bullet should contain one gene.
- Format genes as:
  - If gene symbol exists: - **SYMBOL** (ENSEMBL_ID) — Name
  - If no gene symbol: - *(no symbol)* (ENSEMBL_ID) — Name
  - If no symbol AND no name: - *(unnamed)* (ENSEMBL_ID)
- Display biotypes in readable form (e.g., protein_coding → Protein Coding).
- Add a blank line between sections.
- For aggregation results: - **Value**: N genes
- Always add a blank line before the final insight.
- Do not use markdown tables.
"""

# Tool schemas (Anthropic format)
TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_genes_by_chromosome",
        "description": (
            "Return all genes located on a specific chromosome. "
            "Use this when the user asks about genes on a particular chromosome."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "chromosome": {
                    "type": "string",
                    "description": "Chromosome identifier, e.g. '17', 'X', 'Y', '1'.",
                }
            },
            "required": ["chromosome"],
        },
    },
    {
        "name": "filter_by_biotype",
        "description": (
            "Return genes of a specific biotype. Valid biotypes include: "
            "protein_coding, linc_rna, processed_pseudogene, "
            "unprocessed_pseudogene, antisense. "
            "Use this when the user asks for genes of a particular type."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "biotype": {
                    "type": "string",
                    "description": "Biotype in snake_case, e.g. 'protein_coding'.",
                }
            },
            "required": ["biotype"],
        },
    },
    {
        "name": "filter_genes",
        "description": "Filter genes by any combination of chromosome, biotype, and/or name keyword. Prefer this over calling multiple tools separately when the question involves multiple criteria.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chromosome": {"type": "string", "description": "Chromosome e.g. '17', 'X'"},
                "biotype": {"type": "string", "description": "e.g. 'protein_coding', 'linc_rna'"},
                "name": {"type": "string", "description": "Keyword to search in gene names/symbols"},
            },
        },
    },
    {
        "name": "search_gene_name",
        "description": (
            "Search gene names and symbols using a keyword or phrase. "
            "Use this when the user mentions a gene function, pathway, or "
            "partial gene name such as 'G protein-coupled receptor'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Search term to find in gene names or symbols.",
                }
            },
            "required": ["name"],
        },
    },
    {
        "name": "aggregate_gene_counts",
        "description": (
            "Count genes grouped by a field. "
            "Supported fields: 'chromosome', 'biotype'. "
            "Use this for summary statistics questions like 'how many genes per chromosome'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "field": {
                    "type": "string",
                    "enum": ["chromosome", "biotype"],
                    "description": "Field to aggregate by.",
                }
            },
            "required": ["field"],
        },
    },
]

# Tool dispatcher
def _dispatch_tool(name: str, tool_input: dict[str, Any]) -> str:
    """
    Call the appropriate tool function and return the result as a JSON string.

    Parameters
    ----------
    name : str
        Tool name as declared in TOOL_SCHEMAS.
    tool_input : dict
        Arguments the LLM decided to pass.

    Returns
    -------
    str
        JSON-serialised tool output (to be sent back to the model as a
        tool_result block).
    """
    try:
        if name == "get_genes_by_chromosome":
            result = gene_tools.get_genes_by_chromosome(**tool_input)
        elif name == "filter_by_biotype":
            result = gene_tools.filter_by_biotype(**tool_input)
        elif name == "filter_genes":
            result = gene_tools.filter_genes(**tool_input)
        elif name == "search_gene_name":
            result = gene_tools.search_gene_name(**tool_input)
        elif name == "aggregate_gene_counts":
            result = gene_tools.aggregate_gene_counts(**tool_input)
        else:
            result = {"error": f"Unknown tool: {name}"}
    except Exception as exc:  
        logger.exception("Tool %s raised an error", name)
        result = {"error": str(exc)}

    return json.dumps(result, ensure_ascii=False)


# Synchronous agent (used by evals)
def run_agent(question: str) -> str:
    """
    Run the full agentic loop synchronously and return the final answer.

    The loop:
    1. Send user question + tool schemas to the model.
    2. If model calls tools → execute them, append results, repeat.
    3. When model stops calling tools → return its text response.

    Parameters
    ----------
    question : str
        Natural language user question.

    Returns
    -------
    str
        Final answer text from the model.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    while True:
        response = _client.messages.create(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

        # Append assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract plain text
            text_blocks = [
                block.text for block in response.content if hasattr(block, "text")
            ]
            return "\n".join(text_blocks)

        if response.stop_reason == "tool_use":
            # Execute every tool the model requested
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    logger.info("Calling tool: %s(%s)", block.name, block.input)
                    output = _dispatch_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
            # Loop again with tool results in context
        else:
            # Unexpected stop reason: return whatever text we have
            text_blocks = [
                block.text for block in response.content if hasattr(block, "text")
            ]
            return "\n".join(text_blocks) or "No answer produced."


# Async streaming agent (used by FastAPI endpoint)
async def stream_agent(question: str) -> AsyncGenerator[str, None]:
    """
    Async generator that runs the agentic loop and streams the final answer
    token-by-token using the Anthropic streaming API.

    Yields
    ------
    str
        Incremental text chunks from the final LLM response.
    """
    messages: list[dict[str, Any]] = [{"role": "user", "content": question}]

    while True:
        # Always stream: handle tool_use blocks as they arrive
        tool_calls: list[dict[str, Any]] = []
        collected_content: list[Any] = []

        async with _async_client.messages.stream(
            model=MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        ) as stream:
            async for text in stream.text_stream:
                yield text

            final = await stream.get_final_message()
            collected_content = final.content

            if final.stop_reason == "tool_use":
                for block in final.content:
                    if block.type == "tool_use":
                        logger.info("Calling tool: %s(%s)", block.name, block.input)
                        output = _dispatch_tool(block.name, block.input)
                        tool_calls.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        })

        if tool_calls:
            messages.append({"role": "assistant", "content": collected_content})
            messages.append({"role": "user", "content": tool_calls})
            # Loop: next iteration streams the final answer directly
        else:
            return  # end_turn, already streamed