"""
evals.py - Evaluation script for the Gene Research Assistant.

Runs a fixed set of questions against the agent and checks whether
expected keywords / facts appear in the response.

Usage
-----
    python evals.py

Output
------
Prints per-question PASS / FAIL and an overall accuracy score.
"""

from __future__ import annotations

import sys
import time

from dotenv import load_dotenv
load_dotenv()

from agent import run_agent

# Evaluation cases
# ---------------------------------------------------------------------------
# Each case has:
#   question  - sent to the agent
#   must_contain - list of strings that MUST appear in the answer (case-insensitive)
#   must_not_contain - list of strings that MUST NOT appear (hallucination guard)
# ---------------------------------------------------------------------------
EVAL_CASES: list[dict] = [
    {
        "id": "E01",
        "question": "How many genes are on chromosome 17?",
        "must_contain": ["17"],          # answer must mention chrom 17
        "must_not_contain": [],
        "note": "Count check for chromosome 17",
    },
    {
        "id": "E02",
        "question": "List all protein coding genes on chromosome X.",
        "must_contain": ["protein coding", "X"],
        "must_not_contain": [],
        "note": "Filter by biotype + chromosome",
    },
    {
        "id": "E03",
        "question": "Which genes are associated with G protein-coupled receptors?",
        "must_contain": ["GPR"],          # GPR prefix genes expected
        "must_not_contain": [],
        "note": "Name search – G protein receptors",
    },
    {
        "id": "E04",
        "question": "What biotypes are present in the dataset and how many genes does each have?",
        "must_contain": ["protein coding", "linc"],
        "must_not_contain": [],
        "note": "Aggregation by biotype",
    },
    {
        "id": "E05",
        "question": "Find all pseudogenes in the dataset.",
        "must_contain": ["pseudogene"],
        "must_not_contain": [],
        "note": "Biotype filter – pseudogenes",
    },
    {
        "id": "E06",
        "question": "What is the Ensembl ID for GPR88?",
        "must_contain": ["ENSG00000181656"],
        "must_not_contain": [],
        "note": "Specific gene lookup by symbol",
    },
    {
        "id": "E07",
        "question": "Which chromosomes have the most genes?",
        "must_contain": [],              # dynamic data; just check no crash
        "must_not_contain": ["I don't know"],
        "note": "Aggregation by chromosome, ranked",
    },
    {
        "id": "E08",
        "question": "Tell me about glutathione peroxidase genes.",
        "must_contain": ["GPX"],
        "must_not_contain": [],
        "note": "Name keyword search",
    },
    {
        "id": "E09",
        "question": "Are there any antisense genes in the dataset?",
        "must_contain": ["antisense"],
        "must_not_contain": [],
        "note": "Biotype presence check",
    },
    {
        "id": "E10",
        "question": "What is the capital of France?",   # out-of-scope
        "must_contain": ["gene"],      
        "must_not_contain": ["Paris"],
        "note": "Out-of-scope refusal – no hallucination",
    },
]


# Runner
def evaluate() -> None:
    passed = 0
    failed = 0
    results: list[dict] = []

    print("=" * 70)
    print("Gene Research Assistant – Evaluation Suite")
    print("=" * 70)

    for case in EVAL_CASES:
        eid = case["id"]
        question = case["question"]
        must_contain = [kw.lower() for kw in case.get("must_contain", [])]
        must_not_contain = [kw.lower() for kw in case.get("must_not_contain", [])]

        print(f"\n[{eid}] {case.get('note', question)}")
        print(f"  Q: {question}")

        start = time.perf_counter()
        try:
            answer = run_agent(question)
        except Exception as exc:  # noqa: BLE001
            answer = f"[EXCEPTION] {exc}"
        elapsed = time.perf_counter() - start

        answer_lower = answer.lower()

        # Check must_contain
        missing = [kw for kw in must_contain if kw not in answer_lower]
        # Check must_not_contain
        found_bad = [kw for kw in must_not_contain if kw in answer_lower]

        ok = not missing and not found_bad
        status = "PASS ✓" if ok else "FAIL ✗"
        if ok:
            passed += 1
        else:
            failed += 1

        print(f"  A: {answer[:200]}{'…' if len(answer) > 200 else ''}")
        print(f"  → {status}  ({elapsed:.1f}s)")
        if missing:
            print(f"     Missing keywords: {missing}")
        if found_bad:
            print(f"     Unexpected content: {found_bad}")

        results.append({"id": eid, "status": status, "elapsed": elapsed})

    total = passed + failed
    accuracy = passed / total * 100 if total else 0

    print("\n" + "=" * 70)
    print(f"Results: {passed}/{total} passed  ({accuracy:.0f}% accuracy)")
    print("=" * 70)

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    evaluate()
