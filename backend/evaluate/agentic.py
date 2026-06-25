"""
Agentic metrics — decision-quality scoring.

Where answer-quality metrics score the final ANSWER, these score the agent's
DECISION PROCESS, computed entirely from execution traces (EvalTrace):

  * Tool Selection Accuracy        - did the first tool match the expected tool?
  * Query Reformulation Success    - of WEAK retrievals, how many recovered to GOOD?
  * Web Escalation Accuracy        - did out-of-corpus queries escalate to web?
  * Avg Tool Calls / Query         - efficiency
  * Citation Accuracy              - inline [N] markers that resolve to real chunks
  * Retrieval Confidence stats     - mean / min / max top-3 vector similarity

Each benchmark item carries an `expected_tool` and `category` label so the
trace-derived behavior can be scored against ground-truth intent.

Outputs:
  results/agentic_metrics.csv    (per-query detail)
  results/agentic_summary.json   (aggregate metrics)
"""
from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from loguru import logger

from evaluate.agent_runner import run_query
from evaluate.trace_schema import EvalTrace

_HERE = Path(__file__).parent
_RESULTS_DIR = _HERE / "results"

# Map a benchmark category to the tool we expect the agent to reach for.
# None means "no tool should be called" (conversational / general knowledge).
_EXPECTED_TOOL_BY_CATEGORY = {
    "single_hop": "retrieve_documents",
    "multi_hop": "retrieve_documents",
    "out_of_corpus": "web_search",
    "conversational": None,
    "adversarial": "retrieve_documents",  # best-effort; soft-scored
}


def _expected_tool(item: dict) -> Any:
    if "expected_tool" in item:
        return item["expected_tool"]  # may be explicit null
    return _EXPECTED_TOOL_BY_CATEGORY.get(item.get("category", ""), "retrieve_documents")


def _keyword_recall(answer: str, required_keywords: list[str]) -> float | None:
    """Deterministic metric: fraction of required_keywords present in the answer (case-insensitive).
    Returns None when no keywords are defined for the question."""
    if not required_keywords:
        return None
    answer_lower = answer.lower()
    hits = sum(1 for kw in required_keywords if kw.lower() in answer_lower)
    return hits / len(required_keywords)


def _citation_accuracy(trace: EvalTrace) -> float | None:
    """Fraction of inline [N] markers that resolve to a real citation id."""
    inline = trace.inline_citation_ids()
    if not inline:
        return None  # not applicable
    cited_ids = {str(c.get("id")) for c in trace.citations}
    resolved = sum(1 for i in inline if i in cited_ids)
    return resolved / len(inline)


async def run_agentic_suite(items: list[dict], retriever: Any, citation_manager: Any) -> dict:
    logger.info(f"[agentic] scoring {len(items)} queries on decision quality")

    rows = []
    for item in items:
        trace = await run_query(item["question"], retriever, citation_manager)
        expected_tool = _expected_tool(item)

        # Tool selection: for "no tool expected", success = zero tool calls
        if expected_tool is None:
            tool_correct = trace.num_tool_calls == 0
        else:
            tool_correct = trace.first_tool == expected_tool

        reform_recovered = trace.reformulation_recovered()   # True / False / None
        cite_acc = _citation_accuracy(trace)
        kw_recall = _keyword_recall(trace.final_answer or "", item.get("required_keywords", []))

        rows.append({
            "id": item.get("id", ""),
            "category": item.get("category", ""),
            "question": item["question"],
            "expected_tool": expected_tool,
            "first_tool": trace.first_tool,
            "tool_correct": tool_correct,
            "num_tool_calls": trace.num_tool_calls,
            "did_reformulate": trace.did_reformulate(),
            "reformulation_recovered": reform_recovered,
            "escalated_to_web": trace.escalated_to_web(),
            "citation_accuracy": cite_acc,
            "keyword_recall": kw_recall,
            "retrieval_confidence": round(trace.retrieval_confidence, 3),
            "latency_s": round(trace.latency_s, 2),
            "intent": trace.intent,
            "error": trace.error,
        })

    summary = _aggregate(rows, items)
    _write_outputs(rows, summary)
    logger.info("[agentic] DONE")
    return summary


def _aggregate(rows: list[dict], items: list[dict]) -> dict:
    n = len(rows)
    if n == 0:
        return {}

    # Tool selection accuracy
    tool_acc = sum(1 for r in rows if r["tool_correct"]) / n

    # Reformulation success rate (only over rows that had a WEAK to recover from)
    reform_rows = [r for r in rows if r["reformulation_recovered"] is not None]
    reform_success = (
        sum(1 for r in reform_rows if r["reformulation_recovered"]) / len(reform_rows)
        if reform_rows else None
    )

    # Web escalation accuracy (only over out_of_corpus items)
    ooc = [r for r in rows if r["category"] == "out_of_corpus"]
    web_escalation_acc = (
        sum(1 for r in ooc if r["escalated_to_web"]) / len(ooc)
        if ooc else None
    )

    # Avg tool calls per query
    avg_tool_calls = statistics.mean(r["num_tool_calls"] for r in rows)

    # Citation accuracy (only where applicable)
    cite_vals = [r["citation_accuracy"] for r in rows if r["citation_accuracy"] is not None]
    citation_acc = statistics.mean(cite_vals) if cite_vals else None

    # Keyword recall — deterministic, no LLM (only factual questions with required_keywords)
    kw_vals = [r["keyword_recall"] for r in rows if r["keyword_recall"] is not None]
    keyword_recall = statistics.mean(kw_vals) if kw_vals else None

    # Retrieval confidence stats (only doc-QA rows that retrieved something)
    conf_vals = [r["retrieval_confidence"] for r in rows if r["retrieval_confidence"] > 0]
    conf_stats = {
        "mean": round(statistics.mean(conf_vals), 3) if conf_vals else 0.0,
        "min": round(min(conf_vals), 3) if conf_vals else 0.0,
        "max": round(max(conf_vals), 3) if conf_vals else 0.0,
        "n": len(conf_vals),
    }

    # Latency stats (P50 / P95) over all successful queries
    lat_vals = sorted(r["latency_s"] for r in rows if not r["error"])
    def _pct(p: float) -> float:
        if not lat_vals:
            return 0.0
        k = max(0, min(len(lat_vals) - 1, int(round(p * (len(lat_vals) - 1)))))
        return round(lat_vals[k], 2)
    latency_stats = {
        "p50": _pct(0.50),
        "p95": _pct(0.95),
        "mean": round(statistics.mean(lat_vals), 2) if lat_vals else 0.0,
        "max": round(max(lat_vals), 2) if lat_vals else 0.0,
        "n": len(lat_vals),
    }

    return {
        "n_queries": n,
        "tool_selection_accuracy": round(tool_acc, 3),
        "query_reformulation_success_rate": (
            round(reform_success, 3) if reform_success is not None else None
        ),
        "reformulation_sample_size": len(reform_rows),
        "web_escalation_accuracy": (
            round(web_escalation_acc, 3) if web_escalation_acc is not None else None
        ),
        "web_escalation_sample_size": len(ooc),
        "avg_tool_calls_per_query": round(avg_tool_calls, 2),
        "citation_accuracy": round(citation_acc, 3) if citation_acc is not None else None,
        "keyword_recall": round(keyword_recall, 3) if keyword_recall is not None else None,
        "keyword_recall_n": len(kw_vals),
        "retrieval_confidence": conf_stats,
        "latency": latency_stats,
        "errors": sum(1 for r in rows if r["error"]),
    }


def _write_outputs(rows: list[dict], summary: dict) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (_RESULTS_DIR / "agentic_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    with (_RESULTS_DIR / "agentic_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "id", "category", "expected_tool", "first_tool", "tool_correct",
            "num_tool_calls", "did_reformulate", "reformulation_recovered",
            "escalated_to_web", "citation_accuracy", "keyword_recall",
            "retrieval_confidence", "latency_s", "intent", "error",
        ])
        for r in rows:
            w.writerow([
                r["id"], r["category"], r["expected_tool"], r["first_tool"],
                r["tool_correct"], r["num_tool_calls"], r["did_reformulate"],
                r["reformulation_recovered"], r["escalated_to_web"],
                "" if r["citation_accuracy"] is None else round(r["citation_accuracy"], 3),
                "" if r["keyword_recall"] is None else round(r["keyword_recall"], 3),
                r["retrieval_confidence"], r["latency_s"], r["intent"], r["error"] or "",
            ])
