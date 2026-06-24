"""
CLI entrypoint for the full evaluation suite.

Usage (inside the backend container, where Qdrant is reachable):

    docker compose exec backend python -m evaluate.run_all
    docker compose exec backend python -m evaluate.run_all --suite functional
    docker compose exec backend python -m evaluate.run_all --suite agentic
    docker compose exec backend python -m evaluate.run_all --suite ragas

Runs the functional harness, agentic metrics, and RAGAS pipeline, then prints a
consolidated report. All artifacts are written to evaluate/results/.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from evaluate.services import EvalServices
from evaluate.functional import run_functional_suite
from evaluate.agentic import run_agentic_suite
from evaluate.ragas_eval import run_ragas_suite

_HERE = Path(__file__).parent
_BENCHMARK = _HERE / "datasets" / "benchmark_dataset.json"
_RESULTS_DIR = _HERE / "results"


def _load_benchmark_items() -> list[dict]:
    data = json.loads(_BENCHMARK.read_text(encoding="utf-8"))["questions"]
    # agentic.py expects 'question' key (already present in benchmark)
    return data


def _print_header(title: str) -> None:
    bar = "=" * 64
    print(f"\n{bar}\n  {title}\n{bar}")


async def main(suite: str) -> None:
    services = await EvalServices().initialize()
    retriever = services.retriever
    cm = services.citation_manager

    reports: dict = {}

    if suite in ("all", "functional"):
        _print_header("FUNCTIONAL TEST SUITE (A1-A8)")
        reports["functional"] = await run_functional_suite(retriever, cm)

    if suite in ("all", "agentic"):
        _print_header("AGENTIC METRICS SUITE")
        items = _load_benchmark_items()
        reports["agentic"] = await run_agentic_suite(items, retriever, cm)

    if suite in ("all", "ragas"):
        _print_header("RAGAS QUANTITATIVE SUITE")
        reports["ragas"] = await run_ragas_suite(retriever, cm)

    # ── Consolidated console report ─────────────────────────────────────────
    _print_header("CONSOLIDATED REPORT")

    if "functional" in reports:
        f = reports["functional"]
        print(f"\nFunctional: {f['passed']}/{f['total']} passed ({f['pass_rate']:.0%})")
        for cat, stats in f["by_category"].items():
            print(f"   {cat:38s} {stats['passed']}/{stats['total']}")

    if "agentic" in reports:
        a = reports["agentic"]
        print("\nAgentic decision metrics:")
        print(f"   Tool selection accuracy        {_pct(a.get('tool_selection_accuracy'))}")
        print(f"   Query reformulation success    {_pct(a.get('query_reformulation_success_rate'))}"
              f"  (n={a.get('reformulation_sample_size', 0)})")
        print(f"   Web escalation accuracy        {_pct(a.get('web_escalation_accuracy'))}"
              f"  (n={a.get('web_escalation_sample_size', 0)})")
        print(f"   Avg tool calls / query         {a.get('avg_tool_calls_per_query')}")
        print(f"   Citation accuracy              {_pct(a.get('citation_accuracy'))}")
        rc = a.get("retrieval_confidence", {})
        print(f"   Retrieval confidence (mean)    {rc.get('mean')}  (min={rc.get('min')}, max={rc.get('max')})")

    if "ragas" in reports:
        r = reports["ragas"]
        print("\nRAGAS answer-quality metrics:")
        if r.get("status") == "completed" and "aggregate" in r:
            for metric, val in r["aggregate"].items():
                print(f"   {metric:32s} {val}")
        else:
            print(f"   {r.get('status', 'unknown')}: {r.get('reason') or r.get('error', '')}")

    print(f"\nArtifacts written to: {_RESULTS_DIR}\n")


def _pct(v) -> str:
    return "n/a" if v is None else f"{v:.0%}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agentic RAG evaluation suite")
    parser.add_argument(
        "--suite", choices=["all", "functional", "agentic", "ragas"],
        default="all", help="Which suite to run (default: all)",
    )
    args = parser.parse_args()
    asyncio.run(main(args.suite))
