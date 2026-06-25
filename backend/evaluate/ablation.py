"""
Retrieval ablation study — BM25-only vs Vector-only vs Hybrid.

Controlled experiment: the SAME evaluation pipeline (agentic suite) and the SAME
dataset are used for all three runs. The only variable changed between runs is
`retriever.mode`, which selects which ranked list(s) feed RRF inside
HybridRetriever.retrieve(). Reranker, thresholds, prompts, and model logic are
identical across modes.

Scope: only document-grounded questions (single_hop + multi_hop) are used, because
those are the only category where retrieval strategy is actually exercised.
out_of_corpus / conversational / adversarial test escalation and refusal, not
retrieval quality, so they would dilute the comparison.

Metrics per mode:
  * Keyword recall (primary, deterministic — no LLM in the loop)
  * Citation accuracy
  * Tool selection accuracy
  * Retrieval confidence (mean)
  * Latency P50 / P95

Usage (inside the backend container):
    docker compose exec backend python -m evaluate.ablation

Outputs:
  results/ablation_summary.json   (per-mode aggregate)
  results/ablation_summary.csv    (comparison table)
  results/ablation_<mode>.csv      (per-query detail for each mode)
"""
from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

from loguru import logger

from evaluate.services import EvalServices
from evaluate.agentic import run_agentic_suite

_HERE = Path(__file__).parent
_BENCHMARK = _HERE / "datasets" / "benchmark_dataset.json"
_RESULTS = _HERE / "results"

_MODES = ["bm25", "vector", "hybrid"]
_RETRIEVAL_CATEGORIES = {"single_hop", "multi_hop"}


def _load_doc_grounded_items() -> list[dict]:
    data = json.loads(_BENCHMARK.read_text(encoding="utf-8"))["questions"]
    return [q for q in data if q.get("category") in _RETRIEVAL_CATEGORIES]


async def main() -> None:
    services = await EvalServices().initialize()
    retriever = services.retriever
    cm = services.citation_manager

    items = _load_doc_grounded_items()
    logger.info(f"[ablation] {len(items)} doc-grounded questions × {len(_MODES)} modes")

    summaries: dict[str, dict] = {}
    for mode in _MODES:
        retriever.mode = mode
        logger.info(f"[ablation] ===================== MODE = {mode} =====================")
        summaries[mode] = await run_agentic_suite(items, retriever, cm)
        # Preserve per-mode per-query detail (run_agentic_suite overwrites the shared file)
        src = _RESULTS / "agentic_metrics.csv"
        if src.exists():
            src.replace(_RESULTS / f"ablation_{mode}.csv")

    retriever.mode = "hybrid"  # restore default for any later use
    _write_and_print(summaries)


def _write_and_print(summaries: dict[str, dict]) -> None:
    rows = []
    for mode in _MODES:
        s = summaries[mode]
        lat = s.get("latency", {})
        rc = s.get("retrieval_confidence", {})
        rows.append({
            "mode": mode,
            "keyword_recall": s.get("keyword_recall"),
            "citation_accuracy": s.get("citation_accuracy"),
            "tool_selection_accuracy": s.get("tool_selection_accuracy"),
            "retrieval_confidence_mean": rc.get("mean"),
            "latency_p50": lat.get("p50"),
            "latency_p95": lat.get("p95"),
        })

    _RESULTS.mkdir(parents=True, exist_ok=True)
    (_RESULTS / "ablation_summary.json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    with (_RESULTS / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # ── Console comparison table ────────────────────────────────────────────
    def _pct(v) -> str:
        return "  n/a" if v is None else f"{v:6.1%}"

    def _num(v) -> str:
        return "  n/a" if v is None else f"{v:6.3f}"

    bar = "=" * 72
    print(f"\n{bar}\n  RETRIEVAL ABLATION — BM25 vs VECTOR vs HYBRID  (n={len(_load_doc_grounded_items())})\n{bar}")
    print(f"\n{'Metric':<28}{'BM25':>12}{'Vector':>12}{'Hybrid':>12}")
    print("-" * 64)
    by_mode = {r["mode"]: r for r in rows}

    def _line(label: str, key: str, fmt) -> None:
        print(f"{label:<28}"
              f"{fmt(by_mode['bm25'][key]):>12}"
              f"{fmt(by_mode['vector'][key]):>12}"
              f"{fmt(by_mode['hybrid'][key]):>12}")

    _line("Keyword recall (primary)", "keyword_recall", _pct)
    _line("Citation accuracy", "citation_accuracy", _pct)
    _line("Tool selection accuracy", "tool_selection_accuracy", _pct)
    _line("Retrieval confidence (mean)", "retrieval_confidence_mean", _num)
    _line("Latency P50 (s)", "latency_p50", _num)
    _line("Latency P95 (s)", "latency_p95", _num)

    # Verdict on the primary metric
    kr = {m: by_mode[m]["keyword_recall"] or 0.0 for m in _MODES}
    best = max(kr, key=kr.get)
    print("-" * 64)
    print(f"\nBest keyword recall: {best.upper()} ({kr[best]:.1%})")
    print(f"   Hybrid vs BM25:   {kr['hybrid'] - kr['bm25']:+.1%}")
    print(f"   Hybrid vs Vector: {kr['hybrid'] - kr['vector']:+.1%}")
    print(f"\nArtifacts: {_RESULTS}/ablation_summary.csv (+ ablation_<mode>.csv)\n")


if __name__ == "__main__":
    asyncio.run(main())
