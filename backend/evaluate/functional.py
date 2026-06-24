"""
Functional test harness (A1-A8).

Loads declarative test cases from datasets/functional_cases.json, runs each
through the real agent graph, and evaluates pass/fail assertions against the
captured EvalTrace.

Outputs:
  results/functional_results.json   (full per-case detail)
  results/functional_summary.csv    (one row per case)
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger

from evaluate.agent_runner import run_query
from evaluate.trace_schema import EvalTrace

_HERE = Path(__file__).parent
_CASES_FILE = _HERE / "datasets" / "functional_cases.json"
_RESULTS_DIR = _HERE / "results"


# ── Individual assertion checks ─────────────────────────────────────────────
# Each returns (passed: bool, detail: str)

def _check_no_error(trace: EvalTrace, expected: bool) -> tuple[bool, str]:
    ok = (trace.error is None) == expected
    return ok, f"error={trace.error!r}"


def _check_first_tool(trace: EvalTrace, expected: str) -> tuple[bool, str]:
    ok = trace.first_tool == expected
    return ok, f"first_tool={trace.first_tool!r} expected={expected!r}"


def _check_tool_calls_count(trace: EvalTrace, spec: dict) -> tuple[bool, str]:
    op, val = spec["op"], spec["value"]
    n = trace.num_tool_calls
    ok = {
        "==": n == val, "<=": n <= val, ">=": n >= val,
        "<": n < val, ">": n > val,
    }[op]
    return ok, f"tool_calls={n} {op} {val}"


def _check_intent_in(trace: EvalTrace, allowed: list[str]) -> tuple[bool, str]:
    ok = trace.intent in allowed
    return ok, f"intent={trace.intent!r} allowed={allowed}"


def _check_sources_all_from(trace: EvalTrace, tokens: list[str]) -> tuple[bool, str]:
    """All retrieved sources must match at least one allowed token (substring)."""
    srcs = trace.retrieved_sources
    if not srcs:
        return False, "no sources retrieved"
    toks = [t.lower() for t in tokens]
    bad = [s for s in srcs if not any(t in s.lower() for t in toks)]
    return len(bad) == 0, f"unexpected_sources={bad}" if bad else f"all {len(srcs)} sources ok"


def _check_answer_contains_any(trace: EvalTrace, needles: list[str]) -> tuple[bool, str]:
    ans = trace.final_answer.lower()
    hits = [n for n in needles if n.lower() in ans]
    return len(hits) > 0, f"matched={hits}" if hits else f"none of {needles} in answer"


def _check_reflection_passed(trace: EvalTrace, expected: bool) -> tuple[bool, str]:
    ok = trace.reflection_passed == expected
    return ok, f"reflection_passed={trace.reflection_passed}"


def _check_citations_resolve(trace: EvalTrace, expected: bool) -> tuple[bool, str]:
    """Every inline [N] in the answer must exist in the citations id set."""
    inline = trace.inline_citation_ids()
    cited_ids = {str(c.get("id")) for c in trace.citations}
    if not inline:
        # No inline refs is acceptable (some answers cite nothing) -> treat as pass
        return expected, "no inline [N] references"
    unresolved = [i for i in inline if i not in cited_ids]
    ok = (len(unresolved) == 0) == expected
    return ok, f"unresolved={unresolved}" if unresolved else f"all {len(inline)} refs resolve"


def _check_min_citations(trace: EvalTrace, minimum: int) -> tuple[bool, str]:
    n = len(trace.citations)
    return n >= minimum, f"citations={n} >= {minimum}"


def _check_max_latency(trace: EvalTrace, max_s: float) -> tuple[bool, str]:
    return trace.latency_s <= max_s, f"latency={trace.latency_s:.1f}s <= {max_s}s"


_CHECKS = {
    "no_error": _check_no_error,
    "first_tool": _check_first_tool,
    "tool_calls_count": _check_tool_calls_count,
    "intent_in": _check_intent_in,
    "sources_all_from": _check_sources_all_from,
    "answer_contains_any": _check_answer_contains_any,
    "reflection_passed": _check_reflection_passed,
    "citations_resolve": _check_citations_resolve,
    "min_citations": _check_min_citations,
    "max_latency_s": _check_max_latency,
}


def _evaluate_assertions(trace: EvalTrace, asserts: dict) -> tuple[bool, list[dict]]:
    checks = []
    all_passed = True
    for key, expected in asserts.items():
        fn = _CHECKS.get(key)
        if fn is None:
            checks.append({"check": key, "passed": False, "detail": "unknown assertion"})
            all_passed = False
            continue
        passed, detail = fn(trace, expected)
        checks.append({"check": key, "passed": passed, "detail": detail})
        all_passed = all_passed and passed
    return all_passed, checks


async def run_functional_suite(retriever: Any, citation_manager: Any) -> dict:
    cases = json.loads(_CASES_FILE.read_text(encoding="utf-8"))["cases"]
    logger.info(f"[functional] running {len(cases)} test cases")

    results = []
    for case in cases:
        trace = await run_query(case["query"], retriever, citation_manager)
        passed, checks = _evaluate_assertions(trace, case.get("assert", {}))
        status = "PASS" if passed else "FAIL"
        logger.info(f"[functional] {case['id']} [{status}] {case['category']}")
        results.append({
            "id": case["id"],
            "category": case["category"],
            "objective": case["objective"],
            "query": case["query"],
            "component": case.get("component", ""),
            "status": status,
            "checks": checks,
            "trace": trace.to_dict(),
        })

    total = len(results)
    passed = sum(1 for r in results if r["status"] == "PASS")

    # Per-category breakdown
    categories: dict[str, dict] = {}
    for r in results:
        c = categories.setdefault(r["category"], {"total": 0, "passed": 0})
        c["total"] += 1
        if r["status"] == "PASS":
            c["passed"] += 1

    summary = {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 3) if total else 0.0,
        "by_category": categories,
    }

    _write_outputs(results, summary)
    logger.info(f"[functional] DONE — {passed}/{total} passed ({summary['pass_rate']:.0%})")
    return summary


def _write_outputs(results: list[dict], summary: dict) -> None:
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    (_RESULTS_DIR / "functional_results.json").write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )

    with (_RESULTS_DIR / "functional_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "category", "status", "first_tool", "num_tool_calls",
                    "intent", "latency_s", "objective"])
        for r in results:
            t = r["trace"]
            w.writerow([
                r["id"], r["category"], r["status"],
                t.get("first_tool"), t.get("num_tool_calls"),
                t.get("intent"), round(t.get("latency_s", 0), 2),
                r["objective"],
            ])
