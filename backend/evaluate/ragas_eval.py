"""
RAGAS quantitative evaluation pipeline (answer-quality axis).

Runs each benchmark question through the real agent, collects
(question, retrieved_contexts, answer, ground_truth), then scores with RAGAS
using Gemini as the judge LLM.

Metrics: Context Precision, Context Recall, Faithfulness,
         Answer Relevancy, Answer Correctness.

RAGAS + LangChain-Google are OPTIONAL heavy deps. If they are not installed,
this module degrades gracefully: it still produces the agent outputs needed for
scoring and prints clear install instructions, so functional + agentic suites
are unaffected.

Install:  pip install ragas langchain-google-genai datasets

Outputs:
  results/ragas_results.csv
  results/ragas_summary.json
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from loguru import logger

from config import settings
from evaluate.agent_runner import run_query

_HERE = Path(__file__).parent
_RESULTS_DIR = _HERE / "results"
_BENCHMARK = _HERE / "datasets" / "benchmark_dataset.json"

# Only document-grounded questions are meaningful for RAGAS context metrics.
_RAGAS_CATEGORIES = {"single_hop", "multi_hop"}


def _ragas_available() -> bool:
    try:
        import ragas  # noqa: F401
        import langchain_google_genai  # noqa: F401
        import datasets  # noqa: F401
        return True
    except Exception:
        return False


async def _collect_samples(retriever: Any, citation_manager: Any) -> list[dict]:
    """Run the agent over document-grounded questions and gather RAGAS inputs."""
    data = json.loads(_BENCHMARK.read_text(encoding="utf-8"))["questions"]
    items = [q for q in data if q["category"] in _RAGAS_CATEGORIES]
    logger.info(f"[ragas] collecting agent outputs for {len(items)} grounded questions")

    samples = []
    for q in items:
        trace = await run_query(q["question"], retriever, citation_manager)
        contexts = trace.context_texts or [c.get("excerpt", "") for c in trace.citations]
        samples.append({
            "id": q["id"],
            "category": q["category"],
            "question": q["question"],
            "answer": trace.final_answer,
            "contexts": [c for c in contexts if c],
            "ground_truth": q["ground_truth"],
            "error": trace.error,
        })
    return samples


def _score_with_ragas(samples: list[dict]) -> dict:
    """Run RAGAS scoring. Assumes deps are present (checked by caller)."""
    from datasets import Dataset
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
    from ragas import evaluate as ragas_evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.metrics import (
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_correctness,
    )

    # Drop failed runs (no answer) so they don't poison the judge
    usable = [s for s in samples if s["answer"] and not s["error"] and s["contexts"]]
    if not usable:
        logger.warning("[ragas] no usable samples (empty answers/contexts) — skipping scoring")
        return {"error": "no usable samples"}

    ds = Dataset.from_dict({
        "question":     [s["question"] for s in usable],
        "answer":       [s["answer"] for s in usable],
        "contexts":     [s["contexts"] for s in usable],
        "ground_truth": [s["ground_truth"] for s in usable],
    })

    judge_llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=0.0,
    ))
    judge_emb = LangchainEmbeddingsWrapper(GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=settings.gemini_api_key,
    ))

    metrics = [context_precision, context_recall, faithfulness,
               answer_relevancy, answer_correctness]

    logger.info(f"[ragas] scoring {len(usable)} samples with Gemini judge...")
    result = ragas_evaluate(ds, metrics=metrics, llm=judge_llm, embeddings=judge_emb)

    df = result.to_pandas()
    metric_cols = [c for c in df.columns
                   if c not in ("question", "answer", "contexts", "ground_truth")]
    aggregate = {col: round(float(df[col].mean()), 4) for col in metric_cols}

    # Persist per-sample CSV
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_RESULTS_DIR / "ragas_results.csv", index=False)

    return {"aggregate": aggregate, "n_samples": len(usable)}


async def run_ragas_suite(retriever: Any, citation_manager: Any) -> dict:
    samples = await _collect_samples(retriever, citation_manager)

    if not _ragas_available():
        msg = (
            "RAGAS not installed — skipping quantitative scoring. "
            "Install with: pip install ragas langchain-google-genai datasets"
        )
        logger.warning(f"[ragas] {msg}")
        # Still save the collected agent outputs so they aren't lost
        _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (_RESULTS_DIR / "ragas_samples.json").write_text(
            json.dumps(samples, indent=2), encoding="utf-8"
        )
        summary = {"status": "skipped", "reason": msg, "samples_collected": len(samples)}
        (_RESULTS_DIR / "ragas_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary

    try:
        scored = _score_with_ragas(samples)
    except Exception as e:
        logger.error(f"[ragas] scoring failed: {e}")
        scored = {"error": str(e)}

    summary = {"status": "completed", **scored}
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (_RESULTS_DIR / "ragas_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info(f"[ragas] DONE — {summary}")
    return summary
