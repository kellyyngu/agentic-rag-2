#!/usr/bin/env python3
"""
Evaluation runner for Agentic RAG v2.

Usage:
    python evaluation/run_eval.py --api http://localhost:8000

Metrics:
    Precision@K  — fraction of retrieved chunks that are relevant
    Recall@K     — fraction of relevant documents found
    Groundedness — how much of the answer is grounded in retrieved context
    Citation Acc — fraction of cited sources matching ground truth

Results are printed to stdout and saved to evaluation/results.json.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Install httpx: pip install httpx")
    sys.exit(1)


DATASET_PATH = Path(__file__).parent / "dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.json"


async def run_query(client: httpx.AsyncClient, api_base: str, query: str) -> dict:
    answer = ""
    citations = []
    confidence = 0.0
    t0 = time.time()

    async with client.stream(
        "POST",
        f"{api_base}/api/chat",
        json={"query": query, "conversation_history": []},
        timeout=120.0,
    ) as resp:
        resp.raise_for_status()
        buffer = ""
        async for chunk in resp.aiter_text():
            buffer += chunk
            parts = buffer.split("\n\n")
            buffer = parts.pop()
            for part in parts:
                lines = part.strip().split("\n")
                event = ""
                data_str = ""
                for line in lines:
                    if line.startswith("event: "):
                        event = line[7:]
                    elif line.startswith("data: "):
                        data_str = line[6:]
                if not event or not data_str:
                    continue
                try:
                    data = json.loads(data_str)
                except Exception:
                    continue
                if event == "token":
                    answer += data.get("text", "")
                elif event == "citations":
                    citations = data.get("citations", [])
                elif event == "reflection":
                    confidence = data.get("confidence", 0.0)

    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
        "latency_s": round(time.time() - t0, 2),
    }


def precision_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    if not retrieved or not relevant:
        return 0.0
    top_k = retrieved[:k]
    rel_set = set(s.lower() for s in relevant)
    hits = sum(1 for s in top_k if s.lower() in rel_set)
    return hits / k


def recall_at_k(retrieved: list, relevant: list, k: int = 5) -> float:
    if not retrieved or not relevant:
        return 0.0
    top_k = set(s.lower() for s in retrieved[:k])
    rel_set = set(s.lower() for s in relevant)
    hits = len(top_k & rel_set)
    return hits / len(rel_set)


def groundedness(answer: str, expected: str) -> float:
    import re
    if not answer or not expected:
        return 0.0
    expected_words = set(expected.lower().split())
    answer_words = set(answer.lower().split())
    overlap = len(expected_words & answer_words)
    return min(overlap / max(len(expected_words), 1), 1.0)


async def main(api_base: str):
    with open(DATASET_PATH) as f:
        dataset = json.load(f)

    print(f"\n{'='*60}")
    print(f"Agentic RAG v2 — Evaluation Suite")
    print(f"API: {api_base}  |  Items: {len(dataset)}")
    print(f"{'='*60}\n")

    results = []
    totals = {"p5": 0, "r5": 0, "gs": 0, "ca": 0, "latency": 0}

    async with httpx.AsyncClient() as client:
        for item in dataset:
            print(f"[{item['id']}] {item['query'][:60]}...")
            try:
                resp = await run_query(client, api_base, item["query"])
                cited_sources = [c.get("source", "") for c in resp["citations"]]
                p5 = precision_at_k(cited_sources, item["ground_truth_sources"])
                r5 = recall_at_k(cited_sources, item["ground_truth_sources"])
                gs = groundedness(resp["answer"], item["expected_answer"])
                ca = 1.0 if not item["ground_truth_sources"] else precision_at_k(cited_sources, item["ground_truth_sources"])

                print(f"  Precision@5={p5:.2f}  Recall@5={r5:.2f}  Groundedness={gs:.2f}  Latency={resp['latency_s']}s")

                result = {
                    **item,
                    "generated_answer": resp["answer"][:300],
                    "citations": cited_sources,
                    "confidence": resp["confidence"],
                    "metrics": {
                        "precision_at_5": p5,
                        "recall_at_5": r5,
                        "groundedness": gs,
                        "citation_accuracy": ca,
                        "latency_s": resp["latency_s"],
                    },
                }
                results.append(result)
                totals["p5"] += p5
                totals["r5"] += r5
                totals["gs"] += gs
                totals["ca"] += ca
                totals["latency"] += resp["latency_s"]
            except Exception as e:
                print(f"  ERROR: {e}")

    n = max(len(results), 1)
    print(f"\n{'='*60}")
    print(f"AGGREGATE RESULTS  (n={len(results)})")
    print(f"{'='*60}")
    print(f"  Avg Precision@5  : {totals['p5']/n:.3f}")
    print(f"  Avg Recall@5     : {totals['r5']/n:.3f}")
    print(f"  Avg Groundedness : {totals['gs']/n:.3f}")
    print(f"  Avg Citation Acc : {totals['ca']/n:.3f}")
    print(f"  Avg Latency      : {totals['latency']/n:.1f}s")
    print(f"{'='*60}\n")

    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://localhost:8000", help="Backend API base URL")
    args = parser.parse_args()
    asyncio.run(main(args.api))
