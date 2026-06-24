import time
from typing import Dict, Any, List
from loguru import logger

from evaluation.metrics import precision_at_k, recall_at_k, groundedness_score, citation_accuracy
from agent.graph import run_agent


class Evaluator:
    async def evaluate_single(
        self,
        query: str,
        expected_answer: str,
        ground_truth_sources: List[str],
        retriever,
    ) -> Dict[str, Any]:
        t0 = time.time()
        logger.info(f"[eval] query='{query[:60]}'")

        final_state = None
        async for event in run_agent(query, [], retriever):
            # Collect the last state implicitly; we just need the final answer
            pass

        # After the graph runs, we need to get the answer and citations
        # We re-run with a collector approach
        answer = ""
        citations_raw = []
        chunks_used = []

        async for event in run_agent(query, [], retriever):
            if event.get("event") == "citations":
                citations_raw = event["data"].get("citations", [])
            elif event.get("event") == "token":
                answer += event["data"].get("text", "")

        cited_sources = [c.get("source", "") for c in citations_raw]
        retrieved_ids = [c.get("source", "") for c in citations_raw]
        relevant_ids = set(ground_truth_sources)

        p5 = precision_at_k(retrieved_ids, relevant_ids, k=5)
        r5 = recall_at_k(retrieved_ids, relevant_ids, k=5)
        gs = groundedness_score(answer, [expected_answer])
        ca = citation_accuracy(cited_sources, ground_truth_sources)
        latency = time.time() - t0

        return {
            "query": query,
            "precision_at_5": round(p5, 4),
            "recall_at_5": round(r5, 4),
            "groundedness": round(gs, 4),
            "citation_accuracy": round(ca, 4),
            "latency_s": round(latency, 2),
        }

    def get_sample_dataset(self) -> List[Dict[str, Any]]:
        return [
            {
                "query": "What is retrieval-augmented generation?",
                "expected_answer": "RAG combines information retrieval with language model generation to produce grounded, factual responses.",
                "ground_truth_sources": [],
            },
            {
                "query": "How does BM25 work?",
                "expected_answer": "BM25 is a probabilistic ranking function that scores documents based on term frequency and document length normalization.",
                "ground_truth_sources": [],
            },
            {
                "query": "What are the advantages of hybrid retrieval?",
                "expected_answer": "Hybrid retrieval combines sparse (BM25) and dense (vector) search to capture both keyword and semantic matches.",
                "ground_truth_sources": [],
            },
        ]
