import time
from typing import Any
from loguru import logger

from agent.state import AgentState, RetrievedChunk
from config import settings


async def run(state: AgentState, retriever: Any) -> AgentState:
    t0 = time.time()

    # Use sub-questions as search queries, fall back to original query
    queries = state.get("sub_questions") or [state["query"]]
    # Don't re-search if we already have chunks from a prior iteration
    if state.get("retrieved_chunks") and state.get("iteration_count", 0) > 0:
        queries = [state["reflection_feedback"] or state["query"]]

    logger.info(f"[retriever] queries={queries}")

    all_chunks: list[RetrievedChunk] = []
    seen_ids: set[str] = {c.chunk_id for c in state.get("retrieved_chunks", [])}

    for q in queries:
        chunks = await retriever.retrieve(q, top_k=settings.final_top_k)
        for chunk in chunks:
            if chunk.chunk_id not in seen_ids:
                all_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)

    # Merge with existing chunks, keep top-scored
    merged = list(state.get("retrieved_chunks", [])) + all_chunks
    merged.sort(key=lambda c: c.score, reverse=True)
    top_chunks = merged[: settings.final_top_k * 2]  # keep more for generation context

    elapsed = time.time() - t0
    logger.info(f"[retriever] found={len(top_chunks)} chunks t={elapsed:.2f}s")

    state["retrieved_chunks"] = top_chunks
    state["search_queries_used"] = list(state.get("search_queries_used", [])) + queries

    state["trace"].setdefault("retrieval_iterations", []).append({
        "queries": queries,
        "chunks_found": len(top_chunks),
        "latency_s": elapsed,
    })

    # Stream chunk metadata to frontend (not full content)
    q_stream = state.get("stream_queue")
    if q_stream and all_chunks:
        await q_stream.put({
            "event": "chunks",
            "data": {
                "count": len(all_chunks),
                "sources": list({c.source for c in all_chunks}),
            },
        })

    return state
