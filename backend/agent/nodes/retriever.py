import re
import time
from typing import Any
from loguru import logger

from agent.state import AgentState, RetrievedChunk
from config import settings

# If the query explicitly references a document, never reclassify away from document_qa
_DOC_REF_RE = re.compile(
    r"\b(report|document|doc|file|pdf|paper|article|upload|uploaded)\b",
    re.IGNORECASE,
)


async def run(state: AgentState, retriever: Any) -> AgentState:
    t0 = time.time()

    # On the reflection retry, search the reflector's feedback; else the original query.
    if state.get("retrieved_chunks") and state.get("iteration_count", 0) > 0:
        queries = [state["reflection_feedback"] or state["query"]]
    else:
        queries = [state["query"]]

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
    top_chunks = merged[: settings.final_top_k * 2]

    # Filter out cross-document contamination: chunks with very low vector similarity
    # are genuinely unrelated to the query and will only confuse the generator.
    # Keep at least 2 chunks even if all are below threshold (e.g. doc-summary queries).
    MIN_VECTOR_SCORE = 0.25
    filtered = [c for c in top_chunks if c.vector_score >= MIN_VECTOR_SCORE]
    if len(filtered) < 2 and top_chunks:
        filtered = top_chunks[:2]  # fallback: keep top-2 by reranker score
    top_chunks = filtered

    elapsed = time.time() - t0
    logger.info(f"[retriever] found={len(top_chunks)} chunks t={elapsed:.2f}s")

    state["retrieved_chunks"] = top_chunks
    state["search_queries_used"] = list(state.get("search_queries_used", [])) + queries

    state["trace"].setdefault("retrieval_iterations", []).append({
        "queries": queries,
        "chunks_found": len(top_chunks),
        "latency_s": elapsed,
    })

    # Compute retrieval confidence from vector cosine similarity (stable [0,1]).
    # Reranker scores are query-style dependent and can be near-zero for valid
    # doc-summary queries; vector scores are more reliable for this gate.
    if top_chunks:
        top3_vec = [c.vector_score for c in top_chunks[:3] if c.vector_score > 0]
        retrieval_confidence = sum(top3_vec) / len(top3_vec) if top3_vec else 0.0
    else:
        retrieval_confidence = 0.0

    state["retrieval_confidence"] = retrieval_confidence

    # Confidence gate: if best chunk is below threshold, documents don't cover this query.
    # Reclassify to general_knowledge so the LLM answers from its own knowledge.
    # Exception: if the query explicitly mentions a document artifact, trust the user — don't reclassify.
    query_references_doc = bool(_DOC_REF_RE.search(state["query"]))
    if retrieval_confidence < settings.retrieval_relevance_threshold and not query_references_doc:
        logger.info(
            f"[retriever] low relevance ({retrieval_confidence:.2f} < {settings.retrieval_relevance_threshold}) "
            f"— reclassifying '{state['query']}' as general_knowledge"
        )
        state["intent"] = "general_knowledge"
    elif retrieval_confidence < settings.retrieval_relevance_threshold and query_references_doc:
        logger.info(
            f"[retriever] low relevance ({retrieval_confidence:.2f}) but query references a document "
            f"— keeping document_qa"
        )

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
