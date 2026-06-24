from typing import List, Dict, Tuple, Optional
from loguru import logger

from agent.state import RetrievedChunk
from retrieval.vector_store import VectorStore
from retrieval.bm25_index import BM25Index
from retrieval.reranker import Reranker
from config import settings


def _reciprocal_rank_fusion(
    *ranked_lists: List[Tuple[str, float]],
    k: int = 60,
) -> Dict[str, float]:
    """RRF fusion — k=60 is the standard constant."""
    scores: Dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, (doc_id, _) in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


class HybridRetriever:
    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        reranker: Reranker,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.reranker = reranker

        # Cache chunk content for reranking (chunk_id → (content, source, page))
        self._chunk_cache: Dict[str, Tuple[str, Optional[str], Optional[int]]] = {}

    async def retrieve(self, query: str, top_k: int = 5) -> List[RetrievedChunk]:
        # 1. BM25 retrieval
        bm25_results = self.bm25_index.search(query, top_k=settings.bm25_top_k)
        bm25_ranked = [(chunk_id, score) for chunk_id, _, _, _, score in bm25_results]
        for chunk_id, content, source, page, score in bm25_results:
            self._chunk_cache[chunk_id] = (content, source, page)

        # 2. Vector retrieval
        vector_results = await self.vector_store.search(query, top_k=settings.vector_top_k)
        vector_ranked = [(chunk_id, score) for chunk_id, _, _, _, score in vector_results]
        for chunk_id, content, source, page, score in vector_results:
            self._chunk_cache[chunk_id] = (content, source, page)

        # 3. RRF fusion
        fused_scores = _reciprocal_rank_fusion(bm25_ranked, vector_ranked)
        fused_sorted = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        # 4. Build candidates for reranking
        candidates: List[RetrievedChunk] = []
        for chunk_id, rrf_score in fused_sorted[: settings.rerank_top_k]:
            content, source, page = self._chunk_cache.get(chunk_id, ("", None, None))
            if content:
                candidates.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    content=content,
                    source=source or "unknown",
                    page=page,
                    score=rrf_score,
                ))

        if not candidates:
            logger.warning(f"[hybrid] no candidates for query='{query}'")
            return []

        # 5. CrossEncoder reranking
        reranked = self.reranker.rerank(query, candidates, top_k=top_k)
        logger.info(f"[hybrid] query='{query[:50]}' bm25={len(bm25_results)} vec={len(vector_results)} after_rerank={len(reranked)}")
        return reranked
