import math
from typing import List
from loguru import logger
from sentence_transformers import CrossEncoder

from agent.state import RetrievedChunk
from config import settings


def _sigmoid(x: float) -> float:
    """Normalize a raw cross-encoder logit to [0, 1]."""
    return 1.0 / (1.0 + math.exp(-x))


class Reranker:
    def __init__(self):
        logger.info(f"[reranker] loading {settings.reranker_model}")
        self.model = CrossEncoder(settings.reranker_model, max_length=512)
        logger.info("[reranker] ready")

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.content) for chunk in chunks]
        raw_scores = self.model.predict(pairs)

        scored = sorted(
            zip(chunks, raw_scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, raw in scored[:top_k]:
            chunk.score = _sigmoid(float(raw))
            results.append(chunk)

        return results
