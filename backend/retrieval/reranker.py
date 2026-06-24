from typing import List
from loguru import logger
from sentence_transformers import CrossEncoder

from agent.state import RetrievedChunk
from config import settings


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
        scores = self.model.predict(pairs)

        scored = sorted(
            zip(chunks, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for chunk, score in scored[:top_k]:
            chunk.score = float(score)
            results.append(chunk)

        return results
