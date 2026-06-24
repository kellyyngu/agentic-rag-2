"""
Standalone service bootstrap for the evaluation CLI.

Mirrors the FastAPI lifespan in main.py so the evaluation harness can build the
exact same retriever stack outside the web server. Requires Qdrant to be
reachable (run inside the backend container, or point QDRANT_HOST at localhost).
"""
from __future__ import annotations

from loguru import logger

from retrieval.vector_store import VectorStore
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from agent.citation_manager import CitationManager
from config import settings


class EvalServices:
    """Holds the retriever stack + citation manager for evaluation runs."""

    def __init__(self) -> None:
        self.retriever: HybridRetriever | None = None
        self.citation_manager: CitationManager | None = None

    async def initialize(self) -> "EvalServices":
        logger.info("[eval-services] initializing retriever stack...")
        vector_store = VectorStore()
        await vector_store.initialize()

        bm25 = BM25Index(data_dir=settings.data_dir)
        bm25.load()

        reranker = Reranker()

        self.retriever = HybridRetriever(
            vector_store=vector_store,
            bm25_index=bm25,
            reranker=reranker,
        )
        self.citation_manager = CitationManager()

        doc_count = await vector_store.count()
        logger.info(f"[eval-services] ready — {doc_count} vectors in store")
        if doc_count == 0:
            logger.warning(
                "[eval-services] vector store is EMPTY. Upload documents first, "
                "otherwise document-QA test cases will all fail."
            )
        return self
