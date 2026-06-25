import asyncio
import hashlib
from typing import List, Tuple, Optional
import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    ScoredPoint,
)

from config import settings
from retrieval.document_processor import DocumentChunk


def _point_id(chunk_id: str) -> int:
    """Deterministic, collision-resistant Qdrant point ID derived from chunk_id.

    Python's built-in hash() is randomized per process (PYTHONHASHSEED), so the
    same chunk produced a *different* point ID after every restart — re-ingestion
    silently created duplicate vectors instead of upserting in place. SHA-256 is
    stable across processes; we take the top 63 bits (Qdrant point IDs are
    unsigned integers) which keeps the value well within range while preserving
    collision resistance for any realistic corpus size.
    """
    digest = hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()
    return int(digest[:16], 16) & 0x7FFFFFFFFFFFFFFF


class VectorStore:
    def __init__(self):
        self.client: Optional[QdrantClient] = None
        self.model: Optional[SentenceTransformer] = None
        self.collection = settings.qdrant_collection

    async def initialize(self):
        logger.info(f"[vector_store] connecting to qdrant at {settings.qdrant_host}:{settings.qdrant_port}")
        self.client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        self.model = SentenceTransformer(settings.embedding_model)

        try:
            self.client.get_collection(self.collection)
            logger.info(f"[vector_store] collection '{self.collection}' exists")
        except Exception:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=settings.embedding_dim, distance=Distance.COSINE),
            )
            logger.info(f"[vector_store] created collection '{self.collection}'")

    def embed(self, texts: List[str]) -> np.ndarray:
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    async def upsert(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        # Embedding + Qdrant upsert are blocking CPU/IO — run off the event loop.
        return await asyncio.to_thread(self._upsert_blocking, chunks)

    def _upsert_blocking(self, chunks: List[DocumentChunk]) -> int:
        texts = [c.content for c in chunks]
        vectors = self.embed(texts)

        points = [
            PointStruct(
                id=_point_id(c.chunk_id),
                vector=vectors[i].tolist(),
                payload={
                    "chunk_id": c.chunk_id,
                    "content": c.content,
                    "source": c.source,
                    "page": c.page,
                    "chunk_index": c.chunk_index,
                    **c.metadata,
                },
            )
            for i, c in enumerate(chunks)
        ]

        self.client.upsert(collection_name=self.collection, points=points)
        logger.info(f"[vector_store] upserted {len(points)} points")
        return len(points)

    async def search(self, query: str, top_k: int = 20) -> List[Tuple[str, str, Optional[str], Optional[int], float]]:
        """Returns list of (chunk_id, content, source, page, score)."""
        if not self.client:
            return []
        # Embedding inference + vector search are blocking — run off the event loop
        # so concurrent SSE streams aren't stalled. Results/ordering are unchanged.
        return await asyncio.to_thread(self._search_blocking, query, top_k)

    def _search_blocking(self, query: str, top_k: int) -> List[Tuple[str, str, Optional[str], Optional[int], float]]:
        vec = self.embed([query])[0].tolist()
        results: List[ScoredPoint] = self.client.search(
            collection_name=self.collection,
            query_vector=vec,
            limit=top_k,
        )

        return [
            (
                r.payload.get("chunk_id", str(r.id)),
                r.payload.get("content", ""),
                r.payload.get("source"),
                r.payload.get("page"),
                float(r.score),
            )
            for r in results
        ]

    async def count(self) -> int:
        if not self.client:
            return 0
        info = self.client.get_collection(self.collection)
        return info.points_count or 0

    async def list_sources(self) -> list[dict]:
        """Return unique documents: [{source, chunk_count, pages}]."""
        if not self.client:
            return []
        seen: dict[str, dict] = {}
        offset = None
        while True:
            results, next_offset = self.client.scroll(
                collection_name=self.collection,
                with_payload=True,
                limit=100,
                offset=offset,
            )
            for r in results:
                src = r.payload.get("source", "unknown")
                if src not in seen:
                    seen[src] = {"source": src, "chunk_count": 0, "pages": set()}
                seen[src]["chunk_count"] += 1
                page = r.payload.get("page")
                if page:
                    seen[src]["pages"].add(page)
            if next_offset is None:
                break
            offset = next_offset
        return [
            {"source": v["source"], "chunk_count": v["chunk_count"], "page_count": len(v["pages"])}
            for v in seen.values()
        ]

    async def delete_collection(self):
        if self.client:
            self.client.delete_collection(self.collection)
            await self.initialize()
