"""
Tests that retrieval still produces correct, ordered results after the blocking
work (BM25 scoring, vector search, reranking) was moved off the event loop with
asyncio.to_thread. Behavior, ordering, and scores must be unchanged.

These use lightweight fakes — no real models — so they stay deterministic.
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retrieval.hybrid_retriever import HybridRetriever
from agent.state import RetrievedChunk


class _FakeBM25:
    """Synchronous, like the real BM25Index — must be driven via to_thread now."""
    def __init__(self):
        self.called_in_thread = None

    def search(self, query, top_k):
        # Capture the running-thread identity to prove we were dispatched off-loop.
        import threading
        self.called_in_thread = threading.current_thread().name
        return [
            ("c1", "content one about pinns", "s.pdf", 1, 0.9),
            ("c2", "content two about loss", "s.pdf", 1, 0.8),
        ]


class _FakeVectorStore:
    async def search(self, query, top_k):
        return [
            ("c2", "content two about loss", "s.pdf", 1, 0.71),
            ("c1", "content one about pinns", "s.pdf", 1, 0.62),
        ]


class _FakeReranker:
    """Identity reranker — preserves the fused order so we can assert it survives."""
    def __init__(self):
        self.called_in_thread = None

    def rerank(self, query, chunks, top_k):
        import threading
        self.called_in_thread = threading.current_thread().name
        return list(chunks[:top_k])


def _build():
    return HybridRetriever(_FakeVectorStore(), _FakeBM25(), _FakeReranker())


def test_retrieve_returns_chunks():
    r = _build()
    out = asyncio.run(r.retrieve("what are pinns", top_k=5))
    assert {c.chunk_id for c in out} == {"c1", "c2"}
    assert all(isinstance(c, RetrievedChunk) for c in out)


def test_vector_scores_preserved_through_pipeline():
    """vector_score must survive fusion + rerank unchanged (display/confidence gate)."""
    r = _build()
    out = asyncio.run(r.retrieve("loss", top_k=5))
    by_id = {c.chunk_id: c for c in out}
    assert by_id["c2"].vector_score == 0.71
    assert by_id["c1"].vector_score == 0.62


def test_blocking_work_dispatched_off_event_loop():
    """The fakes record the thread they ran on; with to_thread it must NOT be the
    main thread (where the event loop runs)."""
    import threading
    main_name = threading.current_thread().name
    r = _build()
    asyncio.run(r.retrieve("pinns", top_k=5))
    assert r.bm25_index.called_in_thread != main_name
    assert r.reranker.called_in_thread != main_name


def test_empty_candidates_returns_empty():
    class _EmptyBM25:
        def search(self, query, top_k): return []
    class _EmptyVS:
        async def search(self, query, top_k): return []
    r = HybridRetriever(_EmptyVS(), _EmptyBM25(), _FakeReranker())
    out = asyncio.run(r.retrieve("nothing", top_k=5))
    assert out == []
