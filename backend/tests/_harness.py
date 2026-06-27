"""
Shared offline builders for the safety test layer (tests/test_safety.py).

Self-contained copies of the minimal builders the safety tests need, so the
existing trajectory suite is left untouched — TrajectoryRunner and its topology
tables stay in test_trajectories.py. Everything here is offline and
deterministic: the external boundaries (Gemini, retriever, Tavily) are mocked.
"""
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.nodes import intent_router, generator, reflector
from agent import orchestrator
from agent.state import RetrievedChunk
from agent.citation_manager import CitationManager


def chunk(cid, vscore, score=None, content="OSM-PINN reduces RUL violation rates.",
          source="OSM_PINN.pdf", page=1) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid, content=content, source=source, page=page,
        score=score if score is not None else vscore, vector_score=vscore,
    )


def fresh_state(query, history=None) -> dict:
    """Mirror run_agent()'s initial_state, but offline (no queue)."""
    return {
        "query": query,
        "conversation_history": history or [],
        "intent": "document_qa",
        "retrieved_chunks": [],
        "search_queries_used": [],
        "web_search_results": [],
        "answer": "",
        "citations": [],
        "follow_up_questions": [],
        "retrieval_confidence": 0.0,
        "reflection_passed": False,
        "reflection_feedback": None,
        "confidence_score": 0.0,
        "iteration_count": 0,
        "trace": {"start_time": time.time()},
        "stream_queue": None,
        "citation_manager": CitationManager(),
    }


class FakeRetriever:
    """Returns each batch in turn; repeats the last batch once exhausted."""

    def __init__(self, *batches):
        self._queue = list(batches)
        self._last: list = []
        self.calls: list[str] = []

    async def retrieve(self, query, top_k=5):
        self.calls.append(query)
        if self._queue:
            self._last = self._queue.pop(0)
        return list(self._last)


def _ns(text):
    return SimpleNamespace(text=text)


def _stream(text):
    return [SimpleNamespace(text=text)]


def install_genai(monkeypatch, *, router_texts=None, orch_stop=True,
                  gen_streams=None, refl_passed=None):
    """Mock the per-module Gemini `_client`s at exactly the call shapes each uses."""
    if router_texts is not None:
        m = MagicMock()
        m.models.generate_content.side_effect = [_ns(t) for t in router_texts]
        monkeypatch.setattr(intent_router, "_client", m)
    if orch_stop:
        # candidates=[] → orchestrator parses zero function-calls → stops the loop.
        mo = MagicMock()
        mo.models.generate_content.return_value = SimpleNamespace(candidates=[])
        monkeypatch.setattr(orchestrator, "_client", mo)
    if gen_streams is not None:
        mg = MagicMock()
        mg.models.generate_content_stream.side_effect = [_stream(t) for t in gen_streams]
        monkeypatch.setattr(generator, "_client", mg)
    if refl_passed is not None:
        mr = MagicMock()
        mr.models.generate_content.return_value = _ns(
            json.dumps({"passed": refl_passed, "should_retrieve_more": not refl_passed})
        )
        monkeypatch.setattr(reflector, "_client", mr)


def fc_response(name, query):
    """Build a Gemini response carrying a single function call — lets loop-safety
    tests simulate an LLM that keeps requesting tools instead of stopping."""
    part = SimpleNamespace(
        function_call=SimpleNamespace(name=name, args={"query": query})
    )
    content = SimpleNamespace(parts=[part])
    return SimpleNamespace(candidates=[SimpleNamespace(content=content)])
