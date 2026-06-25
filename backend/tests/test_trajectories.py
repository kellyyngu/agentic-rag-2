"""
Trajectory (multi-step) tests for the agent graph.

WHY THIS EXISTS
---------------
The unit suite (RRF, citations, router, safe-fail routing, …) validates
individual functions. But several production bugs lived in the *interaction*
between nodes across steps — things no single-function test could catch:

  1. A correct web answer was discarded after reflection.
  2. Reflection re-retrieved documents even though web evidence was sufficient.
  3. The system downgraded to general_knowledge after already having an answer.
  4. Confidence/routing interactions produced the wrong sink.

These tests drive a full trajectory and assert BOTH the node sequence and the
final state fields.

HARNESS DESIGN
--------------
`TrajectoryRunner` executes the REAL node `run()` functions and the REAL routing
functions imported from `agent.graph`, walking the EXACT topology that
`build_graph()` wires up (see agent/graph.py — the CONDITIONAL/FIXED tables
below are a 1:1 transcription of its edges). Only the external boundaries are
mocked: the Gemini client (`_client` in each node module), the retriever
service, and Tavily web search. No API keys, fully deterministic, offline.

We drive the nodes directly rather than LangGraph's compiled runtime because
(a) the existing conftest stubs `langgraph` for offline CI, and (b) these bugs
live in our routing/state logic, not in LangGraph's generic executor — so this
gives precise, assertable path recording over exactly the code under test.
"""
import asyncio
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.graph import (
    _route_after_intent,
    _route_after_orchestrator,
    _route_after_retrieval,
    _route_after_web_search,
    _should_continue,
    direct_node,
    safe_fail_node,
)
from agent.nodes import intent_router, retriever, web_search, generator, reflector
from agent import orchestrator
from agent.state import RetrievedChunk
from agent.citation_manager import CitationManager


# ── Topology: a 1:1 transcription of build_graph() in agent/graph.py ─────────
# Conditional edges: node -> (routing_fn, {routing_key: next_node})
CONDITIONAL = {
    "router":       (_route_after_intent,       {"direct": "direct", "orchestrator": "orchestrator", "web_search": "web_search"}),
    "orchestrator": (_route_after_orchestrator,  {"direct": "direct", "generate": "generator", "safe_fail": "safe_fail"}),
    "retriever":    (_route_after_retrieval,     {"direct": "direct", "web_search": "web_search", "generate": "generator"}),
    "web_search":   (_route_after_web_search,    {"generate": "generator", "safe_fail": "safe_fail"}),
    "reflector":    (_should_continue,           {"retrieve": "retriever", "end": "END"}),
}
# Fixed edges: node -> next_node
FIXED = {"generator": "reflector", "direct": "END", "safe_fail": "END"}


class TrajectoryRunner:
    """Walk the real graph topology with the real nodes, recording the path."""

    def __init__(self, retriever_service):
        self.rs = retriever_service
        self.path: list[str] = []
        self.handlers = {
            "router":       lambda s: intent_router.run(s),
            "orchestrator": lambda s: orchestrator.run(s, self.rs),
            "retriever":    lambda s: retriever.run(s, self.rs),
            "web_search":   lambda s: web_search.run(s),
            "generator":    lambda s: generator.run(s),
            "reflector":    lambda s: reflector.run(s),
            "direct":       lambda s: direct_node(s),
            "safe_fail":    lambda s: safe_fail_node(s),
        }

    async def run(self, state, max_steps: int = 25):
        node = "router"
        while node != "END":
            if len(self.path) >= max_steps:
                raise AssertionError(
                    f"trajectory exceeded {max_steps} steps (possible infinite loop): {self.path}"
                )
            self.path.append(node)
            state = await self.handlers[node](state)
            node = self._next(node, state)
        return state

    def _next(self, node, state) -> str:
        if node in CONDITIONAL:
            fn, mapping = CONDITIONAL[node]
            return mapping[fn(state)]
        return FIXED[node]


# ── Fixtures / builders ──────────────────────────────────────────────────────

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


def gen_text(answer, follow_ups=None, conf=0.8) -> str:
    """A generator output: prose + the trailing <<<JSON>>> metadata block."""
    meta = json.dumps({"follow_up_questions": follow_ups or [], "confidence_score": conf})
    return f"{answer}\n<<<JSON\n{meta}\n>>>"


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


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: a correct web answer is preserved (not discarded by reflection)
# ─────────────────────────────────────────────────────────────────────────────
def test_t1_web_answer_preserved(monkeypatch):
    async def fake_tavily(q, max_results=5):
        return [{"title": "KL weather", "body": "32°C and sunny in Kuala Lumpur today", "href": "http://w"}]
    monkeypatch.setattr(web_search, "tavily_search", fake_tavily)

    # Docs would only matter if the retriever ran on this path — it must not.
    rs = FakeRetriever([chunk("d1", 0.05)])
    install_genai(
        monkeypatch,
        gen_streams=[gen_text("It is 32°C and sunny in Kuala Lumpur today.", ["Humidity?"], conf=0.1)],
        refl_passed=False,  # reflection FAILS — the bug was that this discarded the answer
    )

    runner = TrajectoryRunner(rs)
    state = asyncio.run(runner.run(fresh_state("What is the weather in Kuala Lumpur today?")))

    assert runner.path == ["router", "web_search", "generator", "reflector"]
    assert "retriever" not in runner.path        # NOT re-triggered after reflection
    assert rs.calls == []                          # retriever never touched
    assert "Kuala Lumpur" in state["answer"]       # final answer is the web answer
    assert state["web_search_results"]             # grounded in live web evidence
    assert state["citations"] == []                # web answer carries no doc citations


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: a document answer stays document-grounded (no web escalation)
# ─────────────────────────────────────────────────────────────────────────────
def test_t2_document_answer_grounded(monkeypatch):
    rs = FakeRetriever([
        chunk("c1", 0.55, 0.80, "OSM-PINN applies an asymmetric loss penalising under-degradation."),
        chunk("c2", 0.52, 0.75, "Evaluated on the NASA C-MAPSS benchmark FD001-FD004."),
        chunk("c3", 0.50, 0.70, "Violation rate reduced from 3.62% to 2.06%."),
    ])
    install_genai(
        monkeypatch,
        router_texts=["document_qa"],
        gen_streams=[gen_text("OSM-PINN uses an asymmetric loss formulation [1, 2].", ["Datasets?"], conf=0.8)],
    )

    runner = TrajectoryRunner(rs)
    state = asyncio.run(runner.run(fresh_state("What is OSM PINN?")))

    assert runner.path == ["router", "orchestrator", "generator", "reflector"]
    assert "web_search" not in runner.path
    assert state["web_search_results"] == []       # no unnecessary web escalation
    assert len(state["citations"]) >= 1            # grounded with citations
    assert state["intent"] == "document_qa"


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: reflection retry uses improved retrieval (retry happens exactly once)
# ─────────────────────────────────────────────────────────────────────────────
def test_t3_reflection_retry_improves_retrieval(monkeypatch):
    weak = [
        chunk("w1", 0.20, 0.30, "OSM-PINN is broadly related to RUL."),
        chunk("w2", 0.18, 0.25, "Some tangential background."),
    ]
    strong = [
        chunk("s1", 0.55, 0.80, "OSM-PINN reduces the violation rate from 3.62% to 2.06%."),
        chunk("s2", 0.53, 0.78, "Asymmetric ReLU-based penalty design."),
        chunk("s3", 0.51, 0.76, "Evaluated across C-MAPSS subsets."),
    ]
    rs = FakeRetriever(weak, strong)
    install_genai(
        monkeypatch,
        router_texts=["document_qa"],
        gen_streams=[
            gen_text("Based on limited context, OSM-PINN relates to RUL [1].", conf=0.5),
            gen_text("OSM-PINN reduces the violation rate from 3.62% to 2.06% [1].", conf=0.85),
        ],
        refl_passed=False,  # first reflection fails → triggers the retry
    )

    runner = TrajectoryRunner(rs)
    state = asyncio.run(runner.run(fresh_state("What are the OSM PINN results?")))

    assert runner.path.count("retriever") == 1     # retried EXACTLY once
    assert runner.path.count("generator") == 2     # re-generated after retry
    assert runner.path[-1] == "reflector"          # terminated at reflection cap
    assert len(rs.calls) == 2                       # forced retrieve + one retry
    assert len(state["citations"]) >= 1            # final answer grounded in strong evidence
    assert state["retrieval_confidence"] >= 0.5    # improved retrieval confidence
    assert "2.06%" in state["answer"]              # final answer uses the improved chunk


# ─────────────────────────────────────────────────────────────────────────────
# Test 4a: both sources empty → graceful general_knowledge downgrade
# ─────────────────────────────────────────────────────────────────────────────
def test_t4a_both_empty_downgrades_to_general_knowledge(monkeypatch):
    rs = FakeRetriever([])  # documents return nothing
    install_genai(
        monkeypatch,
        router_texts=["document_qa", "A flux capacitor is fictional tech from Back to the Future."],
    )

    runner = TrajectoryRunner(rs)
    state = asyncio.run(runner.run(fresh_state("What are the specs of the Flux Capacitor 9000?")))

    assert state["intent"] == "general_knowledge"  # downgraded after finding nothing
    assert runner.path[-1] == "direct"
    assert "generator" not in runner.path          # generator never ran
    assert state["citations"] == []                # → no hallucinated citations


# ─────────────────────────────────────────────────────────────────────────────
# Test 4b: weak retrieval + no web → safe_fail (refusal, no fabricated content)
# ─────────────────────────────────────────────────────────────────────────────
def test_t4b_weak_no_web_triggers_safe_fail(monkeypatch):
    rs = FakeRetriever([chunk("c1", 0.10, 0.20, "barely related fragment")])
    install_genai(monkeypatch, router_texts=["document_qa"])

    runner = TrajectoryRunner(rs)
    state = asyncio.run(runner.run(fresh_state("What does the report say about quantum gravity?")))

    assert runner.path[-1] == "safe_fail"
    assert "generator" not in runner.path
    assert state["citations"] == []
    assert state["confidence_score"] == 0.0
    assert "couldn't find" in state["answer"].lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: a valid web answer is NOT downgraded into general_knowledge
#          (reproduces the exchange-rate bug from the logs)
# ─────────────────────────────────────────────────────────────────────────────
def test_t5_web_answer_not_downgraded(monkeypatch):
    async def fake_tavily(q, max_results=5):
        return [{"title": "USD/MYR", "body": "1 USD = 4.7 MYR today", "href": "http://fx"}]
    monkeypatch.setattr(web_search, "tavily_search", fake_tavily)

    rs = FakeRetriever()  # must never be consulted
    install_genai(
        monkeypatch,
        gen_streams=[gen_text("1 USD is about 4.7 MYR today.", ["Euro rate?"], conf=0.1)],
        refl_passed=False,  # low-confidence reflection — the trap that caused the bug
    )

    runner = TrajectoryRunner(rs)
    state = asyncio.run(runner.run(fresh_state("What is the current USD to MYR exchange rate?")))

    assert runner.path == ["router", "web_search", "generator", "reflector"]
    assert "retriever" not in runner.path          # not downgraded into doc re-retrieval
    assert state["intent"] == "web_search"         # NOT downgraded to general_knowledge
    assert state["reflection_passed"] is False     # reflection DID fail…
    assert "4.7" in state["answer"]                # …yet the valid web answer survived
    assert state["confidence_score"] < 0.5         # honestly low confidence, but preserved
    assert rs.calls == []


# ─────────────────────────────────────────────────────────────────────────────
# Test 6: negative ("not found") answer → citations suppressed, confidence low
# ─────────────────────────────────────────────────────────────────────────────
def test_t6_negative_answer_suppresses_citations(monkeypatch):
    rs = FakeRetriever([
        chunk("c1", 0.22, 0.40, "OSM-PINN discusses RUL prediction and monotonicity."),
        chunk("c2", 0.20, 0.35, "Asymmetric loss design details."),
    ])
    install_genai(
        monkeypatch,
        router_texts=["document_qa"],
        # Adversarial: a 'not found' answer that still tries to cite [1] with high self-confidence.
        gen_streams=[gen_text("The provided documents do not mention quantum entanglement [1].", conf=0.85)],
        refl_passed=True,
    )

    runner = TrajectoryRunner(rs)
    state = asyncio.run(runner.run(fresh_state("Does the document discuss quantum entanglement?")))

    assert state["citations"] == []                # citations suppressed for a non-answer
    assert state["confidence_score"] <= 0.25       # low, despite the LLM's 0.85 self-rating
    assert "[1]" not in state["answer"]            # dangling citation marker stripped
    assert "do not mention" in state["answer"].lower()
