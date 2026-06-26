import asyncio
import time
from typing import Any, AsyncGenerator
from langgraph.graph import StateGraph, END
from loguru import logger

from agent.state import AgentState
from agent.nodes import intent_router, retriever, web_search, generator, reflector
from agent import orchestrator   # replaces planner
from config import settings

DIRECT_INTENTS = intent_router.DIRECT_INTENTS


def _route_after_intent(state: AgentState) -> str:
    intent = state.get("intent", "document_qa")
    if intent in DIRECT_INTENTS:
        return "direct"
    if intent == "web_search":
        # Explicit "search the web for X" queries skip the orchestrator
        return "web_search"
    return "orchestrator"


def _route_after_orchestrator(state: AgentState) -> str:
    """Route after the tool loop.

    1. Nothing found → downgraded to general_knowledge → answer from LLM knowledge.
    2. Safe-fail gate: retrieval is weak AND no web evidence → refuse instead of
       sending barely-relevant chunks to the generator (prevents hallucination).
    3. Otherwise → generate.
    """
    if state.get("intent") == "general_knowledge":
        return "direct"
    conf = state.get("retrieval_confidence", 0.0)
    has_web = bool(state.get("web_search_results"))
    if conf < settings.safe_fail_threshold and not has_web:
        return "safe_fail"
    return "generate"


def _route_after_retrieval(state: AgentState) -> str:
    """Used only for the reflector retry path, not the initial pass."""
    intent = state.get("intent", "document_qa")
    if intent == "general_knowledge":
        return "direct"
    chunks = state.get("retrieved_chunks", [])
    low_coverage = len(chunks) < settings.web_search_fallback_threshold
    return "web_search" if low_coverage else "generate"


def _route_after_web_search(state: AgentState) -> str:
    """After a web search: if we have neither web results nor document chunks,
    there is nothing to ground an answer on → safe_fail with a clear message
    instead of sending an empty context to the generator (which would refuse
    with a confusing 'the documents are about X' reply)."""
    has_web    = bool(state.get("web_search_results"))
    has_chunks = bool(state.get("retrieved_chunks"))
    return "generate" if (has_web or has_chunks) else "safe_fail"


def _should_continue(state: AgentState) -> str:
    passed    = state.get("reflection_passed", True)
    iteration = state.get("iteration_count", 0)
    if passed or iteration >= settings.max_reflection_iterations:
        return "end"
    # A web-grounded answer won't be improved by re-retrieving documents — retrying
    # would discard the live web result and fall back to stale general knowledge.
    if state.get("web_search_results"):
        return "end"
    return "retrieve"


# ── Terminal sink nodes ─────────────────────────────────────────────────────
# Module-level (not closures) so trajectory tests can drive the real code path.
# These capture no per-request dependencies (no retriever_service), so extracting
# them from build_graph is purely structural — behaviour is identical.

async def direct_node(state: AgentState) -> AgentState:
    intent = state.get("intent", "general_knowledge")
    if not state.get("answer"):
        await intent_router._handle_direct(state, intent)
    return state


async def safe_fail_node(state: AgentState) -> AgentState:
    """Production-safe fallback: weak retrieval + no web → refuse, skip the generator."""
    if state.get("intent") == "web_search":
        # Real-time query but web search produced nothing — usually because
        # TAVILY_API_KEY isn't configured. Be explicit rather than blaming the docs.
        msg = (
            "This looks like a question that needs live web data, but I couldn't "
            "retrieve any web results. If web search isn't configured, set "
            "TAVILY_API_KEY to enable real-time answers (weather, news, prices)."
        )
    else:
        msg = (
            "I couldn't find enough relevant information in the uploaded documents to "
            "answer this confidently, and no web results were available. Try rephrasing "
            "the question or adding more detail."
        )
    state["answer"] = msg
    state["citations"] = []
    state["follow_up_questions"] = []
    state["confidence_score"] = 0.0
    state["reflection_passed"] = True  # terminal — no reflection retry
    state["trace"]["safe_fail"] = {
        "triggered": True,
        "retrieval_confidence": round(state.get("retrieval_confidence", 0.0), 3),
        "reason": "low_confidence_no_web",
    }
    q = state.get("stream_queue")
    if q:
        await q.put({"event": "answer", "data": {"text": msg}})
        await q.put({"event": "citations", "data": {"citations": []}})
        await q.put({"event": "follow_ups", "data": {"questions": []}})
    return state


def build_graph(retriever_service: Any) -> Any:
    async def router_node(state: AgentState) -> AgentState:
        return await intent_router.run(state)

    async def orchestrator_node(state: AgentState) -> AgentState:
        """LLM-driven tool loop — replaces the old planner + initial retriever."""
        return await orchestrator.run(state, retriever_service)

    async def retriever_node(state: AgentState) -> AgentState:
        """Used only by the reflector retry loop, not the initial pass."""
        return await retriever.run(state, retriever_service)

    async def web_search_node(state: AgentState) -> AgentState:
        return await web_search.run(state)

    async def generator_node(state: AgentState) -> AgentState:
        return await generator.run(state)

    async def reflector_node(state: AgentState) -> AgentState:
        return await reflector.run(state)

    workflow = StateGraph(AgentState)
    workflow.add_node("router",       router_node)
    workflow.add_node("direct",       direct_node)
    workflow.add_node("orchestrator", orchestrator_node)
    workflow.add_node("retriever",    retriever_node)
    workflow.add_node("web_search",   web_search_node)
    workflow.add_node("generator",    generator_node)
    workflow.add_node("reflector",    reflector_node)
    workflow.add_node("safe_fail",    safe_fail_node)

    workflow.set_entry_point("router")

    # Initial routing: conversational → direct, web_search intent → web_search, else → orchestrator
    workflow.add_conditional_edges(
        "router",
        _route_after_intent,
        {"direct": "direct", "orchestrator": "orchestrator", "web_search": "web_search"},
    )
    workflow.add_edge("direct", END)

    # After orchestrator: nothing found → direct; weak+no-web → safe_fail; else → generate
    workflow.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"direct": "direct", "generate": "generator", "safe_fail": "safe_fail"},
    )
    workflow.add_edge("safe_fail", END)

    # Reflector retry loop (unchanged): poor answer → re-retrieve → re-generate
    workflow.add_conditional_edges(
        "retriever",
        _route_after_retrieval,
        {"direct": "direct", "web_search": "web_search", "generate": "generator"},
    )
    # After web search: ground the answer if we have evidence, else safe_fail
    workflow.add_conditional_edges(
        "web_search",
        _route_after_web_search,
        {"generate": "generator", "safe_fail": "safe_fail"},
    )
    workflow.add_edge("generator",  "reflector")
    workflow.add_conditional_edges(
        "reflector",
        _should_continue,
        {"retrieve": "retriever", "end": END},
    )

    return workflow.compile()


async def run_agent(
    query: str,
    conversation_history: list,
    retriever_service: Any,
    citation_manager: Any = None,
    graph: Any = None,
) -> AsyncGenerator[dict, None]:
    queue: asyncio.Queue = asyncio.Queue(maxsize=0)

    initial_state: AgentState = {
        "query":               query,
        "conversation_history": conversation_history,
        "intent":              "document_qa",
        "retrieved_chunks":    [],
        "search_queries_used": [],
        "web_search_results":  [],
        "answer":              "",
        "citations":           [],
        "follow_up_questions": [],
        "retrieval_confidence": 0.0,
        "reflection_passed":   False,
        "reflection_feedback": None,
        "confidence_score":    0.0,
        "iteration_count":     0,
        "trace":               {"start_time": time.time()},
        "stream_queue":        queue,
        "citation_manager":    citation_manager,
    }

    # Reuse the graph compiled once at startup (see main.py lifespan). Fall back to
    # building one on demand so direct callers (tests, eval harness) still work.
    if graph is None:
        graph = build_graph(retriever_service)

    async def run_graph():
        try:
            await graph.ainvoke(initial_state)
        except Exception as e:
            logger.error(f"[graph] error: {e}")
            await queue.put({"event": "error", "data": {"message": str(e)}})
        finally:
            await queue.put(None)

    task = asyncio.create_task(run_graph())

    while True:
        item = await queue.get()
        if item is None:
            break
        yield item

    await task
