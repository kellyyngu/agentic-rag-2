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
    """If orchestrator found nothing and downgraded intent, answer from LLM directly."""
    if state.get("intent") == "general_knowledge":
        return "direct"
    return "generate"


def _route_after_retrieval(state: AgentState) -> str:
    """Used only for the reflector retry path, not the initial pass."""
    intent = state.get("intent", "document_qa")
    if intent == "general_knowledge":
        return "direct"
    chunks = state.get("retrieved_chunks", [])
    explicit_web = state.get("needs_web_search", False)
    low_coverage = len(chunks) < settings.web_search_fallback_threshold
    return "web_search" if (explicit_web or low_coverage) else "generate"


def _should_continue(state: AgentState) -> str:
    passed    = state.get("reflection_passed", True)
    iteration = state.get("iteration_count", 0)
    if passed or iteration >= settings.max_reflection_iterations:
        return "end"
    return "retrieve"


def build_graph(retriever_service: Any) -> Any:
    async def router_node(state: AgentState) -> AgentState:
        return await intent_router.run(state)

    async def direct_node(state: AgentState) -> AgentState:
        intent = state.get("intent", "general_knowledge")
        if not state.get("answer"):
            await intent_router._handle_direct(state, intent)
        return state

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

    workflow.set_entry_point("router")

    # Initial routing: conversational → direct, web_search intent → web_search, else → orchestrator
    workflow.add_conditional_edges(
        "router",
        _route_after_intent,
        {"direct": "direct", "orchestrator": "orchestrator", "web_search": "web_search"},
    )
    workflow.add_edge("direct", END)

    # After orchestrator: if nothing found, downgraded to general_knowledge → direct
    workflow.add_conditional_edges(
        "orchestrator",
        _route_after_orchestrator,
        {"direct": "direct", "generate": "generator"},
    )

    # Reflector retry loop (unchanged): poor answer → re-retrieve → re-generate
    workflow.add_conditional_edges(
        "retriever",
        _route_after_retrieval,
        {"direct": "direct", "web_search": "web_search", "generate": "generator"},
    )
    workflow.add_edge("web_search", "generator")
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
) -> AsyncGenerator[dict, None]:
    queue: asyncio.Queue = asyncio.Queue(maxsize=0)

    initial_state: AgentState = {
        "query":               query,
        "conversation_history": conversation_history,
        "intent":              "document_qa",
        "sub_questions":       [],
        "retrieval_strategy":  "agentic",
        "needs_web_search":    False,
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
