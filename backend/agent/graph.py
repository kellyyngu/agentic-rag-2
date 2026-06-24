import asyncio
import time
from typing import Any, AsyncGenerator
from langgraph.graph import StateGraph, END
from loguru import logger

from agent.state import AgentState
from agent.nodes import intent_router, planner, retriever, web_search, generator, reflector
from config import settings

CONVERSATIONAL_INTENTS = intent_router.CONVERSATIONAL_INTENTS


def _route_after_intent(state: AgentState) -> str:
    return "end" if state.get("intent") in CONVERSATIONAL_INTENTS else "planner"


def _should_use_web_search(state: AgentState) -> str:
    chunks = state.get("retrieved_chunks", [])
    low_coverage = len(chunks) < settings.web_search_fallback_threshold
    explicit = state.get("needs_web_search", False)
    return "web_search" if (explicit or low_coverage) else "generate"


def _should_continue(state: AgentState) -> str:
    passed = state.get("reflection_passed", True)
    iteration = state.get("iteration_count", 0)
    if passed or iteration >= settings.max_reflection_iterations:
        return "end"
    return "retrieve"


def build_graph(retriever_service: Any) -> Any:
    async def router_node(state: AgentState) -> AgentState:
        return await intent_router.run(state)

    async def planner_node(state: AgentState) -> AgentState:
        return await planner.run(state)

    async def retriever_node(state: AgentState) -> AgentState:
        return await retriever.run(state, retriever_service)

    async def web_search_node(state: AgentState) -> AgentState:
        return await web_search.run(state)

    async def generator_node(state: AgentState) -> AgentState:
        return await generator.run(state)

    async def reflector_node(state: AgentState) -> AgentState:
        return await reflector.run(state)

    workflow = StateGraph(AgentState)
    workflow.add_node("router", router_node)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("generator", generator_node)
    workflow.add_node("reflector", reflector_node)

    workflow.set_entry_point("router")

    # Conversational → END immediately, everything else → planner → RAG pipeline
    workflow.add_conditional_edges(
        "router",
        _route_after_intent,
        {"end": END, "planner": "planner"},
    )

    workflow.add_edge("planner", "retriever")
    workflow.add_conditional_edges(
        "retriever",
        _should_use_web_search,
        {"web_search": "web_search", "generate": "generator"},
    )
    workflow.add_edge("web_search", "generator")
    workflow.add_edge("generator", "reflector")
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
) -> AsyncGenerator[dict, None]:
    queue: asyncio.Queue = asyncio.Queue(maxsize=0)  # 0 = unlimited

    initial_state: AgentState = {
        "query": query,
        "conversation_history": conversation_history,
        "intent": "document_qa",
        "sub_questions": [],
        "retrieval_strategy": "factual_lookup",
        "needs_web_search": False,
        "retrieved_chunks": [],
        "search_queries_used": [],
        "web_search_results": [],
        "answer": "",
        "citations": [],
        "follow_up_questions": [],
        "reflection_passed": False,
        "reflection_feedback": None,
        "confidence_score": 0.0,
        "iteration_count": 0,
        "trace": {"start_time": time.time()},
        "stream_queue": queue,
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
