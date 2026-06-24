"""
Drives the real LangGraph orchestrator and captures a normalized EvalTrace.

Key design choice: we call `graph.ainvoke(state)` DIRECTLY rather than consuming
the SSE event stream. ainvoke returns the full final AgentState (with
retrieved_chunks, citations, answer, and the orchestrator trace), which is
everything the metrics need — and it never streams, so stream_queue is None.

This means ZERO changes to core RAG logic; we only read what the graph produces.
"""
from __future__ import annotations

import time
from typing import Any

from loguru import logger

from agent.graph import build_graph
from evaluate.trace_schema import EvalTrace


def _build_initial_state(query: str, citation_manager: Any) -> dict:
    """Mirror the initial_state created inside agent.graph.run_agent."""
    return {
        "query": query,
        "conversation_history": [],
        "intent": "document_qa",
        "sub_questions": [],
        "retrieval_strategy": "agentic",
        "needs_web_search": False,
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
        "stream_queue": None,        # <-- no streaming during evaluation
        "citation_manager": citation_manager,
    }


async def run_query(query: str, retriever: Any, citation_manager: Any) -> EvalTrace:
    """Execute one query through the full agent graph and return its EvalTrace."""
    graph = build_graph(retriever)
    state = _build_initial_state(query, citation_manager)

    t0 = time.time()
    try:
        final_state = await graph.ainvoke(state)
        latency = time.time() - t0
        trace = EvalTrace.from_state(query, final_state, latency_s=latency)
        logger.info(
            f"[eval-runner] '{query[:50]}' intent={trace.intent} "
            f"tools={trace.num_tool_calls} chunks={len(trace.retrieved_chunks)} "
            f"t={latency:.1f}s"
        )
        return trace
    except Exception as e:
        latency = time.time() - t0
        logger.error(f"[eval-runner] '{query[:50]}' FAILED: {e}")
        return EvalTrace.from_state(query, None, latency_s=latency, error=str(e))
