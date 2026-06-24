import time
from loguru import logger
from duckduckgo_search import DDGS

from agent.state import AgentState


async def run(state: AgentState) -> AgentState:
    t0 = time.time()
    query = state["query"]
    logger.info(f"[web_search] query='{query}'")

    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=5):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("body", ""),
                    "href": r.get("href", ""),
                })
    except Exception as e:
        logger.warning(f"[web_search] failed: {e}")

    elapsed = time.time() - t0
    logger.info(f"[web_search] found={len(results)} results t={elapsed:.2f}s")

    state["web_search_results"] = results
    state["trace"]["web_search"] = {
        "query": query,
        "results_count": len(results),
        "latency_s": elapsed,
    }

    q_stream = state.get("stream_queue")
    if q_stream and results:
        await q_stream.put({
            "event": "web_search",
            "data": {"count": len(results)},
        })

    return state
