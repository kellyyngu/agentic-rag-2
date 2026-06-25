import os
import time
import asyncio
import httpx
from loguru import logger

from agent.state import AgentState


async def run(state: AgentState) -> AgentState:
    t0 = time.time()
    query = state["query"]
    logger.info(f"[web_search] query='{query}'")

    results = []
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        logger.warning("[web_search] no TAVILY_API_KEY — skipping web search")
    else:
        try:
            resp = await asyncio.to_thread(lambda: httpx.post(
                "https://api.tavily.com/search",
                json={"api_key": key, "query": query, "max_results": 5},
                timeout=10.0,
            ))
            for r in resp.json().get("results", []):
                results.append({
                    "title": r.get("title", ""),
                    "body": r.get("content", ""),
                    "href": r.get("url", ""),
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
