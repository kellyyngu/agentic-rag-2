"""
Minimal ReAct-style orchestrator — replaces the static planner → retriever chain.

The LLM decides at each iteration whether to:
  - retrieve_documents (search uploaded PDFs)
  - web_search         (search the internet)
  - stop               (respond with no tool call → context is ready)

After the loop exits, state['retrieved_chunks'] and state['web_search_results']
are populated. The downstream generator.py runs unchanged.
"""
import asyncio
import time
from typing import Any

from google import genai
from google.genai import types
from loguru import logger

from agent.state import AgentState, RetrievedChunk
from config import settings

_client = genai.Client(api_key=settings.gemini_api_key)

MAX_ITERATIONS = 3
QUALITY_THRESHOLD = 0.40  # avg vector_score; at/above this = sufficient context
MIN_GOOD_CHUNKS = 3       # stop early once we have this many high-quality chunks

# ─────────────────────────────────────────────────────────────
# Tool declarations — Gemini native function calling
# ─────────────────────────────────────────────────────────────

_TOOLS = types.Tool(function_declarations=[
    types.FunctionDeclaration(
        name="retrieve_documents",
        description=(
            "Search the uploaded document knowledge base (PDFs, reports) for relevant chunks. "
            "Use first for questions about specific files or domain content. "
            "The response includes quality=GOOD (≥40% avg relevance) or WEAK (<40%). "
            "If WEAK, either rewrite the query and call again, or switch to web_search."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "The search query. Be specific and targeted. "
                        "If a prior call returned WEAK, rephrase to be more precise."
                    ),
                ),
            },
            required=["query"],
        ),
    ),
    types.FunctionDeclaration(
        name="web_search",
        description=(
            "Search the internet for real-time, current, or general information. "
            "Use when: the question is about current events, live data, "
            "or when retrieve_documents consistently returns WEAK results."
        ),
        parameters=types.Schema(
            type=types.Type.OBJECT,
            properties={
                "query": types.Schema(type=types.Type.STRING),
            },
            required=["query"],
        ),
    ),
])

_SYSTEM = """\
You are a research orchestrator. Your only job is to gather context by calling tools.

Rules:
1. For questions about uploaded files, reports, or specific domain terms → call retrieve_documents.
2. For current events, live data, or if documents clearly won't help → call web_search.
3. If retrieve_documents returns WEAK quality: rewrite the query and retry, OR switch to web_search.
4. Stop calling tools (respond with no function call) when:
   - You received quality=GOOD from retrieval (≥40% avg relevance), OR
   - You have useful web results, OR
   - The question is a greeting or needs no external context.
5. Never write the final answer — only decide WHEN to stop and WHICH tools to call."""


# ─────────────────────────────────────────────────────────────
# Main entry point
# ─────────────────────────────────────────────────────────────

async def run(state: AgentState, retriever_service: Any) -> AgentState:
    """
    LLM-driven ReAct loop. Populates retrieved_chunks + web_search_results.
    Exits when the LLM stops calling tools or MAX_ITERATIONS is reached.
    """
    t0 = time.time()
    q_stream = state.get("stream_queue")
    query = state["query"]

    history_text = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in state.get("conversation_history", [])[-2:]
    ) or "None"

    contents: list[types.Content] = [
        types.Content(role="user", parts=[types.Part(text=(
            f"Recent conversation:\n{history_text}\n\n"
            f"User question: {query}\n\n"
            "Decide which tools (if any) to call to gather context."
        ))])
    ]

    accumulated_chunks: list[RetrievedChunk] = []
    accumulated_web:    list[dict]           = []
    seen_ids:           set[str]             = set()
    call_log:           list[dict]           = []
    call_signatures:    set[str]             = set()  # loop-guard: same call twice → break

    for iteration in range(MAX_ITERATIONS):
        logger.info(
            f"[orchestrator] iter={iteration} "
            f"chunks={len(accumulated_chunks)} web={len(accumulated_web)}"
        )

        # ── LLM decision step ──────────────────────────────────────────────
        response = await asyncio.to_thread(
            _client.models.generate_content,
            model=settings.llm_model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=_SYSTEM,
                tools=[_TOOLS],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(mode="AUTO")
                ),
                max_output_tokens=256,
                temperature=0.0,
            ),
        )

        # Gemini can return None content when rate-limited or on safety blocks
        candidate = response.candidates[0] if response.candidates else None
        parts = []
        if candidate and getattr(candidate, "content", None) and getattr(candidate.content, "parts", None):
            parts = candidate.content.parts

        fn_calls = [
            p for p in parts
            if getattr(p, "function_call", None) and p.function_call.name
        ]

        if not fn_calls:
            # LLM chose not to call any tool — context is ready
            logger.info(f"[orchestrator] iter={iteration} — LLM stopped (no tool call)")
            break

        # Feed model turn back into conversation
        contents.append(types.Content(role="model", parts=parts))

        result_parts: list[types.Part] = []

        for part in fn_calls:
            fc   = part.function_call
            name = fc.name
            args = dict(fc.args or {})
            tool_query = args.get("query", query)

            # Loop-guard: identical call twice → break immediately
            sig = f"{name}:{tool_query.lower().strip()}"
            if sig in call_signatures:
                logger.warning(f"[orchestrator] duplicate call detected ({sig!r}) — stopping loop")
                fn_calls = []  # signal outer loop to break
                break
            call_signatures.add(sig)

            logger.info(f"[orchestrator] → {name}(query={tool_query!r})")

            # Emit tool call to SSE so frontend can show it in the trace
            if q_stream:
                await q_stream.put({
                    "event": "agent_action",
                    "data": {"tool": name, "args": args, "iteration": iteration},
                })

            # ── Execute tool ───────────────────────────────────────────────
            if name == "retrieve_documents":
                obs = await _exec_retrieve(
                    tool_query, retriever_service, accumulated_chunks, seen_ids
                )
            elif name == "web_search":
                obs = await _exec_web(tool_query, accumulated_web)
            else:
                obs = {
                    "summary": f"Unknown tool '{name}'.",
                    "for_llm": "Tool not found. Use retrieve_documents or web_search.",
                }

            call_log.append({"tool": name, "args": args, "obs": obs["summary"]})
            logger.info(f"[orchestrator] ← {obs['summary']}")

            # Emit observation to SSE
            if q_stream:
                await q_stream.put({
                    "event": "agent_observation",
                    "data": {"tool": name, "result": obs["summary"], "iteration": iteration},
                })

            result_parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=name,
                    response={"result": obs["for_llm"]},
                )
            ))

        # If duplicate-call guard fired, stop outer loop too
        if not fn_calls:
            break

        # Feed all tool observations back
        if result_parts:
            contents.append(types.Content(role="user", parts=result_parts))

        # ── Early exit: sufficient good chunks ──────────────────────────────
        good_chunks = [c for c in accumulated_chunks if c.vector_score >= QUALITY_THRESHOLD]
        if len(good_chunks) >= MIN_GOOD_CHUNKS:
            logger.info(
                f"[orchestrator] {len(good_chunks)} good chunks (≥{QUALITY_THRESHOLD:.0%}) — "
                "stopping early"
            )
            break

    # ── Finalise state ──────────────────────────────────────────────────────
    accumulated_chunks.sort(key=lambda c: c.vector_score, reverse=True)
    state["retrieved_chunks"]    = accumulated_chunks
    state["web_search_results"]  = accumulated_web
    state["search_queries_used"] = [
        e["args"].get("query", "") for e in call_log
    ]

    if accumulated_chunks:
        top3 = [c.vector_score for c in accumulated_chunks[:3] if c.vector_score > 0]
        state["retrieval_confidence"] = sum(top3) / len(top3) if top3 else 0.0

    # Nothing found — downgrade so graph routes to direct_node (LLM answers from knowledge)
    if not accumulated_chunks and not accumulated_web:
        logger.info("[orchestrator] no context found — downgrading to general_knowledge")
        state["intent"] = "general_knowledge"

    elapsed = time.time() - t0
    logger.info(
        f"[orchestrator] done iters={len(call_log)} "
        f"chunks={len(accumulated_chunks)} web={len(accumulated_web)} t={elapsed:.2f}s"
    )

    state["trace"]["orchestrator"] = {
        "iterations": len(call_log),
        "tool_calls": call_log,
        "elapsed_s":  round(elapsed, 2),
    }

    # Backward-compat: emit `plan` + `chunks` so existing AgentTrace works unchanged
    if q_stream and call_log:
        retrieval_queries = [
            e["args"].get("query", "") for e in call_log
            if e["tool"] == "retrieve_documents"
        ]
        await q_stream.put({"event": "plan", "data": {
            "sub_questions": retrieval_queries or [query],
            "strategy":      "agentic",
        }})
        if accumulated_chunks:
            await q_stream.put({"event": "chunks", "data": {
                "count":   len(accumulated_chunks),
                "sources": list({c.source for c in accumulated_chunks}),
            }})

    return state


# ─────────────────────────────────────────────────────────────
# Tool implementations
# ─────────────────────────────────────────────────────────────

async def _exec_retrieve(
    query: str,
    retriever_service: Any,
    accumulated_chunks: list[RetrievedChunk],
    seen_ids: set[str],
) -> dict:
    """Call the existing hybrid retriever and accumulate new chunks."""
    chunks = await retriever_service.retrieve(query, top_k=5)
    new = [c for c in chunks if c.chunk_id not in seen_ids]
    for c in new:
        accumulated_chunks.append(c)
        seen_ids.add(c.chunk_id)

    if not new:
        return {
            "summary": "No new chunks found (all already retrieved). Rewrite query or switch to web_search.",
            "for_llm": "Duplicate results only. Try a different query angle or use web_search.",
        }

    avg_score = sum(c.vector_score for c in new) / len(new)
    quality   = "GOOD" if avg_score >= QUALITY_THRESHOLD else "WEAK"
    sources   = list({c.source for c in new})

    return {
        "summary": (
            f"Retrieved {len(new)} new chunk(s) from {sources}. "
            f"Avg relevance: {avg_score:.0%} — quality={quality}"
        ),
        "for_llm": (
            f"quality={quality} avg_relevance={avg_score:.0%} new_chunks={len(new)}\n"
            f"Top result [{new[0].source} p.{new[0].page}]: {new[0].content[:300]}"
        ),
    }


async def _exec_web(query: str, accumulated_web: list[dict]) -> dict:
    """Run a DuckDuckGo search in a thread (blocking SDK)."""
    try:
        from duckduckgo_search import DDGS

        raw = await asyncio.to_thread(
            lambda: list(DDGS().text(query, max_results=4))
        )
        snippets = [
            {"title": r.get("title", ""), "body": r.get("body", ""), "href": r.get("href", "")}
            for r in raw
        ]
        accumulated_web.extend(snippets)
        preview = "\n".join(
            f"- {s['title']}: {s['body'][:150]}" for s in snippets[:2]
        )
        return {
            "summary": f"Web search returned {len(snippets)} result(s) for '{query}'",
            "for_llm": f"Found {len(snippets)} web results:\n{preview}",
        }
    except Exception as e:
        logger.warning(f"[orchestrator] web_search failed: {e}")
        return {
            "summary": "Web search unavailable (rate-limited). Proceeding with document context.",
            # Tell the LLM explicitly to stop calling tools and use what it has
            "for_llm": (
                "Web search is rate-limited and unavailable right now. "
                "STOP calling tools. Use the document chunks already retrieved to answer."
            ),
        }
