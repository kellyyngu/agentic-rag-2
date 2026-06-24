import json
import time
import asyncio
from typing import Any
from google import genai
from google.genai import types
from loguru import logger

from config import settings
from agent.state import AgentState, Citation, RetrievedChunk

_client = genai.Client(api_key=settings.gemini_api_key)

GENERATOR_SYSTEM = """You are an expert AI assistant that answers questions using the provided context.

RULES:
1. Answer ONLY from the provided context. If context is insufficient, say so clearly.
2. Cite sources inline using [1], [2], etc. — every factual claim must have a citation.
3. If partial information is available, provide what you know and indicate gaps.
4. Be concise but complete. Use markdown for structure when appropriate.
5. End with a JSON block (after your answer) containing citations and follow-up questions."""

GENERATOR_PROMPT = """CONTEXT:
{context}

WEB SEARCH RESULTS (supplementary):
{web_results}

CONVERSATION HISTORY:
{history}

USER QUERY: {query}

Instructions:
- Write a comprehensive, grounded answer with inline citations [1], [2], etc.
- After the answer, on a new line write ONLY valid JSON:
{{
  "citations": [
    {{"id": "1", "source": "filename.pdf", "page": 3, "excerpt": "relevant quote..."}}
  ],
  "follow_up_questions": ["question 1?", "question 2?", "question 3?"],
  "confidence_score": 0.85
}}"""


def _build_context(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return "No relevant documents found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        page_info = f", page {chunk.page}" if chunk.page else ""
        parts.append(f"[{i}] Source: {chunk.source}{page_info}\n{chunk.content}")
    return "\n\n---\n\n".join(parts)


def _build_web_context(results: list[dict]) -> str:
    if not results:
        return "None"
    return "\n".join(f"- {r['title']}: {r['body'][:300]}" for r in results[:3])


async def run(state: AgentState) -> AgentState:
    t0 = time.time()
    logger.info(f"[generator] chunks={len(state.get('retrieved_chunks', []))}")

    context = _build_context(state.get("retrieved_chunks", []))
    web_results = _build_web_context(state.get("web_search_results", []))
    history = "\n".join(
        f"{m['role'].upper()}: {m['content']}"
        for m in state.get("conversation_history", [])[-4:]
    ) or "None"

    prompt = GENERATOR_PROMPT.format(
        context=context,
        web_results=web_results,
        history=history,
        query=state["query"],
    )

    full_text = ""
    q_stream = state.get("stream_queue")

    try:
        response = _client.models.generate_content_stream(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM,
                max_output_tokens=4096,
            ),
        )
        for chunk in response:
            if chunk.text:
                for char in chunk.text:
                    full_text += char
                    if q_stream:
                        try:
                            q_stream.put_nowait({"event": "token", "data": {"text": char}})
                        except asyncio.QueueFull:
                            pass
    except Exception as e:
        logger.error(f"[generator] streaming failed: {e}")
        full_text = f"I encountered an error generating the response: {e}"

    answer = full_text
    citations: list[Citation] = []
    follow_ups: list[str] = []
    confidence = 0.75

    # Build a lookup from 1-based citation index → actual retrieved chunk
    chunks = state.get("retrieved_chunks", [])
    chunk_map = {str(i + 1): chunk for i, chunk in enumerate(chunks)}

    json_start = full_text.rfind("\n{")
    if json_start != -1:
        try:
            meta = json.loads(full_text[json_start:].strip())
            answer = full_text[:json_start].strip()
            for c in meta.get("citations", []):
                cid = str(c.get("id", ""))
                chunk = chunk_map.get(cid)
                # Always use the actual chunk text — never trust LLM-generated excerpts
                excerpt = chunk.content[:500].strip() if chunk else c.get("excerpt", "")
                citations.append(Citation(
                    id=cid,
                    source=chunk.source if chunk else c.get("source", ""),
                    page=chunk.page if chunk else c.get("page"),
                    excerpt=excerpt,
                    relevance_score=round(chunk.score, 3) if chunk else 0.9,
                ))
            follow_ups = meta.get("follow_up_questions", [])
            confidence = meta.get("confidence_score", 0.75)
        except Exception as e:
            logger.warning(f"[generator] JSON parse failed: {e}")

    elapsed = time.time() - t0
    logger.info(f"[generator] answer_len={len(answer)} citations={len(citations)} t={elapsed:.2f}s")

    state["answer"] = answer
    state["citations"] = citations
    state["follow_up_questions"] = follow_ups
    state["confidence_score"] = confidence
    state["trace"]["generator"] = {
        "answer_length": len(answer),
        "citations_count": len(citations),
        "confidence_score": confidence,
        "latency_s": elapsed,
    }

    if q_stream:
        await q_stream.put({
            "event": "citations",
            "data": {"citations": [c.to_dict() for c in citations]},
        })
        await q_stream.put({
            "event": "follow_ups",
            "data": {"questions": follow_ups},
        })

    return state
