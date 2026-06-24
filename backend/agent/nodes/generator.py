import json
import re
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
5. After your answer, output a metadata block that starts with the exact line: <<<JSON
   Then the raw JSON object (no code fences, no backticks), then: >>>"""

GENERATOR_PROMPT = """CONTEXT:
{context}

WEB SEARCH RESULTS (supplementary):
{web_results}

CONVERSATION HISTORY:
{history}

USER QUERY: {query}

Instructions:
- Write a comprehensive, grounded answer with inline citations [1], [2], etc.
- After your answer output EXACTLY this block (no backticks, no markdown fences):
<<<JSON
{{"citations":[{{"id":"1","source":"filename.pdf","page":1,"excerpt":"quote"}}],"follow_up_questions":["q1?","q2?","q3?"],"confidence_score":0.85}}
>>>"""


def _sanitize_json(s: str) -> str:
    """Escape raw newlines/tabs inside JSON string values that LLMs sometimes emit."""
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string:
            # Already-escaped sequence — keep both chars as-is
            result.append(c)
            if i + 1 < len(s):
                result.append(s[i + 1])
                i += 2
            else:
                i += 1
            continue
        if c == '"':
            in_string = not in_string
            result.append(c)
            i += 1
            continue
        if in_string:
            if c == '\n':
                result.append('\\n')
            elif c == '\r':
                result.append('\\r')
            elif c == '\t':
                result.append('\\t')
            else:
                result.append(c)
        else:
            result.append(c)
        i += 1
    return ''.join(result)


def _extract_meta(text: str) -> tuple[dict | None, int]:
    """Return (parsed_meta_dict, split_position) or (None, -1)."""
    # 1. Custom delimiter <<<JSON ... >>>
    m = re.search(r'<<<JSON\s*\n?([\s\S]*?)>>>', text)
    if m:
        try:
            return json.loads(_sanitize_json(m.group(1).strip())), m.start()
        except Exception:
            pass

    # 2. Markdown fenced ```json ... ``` or ``` ... ```
    m = re.search(r'```(?:json)?\s*\n(\{[\s\S]*?\})\s*\n?```\s*$', text, re.IGNORECASE)
    if m:
        try:
            return json.loads(_sanitize_json(m.group(1).strip())), m.start()
        except Exception:
            pass

    # 3. Raw trailing JSON object
    m = re.search(r'\n(\{[\s\S]*\})\s*$', text)
    if m:
        try:
            return json.loads(_sanitize_json(m.group(1).strip())), m.start()
        except Exception:
            pass

    return None, -1


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
                full_text += chunk.text
                if q_stream:
                    # Stream whole chunk — avoids queue overflow from char-by-char
                    await q_stream.put({"event": "token", "data": {"text": chunk.text}})
    except Exception as e:
        logger.error(f"[generator] streaming failed: {e}")
        full_text = f"I encountered an error generating the response: {e}"

    answer = full_text
    citations: list[Citation] = []
    follow_ups: list[str] = []
    confidence = 0.75

    # Build a lookup from 1-based citation index → actual retrieved chunk.
    # This index is canonical: it matches the [N] numbers given to the LLM in the context.
    chunks = state.get("retrieved_chunks", [])
    chunk_map = {str(i + 1): chunk for i, chunk in enumerate(chunks)}

    # Pre-build local→global ID mapping using the CitationManager singleton.
    # This is the only place IDs are assigned — all queries share the same counter.
    manager = state.get("citation_manager")
    local_to_global: dict[str, str] = {}
    for local_id, chunk in chunk_map.items():
        if manager:
            local_to_global[local_id] = manager.get_or_assign(chunk.chunk_id)
        else:
            local_to_global[local_id] = local_id  # graceful fallback

    # --- Step 1: Strip the metadata block from the visible answer text ---
    meta, split_pos = _extract_meta(full_text)
    if meta is not None:
        answer = full_text[:split_pos].strip()
        follow_ups = meta.get("follow_up_questions", [])
        llm_score = float(meta.get("confidence_score", 0.75))
        if llm_score > 1.0:
            llm_score /= 10.0
        llm_score = max(0.0, min(1.0, llm_score))
    else:
        answer = re.sub(r'\s*<<<JSON[\s\S]*?>>>\s*', '', full_text).strip()
        answer = re.sub(r'\s*```(?:json)?\s*\{[\s\S]*?\}\s*```\s*', '', answer).strip()
        llm_score = 0.75

    # --- Step 2: Build citations from inline [N] references in the answer ---
    # Extract which context positions the LLM actually cited, then:
    #   a) Map each local [N] → global ID via CitationManager
    #   b) Rewrite [N] in the answer text to [global_id]
    # This guarantees per-document re-numbering by the LLM can never cause duplicates.
    cited_local_ids = sorted(set(re.findall(r'\[(\d+)\]', answer)), key=lambda x: int(x))
    seen_global: set[str] = set()
    for local_cid in cited_local_ids:
        chunk = chunk_map.get(local_cid)
        global_id = local_to_global.get(local_cid, local_cid)
        if chunk and global_id not in seen_global:
            # Use vector cosine similarity as the displayed relevance score.
            # The reranker score (chunk.score) is for ranking only — it gives
            # near-zero values for meta-queries like "what is my report about"
            # even when the chunks ARE relevant. Vector cosine similarity is
            # a more robust and meaningful signal for the user-facing percentage.
            display_score = chunk.vector_score if chunk.vector_score > 0 else chunk.score
            citations.append(Citation(
                id=global_id,
                source=chunk.source,
                page=chunk.page,
                excerpt=chunk.content[:500].strip(),
                relevance_score=round(display_score, 3),
            ))
            seen_global.add(global_id)

    # Remap [N] references in answer text to global IDs.
    # Sort by descending local ID value to avoid partial replacement
    # (e.g., replacing [1] before [10] would corrupt [10] → [global_1]0).
    for local_id in sorted(local_to_global.keys(), key=lambda x: -int(x)):
        global_id = local_to_global[local_id]
        if local_id != global_id:
            answer = re.sub(rf'\[{re.escape(local_id)}\]', f'[{global_id}]', answer)

    # --- Step 3: Compute confidence — all inputs clamped to [0,1] ---
    retrieval_conf = max(0.0, min(1.0, float(state.get("retrieval_confidence", 0.5))))
    citation_support = min(len(citations) / max(len(chunks), 1), 1.0)
    confidence = max(0.0, min(1.0, round(
        llm_score * 0.4 + retrieval_conf * 0.4 + citation_support * 0.2,
        3,
    )))

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
        # Send the clean answer (JSON block stripped) so the frontend replaces
        # the raw streamed content which may include the trailing JSON metadata.
        await q_stream.put({
            "event": "answer",
            "data": {"text": answer},
        })
        await q_stream.put({
            "event": "citations",
            "data": {"citations": [c.to_dict() for c in citations]},
        })
        await q_stream.put({
            "event": "follow_ups",
            "data": {"questions": follow_ups},
        })

    return state
