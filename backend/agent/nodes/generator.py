import json
import re
import time
import asyncio
from google import genai
from google.genai import types
from loguru import logger

from config import settings
from agent.state import AgentState, Citation, RetrievedChunk
from agent.nodes.generator_prompts import GENERATOR_SYSTEM, GENERATOR_PROMPT
from agent.nodes.evidence import _evidence_snippet, _claim_for
from agent.nodes.citation_logic import (
    _is_negative_answer,
    _extract_cited_ids,
    _remap_citation_groups,
)

# Re-export private helpers consumed by test_citations.py so import paths stay stable.
from agent.nodes.evidence import _clean_pdf_text, _keywords
from agent.nodes.citation_logic import _NEGATIVE_MARKERS

_client = genai.Client(api_key=settings.gemini_api_key)


def _sanitize_json(s: str) -> str:
    """Escape raw newlines/tabs inside JSON string values that LLMs sometimes emit."""
    result = []
    in_string = False
    i = 0
    while i < len(s):
        c = s[i]
        if c == '\\' and in_string:
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

    # 4. Truncated/unclosed <<<JSON block — the model hit the token limit before
    #    emitting >>>. We can't parse the partial JSON, but we MUST still cut it
    #    from the visible answer. Return no metadata but a valid split position.
    m = re.search(r'<<<JSON[\s\S]*$', text)
    if m:
        return None, m.start()

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


# ── Post-generation pipeline (pure, independently testable) ─────────────────
# These are extracted from run() so the confidence policy can be unit-tested
# without driving a token stream. Behaviour is identical to the inline version.

def parse_generation(full_text: str) -> tuple[str, list[str], float]:
    """Strip the trailing metadata block; return (answer, follow_ups, llm_score)."""
    follow_ups: list[str] = []
    meta, split_pos = _extract_meta(full_text)
    if meta is not None:
        answer = full_text[:split_pos].strip()
        follow_ups = meta.get("follow_up_questions", [])
        llm_score = float(meta.get("confidence_score", 0.75))
        if llm_score > 1.0:
            llm_score /= 10.0
        llm_score = max(0.0, min(1.0, llm_score))
    elif split_pos >= 0:
        # Truncated/unclosed JSON block (pattern 4): no parsable metadata, but we
        # still know where it starts — cut everything from there so it never shows.
        answer = full_text[:split_pos].strip()
        llm_score = 0.75
    else:
        answer = re.sub(r'\s*<<<JSON[\s\S]*?>>>\s*', '', full_text).strip()
        answer = re.sub(r'\s*```(?:json)?\s*\{[\s\S]*?\}\s*```\s*', '', answer).strip()
        answer = re.sub(r'\s*<<<JSON[\s\S]*$', '', answer).strip()
        llm_score = 0.75
    return answer, follow_ups, llm_score


def build_citations(
    answer: str,
    chunk_map: dict[str, RetrievedChunk],
    local_to_global: dict[str, str],
    query: str,
) -> tuple[list[Citation], str]:
    """Build Citation objects from the inline [N] markers the answer actually cites,
    then rewrite those markers from local context indices to global IDs.

    Grouped citations like [2, 4] are supported; the source's own bibliography
    numbers (not in chunk_map) are filtered out. Returns (citations, remapped_answer).
    """
    cited_local_ids = _extract_cited_ids(answer, set(chunk_map))
    citations: list[Citation] = []
    seen_global: set[str] = set()
    for local_cid in cited_local_ids:
        chunk = chunk_map[local_cid]
        global_id = local_to_global.get(local_cid, local_cid)
        if global_id not in seen_global:
            # Use vector cosine similarity as the displayed relevance score.
            # The reranker score (chunk.score) is for ranking only — it gives
            # near-zero values for meta-queries like "what is my report about"
            # even when the chunks ARE relevant. Vector cosine similarity is
            # a more robust and meaningful signal for the user-facing percentage.
            display_score = chunk.vector_score if chunk.vector_score > 0 else chunk.score
            # Evidence-centric excerpt: show the passage of this chunk that actually
            # supports the sentence(s) citing it, not just the chunk's opening lines.
            claim = _claim_for(local_cid, answer, query)
            citations.append(Citation(
                id=global_id,
                source=chunk.source,
                page=chunk.page,
                excerpt=_evidence_snippet(chunk.content, claim),
                relevance_score=round(display_score, 3),
            ))
            seen_global.add(global_id)

    answer = _remap_citation_groups(answer, local_to_global)
    return citations, answer


def calibrate_confidence(
    answer: str,
    citations: list[Citation],
    retrieval_conf: float,
    has_web: bool,
    llm_score: float,
    num_chunks: int,
) -> tuple[list[Citation], str, float]:
    """Grounding-aware confidence + citation calibration.

    An answer is "grounded" only if it cites a sufficiently-relevant chunk OR is
    backed by web results — AND is not an information-absence reply. This stops
    adversarial "X is not mentioned" answers from carrying high confidence and a
    fistful of citations, and keeps confidence honest for ungrounded answers.

    All policy weights come from config (settings.confidence_*) so the scoring can
    be retuned without code changes. Returns (citations, answer, confidence).
    """
    retrieval_conf = max(0.0, min(1.0, float(retrieval_conf)))
    top_cited      = max((c.relevance_score for c in citations), default=0.0)
    is_negative    = _is_negative_answer(answer)
    is_grounded    = (top_cited >= settings.grounding_threshold or has_web) and not is_negative

    if not is_grounded:
        citations = []
        answer = re.sub(r'\s*\[[\d,\s]+\]', '', answer).strip()
        confidence = round(min(settings.confidence_ungrounded_cap, retrieval_conf), 3)
    elif has_web and not citations:
        confidence = max(0.0, min(1.0, round(
            settings.confidence_web_base + llm_score * settings.confidence_web_llm_weight, 3
        )))
    else:
        citation_support = min(len(citations) / max(num_chunks, 1), 1.0)
        confidence = max(0.0, min(1.0, round(
            llm_score * settings.confidence_doc_llm_weight
            + retrieval_conf * settings.confidence_doc_retrieval_weight
            + citation_support * settings.confidence_doc_citation_weight,
            3,
        )))
    return citations, answer, confidence


async def run(state: AgentState) -> AgentState:
    t0 = time.time()
    chunks = state.get("retrieved_chunks", [])
    logger.info(f"[generator] chunks={len(chunks)}")

    # Cap context at top-4 chunks by score — feeding 6-8 chunks makes the model
    # feel obligated to cover every point, inflating answer length significantly.
    context = _build_context(chunks[:4])
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

    # The trailing "<<<JSON ... >>>" metadata block must never reach the user.
    # We accumulate it in full_text (for citation parsing below) but stop streaming
    # to the frontend the moment the sentinel appears. A small tail is buffered so a
    # marker split across two chunks (e.g. "<<" then "<JSON") is never leaked.
    _SENTINEL = "<<<JSON"
    _KEEP = len(_SENTINEL) - 1
    pending = ""
    stopped_emit = False

    try:
        response = _client.models.generate_content_stream(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM,
                max_output_tokens=2048,
                temperature=0.1,
            ),
        )
        for chunk in response:
            if not chunk.text:
                continue
            full_text += chunk.text
            if not q_stream or stopped_emit:
                continue
            pending += chunk.text
            idx = pending.find(_SENTINEL)
            if idx != -1:
                visible = pending[:idx]
                if visible:
                    await q_stream.put({"event": "token", "data": {"text": visible}})
                stopped_emit = True
                pending = ""
            elif len(pending) > _KEEP:
                await q_stream.put({"event": "token", "data": {"text": pending[:-_KEEP]}})
                pending = pending[-_KEEP:]
        if q_stream and not stopped_emit and pending:
            await q_stream.put({"event": "token", "data": {"text": pending}})
    except Exception as e:
        logger.error(f"[generator] streaming failed: {e}")
        full_text = f"I encountered an error generating the response: {e}"

    answer, follow_ups, llm_score = parse_generation(full_text)

    chunks = state.get("retrieved_chunks", [])
    chunk_map = {str(i + 1): chunk for i, chunk in enumerate(chunks)}

    manager = state.get("citation_manager")
    local_to_global: dict[str, str] = {}
    for local_id, chunk in chunk_map.items():
        if manager:
            local_to_global[local_id] = manager.get_or_assign(chunk.chunk_id)
        else:
            local_to_global[local_id] = local_id

    citations, answer = build_citations(answer, chunk_map, local_to_global, state["query"])

    citations, answer, confidence = calibrate_confidence(
        answer=answer,
        citations=citations,
        retrieval_conf=state.get("retrieval_confidence", 0.0),
        has_web=bool(state.get("web_search_results")),
        llm_score=llm_score,
        num_chunks=len(chunks),
    )

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
