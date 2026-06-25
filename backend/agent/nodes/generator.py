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
2. Cite sources inline using ONLY the bracketed source numbers shown in the CONTEXT
   (the [1], [2], ... that label each context block). Group multiple sources as [1, 3].
   NEVER reproduce reference numbers that appear *inside* the source text itself
   (e.g. a paper's own bibliography markers like "[7]") — only the context labels.
3. If partial information is available, provide what you know and indicate gaps.
4. Keep answers concise: 150–400 words for most questions. Only go longer if the user
   explicitly asks for a detailed summary or comparison.
5. Use markdown structure only when the answer has multiple distinct sections.
6. METADATA — output this block ONLY AFTER your answer text is 100% complete.
   Every sentence in your answer must be finished before you write <<<JSON.
   Format (one line of JSON, no backticks, no excerpts):
   <<<JSON
   {"follow_up_questions":["q1?","q2?","q3?"],"confidence_score":0.0}
   >>>"""

GENERATOR_PROMPT = """CONTEXT:
{context}

WEB SEARCH RESULTS (supplementary):
{web_results}

CONVERSATION HISTORY:
{history}

USER QUERY: {query}

Write a complete, grounded answer with inline citations [1], [2], etc.
Finish every sentence before outputting the <<<JSON metadata block."""


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


# Phrases that signal the answer is a "not found" / information-absence reply.
# Deliberately about *missing information*, not about content (so "OSM-PINN does not
# use symmetric penalties" — a substantive claim — does NOT match).
_NEGATIVE_MARKERS = (
    "does not mention", "doesn't mention", "not mentioned", "no mention of",
    "does not discuss", "doesn't discuss", "does not contain", "doesn't contain",
    "does not provide", "doesn't provide", "does not specify", "doesn't specify",
    "does not appear", "doesn't appear", "not appear in",
    "no information", "no relevant information", "could not find", "couldn't find",
    "not found in", "is not mentioned", "isn't mentioned", "not contain any",
    "i cannot provide", "i can't provide", "i could not find", "unable to find",
    "context does not", "context doesn't", "documents do not", "documents don't",
)


def _is_negative_answer(answer: str) -> bool:
    """True when the answer is an information-absence reply ("X is not mentioned").

    Such answers must not carry citations or high confidence — the cited chunks
    do not *support* the claim, they merely failed to contain the requested fact.
    """
    low = answer.lower()
    return any(marker in low for marker in _NEGATIVE_MARKERS)


def _extract_cited_ids(answer: str, valid_ids: set[str]) -> list[str]:
    """Return the ordered, de-duplicated chunk indices the answer actually cites.

    Handles grouped citations — [3], [2, 4], [2, 4, 6] are all parsed. Any bracketed
    number that is NOT a real retrieved-chunk index is ignored (academic papers embed
    their own bibliography markers like "[7]" in the body text, which must never
    become citations). Returns indices sorted ascending by value.
    """
    out: list[str] = []
    for group in re.findall(r'\[([\d,\s]+)\]', answer):
        for num in re.findall(r'\d+', group):
            if num in valid_ids and num not in out:
                out.append(num)
    out.sort(key=int)
    return out


def _remap_citation_groups(answer: str, local_to_global: dict[str, str]) -> str:
    """Rewrite every [N] / [N, M, ...] group from local context indices to global IDs.

    Numbers that don't map to a retrieved chunk (the source's own references) are
    dropped from the group. A single regex pass with a replacement function avoids
    the partial-replacement bug where rewriting [1] before [10] corrupts "[10]".
    """
    def _sub(match: re.Match) -> str:
        mapped: list[str] = []
        for num in re.findall(r'\d+', match.group(1)):
            g = local_to_global.get(num)
            if g and g not in mapped:
                mapped.append(g)
        return "[" + ", ".join(mapped) + "]" if mapped else ""

    return re.sub(r'\[([\d,\s]+)\]', _sub, answer)


# ── Evidence-centric snippet extraction ────────────────────────────────────
# When a source is cited, the card should show the passage of that chunk most
# relevant to the cited claim — not just the chunk's opening 500 chars. This is
# deterministic and LLM-free (keyword overlap), consistent with the project's
# other scoring heuristics, and adds no latency or model calls.

_SENT_SPLIT = re.compile(r'(?<=[.!?])\s+(?=[A-Z(0-9])')

# Small stopword set so overlap scoring keys on meaningful terms, not filler.
_STOPWORDS = frozenset("""
a an the and or but of to in on at for with from by as is are was were be been
this that these those it its their they them which who what when where how why
we our you your i me my be can could would should may might will not no
into than then over under such only also more most other some any each both
""".split())


# PyPDF2 Symbol-font glyph codes → readable characters.
# These appear when a PDF embeds math/symbol characters in a non-standard font
# that PyPDF2 cannot decode (e.g. APA-style stats: r = .58, p < .001).
_PDF_SYMBOL_MAP: dict[str, str] = {
    '/H11005': '=',   # equals sign
    '/H11021': '<',   # less than
    '/H11022': '>',   # greater than
    '/H11349': '±',   # plus-minus
    '/H11011': '−',   # minus (en-dash)
    '/H11015': '≈',   # approximately equal
    '/H11003': '×',   # multiplication
    '/H11001': '+',   # plus (sometimes encoded)
    '/H11002': '−',   # minus (alternate)
    '/H11032': '′',   # prime
    '/H11018': '≤',   # less-than-or-equal
    '/H11019': '≥',   # greater-than-or-equal
    '/H9251':  'α',   # alpha
    '/H9252':  'β',   # beta
    '/H9254':  'δ',   # delta
    '/H9262':  'μ',   # mu
    '/H9268':  'σ',   # sigma
    '/H9273':  'χ',   # chi
    '/H9274':  'ψ',   # psi
}
# Compiled once — matches any known glyph code surrounded by word boundaries.
_PDF_SYMBOL_RE = re.compile(
    '|'.join(re.escape(k) for k in _PDF_SYMBOL_MAP),
)


def _clean_pdf_text(text: str) -> str:
    """Repair PDF extraction artifacts for readable display.

    Replaces Symbol-font glyph codes (e.g. /H11005 → =, /H11021 → <) that
    PyPDF2 emits when it cannot decode embedded math fonts. Also joins
    line-break hyphenation ("funda- mental" → "fundamental") while keeping
    real compound hyphens ("fixed-weight"), then collapses whitespace.
    """
    text = _PDF_SYMBOL_RE.sub(lambda m: _PDF_SYMBOL_MAP[m.group(0)], text)
    text = re.sub(r'(\w)-\s+(\w)', r'\1\2', text)
    return re.sub(r'\s+', ' ', text).strip()


def _keywords(text: str) -> set[str]:
    return {w for w in re.findall(r'[a-z]+', text.lower()) if len(w) > 2 and w not in _STOPWORDS}


def _claim_for(local_id: str, answer: str, query: str) -> str:
    """The text a citation should be matched against: the answer sentence(s) that
    cite this source. Falls back to the user query if none can be isolated."""
    citing = []
    for sent in re.split(r'(?<=[.!?])\s+', answer):
        if any(local_id in re.findall(r'\d+', g) for g in re.findall(r'\[([\d,\s]+)\]', sent)):
            citing.append(sent)
    return " ".join(citing).strip() or query


def _evidence_snippet(content: str, claim: str, max_chars: int = 600) -> str:
    """Return the passage of `content` most relevant to `claim`, with neighbouring
    context — picked by keyword overlap, trimmed to whole sentences (never mid-word).
    """
    clean = _clean_pdf_text(content)
    sentences = [s.strip() for s in _SENT_SPLIT.split(clean) if s.strip()]
    if not sentences:
        return clean[:max_chars]

    kw = _keywords(claim)
    scores = [len(kw & _keywords(s)) for s in sentences] if kw else [0] * len(sentences)
    best = max(range(len(sentences)), key=lambda i: scores[i]) if any(scores) else 0

    # Window: best sentence plus one neighbour on each side (1–3 sentences).
    lo, hi = max(0, best - 1), min(len(sentences), best + 2)
    snippet = " ".join(sentences[lo:hi]).strip()

    if len(snippet) > max_chars:
        cut = snippet[:max_chars]
        boundary = max(cut.rfind('. '), cut.rfind('! '), cut.rfind('? '))
        snippet = (cut[:boundary + 1] if boundary > max_chars // 2 else cut.rsplit(' ', 1)[0]).strip()

    prefix = "… " if lo > 0 else ""
    suffix = " …" if hi < len(sentences) else ""
    return f"{prefix}{snippet}{suffix}".strip()


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

    # Rewrite inline [N] / [N, M] references to global IDs in a single pass.
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
        # Weak / negative / off-topic answer: drop the misleading citations and the
        # now-dangling inline markers, and report low, honest confidence.
        citations = []
        answer = re.sub(r'\s*\[[\d,\s]+\]', '', answer).strip()
        confidence = round(min(settings.confidence_ungrounded_cap, retrieval_conf), 3)
    elif has_web and not citations:
        # Web-grounded answer (no document citations). Base confidence on the LLM's
        # self-rating with a floor — we DID retrieve live results to ground it.
        confidence = max(0.0, min(1.0, round(
            settings.confidence_web_base + llm_score * settings.confidence_web_llm_weight, 3
        )))
    else:
        # Document-grounded answer: blend LLM self-rating, retrieval relevance, and
        # citation coverage.
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
    pending = ""          # not-yet-emitted text (may hold a partial sentinel)
    stopped_emit = False  # once the sentinel is seen, never emit again

    try:
        response = _client.models.generate_content_stream(
            model=settings.llm_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=GENERATOR_SYSTEM,
                # 2048 tokens gives room for complex multi-part questions (e.g.
                # "compare all three models and their trade-offs") without truncation.
                # Sentinel-stop keeps the JSON metadata invisible in the live stream.
                # Prompt instruction keeps simple answers under 400 words.
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
                # Sentinel reached — emit everything before it, then go silent.
                visible = pending[:idx]
                if visible:
                    await q_stream.put({"event": "token", "data": {"text": visible}})
                stopped_emit = True
                pending = ""
            elif len(pending) > _KEEP:
                # Emit all but a short tail that might be the start of the sentinel.
                await q_stream.put({"event": "token", "data": {"text": pending[:-_KEEP]}})
                pending = pending[-_KEEP:]
        # No sentinel ever appeared — flush whatever is left.
        if q_stream and not stopped_emit and pending:
            await q_stream.put({"event": "token", "data": {"text": pending}})
    except Exception as e:
        logger.error(f"[generator] streaming failed: {e}")
        full_text = f"I encountered an error generating the response: {e}"

    # --- Step 1: parse the streamed text into answer + metadata ---
    answer, follow_ups, llm_score = parse_generation(full_text)

    # Build a lookup from 1-based citation index → actual retrieved chunk.
    # This index is canonical: it matches the [N] numbers given to the LLM in the context.
    chunks = state.get("retrieved_chunks", [])
    chunk_map = {str(i + 1): chunk for i, chunk in enumerate(chunks)}

    # Pre-build local→global ID mapping using the per-session CitationManager.
    # This is the only place IDs are assigned — all turns in a session share its counter.
    manager = state.get("citation_manager")
    local_to_global: dict[str, str] = {}
    for local_id, chunk in chunk_map.items():
        if manager:
            local_to_global[local_id] = manager.get_or_assign(chunk.chunk_id)
        else:
            local_to_global[local_id] = local_id  # graceful fallback

    # --- Step 2: build citations from inline [N] references and remap to global IDs ---
    citations, answer = build_citations(answer, chunk_map, local_to_global, state["query"])

    # --- Step 3: grounding-aware confidence + citation calibration ---
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
