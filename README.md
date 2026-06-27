# Agentic RAG — Document Question-Answering with an LLM-Driven Tool Loop

A production-oriented Retrieval-Augmented Generation system where an LLM **agent decides
how to gather evidence** — which tool to call, how to reformulate a weak query, when to
escalate to web search, and when it has enough context to answer. It is not a fixed
`retrieve → generate` pipeline; retrieval is a *decision*, not a hardcoded step.

**Stack:** FastAPI · LangGraph · Google Gemini 2.0 Flash (function calling) · Qdrant · BM25 ·
CrossEncoder reranker · Tavily · React + Vite (SSE streaming) · Docker Compose.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Problem Statement — why Agentic RAG](#2-problem-statement--why-agentic-rag)
3. [System Architecture](#3-system-architecture)
4. [How the Agent Works — think · act · reflect](#4-how-the-agent-works--think--act--reflect)
5. [Retrieval System — hybrid + reranking](#5-retrieval-system--hybrid--reranking)
6. [Citation & Grounding System](#6-citation--grounding-system)
7. [Generator Streaming & Output Guard](#7-generator-streaming--output-guard)
8. [Web Search Augmentation](#8-web-search-augmentation)
9. [Evaluation Framework](#9-evaluation-framework)
10. [Agentic RAG vs Traditional RAG](#10-agentic-rag-vs-traditional-rag)
11. [System Design Tradeoffs](#11-system-design-tradeoffs)
12. [Demo / How to Run](#12-demo--how-to-run)
13. [Testing Strategy — how quality is ensured](#13-testing-strategy--how-quality-is-ensured)
14. [Limitations & Future Improvements](#14-limitations--future-improvements)
15. [Repository Layout](#15-repository-layout)
16. [Tech Stack](#16-tech-stack)
17. [Conclusion](#17-conclusion)

---

## 1. Overview

The system answers questions over a corpus of uploaded PDFs (research reports) and
falls back to live web search for out-of-corpus questions. An orchestration agent runs a
bounded **think → act → reflect** loop over two tools (`retrieve_documents`,
`web_search`), grounds every factual claim with inline citations, and refuses rather than
hallucinates when retrieval is too weak to support an answer.

The repository includes a working full-stack prototype (React UI + streaming API) and a
three-axis evaluation harness (functional correctness, agentic decision quality, retrieval
ablation) that produces reproducible CSV/JSON artifacts.

---

## 2. Problem Statement — why Agentic RAG

Traditional RAG is a static chain: embed the query, retrieve top-k, stuff context into the
prompt, generate. This breaks down in several common situations:

| Failure mode | Why static RAG fails |
|---|---|
| **Vague or compound query** | A single retrieval pass over the raw query returns a diluted, low-precision chunk set. There is no opportunity to reformulate. |
| **Out-of-corpus question** | The pipeline still retrieves *something* and generates over irrelevant chunks, producing confident hallucinations. |
| **Weak first retrieval** | No mechanism to recognise low-quality evidence and try again or switch strategy. |
| **No notion of "I don't know"** | Static RAG always answers. There is no refusal path. |

Agentic RAG addresses these by giving the LLM **agency over the retrieval process**: it can
inspect retrieval quality, rewrite queries, choose a different tool, and decide to stop.
The cost is added latency and orchestration complexity — a tradeoff this system manages
explicitly with iteration bounds and early-exit conditions.

---

## 3. System Architecture

```
                         ┌─────────────────────────────────────────────┐
   user query  ─────────▶│              Intent Router                   │
                         │  document_qa │ general_knowledge │           │
                         │  web_search  │ conversational                │
                         └───────┬───────────────┬───────────────┬──────┘
                                 │               │               │
                    document_qa  │   conversational/general      │ web_search
                                 ▼               ▼               ▼
                     ┌──────────────────┐   ┌─────────┐    ┌───────────┐
                     │   Orchestrator   │   │ Direct  │    │Web Search │
                     │ (ReAct tool loop)│   │ answer  │    │ (Tavily)  │
                     └────────┬─────────┘   └────┬────┘    └─────┬─────┘
                              │                   │               │
              ┌───────────────┤                   │        ┌──────┴──────┐
              │ weak+no web   │ general_knowledge │        │  has result?│
              ▼               ▼                   │        ├─────────────┤
   ┌─────────────────┐  ┌─────────┐               │  yes → │  Generator  │
   │  Safe-Fail Gate │  │ Direct  │               │        └─────┬───────┘
   │  (refuse, no    │  └─────────┘               │  no  → │ Safe-Fail   │
   │   hallucination)│                             │        └─────────────┘
   └─────────────────┘                             │
                                                   │
              ┌──────────────────────────────────── ┘
              │
              ▼
   ┌──────────────────┐
   │ Hybrid Retriever │
   │ BM25 + Vector    │
   │ → RRF → Rerank   │
   └────────┬─────────┘
            │
            ▼
   ┌──────────────────┐
   │   Generator      │
   │ grounded answer  │
   │ + inline [N]     │
   │ grounding gate   │
   └────────┬─────────┘
            ▼
   ┌──────────────────┐   fail & iter<max & no web
   │   Reflector      │──────────────┐
   │ quality verdict  │              │ re-retrieve
   └────────┬─────────┘              ▼
            │ pass            ┌──────────────┐
            ▼                 │  Retriever   │
          DONE                │ (retry pass) │
                              └──────────────┘
```

The graph is implemented as a **LangGraph `StateGraph`** with conditional edges, compiled
**once at FastAPI startup** (`app.state.graph` in `main.py`) and reused across all requests
— the topology is static so per-request recompilation would be wasted work. State flows
through a typed `AgentState` (`backend/agent/state.py`); every node emits **Server-Sent
Events** so the React frontend renders the agent's decisions (tool calls, retrieval quality,
reflection verdict) live.

**Key routing functions** (`backend/agent/graph.py`):

| Function | Controls |
|---|---|
| `_route_after_intent` | conversational/general → direct; web_search → web_search; else → orchestrator |
| `_route_after_orchestrator` | general_knowledge → direct; weak+no web → safe_fail; else → generate |
| `_route_after_retrieval` | (reflector retry path) low chunks → web_search; else → generate |
| `_route_after_web_search` | has web or chunks → generate; nothing → safe_fail |
| `_should_continue` | passed or budget exhausted or web-grounded → end; else → retrieve |

**Why LangGraph over a hand-rolled loop:** the routing logic is expressed as explicit,
inspectable edges rather than nested conditionals. The control flow *is* the graph, making
the agent's behaviour auditable — a property that matters more than raw line count.

---

## 4. How the Agent Works — think · act · reflect

The orchestrator (`backend/agent/orchestrator.py`) is a bounded ReAct loop built on
**Gemini native function calling**:

```
for iteration in range(orchestrator_max_iterations):     # default 3
    if iteration == 0:
        # FORCED first action: always retrieve documents before the LLM decides.
        # Queries routed to orchestrator are document_qa; without forcing retrieval
        # the LLM sometimes web-searches questions the uploaded docs could answer.
        obs = retrieve_documents(query)
        if len(good_chunks) >= MIN_GOOD_CHUNKS: break
        continue

    decision = LLM.decide(tools=[retrieve_documents, web_search], context)

    if decision is "no tool call":           # THINK: context is sufficient
        break

    observation = execute(decision.tool, decision.query)  # ACT
    feed observation back into the conversation

    if len(good_chunks) >= MIN_GOOD_CHUNKS:   # early exit on strong evidence
        break
    if duplicate_call(decision):              # loop guard: same call twice → stop
        break
```

Key control mechanisms:

- **Quality signalling:** each `retrieve_documents` observation reports
  `quality=GOOD/WEAK` based on average vector relevance against
  `orchestrator_quality_threshold` (0.30, calibrated to the MiniLM cosine range). The LLM
  uses this to decide whether to reformulate, escalate, or stop.
- **Loop guard:** an identical `(tool, query)` signature seen twice breaks the loop —
  prevents the model from spinning on the same failing call.
- **Early exit:** once ≥3 high-quality chunks are accumulated, the loop stops without
  burning the remaining iteration budget.
- **Forced first retrieval:** iteration 0 always retrieves documents; the LLM only makes
  autonomous decisions from iteration 1 onward.

**Reflect (separate node):** after generation, the reflector
(`backend/agent/nodes/reflector.py`) scores the answer for completeness and grounding. If it
fails *and* the iteration budget allows *and* the answer is not web-grounded, the graph
**loops back to re-retrieve** using the reflector's feedback as a refined query, then
regenerates. This is a real closed loop (`generator → reflector → retriever → generator`),
bounded by `max_reflection_iterations` (default 2).

**Web-guard:** if the current answer is grounded in web search results,
`_should_continue` returns `"end"` even on a failed reflection — re-retrieving documents
would discard the live web result and fall back to stale general knowledge. The web answer
is preserved.

**Robustness detail:** the reflector uses Gemini **constrained decoding**
(`response_schema + response_mime_type=application/json`) combined with
`thinking_budget=0`. On Gemini Flash, thinking tokens otherwise consume the
`max_output_tokens` budget and truncate the JSON — disabling them for simple
classification eliminated a class of intermittent parse failures. The same pattern applies
to the intent router.

**Event-loop safety:** every blocking Gemini call — orchestrator, reflector, and intent
router — runs via `await asyncio.to_thread(...)` so no LLM round-trip stalls the async
event loop. The orchestrator, BM25 search, embedding inference, and Qdrant upsert were
already off-loop; the reflector and intent router LLM calls were the remaining gaps and are
now fixed. Under concurrent SSE streams, one slow reflection no longer stalls other users.

---

## 5. Retrieval System — hybrid + reranking

Retrieval is a three-stage pipeline (`backend/retrieval/hybrid_retriever.py`):

```
query
  │
  ├─▶ BM25 (rank-bm25)        top-20 lexical matches
  │
  ├─▶ Qdrant vector search    top-20 dense matches (all-MiniLM-L6-v2, 384-d cosine)
  │
  ▼
Reciprocal Rank Fusion (k=60)  rank-based merge, no score normalisation needed
  │
  ▼
CrossEncoder rerank            ms-marco-MiniLM-L-6-v2, joint query–chunk scoring
  │
  ▼
top-5 chunks  (vector_score preserved for the confidence gate + citation display)
```

Chunks below `min_vector_score = 0.10` are filtered out before reranking to prevent
off-topic cross-document contamination.

**Design rationale:**

- **Hybrid (BM25 + vector)** covers complementary failure modes: BM25 nails exact technical
  terms and acronyms (e.g. `FD004`, `OSM-PINN`); dense vectors handle paraphrase and
  semantic similarity. Domain corpora are acronym-heavy, so lexical signal genuinely
  matters.
- **RRF over weighted score fusion** because RRF is rank-based — it needs no per-corpus
  weight tuning and is robust to the fact that BM25 and cosine scores live on different,
  non-comparable scales.
- **CrossEncoder reranking** adds a precision pass that bi-encoder retrieval cannot: it
  scores the query and chunk *jointly* rather than comparing independent embeddings. This is
  the single biggest precision lever in the pipeline.
- **`vector_score` is preserved through reranking** because the reranker's logit is for
  *ordering* only and goes near-zero for valid meta/summary queries. Cosine similarity is the
  stable signal used for the confidence gate and the user-facing relevance percentage.

---

## 6. Citation & Grounding System

Every factual sentence in an answer carries an inline `[N]` marker that resolves to a real
retrieved chunk (logic split across `backend/agent/nodes/citation_logic.py`,
`backend/agent/nodes/evidence.py`, and `backend/agent/nodes/generator.py`):

1. The generator is shown context numbered `[1]…[k]` and instructed to cite inline.
2. Local `[N]` markers are remapped to **session-stable global IDs** via a per-session
   `CitationManager`, so the same chunk keeps the same citation number across turns of a
   conversation.
3. The displayed relevance score per citation is the chunk's **vector cosine similarity**,
   not the reranker logit (more meaningful to a human reader).

**Grounding gate (`_is_negative_answer`):** before finalising citations, the generator
checks whether the answer is a "not found" reply (e.g. *"the paper does not mention X"*,
*"I could not find any relevant information"*). If so:
- All citations are suppressed (no dangling `[N]` markers on a non-answer).
- `confidence_score` is capped at `≤ 0.25` regardless of the LLM's self-reported score.
- The `[N]` markers are stripped from the answer text.

This prevents the adversarial pattern where an LLM assigns high confidence (e.g. 0.85) to
a "not found" answer that happens to cite a retrieved chunk.

A separate **`grounding_threshold`** (0.30) gates citation inclusion for positive answers:
if the top cited chunk's vector score is below this threshold, citations are also suppressed
— the answer is not grounded well enough to warrant them.

**Evidence-centric snippets:** each citation card in the UI shows a passage selected from
the chunk by `_evidence_snippet` (`evidence.py`), which:
1. Tokenises the chunk into sentences.
2. Scores each sentence by keyword overlap with the *claiming sentence* from the answer
   (identified by `_claim_for`).
3. Returns the best-matching sentence ± one neighbour (for context), trimmed to whole
   sentences, max 600 characters.

This replaces a blind first-500-characters truncation; users see the passage that
actually supports the claim, not the opening boilerplate.

**Session safety:** `CitationManager` is scoped per `session_id` (or per-request when no
session id is supplied) — there is **no process-global shared citation state**, so
concurrent users never collide on or leak each other's citation IDs.

---

## 7. Generator Streaming & Output Guard

The generator streams tokens to the SSE queue via `generate_content_stream`. To prevent
the trailing JSON metadata block from appearing in the chat, the stream is intercepted by a
**sentinel-stop guard**:

```python
_SENTINEL = "<<<JSON"
_KEEP = len(_SENTINEL) - 1   # 6-char lookahead buffer

pending = ""
for chunk in response:
    pending += chunk.text
    idx = pending.find(_SENTINEL)
    if idx != -1:
        emit(pending[:idx])       # flush text before sentinel
        stop_streaming()          # suppress everything after
    elif len(pending) > _KEEP:
        emit(pending[:-_KEEP])    # emit safe prefix, hold possible sentinel start
        pending = pending[-_KEEP:]
```

The complete output (answer + JSON block) is reassembled server-side. `_extract_meta`
parses the metadata with a three-tier fallback: `<<<JSON…>>>` delimiters, fenced
` ```json ``` ` blocks, and raw trailing JSON. A fourth pattern catches **truncated**
`<<<JSON` blocks (model ran out of token budget mid-JSON) — `split_pos` is still returned
so the partial block is stripped before the answer is stored.

Generator settings: `max_output_tokens=2048`, `temperature=0.1`. The system and user prompt
templates live in `backend/agent/nodes/generator_prompts.py`.

---

## 8. Web Search Augmentation

When the intent router classifies a query as `web_search`, or when the orchestrator decides
document retrieval is consistently WEAK, the agent calls **Tavily**
(`backend/agent/nodes/web_search.py`, `orchestrator._exec_web`). Tavily is used over
DuckDuckGo because it returns reliably from datacenter IPs. Web results are fed back into the
orchestrator loop as observations and into the generator as supplementary context, clearly
separated from document context in the prompt. If no `TAVILY_API_KEY` is configured, web
search degrades gracefully and the agent answers from document context only.

---

## 9. Evaluation Framework

The harness (`backend/evaluate/`) drives the **real** LangGraph agent over a 40-question
benchmark (`single_hop`, `multi_hop`, `out_of_corpus`, `conversational`, `adversarial`) and
measures three independent axes. All runs emit reproducible CSV/JSON to
`backend/evaluate/results/`.

```bash
docker compose exec backend python -m evaluate.run_all                 # functional + agentic
docker compose exec backend python -m evaluate.run_all --suite agentic
docker compose exec backend python -m evaluate.ablation                # retrieval ablation
```

### 9.1 Functional suite
Behavioural assertions (e.g. *loop is bounded ≤ N tool calls*, *out-of-corpus escalates to
web*, *greeting calls no tool*). Pass/fail per case — a regression tripwire.

### 9.2 Agentic decision metrics (`agentic.py`)
Computed entirely from execution traces — they score the **decision process**, not just the
answer:

| Metric | What it measures |
|---|---|
| **Tool selection accuracy** | Did the first tool match the expected tool for the query category? |
| **Query reformulation success** | Of WEAK retrievals, how many recovered to GOOD after a rewrite? |
| **Web escalation accuracy** | Did out-of-corpus queries correctly escalate to web search? |
| **Citation accuracy** | Fraction of inline `[N]` markers that resolve to a real chunk. |
| **Keyword recall** *(deterministic)* | Fraction of required ground-truth keywords present in the answer. |
| **Latency P50 / P95** | End-to-end query latency distribution. |

### 9.3 Retrieval ablation (`ablation.py`)
A controlled experiment: the **same** pipeline and dataset are run three times, changing
**only** which ranked list(s) feed RRF (`bm25` / `vector` / `hybrid`). Reranker, thresholds,
and prompts are held constant, isolating retrieval strategy as the single variable.

**Measured results (n = 26 document-grounded questions):**

| Metric | BM25 | Vector | **Hybrid** |
|---|---|---|---|
| Keyword recall (primary) | 91.0% | 88.5% | **92.9%** |
| Citation accuracy | **95.5%** | 90.3% | 92.0% |
| Tool selection accuracy | 96.2% | 100% | 100% |
| Retrieval confidence (mean) | 0.383 | 0.365 | 0.360 |
| Latency P50 (s) | 12.73 | 11.85 | **10.84** |
| Latency P95 (s) | 26.06 | 20.22 | 22.36 |

Hybrid wins the primary metric (+1.9% over BM25, +4.4% over vector) **and** has the lowest
P50 latency. Honest nuance: BM25 has the best citation accuracy (lexical match pins exact
source chunks), so the choice of hybrid is a deliberate recall/latency-vs-pinpoint-citation
tradeoff, not a free lunch.

### Why deterministic metrics matter
**Keyword recall** is the headline metric precisely because **no LLM sits in the scoring
loop** — it is exact string matching against keywords verifiable directly in the source PDFs.
LLM-as-judge metrics are useful but have a structural weakness: the judge can quietly agree
with the generator's own phrasing and failure modes, and scores drift as the judge model
changes. A deterministic metric is reproducible, auditable, and immune to judge bias — the
right anchor when you need to *defend* a number, not just report one.

---

## 10. Agentic RAG vs Traditional RAG

| Dimension | Traditional RAG | Agentic RAG (this system) |
|---|---|---|
| Retrieval | Single fixed top-k pass | LLM-driven loop; reformulate, retry, switch tool |
| Query handling | Raw query embedded as-is | Agent rewrites weak queries; quality-gated |
| Tool choice | None (retrieval only) | `retrieve_documents` ⟷ `web_search`, chosen per query |
| Out-of-corpus | Hallucinates over irrelevant chunks | Escalates to web, or **refuses** (safe-fail gate) |
| Self-correction | None | Reflect → re-retrieve loop (bounded) |
| Failure behaviour | Always answers | Can decline when evidence is too weak |
| Cost / latency | Low, predictable | Higher; bounded by iteration caps + early exit |
| Observability | Opaque | Per-stage SSE trace + structured execution traces |

Agentic RAG is not strictly superior — it trades latency and complexity for adaptivity and
safety. For a narrow, well-formed FAQ over a clean corpus, traditional RAG is the correct,
cheaper choice. The agentic design pays off when queries are heterogeneous (some in-corpus,
some live, some adversarial) and **confident-but-wrong answers are expensive**.

---

## 11. System Design Tradeoffs

| Decision | Chosen | Why / cost |
|---|---|---|
| Orchestration | LangGraph explicit graph | Auditable control flow vs more boilerplate than a bare loop. |
| Loop bound | `max_iterations = 3` + early exit | Caps worst-case latency; risks under-retrieval on very complex multi-hop queries. |
| Fusion | RRF (rank-based) | No weight tuning, scale-invariant vs loses raw-score magnitude. |
| Reranker | CrossEncoder | Large precision gain vs added inference cost per query. |
| Embedder | all-MiniLM-L6-v2 (384-d) | Fast, CPU-friendly, cheap vs a larger model would lift the recall ceiling. |
| Safe-fail threshold | `safe_fail_threshold = 0.15` | Conservative to avoid wrongly refusing valid low-cosine meta/summary queries. |
| Grounding threshold | `grounding_threshold = 0.30` | Citations suppressed below this cosine — only assert grounding when evidence is meaningful. |
| Confidence threshold | `confidence_threshold = 0.50` | Calibrated for all-MiniLM-L6-v2 on academic text (0.50–0.65 range); naive 0.70 causes correct answers to be retried and worsened. |
| Config | All thresholds/iterations in `config.py` (env-overridable) | Operators tune without code changes. |
| Citations | Per-session manager | Multi-user safe. Session map is bounded by `LRUCache(max_session_cache)` — LRU eviction prevents unbounded growth. |
| Graph lifecycle | Compiled once at startup | `build_graph()` runs in the FastAPI lifespan; `run_agent()` accepts the compiled graph as an optional param so tests and the eval harness still call `build_graph()` directly without ceremony. |
| Retrieval confidence | `compute_retrieval_confidence()` in `state.py` | Single source of truth for the top-3 cosine mean used by both the orchestrator and the reflection-retry retriever — eliminates drift between the two paths. |
| Async LLM calls | `asyncio.to_thread` on all Gemini calls | Reflector and intent router previously blocked the event loop; now all network-bound LLM calls are off-loop, keeping concurrent SSE streams independent. |

---

## 12. Demo / How to Run

**Prerequisites:** Docker + Docker Compose, a Google Gemini API key, and (optional, for web
search) a Tavily API key.

```bash
# 1. Configure secrets
cp .env.example .env        # then edit:
#    GEMINI_API_KEY=...
#    TAVILY_API_KEY=...      (optional — web search degrades gracefully if absent)

# 2. Launch the full stack (Qdrant + FastAPI backend + React frontend)
docker compose up --build

# 3. Open the UI
#    Frontend : http://localhost:3000
#    API docs : http://localhost:8000/docs
#    Health   : http://localhost:8000/health
```

Upload PDFs through the UI (or `POST /api/documents/upload`); they are chunked, embedded
into Qdrant, and indexed for BM25. Ask questions in the chat — the UI streams the agent's
tool calls, retrieved-chunk metadata, the grounded answer with clickable citations, and the
reflection verdict in real time.

**API (no UI):**
```bash
curl -N http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What does OSM-PINN stand for?", "session_id": "demo-1"}'
# → Server-Sent Events: agent_action, chunks, token, citations, reflection, done
```

### Local development (without Docker)

```bash
# Backend — start Qdrant separately: docker run -p 6333:6333 qdrant/qdrant
cd backend
pip install -r requirements.txt
GEMINI_API_KEY=... QDRANT_HOST=localhost uvicorn main:app --reload   # :8000

# Frontend
cd frontend
npm install && npm run dev                                          # :5173
```

### API Reference

All application routes are served under the `/api` prefix; `/health` is the unprefixed
liveness probe.

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe (`{"status": "ok"}`) |
| `POST` | `/api/chat` | Ask a question — **streams** Server-Sent Events (`agent_action`, `chunks`, `token`, `citations`, `reflection`, `done`) |
| `GET` | `/api/chat/health` | Chat-subsystem readiness check |
| `POST` | `/api/documents/upload` | Upload a PDF — chunk, embed into Qdrant, index for BM25 |
| `POST` | `/api/documents/text` | Ingest raw text (same pipeline, no file) |
| `GET` | `/api/documents/list` | List indexed documents |
| `GET` | `/api/documents/stats` | Corpus statistics (chunk/document counts) |
| `DELETE` | `/api/documents` | Clear the corpus |

Interactive OpenAPI docs are available at `http://localhost:8000/docs` when the backend is
running.

---

## 13. Testing Strategy — how quality is ensured

Quality assurance is organised into **four layers** — unit, trajectory, adversarial safety,
and the evaluation harness — and is deliberately **not** dependent on a single LLM judge.
The first three layers are fully offline and deterministic: **245 tests collect and run in
~1 s with zero API cost**.

| Layer | Count | Scope |
|---|---|---|
| Unit | 233 | Component & config correctness in isolation |
| Trajectory | 7 | Real node + routing functions over the graph topology |
| Safety | 5 | Adversarial contracts under hostile input |
| Evaluation harness | 40-q benchmark | End-to-end, real LLM (see §9) |

### 13.1 Unit tests — 233 tests, 9 files

```
backend/tests/
├── test_retrieval.py    — RRF fusion logic (rank ordering, score accumulation, degenerate inputs)
├── test_router.py       — intent router: conversational regex, web-query regex, DIRECT_INTENTS set
├── test_safe_fail.py    — graph routing functions: safe-fail gate, reflection loop, web-guard
├── test_citations.py    — citation pipeline: CitationManager, _extract_meta (3-tier fallback + truncation),
│                          _remap_citation_groups, _is_negative_answer (grounding gate),
│                          _clean_pdf_text (PDF symbol fonts), _evidence_snippet, _keyword_recall
├── test_vector_store.py — _point_id() determinism, restart-stability, Qdrant uint63 range
├── test_hybrid_async.py — async retrieval correctness, off-loop dispatch, score preservation
├── test_confidence.py   — parse_generation, build_citations, calibrate_confidence; config weight respect
├── test_bounded_cache.py — LRU eviction, recency refresh, thread safety, cap under load
└── test_config.py       — config contracts: weight sums (=1.0), threshold ranges [0,1], ordering invariants
```

Key coverage areas per file:

**`test_retrieval.py`** — verifies RRF is rank-based (not score-based), accumulates scores
correctly across two lists, handles single-list and empty-list inputs.

**`test_router.py`** — protects the deterministic fast-paths: 32 conversational patterns that
must bypass the LLM, 10 real-time patterns that must route to web_search, 5 intents that
must (not) be in `DIRECT_INTENTS`.

**`test_safe_fail.py`** — exercises all nine routing transitions across four routing functions.
Includes a regression test for the exchange-rate bug: a web-grounded answer with
`reflection_passed=False` must not re-trigger document retrieval.

**`test_citations.py`** — 68 assertions across seven classes: idempotency and thread-safety of
`CitationManager`, three-tier metadata parser including the truncated-`<<<JSON` regression,
grouped citation remapping (the `[2, 4]` parser bug), grounding gate detection, PDF glyph
decoding, evidence snippet keyword scoring, and the `_keyword_recall` eval metric.

**`test_config.py`** — 21 contract tests that validate `config.py` *itself* (not behaviour):
weights intended to combine must sum to 1.0 (doc confidence weights, web formula ceiling
≤ 1.0); every probability/cosine threshold must fall in `[0, 1]`; and load-bearing
**ordering invariants** hold (`safe_fail < grounding < confidence`, `rerank_top_k ≥
final_top_k`, `chunk_cache_size > bm25_top_k + vector_top_k`). These catch a class of bug
no behaviour test surfaces: a threshold value that is individually legal but breaks a
cross-threshold relationship — e.g. raising `safe_fail_threshold` above `grounding_threshold`
silently makes the grounded-answer path unreachable.

### 13.2 Trajectory (integration) tests — 7 scenarios

`TrajectoryRunner` (`backend/tests/test_trajectories.py`) executes the **real** node
`run()` functions and **real** routing functions, walking the actual graph topology
expressed as `CONDITIONAL` + `FIXED` edge tables (a 1:1 transcription of `build_graph()`).
Only three external boundaries are mocked: the Gemini `_client` per module, the retriever
service, and Tavily. Fully offline, deterministic, no API keys.

Each test asserts **both** the execution path (`runner.path`) and the final state fields —
the two things unit tests cannot see.

| Test | Scenario | Path asserted | State asserted |
|---|---|---|---|
| **T1** | Web answer preserved after failed reflection | `router→web_search→generator→reflector` | `"Kuala Lumpur" in answer`, `citations == []` |
| **T2** | Document answer stays doc-grounded, no web escalation | `router→orchestrator→generator→reflector` | `web_search_results == []`, citations present |
| **T3** | Reflection retry uses improved retrieval, exactly once | retriever count = 1, generator count = 2 | stronger chunks used, confidence ≥ 0.5 |
| **T4a** | Both sources empty → graceful general_knowledge downgrade | ends at `direct`, generator never ran | `citations == []`, no hallucinated refs |
| **T4b** | Weak retrieval + no web → safe_fail refusal | ends at `safe_fail` | `confidence_score == 0.0`, "couldn't find" |
| **T5** | Valid web answer NOT downgraded after reflection failure | `router→web_search→generator→reflector` | `intent == "web_search"` (not general_knowledge) |
| **T6** | Negative ("not found") answer → citations suppressed, confidence low | orchestrator path | `citations == []`, `confidence ≤ 0.25`, no `[1]` in answer |

**Test teeth:** reintroducing the pre-fix `_should_continue` (without the web-guard) causes
T5 to raise `AssertionError` immediately — regressions are caught, not just described.

### 13.3 Adversarial safety tests — 5 contracts

`backend/tests/test_safety.py` (marked `pytest.mark.safety`) adds a thin layer of
adversarial contracts — guarantees the system must hold under hostile input. It implements
**only the net-new contracts** not already enforced elsewhere, cross-referencing existing
coverage (e.g. reflection-loop termination in `test_safe_fail.py`, example-based citation
integrity in `test_citations.py`) so the layer stays the single index of safety guarantees
without duplicating tests. Fully offline and deterministic; shares builders via
`tests/_harness.py`.

| Contract | Guarantee under hostile input |
|---|---|
| **Injection → safe label** | A coerced classifier output that isn't a real label is clamped to `document_qa` — injection can't mint a new intent or seize a direct-answer path. |
| **Hallucinated tool name** | If the LLM is coerced into naming a non-existent tool, the orchestrator returns an "unknown tool" observation and terminates cleanly instead of crashing. |
| **Duplicate-call loop guard** | A repeated `(tool, query)` signature breaks the loop — only the forced first retrieve runs. |
| **Iteration cap** | A non-stopping LLM (distinct query every turn) is hard-capped at `orchestrator_max_iterations`. |
| **Citation integrity (fuzzed)** | 200 randomised valid/invalid inline-marker mixes (fixed seed `1234`) — no emitted citation may ever reference a chunk that wasn't retrieved. |

### 13.4 Evaluation harness (`backend/evaluate/`)

Drives the real LangGraph agent over 40 benchmark questions:

- **Functional suite** — pass/fail behavioural assertions (loop bounds, escalation,
  greeting routing).
- **Agentic metrics** — tool selection, reformulation recovery, web escalation, citation
  resolution, keyword recall, latency P50/P95.
- **Retrieval ablation** — controlled BM25 vs vector vs hybrid comparison (see §9.3).

### Why this test architecture

| Layer | What it catches | What it misses |
|---|---|---|
| Unit | Function-level logic, edge cases, regressions, config invariants | Multi-node interaction bugs |
| Trajectory | Inter-node interaction bugs, routing chains, state propagation | Real LLM behaviour |
| Safety | Adversarial robustness: injection, loop safety, citation integrity | Real-world attack variety |
| Evaluation harness | End-to-end correctness with real LLM calls | Slow, non-deterministic, expensive |

Each layer catches what the others structurally cannot. The three offline layers (245 tests:
unit + trajectory + safety) run in ~1 s with zero API cost and catch the overwhelming
majority of regressions; the eval harness produces the numbers you defend in a demo or
report. Crucially, the headline metrics are deterministic — config invariants and keyword
recall are asserted by exact computation, not by an LLM judge — so results are reproducible
on every commit.

---

## 14. Limitations & Future Improvements

- **Embedding ceiling.** all-MiniLM-L6-v2 (384-d) is lightweight; upgrading to a stronger
  embedder (e.g. `bge`/`mxbai`-class, 768–1024-d) and re-indexing is the highest-leverage
  retrieval improvement available.
- **No query decomposition.** Compound multi-hop questions are handled by in-loop
  reformulation, not explicit sub-query planning with per-sub-query evidence tracking.
- **Citation manager TTL.** The per-session `LRUCache` evicts the least-recently-used session
  past the cap (`max_session_cache=1000`); a returning evicted session simply restarts its
  citation counter from 1. Production would add explicit TTL expiry to evict stale sessions
  proactively rather than on overflow.
- **No persistent trace store.** Execution traces stream to the UI and stdout but are not
  persisted behind a `GET /trace/{id}` API for post-hoc debugging.
- **Latency.** P95 ≈ 22–26 s under multi-tool, multi-iteration paths — acceptable for an
  asynchronous, streamed UX, but not for a low-latency synchronous API without further
  caching and model-tier tuning.

---

## 15. Repository Layout

```
rag-agentic-2/
├── backend/
│   ├── main.py                 # FastAPI app: lifespan, service init, graph compiled once
│   ├── config.py               # All settings/thresholds (Pydantic, env-overridable)
│   ├── api/
│   │   ├── chat.py             # POST /api/chat — SSE streaming endpoint
│   │   └── documents.py        # upload / text / list / stats / delete
│   ├── agent/
│   │   ├── graph.py            # LangGraph StateGraph: nodes, conditional edges, run_agent()
│   │   ├── orchestrator.py     # ReAct tool loop (think · act), forced-first-retrieve, loop guard
│   │   ├── state.py            # Typed AgentState + compute_retrieval_confidence()
│   │   ├── citation_manager.py # Per-session stable citation IDs
│   │   ├── bounded_cache.py    # Thread-safe LRU cache (sessions, chunk cache)
│   │   ├── nodes/
│   │   │   ├── intent_router.py    # Classify → document_qa / web_search / direct
│   │   │   ├── retriever.py        # Reflection-retry retrieval node
│   │   │   ├── generator.py        # Grounded answer synthesis + SSE sentinel guard
│   │   │   ├── generator_prompts.py
│   │   │   ├── reflector.py        # Answer-quality verdict (constrained JSON)
│   │   │   ├── web_search.py       # Tavily integration
│   │   │   ├── citation_logic.py   # [N] marker parse / remap / grounding gate
│   │   │   └── evidence.py         # Evidence-snippet selection for citation cards
│   │   └── tools/              # Tool definitions exposed to the orchestrator
│   ├── retrieval/
│   │   ├── hybrid_retriever.py # BM25 + vector → RRF → CrossEncoder rerank
│   │   ├── bm25_index.py       # Lexical index (rank-bm25), RLock-guarded
│   │   ├── vector_store.py     # Qdrant wrapper; SHA-256 stable point IDs
│   │   ├── reranker.py         # CrossEncoder (ms-marco-MiniLM-L-6-v2)
│   │   └── document_processor.py  # PDF parse + chunking
│   ├── evaluate/
│   │   ├── run_all.py          # Functional + agentic suites
│   │   ├── functional.py       # Pass/fail behavioural assertions
│   │   ├── agentic.py          # Trace-derived decision metrics
│   │   ├── ablation.py         # BM25 vs vector vs hybrid controlled experiment
│   │   ├── agent_runner.py     # Drives the real compiled graph
│   │   ├── trace_schema.py     # EvalTrace schema
│   │   ├── datasets/           # Benchmark questions + ground-truth keywords
│   │   └── results/            # Reproducible CSV/JSON output
│   ├── tests/                  # 245 tests (unit · trajectory · safety) + _harness.py
│   ├── pytest.ini              # Test markers (e.g. `safety`)
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── src/
│   │   ├── App.tsx
│   │   ├── components/         # Chat, Citations, Layout, UI
│   │   ├── hooks/              # SSE stream consumption
│   │   ├── store/             # Zustand state
│   │   └── types/
│   ├── package.json
│   ├── vite.config.ts
│   ├── nginx.conf             # Production static serving
│   └── Dockerfile
├── diagrams/                   # Architecture / pipeline figures
├── docker-compose.yml          # Qdrant + backend + frontend
├── .env.example
└── README.md
```

---

## 16. Tech Stack

- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS, Zustand; SSE streaming UI with a
  live agent-trace panel.
- **Backend:** FastAPI (Python 3.11), LangGraph `StateGraph` orchestration.
- **LLM:** Google Gemini 2.0 Flash (`gemini-2.0-flash`) via `google-genai` (native function
  calling, constrained decoding for classification nodes).
- **Vector store:** Qdrant. **Lexical:** `rank-bm25`. **Reranker:**
  `cross-encoder/ms-marco-MiniLM-L-6-v2`. **Embeddings:** `all-MiniLM-L6-v2`.
- **Web search:** Tavily. **Deployment:** Docker Compose (Qdrant + backend + frontend).

---

## 17. Conclusion

This project implements Agentic RAG as a **bounded, observable decision system** rather than
a static retrieval chain: an LLM agent that selects tools, reformulates weak queries,
escalates to the web, self-corrects through a reflection loop, grounds every claim with
session-stable citations, and **refuses when evidence is insufficient**. Design choices —
hybrid retrieval with RRF and cross-encoder reranking, iteration bounds with early exit,
constrained-decoding calibration, per-session citation scoping, grounding gate with negative
answer detection, confidence threshold calibrated to the embedder's actual score distribution
— are made explicitly and, where it matters, **validated empirically** with both a
deterministic reproducible evaluation harness and a trajectory test suite that locks in
previously discovered inter-node bugs. The result is a system whose behaviour can be
defended with measurements, not adjectives.
