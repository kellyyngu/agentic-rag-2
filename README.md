# Agentic RAG — Document Question-Answering with an LLM-Driven Tool Loop

A production-oriented Retrieval-Augmented Generation system where an LLM **agent decides
how to gather evidence** — which tool to call, how to reformulate a weak query, when to
escalate to web search, and when it has enough context to answer. It is not a fixed
`retrieve → generate` pipeline; retrieval is a *decision*, not a hardcoded step.

**Stack:** FastAPI · LangGraph · Google Gemini (function calling) · Qdrant · BM25 ·
CrossEncoder reranker · Tavily · React + Vite (SSE streaming) · Docker Compose.

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
            ┌─────────────────┼──────────────┐   │               │
            │ retrieve_documents              │   │               │
            │ web_search                      │   │               │
            ▼                                 ▼   │               │
   ┌──────────────────┐              ┌──────────────────┐         │
   │ Hybrid Retriever │              │  Safe-Fail Gate  │         │
   │ BM25 + Vector    │              │ weak + no web →  │         │
   │ → RRF → Rerank   │              │ refuse, skip gen │         │
   └────────┬─────────┘              └────────┬─────────┘         │
            │                                 │                   │
            ▼                                 │                   │
   ┌──────────────────┐                       │                   │
   │   Generator      │◀──────────────────────┼───────────────────┘
   │ grounded answer  │                       │
   │ + inline [N]     │                       │
   └────────┬─────────┘                       │
            ▼                                  │
   ┌──────────────────┐   fail & iter<max     │
   │   Reflector      │──────────┐            │
   │ quality verdict  │          │ re-retrieve│
   └────────┬─────────┘          ▼            │
            │ pass        ┌──────────────┐    │
            ▼             │  Retriever   │────┘
          DONE            │ (retry pass) │
                          └──────────────┘
```

The graph is implemented as a **LangGraph `StateGraph`** with conditional edges. State
flows through a typed `AgentState` (`backend/agent/state.py`); every node emits
**Server-Sent Events** so the React frontend renders the agent's decisions (tool calls,
retrieval quality, reflection verdict) live.

**Why LangGraph over a hand-rolled loop:** the routing logic (intent → orchestrator →
safe-fail / generate → reflect → retry) is expressed as explicit, inspectable edges rather
than nested conditionals. The control flow *is* the graph, which makes the agent's behaviour
auditable — a property that matters more than raw line count.

---

## 4. How the Agent Works — think · act · reflect

The orchestrator (`backend/agent/orchestrator.py`) is a bounded ReAct loop built on
**Gemini native function calling**:

```
for iteration in range(orchestrator_max_iterations):     # default 3
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

**Reflect (separate node):** after generation, the reflector
(`backend/agent/nodes/reflector.py`) scores the answer for completeness and grounding. If it
fails *and* the iteration budget allows, the graph **loops back to re-retrieve** using the
reflector's feedback as a refined query, then regenerates. This is a real closed loop
(`generator → reflector → retriever → generator`), bounded by `max_reflection_iterations`
(default 2), not a dead-end verdict.

**Robustness detail:** all classification/short-output LLM calls (router, reflector) set
`thinking_budget=0` and use constrained decoding (`response_schema` +
`response_mime_type=application/json`). On Gemini Flash, "thinking" tokens otherwise consume
the `max_output_tokens` budget and truncate the JSON — disabling them for simple
classification eliminated a class of intermittent parse failures.

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
retrieved chunk (`backend/agent/nodes/generator.py`):

1. The generator is shown context numbered `[1]…[k]` and instructed to cite inline.
2. Local `[N]` markers are remapped to **session-stable global IDs** via a per-session
   `CitationManager`, so the same chunk keeps the same citation number across turns of a
   conversation.
3. The displayed relevance score per citation is the chunk's **vector cosine similarity**,
   not the reranker logit (more meaningful to a human reader).

**Session safety:** `CitationManager` is scoped per `session_id` (or per-request when no
session id is supplied) — there is **no process-global shared citation state**, so
concurrent users never collide on or leak each other's citation IDs.

---

## 7. Web Search Augmentation

When the intent router classifies a query as `web_search`, or when the orchestrator decides
document retrieval is consistently WEAK, the agent calls **Tavily**
(`backend/agent/nodes/web_search.py`, `orchestrator._exec_web`). Tavily is used over
DuckDuckGo because it returns reliably from datacenter IPs. Web results are fed back into the
orchestrator loop as observations and into the generator as supplementary context, clearly
separated from document context in the prompt. If no `TAVILY_API_KEY` is configured, web
search degrades gracefully and the agent answers from document context only.

---

## 8. Evaluation Framework

The harness (`backend/evaluate/`) drives the **real** LangGraph agent over a 40-question
benchmark (`single_hop`, `multi_hop`, `out_of_corpus`, `conversational`, `adversarial`) and
measures three independent axes. All runs emit reproducible CSV/JSON to
`backend/evaluate/results/`.

```bash
docker compose exec backend python -m evaluate.run_all                 # functional + agentic
docker compose exec backend python -m evaluate.run_all --suite agentic
docker compose exec backend python -m evaluate.ablation                # retrieval ablation
```

### 8.1 Functional suite
Behavioural assertions (e.g. *loop is bounded ≤ N tool calls*, *out-of-corpus escalates to
web*, *greeting calls no tool*). Pass/fail per case — a regression tripwire.

### 8.2 Agentic decision metrics (`agentic.py`)
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

### 8.3 Retrieval ablation (`ablation.py`)
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

## 9. Agentic RAG vs Traditional RAG

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

## 10. System Design Tradeoffs

| Decision | Chosen | Why / cost |
|---|---|---|
| Orchestration | LangGraph explicit graph | Auditable control flow vs more boilerplate than a bare loop. |
| Loop bound | `max_iterations = 3` + early exit | Caps worst-case latency; risks under-retrieval on very complex multi-hop queries. |
| Fusion | RRF (rank-based) | No weight tuning, scale-invariant vs loses raw-score magnitude. |
| Reranker | CrossEncoder | Large precision gain vs added inference cost per query. |
| Embedder | all-MiniLM-L6-v2 (384-d) | Fast, CPU-friendly, cheap vs a larger model would lift the recall ceiling. |
| Safe-fail threshold | Conservative (0.15) | Avoids wrongly refusing valid low-cosine summary queries vs lets some weak answers through to the generator's own "insufficient context" handling. |
| Config | Thresholds/iterations in `config.py` (env-overridable) | Operators tune without code changes. |
| Citations | Per-session manager | Multi-user safe vs unbounded session map growth over a long-lived process (acceptable for the demo; would add TTL eviction in production). |

---

## 11. Demo / How to Run

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

Upload PDFs through the UI (or `POST /api/documents`); they are chunked, embedded into
Qdrant, and indexed for BM25. Ask questions in the chat — the UI streams the agent's tool
calls, retrieved-chunk metadata, the grounded answer with clickable citations, and the
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

---

## 12. Testing Strategy — how quality is ensured

Quality assurance is layered, and deliberately **not** dependent on a single LLM judge:

1. **Functional behavioural tests** — assert structural invariants of the agent: the tool
   loop is bounded, out-of-corpus queries escalate, greetings call no tool, the safe-fail
   gate triggers on weak retrieval. These catch *control-flow* regressions.

2. **Deterministic answer metrics** — keyword recall against source-verifiable ground-truth
   keywords. No model in the scoring loop → reproducible, bias-free, defensible.

3. **Trace-derived decision metrics** — tool selection, reformulation recovery, web
   escalation, citation resolution. These verify the agent *reasoned* correctly, not just
   that the final string looked plausible.

4. **Controlled ablation** — isolates one variable (retrieval strategy) to produce evidence
   that the hybrid design choice is justified, rather than asserted.

**Test-case design principles:** the benchmark is stratified by category so each capability
is exercised independently — `single_hop` (precision), `multi_hop` (synthesis),
`out_of_corpus` (escalation), `conversational` (no-tool routing), and `adversarial`
(false-premise and gibberish inputs that must not produce fabricated content). Each item
carries an `expected_tool` and `required_keywords` so both the *path* and the *content* are
checkable.

---

## 13. Limitations & Future Improvements

Stated honestly — these are known, not hidden:

- **Embedding ceiling.** all-MiniLM-L6-v2 (384-d) is lightweight; upgrading to a stronger
  embedder (e.g. `bge`/`mxbai`-class, 768–1024-d) and re-indexing is the highest-leverage
  retrieval improvement available.
- **No query decomposition.** Compound multi-hop questions are handled by in-loop
  reformulation, not explicit sub-query planning with per-sub-query evidence tracking.
- **Citation manager growth.** The per-session map is unbounded over a long-lived process;
  production would add TTL/LRU eviction.
- **No persistent trace store.** Execution traces stream to the UI and stdout but are not
  persisted behind a `GET /trace/{id}` API for post-hoc debugging.
- **Latency.** P95 ≈ 22–26 s under multi-tool, multi-iteration paths — acceptable for an
  asynchronous, streamed UX, but not for a low-latency synchronous API without further
  caching and model-tier tuning.

---

## 14. Tech Stack

- **Frontend:** React 18, Vite, TypeScript, Tailwind CSS, Zustand; SSE streaming UI with a
  live agent-trace panel.
- **Backend:** FastAPI (Python 3.11), LangGraph `StateGraph` orchestration.
- **LLM:** Google Gemini Flash via `google-genai` (native function calling, constrained
  decoding).
- **Vector store:** Qdrant. **Lexical:** `rank-bm25`. **Reranker:**
  `cross-encoder/ms-marco-MiniLM-L-6-v2`. **Embeddings:** `all-MiniLM-L6-v2`.
- **Web search:** Tavily. **Deployment:** Docker Compose (Qdrant + backend + frontend).

---

## 15. Conclusion

This project implements Agentic RAG as a **bounded, observable decision system** rather than
a static retrieval chain: an LLM agent that selects tools, reformulates weak queries,
escalates to the web, self-corrects through a reflection loop, grounds every claim with
session-stable citations, and **refuses when evidence is insufficient**. Design choices —
hybrid retrieval with RRF and cross-encoder reranking, iteration bounds with early exit,
constrained-decoding calibration, per-session citation scoping — are made explicitly and,
where it matters, **validated empirically** with a deterministic, reproducible evaluation
harness. The result is a system whose behaviour can be defended with measurements, not
adjectives.
