# Agentic RAG v2

Production-grade Retrieval-Augmented Generation with LangGraph agent orchestration, hybrid retrieval, and a modern React UI.

## Architecture

```
Frontend (React + Vite + Tailwind)
    ↓ SSE streaming
Backend (FastAPI)
    ↓
LangGraph Agent
  ├── Planner   → query decomposition + strategy
  ├── Retriever → hybrid BM25 + Qdrant + RRF + CrossEncoder rerank
  ├── Web Search → DuckDuckGo fallback
  ├── Generator → grounded answer with streaming (Claude Opus 4.8)
  └── Reflector → confidence check + optional re-retrieval

Storage: Qdrant (vector) + BM25 (in-memory, persisted to disk)
```

## Quick Start

```bash
# 1. Copy env
cp .env.example .env
# Edit .env and set GEMINI_API_KEY=AIzaSy...

# 2. Start everything
docker-compose up --build

# 3. Open browser
open http://localhost:3000
```

## Local Development (no Docker)

```bash
# Backend
cd backend
pip install -r requirements.txt
# Start Qdrant separately: docker run -p 6333:6333 qdrant/qdrant
GEMINI_API_KEY=AIzaSy... QDRANT_HOST=localhost uvicorn main:app --reload

# Frontend
cd frontend
npm install
npm run dev
# Open http://localhost:5173
```

## Key Features

| Feature | Implementation |
|---|---|
| Query planning | LLM decomposes query into sub-questions |
| Hybrid retrieval | BM25 (rank_bm25) + Qdrant vectors |
| Rank fusion | Reciprocal Rank Fusion (RRF, k=60) |
| Reranking | CrossEncoder `ms-marco-MiniLM-L-6-v2` |
| LLM | Google Gemini (gemini-2.0-flash) |
| Streaming | SSE (FastAPI → React) + Gemini streaming |
| Reflection | Confidence-based re-retrieval loop |
| Citations | Clickable inline badges + side panel |
| Follow-ups | Model-generated suggested questions |
| Dark mode | CSS variables + system detection |

## Evaluation

```bash
# Ingest some documents first, then:
python evaluation/run_eval.py --api http://localhost:8000
```

Metrics: **Precision@5**, **Recall@5**, **Groundedness**, **Citation Accuracy**, **Latency**

## Stack

- **Frontend**: React 18, Vite, TypeScript, Tailwind CSS, Zustand
- **Backend**: FastAPI, Python 3.11
- **Agent**: LangGraph
- **LLM**: Google Gemini (gemini-2.0-flash)
- **Vector DB**: Qdrant
- **Embeddings**: `all-MiniLM-L6-v2` (sentence-transformers)
- **BM25**: `rank_bm25`
- **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- **Web search**: DuckDuckGo (no API key needed)
- **Deployment**: Docker Compose
