from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from loguru import logger
import sys

from config import settings
from api import chat, documents, evaluate
from retrieval.vector_store import VectorStore
from retrieval.bm25_index import BM25Index
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.reranker import Reranker
from agent.citation_manager import CitationManager


logger.remove()
logger.add(sys.stderr, level=settings.log_level, format="{time:HH:mm:ss} | {level} | {message}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing services...")
    app.state.vector_store = VectorStore()
    await app.state.vector_store.initialize()

    app.state.bm25_index = BM25Index(data_dir=settings.data_dir)
    app.state.bm25_index.load()

    app.state.reranker = Reranker()

    app.state.retriever = HybridRetriever(
        vector_store=app.state.vector_store,
        bm25_index=app.state.bm25_index,
        reranker=app.state.reranker,
    )
    app.state.citation_manager = CitationManager()
    logger.info("All services ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Agentic RAG v2",
    description="Production-grade Agentic RAG with LangGraph, hybrid retrieval, and streaming",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat.router, prefix="/api", tags=["chat"])
app.include_router(documents.router, prefix="/api", tags=["documents"])
app.include_router(evaluate.router, prefix="/api", tags=["evaluate"])


@app.get("/health")
async def health():
    return {"status": "ok", "model": settings.llm_model}
