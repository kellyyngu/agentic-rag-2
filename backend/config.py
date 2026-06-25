from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    # Google Gemini
    gemini_api_key: str = Field(..., env="GEMINI_API_KEY")
    llm_model: str = Field("gemini-2.0-flash", env="LLM_MODEL")

    # Qdrant
    qdrant_host: str = Field("qdrant", env="QDRANT_HOST")
    qdrant_port: int = Field(6333, env="QDRANT_PORT")
    qdrant_collection: str = Field("documents", env="QDRANT_COLLECTION")

    # Embeddings
    embedding_model: str = Field("all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    embedding_dim: int = Field(384, env="EMBEDDING_DIM")

    # Reranker
    reranker_model: str = Field(
        "cross-encoder/ms-marco-MiniLM-L-6-v2", env="RERANKER_MODEL"
    )

    # Retrieval
    bm25_top_k: int = Field(20, env="BM25_TOP_K")
    vector_top_k: int = Field(20, env="VECTOR_TOP_K")
    rerank_top_k: int = Field(8, env="RERANK_TOP_K")
    final_top_k: int = Field(5, env="FINAL_TOP_K")
    chunk_size: int = Field(512, env="CHUNK_SIZE")
    chunk_overlap: int = Field(100, env="CHUNK_OVERLAP")

    # Agent
    max_reflection_iterations: int = Field(2, env="MAX_REFLECTION_ITERATIONS")
    # Reflector fast-pass threshold. Calibrated for all-MiniLM-L6-v2 on academic
    # text — good answers consistently score 0.50–0.65 with this embedder.
    # Setting too high (e.g. 0.7) causes correct answers to be retried needlessly,
    # and the retry often produces a shorter, worse answer.
    confidence_threshold: float = Field(0.50, env="CONFIDENCE_THRESHOLD")
    retrieval_relevance_threshold: float = Field(0.2, env="RETRIEVAL_RELEVANCE_THRESHOLD")
    web_search_fallback_threshold: int = Field(2, env="WEB_SEARCH_FALLBACK_THRESHOLD")

    # Orchestrator (ReAct tool loop)
    orchestrator_max_iterations: int = Field(3, env="ORCHESTRATOR_MAX_ITERATIONS")
    orchestrator_quality_threshold: float = Field(0.30, env="ORCHESTRATOR_QUALITY_THRESHOLD")

    # Safe-fail gate: refuse rather than answer when retrieval is this weak AND no web evidence.
    # Kept conservative so valid low-cosine meta/summary queries are not wrongly refused.
    safe_fail_threshold: float = Field(0.15, env="SAFE_FAIL_THRESHOLD")

    # Grounding gate (generator): a document answer is only "grounded" if its top cited
    # chunk reaches this cosine relevance. Below it (or a negative "not found" answer),
    # citations are suppressed and confidence is capped low — no false certainty.
    grounding_threshold: float = Field(0.30, env="GROUNDING_THRESHOLD")

    # ── Confidence calibration weights (generator) ──────────────────────────────
    # The generator scores answer confidence with an explicit, tunable policy rather
    # than hardcoded literals. All values calibrated for all-MiniLM-L6-v2 on the
    # academic corpus; expose them so operators can retune without code changes.
    #
    # Ungrounded / negative / off-topic answers: confidence is capped at this value
    # (and then floored by retrieval relevance) so a "not found" reply can never look
    # certain.
    confidence_ungrounded_cap: float = Field(0.25, env="CONFIDENCE_UNGROUNDED_CAP")
    # Web-grounded answers (no document citations): confidence = base + llm_self_rating*llm_weight.
    # The base is a floor that credits having retrieved live evidence at all.
    confidence_web_base: float = Field(0.45, env="CONFIDENCE_WEB_BASE")
    confidence_web_llm_weight: float = Field(0.35, env="CONFIDENCE_WEB_LLM_WEIGHT")
    # Document-grounded answers: a weighted blend of the LLM self-rating, the retrieval
    # cosine relevance, and citation coverage. The three weights are intended to sum to 1.
    confidence_doc_llm_weight: float = Field(0.40, env="CONFIDENCE_DOC_LLM_WEIGHT")
    confidence_doc_retrieval_weight: float = Field(0.40, env="CONFIDENCE_DOC_RETRIEVAL_WEIGHT")
    confidence_doc_citation_weight: float = Field(0.20, env="CONFIDENCE_DOC_CITATION_WEIGHT")

    # Bounded registries — cap long-lived in-memory maps so a long-running process
    # cannot grow without limit. See agent/bounded_cache.py.
    # One CitationManager per active session; LRU-evict the least-recently-used.
    max_session_cache: int = Field(1000, env="MAX_SESSION_CACHE")
    # Per-retriever chunk-content cache. Must comfortably exceed one query's working
    # set (bm25_top_k + vector_top_k candidates) so no in-flight lookup is evicted.
    chunk_cache_size: int = Field(10000, env="CHUNK_CACHE_SIZE")

    # Minimum vector cosine score for a chunk to survive the retriever filter.
    # Lower = keeps more chunks (better for non-academic docs like menus, reports).
    # Higher = stricter cross-document contamination filtering.
    min_vector_score: float = Field(0.10, env="MIN_VECTOR_SCORE")

    # Data
    data_dir: str = Field("/app/data", env="DATA_DIR")

    # CORS
    cors_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:5173"],
        env="CORS_ORIGINS",
    )

    # Debug
    debug: bool = Field(False, env="DEBUG")
    log_level: str = Field("INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
