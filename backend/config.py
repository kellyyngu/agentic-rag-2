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
    confidence_threshold: float = Field(0.7, env="CONFIDENCE_THRESHOLD")
    web_search_fallback_threshold: int = Field(2, env="WEB_SEARCH_FALLBACK_THRESHOLD")

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
