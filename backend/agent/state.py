from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RetrievedChunk:
    chunk_id: str
    content: str
    source: str
    page: Optional[int]
    score: float          # sigmoid-normalised cross-encoder score (used for ranking)
    vector_score: float = 0.0  # cosine similarity from Qdrant (used for display + filtering)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "source": self.source,
            "page": self.page,
            "score": self.score,
            "metadata": self.metadata,
        }


@dataclass
class Citation:
    id: str
    source: str
    page: Optional[int]
    excerpt: str
    relevance_score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "page": self.page,
            "excerpt": self.excerpt,
            "relevance_score": self.relevance_score,
        }


class AgentState(TypedDict):
    # Input
    query: str
    conversation_history: List[Dict[str, str]]

    # Routing
    intent: str

    # Planning
    sub_questions: List[str]
    retrieval_strategy: str
    needs_web_search: bool

    # Retrieval
    retrieved_chunks: List[RetrievedChunk]
    search_queries_used: List[str]
    web_search_results: List[Dict[str, str]]

    # Generation
    answer: str
    citations: List[Citation]
    follow_up_questions: List[str]

    # Retrieval quality gate
    retrieval_confidence: float

    # Reflection
    reflection_passed: bool
    reflection_feedback: Optional[str]
    confidence_score: float
    iteration_count: int

    # Observability
    trace: Dict[str, Any]
    stream_queue: Optional[Any]       # asyncio.Queue for SSE events
    citation_manager: Optional[Any]   # per-session CitationManager (scoped by session_id)
