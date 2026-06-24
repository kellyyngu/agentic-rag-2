"""
Standardized trace schema.

Every metric in this package is computed from a single normalized structure:

    EvalTrace(
        query, intent, tool_calls, retrieved_chunks, citations,
        final_answer, retrieval_confidence, confidence_score,
        reflection_passed, latency_s, error
    )

`EvalTrace.from_state()` adapts the raw AgentState dict returned by the
LangGraph orchestrator into this schema. If the underlying state format ever
changes, this is the ONLY file that needs updating.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class ToolCall:
    tool: str
    args: dict
    observation: str = ""

    @property
    def query(self) -> str:
        return str(self.args.get("query", "")).strip()

    @property
    def quality(self) -> Optional[str]:
        """Parse GOOD / WEAK out of the observation summary, if present."""
        if "quality=GOOD" in self.observation or "GOOD" in self.observation:
            return "GOOD"
        if "quality=WEAK" in self.observation or "WEAK" in self.observation:
            return "WEAK"
        return None


@dataclass
class EvalTrace:
    query: str
    intent: str = "document_qa"
    tool_calls: list[ToolCall] = field(default_factory=list)
    retrieved_chunks: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    final_answer: str = ""
    retrieval_confidence: float = 0.0
    confidence_score: float = 0.0
    reflection_passed: bool = True
    latency_s: float = 0.0
    error: Optional[str] = None

    # ── Derived helpers used by the metric modules ──────────────────────────

    @property
    def num_tool_calls(self) -> int:
        return len(self.tool_calls)

    @property
    def first_tool(self) -> Optional[str]:
        return self.tool_calls[0].tool if self.tool_calls else None

    @property
    def retrieve_calls(self) -> list[ToolCall]:
        return [t for t in self.tool_calls if t.tool == "retrieve_documents"]

    @property
    def web_calls(self) -> list[ToolCall]:
        return [t for t in self.tool_calls if t.tool == "web_search"]

    @property
    def retrieved_sources(self) -> list[str]:
        return [c.get("source", "") for c in self.retrieved_chunks]

    @property
    def cited_sources(self) -> list[str]:
        return [c.get("source", "") for c in self.citations]

    @property
    def context_texts(self) -> list[str]:
        """Full chunk contents — used as RAGAS `retrieved_contexts`."""
        return [c.get("content", "") for c in self.retrieved_chunks if c.get("content")]

    def inline_citation_ids(self) -> list[str]:
        """Distinct [N] markers actually present in the answer text."""
        return sorted(set(re.findall(r"\[(\d+)\]", self.final_answer)), key=lambda x: int(x))

    def did_reformulate(self) -> bool:
        """≥2 retrieve calls with distinct query strings."""
        qs = [t.query.lower() for t in self.retrieve_calls if t.query]
        return len(set(qs)) >= 2

    def reformulation_recovered(self) -> Optional[bool]:
        """
        Among retrieve calls: was there a WEAK result followed by a GOOD one?
        Returns None if there was never a WEAK to recover from.
        """
        qualities = [t.quality for t in self.retrieve_calls]
        if "WEAK" not in qualities:
            return None
        weak_idx = qualities.index("WEAK")
        return "GOOD" in qualities[weak_idx + 1:]

    def escalated_to_web(self) -> bool:
        """web_search invoked at any point after a document retrieval attempt."""
        seen_retrieve = False
        for t in self.tool_calls:
            if t.tool == "retrieve_documents":
                seen_retrieve = True
            elif t.tool == "web_search" and seen_retrieve:
                return True
        return bool(self.web_calls)  # also count direct web-first escalation

    # ── Construction from the raw orchestrator state ────────────────────────

    @classmethod
    def from_state(cls, query: str, state: dict[str, Any], latency_s: float = 0.0,
                   error: Optional[str] = None) -> "EvalTrace":
        if error is not None or state is None:
            return cls(query=query, latency_s=latency_s, error=error or "no state returned")

        # Tool calls live in trace["orchestrator"]["tool_calls"] (may be absent
        # for direct conversational / general_knowledge paths).
        orch = (state.get("trace") or {}).get("orchestrator") or {}
        raw_calls = orch.get("tool_calls", []) or []
        tool_calls = [
            ToolCall(
                tool=c.get("tool", ""),
                args=c.get("args", {}) or {},
                observation=c.get("obs", "") or c.get("observation", ""),
            )
            for c in raw_calls
        ]

        # retrieved_chunks are dataclass instances → normalize to plain dicts
        chunks = []
        for ch in state.get("retrieved_chunks", []) or []:
            chunks.append({
                "chunk_id": getattr(ch, "chunk_id", None),
                "content": getattr(ch, "content", "") or "",
                "source": getattr(ch, "source", "") or "",
                "page": getattr(ch, "page", None),
                "score": float(getattr(ch, "score", 0.0) or 0.0),
                "vector_score": float(getattr(ch, "vector_score", 0.0) or 0.0),
            })

        citations = []
        for c in state.get("citations", []) or []:
            citations.append({
                "id": getattr(c, "id", None),
                "source": getattr(c, "source", "") or "",
                "page": getattr(c, "page", None),
                "excerpt": getattr(c, "excerpt", "") or "",
                "relevance_score": float(getattr(c, "relevance_score", 0.0) or 0.0),
            })

        return cls(
            query=query,
            intent=state.get("intent", "document_qa"),
            tool_calls=tool_calls,
            retrieved_chunks=chunks,
            citations=citations,
            final_answer=state.get("answer", "") or "",
            retrieval_confidence=float(state.get("retrieval_confidence", 0.0) or 0.0),
            confidence_score=float(state.get("confidence_score", 0.0) or 0.0),
            reflection_passed=bool(state.get("reflection_passed", True)),
            latency_s=latency_s,
            error=None,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["num_tool_calls"] = self.num_tool_calls
        d["first_tool"] = self.first_tool
        return d
