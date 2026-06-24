import time
import uuid
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Span:
    name: str
    start: float = field(default_factory=time.time)
    end: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def finish(self, **kwargs):
        self.end = time.time()
        self.metadata.update(kwargs)
        return self

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            return (time.time() - self.start) * 1000
        return (self.end - self.start) * 1000


@dataclass
class Trace:
    request_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    query: str = ""
    spans: list = field(default_factory=list)
    start: float = field(default_factory=time.time)

    def span(self, name: str) -> Span:
        s = Span(name=name)
        self.spans.append(s)
        return s

    def to_dict(self) -> Dict[str, Any]:
        total_ms = (time.time() - self.start) * 1000
        return {
            "request_id": self.request_id,
            "query": self.query[:100],
            "total_ms": round(total_ms, 1),
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 1),
                    **s.metadata,
                }
                for s in self.spans
            ],
        }
