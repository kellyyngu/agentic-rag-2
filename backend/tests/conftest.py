"""
Stub out all heavy/external dependencies before any test module is collected.

The package __init__.py files eagerly import the full chain
(loguru → langgraph → google-genai → sentence-transformers → qdrant-client).
None of those are installed in a plain CI environment without Docker.

We stub exactly the modules that are imported at module load time — not the
functions we're testing (those are pure Python with no external calls).
"""
import sys
import types


def _make_stub(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    # Module-level __getattr__ (PEP 562): called when attribute not found normally.
    # Signature is (key) — no self.
    mod.__getattr__ = lambda key: _StubAttr()  # type: ignore[assignment]
    return mod


class _StubAttr:
    """Returned for any attribute on a stub module. Callable, iterable, usable as a base class."""
    def __call__(self, *a, **kw):
        return _StubAttr()
    def __iter__(self):
        return iter([])
    def __getattr__(self, key):
        return _StubAttr()
    def __class_getitem__(cls, item):
        return cls


# ── Logging ────────────────────────────────────────────────────────────────
loguru_mod = _make_stub("loguru")
loguru_mod.logger = _StubAttr()
sys.modules.setdefault("loguru", loguru_mod)

# ── LangGraph ──────────────────────────────────────────────────────────────
for name in ["langgraph", "langgraph.graph"]:
    sys.modules.setdefault(name, _make_stub(name))

# Provide StateGraph and END that the graph module needs at import time
lg_graph = sys.modules["langgraph.graph"]
lg_graph.StateGraph = _StubAttr  # type: ignore[attr-defined]
lg_graph.END = "END"             # type: ignore[attr-defined]

# ── Google Generative AI ───────────────────────────────────────────────────
for name in ["google", "google.genai", "google.genai.types"]:
    sys.modules.setdefault(name, _make_stub(name))

# ── ML / retrieval heavy deps ──────────────────────────────────────────────
for name in [
    "sentence_transformers",
    "qdrant_client", "qdrant_client.models",
    "rank_bm25",
    "sklearn", "sklearn.preprocessing",
    "pydantic", "pydantic_settings",
    "pypdf2", "PyPDF2",
    "docx",
    "httpx",
    "aiofiles",
    "fastapi",
    "uvicorn",
    "tenacity",
]:
    sys.modules.setdefault(name, _make_stub(name))

# ── pydantic_settings (config.py uses BaseSettings) ───────────────────────
ps = sys.modules["pydantic_settings"]
ps.BaseSettings = object  # type: ignore[attr-defined]

# ── config.settings (referenced by graph/router/orchestrator at import) ───
# Provide a minimal real settings-like object with the values the tests need.
class _Settings:
    gemini_api_key: str = "test-key"
    llm_model: str = "gemini-2.0-flash"
    safe_fail_threshold: float = 0.15
    grounding_threshold: float = 0.30
    min_vector_score: float = 0.10
    web_search_fallback_threshold: int = 2
    max_reflection_iterations: int = 2
    confidence_threshold: float = 0.50
    retrieval_relevance_threshold: float = 0.2
    orchestrator_max_iterations: int = 3
    orchestrator_quality_threshold: float = 0.30
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rerank_top_k: int = 8
    final_top_k: int = 5


_settings_instance = _Settings()
config_mod = types.ModuleType("config")
config_mod.settings = _settings_instance  # type: ignore[attr-defined]
sys.modules["config"] = config_mod
