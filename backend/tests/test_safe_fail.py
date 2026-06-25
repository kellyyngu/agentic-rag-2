"""
Unit tests for graph routing functions.

Tests _route_after_orchestrator (safe-fail gate), _route_after_retrieval
(reflection retry), and _should_continue (reflection loop termination).

No LLM, no external deps. Settings values are read from config but can be
overridden with monkeypatch. Tests use the ACTUAL threshold values so they
catch misconfiguration, not just logic.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.graph import _route_after_orchestrator, _route_after_retrieval, _should_continue
from config import settings


# ─── _route_after_orchestrator (safe-fail gate) ────────────────────────────

class TestRouteAfterOrchestrator:
    """
    Priority order (from code):
      1. intent == general_knowledge → "direct"
      2. conf < safe_fail_threshold AND no web → "safe_fail"
      3. otherwise → "generate"
    """

    def _state(self, intent="document_qa", conf=0.5, web=None):
        return {
            "intent": intent,
            "retrieval_confidence": conf,
            "web_search_results": web or [],
        }

    def test_general_knowledge_routes_to_direct(self):
        state = self._state(intent="general_knowledge")
        assert _route_after_orchestrator(state) == "direct"

    def test_general_knowledge_overrides_low_confidence(self):
        """general_knowledge takes priority even if confidence is 0."""
        state = self._state(intent="general_knowledge", conf=0.0)
        assert _route_after_orchestrator(state) == "direct"

    def test_low_confidence_no_web_triggers_safe_fail(self):
        conf = settings.safe_fail_threshold - 0.01  # just below threshold
        state = self._state(conf=conf, web=[])
        assert _route_after_orchestrator(state) == "safe_fail"

    def test_exactly_at_threshold_does_not_trigger_safe_fail(self):
        """Gate is strictly less-than — at the threshold value, generate."""
        conf = settings.safe_fail_threshold
        state = self._state(conf=conf, web=[])
        assert _route_after_orchestrator(state) == "generate"

    def test_low_confidence_with_web_routes_to_generate(self):
        """Web results save a low-confidence retrieval — do not safe-fail."""
        conf = settings.safe_fail_threshold - 0.01
        state = self._state(conf=conf, web=[{"title": "result", "body": "text"}])
        assert _route_after_orchestrator(state) == "generate"

    def test_good_confidence_routes_to_generate(self):
        state = self._state(conf=0.8)
        assert _route_after_orchestrator(state) == "generate"

    def test_zero_confidence_no_web_safe_fails(self):
        state = self._state(conf=0.0, web=[])
        assert _route_after_orchestrator(state) == "safe_fail"

    def test_missing_confidence_defaults_to_zero(self):
        """AgentState may not have retrieval_confidence set — defaults to 0.0."""
        state = {"intent": "document_qa", "web_search_results": []}
        # 0.0 < safe_fail_threshold → safe_fail
        assert _route_after_orchestrator(state) == "safe_fail"

    def test_empty_web_list_counts_as_no_web(self):
        conf = 0.0
        state = self._state(conf=conf, web=[])
        assert _route_after_orchestrator(state) == "safe_fail"


# ─── _route_after_retrieval (reflection retry routing) ─────────────────────

class TestRouteAfterRetrieval:
    def _state(self, intent="document_qa", chunks=None):
        return {
            "intent": intent,
            "retrieved_chunks": chunks if chunks is not None else [],
        }

    def test_general_knowledge_routes_to_direct(self):
        state = self._state(intent="general_knowledge")
        assert _route_after_retrieval(state) == "direct"

    def test_low_chunk_count_routes_to_web_search(self):
        # web_search_fallback_threshold is 2 by default; 1 chunk < threshold
        state = self._state(chunks=["chunk"])
        assert _route_after_retrieval(state) == "web_search"

    def test_sufficient_chunks_routes_to_generate(self):
        chunks = ["a", "b", "c"]  # 3 >= threshold (2)
        state = self._state(chunks=chunks)
        assert _route_after_retrieval(state) == "generate"

    def test_exactly_at_threshold_routes_to_generate(self):
        chunks = ["a"] * settings.web_search_fallback_threshold
        state = self._state(chunks=chunks)
        assert _route_after_retrieval(state) == "generate"

    def test_empty_chunks_routes_to_web_search(self):
        state = self._state(chunks=[])
        assert _route_after_retrieval(state) == "web_search"


# ─── _should_continue (reflection loop termination) ────────────────────────

class TestShouldContinue:
    def _state(self, passed=True, iteration=0):
        return {
            "reflection_passed": passed,
            "iteration_count": iteration,
        }

    def test_passed_true_returns_end(self):
        assert _should_continue(self._state(passed=True, iteration=0)) == "end"

    def test_failed_within_budget_returns_retrieve(self):
        state = self._state(passed=False, iteration=1)
        assert _should_continue(state) == "retrieve"

    def test_failed_at_max_iterations_returns_end(self):
        max_iter = settings.max_reflection_iterations
        state = self._state(passed=False, iteration=max_iter)
        assert _should_continue(state) == "end"

    def test_failed_beyond_max_returns_end(self):
        max_iter = settings.max_reflection_iterations + 5
        state = self._state(passed=False, iteration=max_iter)
        assert _should_continue(state) == "end"

    def test_passed_at_max_iterations_returns_end(self):
        max_iter = settings.max_reflection_iterations
        state = self._state(passed=True, iteration=max_iter)
        assert _should_continue(state) == "end"

    def test_missing_reflection_passed_defaults_to_true(self):
        """Default of True means the loop exits — safe default."""
        state = {"iteration_count": 0}
        assert _should_continue(state) == "end"
