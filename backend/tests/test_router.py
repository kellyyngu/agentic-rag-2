"""
Unit tests for intent router logic.

Tests only the deterministic parts — the regex fast path and the DIRECT_INTENTS
set. The LLM classifier (_classify_with_llm) is NOT tested here (non-deterministic).

Protects against: regex regressions that would send greetings to the RAG pipeline,
DIRECT_INTENTS set drift that would strip routes from the graph.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.nodes.intent_router import _CONVERSATIONAL_RE, _WEB_QUERY_RE, DIRECT_INTENTS


class TestConversationalRegex:
    """The regex fast-path avoids an LLM call for obvious conversational queries."""

    # ── Should match (conversational, no LLM needed) ──────────────────────

    @pytest.mark.parametrize("query", [
        "hi", "Hi!", "HI", "hello", "hey", "howdy", "hiya", "yo", "sup",
        "good morning", "good afternoon", "good evening", "good night", "good day",
        "how are you", "how are u", "how are you doing", "how are you today",
        "what's up", "whats up",
        "thanks", "thank you", "ty", "thx", "cheers", "thanks!",
        "bye", "goodbye", "see ya", "later", "cya", "farewell",
        "nice", "great", "awesome", "cool", "ok", "okay", "got it", "sounds good",
        "lol", "haha", ":)",
    ])
    def test_matches_conversational(self, query):
        assert _CONVERSATIONAL_RE.match(query), f"Expected match for: {query!r}"

    # ── Should NOT match (must go to LLM classifier) ──────────────────────

    @pytest.mark.parametrize("query", [
        "what does OSM-PINN stand for",
        "explain the methodology",
        "what is machine learning",
        "summarize my document",
        "what happened in the news today",
        "who are you",
        "what can you do",
        # Longer conversational-ish sentences that contain real questions
        "hi can you help me understand BiLSTM",
        "thanks for that, now explain ReLU",
        "okay but what is the violation rate",
    ])
    def test_does_not_match_non_conversational(self, query):
        assert not _CONVERSATIONAL_RE.match(query), f"Expected no match for: {query!r}"


class TestWebQueryRegex:
    """Real-time queries must route to web_search deterministically — the LLM
    classifier occasionally mislabels them as document_qa (the weather bug)."""

    @pytest.mark.parametrize("query", [
        "how is the weather in malaysia today",
        "what's the weather like",
        "weather forecast for tomorrow",
        "latest news on the election",
        "breaking news",
        "today's news headlines",
        "stock price of AAPL",
        "what is the share price of Tesla",
        "current exchange rate USD to MYR",
        "current events in europe",
    ])
    def test_matches_web_query(self, query):
        assert _WEB_QUERY_RE.search(query), f"Expected web match for: {query!r}"

    @pytest.mark.parametrize("query", [
        "what does OSM-PINN stand for",
        "summarize my document",
        "explain the methodology",
        "what is machine learning",
        # Must NOT steal document queries that happen to mention a report
        "what does the report say about the model",
    ])
    def test_does_not_match_document_query(self, query):
        assert not _WEB_QUERY_RE.search(query), f"Expected no web match for: {query!r}"


class TestDirectIntents:
    """DIRECT_INTENTS controls which intents bypass the RAG pipeline entirely.

    If an intent is removed or renamed here, the graph silently routes it to
    the orchestrator — a silent behavior change. These tests catch that.
    """

    def test_conversational_is_direct(self):
        assert "conversational" in DIRECT_INTENTS

    def test_general_knowledge_is_direct(self):
        assert "general_knowledge" in DIRECT_INTENTS

    def test_assistant_identity_is_direct(self):
        assert "assistant_identity" in DIRECT_INTENTS

    def test_document_qa_is_not_direct(self):
        assert "document_qa" not in DIRECT_INTENTS

    def test_web_search_is_not_direct(self):
        assert "web_search" not in DIRECT_INTENTS

    def test_document_summary_is_not_direct(self):
        # document_summary falls through to orchestrator — confirm not in DIRECT_INTENTS
        assert "document_summary" not in DIRECT_INTENTS
