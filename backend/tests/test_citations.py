"""
Unit tests for citation logic.

Covers three independent components:
  1. CitationManager  — idempotency, sequential IDs, thread safety
  2. _extract_meta    — 3-tier fallback parser (<<<JSON>>>, ```json```, raw trailing)
  3. Citation remap   — the descending-ID replacement that prevents [1] corrupting [10]
  4. _keyword_recall  — deterministic eval metric

No LLM calls, no external deps.
"""
import re
import threading
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.citation_manager import CitationManager
from agent.nodes.generator import _extract_meta, _sanitize_json
from evaluate.agentic import _keyword_recall


# ─── CitationManager ────────────────────────────────────────────────────────

class TestCitationManager:
    def test_first_assignment_returns_1(self):
        cm = CitationManager()
        assert cm.get_or_assign("chunk-A") == "1"

    def test_same_chunk_returns_same_id(self):
        cm = CitationManager()
        first = cm.get_or_assign("chunk-A")
        second = cm.get_or_assign("chunk-A")
        assert first == second == "1"

    def test_different_chunks_get_different_ids(self):
        cm = CitationManager()
        id_a = cm.get_or_assign("chunk-A")
        id_b = cm.get_or_assign("chunk-B")
        assert id_a != id_b

    def test_ids_are_sequential(self):
        cm = CitationManager()
        id1 = cm.get_or_assign("chunk-1")
        id2 = cm.get_or_assign("chunk-2")
        id3 = cm.get_or_assign("chunk-3")
        assert [id1, id2, id3] == ["1", "2", "3"]

    def test_size_tracks_unique_chunks(self):
        cm = CitationManager()
        cm.get_or_assign("a")
        cm.get_or_assign("b")
        cm.get_or_assign("a")  # duplicate — should not increment
        assert cm.size == 2

    def test_thread_safety_no_duplicate_ids(self):
        """Concurrent assigns must never produce the same ID for different chunks."""
        cm = CitationManager()
        results = {}

        def assign(chunk_id):
            results[chunk_id] = cm.get_or_assign(chunk_id)

        chunks = [f"chunk-{i}" for i in range(50)]
        threads = [threading.Thread(target=assign, args=(c,)) for c in chunks]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        ids = list(results.values())
        assert len(set(ids)) == 50  # all unique

    def test_fresh_instance_resets_counter(self):
        cm1 = CitationManager()
        cm1.get_or_assign("x")
        cm2 = CitationManager()
        assert cm2.get_or_assign("x") == "1"  # fresh counter


# ─── _extract_meta (3-tier fallback parser) ────────────────────────────────

class TestExtractMeta:
    def test_custom_delimiter(self):
        text = 'Answer text\n<<<JSON\n{"citations": [], "confidence_score": 0.9}\n>>>'
        meta, pos = _extract_meta(text)
        assert meta is not None
        assert meta["confidence_score"] == 0.9
        assert pos == text.index("<<<JSON")

    def test_markdown_fenced_json(self):
        text = 'Answer text\n```json\n{"citations": [], "confidence_score": 0.8}\n```'
        meta, pos = _extract_meta(text)
        assert meta is not None
        assert meta["confidence_score"] == 0.8

    def test_raw_trailing_json(self):
        text = 'Answer text\n{"citations": [], "follow_up_questions": ["q?"]}'
        meta, pos = _extract_meta(text)
        assert meta is not None
        assert meta["follow_up_questions"] == ["q?"]

    def test_no_json_returns_none(self):
        meta, pos = _extract_meta("Just plain answer text with no metadata.")
        assert meta is None
        assert pos == -1

    def test_empty_string(self):
        meta, pos = _extract_meta("")
        assert meta is None
        assert pos == -1

    def test_malformed_json_falls_through(self):
        # Malformed <<<JSON block should not raise — falls through to None
        text = 'Answer\n<<<JSON\n{broken json\n>>>'
        meta, pos = _extract_meta(text)
        assert meta is None

    def test_split_pos_is_before_delimiter(self):
        """Answer text before split_pos must not contain the JSON block."""
        answer_part = "The answer is here."
        text = answer_part + "\n<<<JSON\n{\"confidence_score\": 0.7}\n>>>"
        meta, pos = _extract_meta(text)
        assert text[:pos].strip() == answer_part

    def test_confidence_score_present(self):
        text = 'OK\n<<<JSON\n{"citations":[],"follow_up_questions":[],"confidence_score":0.85}\n>>>'
        meta, _ = _extract_meta(text)
        assert meta["confidence_score"] == 0.85


# ─── Citation remap ([local] → [global]) ───────────────────────────────────

class TestCitationRemap:
    """
    Tests the descending-sort replacement logic in generator.run().
    The bug it prevents: replacing [1] before [10] turns "…[10]…" into
    "…[global_1]0…". Replacement must go from highest local ID downward.
    """

    def _remap(self, answer: str, local_to_global: dict[str, str]) -> str:
        """Inline replica of the generator's remap loop for unit testing."""
        for local_id in sorted(local_to_global.keys(), key=lambda x: -int(x)):
            global_id = local_to_global[local_id]
            if local_id != global_id:
                answer = re.sub(rf'\[{re.escape(local_id)}\]', f'[{global_id}]', answer)
        return answer

    def test_basic_remap(self):
        answer = "Fact from [1] and another from [2]."
        mapping = {"1": "42", "2": "43"}
        result = self._remap(answer, mapping)
        assert "[42]" in result
        assert "[43]" in result
        assert "[1]" not in result
        assert "[2]" not in result

    def test_no_corruption_with_double_digit_ids(self):
        """[1] must not corrupt [10] when both exist."""
        answer = "See [1] and also [10]."
        mapping = {"1": "A", "10": "B"}
        result = self._remap(answer, mapping)
        assert "[A]" in result
        assert "[B]" in result
        # The [10] replacement must not leave "[A]0" or "[B0]"
        assert "[A]0" not in result
        assert "0]" not in result.replace("[B]", "")

    def test_no_change_when_local_equals_global(self):
        """When IDs already match (no manager), text must be unchanged."""
        answer = "See [1] and [2]."
        mapping = {"1": "1", "2": "2"}
        result = self._remap(answer, mapping)
        assert result == answer

    def test_remap_with_no_citations_in_text(self):
        answer = "Answer with no inline citations."
        mapping = {"1": "99"}
        result = self._remap(answer, mapping)
        assert result == answer

    def test_same_local_id_multiple_occurrences(self):
        """All occurrences of [1] must be replaced, not just the first."""
        answer = "See [1]. Also [1] confirms this."
        mapping = {"1": "7"}
        result = self._remap(answer, mapping)
        assert result.count("[7]") == 2
        assert "[1]" not in result


# ─── _keyword_recall (deterministic eval metric) ───────────────────────────

class TestKeywordRecall:
    def test_all_keywords_present(self):
        assert _keyword_recall("OSM-PINN uses BiLSTM attention", ["BiLSTM", "attention"]) == 1.0

    def test_no_keywords_present(self):
        assert _keyword_recall("Unrelated answer", ["BiLSTM", "attention"]) == 0.0

    def test_partial_match(self):
        result = _keyword_recall("Uses BiLSTM encoder", ["BiLSTM", "attention"])
        assert abs(result - 0.5) < 1e-9

    def test_case_insensitive(self):
        assert _keyword_recall("uses bilstm", ["BiLSTM"]) == 1.0

    def test_empty_keywords_returns_none(self):
        assert _keyword_recall("some answer", []) is None

    def test_empty_answer_no_match(self):
        assert _keyword_recall("", ["BiLSTM"]) == 0.0

    def test_keyword_substring_match(self):
        # "monotonic" is in "monotonicity" — should match
        assert _keyword_recall("monotonicity violations", ["monotonic"]) == 1.0

    def test_single_keyword(self):
        assert _keyword_recall("The violation rate was 3.62%", ["3.62"]) == 1.0
