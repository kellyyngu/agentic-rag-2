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
from agent.nodes.generator import (
    _extract_meta,
    _sanitize_json,
    _extract_cited_ids,
    _remap_citation_groups,
    _is_negative_answer,
)
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


# ─── _extract_cited_ids (grouped citation parser) ──────────────────────────

class TestExtractCitedIds:
    """The real bug this fixes: an answer that cites in groups like [2, 4] used to
    yield ZERO citations because the old regex only matched single [N] brackets.
    """

    def test_single_brackets(self):
        ids = _extract_cited_ids("Fact [1] and [3].", {"1", "2", "3"})
        assert ids == ["1", "3"]

    def test_grouped_brackets(self):
        """[2, 4] must yield BOTH 2 and 4 — this is the reported bug."""
        ids = _extract_cited_ids("PINNs were defined [2, 4] and applied [2, 6].", {str(i) for i in range(1, 7)})
        assert ids == ["2", "4", "6"]

    def test_mixed_single_and_grouped(self):
        ids = _extract_cited_ids("A [3]. B [2, 4]. C [5, 6].", {str(i) for i in range(1, 7)})
        assert ids == ["2", "3", "4", "5", "6"]

    def test_filters_out_of_range_paper_refs(self):
        """Source's own bibliography markers ([7], [8] when only 6 chunks) are dropped."""
        ids = _extract_cited_ids("Defined [2, 7] and applied [3, 8].", {str(i) for i in range(1, 7)})
        assert ids == ["2", "3"]
        assert "7" not in ids
        assert "8" not in ids

    def test_deduplicates(self):
        ids = _extract_cited_ids("A [2]. B [2, 4]. C [4].", {"2", "4"})
        assert ids == ["2", "4"]

    def test_no_citations(self):
        assert _extract_cited_ids("Plain answer, no brackets.", {"1", "2"}) == []

    def test_sorted_ascending(self):
        ids = _extract_cited_ids("[6, 1] then [3].", {str(i) for i in range(1, 7)})
        assert ids == ["1", "3", "6"]


# ─── _remap_citation_groups ([local] → [global], grouped) ──────────────────

class TestRemapCitationGroups:
    def test_basic_remap(self):
        result = _remap_citation_groups("Fact [1] and [2].", {"1": "42", "2": "43"})
        assert "[42]" in result
        assert "[43]" in result
        assert "[1]" not in result and "[2]" not in result

    def test_grouped_remap(self):
        """[1, 2] must remap to both global IDs in one bracket."""
        result = _remap_citation_groups("See [1, 2].", {"1": "7", "2": "8"})
        assert result == "See [7, 8]."

    def test_no_corruption_with_double_digit_ids(self):
        """[1] must not corrupt [10] — single-pass re.sub guarantees this."""
        result = _remap_citation_groups("See [1] and [10].", {"1": "A", "10": "B"})
        assert "[A]" in result
        assert "[B]" in result
        assert "[A]0" not in result

    def test_drops_unmapped_paper_refs(self):
        """A number with no chunk mapping (paper's own ref) is removed from the group."""
        result = _remap_citation_groups("Defined [2, 8].", {"2": "2"})  # 8 not in map
        assert result == "Defined [2]."

    def test_bracket_removed_when_all_unmapped(self):
        result = _remap_citation_groups("Stray ref [7, 8] here.", {"2": "2"})
        assert "[7" not in result and "8]" not in result

    def test_no_citations_unchanged(self):
        answer = "Answer with no inline citations."
        assert _remap_citation_groups(answer, {"1": "99"}) == answer

    def test_multiple_occurrences(self):
        result = _remap_citation_groups("See [1]. Also [1].", {"1": "7"})
        assert result.count("[7]") == 2
        assert "[1]" not in result


# ─── _is_negative_answer (grounding gate for "not found" replies) ──────────

class TestIsNegativeAnswer:
    """Negative answers must be detected so citations/confidence can be suppressed.
    The GPT-4 adversarial bug: "the paper does not mention GPT-4" got 71% confidence
    and 6 citations. This detector is what forces those to drop."""

    @pytest.mark.parametrize("answer", [
        "The OSM-PINN paper does not mention GPT-4.",
        "There is no mention of climate change in the report.",
        "The context does not contain information about the weather.",
        "I cannot provide information about today's exchange rate.",
        "The documents do not discuss this topic.",
        "I could not find any relevant information in the provided context.",
        "This is not mentioned in the uploaded documents.",
    ])
    def test_detects_negative(self, answer):
        assert _is_negative_answer(answer) is True

    @pytest.mark.parametrize("answer", [
        "OSM-PINN stands for One-Sided Monotonic Physics-Informed Neural Network.",
        "The loss function combines data loss and physics loss.",
        # Substantive "does not" claim about CONTENT — must NOT trip the detector
        "OSM-PINN does not use symmetric penalties, unlike conventional PINNs.",
        "The model does not require labeled data during the physics phase.",
    ])
    def test_ignores_substantive_answers(self, answer):
        assert _is_negative_answer(answer) is False

    def test_case_insensitive(self):
        assert _is_negative_answer("The Paper DOES NOT MENTION it.") is True

    def test_empty_answer(self):
        assert _is_negative_answer("") is False


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
