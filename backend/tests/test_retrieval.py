"""
Unit tests for retrieval fusion logic.

Tests _reciprocal_rank_fusion directly — pure function, no external deps.
Protects against: wrong rank ordering, score accumulation bugs, single-list
degeneration, empty-input crashes.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from retrieval.hybrid_retriever import _reciprocal_rank_fusion


class TestRRF:
    def _approx(self, a: float, b: float, tol: float = 1e-9) -> bool:
        return abs(a - b) < tol

    def test_single_list_scores_correctly(self):
        ranked = [("a", 1.0), ("b", 0.9), ("c", 0.8)]
        scores = _reciprocal_rank_fusion(ranked)
        # rank 1: 1/(60+1), rank 2: 1/(60+2), rank 3: 1/(60+3)
        assert self._approx(scores["a"], 1 / 61)
        assert self._approx(scores["b"], 1 / 62)
        assert self._approx(scores["c"], 1 / 63)

    def test_two_lists_accumulate_scores(self):
        list1 = [("a", 1.0), ("b", 0.9)]
        list2 = [("b", 1.0), ("a", 0.9)]
        scores = _reciprocal_rank_fusion(list1, list2)
        # "a": rank1 in list1 + rank2 in list2 = 1/61 + 1/62
        # "b": rank2 in list1 + rank1 in list2 = 1/62 + 1/61
        assert self._approx(scores["a"], 1 / 61 + 1 / 62)
        assert self._approx(scores["b"], 1 / 62 + 1 / 61)
        assert self._approx(scores["a"], scores["b"])

    def test_doc_in_one_list_only(self):
        list1 = [("a", 1.0)]
        list2 = [("b", 1.0)]
        scores = _reciprocal_rank_fusion(list1, list2)
        assert "a" in scores
        assert "b" in scores
        assert self._approx(scores["a"], scores["b"])

    def test_hybrid_beats_single_list_for_shared_doc(self):
        """A doc ranked #1 in both lists gets a higher score than one ranked #1 in only one."""
        shared = [("x", 1.0)]
        exclusive = [("y", 1.0)]
        scores = _reciprocal_rank_fusion(shared, shared)   # x appears in both
        scores_single = _reciprocal_rank_fusion(exclusive)  # y appears once
        assert scores["x"] > scores_single["y"]

    def test_empty_input_returns_empty(self):
        assert _reciprocal_rank_fusion([]) == {}

    def test_ordering_is_rank_based_not_score_based(self):
        """RRF ignores the float score value — only rank position matters."""
        same_rank_diff_score = [("a", 100.0), ("b", 0.001)]
        scores = _reciprocal_rank_fusion(same_rank_diff_score)
        # "a" is rank 1, "b" is rank 2 → "a" wins regardless of raw scores
        assert scores["a"] > scores["b"]

    def test_custom_k_changes_scores(self):
        ranked = [("a", 1.0)]
        default = _reciprocal_rank_fusion(ranked, k=60)
        custom = _reciprocal_rank_fusion(ranked, k=1)
        assert self._approx(custom["a"], 1 / 2)   # 1/(1+1)
        assert self._approx(default["a"], 1 / 61)
        assert custom["a"] > default["a"]
