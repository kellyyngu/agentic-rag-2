"""
Configuration contract tests.

These do NOT test behaviour — they test the structural invariants of config.py
itself. Three classes, each a distinct kind of guarantee:

  TestWeightInvariants   — mathematical: weights that must sum to 1.0
  TestThresholdRanges    — range: every float threshold must be in [0.0, 1.0]
  TestOrderingConstraints — relational: threshold pairs whose relative order is
                            load-bearing (wrong ordering causes silent misbehavior
                            that no single behaviour test surfaces)

WHY THIS IS NOT REDUNDANT WITH EXISTING TESTS
Existing tests (test_safe_fail.py, test_confidence.py) assert *behaviour* by
reading threshold values from settings — they pass as long as the thresholds are
internally consistent with the logic. None of them assert the values themselves.
This file catches the complementary failure: a threshold value that is
individually legal but breaks a cross-threshold invariant.

EXAMPLE: setting safe_fail_threshold = 0.35 (above grounding_threshold 0.30)
would let test_safe_fail.py pass (the routing logic still works at any value
< 1.0) while silently making it impossible for any answer to be grounded —
because the gate that rejects weak retrievals would fire before grounding kicks
in. This file catches that in < 1 ms with no LLM, no retriever, no graph.
"""
import pytest
from config import settings


class TestWeightInvariants:
    """Weights that are intended to combine into a single score must add up.
    config.py line 70: 'The three weights are intended to sum to 1.'"""

    def test_doc_confidence_weights_sum_to_one(self):
        total = (
            settings.confidence_doc_llm_weight
            + settings.confidence_doc_retrieval_weight
            + settings.confidence_doc_citation_weight
        )
        assert abs(total - 1.0) < 1e-9, (
            f"doc confidence weights sum to {total:.4f}, expected 1.0 — "
            f"(llm={settings.confidence_doc_llm_weight}, "
            f"retrieval={settings.confidence_doc_retrieval_weight}, "
            f"citation={settings.confidence_doc_citation_weight})"
        )

    def test_web_confidence_formula_cannot_exceed_one(self):
        # max web score = base + llm_weight * 1.0 (when LLM self-rates 1.0)
        max_web = settings.confidence_web_base + settings.confidence_web_llm_weight
        assert max_web <= 1.0, (
            f"web confidence formula max = {max_web:.3f} > 1.0 — "
            "answers would receive impossible confidence scores"
        )


class TestThresholdRanges:
    """Every float that is used as a probability or cosine threshold must be
    in [0.0, 1.0]. An out-of-range value would silently corrupt routing
    decisions or confidence scores without raising an exception."""

    FLOAT_THRESHOLDS = [
        ("safe_fail_threshold",              0.0, 1.0),
        ("grounding_threshold",              0.0, 1.0),
        ("confidence_threshold",             0.0, 1.0),
        ("orchestrator_quality_threshold",   0.0, 1.0),
        ("retrieval_relevance_threshold",    0.0, 1.0),
        ("min_vector_score",                 0.0, 1.0),
        ("confidence_ungrounded_cap",        0.0, 1.0),
        ("confidence_web_base",              0.0, 1.0),
        ("confidence_web_llm_weight",        0.0, 1.0),
        ("confidence_doc_llm_weight",        0.0, 1.0),
        ("confidence_doc_retrieval_weight",  0.0, 1.0),
        ("confidence_doc_citation_weight",   0.0, 1.0),
    ]

    @pytest.mark.parametrize("name,lo,hi", FLOAT_THRESHOLDS)
    def test_float_threshold_in_range(self, name, lo, hi):
        val = getattr(settings, name)
        assert lo <= val <= hi, f"settings.{name} = {val} is outside [{lo}, {hi}]"

    def test_iteration_limits_are_positive(self):
        assert settings.orchestrator_max_iterations >= 1
        assert settings.max_reflection_iterations >= 1

    def test_retrieval_pipeline_sizes_are_positive(self):
        assert settings.bm25_top_k >= 1
        assert settings.vector_top_k >= 1
        assert settings.rerank_top_k >= 1
        assert settings.final_top_k >= 1


class TestOrderingConstraints:
    """Relational invariants between threshold pairs. Violating any of these
    would create unreachable code paths or contradictory routing decisions that
    no single-function behaviour test would surface directly.

    Each assertion includes an explanation of exactly what breaks when violated.
    """

    def test_safe_fail_below_grounding(self):
        # safe_fail fires when retrieval_conf < safe_fail_threshold AND no web.
        # grounding fires when top_cited_score < grounding_threshold.
        # If safe_fail >= grounding, every query below the grounding floor is
        # refused before it can attempt grounding — the grounded-answer path
        # becomes unreachable for any query that borders on the grounding gate.
        assert settings.safe_fail_threshold < settings.grounding_threshold, (
            f"safe_fail_threshold ({settings.safe_fail_threshold}) must be < "
            f"grounding_threshold ({settings.grounding_threshold})"
        )

    def test_safe_fail_below_confidence_threshold(self):
        # confidence_threshold is the reflector fast-pass bar. If safe_fail >=
        # confidence_threshold, answers that survive the gate could still score
        # below the fast-pass bar, causing unnecessary reflection retries on
        # already-borderline retrievals.
        assert settings.safe_fail_threshold < settings.confidence_threshold, (
            f"safe_fail_threshold ({settings.safe_fail_threshold}) must be < "
            f"confidence_threshold ({settings.confidence_threshold})"
        )

    def test_ungrounded_cap_below_web_base(self):
        # ungrounded_cap is the confidence ceiling for negative/off-topic answers.
        # web_base is the confidence floor for any web-grounded answer. If
        # ungrounded_cap >= web_base, a "not found" answer could match or exceed
        # the confidence of a live-web-grounded answer — inverting the signal.
        assert settings.confidence_ungrounded_cap < settings.confidence_web_base, (
            f"confidence_ungrounded_cap ({settings.confidence_ungrounded_cap}) must be < "
            f"confidence_web_base ({settings.confidence_web_base})"
        )

    def test_rerank_top_k_gte_final_top_k(self):
        # The reranker selects final_top_k from a candidate set of rerank_top_k.
        # If rerank_top_k < final_top_k, the reranker can never return enough
        # chunks to fill the pipeline — final retrieval is silently truncated.
        assert settings.rerank_top_k >= settings.final_top_k, (
            f"rerank_top_k ({settings.rerank_top_k}) must be >= "
            f"final_top_k ({settings.final_top_k})"
        )

    def test_chunk_cache_covers_retrieval_working_set(self):
        # The chunk cache must comfortably hold one query's working set so no
        # in-flight lookup is evicted mid-retrieve. config.py documents this
        # invariant explicitly; this test locks it in.
        working_set = settings.bm25_top_k + settings.vector_top_k
        assert settings.chunk_cache_size > working_set, (
            f"chunk_cache_size ({settings.chunk_cache_size}) must be > "
            f"bm25_top_k + vector_top_k ({working_set})"
        )

