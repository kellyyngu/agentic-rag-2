"""
Unit tests for the extracted generation-pipeline functions:

  parse_generation      — split streamed text into answer + metadata
  build_citations       — inline [N] → Citation objects + global-ID remap
  calibrate_confidence  — grounding-aware confidence + citation suppression

These exercise the confidence policy WITHOUT a token stream, and verify that the
calibration weights are read from config (Refactor 4) rather than hardcoded.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from agent.nodes.generator import (
    parse_generation,
    build_citations,
    calibrate_confidence,
)
from agent.state import RetrievedChunk, Citation
from config import settings


def _chunk(cid, vscore, content, score=None, source="s.pdf", page=1):
    return RetrievedChunk(
        chunk_id=cid, content=content, source=source, page=page,
        score=score if score is not None else vscore, vector_score=vscore,
    )


def _meta_text(answer, conf, follow_ups=None):
    import json
    meta = json.dumps({"follow_up_questions": follow_ups or [], "confidence_score": conf})
    return f"{answer}\n<<<JSON\n{meta}\n>>>"


# ─── parse_generation ───────────────────────────────────────────────────────

class TestParseGeneration:
    def test_parses_metadata_block(self):
        answer, fups, score = parse_generation(_meta_text("The answer.", 0.9, ["q1?"]))
        assert answer == "The answer."
        assert fups == ["q1?"]
        assert score == 0.9

    def test_normalizes_out_of_range_score(self):
        """A model that emits 9 instead of 0.9 must be divided down."""
        _, _, score = parse_generation(_meta_text("X.", 9))
        assert score == 0.9

    def test_clamps_score_to_unit_interval(self):
        _, _, score = parse_generation(_meta_text("X.", 50))  # 50 → 5.0 → clamp 1.0
        assert score == 1.0

    def test_no_metadata_defaults(self):
        answer, fups, score = parse_generation("Plain answer, no metadata.")
        assert answer == "Plain answer, no metadata."
        assert fups == []
        assert score == 0.75

    def test_truncated_json_still_strips(self):
        answer, fups, score = parse_generation('Real answer.\n<<<JSON\n{"follow_up')
        assert answer == "Real answer."
        assert "<<<JSON" not in answer
        assert score == 0.75


# ─── build_citations ────────────────────────────────────────────────────────

class TestBuildCitations:
    def test_citation_rich_answer(self):
        chunk_map = {"1": _chunk("a", 0.6, "OSM-PINN uses asymmetric loss for monotonicity.")}
        cits, answer = build_citations("OSM-PINN uses asymmetric loss [1].", chunk_map, {"1": "7"}, "q")
        assert len(cits) == 1
        assert cits[0].id == "7"
        assert cits[0].relevance_score == 0.6      # vector_score used for display
        assert "[7]" in answer and "[1]" not in answer

    def test_citation_free_answer(self):
        chunk_map = {"1": _chunk("a", 0.6, "content")}
        cits, answer = build_citations("No citations here.", chunk_map, {"1": "7"}, "q")
        assert cits == []
        assert answer == "No citations here."

    def test_grouped_citations(self):
        chunk_map = {"1": _chunk("a", 0.6, "fact one"), "2": _chunk("b", 0.5, "fact two")}
        cits, answer = build_citations("Both apply [1, 2].", chunk_map, {"1": "3", "2": "4"}, "q")
        assert {c.id for c in cits} == {"3", "4"}
        assert "[3, 4]" in answer


# ─── calibrate_confidence ───────────────────────────────────────────────────

class TestCalibrateConfidence:
    def test_document_grounded_blend(self):
        cits = [Citation(id="1", source="s", page=1, excerpt="e", relevance_score=0.6)]
        out_cits, answer, conf = calibrate_confidence(
            answer="Grounded answer [1].", citations=cits,
            retrieval_conf=0.5, has_web=False, llm_score=0.8, num_chunks=2,
        )
        # 0.8*0.4 + 0.5*0.4 + (1/2)*0.2 = 0.32 + 0.20 + 0.10 = 0.62
        assert conf == 0.62
        assert len(out_cits) == 1

    def test_web_grounded_floor(self):
        out_cits, answer, conf = calibrate_confidence(
            answer="1 USD is 4.7 MYR today.", citations=[],
            retrieval_conf=0.0, has_web=True, llm_score=0.1, num_chunks=0,
        )
        # 0.45 + 0.1*0.35 = 0.485
        assert conf == 0.485
        assert out_cits == []

    def test_negative_answer_suppressed(self):
        cits = [Citation(id="1", source="s", page=1, excerpt="e", relevance_score=0.6)]
        out_cits, answer, conf = calibrate_confidence(
            answer="The documents do not mention quantum gravity [1].", citations=cits,
            retrieval_conf=0.4, has_web=False, llm_score=0.85, num_chunks=2,
        )
        assert out_cits == []                       # citations dropped
        assert "[1]" not in answer                  # dangling marker stripped
        assert conf == 0.25                          # capped, despite 0.85 self-rating

    def test_citation_free_doc_answer_is_ungrounded(self):
        """No citations + no web → not grounded → confidence capped."""
        out_cits, answer, conf = calibrate_confidence(
            answer="A confident-sounding but uncited answer.", citations=[],
            retrieval_conf=0.18, has_web=False, llm_score=0.9, num_chunks=3,
        )
        assert out_cits == []
        assert conf == 0.18                          # min(0.25, retrieval_conf)

    def test_weak_citation_below_grounding_threshold(self):
        cits = [Citation(id="1", source="s", page=1, excerpt="e", relevance_score=0.1)]
        out_cits, _, conf = calibrate_confidence(
            answer="Weakly grounded [1].", citations=cits,
            retrieval_conf=0.2, has_web=False, llm_score=0.9, num_chunks=2,
        )
        assert out_cits == []                        # 0.1 < grounding_threshold(0.30)
        assert conf == 0.2


# ─── Refactor 4: config weights are respected ───────────────────────────────

class TestConfigDrivenWeights:
    def test_web_base_is_configurable(self, monkeypatch):
        monkeypatch.setattr(settings, "confidence_web_base", 0.90)
        monkeypatch.setattr(settings, "confidence_web_llm_weight", 0.0)
        _, _, conf = calibrate_confidence(
            answer="web answer", citations=[],
            retrieval_conf=0.0, has_web=True, llm_score=0.5, num_chunks=0,
        )
        assert conf == 0.9                           # base only, llm weight zeroed

    def test_ungrounded_cap_is_configurable(self, monkeypatch):
        monkeypatch.setattr(settings, "confidence_ungrounded_cap", 0.05)
        _, _, conf = calibrate_confidence(
            answer="uncited answer", citations=[],
            retrieval_conf=0.4, has_web=False, llm_score=0.9, num_chunks=2,
        )
        assert conf == 0.05                          # capped lower by config

    def test_doc_weights_are_configurable(self, monkeypatch):
        monkeypatch.setattr(settings, "confidence_doc_llm_weight", 1.0)
        monkeypatch.setattr(settings, "confidence_doc_retrieval_weight", 0.0)
        monkeypatch.setattr(settings, "confidence_doc_citation_weight", 0.0)
        cits = [Citation(id="1", source="s", page=1, excerpt="e", relevance_score=0.6)]
        _, _, conf = calibrate_confidence(
            answer="Grounded [1].", citations=cits,
            retrieval_conf=0.5, has_web=False, llm_score=0.7, num_chunks=2,
        )
        assert conf == 0.7                           # now purely the llm self-rating
