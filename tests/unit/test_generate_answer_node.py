# HCR-C2-002 — Unit Tests: GenerateAnswerNode (inner domain node 4)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# `criterion_assessment` / `grounded_answer` / `citations` are DOMAIN fields
# (not PII-mask scan targets), so Title-Case criterion text is safe to assert on.
#
# ⚠️ SAFETY-BOUNDARY COVERAGE: this node NEVER emits an eligibility verdict —
# proven both by pinning the exact rendered phrasing (which uses only the
# hedged status labels) and by a closed-vocabulary scan of the FULL rendered
# answer body across every status combination.
#
# Mirrors docs/03_test_spec.md §2.5 (GEN-01..GEN-07).
# Deterministic — rule-assembled from criterion_assessment only (grounded by
# construction; no LLM, no network). framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.trust_level import TrustLevel

import src.nodes.generate_answer_node
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.schemas.state import from_json, to_json

_FORBIDDEN_VERDICT_WORDS = ("eligible", "ineligible", " meets ", " fails ", "enrolled")


def _entry(cid, criterion_type, status, matched_term="matched phrase", text=None, source="Protocol note"):
    return {
        "id": cid,
        "trial_id": "XYZ-123",
        "criterion_type": criterion_type,
        "category": "diagnosis",
        "criterion_text": text or f"criterion text for {cid}",
        "status": status,
        "score": 1.0 if matched_term else 0.0,
        "matched_term": matched_term,
        "source": source,
    }


def _make_state(assessment, overall_abstain=False, **extra) -> dict:
    state = {
        "criterion_assessment": to_json(assessment),
        "overall_abstain": overall_abstain,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


_MIXED_ASSESSMENT = [
    _entry("inc1", "inclusion", "appears_satisfied", matched_term="adult", text="Adult 18+"),
    _entry("inc2", "inclusion", "insufficient_info", matched_term=None, text="NSCLC stage"),
    _entry("exc1", "exclusion", "flagged_for_review", matched_term="chemo", text="Prior chemo"),
    _entry("exc2", "exclusion", "appears_satisfied", matched_term="brain mets", text="Brain mets"),
]


class TestGroundedAnswerComposition:
    def test_gen_01_answer_is_grouped_inclusion_then_exclusion(self):
        answer = GenerateAnswerNode()(_make_state(_MIXED_ASSESSMENT))["grounded_answer"]
        assert answer.index("## Inclusion Criteria") < answer.index("## Exclusion Criteria")
        assert answer.index("Adult 18+") < answer.index("Prior chemo")

    def test_gen_02_citation_markers_are_numbered_in_render_order(self):
        answer = GenerateAnswerNode()(_make_state(_MIXED_ASSESSMENT))["grounded_answer"]
        assert "Adult 18+ — APPEARS SATISFIED [1]:" in answer
        assert "Prior chemo — FLAGGED FOR REVIEW [2]:" in answer
        assert "Brain mets — APPEARS SATISFIED [3]:" in answer

    def test_gen_03_insufficient_info_entries_carry_no_citation_marker(self):
        answer = GenerateAnswerNode()(_make_state(_MIXED_ASSESSMENT))["grounded_answer"]
        assert "NSCLC stage — INSUFFICIENT INFO:" in answer
        assert "NSCLC stage — INSUFFICIENT INFO [" not in answer

    def test_gen_04_citations_list_mirrors_the_markers(self):
        citations = from_json(GenerateAnswerNode()(_make_state(_MIXED_ASSESSMENT))["citations"])
        assert [c["ref"] for c in citations] == [1, 2, 3]
        assert [c["id"] for c in citations] == ["inc1", "exc1", "exc2"]
        # insufficient_info (inc2) never appears in citations.
        assert "inc2" not in [c["id"] for c in citations]

    def test_citations_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = GenerateAnswerNode()(_make_state(_MIXED_ASSESSMENT))
        assert isinstance(result["citations"], str)

    def test_gen_05_flagged_summary_line_present_with_correct_count(self):
        answer = GenerateAnswerNode()(_make_state(_MIXED_ASSESSMENT))["grounded_answer"]
        assert "**1 criterion/criteria flagged for review above.**" in answer
        assert "This is a pre-screen signal, not a determination." in answer

    def test_no_flagged_summary_line_when_nothing_flagged(self):
        assessment = [_entry("inc1", "inclusion", "appears_satisfied")]
        answer = GenerateAnswerNode()(_make_state(assessment))["grounded_answer"]
        assert "flagged for review above" not in answer


class TestAbstainPath:
    def test_gen_06_abstain_emits_only_the_abstention_message(self):
        result = GenerateAnswerNode()(_make_state([], overall_abstain=True))
        assert "too short/sparse" in result["grounded_answer"]
        assert "study team for manual review" in result["grounded_answer"]
        assert from_json(result["citations"]) == []

    def test_abstain_ignores_any_leftover_assessment_data(self):
        """Even if criterion_assessment were non-empty, overall_abstain=True
        must short-circuit before any per-criterion synthesis."""
        result = GenerateAnswerNode()(_make_state(_MIXED_ASSESSMENT, overall_abstain=True))
        assert "Adult 18+" not in result["grounded_answer"]
        assert from_json(result["citations"]) == []


class TestNoCriteriaAssessed:
    def test_empty_non_abstain_assessment_yields_explicit_no_criteria_message(self):
        result = GenerateAnswerNode()(_make_state([], overall_abstain=False))
        assert "No trial criteria were assessed" in result["grounded_answer"]


class TestSafetyBoundaryVocabulary:
    """This node must NEVER emit an eligibility verdict — see
    docs/02_design.md "Safety Boundary" #1."""

    def test_gen_07_rendered_answer_never_contains_verdict_language(self):
        for assessment, abstain in (
            (_MIXED_ASSESSMENT, False),
            ([], True),
            ([], False),
        ):
            answer = GenerateAnswerNode()(_make_state(assessment, overall_abstain=abstain))["grounded_answer"]
            lowered = f" {answer.lower()} "
            for forbidden in _FORBIDDEN_VERDICT_WORDS:
                assert forbidden not in lowered, f"forbidden verdict word {forbidden!r} leaked into: {answer}"


class TestGenerateAnswerAudit:
    def test_gen_08_domain_audit_payload_non_abstain(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.generate_answer_node, "emit_trace_event", spy)
        GenerateAnswerNode()(_make_state(_MIXED_ASSESSMENT))
        events = [call.args[0] for call in spy.call_args_list]
        assert "generate_answer_complete" in events
        payload = spy.call_args_list[events.index("generate_answer_complete")].args[1]
        assert payload["overall_abstain"] is False
        assert payload["citation_count"] == 3
        assert payload["flagged_for_review_count"] == 1

    def test_gen_09_domain_audit_payload_abstain(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.generate_answer_node, "emit_trace_event", spy)
        GenerateAnswerNode()(_make_state([], overall_abstain=True))
        events = [call.args[0] for call in spy.call_args_list]
        payload = spy.call_args_list[events.index("generate_answer_complete")].args[1]
        assert payload["overall_abstain"] is True
        assert payload["citation_count"] == 0
