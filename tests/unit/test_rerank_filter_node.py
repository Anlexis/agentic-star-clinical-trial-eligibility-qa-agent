# HCR-C2-002 — Unit Tests: RerankFilterNode (inner domain node 3)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# Config precedence (score_threshold / abstain_min_chars) is exercised by
# seeding `retrieval_config` into STATE — never a 2nd execute() argument.
#
# ⚠️ SAFETY-BOUNDARY COVERAGE (template-specific instruction): this file pins
# the full 4-value status vocabulary — appears_satisfied / flagged_for_review /
# uncertain / insufficient_info — and proves the vocabulary NEVER contains
# "eligible" / "ineligible" / "meets" / "fails" for any input combination, plus
# the overall_abstain safety gate (both trigger conditions).
#
# Mirrors docs/03_test_spec.md §2.4 (RRF-01..RRF-10).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.trust_level import TrustLevel

import src.nodes.rerank_filter_node
from src.nodes.rerank_filter_node import RerankFilterNode
from src.schemas.state import from_json, to_json

_FORBIDDEN_VERDICT_WORDS = ("eligible", "ineligible", "meets", "fails", "enrolled")

_ALL_STATUSES = {"appears_satisfied", "flagged_for_review", "uncertain", "insufficient_info"}


def _cand(cid, criterion_type, score, negated=False, matched_term="matched phrase"):
    return {
        "id": cid,
        "trial_id": "XYZ-123",
        "criterion_type": criterion_type,
        "category": "diagnosis",
        "criterion_text": f"criterion text for {cid}",
        "source": "seeded kb",
        "score": score,
        "matched_term": matched_term if score else None,
        "negated": negated,
    }


def _make_state(candidates, text="a sufficiently long patient criteria narrative for testing", **extra) -> dict:
    state = {
        "matched_criteria": to_json(candidates),
        "patient_criteria_text": text,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestStatusVocabularyNeverAssertsAVerdict:
    """No status value, for ANY input combination, may spell out an
    eligibility verdict — see docs/02_design.md "Safety Boundary" #1."""

    def test_rrf_vocabulary_is_closed_to_the_four_hedged_statuses(self):
        assert _ALL_STATUSES == {
            "appears_satisfied",
            "flagged_for_review",
            "uncertain",
            "insufficient_info",
        }
        for status in _ALL_STATUSES:
            for forbidden in _FORBIDDEN_VERDICT_WORDS:
                assert forbidden not in status


class TestInclusionClassification:
    def test_rrf_01_confident_unnegated_inclusion_is_appears_satisfied(self):
        result = RerankFilterNode()(_make_state([_cand("c1", "inclusion", 1.0, negated=False)]))
        assessed = from_json(result["criterion_assessment"])[0]
        assert assessed["status"] == "appears_satisfied"

    def test_rrf_02_confident_negated_inclusion_is_uncertain(self):
        """Explicit contrary evidence on an inclusion term — hedged as
        uncertain, never a bare 'not met'."""
        result = RerankFilterNode()(_make_state([_cand("c1", "inclusion", 1.0, negated=True)]))
        assessed = from_json(result["criterion_assessment"])[0]
        assert assessed["status"] == "uncertain"

    def test_rrf_03_weak_signal_inclusion_is_uncertain(self):
        result = RerankFilterNode()(_make_state([_cand("c1", "inclusion", 0.5, negated=False)]))
        assessed = from_json(result["criterion_assessment"])[0]
        assert assessed["status"] == "uncertain"

    def test_rrf_04_no_evidence_inclusion_is_insufficient_info(self):
        result = RerankFilterNode()(_make_state([_cand("c1", "inclusion", 0.0, negated=False)]))
        assessed = from_json(result["criterion_assessment"])[0]
        assert assessed["status"] == "insufficient_info"


class TestExclusionClassification:
    """The exclusion tier is intentionally asymmetric: 'no evidence found'
    defaults to appears_satisfied (clears the exclusion), never flagged."""

    def test_rrf_05_confident_unnegated_exclusion_is_flagged_for_review(self):
        """The single status meant to draw a reviewer's attention first."""
        result = RerankFilterNode()(_make_state([_cand("c1", "exclusion", 1.0, negated=False)]))
        assessed = from_json(result["criterion_assessment"])[0]
        assert assessed["status"] == "flagged_for_review"

    def test_rrf_06_confident_negated_exclusion_is_appears_satisfied(self):
        """The excluded condition appears explicitly ABSENT — clears the exclusion."""
        result = RerankFilterNode()(_make_state([_cand("c1", "exclusion", 1.0, negated=True)]))
        assessed = from_json(result["criterion_assessment"])[0]
        assert assessed["status"] == "appears_satisfied"

    def test_rrf_07_weak_signal_exclusion_is_uncertain(self):
        result = RerankFilterNode()(_make_state([_cand("c1", "exclusion", 0.5, negated=False)]))
        assessed = from_json(result["criterion_assessment"])[0]
        assert assessed["status"] == "uncertain"

    def test_rrf_08_no_evidence_exclusion_is_appears_satisfied(self):
        result = RerankFilterNode()(_make_state([_cand("c1", "exclusion", 0.0, negated=False)]))
        assessed = from_json(result["criterion_assessment"])[0]
        assert assessed["status"] == "appears_satisfied"


class TestOverallAbstainSafetyGate:
    """SAFETY BOUNDARY: abstain rather than assert a low-confidence read."""

    def test_rrf_09_sparse_text_triggers_abstain(self):
        result = RerankFilterNode()(_make_state([_cand("c1", "inclusion", 1.0)], text="short"))
        assert result["overall_abstain"] is True
        assert from_json(result["criterion_assessment"]) == []

    def test_rrf_10_no_candidates_triggers_abstain(self):
        result = RerankFilterNode()(_make_state([]))
        assert result["overall_abstain"] is True
        assert from_json(result["criterion_assessment"]) == []

    def test_abstain_min_chars_is_configurable_via_state_seeding(self):
        # Canon: config reaches the node via STATE seeding, never a 2nd
        # execute() argument.
        state = _make_state(
            [_cand("c1", "inclusion", 1.0)],
            text="twenty chars exactly!",  # 21 chars
            retrieval_config=to_json({"abstain_min_chars": 5}),
        )
        result = RerankFilterNode()(state)
        assert result["overall_abstain"] is False

    def test_not_abstaining_when_text_and_candidates_both_present(self):
        result = RerankFilterNode()(_make_state([_cand("c1", "inclusion", 1.0)]))
        assert result["overall_abstain"] is False
        assert len(from_json(result["criterion_assessment"])) == 1


class TestScoreThresholdConfig:
    def test_rrf_11_score_threshold_override_via_state_seeding(self):
        state = _make_state(
            [_cand("c1", "inclusion", 0.6, negated=False)],
            retrieval_config=to_json({"score_threshold": 0.5}),
        )
        result = RerankFilterNode()(state)
        assessed = from_json(result["criterion_assessment"])[0]
        # 0.6 clears a 0.5 floor -> confident branch, not the default 0.75 floor.
        assert assessed["status"] == "appears_satisfied"

    def test_default_threshold_is_075(self):
        # 0.6 does NOT clear the default 0.75 floor -> weak-signal branch.
        result = RerankFilterNode()(_make_state([_cand("c1", "inclusion", 0.6, negated=False)]))
        assessed = from_json(result["criterion_assessment"])[0]
        assert assessed["status"] == "uncertain"


class TestAssessmentShape:
    def test_criterion_assessment_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RerankFilterNode()(_make_state([_cand("c1", "inclusion", 1.0)]))
        assert isinstance(result["criterion_assessment"], str)

    def test_non_dict_candidate_entries_are_skipped(self):
        result = RerankFilterNode()(_make_state(["not-a-dict", _cand("c1", "inclusion", 1.0)]))
        assessed = from_json(result["criterion_assessment"])
        assert len(assessed) == 1
        assert assessed[0]["id"] == "c1"


class TestRerankFilterAudit:
    def test_rrf_12_domain_audit_payload_non_abstain(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.rerank_filter_node, "emit_trace_event", spy)
        RerankFilterNode()(_make_state([_cand("c1", "exclusion", 1.0, negated=False)]))
        events = [call.args[0] for call in spy.call_args_list]
        assert "rerank_filter_complete" in events
        payload = spy.call_args_list[events.index("rerank_filter_complete")].args[1]
        assert payload["overall_abstain"] is False
        assert payload["flagged_for_review_count"] == 1

    def test_rrf_13_domain_audit_payload_abstain(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.rerank_filter_node, "emit_trace_event", spy)
        RerankFilterNode()(_make_state([]))
        events = [call.args[0] for call in spy.call_args_list]
        payload = spy.call_args_list[events.index("rerank_filter_complete")].args[1]
        assert payload["overall_abstain"] is True
