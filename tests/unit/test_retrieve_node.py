# HCR-C2-002 — Unit Tests: RetrieveNode (inner domain node 2)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# `patient_criteria_text` is NOT one of the framework's PII-scan fields
# (_PII_SCAN_FIELDS = user_input/validated_input/llm_response only), so these
# payloads are safe to write in realistic clinical phrasing without PII-mask
# interference. Config-precedence tests seed `retrieval_config` into STATE
# (never a 2nd `execute(state, config=...)` argument — nodes take
# `execute(self, state)` only; config reaches nodes via ctor or State).
#
# Mirrors docs/03_test_spec.md §2.3 (RET-01..RET-09).
# Deterministic — keyword/negation matching over the seeded
# config/kb/clinical_trials_kb.json; no LLM, no network. framework.* / src.*
# imports only.

from unittest.mock import MagicMock

from framework.schemas.trust_level import TrustLevel

import src.nodes.retrieve_node
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import from_json, to_json

# Mirrors the VALID_PAYLOAD clinical narrative (minus the "Eligibility
# pre-screen request against trial XYZ-123:" preamble, which InputValidateNode
# strips into query_filters upstream — RetrieveNode only ever sees the
# resolved patient_criteria_text + trial_id).
_XYZ_PATIENT_TEXT = (
    "68-year-old adult patient with histologically confirmed stage ii nsclc, "
    "ecog performance status 1, no prior chemotherapy, no known brain "
    "metastases, adequate organ function on recent labs."
)


def _make_state(text, trial_id="XYZ-123", **extra) -> dict:
    state = {
        "patient_criteria_text": text,
        "query_filters": to_json({"trial_id": trial_id}),
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestEveryCriterionIsAlwaysACandidate:
    """No top_k/tier cap: every criterion of the resolved trial is always
    assessed — dropping one (e.g. an exclusion ground) is a patient-safety gap."""

    def test_ret_01_all_seven_xyz123_criteria_are_candidates(self):
        result = RetrieveNode()(_make_state(_XYZ_PATIENT_TEXT))
        candidates = from_json(result["matched_criteria"])
        assert len(candidates) == 7
        assert {c["id"] for c in candidates} == {
            "xyz123-inc-01",
            "xyz123-inc-02",
            "xyz123-inc-03",
            "xyz123-inc-04",
            "xyz123-exc-01",
            "xyz123-exc-02",
            "xyz123-exc-03",
        }

    def test_ret_02_deterministic_order_inclusion_before_exclusion_then_id_asc(self):
        candidates = from_json(RetrieveNode()(_make_state(_XYZ_PATIENT_TEXT))["matched_criteria"])
        assert [c["id"] for c in candidates] == [
            "xyz123-inc-01",
            "xyz123-inc-02",
            "xyz123-inc-03",
            "xyz123-inc-04",
            "xyz123-exc-01",
            "xyz123-exc-02",
            "xyz123-exc-03",
        ]

    def test_matched_criteria_is_json_string(self):
        # List-shaped State fields travel as JSON strings.
        result = RetrieveNode()(_make_state(_XYZ_PATIENT_TEXT))
        assert isinstance(result["matched_criteria"], str)


class TestExplicitAndImplicitScoring:
    def test_ret_03_explicit_term_hit_scores_one(self):
        candidates = from_json(RetrieveNode()(_make_state(_XYZ_PATIENT_TEXT))["matched_criteria"])
        inc_02 = next(c for c in candidates if c["id"] == "xyz123-inc-02")
        assert inc_02["score"] == 1.0
        assert inc_02["matched_term"] == "nsclc"

    def test_ret_04_implicit_hint_term_scores_half(self):
        # "chemo" is an implicit_hint_terms entry on xyz123-exc-01, and no
        # explicit_terms phrase is present in this text.
        result = RetrieveNode()(_make_state("patient has a history of chemo treatment"))
        exc_01 = next(c for c in from_json(result["matched_criteria"]) if c["id"] == "xyz123-exc-01")
        assert exc_01["score"] == 0.5
        assert exc_01["matched_term"] == "chemo"

    def test_ret_05_no_textual_evidence_scores_zero(self):
        candidates = from_json(RetrieveNode()(_make_state(_XYZ_PATIENT_TEXT))["matched_criteria"])
        exc_03 = next(c for c in candidates if c["id"] == "xyz123-exc-03")  # hypersensitivity
        assert exc_03["score"] == 0.0
        assert exc_03["matched_term"] is None


class TestNegationAwareness:
    """Safety-critical: clinical text routinely negates a criterion term."""

    def test_ret_06_negated_exclusion_terms_are_flagged_negated(self):
        candidates = from_json(RetrieveNode()(_make_state(_XYZ_PATIENT_TEXT))["matched_criteria"])
        exc_01 = next(c for c in candidates if c["id"] == "xyz123-exc-01")  # no prior chemotherapy
        exc_02 = next(c for c in candidates if c["id"] == "xyz123-exc-02")  # no known brain metastases
        assert exc_01["negated"] is True
        assert exc_02["negated"] is True

    def test_ret_06_positive_inclusion_terms_are_not_negated(self):
        candidates = from_json(RetrieveNode()(_make_state(_XYZ_PATIENT_TEXT))["matched_criteria"])
        for cid in ("xyz123-inc-01", "xyz123-inc-02", "xyz123-inc-03", "xyz123-inc-04"):
            entry = next(c for c in candidates if c["id"] == cid)
            assert entry["negated"] is False, f"{cid} should not be negated"

    def test_ret_07_negation_window_is_clause_scoped(self):
        """The negation cue from an EARLIER clause must not bleed across a
        comma boundary into a later, unrelated clause — this is the exact
        false-positive scenario documented in retrieve_node.py's own
        docstring: "no known brain metastases, adequate organ function..."
        must NOT read organ function as negated by the preceding "no"."""
        result = RetrieveNode()(_make_state("no known brain metastases, adequate organ function on recent labs"))
        candidates = from_json(result["matched_criteria"])
        organ = next(c for c in candidates if c["id"] == "xyz123-inc-04")
        brain = next(c for c in candidates if c["id"] == "xyz123-exc-02")
        assert organ["negated"] is False, "organ function must not inherit the prior clause's negation"
        assert brain["negated"] is True


class TestUnresolvedOrUnseededTrial:
    def test_ret_08_no_trial_id_yields_no_candidates_and_a_note(self):
        result = RetrieveNode()(_make_state("some patient criteria text", trial_id=None))
        assert from_json(result["matched_criteria"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("no trial_id resolved" in n for n in notes)

    def test_ret_09_trial_id_not_in_seeded_kb_yields_no_candidates_and_a_note(self):
        result = RetrieveNode()(_make_state("some patient criteria text", trial_id="ZZZ-999"))
        assert from_json(result["matched_criteria"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("matched no seeded protocol criteria" in n for n in notes)


class TestConfigAndKbRobustness:
    def test_missing_kb_file_degrades_gracefully_via_state_seeding(self):
        # Canon: config reaches the node via STATE seeding (retrieval_config),
        # never a 2nd execute() argument.
        state = _make_state(
            "some patient criteria text",
            retrieval_config=to_json({"kb_path": "config/kb/does_not_exist.json"}),
        )
        result = RetrieveNode()(state)
        assert from_json(result["matched_criteria"]) == []
        notes = from_json(result.get("intake_notes"), [])
        assert any("not readable" in n for n in notes)

    def test_intake_notes_append_never_clobber(self):
        state = _make_state("some patient criteria text", trial_id=None, intake_notes=to_json(["earlier note"]))
        result = RetrieveNode()(state)
        notes = from_json(result["intake_notes"])
        assert notes[0] == "earlier note"
        assert len(notes) == 2


class TestRetrieveAudit:
    def test_ret_10_domain_audit_payload(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.retrieve_node, "emit_trace_event", spy)
        RetrieveNode()(_make_state(_XYZ_PATIENT_TEXT))
        events = [call.args[0] for call in spy.call_args_list]
        assert "retrieve_complete" in events
        payload = spy.call_args_list[events.index("retrieve_complete")].args[1]
        assert payload["candidates"] == 7
        assert payload["has_trial_id"] is True
        assert payload["kb_entries"] > 0
