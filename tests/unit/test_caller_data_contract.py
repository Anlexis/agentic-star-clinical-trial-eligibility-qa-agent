# HCR-C2-002 — Unit Tests: caller-data input contract (input_context)
#
# InputValidateNode accepts structured caller data via input_context
# (bridged into inner state — see src/graph/context_bridge.py) and
# validates every field against the explicit contract, fail-CLOSED:
#
#   patient_profile -> must be a string (free text; length-capped downstream)
#   trial_id        -> must be a string matching the trial-identifier grammar
#                      (2-6 letters, hyphen, 2-4 digits) — an inert
#                      identifier shape, never free text
#
# Rejections name the FIELD and never echo the rejected VALUE (a rejected
# value round-tripping into error logs would defeat the PHI/identifier
# hygiene of this template). Absent caller data degrades to the string
# payload path — never a fabricated assessment.
#
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json

import pytest

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json


def _make_state(payload: str, input_context=None, **extra) -> dict:
    state = {
        "validated_input": payload,
        "input_context": input_context or {},
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestValidCallerData:
    def test_context_profile_and_trial_id_are_used(self):
        result = InputValidateNode()(
            _make_state(
                "run a pre-screen for the attached profile",
                input_context={
                    "patient_profile": "adult patient with stage ii nsclc, no prior chemotherapy",
                    "trial_id": "xyz-123",
                },
            )
        )
        assert result["patient_criteria_text"] == ("adult patient with stage ii nsclc, no prior chemotherapy")
        assert from_json(result["query_filters"])["trial_id"] == "XYZ-123"

    def test_context_values_win_over_the_json_envelope(self):
        envelope = json.dumps({"patient_profile": "envelope text", "trial_id": "abc-456"})
        result = InputValidateNode()(
            _make_state(
                envelope,
                input_context={"patient_profile": "context text wins", "trial_id": "XYZ-123"},
            )
        )
        assert result["patient_criteria_text"] == "context text wins"
        assert from_json(result["query_filters"])["trial_id"] == "XYZ-123"

    def test_absent_context_degrades_to_the_string_payload(self):
        result = InputValidateNode()(_make_state("screening under trial protocol abc-456 for this adult patient"))
        assert from_json(result["query_filters"])["trial_id"] == "ABC-456"

    def test_empty_string_trial_id_is_treated_as_absent(self):
        result = InputValidateNode()(
            _make_state(
                "adult patient with stage ii nsclc under xyz-123",
                input_context={"trial_id": "   "},
            )
        )
        # Falls through to free-text extraction rather than rejecting.
        assert from_json(result["query_filters"])["trial_id"] == "XYZ-123"

    def test_context_profile_gets_the_same_phi_identifier_strip(self):
        # The string payload is PHI-identifier-stripped by PreProcessNode;
        # the input_context channel bypasses that node, so the SAME surface
        # strip must run here - raw identifiers never reach inner state on
        # either channel.
        result = InputValidateNode()(
            _make_state(
                "run a pre-screen",
                input_context={
                    "patient_profile": (
                        "adult patient with stage ii nsclc, MRN-1234567, "
                        "ssn 123-45-6789, dob 1958-03-14, under xyz-123"
                    )
                },
            )
        )
        text = result["patient_criteria_text"]
        assert "MRN-1234567" not in text
        assert "123-45-6789" not in text
        assert "1958-03-14" not in text
        assert "[REDACTED]" in text
        assert "stage ii nsclc" in text  # clinical phrasing left intact

    def test_unknown_context_keys_are_ignored(self):
        result = InputValidateNode()(
            _make_state(
                "adult patient with stage ii nsclc under xyz-123",
                input_context={"unrelated_key": object()},
            )
        )
        assert result.get("status") != AgentStatus.ERROR.value
        assert result["patient_criteria_text"]


class TestRejectionMatrix:
    """Fail-closed rejection of every non-conforming caller value, per field.

    Numbers (including NaN/Infinity, which parse fine through raw JSON and
    float()) are NOT valid for either field — there is no numeric semantics
    in this contract, so every non-string is refused outright rather than
    coerced.
    """

    @pytest.mark.parametrize(
        "bad_profile",
        [123, 1.5, float("nan"), float("inf"), float("-inf"), True, ["txt"], {"a": 1}],
        ids=["int", "float", "raw-nan", "raw-inf", "raw-neginf", "bool", "list", "dict"],
    )
    def test_non_string_patient_profile_is_rejected(self, bad_profile):
        result = InputValidateNode()(_make_state("text", input_context={"patient_profile": bad_profile}))
        assert result.get("status") == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("input_context.patient_profile" in str(e) for e in result.get("error_log", []))
        assert "patient_criteria_text" not in result

    @pytest.mark.parametrize(
        "bad_trial_id",
        [42, 2.5, float("nan"), float("inf"), True, ["XYZ-123"], {"id": "XYZ-123"}],
        ids=["int", "float", "raw-nan", "raw-inf", "bool", "list", "dict"],
    )
    def test_non_string_trial_id_is_rejected(self, bad_trial_id):
        result = InputValidateNode()(_make_state("text", input_context={"trial_id": bad_trial_id}))
        assert result.get("status") == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("input_context.trial_id" in str(e) for e in result.get("error_log", []))

    @pytest.mark.parametrize(
        "malformed",
        [
            "XYZ_123",  # wrong separator
            "X-123",  # too few letters
            "TOOLONGX-123",  # too many letters
            "XYZ-1",  # too few digits
            "XYZ-12345",  # too many digits
            "XYZ-123 extra",  # trailing free text
            "NaN",  # numeric-looking string, not an identifier
            "Infinity",
            "-Infinity",
            "<script>x</script>",
        ],
    )
    def test_malformed_trial_id_string_is_rejected(self, malformed):
        result = InputValidateNode()(_make_state("text", input_context={"trial_id": malformed}))
        assert result.get("status") == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("input_context.trial_id" in str(e) for e in result.get("error_log", []))

    def test_malformed_envelope_trial_id_is_rejected_too(self):
        envelope = json.dumps({"patient_profile": "adult patient", "trial_id": "not a trial id"})
        result = InputValidateNode()(_make_state(envelope))
        assert result.get("status") == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("trial_id" in str(e) for e in result.get("error_log", []))

    def test_rejected_values_are_never_echoed(self):
        marker = "zqxv_marker_never_echoed_9917"
        for state in (
            _make_state("text", input_context={"trial_id": marker}),
            _make_state(json.dumps({"trial_id": marker})),
        ):
            result = InputValidateNode()(state)
            assert result.get("status") == AgentStatus.SUCCESS.value
            # Completes carrying the reason, so the caller can
            # correct the value and send the request again.
            assert result.get("error_code")
            assert marker not in json.dumps(result.get("error_log", []))

    def test_rejection_produces_no_partial_domain_output(self):
        result = InputValidateNode()(_make_state("text", input_context={"trial_id": "bad id"}))
        assert result.get("status") == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert "patient_criteria_text" not in result
        assert "query_filters" not in result
