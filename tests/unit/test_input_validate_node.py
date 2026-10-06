# HCR-C2-002 — Unit Tests: InputValidateNode (inner domain node 1)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller
# (inner Cat-2 domain node). Payloads are lowercase / PII-free so the PII mask
# leaves them untouched — `validated_input` is one of the framework's scanned
# fields (_PII_SCAN_FIELDS) exactly like `user_input`.
#
# Mirrors docs/03_test_spec.md §2.2 (VAL-01..VAL-09).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import json
from unittest.mock import MagicMock

from framework.schemas.trust_level import TrustLevel

import src.nodes.input_validate_node
from src.nodes.input_validate_node import InputValidateNode
from src.schemas.state import from_json


def _make_state(payload, **extra) -> dict:
    state = {
        "validated_input": payload,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPlainTextParsing:
    def test_val_01_plain_text_becomes_patient_criteria_text(self):
        text = "does the patient meet inclusion and exclusion criteria for the trial"
        result = InputValidateNode()(_make_state(text))
        assert result["patient_criteria_text"] == text
        filters = from_json(result["query_filters"])
        assert filters == {"trial_id": None}

    def test_val_02_whitespace_is_collapsed(self):
        result = InputValidateNode()(_make_state("  adult   patient\n with stage ii nsclc  "))
        assert result["patient_criteria_text"] == "adult patient with stage ii nsclc"

    def test_query_filters_is_json_string(self):
        # Structured State fields travel as JSON strings, never dicts.
        result = InputValidateNode()(_make_state("adult patient screening"))
        assert isinstance(result["query_filters"], str)
        assert isinstance(from_json(result["query_filters"]), dict)


class TestTrialIdExtraction:
    def test_val_03_trial_id_extracted_from_free_text(self):
        result = InputValidateNode()(_make_state("eligibility screen against trial protocol xyz-123 for this patient"))
        assert from_json(result["query_filters"])["trial_id"] == "XYZ-123"

    def test_val_03_trial_id_extraction_is_case_insensitive_then_uppercased(self):
        result = InputValidateNode()(_make_state("screening under abc-456 for diabetes"))
        assert from_json(result["query_filters"])["trial_id"] == "ABC-456"

    def test_unresolved_trial_id_produces_a_note(self):
        result = InputValidateNode()(_make_state("no protocol number mentioned anywhere here"))
        assert from_json(result["query_filters"]) == {"trial_id": None}
        notes = from_json(result.get("intake_notes"), [])
        assert any("no trial_id resolved" in n for n in notes)


class TestJsonEnvelopeParsing:
    def test_val_04_envelope_patient_profile_and_trial_id(self):
        payload = json.dumps({"patient_profile": "adult with confirmed diagnosis", "trial_id": "xyz-123"})
        result = InputValidateNode()(_make_state(payload))
        assert result["patient_criteria_text"] == "adult with confirmed diagnosis"
        assert from_json(result["query_filters"])["trial_id"] == "XYZ-123"

    def test_patient_criteria_alias_accepted(self):
        payload = json.dumps({"patient_criteria": "stage ii diagnosis, no prior chemo"})
        result = InputValidateNode()(_make_state(payload))
        assert result["patient_criteria_text"] == "stage ii diagnosis, no prior chemo"

    def test_query_alias_accepted(self):
        payload = json.dumps({"query": "adult patient eligibility pre-screen"})
        result = InputValidateNode()(_make_state(payload))
        assert result["patient_criteria_text"] == "adult patient eligibility pre-screen"

    def test_val_05_malformed_json_falls_back_to_plain_text(self):
        payload = "{ this is not valid json but starts like it"
        result = InputValidateNode()(_make_state(payload))
        assert result["patient_criteria_text"] == payload
        notes = from_json(result.get("intake_notes"), [])
        assert any("did not parse" in n for n in notes)

    def test_non_dict_json_falls_back_to_plain_text(self):
        payload = json.dumps(["not", "an", "object"])
        result = InputValidateNode()(_make_state(payload))
        assert result["patient_criteria_text"] == payload


class TestSizeAndEmptyGuards:
    def test_val_06_oversize_text_is_truncated(self):
        payload = "adult patient " * 400  # ~5600 chars after collapse
        result = InputValidateNode()(_make_state(payload))
        assert len(result["patient_criteria_text"]) == 4000
        notes = from_json(result.get("intake_notes"), [])
        assert any("truncated" in n for n in notes)

    def test_val_07_empty_request_yields_notes_not_error(self):
        result = InputValidateNode()(_make_state(""))
        assert result["patient_criteria_text"] == ""
        notes = from_json(result.get("intake_notes"), [])
        assert any("empty request" in n for n in notes)
        assert any("no trial_id resolved" in n for n in notes)


class TestInputValidateAudit:
    def test_val_08_domain_audit_payload(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.input_validate_node, "emit_trace_event", spy)
        text = "trial xyz-123 eligibility pre-screen"
        InputValidateNode()(_make_state(text))
        events = [call.args[0] for call in spy.call_args_list]
        assert "input_validate_complete" in events
        payload = spy.call_args_list[events.index("input_validate_complete")].args[1]
        assert payload["has_trial_id"] is True
        assert payload["text_chars"] == len(text)
