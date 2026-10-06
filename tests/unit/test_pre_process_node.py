# HCR-C2-002 — Unit Tests: PreProcessNode (outer pre_process slot)
#
# Invocation canon: every test invokes the node via node(state) —
# BaseNode.__call__ -> trust gate -> PII input mask -> execute()
# -> output gate — never a bare node.execute(state). PreProcessNode
# requires VERIFIED_EXTERNAL, so its behavioural tests build the state at that
# level (the ANONYMOUS rejection lives in test_trust_gate.py).
#
# ⚠️ THE CRITICAL TEST IN THIS FILE (TestPHIIdentifierScreen): the PHI strip
# must be REAL, not a doc claim (docs/02_design.md "Safety Boundary" #3).
# This file proves exactly that:
# TestPHIIdentifierScreenDirect exercises the module-level
# _surface_strip_identifiers() helper directly against every one of the 5
# documented pattern types, and TestPHIIdentifierScreenEndToEnd drives the
# SAME behaviour through node(state) (full __call__ pipeline) for the two
# pattern types the framework's OWN input gate does not already cover, so the
# node's strip is proven WIRED, not just present as an orphaned helper.
#
# PII layering note: the FRAMEWORK input gate (framework.nodes.function_node,
# shared.security.pii_detector.detect_pii) ALSO masks user_input /
# validated_input to [MASKED] before execute() runs, for its own pattern set:
# e-mail, phone (JP/US), SSN (US, `\d{3}-\d{2}-\d{4}`), My Number (JP),
# credit-card digit runs, and 2+-word Title-Case name bigrams. This node's
# OWN _surface_strip_identifiers() additionally (and independently) catches
# MRN-style IDs and calendar-date-shaped tokens (`MM/DD/YYYY`, `YYYY-MM-DD`) —
# NEITHER of which the framework gate's patterns match — replacing them with
# [REDACTED]. For the SSN / e-mail / phone pattern types the two gates
# overlap (the framework gate runs FIRST and already masks the token to
# [MASKED] before this node's regex ever sees the raw digits), which is why
# the end-to-end proof below uses the framework-independent pattern types
# (MRN, dates) — the direct-function tests below independently pin all 5.
#
# Mirrors docs/03_test_spec.md §2.1 (PRE-01..PRE-09).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.pre_process_node
from src.nodes.pre_process_node import PreProcessNode, _surface_strip_identifiers

# Lowercase phrasing on purpose: PII-free (no Title-Case bigram, no @, no
# digit run), so the framework mask leaves the payload untouched end-to-end.
_VALID_QUERY = (
    "does the patient meet the inclusion and exclusion criteria for the "
    "trial based on diagnosis and prior treatment history"
)


def _make_state(user_input=_VALID_QUERY, **extra) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestPreProcessSuccess:
    def test_pre_01_valid_query_accepted(self):
        result = PreProcessNode()(_make_state())
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum
        # (AgentStatus subclasses str, so a bare isinstance would not catch it).
        assert isinstance(result["status"], str)
        assert not isinstance(result["status"], AgentStatus)
        assert result["validated_input"] == _VALID_QUERY

    def test_enriched_context_carries_channel(self):
        result = PreProcessNode()(_make_state(input_context={"channel": "ehr-portal"}))
        assert result["enriched_context"]["channel"] == "ehr-portal"
        assert result["enriched_context"]["source"] == "ClinicalTrialEligibilityQAAgent"

    def test_missing_channel_defaults_to_unknown(self):
        result = PreProcessNode()(_make_state())
        assert result["enriched_context"]["channel"] == "unknown"


class TestPreProcessRejection:
    def test_pre_02_empty_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input=""))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert result["error_log"]
        # No validated_input is produced on the reject path.
        assert "validated_input" not in result

    def test_whitespace_only_is_error(self):
        result = PreProcessNode()(_make_state(user_input="   \n\t "))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")

    def test_pre_03_missing_user_input_is_error(self):
        state = _make_state()
        del state["user_input"]
        result = PreProcessNode()(state)
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")

    def test_non_string_input_is_error(self):
        result = PreProcessNode()(_make_state(user_input={"malicious": "dict"}))
        assert result["status"] == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")


class TestPHIIdentifierScreenDirect:
    """PRE-04..PRE-08: _surface_strip_identifiers() is exercised DIRECTLY for
    every one of the 5 documented pattern types — this is the provably-real
    proof the Safety Boundary demands: each pattern's regex genuinely
    matches and redacts, independent of what any upstream gate also does."""

    def test_pre_04_mrn_style_id_is_redacted(self):
        out = _surface_strip_identifiers("patient record MRN-1234567 flagged for review")
        assert "MRN-1234567" not in out
        assert "[REDACTED]" in out

    def test_mrn_variant_without_hyphen_is_redacted(self):
        out = _surface_strip_identifiers("see MRN 987654321 for prior labs")
        assert "987654321" not in out
        assert "[REDACTED]" in out

    def test_pre_05_ssn_shaped_sequence_is_redacted(self):
        out = _surface_strip_identifiers("identifier on file: 123-45-6789")
        assert "123-45-6789" not in out
        assert "[REDACTED]" in out

    def test_pre_06_email_is_redacted(self):
        out = _surface_strip_identifiers("contact the coordinator at trial.coord@example.org")
        assert "trial.coord@example.org" not in out
        assert "[REDACTED]" in out

    def test_pre_07_phone_number_is_redacted(self):
        out = _surface_strip_identifiers("callback number (555) 123-4567 on file")
        assert "123-4567" not in out
        assert "[REDACTED]" in out

    def test_pre_08_slash_date_is_redacted(self):
        out = _surface_strip_identifiers("date of birth 05/12/1958 recorded at intake")
        assert "05/12/1958" not in out
        assert "[REDACTED]" in out

    def test_pre_08_iso_date_is_redacted(self):
        out = _surface_strip_identifiers("consented on 1958-05-12 per the chart note")
        assert "1958-05-12" not in out
        assert "[REDACTED]" in out

    def test_bare_age_is_left_untouched(self):
        """A bare age like '68yo' / '68-year-old' is NOT a direct identifier —
        the domain nodes need age phrasing intact for criterion matching."""
        out = _surface_strip_identifiers("68-year-old patient, 68yo, presents for screening")
        assert "68-year-old" in out
        assert "68yo" in out
        assert "[REDACTED]" not in out

    def test_multiple_identifier_types_all_redacted_in_one_pass(self):
        raw = "MRN-1234567, SSN 123-45-6789, contact patient@example.org or " "(555) 123-4567, DOB 05/12/1958"
        out = _surface_strip_identifiers(raw)
        for leaked in (
            "MRN-1234567",
            "123-45-6789",
            "patient@example.org",
            "123-4567",
            "05/12/1958",
        ):
            assert leaked not in out
        assert out.count("[REDACTED]") == 5


class TestPHIIdentifierScreenEndToEnd:
    """PRE-09: the strip is WIRED into the real node(state) pipeline, not just
    a standalone helper — proven for the two pattern types the framework's OWN
    input gate does not already mask (MRN, calendar dates), so what we observe
    here is genuinely THIS node's behaviour."""

    def test_pre_09_mrn_stripped_via_full_call_pipeline(self):
        raw = "eligibility review for record MRN-2004587, please advise"
        result = PreProcessNode()(_make_state(user_input=raw))
        assert result["status"] == AgentStatus.SUCCESS.value
        vi = result["validated_input"]
        assert "MRN-2004587" not in vi
        assert "[REDACTED]" in vi

    def test_pre_09_date_stripped_via_full_call_pipeline(self):
        raw = "consent obtained 2024-01-15 ahead of the pre-screen"
        result = PreProcessNode()(_make_state(user_input=raw))
        assert result["status"] == AgentStatus.SUCCESS.value
        vi = result["validated_input"]
        assert "2024-01-15" not in vi
        assert "[REDACTED]" in vi

    def test_pre_09_email_masked_by_framework_gate_before_execute(self):
        """The framework input gate masks e-mail before execute() sees it — the
        raw address must never survive, whichever layer is responsible."""
        raw = "escalate to the study coordinator at coordinator@example.org today"
        result = PreProcessNode()(_make_state(user_input=raw))
        assert result["status"] == AgentStatus.SUCCESS.value
        vi = result["validated_input"]
        assert "coordinator@example.org" not in vi
        assert "[MASKED]" in vi


class TestPreProcessAudit:
    def test_pre_10_domain_audit_payload(self, monkeypatch):
        """Audit: the accepted request emits pre_process_complete; the assertion
        targets call.args[1] — the event payload — never the whole call repr."""
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.pre_process_node, "emit_trace_event", spy)
        PreProcessNode()(_make_state())
        events = [call.args[0] for call in spy.call_args_list]
        assert "pre_process_complete" in events
        payload = spy.call_args_list[events.index("pre_process_complete")].args[1]
        assert payload["input_chars"] == len(_VALID_QUERY)
