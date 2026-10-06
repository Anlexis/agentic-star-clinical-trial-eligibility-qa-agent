"""AgentCore Platform v1.0"""

# Node contract:
#  - Extend FunctionNode; implement execute(state) -> dict
#  - Return ONLY the fields this node changes (never full state)
#  - Return AgentStatus.<X>.value strings for status assignments
#  - Read input_context via state.get("input_context", {}) — read-only
#  - Never import from mediator/, api/, or other agents
#
# Trust gate: this is an outer backbone gate slot — the manifest declares
# required_trust_level: "VERIFIED_EXTERNAL" (config/agent.yaml), so this node
# gates external callers before the inner domain workflow runs.
#
# PHI screen (real behaviour, not a doc claim): a surface screen redacts
# PHI-shaped direct identifiers (MRN-style IDs, SSNs, e-mails, phone
# numbers, calendar-date-shaped tokens) from the free-text payload BEFORE
# validated_input is written, so raw identifiers never reach the inner
# domain workflow or the checkpoint DB. The strip is exercised inline in
# execute() and pinned by dedicated tests, not only described in prose.

import re
from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT

# PHI-shaped direct-identifier patterns redacted before validated_input is
# written. Downstream domain nodes only ever operate on the normalised
# patient-criteria text and KB citation summaries, never raw identifiers.
# Deliberately limited to HIGH-CONFIDENCE structural patterns (never
# freeform name-guessing, which is out of scope and error-prone for a
# keyword-based v1 — see docs/02_design.md "Safety Boundary").
_PHI_PATTERNS: List[re.Pattern[str]] = [
    # Medical-record-number-style token (mirrors the output-gate pattern
    # in post_process_node.py — kept in sync intentionally).
    re.compile(r"\bMRN[-:\s]?\d{6,10}\b", re.IGNORECASE),
    # SSN-shaped sequence.
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # E-mail addresses.
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    # Phone numbers (loose US-style).
    re.compile(r"\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    # Calendar-date-shaped tokens (a specific DOB-style date is a stronger
    # direct-identifier signal than a bare age like "68yo", which is left
    # untouched — the domain nodes need age/criteria phrasing intact).
    re.compile(r"\b\d{1,2}/\d{1,2}/\d{2,4}\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
]
_PHI_REPLACEMENT = "[REDACTED]"


def _surface_strip_identifiers(text: str) -> str:
    """Redact obvious direct-identifier / PHI-shaped tokens from free text."""
    for pattern in _PHI_PATTERNS:
        text = pattern.sub(_PHI_REPLACEMENT, text)
    return text


class PreProcessNode(FunctionNode):
    """Input validation + PHI identifier screen before main processing."""

    # Explicit by design, not inherited implicitly. Outer backbone gate slot
    # — matches the manifest's declared required_trust_level
    # (config/agent.yaml).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        emit_progress("Checking the request...")
        user_input = state.get("user_input", "")
        input_context = state.get("input_context", {})  # read-only

        if not user_input or not isinstance(user_input, str) or not user_input.strip():
            # Nothing to work with, but the caller can simply send a question
            # and try again - so the run completes carrying the reason rather
            # than terminating. Terminating would end the caller's turn and
            # surface only an exception type, leaving the reason reachable
            # solely from the audit trail.
            emit_progress(EMPTY_INPUT)
            emit_trace_event("pre_process_declined", {"reason": "EMPTY_INPUT"}, state)
            return {
                "status": AgentStatus.SUCCESS.value,
                "error_code": "EMPTY_INPUT",
                "error_log": ["PreProcessNode: user_input is empty or missing"],
            }

        validated_input = _surface_strip_identifiers(user_input.strip())

        # Domain audit: a de-identified eligibility query was accepted
        # and surface-redacted. NEVER include the input text itself here.
        emit_trace_event(
            "pre_process_complete",
            {"input_chars": len(validated_input)},
            state,
        )

        return {
            "validated_input": validated_input,
            "enriched_context": {
                "source": "ClinicalTrialEligibilityQAAgent",
                "channel": input_context.get("channel", "unknown"),
            },
            "status": AgentStatus.SUCCESS.value,
        }
