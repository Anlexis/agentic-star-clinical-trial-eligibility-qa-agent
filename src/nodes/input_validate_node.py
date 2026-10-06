"""AgentCore Platform v1.0"""

# HCR-C2-002 - InputValidateNode
# Domain node 1: parse and normalise the incoming patient-criteria request.
#
# Caller data arrives on TWO channels, both validated here field-by-field:
#
#   input_context (structured invocation parameters, bridged into inner
#   state - see src/graph/context_bridge.py):
#     patient_profile -> de-identified patient-criteria text (string)
#     trial_id        -> explicit trial filter (string, trial-identifier
#                        grammar enforced)
#
#   the string payload (validated_input, PHI-identifier-stripped by
#   PreProcessNode before it reaches the inner graph):
#     plain text            -> the whole string is the patient-criteria text;
#                               a trial_id is additionally extracted via regex
#                               if the text mentions one (e.g. "trial protocol
#                               XYZ-123")
#     {"patient_profile": "...",
#      "trial_id": "..."}   -> patient-criteria text + explicit trial filter
#
# Validation is fail-CLOSED: a supplied field that is not a string, or a
# trial_id that does not match the trial-identifier grammar, returns
# status=ERROR naming the FIELD - never echoing the rejected value. Absent
# caller data degrades to the plain-text path (and, further downstream, to
# the abstention baseline) rather than fabricating an assessment.
#
# Precedence: input_context values (already structured) win over the JSON
# envelope, which wins over free-text extraction.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import INPUT_REJECTED
from src.nodes.pre_process_node import _surface_strip_identifiers
from src.schemas.state import to_json

# Hard cap on the normalised patient-criteria text length (defence-in-depth
# on input size; the server adapter additionally caps the whole serialized
# input_context).
_MAX_TEXT_CHARS = 4000

# Trial-ID grammar: 2-6 letters, a hyphen, 2-4 digits (e.g. "XYZ-123").
# Matched case-insensitively then upper-cased, so "xyz-123" in caller free
# text still resolves. The FULL-match form locks every caller-supplied
# trial_id to this inert identifier shape before it can reach state, notes,
# or KB comparisons.
_TRIAL_ID_RE = re.compile(r"\b([A-Za-z]{2,6}-\d{2,4})\b")
_TRIAL_ID_FULL_RE = re.compile(r"^[A-Za-z]{2,6}-\d{2,4}$")

_WHITESPACE_RE = re.compile(r"\s+")

_TRIAL_ID_FORMAT_MSG = "must match the trial-identifier format: 2-6 letters, a hyphen, " "2-4 digits (e.g. XYZ-123)"


def _extract_trial_id(text: str) -> Optional[str]:
    """Best-effort trial_id extraction from free text (e.g. "...trial XYZ-123...")."""
    m = _TRIAL_ID_RE.search(text)
    if not m:
        return None
    return m.group(1).upper()


def _decline(field: str, reason: str) -> Dict[str, Any]:
    """Decline a caller-supplied value and COMPLETE the run.

    Every call site is a value the caller can correct: a field outside its
    documented contract. The processing path still fails closed - nothing
    downstream runs on an unusable value. What changed is the reporting:
    terminating would end the caller's turn and surface only an exception type,
    leaving the field name reachable solely from the audit trail. Completing
    with the reason lets the caller correct the value and send the request
    again on the same conversation.

    Names the offending FIELD only (no value echo).
    """
    emit_progress(INPUT_REJECTED)
    return {
        "status": AgentStatus.SUCCESS.value,
        "error_code": "INVALID_REQUEST",
        "error_log": [f"InputValidateNode: {field} {reason}"],
    }


def _validate_trial_id(raw: Any, field: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Validate a caller-supplied trial identifier against the grammar.

    Returns (normalised_trial_id, error_dict). None/empty input is not an
    error - it means "not supplied on this channel".
    """
    if raw is None:
        return None, None
    if not isinstance(raw, str):
        return None, _decline(field, "must be a string")
    candidate = raw.strip()
    if not candidate:
        return None, None
    if not _TRIAL_ID_FULL_RE.match(candidate):
        return None, _decline(field, _TRIAL_ID_FORMAT_MSG)
    return candidate.upper(), None


class InputValidateNode(FunctionNode):
    """Parse and validate the caller request into normalised patient-criteria text.

    Input state keys:
        input_context:                caller invocation parameters (read-only)
        validated_input | user_input: PHI-identifier-stripped request payload

    Output state keys (partial dict):
        patient_criteria_text: normalised patient-criteria text
        query_filters:         JSON dict {"trial_id": str|None}
        intake_notes:          (when anomalies were seen) JSON list[str]

    On a validation rejection (fail-closed): status=ERROR + an error_log
    entry naming the field - the rest of the pipeline is skipped by the
    framework and no assessment is produced.
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        emit_progress("Checking the request...")
        raw = state.get("validated_input") or state.get("user_input", "")
        input_context = state.get("input_context", {}) or {}  # read-only
        notes: List[str] = []

        # ── Structured caller data (input_context) - validated field-by-field ──
        ctx_profile = input_context.get("patient_profile")
        if ctx_profile is not None and not isinstance(ctx_profile, str):
            return _decline("input_context.patient_profile", "must be a string")
        trial_id, err = _validate_trial_id(input_context.get("trial_id"), "input_context.trial_id")
        if err:
            return err

        text = ""
        if isinstance(ctx_profile, str) and ctx_profile.strip():
            # input_context does not pass through PreProcessNode (which strips
            # PHI-shaped identifiers from the string payload) or the
            # framework's PII mask (scoped to user_input/validated_input), so
            # the SAME surface strip is applied here - raw identifiers never
            # reach inner state or the checkpoint DB on either channel.
            text = _surface_strip_identifiers(ctx_profile.strip())

        # ── String payload (plain text or JSON envelope) ──────────────────────
        if isinstance(raw, str) and raw.strip():
            payload: Any = None
            stripped = raw.strip()
            if stripped.startswith("{"):
                try:
                    payload = json.loads(stripped)
                except (json.JSONDecodeError, ValueError):
                    notes.append(
                        "InputValidateNode: JSON-looking input did not parse - "
                        "treated as plain patient-criteria text."
                    )
            if isinstance(payload, dict):
                if not text:
                    text = str(
                        payload.get("patient_profile") or payload.get("patient_criteria") or payload.get("query") or ""
                    )
                if trial_id is None:
                    trial_id, err = _validate_trial_id(payload.get("trial_id"), "trial_id")
                    if err:
                        return err
            elif not text:
                text = stripped
        elif not text:
            notes.append("InputValidateNode: empty request - no patient criteria to screen.")

        # Normalise whitespace and cap length.
        text = _WHITESPACE_RE.sub(" ", text).strip()
        if len(text) > _MAX_TEXT_CHARS:
            text = text[:_MAX_TEXT_CHARS]
            notes.append(f"InputValidateNode: patient-criteria text truncated to {_MAX_TEXT_CHARS} chars.")

        # Always also attempt a trial-id extraction from the free text - this
        # covers the plain-text path AND backfills an envelope/context that
        # named the trial only inline in patient_profile.
        if trial_id is None and text:
            trial_id = _extract_trial_id(text)
        if trial_id is None:
            notes.append("InputValidateNode: no trial_id resolved from the request.")

        filters: Dict[str, Any] = {"trial_id": trial_id}

        # Domain audit: request parsed and normalised. NEVER include the
        # patient-criteria text itself or the trial_id in the audit payload -
        # length/booleans only.
        emit_trace_event(
            "input_validate_complete",
            {
                "text_chars": len(text),
                "has_trial_id": trial_id is not None,
            },
            state,
        )

        out: Dict[str, Any] = {
            "patient_criteria_text": text,
            "query_filters": to_json(filters),
        }
        if notes:
            out["intake_notes"] = to_json(notes)
        return out
