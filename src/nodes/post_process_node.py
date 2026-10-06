"""AgentCore Platform v1.0"""

# Output gate: this node calls the MODULE-LEVEL `_security_gate_output()`
# scan from execute() itself. `content` may be a plain string OR a nested
# structure (dict/list/tuple) — the scan walks every level and checks every
# string leaf, so a violation buried inside a nested field (e.g. a
# structured citation/checklist dict) is caught exactly like a top-level
# string, and the structured output path never needs a gate rewrite.
#
# Patterns cover generic credential leakage AND basic PHI-shaped identifiers
# — this is a life-critical/PHI-adjacent template that calls for enhanced
# PHI redaction. This is a deterministic-pipeline floor; a KB-aware domain
# layer may add stricter PHI/disclaimer checks on top.
#
# No _extra_security_gate_input/_output instance methods are defined on this
# node (the framework auto-wraps such hooks into the graph chain).
#
# Trust gate: this is an outer backbone gate slot — the manifest declares
# required_trust_level: "VERIFIED_EXTERNAL" (config/agent.yaml).

import logging
import re
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.services.failure_message import EMPTY_INPUT, INPUT_REJECTED, INVALID_VALUE, OUTPUT_BLOCKED, TOO_LONG

logger = logging.getLogger(__name__)

# Disallowed output-content patterns. Each tuple: (name, compiled regex) —
# order matters (most specific first).
_DISALLOWED_PATTERNS: List[Tuple[str, re.Pattern[str]]] = [
    # API key patterns: sk-..., pk-..., ak-...
    ("api_key", re.compile(r"\b(?:sk|pk|ak)-[A-Za-z0-9]{16,}", re.IGNORECASE)),
    # JWT: three base64url segments separated by dots
    ("jwt", re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}")),
    # Bearer token in Authorization-like context
    ("bearer_token", re.compile(r"Bearer\s+[A-Za-z0-9._~+/]{20,}", re.IGNORECASE)),
    # Credential assignment patterns
    (
        "credential_assignment",
        re.compile(
            r"\b(?:password|passwd|secret|api_key|token|access_key|private_key)\s*[:=]\s*\S{8,}",
            re.IGNORECASE,
        ),
    ),
    # PHI-shaped identifiers (enhanced redaction for a PHI-adjacent
    # domain) — a plausible medical-record-number-style token, or an
    # SSN-shaped sequence. Deterministic heuristic; the read-only KB carries
    # no real PHI, so this is a defensive floor against an identifier echoed
    # back from caller-supplied input.
    ("mrn_like_id", re.compile(r"\bMRN[-:\s]?\d{6,10}\b", re.IGNORECASE)),
    ("ssn_like", re.compile(r"\b\d{3}-\d{2}-\d{4}\b")),
]

_SANITISED_STUB = (
    "[OUTPUT BLOCKED by the output security gate - disallowed content detected. "
    "Review the generated output and retry without credential-like or "
    "PHI-identifier-like strings.]"
)

# Mandatory, non-suppressible PI-review disclaimer (docs/02_design.md
# "Safety Boundary"):
# attached to EVERY successful output HERE — the one gate every response
# passes through unconditionally — never by a domain node that a future
# change could accidentally skip. This template is decision-support only;
# it must never read as an enrollment decision or a definitive eligibility
# verdict.
_PI_REVIEW_DISCLAIMER = (
    "DECISION SUPPORT ONLY — this is an automated eligibility "
    "PRE-SCREEN, not an enrollment decision. The Principal Investigator "
    "(PI) or study team must independently review the full patient record "
    "and confirm final trial eligibility before any enrollment action."
)


def _security_gate_output(content: Any) -> Optional[str]:
    """Run the output content gate, RECURSIVELY.

    `content` may be a str, or a dict/list/tuple that nests strings at any
    depth. Every string leaf is scanned — a violation nested inside a
    dict/list is caught exactly like a top-level string.

    Returns the name of the first matched violation, or None if clean.
    """
    if isinstance(content, str):
        for name, pattern in _DISALLOWED_PATTERNS:
            if pattern.search(content):
                return name
        return None
    if isinstance(content, dict):
        for value in content.values():
            violation = _security_gate_output(value)
            if violation:
                return violation
        return None
    if isinstance(content, (list, tuple)):
        for item in content:
            violation = _security_gate_output(item)
            if violation:
                return violation
        return None
    # Non-string scalars (int/float/bool/None/...) carry no disallowed text.
    return None


# Caller-facing wording for a run that completed without an answer. The marker
# is an internal reason code; this maps it to the sentence the caller sees.
# Static sentences only - no request value is ever substituted, so nothing the
# caller sent can be reflected back through this path.
_DEGRADED_MESSAGES = {
    "EMPTY_INPUT": EMPTY_INPUT,
    "QUESTION_TOO_LONG": TOO_LONG,
    "INVALID_REQUEST": INVALID_VALUE,
}


class PostProcessNode(FunctionNode):
    """Format and finalize the output, behind the output security gate."""

    # Explicit by design, not inherited implicitly. Outer backbone gate slot
    # — matches the manifest's declared required_trust_level
    # (config/agent.yaml).
    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        emit_progress("Finalising the response...")

        # The run completed without an answer because the request could not be
        # accepted as written. Report the reason as the response: the caller
        # needs to know what to change, and an empty body would leave them with
        # nothing. Status stays SUCCESS - the run did what it could with the
        # request it was given, and the caller can correct it and send again on
        # the same conversation.
        marker = state.get("error_code")
        if marker:
            message = _DEGRADED_MESSAGES.get(marker, INPUT_REJECTED)
            emit_trace_event("post_process_degraded", {"reason": marker}, state)
            return {
                "formatted_output": message,
                "result": message,
                "status": AgentStatus.SUCCESS.value,
                "error_code": marker,
            }
        result = state.get("result", "")

        if not result or (isinstance(result, str) and not result.strip()):
            # No result to gate — forward as-is (non-fatal).
            return {
                "formatted_output": result,
                "status": AgentStatus.SUCCESS.value,
            }

        violation = _security_gate_output(result)
        if violation:
            logger.error(
                "PostProcessNode: OUTPUT BLOCKED - violation type: %s",
                violation,
            )
            emit_progress(OUTPUT_BLOCKED)
            return {
                "formatted_output": _SANITISED_STUB,
                "result": _SANITISED_STUB,
                "status": AgentStatus.ERROR.value,
                "error_log": [f"PostProcessNode: output blocked - " f"disallowed content detected ({violation})"],
            }

        # Attach the mandatory PI-review disclaimer, then VERIFY it is
        # actually present before SUCCESS is returned — fail-closed: if a
        # future edit to this method ever drops the append (e.g. a
        # truncation added later), the gate blocks the output instead of
        # silently shipping a disclaimer-less clinical assessment. Only a
        # plain string result gets the text-appended disclaimer; the
        # structured fields (criterion_assessment / citations) carry their
        # own advisory_notice constant via ClinicalTrialEligibilityQAAgent.get_output()
        # in src/graph/graph.py.
        output = result
        if isinstance(output, str):
            output = f"{output}\n\n---\n*{_PI_REVIEW_DISCLAIMER}*"
            if _PI_REVIEW_DISCLAIMER not in output:
                logger.error("PostProcessNode: mandatory PI-review disclaimer " "verification failed post-attach")
                emit_progress(OUTPUT_BLOCKED)
                return {
                    "formatted_output": _SANITISED_STUB,
                    "result": _SANITISED_STUB,
                    "status": AgentStatus.ERROR.value,
                    "error_log": [
                        "PostProcessNode: output blocked - mandatory " "PI-review disclaimer verification failed"
                    ],
                }

        # Clean — domain audit: a finalized output was emitted.
        emit_trace_event(
            "post_process_complete",
            {"output_chars": len(str(output))},
            state,
        )

        return {
            "formatted_output": output,
            "status": AgentStatus.SUCCESS.value,
        }
