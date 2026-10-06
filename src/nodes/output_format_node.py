"""AgentCore Platform v1.0"""

# HCR-C2-002 - OutputFormatNode
# Domain node 5 (terminal): compose the final pre-screen body - the grouped
# per-criterion narrative plus the Sources list.
#
# The mandatory PI-review disclaimer is DELIBERATELY NOT added here. It is
# appended downstream by the OUTER PostProcessNode (output security gate)
# instead, so the disclaimer is genuinely non-suppressible - a domain node
# forgetting to add it can never happen, because the disclaimer lives in the
# one gate every response passes through unconditionally, and PostProcessNode
# additionally VERIFIES the disclaimer is present in the final text before
# returning SUCCESS (fail-closed if a future change ever drops it). See
# docs/02_design.md "Safety Boundary".
#
# Wired by the inner graph (DomainWorkflowGraph). get_output() of the inner
# graph surfaces formatted_answer + status (+ structured fields) to the
# outer merge_output(). Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.schemas.state import from_json


class OutputFormatNode(FunctionNode):
    """Compose the final pre-screen body: grouped statuses + sources.

    Input state keys:
        grounded_answer: pre-screen narrative with [n] citation markers (or
                          the abstention message)
        citations:        JSON list [{ref, id, trial_id, category, source}]

    Output state keys (partial dict):
        formatted_answer: final rendered pre-screen body string
        status:           AgentStatus.SUCCESS.value (plain string —
                          never write the bare enum to State)
    """

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.ANONYMOUS

    def execute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        # The request was already found unacceptable upstream: this run
        # completes without an answer, so there is nothing for this step to
        # do. Returning the marker keeps it on the node's own result dict,
        # which is what the output gate inspects.
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}

        emit_progress("Formatting the response...")
        grounded_answer = state.get("grounded_answer") or (
            "No eligibility pre-screen result is available for this request."
        )
        citations: List[Dict[str, Any]] = from_json(state.get("citations"), []) or []

        lines: List[str] = []
        lines.append("# Clinical Trial Eligibility Pre-Screen")
        lines.append("")
        lines.append(grounded_answer)
        lines.append("")
        lines.append("## Sources")
        if citations:
            for citation in citations:
                if not isinstance(citation, dict):
                    continue
                ref = citation.get("ref", "?")
                trial_id = str(citation.get("trial_id", "")).strip()
                category = str(citation.get("category", "")).strip()
                source = str(citation.get("source", "")).strip()
                label = " / ".join(p for p in (trial_id, category) if p) or "criterion"
                suffix = f" — {source}" if source else ""
                lines.append(f"- [{ref}] {label}{suffix}")
        else:
            lines.append("- none (no criterion reached a citable appears-satisfied/" "flagged/uncertain assessment)")

        formatted_answer = "\n".join(lines)

        # Domain audit: final pre-screen body composed.
        emit_trace_event(
            "output_format_complete",
            {
                "answer_chars": len(formatted_answer),
                "citation_count": len(citations),
            },
            state,
        )

        return {
            "formatted_answer": formatted_answer,
            "status": AgentStatus.SUCCESS.value,
        }
