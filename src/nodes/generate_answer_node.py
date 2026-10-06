"""AgentCore Platform v1.0"""

# HCR-C2-002 - GenerateAnswerNode
# Domain node 4: assemble the cited, criterion-grouped pre-screen narrative
# from the final per-criterion assessment.
#
# v1 is DETERMINISTIC (no live LLM call): the body is rule-assembled from
# criterion_assessment only - a grouped, per-criterion listing with numbered
# citation markers. Nothing outside criterion_assessment reaches the answer
# body, so the output is grounded by construction. The LLM synthesis upgrade
# seam is documented in docs/02_design.md ("v1 Implementation Note - LLM
# synthesis") and config/prompts/eligibility_synthesis_prompt.md.
#
# SAFETY BOUNDARY: this node NEVER emits an eligibility verdict, an
# enrolment decision, or a "meets"/"fails" determination - only factual,
# per-criterion advisory statements, each pointing back to a human
# (Principal Investigator) review requirement. On overall_abstain, it emits
# ONLY an abstention message (no per-criterion synthesis at all, no
# citations) and routes the caller to PI/human review - see
# RerankFilterNode for the abstain gate.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List, Optional

from framework.nodes.function_node import FunctionNode
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.schemas.state import from_json, to_json
from framework.schemas.agent_status import AgentStatus

_ABSTAIN_ANSWER = (
    "The provided patient-criteria text is too short/sparse, or no seeded "
    "trial protocol could be identified, to support a reliable eligibility "
    "pre-screen. No per-criterion assessment has been made. Please provide "
    "the trial identifier and the relevant patient criteria, or route this "
    "request directly to the study team for manual review."
)

_TIER_HEADER = {
    "inclusion": "## Inclusion Criteria",
    "exclusion": "## Exclusion Criteria",
}

_STATUS_LABEL = {
    "appears_satisfied": "APPEARS SATISFIED",
    "flagged_for_review": "FLAGGED FOR REVIEW",
    "uncertain": "UNCERTAIN",
    "insufficient_info": "INSUFFICIENT INFO",
}


def _format_line(ref: Optional[int], entry: Dict[str, Any]) -> str:
    criterion_text = str(entry.get("criterion_text", "")).strip() or "(criterion text unavailable)"
    status = str(entry.get("status", ""))
    label = _STATUS_LABEL.get(status, status.upper())
    matched_term = entry.get("matched_term")
    cite = f" [{ref}]" if ref is not None else ""

    if status == "flagged_for_review":
        detail = (
            f'text appears to reference "{matched_term}"'
            if matched_term
            else "a signal for this exclusion ground was found"
        )
        return f"- {criterion_text} — {label}{cite}: {detail}. Recommend PI review before proceeding."
    if status == "appears_satisfied":
        detail = (
            f'supported by "{matched_term}" in the provided text'
            if matched_term
            else "no evidence of the excluded condition was found in the provided text"
        )
        return f"- {criterion_text} — {label}{cite}: {detail}."
    if status == "uncertain":
        detail = (
            f'only a weak/ambiguous signal ("{matched_term}") was found - confirm with the patient record'
            if matched_term
            else "the provided text gives contradictory or ambiguous signal - confirm with the patient record"
        )
        return f"- {criterion_text} — {label}{cite}: {detail}."
    # insufficient_info
    return f"- {criterion_text} — {label}: the provided text does not address this criterion."


class GenerateAnswerNode(FunctionNode):
    """Rule-based, cited, criterion-grouped pre-screen narrative assembly.

    Input state keys:
        criterion_assessment: JSON list of final per-criterion assessments (from RerankFilterNode)
        overall_abstain:      bool (from RerankFilterNode)

    Output state keys (partial dict):
        grounded_answer: pre-screen narrative with [n] citation markers, or
                          the abstention message
        citations:        JSON list [{ref, id, trial_id, category, source}]
                          (empty on abstain)
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

        emit_progress("Composing the answer...")
        overall_abstain = bool(state.get("overall_abstain", False))
        assessment: List[Dict[str, Any]] = from_json(state.get("criterion_assessment"), []) or []

        if overall_abstain:
            emit_trace_event(
                "generate_answer_complete",
                {"overall_abstain": True, "citation_count": 0},
                state,
            )
            return {"grounded_answer": _ABSTAIN_ANSWER, "citations": to_json([])}

        citations: List[Dict[str, Any]] = []
        lines: List[str] = []

        inclusion = [e for e in assessment if isinstance(e, dict) and e.get("criterion_type") == "inclusion"]
        exclusion = [e for e in assessment if isinstance(e, dict) and e.get("criterion_type") != "inclusion"]

        def _emit_group(header: str, group: List[Dict[str, Any]]) -> None:
            if not group:
                return
            lines.append(header)
            lines.append("")
            for entry in group:
                ref: Optional[int] = None
                if entry.get("status") in ("appears_satisfied", "flagged_for_review", "uncertain"):
                    ref = len(citations) + 1
                    citations.append(
                        {
                            "ref": ref,
                            "id": entry.get("id", ""),
                            "trial_id": entry.get("trial_id", ""),
                            "category": entry.get("category", ""),
                            "source": entry.get("source", ""),
                        }
                    )
                lines.append(_format_line(ref, entry))
            lines.append("")

        if not inclusion and not exclusion:
            lines.append(
                "No trial criteria were assessed for this request (no seeded "
                "knowledge-base entries were available for the identified trial)."
            )
        else:
            _emit_group(_TIER_HEADER["inclusion"], inclusion)
            _emit_group(_TIER_HEADER["exclusion"], exclusion)

        flagged_count = sum(1 for e in assessment if isinstance(e, dict) and e.get("status") == "flagged_for_review")
        if flagged_count:
            lines.append(
                f"**{flagged_count} criterion/criteria flagged for review above.** "
                "This is a pre-screen signal, not a determination."
            )
            lines.append("")

        grounded_answer = "\n".join(lines).rstrip()

        # Domain audit: pre-screen narrative assembled. Counts only.
        emit_trace_event(
            "generate_answer_complete",
            {
                "overall_abstain": False,
                "citation_count": len(citations),
                "inclusion_count": len(inclusion),
                "exclusion_count": len(exclusion),
                "flagged_for_review_count": flagged_count,
                "answer_chars": len(grounded_answer),
            },
            state,
        )

        return {
            "grounded_answer": grounded_answer,
            "citations": to_json(citations),
        }
