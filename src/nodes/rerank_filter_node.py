"""AgentCore Platform v1.0"""

# HCR-C2-002 - RerankFilterNode
# Domain node 3: turn the raw match candidates into final per-criterion
# assessments, and compute the SAFETY-CRITICAL overall_abstain flag.
#
# Status vocabulary (deliberately NOT "meets"/"fails"/"eligible"/"ineligible"
# - this template is decision-support only and must never imply an
# eligibility verdict - see docs/02_design.md "Safety Boundary"):
#
#   appears_satisfied   - available evidence favours this criterion being
#                          satisfied for the patient (an inclusion term found
#                          un-negated, OR an exclusion term either absent
#                          entirely or found NEGATED - i.e. the excluded
#                          condition appears absent)
#   flagged_for_review   - an exclusion criterion's term was found
#                          UN-negated (the excluded condition appears
#                          present) - the single status that should draw a
#                          reviewer's attention first
#   uncertain            - only a weak/implicit signal was found, OR an
#                          inclusion criterion's term was found NEGATED
#                          (explicit contrary evidence, but this node still
#                          never asserts a bare "not met")
#   insufficient_info    - an inclusion criterion the patient text simply
#                          does not address at all (score 0, not negated)
#
# This classification is symmetric-conservative: for BOTH criterion types,
# a status is only asserted off POSITIVE textual evidence (never inferred
# from silence into a confident verdict) - the exclusion-tier "no evidence
# found" default is appears_satisfied (clears the exclusion) rather than
# flagged, and the inclusion-tier "no evidence found" default is
# insufficient_info rather than appears_satisfied. Never a bare "fails".
#
# SAFETY BOUNDARY (docs/02_design.md): overall_abstain is True when the
# patient-criteria text is too short/sparse (len < abstain_min_chars) OR no
# criteria could be retrieved at all (trial_id unresolved / not in the
# seeded KB). On abstain, criterion_assessment is emitted EMPTY (to_json([]))
# - this node never lets a low-confidence / unidentified-trial input reach
# GenerateAnswerNode with a synthesisable per-criterion assessment list.
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

from typing import Any, ClassVar, Dict, List

from framework.nodes.function_node import FunctionNode
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.schemas.state import from_json, to_json
from framework.schemas.agent_status import AgentStatus

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "score_threshold": 0.75,
    "abstain_min_chars": 25,
}

_STATUS_APPEARS_SATISFIED = "appears_satisfied"
_STATUS_FLAGGED_FOR_REVIEW = "flagged_for_review"
_STATUS_UNCERTAIN = "uncertain"
_STATUS_INSUFFICIENT_INFO = "insufficient_info"


def _resolve_retrieval_config(state: Dict[str, Any]) -> Dict[str, Any]:
    """Effective retrieval config: state retrieval_config > module defaults."""
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


def _classify(criterion_type: str, score: float, negated: bool, score_threshold: float) -> str:
    """Map (criterion_type, score, negated) to the status vocabulary. See module docstring."""
    is_exclusion = criterion_type == "exclusion"
    if score >= score_threshold:
        if negated:
            # Confident match, but the matched term was explicitly negated.
            return _STATUS_APPEARS_SATISFIED if is_exclusion else _STATUS_UNCERTAIN
        # Confident match, not negated.
        return _STATUS_FLAGGED_FOR_REVIEW if is_exclusion else _STATUS_APPEARS_SATISFIED
    if score > 0:
        # Only a weak/implicit hint either way - always uncertain.
        return _STATUS_UNCERTAIN
    # No textual evidence at all.
    return _STATUS_APPEARS_SATISFIED if is_exclusion else _STATUS_INSUFFICIENT_INFO


class RerankFilterNode(FunctionNode):
    """Classify match candidates into per-criterion statuses; compute abstain.

    Input state keys:
        matched_criteria: JSON list of match candidates (from RetrieveNode)
        retrieval_config: forwarded runtime retrieval block (JSON)
        patient_criteria_text: normalised patient-criteria text (abstain check)

    Output state keys (partial dict):
        criterion_assessment: JSON list of final per-criterion assessments
                               (empty when overall_abstain is True)
        overall_abstain:      bool - True when input confidence is too low,
                               or the trial is unidentified, for a reliable
                               pre-screen (SAFETY BOUNDARY)
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

        emit_progress("Ranking the results...")
        candidates: List[Dict[str, Any]] = from_json(state.get("matched_criteria"), []) or []
        retrieval_cfg = _resolve_retrieval_config(state)
        text = state.get("patient_criteria_text") or ""
        text = text if isinstance(text, str) else ""

        try:
            abstain_min_chars = int(retrieval_cfg.get("abstain_min_chars", _DEFAULT_RETRIEVAL["abstain_min_chars"]))
        except (TypeError, ValueError):
            abstain_min_chars = int(_DEFAULT_RETRIEVAL["abstain_min_chars"])

        # Abstain when the text is too sparse, OR when RetrieveNode could not
        # resolve any criteria at all (unidentified / unseeded trial) - both
        # are "insufficient basis to assess" cases.
        overall_abstain = len(text.strip()) < max(0, abstain_min_chars) or not candidates

        if overall_abstain:
            emit_trace_event(
                "rerank_filter_complete",
                {"overall_abstain": True, "candidates_seen": len(candidates)},
                state,
            )
            return {"criterion_assessment": to_json([]), "overall_abstain": True}

        try:
            score_threshold = float(retrieval_cfg.get("score_threshold", _DEFAULT_RETRIEVAL["score_threshold"]))
        except (TypeError, ValueError):
            score_threshold = float(_DEFAULT_RETRIEVAL["score_threshold"])
        score_threshold = max(0.0, min(1.0, score_threshold))

        assessment: List[Dict[str, Any]] = []
        status_counts: Dict[str, int] = {}
        for entry in candidates:
            if not isinstance(entry, dict):
                continue
            try:
                score = float(entry.get("score", 0.0))
            except (TypeError, ValueError):
                score = 0.0
            criterion_type = str(entry.get("criterion_type", ""))
            negated = bool(entry.get("negated", False))
            status = _classify(criterion_type, score, negated, score_threshold)
            status_counts[status] = status_counts.get(status, 0) + 1
            assessment.append(
                {
                    "id": entry.get("id", ""),
                    "trial_id": entry.get("trial_id", ""),
                    "criterion_type": criterion_type,
                    "category": entry.get("category", ""),
                    "criterion_text": entry.get("criterion_text", ""),
                    "status": status,
                    "score": round(score, 4),
                    "matched_term": entry.get("matched_term"),
                    "source": entry.get("source", ""),
                }
            )

        # Domain audit: statuses classified, abstain gate evaluated clear.
        # Counts only - never the criterion text or patient text.
        emit_trace_event(
            "rerank_filter_complete",
            {
                "overall_abstain": False,
                "assessed_count": len(assessment),
                "flagged_for_review_count": status_counts.get(_STATUS_FLAGGED_FOR_REVIEW, 0),
                "score_threshold": score_threshold,
            },
            state,
        )

        return {"criterion_assessment": to_json(assessment), "overall_abstain": False}
