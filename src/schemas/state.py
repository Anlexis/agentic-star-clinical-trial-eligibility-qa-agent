"""AgentCore Platform v1.0"""

# State must be a flat TypedDict - never Pydantic BaseModel.
# LangGraph checkpoints use msgpack serialization; Pydantic objects
# cause silent corruption.  Extend AgentState with agent-specific
# fields only.  Do NOT add credentials, secrets, or Pydantic models.
#
# Msgpack safety: structured fields (dict / list[dict]) are stored
# as JSON STRINGS, not bare Python containers - a bare dict/list in a
# checkpointed State field is a state-safety violation. Producers
# serialize with to_json() on write; consumers deserialize with from_json()
# on read.
#
# HCR-C2-002 - Clinical Trial Eligibility QA Agent (Cat 2 RAG). Two-layer
# nested Cat 2 graph: outer backbone (AgentBaseGraph) + inner domain workflow
# (BaseGraph). Fields below cover both layers.
#
# PHI / confidentiality note: this template screens NORMALIZED, caller-
# supplied patient-criteria TEXT only (age range, diagnosis, lab values,
# prior treatment) - never a real medical record or an EMR/registry lookup.
# PreProcessNode surface-strips PHI-shaped direct identifiers
# (MRN-style IDs, SSNs, e-mails, phone numbers, calendar-date-shaped tokens)
# from the payload BEFORE it is written to State - this is a real behaviour
# on the node, not a doc claim. The output gate in
# PostProcessNode additionally recurse-scans the final result for
# credential- and PHI-shaped content before it leaves the agent.
#
# Safety note: this agent is DECISION SUPPORT ONLY - it never issues an
# enrolment decision or a definitive eligible/ineligible verdict. It abstains
# (overall_abstain) on low-confidence / unidentified-trial input rather than
# guessing, and every output carries a mandatory, non-suppressible PI-review
# disclaimer (see docs/02_design.md "Safety Boundary").

import json
from typing import Any, Optional

from framework.schemas.agent_state import AgentState


def to_json(value: Any) -> Optional[str]:
    """Serialize a dict/list State field to a JSON string (msgpack safety).

    None passes through unchanged so an 'unset' field stays distinguishable
    from an empty container.
    """
    if value is None:
        return None
    return json.dumps(value, ensure_ascii=False)


def from_json(value: Optional[str], default: Any = None) -> Any:
    """Deserialize a JSON-string State field back to its dict/list.

    None / empty / malformed input -> the supplied ``default`` so a missing or
    corrupt field is non-fatal for the consuming node.
    """
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


class State(AgentState):
    """Flat TypedDict for HCR-C2-002.

    All shared fields (user_input, status, session_id, node_history,
    error_log, hitl_*, etc.) are inherited from AgentState.
    Domain fields are Optional (default-absent) so the state is valid at
    graph initialisation, before any node has written a value.
    """

    # ------------------------------------------------------------------
    # Outer layer - set by PreProcessNode / EligibilityScreeningGraphNode.merge_output
    # ------------------------------------------------------------------

    # PHI-identifier-stripped, validated patient-criteria payload produced by
    # PreProcessNode. Raw input is NOT persisted beyond it.
    validated_input: Optional[str]

    # Final eligibility pre-screen result, mapped from the inner graph's
    # formatted_answer output via merge_output.
    eligibility_assessment_result: Optional[str]

    # ------------------------------------------------------------------
    # Inner layer - domain nodes (DomainWorkflowGraph)
    # ------------------------------------------------------------------

    # InputValidateNode output
    # Normalised patient-criteria text (whitespace-collapsed, length-capped).
    patient_criteria_text: Optional[str]

    # JSON STRING (to_json) of parsed structured request params. Deserialised
    # dict shape: {"trial_id": str | None}. Consumers (RetrieveNode) read it
    # back via from_json(). trial_id is resolved either from a JSON envelope
    # key or extracted from the free-text payload (e.g. "trial protocol
    # XYZ-123").
    query_filters: Optional[str]

    # Runtime `retrieval` block (config/config.yaml) forwarded by
    # EligibilityScreeningGraphNode._parent_config() ->
    # DomainWorkflowGraph._extra_initial_state().
    # JSON STRING (to_json) of {"score_threshold": float, "kb_path": str,
    # "abstain_min_chars": int}. Consumers (RetrieveNode, RerankFilterNode)
    # read it back via from_json().
    retrieval_config: Optional[str]

    # RetrieveNode output
    # JSON STRING (to_json) of per-criterion match candidates for the
    # resolved trial_id. Deserialised shape: list[dict], each entry
    # {"id": str, "trial_id": str, "criterion_type": "inclusion"|"exclusion",
    # "category": str, "criterion_text": str, "source": str, "score": float,
    # "matched_term": str | None, "negated": bool}. EVERY criterion belonging
    # to the resolved trial is always a candidate (there is no tier/cap
    # concept here - dropping a criterion, e.g. an exclusion ground, would be
    # a patient-safety gap). Empty when trial_id could not be resolved or
    # does not match any seeded trial.
    matched_criteria: Optional[str]

    # RerankFilterNode output
    # JSON STRING (to_json) of the final per-criterion assessment. Deserialised
    # shape: list[dict], each entry {"id": str, "trial_id": str,
    # "criterion_type": str, "category": str, "criterion_text": str,
    # "status": "appears_satisfied"|"uncertain"|"flagged_for_review"|
    # "insufficient_info", "score": float, "matched_term": str | None,
    # "source": str}. Consumers (GenerateAnswerNode) read it back via
    # from_json(). Empty (to_json([])) when overall_abstain is True -
    # abstention emits NO synthesised per-criterion verdict.
    criterion_assessment: Optional[str]

    # RerankFilterNode output (SAFETY BOUNDARY field)
    # True when the patient-criteria text is too short/sparse, or no trial
    # could be identified/matched in the seeded KB, to support a reliable
    # pre-screen (see docs/02_design.md "Safety Boundary"). When True,
    # GenerateAnswerNode emits an abstention message instead of a synthesised
    # assessment - this agent must NEVER assert a per-criterion read on
    # low-confidence input.
    overall_abstain: Optional[bool]

    # GenerateAnswerNode outputs
    # Rule-assembled advisory body grouped by Inclusion/Exclusion, with
    # numbered citation markers. Never an eligibility verdict.
    grounded_answer: Optional[str]

    # JSON STRING (to_json) of citations. Deserialised shape: list[dict],
    # each entry {"ref": int, "id": str, "trial_id": str, "category": str,
    # "source": str}. Consumers (OutputFormatNode) read it back via
    # from_json().
    citations: Optional[str]

    # OutputFormatNode output
    # Final formatted assessment body (grouped statuses + sources). The
    # mandatory, non-suppressible PI-review disclaimer is appended
    # downstream by PostProcessNode, NOT here - see docs/02_design.md
    # "Safety Boundary".
    formatted_answer: Optional[str]

    # Validation / parse notes accumulated during intake (no PHI).
    # JSON STRING (to_json) of list[str].
    intake_notes: Optional[str]

    # ------------------------------------------------------------------
    # Degraded completion marker
    # ------------------------------------------------------------------

    # Set when the run completes WITHOUT producing an answer because the
    # caller's request could not be accepted as written - a rejection the
    # caller can correct and retry. The run still completes: no retrieval is
    # performed, no answer is assembled, and the domain audit event for the
    # rejection is still emitted. Carrying this as a completion marker rather
    # than a terminal error is what lets the caller see the reason and send a
    # corrected request on the same conversation.
    #
    # Content the agent refuses outright, and a breach of a contract the
    # caller cannot influence, are NOT reported here - those stay terminal so
    # they are not mistaken for something a reworded request would get past.
    #
    # Once set, every later domain node passes through without doing work, and
    # the value is carried across the inner/outer boundary by get_output() and
    # merge_output().
    error_code: Optional[str]

    # ------------------------------------------------------------------
    # Tracing / audit - framework-managed; do NOT write from node code
    # ------------------------------------------------------------------

    trace_id: Optional[str]
    correlation_id: Optional[str]
    # node_history inherited from AgentState; listed here for clarity
    # node_history: Optional[List[str]]
