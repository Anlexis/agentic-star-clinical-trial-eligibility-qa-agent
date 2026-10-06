"""AgentCore Platform v1.0"""

# HCR-C2-002 - RetrieveNode
# Domain node 2: deterministic term matching over the seeded clinical-trial
# protocol knowledge base (config/kb/clinical_trials_kb.json). v1 is fully
# deterministic - no embedding model, no vector store, no live EMR / trial-
# registry connection. The match contract (matched_criteria JSON) is
# store-agnostic so a later vector-store upgrade only swaps this node's
# internals (see docs/02_design.md "v1 Implementation Note").
#
# Every criterion belonging to the resolved trial_id is ALWAYS a candidate
# (no tier/top_k cap concept, unlike a generic KB search) - a compliance-
# grade pre-screen needs a status for every inclusion/exclusion criterion of
# the identified trial every time; silently dropping one (e.g. an exclusion
# ground) would be a patient-safety gap, not just a relevance judgement.
#
# NEGATION AWARENESS (safety-critical for this domain): clinical narrative
# text routinely negates a criterion term ("no prior chemotherapy", "denies
# brain metastases") - a bare substring match would misread that as POSITIVE
# evidence. Each match records whether it was found inside a small negation
# window (a negation cue word within ~40 chars before the match); RerankFilterNode
# uses this to avoid flagging an explicitly-absent condition as present. This
# is a documented v1 heuristic (no NLP dependency), not full clinical NLP.
#
# Config: reads `score_threshold` / `kb_path` from the state field
# retrieval_config (republished by DomainWorkflowGraph._extra_initial_state()
# from config/config.yaml, forwarded by EligibilityScreeningGraphNode._parent_config());
# falls back to module defaults that mirror config/config.yaml when the field
# is absent (e.g. this node invoked standalone in a boundary test).
#
# Wired by the inner graph (DomainWorkflowGraph).
# Returns only changed state keys (partial dict).

import json
from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional, Tuple

from framework.nodes.function_node import FunctionNode
from framework.schemas.trust_level import TrustLevel
from shared.utils.audit_logger import emit_trace_event
from src.services.progress import emit_progress
from src.schemas.state import from_json, to_json
from framework.schemas.agent_status import AgentStatus

# Defaults mirror the `retrieval` block in config/config.yaml.
_DEFAULT_RETRIEVAL: Dict[str, Any] = {
    "score_threshold": 0.75,
    "kb_path": "config/kb/clinical_trials_kb.json",
    "abstain_min_chars": 25,
}

# Repo root: src/nodes/retrieve_node.py -> parents[2].
_REPO_ROOT = Path(__file__).resolve().parents[2]

_EXPLICIT_SCORE = 1.0
_IMPLICIT_SCORE = 0.5
_NO_MATCH_SCORE = 0.0

# Negation cue words checked in the window immediately BEFORE a matched term.
# Deliberately short list of unambiguous clinical-negation phrasing - a
# heuristic, not a parser; documented limitation (docs/02_design.md).
_NEGATION_CUES = (
    "no ",
    "not ",
    "denies ",
    "denied ",
    "without ",
    "absence of ",
    "negative for ",
    "never had ",
    "free of ",
    "ruled out ",
    "no evidence of ",
    "no history of ",
)
_NEGATION_WINDOW_CHARS = 40

# Excerpt length carried into matched_criteria (keeps State small).
_TEXT_TRUNC_CHARS = 280


def _resolve_retrieval_config(state: Dict[str, Any]) -> Dict[str, Any]:
    """Effective retrieval config: state retrieval_config > module defaults."""
    effective = dict(_DEFAULT_RETRIEVAL)  # local copy - never mutate the module default
    from_state = from_json(state.get("retrieval_config"), None)
    if isinstance(from_state, dict):
        effective.update(from_state)
    return effective


def _load_kb(kb_path: str) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Load the seeded KB JSON. Missing / malformed file degrades gracefully."""
    notes: List[str] = []
    path = Path(kb_path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        notes.append(f"RetrieveNode: clinical-trial knowledge base not readable at {kb_path}.")
        return [], notes
    if not isinstance(entries, list):
        notes.append("RetrieveNode: clinical-trial knowledge base root must be a JSON list.")
        return [], notes
    return [e for e in entries if isinstance(e, dict)], notes


def _find_term(text_lower: str, term_lower: str) -> Optional[Tuple[int, bool]]:
    """(match_start_idx, negated) for term in text, or None if not found.

    `negated` is True when a negation cue word appears in the small window
    immediately before the match (e.g. "no prior chemotherapy").

    The window is additionally clause-scoped: truncated at the last comma /
    period / semicolon inside it, so a negation cue from an earlier,
    unrelated clause never leaks across a clause boundary. Without this, "no
    known brain metastases, adequate organ function..." would misread
    "adequate organ function" as negated by the "no" that actually belongs
    to the PRECEDING clause - a real false-positive class this
    clause-scoping exists to prevent, pinned by a dedicated test.
    """
    if not term_lower:
        return None
    idx = text_lower.find(term_lower)
    if idx == -1:
        return None
    window_start = max(0, idx - _NEGATION_WINDOW_CHARS)
    window = text_lower[window_start:idx]
    last_boundary = max(window.rfind(","), window.rfind("."), window.rfind(";"))
    if last_boundary != -1:
        window = window[last_boundary + 1 :]
    negated = any(cue in window for cue in _NEGATION_CUES)
    return idx, negated


def _match_entry(entry: Dict[str, Any], text_lower: str) -> Tuple[float, Optional[str], bool]:
    """(score, matched_term, negated) for one KB entry against the patient text.

    Case-insensitive substring matching (both sides lower-cased) - free-text
    clinical narrative carries far more casing variance than a curated label
    corpus, so both sides are normalised rather than relying on pre-curated
    per-case KB term variants.
    """
    if not text_lower:
        return _NO_MATCH_SCORE, None, False
    for term in entry.get("explicit_terms", []) or []:
        if not isinstance(term, str):
            continue
        hit = _find_term(text_lower, term.lower())
        if hit:
            _, negated = hit
            return _EXPLICIT_SCORE, term, negated
    for term in entry.get("implicit_hint_terms", []) or []:
        if not isinstance(term, str):
            continue
        hit = _find_term(text_lower, term.lower())
        if hit:
            _, negated = hit
            return _IMPLICIT_SCORE, term, negated
    return _NO_MATCH_SCORE, None, False


class RetrieveNode(FunctionNode):
    """Match every criterion of the resolved trial against the patient-criteria text.

    Input state keys:
        patient_criteria_text: normalised text (from InputValidateNode)
        query_filters:         JSON dict with the resolved trial_id
        retrieval_config:      forwarded runtime retrieval block (JSON)

    Output state keys (partial dict):
        matched_criteria: JSON list of per-criterion match candidates
        intake_notes:     (on KB anomalies / unresolved trial) JSON list[str]
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

        emit_progress("Searching the knowledge base...")
        text = state.get("patient_criteria_text") or state.get("validated_input") or state.get("user_input", "")
        text = text if isinstance(text, str) else ""
        text_lower = text.lower()

        filters = from_json(state.get("query_filters"), {}) or {}
        trial_id = filters.get("trial_id")

        retrieval_cfg = _resolve_retrieval_config(state)
        entries, notes = _load_kb(str(retrieval_cfg.get("kb_path", _DEFAULT_RETRIEVAL["kb_path"])))

        candidates: List[Dict[str, Any]] = []
        if not trial_id:
            notes.append("RetrieveNode: no trial_id resolved - cannot retrieve protocol criteria.")
        else:
            trial_entries = [
                e for e in entries if str(e.get("trial_id", "")).strip().upper() == str(trial_id).strip().upper()
            ]
            if not trial_entries:
                notes.append(f"RetrieveNode: trial_id '{trial_id}' matched no seeded protocol criteria.")
            for entry in trial_entries:
                score, matched_term, negated = _match_entry(entry, text_lower)
                candidates.append(
                    {
                        "id": str(entry.get("id", "")),
                        "trial_id": str(entry.get("trial_id", "")),
                        "criterion_type": str(entry.get("criterion_type", "")),
                        "category": str(entry.get("category", "")),
                        "criterion_text": str(entry.get("criterion_text", ""))[:_TEXT_TRUNC_CHARS],
                        "source": str(entry.get("source", "")),
                        "score": score,
                        "matched_term": matched_term,
                        "negated": negated,
                    }
                )

        # Deterministic ordering: inclusion before exclusion, then id asc.
        candidates.sort(key=lambda c: (0 if c["criterion_type"] == "inclusion" else 1, c["id"]))

        # Domain audit: criterion-matching pass completed. NEVER include
        # the patient-criteria text or trial_id verbatim in the audit payload.
        emit_trace_event(
            "retrieve_complete",
            {
                "candidates": len(candidates),
                "kb_entries": len(entries),
                "has_trial_id": trial_id is not None,
                "text_chars": len(text),
            },
            state,
        )

        out: Dict[str, Any] = {"matched_criteria": to_json(candidates)}
        if notes:
            # Append to (never clobber) the notes accumulated upstream.
            prior = from_json(state.get("intake_notes"), []) or []
            out["intake_notes"] = to_json(list(prior) + notes)
        return out
