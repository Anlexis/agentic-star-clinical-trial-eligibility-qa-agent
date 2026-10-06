# Template Design Specification — HCR-C2-002

**Template ID:** HCR-C2-002
**Template Name:** ClinicalTrialEligibilityQAAgent
**Category:** Cat 2 (multi-step domain workflow — RAG pattern)
**Industry:** HCR

## Position in AgentCore Architecture

| Field | Value |
|---|---|
| Agent Class | `ClinicalTrialEligibilityQAAgent` (alias `Graph`) |
| L1 Base (framework base class) | AgentBaseGraph — direct framework inheritance |
| Inner graph base | `BaseGraph` — `DomainWorkflowGraph` |
| Pattern | Cat 2 two-layer nested architecture (outer fixed 5-node backbone + `GraphNode` in the `main` slot wrapping an inner `BaseGraph` domain workflow) |

- **Separation of concerns:**
  - State: flat `TypedDict` composition (no Pydantic — msgpack incompatible);
    structured fields stored as JSON strings via `to_json()` / `from_json()`
  - Node: framework inheritance via `FunctionNode` (override `execute(self, state) -> dict` only)
  - Graph: composition (`register_nodes()` for node substitution); outer
    `add_edges()` is NOT overridden

## Purpose

Clinical-trial eligibility pre-screen Q&A: a clinician or trial coordinator
submits de-identified patient criteria (age, diagnosis, lab values, prior
treatment) together with a trial identifier; the agent retrieves the
relevant inclusion/exclusion criteria from a seeded trial-protocol knowledge
base and returns a cited, criterion-by-criterion advisory assessment —
retrieve → classify → grounded narrative with citations + a standing,
non-suppressible PI-review disclaimer. v1 is fully deterministic (keyword/
term matching + rule-based narrative assembly; no live LLM call — see the
v1 Implementation Note below). **Decision-support only — see "Safety
Boundary" below.**

## Architecture Overview

### Outer backbone (AgentBaseGraph)

```
START → initialize → pre_process → main → {route} → post_process → finalize → END
                                     ↓ (retry, max 3)
                                   pre_process
```

| Slot | Class | Responsibility | required_trust_level |
|------|-------|----------------|----------------------|
| initialize | InitializeNode (framework default) | session_id, trust_level, schema_version | — (framework) |
| pre_process | `PreProcessNode` | validate non-empty input; surface-strip PHI-shaped direct identifiers (MRN-style IDs, SSNs, e-mails, phone numbers, DOB-shaped dates) → `validated_input` | `TrustLevel.VERIFIED_EXTERNAL` |
| main | `EligibilityScreeningGraphNode` (`GraphNode`) | delegates to inner `DomainWorkflowGraph`; maps inner `formatted_answer` → outer `result`; stashes the caller's `input_context` on the context bridge | — (GraphNode delegation) |
| post_process | `PostProcessNode` | output gate — module-level `_security_gate_output()` scans `result` for credentials/PHI → ERROR + sanitised stub; on the clean path, attaches AND VERIFIES the mandatory PI-review disclaimer before returning SUCCESS | `TrustLevel.VERIFIED_EXTERNAL` |
| finalize | FinalizeNode (framework default) | response_metadata, total_time_ms | — (framework) |

### Inner graph (DomainWorkflowGraph — BaseGraph, linear)

```
START → input_validate → retrieve → rerank_filter → generate_answer → output_format → END
```

All five inner domain nodes declare `required_trust_level = TrustLevel.ANONYMOUS`
(the external trust gate lives on the outer backbone gate node; a stricter inner
level would deny a real VERIFIED_EXTERNAL invoke at runtime).

| Node | Responsibility | required_trust_level | Input State | Output State |
|------|----------------|----------------------|-------------|---------------|
| `InputValidateNode` | Validate the caller-data contract (`input_context.patient_profile` / `input_context.trial_id`, fail-closed) and parse the (possibly JSON-enveloped) request; normalise whitespace; cap length; resolve `trial_id` (context field, JSON key, or regex-extracted from free text, e.g. "trial protocol XYZ-123" — always grammar-locked) | `TrustLevel.ANONYMOUS` | `input_context`, `validated_input` \| `user_input` | `patient_criteria_text`, `query_filters`, `intake_notes` |
| `RetrieveNode` | Deterministic term matching over the seeded KB (`config/kb/clinical_trials_kb.json`): every criterion of the resolved `trial_id` is ALWAYS a candidate (no cap/tier — dropping a criterion, e.g. an exclusion ground, would be a patient-safety gap); records a match score, the matched term, and whether the match sat inside a negation window | `TrustLevel.ANONYMOUS` | `patient_criteria_text`, `query_filters`, `retrieval_config` | `matched_criteria`, `intake_notes` |
| `RerankFilterNode` | Classify each candidate into the status vocabulary (see "Safety Boundary"); computes the `overall_abstain` safety flag | `TrustLevel.ANONYMOUS` | `matched_criteria`, `retrieval_config`, `patient_criteria_text` | `criterion_assessment`, `overall_abstain` |
| `GenerateAnswerNode` | Rule-based, cited, Inclusion/Exclusion-grouped narrative assembly from `criterion_assessment` only (v1 deterministic — LLM synthesis seam documented below); abstention message when `overall_abstain` | `TrustLevel.ANONYMOUS` | `criterion_assessment`, `overall_abstain` | `grounded_answer`, `citations` |
| `OutputFormatNode` | Compose the final answer: body + Sources list (disclaimer is appended by `post_process`, NOT here — see "Safety Boundary") | `TrustLevel.ANONYMOUS` | `grounded_answer`, `citations` | `formatted_answer`, `status` |

### Data Flow

```
user_input + input_context
  → PreProcessNode (input gates)                  → validated_input (PHI-stripped)
  → EligibilityScreeningGraphNode.extract_input   → stashes input_context on the context bridge
                                                  → inner DomainWorkflowGraph.invoke(validated_input)
        → input_validate                         → patient_criteria_text / query_filters (trial_id)
        → retrieve                                → matched_criteria
        → rerank_filter                           → criterion_assessment / overall_abstain
        → generate_answer                         → grounded_answer / citations
        → output_format                           → formatted_answer
     get_output() → {formatted_answer, citations, criterion_assessment, overall_abstain, status, ...}
  → EligibilityScreeningGraphNode.merge_output    → result = formatted_answer, eligibility_assessment_result
  → PostProcessNode (output gate)                  → formatted_output (gated + disclaimer attached & verified)
```

### Caller data contract (`input_context`)

`/invoke` accepts structured invocation parameters via `input_context` (an
SDK-level `BaseGraph.invoke` parameter; the HTTP adapter caps the serialized
size at 256 KiB). Every field is validated by `InputValidateNode`,
fail-closed — a non-conforming value returns `status=error` naming the field,
never echoing the value:

**Completion is not the same as answering.** A run that ends with
`AgentStatus.SUCCESS` reports that the request was handled safely to a defined
end, not that an answer was produced. A value the caller can correct (an
out-of-contract parameter, an empty or over-long request) ends this way so the
caller receives the reason and can send a corrected request on the same
conversation; terminating instead would end the calling surface's turn and
surface only an exception type, leaving the reason reachable solely from the
audit trail. The reason travels as `error_code` in State, every later domain
node passes through without doing work once it is set, the structured output
fields are withheld, and `PostProcessNode` renders the reason as a static
caller-facing sentence.

Two classes keep terminating, and must not be folded into the above: content
the agent refuses outright (an instruction-override payload — re-sending a
reworded variant is not a correction), and a breach of a contract the caller
cannot influence.

| Field | Type | Contract |
|---|---|---|
| `patient_profile` | `str` | De-identified patient-criteria free text. Surface-stripped of PHI-shaped identifiers on intake (same strip the string payload gets in `PreProcessNode`), whitespace-normalised, length-capped at 4,000 chars. |
| `trial_id` | `str` | Explicit trial filter. Locked to the trial-identifier grammar `[A-Za-z]{2,6}-[0-9]{2,4}` (an inert identifier shape); anything else is rejected. |

`GraphNode.execute()` (SDK 1.0.1) does not forward `input_context` on
`subgraph.invoke()`, so the outer `extract_input()` stashes it on a
per-task ContextVar bridge (`src/graph/context_bridge.py`) and the inner
graph's `_extra_initial_state()` seeds it back into inner state.

Structured parameters may equivalently travel as JSON inside the string
payload: when the caller supplies a JSON envelope
(`{"patient_profile": ..., "trial_id": ...}`), it passes through
`validated_input` as a string and the FIRST inner node
(`InputValidateNode`) parses it back — under the same validation contract.
`input_context` values take precedence over the envelope, which takes
precedence over free-text extraction. Absent caller data degrades to the
plain-text path (and downstream, to the abstention baseline) — never a
fabricated assessment.

### Runtime config forwarding (`_parent_config`)

`EligibilityScreeningGraphNode._parent_config()` receives the runtime config
from the outer graph (threaded in at `register_nodes()` time from `self.config`,
which the entry point loaded from `config/config.yaml`)
and forwards its blocks under `config["configurable"]` (never `{}`):

```
{"configurable": {"retrieval": {score_threshold, kb_path, abstain_min_chars}, "llm": {...}, "max_retry": 3, "timeout_seconds": 30}}
```

(`timeout_s` in the file is mapped to `timeout_seconds`, the key the inner
graph validates; `max_retry` / `timeout_seconds` must be positive integers
when present — `DomainWorkflowGraph._validate_config()` fails loudly at
compile time otherwise.)

`get_subgraph()` passes this into `DomainWorkflowGraph(config=...)`; the inner
graph republishes the `retrieval` block into the inner initial state as the
JSON-string field `retrieval_config` (via `_extra_initial_state()`), so the
declared `score_threshold` / `abstain_min_chars` are live at runtime.
`RetrieveNode` and `RerankFilterNode` read those keys from `retrieval_config`,
falling back to module defaults that mirror the config/config.yaml values.

### State Definition

| Field | Type | Purpose | Layer |
|-------|------|---------|-------|
| `validated_input` | `Optional[str]` | PHI-identifier-stripped request payload | outer |
| `eligibility_assessment_result` | `Optional[str]` | final answer, mapped from inner `formatted_answer` | outer |
| `patient_criteria_text` | `Optional[str]` | normalised patient-criteria text | inner |
| `query_filters` | `Optional[str]` (JSON) | `{"trial_id": str \| None}` | inner |
| `retrieval_config` | `Optional[str]` (JSON) | forwarded runtime `retrieval` block (config/config.yaml) | inner |
| `matched_criteria` | `Optional[str]` (JSON) | per-criterion match candidates for the resolved trial | inner |
| `criterion_assessment` | `Optional[str]` (JSON) | final per-criterion status list | inner |
| `overall_abstain` | `Optional[bool]` | SAFETY BOUNDARY flag | inner |
| `grounded_answer` | `Optional[str]` | rule-assembled pre-screen narrative body | inner |
| `citations` | `Optional[str]` (JSON) | `[{ref, id, trial_id, category, source}]` | inner |
| `formatted_answer` | `Optional[str]` | final answer + sources (no disclaimer yet) | inner |
| `intake_notes` | `Optional[str]` (JSON) | validation / parse notes (no PHI) | inner |
| `trace_id` / `correlation_id` | `Optional[str]` | framework-managed tracing | both |

**State Constraints (mandatory):**
- Flat `TypedDict` only (primitives + JSON-serialisable types).
- Structured fields (dict / list[dict]) stored as JSON STRINGS via `to_json()` /
  `from_json()` — used consistently by every producer AND consumer (msgpack
  safety).
- Domain fields are `Optional[...]` / default-absent (valid state before any node writes).
- `formatted_output` is NOT re-declared (backbone field stays framework-owned).
- No JWT, API keys, credentials, or raw PHI/personal identifiers in State.
- `InvocationContext` via `config["configurable"]` only (never in State).
- No Pydantic models / dataclasses / arbitrary Python objects.

## Safety Boundary

This template is **decision-support only**. It is built against the
following hard constraints — every constraint below is a real, exercised
node behaviour pinned by tests, not a doc-only claim:

1. **Never an enrolment decision or a definitive eligibility verdict.** No
   node, no output field, and no status value ever says "eligible",
   "ineligible", "enrolled", "meets", or "fails". `RerankFilterNode`'s status
   vocabulary is deliberately hedged: `appears_satisfied` / `uncertain` /
   `flagged_for_review` / `insufficient_info` — always advisory, always
   pointing back to a human (Principal Investigator) determination.
2. **Abstain on low-confidence / unidentified-trial input.** `RerankFilterNode`
   sets `overall_abstain = True` when the patient-criteria text is shorter
   than `abstain_min_chars`, OR when no criteria could be retrieved at all
   (no `trial_id` resolved, or the resolved `trial_id` matches no seeded
   protocol). On abstain, `GenerateAnswerNode` emits ONLY an abstention
   message routing to human/PI review — never a synthesised per-criterion
   read on a low-confidence input.
3. **PHI must not reach the inner workflow.** `PreProcessNode`
   surface-strips PHI-shaped direct identifiers (MRN-style IDs, SSNs,
   e-mails, phone numbers, calendar-date-shaped tokens) from the raw
   `user_input` BEFORE `validated_input` is written — this happens inline in
   `execute()`, exercised by `tests/unit/test_pre_process_node.py` and the
   trust-gate pass-through test, not merely described here. The
   `input_context.patient_profile` channel bypasses `PreProcessNode`, so
   `InputValidateNode` applies the SAME surface strip there — raw
   identifiers never reach inner state on either channel.
4. **Mandatory, non-suppressible PI-review disclaimer.** `PostProcessNode`
   attaches the disclaimer to every successful STRING output, then VERIFIES
   the disclaimer is actually present in the final text before returning
   SUCCESS — if a future edit ever drops the append, the gate fails CLOSED
   (blocks the output) instead of silently shipping a disclaimer-less
   clinical assessment. The disclaimer is added HERE — the one gate every
   response passes through unconditionally — not inside a domain node that a
   future change could accidentally bypass.
5. **Structured output is gated the same way as text.** `ClinicalTrialEligibilityQAAgent.get_output()`
   re-scans `criterion_assessment` / `citations` through the SAME recursive
   `_security_gate_output()` before surfacing them, and only on SUCCESS —
   never a raw forwarded dict, only whitelisted scalar fields.
6. **Negation-aware matching.** Clinical narrative text routinely negates a
   criterion term ("no prior chemotherapy", "denies brain metastases").
   `RetrieveNode` checks a small window before each matched term for a
   negation cue and records it; `RerankFilterNode` uses that signal so an
   explicitly-ABSENT excluded condition is never misread as evidence the
   patient is excluded. This is a documented v1 heuristic (no NLP
   dependency), not full clinical NLP — see the Design Decision Record.

## Framework Utilization

### Shared Components Used
- [x] InvocationContext (correlation_id, session_id, permissions, credential handle)
- [x] SecurityViolationError (via the output gate's ERROR path)
- [x] PHI/identifier surface-strip performed inline in `PreProcessNode.execute()`
      (module-level `_surface_strip_identifiers()` helper — NOT an
      `_extra_security_gate_input()` hook method)
- [x] Output gate: module-level `_security_gate_output()` in `PostProcessNode`,
      RECURSIVE (dict/list/tuple at any depth), plus the disclaimer-presence
      check described in "Safety Boundary" — NOT an `_extra_security_gate_output()`
      hook method
- [x] Audit: `emit_trace_event()` — one domain-specific event inside every node's
      `execute()` (mandatory); nodes do NOT emit `node_start` / `node_complete` /
      `node_error` — `BaseNode.__call__()` emits those. Domain event names:
  - `pre_process_complete`
  - `input_validate_complete`
  - `retrieve_complete`
  - `rerank_filter_complete`
  - `generate_answer_complete`
  - `output_format_complete`
  - `post_process_complete`

> **Input/output gate behaviour by node type:**
> - `FunctionNode` subclass → framework `@final` gate always runs automatically;
>   extend via `_extra_security_gate_input()` / `_extra_security_gate_output()` only
> - `GraphNode` / `RemoteAgentNode` → deliberate no-op (upstream or remote node's gate already applied)
> - Custom `BaseNode` subclass → must implement `_security_gate_input()` and
>   `_security_gate_output()` directly (`@abstractmethod` — omission raises `TypeError` at instantiation)
>
> This template does NOT define `_extra_security_gate_input()` /
> `_extra_security_gate_output()` on any node — the PHI strip and the
> disclaimer-presence check are both done as plain module-level helpers
> called inline from `execute()`.

### Entry Points

The agent is reachable through three entry points, all of which build the graph
from the same `config/config.yaml`:

| Entry point | Construction | Notes |
|---|---|---|
| Platform registry | `Graph(config=...)` by the registry | Reads `config/config.yaml` itself |
| Standalone HTTP (`src/api/server.py`) | Loads `config/config.yaml`, passes `Graph(config=...)` | Caller-auth boundary; see Security Design |
| Marketplace (`cli.py`) | `run_agent_marketplace(...)` is handed the graph class and the resolved config | The runner constructs the graph itself, so `cli.py` resolves `config/config.yaml` with `load_agent_config()` and passes it in; `extend_config` is the seam for deployment-specific overrides |

`cli.py` sits at the repository root because the deployment image starts it as
`CMD ["python", "cli.py"]`. It adds no business logic: graph construction,
lifecycle, secret provisioning and the invocation loop belong to
`run_agent_marketplace()`.

## Caller-Facing Events

Nodes report progress and rejection reasons to the caller as non-terminal
events, so a caller watching a run sees the pipeline advance instead of a
silent wait, and learns what to change when a request is refused.

- **Progress** — each node reports its phase at the top of `execute()`.
- **Rejection reason** — a node that returns `status: error` sends the reason
  first. It has to happen there: once the run carries an error status the
  framework skips `execute()` on every later node, so no downstream node could
  send it. Wording separates what the caller can fix (missing question,
  oversized request, malformed value) from what they cannot (retrieval or
  output failures), so a caller is not invited into a pointless retry.

Both are best-effort: the emitter is resolved lazily and failures are
swallowed, because reporting must never change the outcome of a run. Messages
are static phase and reason labels — no request value, record value or
internal identifier is ever included, since these events leave the process and
are not covered by the S-3 output gate. Terminal delivery (success/failure)
belongs to the platform runner alone.

## Composition Pattern

- **Pattern:** `GraphNode` (subgraph) in the outer `main` slot.
- **Composition target:** `DomainWorkflowGraph` (inner `BaseGraph`).
- **Error propagation strategy:** `propagate` (inner errors re-raised as `SubgraphError`).
- Inner domain nodes run at `TrustLevel.ANONYMOUS`; outer pre/post_process run
  at `TrustLevel.VERIFIED_EXTERNAL`.

## v1 Implementation Note — LLM synthesis

v1 of this template is **deterministic end-to-end**: `RetrieveNode` performs
case-insensitive term matching (with negation-window awareness) over the
seeded KB, and `GenerateAnswerNode` assembles the grounded narrative
rule-based from `criterion_assessment` (grouped Inclusion/Exclusion listing).
There is NO live LLM call and no LLM client dependency in v1 — the `llm`
block in `config/config.yaml` is forwarded through `_parent_config()` for
forward-compatibility but is not consumed by any v1 node, and no
`system_prompt` is read at runtime. The LLM synthesis upgrade seam is
documented in `config/prompts/eligibility_synthesis_prompt.md`: a v2
`GenerateAnswerNode` swaps the rule-based assembly for an LLM call that
synthesises over the same `criterion_assessment` input and emits the same
`grounded_answer` / `citations` state contract — the same hedged status
vocabulary and abstain rule carry over unchanged (see that file).

**Also deferred past v1 (separate from LLM synthesis):** numeric-threshold
criterion evaluation (an exact age cutoff, an eGFR/HbA1c numeric range
comparison) is NOT implemented — v1 evaluates every criterion, including
age/lab-value ones, via the same deterministic KB-term matching as
diagnosis/treatment-history criteria, with each KB entry's `explicit_terms`
curated to match common realistic phrasings. This keeps the matching
pipeline to a single, well-tested paradigm rather than mixing keyword
matching with a second numeric-comparison code path in v1. Live
vector-store retrieval and any EMR/clinical-trial-registry integration are
also out of scope for v1.

## Import Isolation Confirmation
- [x] Template does not import the platform-internal SDK.
- [x] Import targets: `framework/` and `shared/` only.
- [x] Base positions hold framework base classes only (`AgentBaseGraph`, `BaseGraph`, `FunctionNode`, `GraphNode`).

## Design Decision Record

| Decision | Option A | Option B | Chosen | Rationale |
|----------|----------|----------|--------|-----------|
| Base type | AgentBaseGraph | AutonomousBaseGraph | **AgentBaseGraph** | Fixed multi-step RAG pre-screen workflow, no autonomous loop |
| Composition pattern | Standalone Cat 1 slots | GraphNode → inner BaseGraph | **GraphNode → inner BaseGraph** | 5-step domain workflow exceeds a single `main` node; nested keeps the outer backbone untouched |
| Answer synthesis | Rule-based assembly | LLM call | **Rule-based (v1)** | Deterministic assembly is testable and auditable for a life-critical domain; v2 swaps the LLM in at the documented seam |
| Status vocabulary | "meets"/"fails"/"eligible" | Hedged advisory statuses | **Hedged (`appears_satisfied`/`uncertain`/`flagged_for_review`/`insufficient_info`)** | A bare pass/fail reads as an eligibility verdict, which this template must never issue (Safety Boundary) |
| Disclaimer placement | Domain `OutputFormatNode` | Outer `PostProcessNode` (output gate) | **Outer `PostProcessNode`** | Non-suppressible: the output gate is the one node every response passes through unconditionally; a domain-node placement could be silently skipped by a future change |
| Negation handling | Bare substring match | Negation-window-aware match | **Negation-window-aware** | Clinical narrative routinely negates criteria ("no prior chemo"); a bare match would misread that as positive evidence — a genuine safety gap for THIS domain |
| KB storage | External vector store | Seeded JSON KB | **Seeded JSON KB (v1)** | Self-contained, deterministic CI; the retrieval contract (`matched_criteria` JSON) is store-agnostic for a later vector-store upgrade |
| Criterion capping | top_k cap (generic KB-search precedent) | No cap — every criterion of the resolved trial is always assessed | **No cap** | Dropping a criterion (e.g. an exclusion ground) to satisfy a relevance cap is a patient-safety gap, not a UX simplification |
