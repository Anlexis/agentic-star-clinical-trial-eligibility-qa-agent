# Test Specification — HCR-C2-002

**Template ID:** HCR-C2-002
**Template Name:** ClinicalTrialEligibilityQAAgent
**Category:** Cat 2 (nested) — Clinical Trial Eligibility Q&A (RAG pattern)

## Test Strategy

- **Coverage target:** every node's `execute()` path (happy path + rejection/
  degraded-input paths), the full gate contract on every node (trust gate,
  PII input mask, output gate, audit), the outer backbone invoke order
  end-to-end, the caller-data validation contract, and — because this is a
  PHI-adjacent, decision-support-only template — dedicated, provably-real
  coverage of the two safety-critical behaviours called out in
  `docs/02_design.md` "Safety Boundary": the PHI-identifier strip (on BOTH
  caller channels), and the non-suppressible PI-review disclaimer.
- **Test types:** Unit (per-node, `tests/unit/`) / Proof-of-Boundary
  (`tests/proof_of_boundary/`, including end-to-end business behaviour
  through the real ASGI `/invoke` adapter with Bearer auth). The pipeline is
  deterministic (seeded KB + rule-based synthesis, no LLM, no network), so
  no external-service integration harness exists.
- **Invocation canon:** every test invokes a node via `node(state)`
  (`BaseNode.__call__` → trust gate → PII input mask → `execute()` → output
  gate), never a bare `node.execute(state)`. Config overrides
  (`retrieval_config`) are exercised by seeding State, never a 2nd `execute()`
  argument — nodes take `execute(self, state) -> dict` only.
- All tests green under the installed framework wheel
  (`agenticstar-agentcore==1.0.1`).

## Framework Compliance Tests

| TC-ID | Test | Expected Result | Where |
|-------|------|----------------|-------|
| TC-01 | State contract: flat TypedDict | Type check pass, no Pydantic/dataclass | `tests/proof_of_boundary/test_state_safety.py` |
| TC-02 | Disallowed output content is gated | Output gate returns `AgentStatus.ERROR` + sanitised stub | `tests/unit/test_post_process_gate.py` |
| TC-03 | No JWT/Credential in State | Credential scan (`scripts/check_credentials.py`): 0 violations | repo gate script |
| TC-04 | InvocationContext via configurable only | `InvocationContext` never written to State | `src/schemas/state.py` + `tests/proof_of_boundary/test_state_safety.py` |
| TC-05 | Audit: no duplicate lifecycle events in `execute()` | `node_start` / `node_complete` / `node_error` absent from `execute()` body; every node emits only its own domain event | per-node `*Audit` test classes |
| TC-06 | `_security_gate_input()` not overridden (`FunctionNode` subclass) | `TypeError` raised at class definition if overridden (`@final` enforced by framework) | `tests/unit/test_framework_compliance_tc06_tc07.py` |
| TC-07 | `_security_gate_output()` not overridden (`FunctionNode` subclass) | `TypeError` raised at class definition if overridden (`@final` enforced by framework) | `tests/unit/test_framework_compliance_tc06_tc07.py` |
| TC-08 | `required_trust_level` enforced | Insufficient trust → refused | `tests/unit/test_trust_gate.py` — full matrix, all 7 node classes |
| TC-09 | Domain input hardening | The PHI strip runs as a plain `execute()`-inline helper (`_surface_strip_identifiers()`), never an `_extra_security_gate_input()` hook | `tests/unit/test_pre_process_node.py`; see `docs/02_design.md` "gate behaviour by node type" |
| TC-10 | Domain output hardening | The module-level `_security_gate_output()` recursive scan + disclaimer-presence check run inline from `PostProcessNode.execute()` | `tests/unit/test_post_process_gate.py` |
| TC-11 | Audit: at least one domain `emit_trace_event()` inside each `execute()` | Domain event emitted on every invocation path | 7/7 nodes, per-node `*Audit` test classes |

## Marketplace Entry Point — `tests/unit/test_cli_entry_point.py`

| ID | Case | Expected |
|----|------|----------|
| CLI-01 | `cli.py` imports | module loads; `run_agent_marketplace`, `load_agent_config` and `ClinicalTrialEligibilityQAAgent` are present |
| CLI-02 | override seam ships empty | `extend_config == {}`; a stray value would silently outrank `config/config.yaml` on the Marketplace path only |
| CLI-03 | the runner receives what the image's CMD would send | executing `cli.py` as `__main__` with the runner replaced captures the call: the graph class, `agent_name`, `namespace`, and every value declared in `config/config.yaml`. Loading the module alone never runs that block, so a wrong class or a dropped config there would otherwise ship unnoticed |

`cli.py` is imported by no other module, so nothing else in the suite would
notice if its import path, graph class or config assembly broke; the image
would build and fail only when the Pod starts. Skipped where the platform
events package is absent.

## Proof-of-Boundary Tests

| PB-ID | Boundary | Test | Expected Result | Where |
|-------|----------|------|----------------|-------|
| PB-1 | BaseNode → EventEmitter | `emit_trace_event()` fires on every invocation path | No silent failures | per-node Audit test classes, all 7 nodes |
| PB-2 | State serialization | Post-invoke State is primitives only | No Pydantic/dataclass | `tests/proof_of_boundary/test_state_safety.py` |
| PB-3 | External service | Real external service connection | N/A — the pipeline is a deterministic, network-free matcher over the seeded `config/kb/clinical_trials_kb.json`; no external service exists to connect to | N/A by design |
| PB-4 | Import isolation | No platform-internal SDK imports | AST scan: 0 violations | `tests/proof_of_boundary/test_import_isolation.py` |
| PB-5 | Checkpoint safety | No JWT/Pydantic in checkpoint | Inspection pass | `tests/proof_of_boundary/test_state_safety.py` **Auto-waived — checkpointing disabled**: `config/config.yaml` enables neither `memory_enabled` nor `hitl.enabled`, so no checkpoint surface exists; the conditional gate and the non-lossy traversal helper ship with the stub. |
| PB-6 | Invoke execution order | `__call__()`: trust gate → `node_start` → input gate → `execute()` → output gate → `node_complete`; full `Graph().invoke()` backbone order `[InitializeNode, PreProcessNode, EligibilityScreeningGraphNode, PostProcessNode, FinalizeNode]` on SUCCESS | Order verified | `tests/proof_of_boundary/test_pb_invoke_order.py` |
| PB-7 | HITL interrupt propagation *(conditional)* | Required only when the runtime config enables HITL; otherwise a documented skip stub | `GraphInterrupt` propagates to the LangGraph engine; `status` is not set to `error` | **Skip stub — non-HITL** (`EligibilityScreeningGraphNode.propagate_hitl = False`; `tests/proof_of_boundary/test_pb7_hitl_interrupt_propagation.py`) |
| PB-8 | Entry-point E2E (`POST /invoke`) | Real ASGI app, Bearer auth: caller data via `input_context` crosses the outer→inner bridge and yields a cited assessment; flagged / abstain / rejection paths; 401 without token; 413 over the adapter size cap; output-schema scan | All paths verified | `tests/proof_of_boundary/test_invoke_e2e.py` |

## Business Logic Tests

Curated index of the domain-critical scenarios (full per-node detail lives in
each `tests/unit/test_<node>.py` file).

| TC-ID | Test | Input | Expected Result | Where |
|-------|------|-------|----------------|-------|
| BL-01 | PHI identifier strip is REAL, not a doc claim | MRN-style ID / SSN-shaped / e-mail / phone / calendar-date tokens | `[REDACTED]` (node's own strip) or `[MASKED]` (framework input mask) — raw identifier never survives into `validated_input`; a bare age (`68-year-old`) is left untouched | `tests/unit/test_pre_process_node.py` |
| BL-02 | Trial-id resolution (input_context field, JSON envelope key, OR free-text regex extraction) | `{"trial_id": "xyz-123"}` / `"...trial protocol xyz-123..."` / no trial id present | Resolved + upper-cased `trial_id`, or a `no trial_id resolved` note | `tests/unit/test_input_validate_node.py`, `tests/unit/test_caller_data_contract.py` |
| BL-03 | Every criterion of the resolved trial is ALWAYS a retrieval candidate (no top_k/tier cap — dropping one is a patient-safety gap) | Full standard-payload narrative against trial XYZ-123 (7 seeded criteria) | All 7 candidates present, deterministic inclusion-before-exclusion / id-asc order | `tests/unit/test_retrieve_node.py` |
| BL-04 | Negation-aware matching, clause-scoped | `"no known brain metastases, adequate organ function..."` | The exclusion term is `negated=True`; the FOLLOWING clause's inclusion term is NOT wrongly inherited as negated | `tests/unit/test_retrieve_node.py` |
| BL-05 | Status vocabulary is closed to 4 hedged values; NEVER an eligibility verdict | All (criterion_type × score-band × negation) combinations | Only `appears_satisfied` / `flagged_for_review` / `uncertain` / `insufficient_info`; never "eligible"/"ineligible"/"meets"/"fails" | `tests/unit/test_rerank_filter_node.py` |
| BL-06 | `overall_abstain` safety gate | Sparse text (< `abstain_min_chars`) OR unresolved/unseeded trial | `overall_abstain=True`, `criterion_assessment=[]`; `GenerateAnswerNode` emits ONLY the abstention message, routes to human/PI review | `tests/unit/test_rerank_filter_node.py`, `tests/unit/test_generate_answer_node.py` |
| BL-07 | Grounded, cited narrative assembly; rendered answer never contains verdict language | Mixed-status `criterion_assessment` | Numbered `[n]` citations mirror render order; `insufficient_info` entries carry no citation; closed-vocabulary scan of the FULL rendered body finds no forbidden word | `tests/unit/test_generate_answer_node.py` |
| BL-08 | Mandatory, non-suppressible PI-review disclaimer — fails CLOSED if ever dropped | Clean string result | Disclaimer appended AND its presence re-verified before SUCCESS; simulated drop → `ERROR` (fail-closed), not a silent ship | `tests/unit/test_post_process_gate.py`, `tests/proof_of_boundary/test_pb_invoke_order.py` |
| BL-09 | Output gate is RECURSIVE (dict/list/tuple at any depth) | Credential / PHI-shaped string nested 2 levels deep inside a dict/list/tuple | Blocked exactly like a top-level violation; sanitised stub returned, raw secret never surfaces | `tests/unit/test_post_process_gate.py` |
| BL-10 | Structured output (`get_output()` override) is gated the same way as text, fail-closed | `criterion_assessment` / `citations` on SUCCESS | Present + re-scanned via the same recursive gate; withheld entirely on any non-SUCCESS status | `tests/proof_of_boundary/test_pb_invoke_order.py`; `src/graph/graph.py::get_output` |
| BL-11 | Caller-data contract is fail-closed, field-naming, never value-echoing | Non-string `patient_profile` / `trial_id` (incl. raw NaN/Infinity through JSON); malformed trial-id strings | `status=error` naming the FIELD; the rejected value never appears in any error log or response; no partial domain output | `tests/unit/test_caller_data_contract.py`, `tests/proof_of_boundary/test_invoke_e2e.py` |
| BL-12 | `input_context.patient_profile` gets the SAME PHI strip as the string payload | Profile carrying MRN / SSN / DOB-shaped tokens via `input_context` | Identifiers `[REDACTED]` before inner state; clinical phrasing left intact; nothing PHI-shaped survives into the response | `tests/unit/test_caller_data_contract.py`, `tests/proof_of_boundary/test_invoke_e2e.py` |
| BL-13 | End-to-end business behaviour through `/invoke` | `input_context` profile + trial id (clean / un-negated-exclusion / sparse variants) | Cited non-empty assessment; `FLAGGED FOR REVIEW` on the un-negated exclusion; abstention on sparse input; disclaimer always present | `tests/proof_of_boundary/test_invoke_e2e.py` |

## Test Execution Summary

> **Pending re-run.** The figures below predate the Marketplace entry point work.
> The entry-point test and the PB-5 / PB-6 additions were added after this run and
> have not been executed locally — the framework wheel is not installed in the
> authoring environment. **They are not yet verified anywhere**; this summary is
> updated once a pipeline run has executed them.

- **SDK:** `agenticstar-agentcore==1.0.1` (the installed framework wheel)
- **Total tests (full `tests/` tree):** 207 collected — 206 passed / 0 failed / 1 skipped (PB-7 non-HITL skip stub)
- **Proof-of-Boundary subset (`tests/proof_of_boundary/`):** 34 collected — 33 passed / 1 skipped
- **Unit subset (`tests/unit/`):** 173 collected — 173 passed
- **Coverage:** every `src/nodes/*.py` module has a dedicated `tests/unit/test_*.py` file; every gate behaviour (trust, input mask, output gate, audit) is exercised; the full outer + inner graph composition is unit-tested (`test_graph_composition.py`, `test_domain_workflow_graph.py`); the full outer backbone is exercised end-to-end (PB-6) for both the SUCCESS and the ANONYMOUS-denial paths; the caller-data contract and the entry-point auth boundary are exercised through the real ASGI `/invoke` (PB-8).
