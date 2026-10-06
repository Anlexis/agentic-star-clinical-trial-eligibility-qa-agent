# PB-6 - Invoke-Order Boundary: a full agent.invoke() must execute the fixed
# AgentBaseGraph backbone in order.
#
# The Cat 1 backbone is fixed and is NEVER overridden by a Cat 2 template
# (add_edges() belongs to the framework):
#
#     START -> initialize -> pre_process -> main -> {route} -> post_process
#           -> finalize -> END
#
# The framework records every executed node in `node_history` (an AgentState
# field whose reducer is operator.add, so entries accumulate in execution
# order). Each entry is the node's CLASS NAME - appended by BaseNode.__call__.
#
# For HCR-C2-002 (Cat 2, two-layer nested) the `main` slot is a GraphNode
# subclass (EligibilityScreeningGraphNode) that delegates to the inner
# DomainWorkflowGraph. The inner graph runs with its own state; its inner
# node_history is NOT merged back into the outer state (merge_output() maps
# only eligibility_assessment_result / result / citations / criterion_assessment
# / overall_abstain / status), so the OUTER node_history contains exactly the
# five backbone slots - never the inner domain nodes.
#
# This test drives a real end-to-end Graph().invoke() over the standard
# deploy payload and asserts the surfaced node_history matches the canonical
# backbone order. A SUCCESS terminal status is required: on any non-SUCCESS
# status route() short-circuits main -> finalize and the post_process (output
# gate) slot is skipped, which is itself an invoke-order violation this test
# would catch.
#
# Trust context: InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL)
# - the manifest's declared caller level. for_internal() is NEVER used here:
# it would over-privilege the run and hide trust-gate regressions on the outer gate.
#
# ⚠️ PAYLOAD / PII-MASK INTERACTION: before wiring this
# payload, the framework masking behaviour was checked EMPIRICALLY against the
# installed SDK's shared.security.pii_detector.detect_pii() rather than
# reasoned about — see test_payload_is_not_masked_by_the_s2_gate below, which
# pins the result as a regression test. Every capitalised span in
# _VALID_PAYLOAD is either a single ALL-CAPS token (XYZ, II, NSCLC, ECOG) or a
# sentence-initial word not followed by a second Title-Case word, so the
# framework's `name` pattern (`[A-Z][a-z]{1,20}(?:\s[A-Z][a-z]{1,20})+` - each
# word needs >=1 lowercase letter) finds no 2-word run to match; no digit
# span is SSN/phone/credit-card/My-Number shaped. detect_pii() confirms ZERO
# findings - the payload is genuinely PII-clean, not masked-then-coincidentally-ok.
# The pin lives in test_payload_is_not_masked_by_the_input_gate below.
#
# docs/03_test_spec.md section 4 (PoB).
# Deterministic - no LLM, no network. framework.* / src.* imports only.
from typing import ClassVar
from framework.nodes.base_node import BaseNode

import json
import pathlib

from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel
from shared.security.pii_detector import detect_pii

from src.graph.graph import Graph
from src.nodes.post_process_node import _PI_REVIEW_DISCLAIMER

# --- TEMPLATE-SPECIFIC ------------------------------------------------------
# The `main`-slot GraphNode class name for THIS template. A sibling template
# mirroring this canonical changes ONLY this one entry (its own domain
# <...>GraphNode); the other four backbone slot names are framework-fixed
# and identical across every Cat 1 / Cat 2 template.
_MAIN_SLOT_NODE = "EligibilityScreeningGraphNode"

# The valid, PII-free domain payload that drives the full eligibility
# pre-screen workflow to a SUCCESS terminal status. MUST stay byte-equal to
# the `input` field of deploy/invoke_payload.json (the standard deploy
# payload) - enforced by test_payload_matches_deploy_invoke_payload below.
_VALID_PAYLOAD = (
    "Eligibility pre-screen request against trial XYZ-123: 68-year-old adult "
    "patient with histologically confirmed Stage II NSCLC, ECOG performance "
    "status 1, no prior chemotherapy, no known brain metastases, adequate "
    "organ function on recent labs."
)

_DEPLOY_PAYLOAD_PATH = pathlib.Path(__file__).resolve().parents[2] / "deploy" / "invoke_payload.json"
# --- END TEMPLATE-SPECIFIC --------------------------------------------------

# Canonical AgentBaseGraph backbone execution order, by node class name as
# recorded in node_history. Four entries are framework-fixed and
# identical for every template; only _MAIN_SLOT_NODE is template-specific.
_EXPECTED_ORDER = [
    "InitializeNode",  # framework default  (initialize slot)
    "PreProcessNode",  # standard           (pre_process slot, input gates)
    _MAIN_SLOT_NODE,  # TEMPLATE-SPECIFIC  (main slot GraphNode)
    "PostProcessNode",  # standard           (post_process slot, output gate)
    "FinalizeNode",  # framework default  (finalize slot)
]


class _PrivilegedTrustGateFixture(BaseNode):
    """Always-present privileged node used to prove the S-1 negative boundary."""

    required_trust_level: ClassVar[TrustLevel] = TrustLevel.VERIFIED_EXTERNAL

    def _security_gate_input(self, state):
        return state

    def execute(self, state):
        return {"status": "success"}

    def _security_gate_output(self, result):
        return result


def _trust_predecessor(required: TrustLevel) -> TrustLevel:
    """Return a lower valid trust level; fail loudly if the framework adds one."""
    predecessors = {
        TrustLevel.VERIFIED_EXTERNAL: TrustLevel.ANONYMOUS,
        TrustLevel.INTERNAL: TrustLevel.VERIFIED_EXTERNAL,
    }
    try:
        return predecessors[required]
    except KeyError as exc:
        raise AssertionError(f"no lower trust level defined for {required!r}") from exc


def _run() -> dict:
    """Run a full end-to-end invocation at the manifest's declared trust level."""
    ctx = InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL, caller_id="pb6-suite")
    return Graph().invoke(_VALID_PAYLOAD, ctx=ctx)


class TestPayloadPreconditions:
    def test_payload_matches_deploy_invoke_payload(self):
        """_VALID_PAYLOAD must be byte-equal to deploy/invoke_payload.json's
        `input` - PB-6 proves invoke-order for the SAME payload deployments verify."""
        deployed = json.loads(_DEPLOY_PAYLOAD_PATH.read_text(encoding="utf-8"))
        assert _VALID_PAYLOAD == deployed["input"]

    def test_payload_is_not_masked_by_the_input_gate(self):
        """Empirical pin (not reasoned-about): the framework's PII scan finds
        NOTHING in _VALID_PAYLOAD, so the backbone-order assertions below are
        exercising the payload verbatim, not a [MASKED] variant of it."""
        assert detect_pii(_VALID_PAYLOAD) == []

    def test_s1_denial_refuses_execution_before_execute(self, monkeypatch):
        """TC-08: an always-present privileged node proves the negative S-1 path."""
        import framework.nodes.base_node as base_node_module

        events: list[str] = []
        execute_calls: list[object] = []
        monkeypatch.setattr(
            base_node_module,
            "emit_trace_event",
            lambda event_type, _payload, _state: events.append(event_type),
        )
        original_execute = _PrivilegedTrustGateFixture.execute

        def spy_execute(self, state):
            execute_calls.append(state)
            return original_execute(self, state)

        monkeypatch.setattr(_PrivilegedTrustGateFixture, "execute", spy_execute)
        result = _PrivilegedTrustGateFixture()(
            {
                "caller_trust_level": _trust_predecessor(_PrivilegedTrustGateFixture.required_trust_level).value,
                "correlation_id": "tc08-s1-denial",
            }
        )

        assert result["status"] == "error"
        assert "S-1 trust gate denied" in result["error_log"][0]
        assert events == ["s1_denied"]
        assert not execute_calls


class TestInvokeOrderBoundary:
    """PB-6: full agent.invoke() executes the backbone in the fixed order."""

    def test_invoke_reaches_success(self):
        """The full run must terminate SUCCESS - otherwise route() short-circuits
        main -> finalize and the post_process (output gate) slot never runs."""
        result = _run()
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected SUCCESS, got {result.get('status')!r}. result={result!r}"

    def test_output_is_non_empty(self):
        """A successful run must surface a non-empty gated output."""
        assert _run().get("output"), "invoke() surfaced an empty output"

    def test_output_carries_the_mandatory_pi_review_disclaimer(self):
        """SAFETY BOUNDARY: the non-suppressible disclaimer must ride with
        every successful text output (docs/02_design.md)."""
        output = _run().get("output", "")
        assert _PI_REVIEW_DISCLAIMER in output

    def test_node_history_is_populated(self):
        """node_history must be a non-empty list of node class-name strings."""
        history = _run().get("node_history")
        assert isinstance(history, list) and history, f"node_history must be a non-empty list, got {history!r}"
        assert all(isinstance(n, str) for n in history), f"node_history entries must be strings, got {history!r}"

    def test_backbone_slot_order(self):
        """Core invoke-order boundary: the pre_process gate slot runs before the
        domain main slot, which runs before the post_process gate slot - as a
        strict ordered subsequence of node_history."""
        history = _run().get("node_history", [])
        ordered_slots = ["PreProcessNode", _MAIN_SLOT_NODE, "PostProcessNode"]
        for name in ordered_slots:
            assert name in history, f"Expected backbone slot {name!r} in node_history, got {history!r}"
        positions = [history.index(name) for name in ordered_slots]
        assert positions == sorted(positions), (
            f"Backbone slots executed out of order: {ordered_slots} at {positions}. " f"node_history={history!r}"
        )

    def test_full_backbone_sequence(self):
        """The complete AgentBaseGraph backbone order:
        initialize -> pre_process -> main -> post_process -> finalize."""
        history = _run().get("node_history", [])
        assert history == _EXPECTED_ORDER, (
            "node_history does not match the canonical backbone order.\n"
            f"  expected: {_EXPECTED_ORDER}\n"
            f"  actual:   {history}"
        )


class TestOuterSecurityBoundaryOnTheSamePayload:
    """Trust gate at the graph level: an ANONYMOUS invoke of the SAME payload
    is refused by the VERIFIED_EXTERNAL pre_process slot, and the output gate
    slot never runs - the mirror boundary case to the SUCCESS path above."""

    def test_anonymous_caller_is_denied_and_post_process_never_runs(self):
        ctx = InvocationContext(caller_trust_level=TrustLevel.ANONYMOUS, caller_id="pb6-anon")
        result = Graph().invoke(_VALID_PAYLOAD, ctx=ctx)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        assert "PostProcessNode" not in result.get("node_history", [])
