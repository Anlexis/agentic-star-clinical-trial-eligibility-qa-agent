# HCR-C2-002 — Unit Tests: trust gate
#
# Trust-gate contract: tests must invoke nodes via node(state) — through
# BaseNode.__call__, which runs the trust gate → the PII input mask →
# execute() → the output gate — never via node.execute(state) directly,
# which bypasses the gate. A denial RETURNS an error dict (never raises)
# with status ERROR and "trust gate denied" in error_log; execute() never
# runs, so execute-only output keys are ABSENT from the returned dict.
#
# The nested Cat-2 shape (EligibilityScreeningGraphNode wraps the inner
# DomainWorkflowGraph's 5 domain nodes) has no standalone main-slot
# FunctionNode, so this suite exercises an inner ANONYMOUS domain node
# (InputValidateNode) for the allowed path, plus the full per-node matrix
# below.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.post_process_node import PostProcessNode
from src.nodes.pre_process_node import PreProcessNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import to_json


def _make_state(
    trust_value: str,
    user_input: str = "does a 68yo patient with stage II NSCLC and no prior chemo meet trial protocol XYZ-123 inclusion criteria?",
    **extra,
) -> dict:
    state = {
        "user_input": user_input,
        "caller_trust_level": trust_value,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestTrustGate:
    """Trust gate tests — all invocations go through node(state) / __call__."""

    def test_anonymous_caller_allowed_on_input_validate_node(self):
        """An ANONYMOUS caller passes the ANONYMOUS inner domain node
        (main slot delegates to the inner graph; InputValidateNode is its
        first node)."""
        node = InputValidateNode()  # required_trust_level = ANONYMOUS
        result = node(_make_state(TrustLevel.ANONYMOUS.value, validated_input="test query"))
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("patient_criteria_text") is not None

    def test_anonymous_caller_denied_on_pre_process(self):
        """Trust-gate rejection: ANONYMOUS caller on the VERIFIED_EXTERNAL PreProcessNode.

        __call__ must RETURN an error dict (never raise) with status ERROR and
        'trust gate denied' in the error_log. execute() never ran, so the
        execute-only output key (validated_input) must be ABSENT.
        """
        node = PreProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(_make_state(TrustLevel.ANONYMOUS.value))
        assert result.get("status") == AgentStatus.ERROR.value
        error_log = result.get("error_log", [])
        assert any(
            "trust gate denied" in str(e) for e in error_log
        ), f"Expected 'trust gate denied' in error_log, got: {error_log}"
        assert "validated_input" not in result, "execute() must not run on a trust-gate denial — validated_input leaked"

    def test_verified_external_caller_passes_pre_process(self):
        """A VERIFIED_EXTERNAL caller clears the pre_process gate and
        the node writes validated_input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("validated_input")

    def test_pre_process_empty_input_rejected_after_gate(self):
        """The trust gate passes, then the node's own input validation rejects empty input."""
        node = PreProcessNode()
        result = node(_make_state(TrustLevel.VERIFIED_EXTERNAL.value, user_input=""))
        assert result.get("status") == AgentStatus.SUCCESS.value
        # Completes carrying the reason, so the caller can
        # correct the value and send the request again.
        assert result.get("error_code")
        assert any("empty" in str(e) for e in result.get("error_log", []))

    def test_anonymous_caller_denied_on_post_process(self):
        """Trust-gate rejection on the other VERIFIED_EXTERNAL outer slot (post_process).

        The denial dict carries no execute-only key (formatted_output ABSENT).
        """
        node = PostProcessNode()  # required_trust_level = VERIFIED_EXTERNAL
        result = node(
            _make_state(
                TrustLevel.ANONYMOUS.value,
                result="a clean processed result",
            )
        )
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("trust gate denied" in str(e) for e in result.get("error_log", []))
        assert (
            "formatted_output" not in result
        ), "execute() must not run on a trust-gate denial — formatted_output leaked"

    def test_verified_external_caller_passes_post_process(self):
        """A VERIFIED_EXTERNAL caller clears the post_process gate.

        (post_process also appends the mandatory PI-review disclaimer —
        formatted_output stays truthy either way; the disclaimer-presence
        behaviour itself is covered in tests/unit/test_post_process_gate.py.)
        """
        node = PostProcessNode()
        result = node(
            _make_state(
                TrustLevel.VERIFIED_EXTERNAL.value,
                result="a clean processed result",
            )
        )
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output")

    # Beyond the InputValidateNode spot-check above, the remaining 4 inner
    # domain nodes are each proven to actually admit an ANONYMOUS caller
    # through __call__, not just declare the trust level in
    # TestTrustLevelMatrix below.

    def test_anonymous_caller_allowed_on_retrieve_node(self):
        node = RetrieveNode()
        state = _make_state(
            TrustLevel.ANONYMOUS.value,
            patient_criteria_text="adult patient with stage ii nsclc",
            query_filters=to_json({"trial_id": "XYZ-123"}),
        )
        result = node(state)
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("matched_criteria") is not None

    def test_anonymous_caller_allowed_on_rerank_filter_node(self):
        node = RerankFilterNode()
        state = _make_state(
            TrustLevel.ANONYMOUS.value,
            matched_criteria=to_json([]),
            patient_criteria_text="adult patient with stage ii nsclc",
        )
        result = node(state)
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert "overall_abstain" in result

    def test_anonymous_caller_allowed_on_generate_answer_node(self):
        node = GenerateAnswerNode()
        state = _make_state(
            TrustLevel.ANONYMOUS.value,
            criterion_assessment=to_json([]),
            overall_abstain=True,
        )
        result = node(state)
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("grounded_answer")

    def test_anonymous_caller_allowed_on_output_format_node(self):
        node = OutputFormatNode()
        state = _make_state(
            TrustLevel.ANONYMOUS.value,
            grounded_answer="a grounded answer",
            citations=to_json([]),
        )
        result = node(state)
        assert "trust gate denied" not in str(result.get("error_log", []))
        assert result.get("status") == AgentStatus.SUCCESS.value


class TestTrustLevelMatrix:
    """The backbone's declared trust matrix (config/agent.yaml required_trust_level).

    Outer S-gate slots (pre_process / post_process) require VERIFIED_EXTERNAL,
    matching the manifest's declared required_trust_level; the main slot
    delegates to the inner Cat-2 domain workflow, whose nodes all declare
    ANONYMOUS (never INTERNAL, never omitted; the external
    gate lives on the outer backbone).
    """

    def test_outer_gate_nodes_require_verified_external(self):
        assert PreProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL
        assert PostProcessNode.required_trust_level is TrustLevel.VERIFIED_EXTERNAL

    def test_inner_domain_nodes_admit_anonymous(self):
        """Every inner Cat-2 domain node declares TrustLevel.ANONYMOUS."""
        inner_nodes = (
            InputValidateNode,
            RetrieveNode,
            RerankFilterNode,
            GenerateAnswerNode,
            OutputFormatNode,
        )
        for node_cls in inner_nodes:
            assert node_cls.required_trust_level is TrustLevel.ANONYMOUS, (
                f"{node_cls.__name__} must declare TrustLevel.ANONYMOUS "
                f"(inner Cat-2 domain nodes never VERIFIED_EXTERNAL/INTERNAL)"
            )
