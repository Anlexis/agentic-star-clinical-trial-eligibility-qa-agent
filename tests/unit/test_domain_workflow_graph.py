# HCR-C2-002 — Unit Tests: DomainWorkflowGraph (inner BaseGraph)
#
# Inner-graph composition + a full inner invoke() over the seeded KB. The
# inner graph runs the 5 domain nodes (all ANONYMOUS) — the outer trust boundary
# is the AgentBaseGraph backbone's concern and is covered in
# test_graph_composition.py / the PoB suite.
#
# Mirrors docs/03_test_spec.md §3 (INT-01..INT-04).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from langgraph.graph import END

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_status import AgentStatus

from src.graph.context_bridge import set_caller_input_context
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import EligibilityScreeningGraphNode
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, from_json

_VALID_PAYLOAD = (
    "Eligibility pre-screen request against trial XYZ-123: 68-year-old adult "
    "patient with histologically confirmed Stage II NSCLC, ECOG performance "
    "status 1, no prior chemotherapy, no known brain metastases, adequate "
    "organ function on recent labs."
)


class TestInnerGraphConstruction:
    def test_int_01_inherits_base_graph(self):
        assert issubclass(DomainWorkflowGraph, BaseGraph)

    def test_int_01_registers_the_five_domain_nodes(self):
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert set(inner._nodes.keys()) == {
            "input_validate",
            "retrieve",
            "rerank_filter",
            "generate_answer",
            "output_format",
        }
        assert isinstance(inner._nodes["input_validate"], InputValidateNode)
        assert isinstance(inner._nodes["retrieve"], RetrieveNode)
        assert isinstance(inner._nodes["rerank_filter"], RerankFilterNode)
        assert isinstance(inner._nodes["generate_answer"], GenerateAnswerNode)
        assert isinstance(inner._nodes["output_format"], OutputFormatNode)

    def test_inner_graph_name_and_schema(self):
        inner = DomainWorkflowGraph()
        assert inner.name == "hcr_c2_002_eligibility_screening_workflow"
        assert inner.state_schema is State

    def test_initialize_finalize_are_not_registered(self):
        # Outer backbone concerns must not leak into the inner topology.
        inner = DomainWorkflowGraph()
        inner.register_nodes()
        assert "initialize" not in inner._nodes
        assert "finalize" not in inner._nodes


class TestConfigForwarding:
    def test_int_02_extra_initial_state_republishes_retrieval_block(self):
        inner = DomainWorkflowGraph(config={"configurable": {"retrieval": {"score_threshold": 0.5}}})
        extra = inner._extra_initial_state()
        assert set(extra.keys()) == {"retrieval_config", "input_context"}
        assert isinstance(extra["retrieval_config"], str)  # JSON string, never a bare dict
        assert from_json(extra["retrieval_config"]) == {"score_threshold": 0.5}

    def test_extra_initial_state_with_no_config_is_empty_block(self):
        assert from_json(DomainWorkflowGraph()._extra_initial_state()["retrieval_config"]) == {}

    def test_extra_initial_state_seeds_the_bridged_input_context(self):
        # GraphNode does not forward input_context to the inner invoke (SDK
        # 1.0.1); extract_input() stashes it on the context bridge and
        # _extra_initial_state() reads it back into inner state.
        set_caller_input_context({"trial_id": "XYZ-123"})
        try:
            extra = DomainWorkflowGraph()._extra_initial_state()
            assert extra["input_context"] == {"trial_id": "XYZ-123"}
        finally:
            set_caller_input_context({})

    def test_int_02b_config_yaml_values_reach_inner_state_end_to_end(self):
        # PROBE (not a declaration check): the repo's real config/config.yaml
        # values must reach the inner graph's seeded retrieval_config through
        # the same path the outer GraphNode uses at runtime.
        import yaml

        declared = yaml.safe_load((pathlib.Path(__file__).resolve().parents[2] / "config" / "config.yaml").read_text())[
            "retrieval"
        ]
        inner = DomainWorkflowGraph(config=EligibilityScreeningGraphNode()._parent_config())
        seeded = from_json(inner._extra_initial_state()["retrieval_config"])
        assert seeded == declared
        assert seeded["score_threshold"] == 0.75
        assert seeded["abstain_min_chars"] == 25


class TestOutputShape:
    def test_int_03_get_output_shapes_the_merge_contract(self):
        inner = DomainWorkflowGraph()
        out = inner.get_output(
            {
                "formatted_answer": "ANSWER",
                "citations": "[]",
                "criterion_assessment": "[]",
                "overall_abstain": False,
                "status": AgentStatus.SUCCESS.value,
                "node_history": ["InputValidateNode"],
            }
        )
        assert out["formatted_answer"] == "ANSWER"
        assert out["citations"] == "[]"
        assert out["criterion_assessment"] == "[]"
        assert out["overall_abstain"] is False
        assert out["status"] == AgentStatus.SUCCESS.value
        assert out["node_history"] == ["InputValidateNode"]

    def test_route_returns_end_on_error(self):
        inner = DomainWorkflowGraph()
        assert inner.route({"status": AgentStatus.ERROR.value}) == END
        assert inner.route({"status": AgentStatus.SUCCESS.value}) == "output_format"


class TestInnerEndToEnd:
    def _invoke(self, payload: str) -> dict:
        # Same construction path the outer GraphNode uses: config/config.yaml-
        # derived config via _parent_config(); domain nodes take NO ctor args.
        inner = DomainWorkflowGraph(config=EligibilityScreeningGraphNode()._parent_config())
        return inner.invoke(payload, session_id="inner-e2e")

    def test_int_04_full_inner_run_produces_the_formatted_answer(self):
        result = self._invoke(_VALID_PAYLOAD)
        assert result["status"] == AgentStatus.SUCCESS.value
        answer = result["formatted_answer"]
        assert answer.startswith("# Clinical Trial Eligibility Pre-Screen")
        assert "[1]" in answer
        citations = from_json(result["citations"])
        assert len(citations) == 7
        assessment = from_json(result["criterion_assessment"])
        assert len(assessment) == 7
        assert result["overall_abstain"] is False

    def test_int_04_inner_node_history_is_the_linear_topology(self):
        history = self._invoke(_VALID_PAYLOAD)["node_history"]
        assert history == [
            "InputValidateNode",
            "RetrieveNode",
            "RerankFilterNode",
            "GenerateAnswerNode",
            "OutputFormatNode",
        ]

    def test_sparse_query_still_terminates_success_but_abstains(self):
        result = self._invoke("no seeded trial protocol can be identified from this free text")
        assert result["status"] == AgentStatus.SUCCESS.value
        assert result["overall_abstain"] is True
        assert "too short/sparse" in result["formatted_answer"]
