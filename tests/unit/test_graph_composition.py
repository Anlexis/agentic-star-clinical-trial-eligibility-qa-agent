# HCR-C2-002 — Unit Tests: nested Cat-2 graph composition (outer + end-to-end)
#
# Drives the REAL outer agent (ClinicalTrialEligibilityQAAgent / Graph)
# end-to-end via AgentBaseGraph.invoke(). The e2e context is
# InvocationContext(caller_trust_level=TrustLevel.VERIFIED_EXTERNAL) — the
# manifest's declared caller level; for_internal() is NEVER used (it would
# over-privilege the run and hide trust-gate regressions).
#
# Mirrors docs/03_test_spec.md §3 (INT-05..INT-12).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

import pathlib

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.schemas.agent_status import AgentStatus
from framework.schemas.invocation_context import InvocationContext
from framework.schemas.trust_level import TrustLevel

import src.graph.graph
from src.graph.domain_workflow_graph import DomainWorkflowGraph
from src.graph.graph import (
    ClinicalTrialEligibilityQAAgent,
    EligibilityScreeningGraphNode,
    Graph,
)
from src.nodes.post_process_node import PostProcessNode, _PI_REVIEW_DISCLAIMER
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json, to_json

_VALID_PAYLOAD = (
    "Eligibility pre-screen request against trial XYZ-123: 68-year-old adult "
    "patient with histologically confirmed Stage II NSCLC, ECOG performance "
    "status 1, no prior chemotherapy, no known brain metastases, adequate "
    "organ function on recent labs."
)


def _run(user_input: str, trust: TrustLevel = TrustLevel.VERIFIED_EXTERNAL) -> dict:
    ctx = InvocationContext(caller_trust_level=trust, caller_id="unit-suite")
    return Graph().invoke(user_input, ctx=ctx)


class TestOuterGraphConstruction:
    def test_int_05_inherits_agent_base_graph_directly(self):
        assert issubclass(ClinicalTrialEligibilityQAAgent, AgentBaseGraph)

    def test_int_05_graph_alias(self):
        assert Graph is ClinicalTrialEligibilityQAAgent

    def test_state_schema_is_state(self):
        assert ClinicalTrialEligibilityQAAgent().state_schema is State

    def test_int_06_compile_fills_all_backbone_slots(self):
        agent = ClinicalTrialEligibilityQAAgent()
        agent.compile()
        for slot in ("initialize", "pre_process", "main", "post_process", "finalize"):
            assert agent._nodes.get(slot) is not None, f"backbone slot not filled: {slot}"
        assert isinstance(agent._nodes["pre_process"], PreProcessNode)
        assert isinstance(agent._nodes["main"], EligibilityScreeningGraphNode)
        assert isinstance(agent._nodes["post_process"], PostProcessNode)

    def test_add_edges_is_not_overridden(self):
        # Backbone wiring belongs to the framework — the template must not
        # redefine it.
        assert "add_edges" not in ClinicalTrialEligibilityQAAgent.__dict__


class TestMainSlotGraphNode:
    def test_int_07_get_subgraph_returns_the_inner_graph(self):
        subgraph = EligibilityScreeningGraphNode().get_subgraph()
        assert isinstance(subgraph, DomainWorkflowGraph)
        assert subgraph.config["configurable"]["retrieval"], "inner config must carry the retrieval block"

    def test_int_08_extract_input_prefers_validated_input(self):
        node = EligibilityScreeningGraphNode()
        assert node.extract_input({"validated_input": "VI", "user_input": "UI"}) == "VI"
        assert node.extract_input({"user_input": "UI"}) == "UI"

    def test_int_09_merge_output_maps_the_inner_contract(self):
        node = EligibilityScreeningGraphNode()
        citations = to_json(
            [{"ref": 1, "id": "xyz123-inc-01", "trial_id": "XYZ-123", "category": "age", "source": "s"}]
        )
        criterion_assessment = to_json([{"id": "xyz123-inc-01", "status": "appears_satisfied"}])
        delta = node.merge_output(
            {},
            {
                "formatted_answer": "ANSWER",
                "citations": citations,
                "criterion_assessment": criterion_assessment,
                "overall_abstain": False,
                "status": AgentStatus.SUCCESS.value,
            },
        )
        # The inner formatted_answer surfaces as BOTH eligibility_assessment_result
        # and result (PostProcessNode's output gate reads state["result"]).
        assert delta == {
            "eligibility_assessment_result": "ANSWER",
            "result": "ANSWER",
            "citations": citations,
            "criterion_assessment": criterion_assessment,
            "overall_abstain": False,
            "status": AgentStatus.SUCCESS.value,
            # Always present so a reason can never be dropped at the boundary.
            "error_code": "",
        }

    def test_error_strategy_is_propagate_and_hitl_is_contained(self):
        assert EligibilityScreeningGraphNode.error_strategy == "propagate"
        assert EligibilityScreeningGraphNode.propagate_hitl is False

    def test_int_10_parent_config_never_empty_without_config_file(self, monkeypatch):
        # Even with an unreadable config/config.yaml the forwarded config
        # carries the fallback retrieval/llm blocks — never {}.
        monkeypatch.setattr(src.graph.graph, "_RUNTIME_CONFIG_PATH", pathlib.Path("/nonexistent/config.yaml"))
        cfg = EligibilityScreeningGraphNode()._parent_config()
        assert cfg["configurable"]["retrieval"]["kb_path"] == "config/kb/clinical_trials_kb.json"
        assert cfg["configurable"]["llm"]


class TestEndToEndInvoke:
    """Full agent run: outer backbone + inner domain workflow, no LLM."""

    def test_int_11_invoke_returns_success(self):
        result = _run(_VALID_PAYLOAD)
        assert (
            result.get("status") == AgentStatus.SUCCESS.value
        ), f"Expected success, got {result.get('status')}. result={result!r}"

    def test_int_11_output_is_the_gated_formatted_answer(self):
        output = _run(_VALID_PAYLOAD).get("output")
        assert isinstance(output, str) and output.strip()
        assert output.startswith("# Clinical Trial Eligibility Pre-Screen")
        assert "[1]" in output
        assert _PI_REVIEW_DISCLAIMER in output

    def test_int_11_e2e_traverses_the_post_process_gate(self):
        history = _run(_VALID_PAYLOAD).get("node_history", [])
        for cls_name in ("PreProcessNode", "EligibilityScreeningGraphNode", "PostProcessNode"):
            assert cls_name in history, f"node_history missing {cls_name}: {history}"

    def test_int_11_structured_fields_surfaced_on_success(self):
        result = _run(_VALID_PAYLOAD)
        assert len(result.get("criterion_assessment", [])) == 7
        assert len(result.get("citations", [])) == 7
        assert result.get("overall_abstain") is False
        assert result.get("advisory_notice")

    def test_no_coverage_query_still_terminates_success_and_abstains(self):
        result = _run("no seeded trial protocol can be identified from this free text")
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("overall_abstain") is True
        assert "too short/sparse" in result.get("output", "")

    def test_int_12_anonymous_caller_is_denied_at_the_outer_boundary(self):
        """Trust gate at graph level: an ANONYMOUS invoke is refused by the
        VERIFIED_EXTERNAL pre_process slot. The error state short-circuits the
        main slot (its input gate sees status=error and skips the inner graph)
        and routes past post_process to finalize — no domain answer is ever
        produced, and NO structured field is surfaced."""
        result = _run(_VALID_PAYLOAD, trust=TrustLevel.ANONYMOUS)
        assert result.get("status") == AgentStatus.ERROR.value
        assert not result.get("output")
        assert "criterion_assessment" not in result
        history = result.get("node_history", [])
        assert "PostProcessNode" not in history
        assert history[:2] == ["InitializeNode", "PreProcessNode"]


class TestStateRoundTrip:
    """JSON-string state helpers: producers to_json() on write, consumers from_json()."""

    def test_to_from_json_list_round_trip(self):
        original = [{"id": "xyz123-inc-01", "score": 1.0, "status": "appears_satisfied"}]
        assert from_json(to_json(original)) == original

    def test_to_from_json_dict_round_trip(self):
        original = {"trial_id": "XYZ-123"}
        assert from_json(to_json(original)) == original

    def test_to_json_none_passes_through(self):
        assert to_json(None) is None

    def test_from_json_malformed_returns_default(self):
        assert from_json("{not valid json", default=[]) == []
        assert from_json(None, default={}) == {}
        assert from_json("", default=[]) == []
