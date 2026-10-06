"""AgentCore Platform v1.0"""

# HCR-C2-002 - DomainWorkflowGraph (inner BaseGraph)
#
# This is the INNER graph for the Cat 2 two-layer nested architecture.
# It encapsulates the full clinical-trial-eligibility pre-screen workflow:
#
#   START -> input_validate -> retrieve -> rerank_filter
#         -> generate_answer -> output_format -> END
#
# Called by EligibilityScreeningGraphNode.get_subgraph() (graph.py).
# get_output() shapes the sub_result dict consumed by merge_output() there.
#
# Rules enforced:
#   - Inherits BaseGraph (fully custom topology - no forced backbone)
#   - Implements all 7 BaseGraph ABC methods
#   - register_nodes() does NOT call super() (abstract in BaseGraph)
#   - register_nodes() instantiates every domain node with NO ctor args
#   - Does NOT register initialize / finalize (outer backbone concerns)
#   - get_output() designed together with EligibilityScreeningGraphNode.merge_output()
#   - No platform-internal SDK imports
#   - Not placed under src/subagents/

from typing import Any

from langgraph.graph import END, START

from framework.graph.base_graph import BaseGraph
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import get_caller_input_context
from src.nodes.generate_answer_node import GenerateAnswerNode
from src.nodes.input_validate_node import InputValidateNode
from src.nodes.output_format_node import OutputFormatNode
from src.nodes.rerank_filter_node import RerankFilterNode
from src.nodes.retrieve_node import RetrieveNode
from src.schemas.state import State, to_json


class DomainWorkflowGraph(BaseGraph):
    """Inner domain workflow graph for HCR-C2-002.

    Inherits BaseGraph directly for a fully custom node topology.
    Called by EligibilityScreeningGraphNode.get_subgraph() in graph.py, which
    passes the config/config.yaml-derived config (`_parent_config()`) into
    the ctor.

    Pipeline (linear):
        START
          -> input_validate  (InputValidateNode)  - parse + normalise patient criteria
          -> retrieve        (RetrieveNode)       - match seeded trial-protocol criteria
          -> rerank_filter   (RerankFilterNode)   - classify statuses + abstain gate
          -> generate_answer (GenerateAnswerNode) - cited, criterion-grouped narrative
          -> output_format   (OutputFormatNode)   - final format (no disclaimer - see graph.py)
          -> END

    All nodes are FunctionNode subclasses returning partial-dict state updates.
    initialize / finalize are outer backbone concerns - not registered here.
    """

    # -- Identity --------------------------------------------------------------

    @property
    def name(self) -> str:
        """Unique identifier for this inner graph."""
        return "hcr_c2_002_eligibility_screening_workflow"

    @property
    def state_schema(self) -> type:
        """TypedDict subclass shared across inner and outer graph."""
        return State

    # -- Config validation -----------------------------------------------------

    def _validate_config(self) -> None:
        """Validate inner graph config before compilation.

        The forwarded `retrieval` block (score_threshold / kb_path /
        abstain_min_chars) is read per-call by the domain nodes with safe
        defaults, so absence is non-fatal. `max_retry` / `timeout_seconds`
        (from config/config.yaml, `timeout_s` mapped on the way through)
        must be positive integers WHEN present - a declared-but-broken
        runtime value fails loudly at compile time instead of silently
        degrading.
        """
        configurable = (self.config or {}).get("configurable", {})
        for key in ("max_retry", "timeout_seconds"):
            if key not in configurable:
                continue
            value = configurable[key]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                file_key = "timeout_s" if key == "timeout_seconds" else key
                raise ValueError(
                    f"DomainWorkflowGraph: config/config.yaml {file_key} " f"must be a positive integer, got {value!r}"
                )

    # -- Config + caller-context forwarding into state -------------------------

    def _extra_initial_state(self) -> dict[str, Any]:
        """Seed inner state with the runtime `retrieval` block + input_context.

        EligibilityScreeningGraphNode._parent_config() forwards the
        config/config.yaml blocks under config["configurable"]; this hook
        makes the `retrieval` block reachable by the domain nodes at runtime
        as the JSON-string state field `retrieval_config` (structured state
        fields are stored as JSON strings for checkpoint/msgpack safety).
        Every inner node reads a passed retrieval_config first and falls back
        to its own module defaults - there is no `config` parameter on
        execute() (config reaches nodes via ctor or State only).

        Also seeds the caller's input_context: GraphNode.execute() does not
        forward input_context on subgraph.invoke() (SDK 1.0.1);
        EligibilityScreeningGraphNode.extract_input() stashes it via the
        context bridge immediately before the inner invoke, and this hook
        (called by BaseGraph.invoke while building initial state) reads it
        back. Inner domain nodes keep their plain state["input_context"]
        reads.
        """
        retrieval = (self.config or {}).get("configurable", {}).get("retrieval") or {}
        return {
            "retrieval_config": to_json(retrieval),
            "input_context": get_caller_input_context(),
        }

    # -- Node registration -----------------------------------------------------

    def register_nodes(self) -> None:
        """Register all 5 domain nodes.

        No super() call - BaseGraph.register_nodes() is abstract.
        Do NOT register initialize or finalize; those are outer backbone
        concerns handled by AgentBaseGraph in graph.py.

        Every node is instantiated with NO constructor arguments - FunctionNode
        subclasses take no __init__; config flows in via State seeding only
        (_extra_initial_state() above), per the execute(self, state) -> dict
        contract.
        """
        self._nodes["input_validate"] = InputValidateNode()
        self._nodes["retrieve"] = RetrieveNode()
        self._nodes["rerank_filter"] = RerankFilterNode()
        self._nodes["generate_answer"] = GenerateAnswerNode()
        self._nodes["output_format"] = OutputFormatNode()

    # -- Edge wiring -----------------------------------------------------------

    def add_edges(self) -> None:
        """Wire the linear eligibility-screening domain topology.

        Each step passes its partial-dict output into the shared State.
        For this template the topology is intentionally linear - no
        conditional branching between domain nodes. route() is implemented
        as required by the ABC but add_conditional_edges() is not used.
        """
        self._sg.add_edge(START, "input_validate")
        self._sg.add_edge("input_validate", "retrieve")
        self._sg.add_edge("retrieve", "rerank_filter")
        self._sg.add_edge("rerank_filter", "generate_answer")
        self._sg.add_edge("generate_answer", "output_format")
        self._sg.add_edge("output_format", END)

    # -- Routing ---------------------------------------------------------------

    def route(self, state: AgentState) -> str:
        """Conditional routing - required by BaseGraph ABC.

        For this linear topology add_conditional_edges() is not used, so this
        method is never called at runtime. It is implemented to satisfy the ABC
        contract. Returns END on error so an unexpected call does not re-enter a
        processing node.
        """
        if state.get("status") == AgentStatus.ERROR.value:
            return str(END)
        return "output_format"

    # -- Output shape ----------------------------------------------------------

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Shape the output dict returned to the outer graph as sub_result.

        This dict is received by EligibilityScreeningGraphNode.merge_output() in
        graph.py as the `sub_result` argument. Both methods are designed
        together to guarantee field-name consistency:

            Inner get_output()  emits: "formatted_answer", "citations",
                                       "criterion_assessment", "overall_abstain",
                                       "status", ...
            Outer merge_output() reads: sub_result.get("formatted_answer"),
                                        sub_result.get("citations"),
                                        sub_result.get("criterion_assessment"),
                                        sub_result.get("overall_abstain"),
                                        sub_result.get("status")

        Additional fields (intake_notes, trace_id, correlation_id,
        node_history) are surfaced for observability / downstream extension.
        """
        return {
            "formatted_answer": state.get("formatted_answer"),
            "citations": state.get("citations"),
            "criterion_assessment": state.get("criterion_assessment"),
            "overall_abstain": state.get("overall_abstain"),
            "status": state.get("status"),
            # Carried explicitly: the boundary only moves the keys named here,
            # so a run that completed without an answer would otherwise arrive
            # at the outer graph indistinguishable from one that answered.
            "error_code": state.get("error_code"),
            "intake_notes": state.get("intake_notes"),
            "trace_id": state.get("trace_id"),
            "correlation_id": state.get("correlation_id"),
            "node_history": state.get("node_history", []),
        }
