"""AgentCore Platform v1.0"""

# HCR-C2-002 - Outer graph (AgentBaseGraph; Cat 2 two-layer nested architecture)
#
# Clinical Trial Eligibility Q&A Agent (Cat 2 RAG decision-support pre-screen
# workflow).
#
# Architecture (Cat 2):
#
#   Outer backbone (fixed - identical to Cat 1, do NOT override add_edges()):
#     START -> initialize -> pre_process -> main -> {route} -> post_process -> finalize -> END
#                                             |  (RETRY, max 3)
#                                             -> pre_process
#
#   `main` slot is a GraphNode subclass (EligibilityScreeningGraphNode) that
#   delegates the full eligibility pre-screen domain workflow to
#   DomainWorkflowGraph (inner BaseGraph: input_validate -> retrieve ->
#   rerank_filter -> generate_answer -> output_format).
#
#   Domain complexity is fully encapsulated inside the inner graph. The outer
#   backbone is never modified.
#
# Directory layout:
#   src/graph/graph.py                 <- outer graph (this file)
#   src/graph/domain_workflow_graph.py <- inner graph (multi-step topology)
#   src/graph/context_bridge.py        <- input_context outer->inner hand-off
#
# Class-name contract:
#   graph.py class:           ClinicalTrialEligibilityQAAgent (this file)
#   config/agent.yaml class:  "src.graph.graph.ClinicalTrialEligibilityQAAgent"  <- must match
#   src/api/server.py import: from src.graph.graph import ClinicalTrialEligibilityQAAgent
#
# Rules enforced:
#   - ClinicalTrialEligibilityQAAgent inherits AgentBaseGraph (framework base
#     class - direct inheritance)
#   - super().register_nodes() called first (fills initialize + finalize)
#   - EligibilityScreeningGraphNode assigned to self._nodes["main"]
#   - _parent_config() forwards the config/config.yaml runtime blocks (never {})
#   - merge_output() returns only changed keys
#   - add_edges() NOT overridden on the outer graph
#   - No platform-internal SDK imports
#
# Structured product: get_output() is overridden below to surface the
# per-criterion criterion_assessment + citations + overall_abstain to a
# programmatic caller, on SUCCESS only, fail-closed (re-scanned via the same
# recursive _security_gate_output() PostProcessNode uses).
#
# SAFETY BOUNDARY: this template is decision-support only. It never emits an
# enrolment decision or a definitive eligible/ineligible verdict; see
# docs/02_design.md "Safety Boundary" for the full statement.

from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from framework.graph.agent_base_graph import AgentBaseGraph
from framework.nodes.graph_node import GraphNode
from framework.schemas.agent_state import AgentState
from framework.schemas.agent_status import AgentStatus
from src.graph.context_bridge import set_caller_input_context
from src.nodes.post_process_node import PostProcessNode, _security_gate_output
from src.nodes.pre_process_node import PreProcessNode
from src.schemas.state import State, from_json

if TYPE_CHECKING:
    from src.graph.domain_workflow_graph import DomainWorkflowGraph

# Runtime-parameter file: src/graph/graph.py -> parents[2] = repo root.
_RUNTIME_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "config.yaml"

# Fallbacks mirror the `retrieval` / `llm` blocks in config/config.yaml so
# _parent_config() never forwards an empty config even if the file is
# unreadable in an exotic deployment layout.
_FALLBACK_RETRIEVAL: dict[str, Any] = {
    "score_threshold": 0.75,
    "kb_path": "config/kb/clinical_trials_kb.json",
    "abstain_min_chars": 25,
}
_FALLBACK_LLM: dict[str, Any] = {
    "temperature": 0.0,
    "max_tokens": 1500,
}

# Non-suppressible advisory notice surfaced on the STRUCTURED payload - a
# constant, never derived from gateable content, so it cannot be withheld by
# any input (mirrors the text-side PI-review disclaimer PostProcessNode
# appends).
_ADVISORY_NOTICE = "decision support only — PI/study-team review required before any " "enrollment action"


def _runtime_config() -> dict[str, Any]:
    """Read the runtime parameters from config/config.yaml.

    Returns an empty dict — never raises — when the file is absent,
    unreadable, not valid YAML, or not a mapping. `timeout_s` is mapped to
    `timeout_seconds`, the key the inner graph validates.
    """
    try:
        import yaml

        loaded = yaml.safe_load(_RUNTIME_CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(loaded, dict):
        return {}
    out: dict[str, Any] = {}
    for key in ("retrieval", "llm", "max_retry"):
        if key in loaded:
            out[key] = loaded[key]
    if "timeout_s" in loaded:
        out["timeout_seconds"] = loaded["timeout_s"]
    return out


class EligibilityScreeningGraphNode(GraphNode):
    """GraphNode subclass assigned to the `main` slot of the outer agent.

    Wraps DomainWorkflowGraph (inner Cat 2 BaseGraph eligibility pre-screen
    pipeline). Called by AgentBaseGraph backbone after pre_process and before
    post_process.

    Contracts:
      get_subgraph()    - instantiate DomainWorkflowGraph with the forwarded
                          runtime config (_parent_config())
      extract_input()   - pull validated_input (PHI-identifier-stripped) from
                          outer state; stash input_context for the inner graph
      merge_output()    - map sub_result fields into outer state delta (changed keys only)
      error_strategy    - "propagate": re-raise inner errors as SubgraphError (fail-fast)
    """

    def __init__(self, runtime_config: dict[str, Any] | None = None) -> None:
        """Receive the runtime config from the outer graph.

        A BaseNode has no config back-reference of its own, so the outer
        AgentBaseGraph reads `self.config` and threads it in here at
        register_nodes() time. Static construction input - not mutable state.
        """
        # A non-mapping runtime config degrades to {} instead of raising: reading and
        # parsing config/config.yaml belongs to the entry point, and this node only has
        # to survive whatever it is handed.
        self._runtime_config = dict(runtime_config) if isinstance(runtime_config, dict) else {}

    # "propagate": re-raise inner graph exceptions as SubgraphError (default - fail fast).
    # "handle": call on_subgraph_error() instead - use for graceful degradation.
    error_strategy: ClassVar[str] = "propagate"

    # False: HITL interrupts are handled inside the inner graph only (this
    # template has no HITL path).
    propagate_hitl: ClassVar[bool] = False

    def _parent_config(self) -> dict[str, Any]:
        """Forward the config/config.yaml runtime blocks to the inner graph.

        Loads config/config.yaml and returns its blocks under
        config["configurable"] - never an empty dict. The inner graph
        republishes the `retrieval` block into inner state
        (DomainWorkflowGraph._extra_initial_state()) so RetrieveNode /
        RerankFilterNode read live score_threshold / kb_path /
        abstain_min_chars values instead of dead declarations. The `llm`
        block is forwarded verbatim for the documented LLM-synthesis
        upgrade (unused by the deterministic pipeline). `max_retry` /
        `timeout_seconds` are validated by the inner graph's
        _validate_config().
        """
        config = self._runtime_config
        retrieval = config.get("retrieval")
        if not isinstance(retrieval, dict) or not retrieval:
            retrieval = dict(_FALLBACK_RETRIEVAL)
        llm = config.get("llm")
        if not isinstance(llm, dict) or not llm:
            llm = dict(_FALLBACK_LLM)
        configurable: dict[str, Any] = {"retrieval": retrieval, "llm": llm}
        for key in ("max_retry", "timeout_seconds"):
            if key in config:
                configurable[key] = config[key]
        return {"configurable": configurable}

    def get_subgraph(self) -> "DomainWorkflowGraph":
        """Instantiate and return the inner domain workflow graph.

        DomainWorkflowGraph is imported lazily (inside the method) to avoid
        circular-import risk at module load time and to match the pattern in
        the Cat 2 sample.

        The inner graph receives the config/config.yaml-derived config via its
        BaseGraph ctor; its domain NODES still take no constructor arguments
        and read config exclusively via State seeding (execute(self, state) ->
        dict - no config parameter).
        """
        from src.graph.domain_workflow_graph import DomainWorkflowGraph

        return DomainWorkflowGraph(config=self._parent_config())

    def execute(self, state: AgentState) -> dict[str, Any]:
        """Skip the inner graph when the request was already found unacceptable.

        A request declined by pre_process has no validated input to work on, so
        running the inner graph would only produce a second, vaguer reason for
        the same rejection - and overwrite the specific one already settled.
        Completing here keeps the original reason intact.

        This override is deliberate: GraphNode.execute() is not final, and the
        marker is the only signal that distinguishes "nothing to do" from "not
        run yet".
        """
        marker = state.get("error_code")
        if marker:
            return {"status": AgentStatus.SUCCESS.value, "error_code": marker}
        result: dict[str, Any] = super().execute(state)
        return result

    def extract_input(self, state: AgentState) -> str:
        """Return the string input passed into inner_graph.invoke().

        PreProcessNode validates and PHI-identifier-strips the raw user_input
        and writes the result to validated_input. Prefer that; fall back to
        user_input if validated_input is absent (e.g. in unit tests).
        Structured params may travel inside this string as a JSON envelope
        and are parsed back by the first inner node (InputValidateNode).

        Also bridges the caller's input_context to the inner graph:
        GraphNode.execute() does not forward input_context on
        subgraph.invoke() (SDK 1.0.1), and extract_input is the last
        our-code hook that sees the outer state before the inner invoke -
        see src/graph/context_bridge.py.
        """
        set_caller_input_context(state.get("input_context") or {})
        return str(state.get("validated_input") or state.get("user_input", ""))

    def merge_output(self, state: AgentState, sub_result: dict[str, Any]) -> dict[str, Any]:
        """Map inner graph sub_result back into the outer state delta.

        sub_result is the dict returned by DomainWorkflowGraph.get_output().
        Returns ONLY changed keys - never the full state.

        Key coupling (designed together with DomainWorkflowGraph.get_output()):
          Inner get_output() emits  -> "formatted_answer", "citations",
                                       "criterion_assessment", "overall_abstain",
                                       "status", ...
          This merge_output() reads -> sub_result.get("formatted_answer"),
                                       sub_result.get("citations"),
                                       sub_result.get("criterion_assessment"),
                                       sub_result.get("overall_abstain"),
                                       sub_result.get("status")

        eligibility_assessment_result (str | None): final rendered pre-screen
          result; written by OutputFormatNode inside the inner graph.
        result: PostProcessNode (outer post_process slot) reads
          state.get("result") - the inner graph emits the rendered result
          under "formatted_answer", so map it to "result" as well; otherwise
          the final output surfaced by PostProcessNode (and the output gate,
          incl. the mandatory PI-review disclaimer) is always empty.
        status (str | None): terminal AgentStatus value from the inner graph run.
        """
        return {
            "eligibility_assessment_result": sub_result.get("formatted_answer"),
            "result": sub_result.get("formatted_answer"),
            "citations": sub_result.get("citations"),
            "criterion_assessment": sub_result.get("criterion_assessment"),
            "overall_abstain": sub_result.get("overall_abstain"),
            "status": sub_result.get("status"),
            # Outer reason wins. A reason already settled before the inner run
            # is the real one; the inner graph only ever sees the downstream
            # consequence of it, so taking the inner value first would replace a
            # specific reason with a generic one - and a plain sub_result.get()
            # would erase the outer reason entirely whenever the inner run did
            # not set its own.
            "error_code": state.get("error_code") or sub_result.get("error_code", ""),
        }


class ClinicalTrialEligibilityQAAgent(AgentBaseGraph):
    """Outer graph for HCR-C2-002 (Cat 2 RAG decision-support pre-screen).

    Inherits AgentBaseGraph (framework base class) directly. Domain logic is
    fully encapsulated in EligibilityScreeningGraphNode (main slot), which
    delegates to DomainWorkflowGraph (inner BaseGraph).

    Backbone (fixed - identical to Cat 1):
        START -> initialize -> pre_process -> main -> post_process -> finalize -> END

    register_nodes() is the ONLY structural override:
      - super().register_nodes() fills: initialize, finalize (framework defaults)
      - pre_process:  PreProcessNode (input validation + PHI identifier strip)
      - main:         EligibilityScreeningGraphNode (delegates to DomainWorkflowGraph)
      - post_process: PostProcessNode (output gate + mandatory PI-review disclaimer)

    add_edges() is NOT overridden - backbone wiring belongs to the framework.
    get_output() IS overridden (structured product) - see below.
    """

    @property
    def name(self) -> str:
        """Agent identifier registered with AgentRegistry."""
        return "ClinicalTrialEligibilityQAAgent"

    @property
    def state_schema(self) -> type:
        return State

    def register_nodes(self) -> None:
        """Fill all 5 backbone slots.

        super().register_nodes() MUST be called first - it injects the
        framework's default InitializeNode (sets schema_version, session_id,
        trust_level) and FinalizeNode (builds response_metadata, total_time_ms).
        """
        super().register_nodes()  # fills: initialize, finalize

        self._nodes["pre_process"] = PreProcessNode()
        self._nodes["main"] = EligibilityScreeningGraphNode(runtime_config=self.config)
        self._nodes["post_process"] = PostProcessNode()

    # add_edges() is NOT overridden - backbone wiring belongs to the framework.

    def get_output(self, state: AgentState) -> dict[str, Any]:
        """Extend the base envelope with the structured pre-screen product.

        base = {output, status, trace_id, correlation_id, node_history} from
        super(); on SUCCESS only, ADD criterion_assessment / citations /
        overall_abstain / advisory_notice - never on a non-SUCCESS status
        (fail-closed: withhold the structured fields entirely rather than
        surface a partial/gated result).

        Each surfaced field is built from explicit, whitelisted, KB-derived
        SCALAR values (ids, status labels, scores, source strings) - never a
        raw forwarded dict - and is defensively re-scanned through the SAME
        recursive `_security_gate_output()` PostProcessNode uses before
        being returned, so a nested credential/PHI leak cannot bypass the
        gate by riding the structured path instead of the text path.

        Never surfaces an eligibility verdict field - criterion_assessment
        entries carry a `status` in {"appears_satisfied", "uncertain",
        "flagged_for_review", "insufficient_info"} only (see
        src/nodes/rerank_filter_node.py), never "eligible"/"ineligible".
        """
        base: dict[str, Any] = super().get_output(state)
        if state.get("status") != AgentStatus.SUCCESS.value:
            return base

        # A run that completed without processing the request has no pre-screen
        # product to surface. SUCCESS reports that the run reached a defined end
        # safely, not that an assessment was produced - so the structured fields
        # are withheld here exactly as they are on a non-SUCCESS status. Only the
        # sentence saying what to correct is returned.
        if state.get("error_code"):
            return base

        criterion_assessment = from_json(state.get("criterion_assessment"), []) or []
        citations = from_json(state.get("citations"), []) or []

        if _security_gate_output(criterion_assessment) or _security_gate_output(citations):
            # Fail-closed: withhold the structured fields entirely on any
            # gate violation. The text-side output already carries the gate
            # verdict (incl. disclaimer) via PostProcessNode; nothing further
            # is added here.
            return base

        base["criterion_assessment"] = criterion_assessment
        base["citations"] = citations
        base["overall_abstain"] = bool(state.get("overall_abstain", False))
        base["advisory_notice"] = _ADVISORY_NOTICE
        return base


# Back-compat alias - config/agent.yaml declares the dotted path to
# ClinicalTrialEligibilityQAAgent, and src/api/server.py imports the class
# directly. Keep both names pointing at the agent.
Graph = ClinicalTrialEligibilityQAAgent
