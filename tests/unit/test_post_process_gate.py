# HCR-C2-002 — Unit Tests: PostProcessNode output gate (recursive scan)
#
# Proves the output gate is real and RECURSIVE (content: Any, walks
# dict/list/tuple at any depth) — never the weaker top-level-string-only
# form. Invoked via node(state) / __call__ so the trust gate
# (VERIFIED_EXTERNAL) is satisfied first, matching the project's
# trust-gate test convention.
#
# A clean STRING result also carries the mandatory, non-suppressible
# PI-review disclaimer (docs/02_design.md "Safety Boundary") appended by
# this same node — test_clean_string_result_passes asserts on that real
# behaviour instead of exact string equality.

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

from src.nodes.post_process_node import (
    PostProcessNode,
    _PI_REVIEW_DISCLAIMER,
    _security_gate_output,
)


def _make_state(result, **extra) -> dict:
    state = {
        "caller_trust_level": TrustLevel.VERIFIED_EXTERNAL.value,
        "result": result,
        "node_history": [],
        "error_log": [],
        "session_id": "test-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestGateCleanPath:
    def test_clean_string_result_passes(self):
        """A clean string passes the gate AND is stamped with the mandatory,
        non-suppressible PI-review disclaimer (docs/02_design.md "Safety
        Boundary") — the original content is preserved verbatim, the
        disclaimer is appended, not substituted."""
        node = PostProcessNode()
        result = node(_make_state("a clean eligibility pre-screen summary"))
        assert result.get("status") == AgentStatus.SUCCESS.value
        output = result.get("formatted_output")
        assert output.startswith("a clean eligibility pre-screen summary")
        assert _PI_REVIEW_DISCLAIMER in output

    def test_clean_nested_structure_passes(self):
        """A nested dict/list result with no violations anywhere must pass —
        proves the recursion doesn't false-positive on ordinary structured content."""
        node = PostProcessNode()
        nested = {
            "summary": "patient appears to meet 4 of 5 criteria",
            "criteria": [
                {"id": "INC-1", "met": True, "note": "age within protocol range"},
                {"id": "EXC-2", "met": False, "note": "no prior chemotherapy on record"},
            ],
        }
        result = node(_make_state(nested))
        assert result.get("status") == AgentStatus.SUCCESS.value
        assert result.get("formatted_output") == nested


class TestGateRecursiveViolation:
    """A violation buried inside a nested dict/list must be caught exactly like
    a top-level string — the defect the top-level-only form let through."""

    def test_credential_nested_two_levels_deep_is_blocked(self):
        node = PostProcessNode()
        nested = {
            "summary": "clean top-level text",
            "criteria": [
                {"id": "INC-1", "note": "token: sk-abcdefghijklmnopqrstuvwx"},
            ],
        }
        result = node(_make_state(nested))
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result.get("error_log", []))
        assert result.get("formatted_output") != nested  # sanitised stub, not raw content

    def test_phi_shaped_id_inside_tuple_is_blocked(self):
        node = PostProcessNode()
        nested = {"citations": ("protocol XYZ-123", "patient SSN 123-45-6789")}
        result = node(_make_state(nested))
        assert result.get("status") == AgentStatus.ERROR.value
        assert any("output blocked" in str(e) for e in result.get("error_log", []))

    def test_gate_function_recurses_directly(self):
        """Direct unit coverage of _security_gate_output's recursion contract."""
        assert _security_gate_output("clean") is None
        assert _security_gate_output({"a": {"b": ["clean", "still clean"]}}) is None
        assert _security_gate_output({"a": ["ok", {"b": "MRN-1234567"}]}) == "mrn_like_id"
        assert _security_gate_output(("ok", ("Bearer " + "x" * 25,))) == "bearer_token"
        assert _security_gate_output(42) is None
        assert _security_gate_output(None) is None
