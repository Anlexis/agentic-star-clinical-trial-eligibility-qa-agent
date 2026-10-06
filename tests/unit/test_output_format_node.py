# HCR-C2-002 — Unit Tests: OutputFormatNode (inner domain node 5, terminal)
#
# Invocation canon: node(state) via BaseNode.__call__ with an ANONYMOUS caller.
# `grounded_answer`/`citations` are DOMAIN fields (not PII-mask scan targets). The
# mandatory PI-review disclaimer is deliberately NOT asserted here — it is
# appended by the OUTER PostProcessNode (see test_post_process_s3_gate.py /
# docs/02_design.md "Safety Boundary"), never by this node.
#
# Mirrors docs/03_test_spec.md §2.6 (FMT-01..FMT-06).
# Deterministic — no LLM, no network. framework.* / src.* imports only.

from unittest.mock import MagicMock

from framework.schemas.agent_status import AgentStatus
from framework.schemas.trust_level import TrustLevel

import src.nodes.output_format_node
from src.nodes.output_format_node import OutputFormatNode
from src.schemas.state import to_json


def _make_state(grounded_answer, citations, **extra) -> dict:
    state = {
        "grounded_answer": grounded_answer,
        "citations": citations,
        "caller_trust_level": TrustLevel.ANONYMOUS.value,
        "node_history": [],
        "error_log": [],
        "session_id": "unit-session",
        "execution_time": {},
    }
    state.update(extra)
    return state


class TestFormattedAnswerComposition:
    def test_fmt_01_composes_header_body_and_sources(self):
        citations = to_json(
            [{"ref": 1, "id": "inc1", "trial_id": "XYZ-123", "category": "age", "source": "Protocol XYZ-123 v2.1"}]
        )
        result = OutputFormatNode()(_make_state("[1] the grounded answer body.", citations))
        answer = result["formatted_answer"]
        assert answer.startswith("# Clinical Trial Eligibility Pre-Screen")
        assert "[1] the grounded answer body." in answer
        assert "## Sources" in answer
        assert "- [1] XYZ-123 / age — Protocol XYZ-123 v2.1" in answer
        assert result["status"] == AgentStatus.SUCCESS.value
        # Regression guard: State carries the plain string, never the enum
        # (AgentStatus subclasses str, so a bare isinstance would not catch it).
        assert isinstance(result["status"], str)
        assert not isinstance(result["status"], AgentStatus)

    def test_fmt_02_label_falls_back_when_category_and_source_blank(self):
        citations = to_json([{"ref": 2, "id": "exc1", "trial_id": "XYZ-123", "category": "", "source": ""}])
        answer = OutputFormatNode()(_make_state("body.", citations))["formatted_answer"]
        assert "- [2] XYZ-123\n" in answer + "\n"
        assert " — " not in answer.split("## Sources")[1].split("\n")[1]

    def test_fmt_03_label_falls_back_to_criterion_when_no_trial_id_or_category(self):
        citations = to_json([{"ref": 1, "id": "exc1", "trial_id": "", "category": "", "source": "a source"}])
        answer = OutputFormatNode()(_make_state("body.", citations))["formatted_answer"]
        assert "- [1] criterion — a source" in answer

    def test_disclaimer_is_not_added_by_this_node(self):
        """The mandatory PI-review disclaimer is added by PostProcessNode,
        NOT here — see docs/02_design.md "Safety Boundary"."""
        answer = OutputFormatNode()(_make_state("body.", to_json([])))["formatted_answer"]
        assert "PI" not in answer
        assert "Principal Investigator" not in answer


class TestDegradedInputs:
    def test_fmt_04_no_citations_renders_explicit_none_line(self):
        answer = OutputFormatNode()(_make_state("no coverage body.", to_json([])))["formatted_answer"]
        assert "- none (no criterion reached a citable appears-satisfied/flagged/uncertain assessment)" in answer

    def test_fmt_05_missing_grounded_answer_uses_fallback_text(self):
        state = _make_state("", to_json([]))
        del state["grounded_answer"]
        result = OutputFormatNode()(state)
        assert "No eligibility pre-screen result is available for this request." in result["formatted_answer"]
        assert result["status"] == AgentStatus.SUCCESS.value


class TestOutputFormatAudit:
    def test_fmt_06_domain_audit_payload(self, monkeypatch):
        spy = MagicMock()
        monkeypatch.setattr(src.nodes.output_format_node, "emit_trace_event", spy)
        citations = to_json([{"ref": 1, "id": "inc1", "trial_id": "XYZ-123", "category": "age", "source": "s"}])
        result = OutputFormatNode()(_make_state("[1] body.", citations))
        events = [call.args[0] for call in spy.call_args_list]
        assert "output_format_complete" in events
        payload = spy.call_args_list[events.index("output_format_complete")].args[1]
        assert payload["citation_count"] == 1
        assert payload["answer_chars"] == len(result["formatted_answer"])
