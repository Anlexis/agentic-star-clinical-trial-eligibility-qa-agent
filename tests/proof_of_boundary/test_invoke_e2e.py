# PB: End-to-end business behaviour through POST /invoke — src/api/server.py
#
# Proves the supported input contract produces REAL outcomes through the full
# nested graph (outer backbone → inner domain pipeline):
#   - caller data supplied via input_context reaches the inner graph (the
#     context bridge) and yields a cited, non-empty pre-screen assessment
#   - an un-negated exclusion ground is FLAGGED FOR REVIEW
#   - sparse / trial-less input abstains instead of fabricating an assessment
#   - malformed caller data is rejected fail-closed, with no value echo
#   - the rendered output honours the safety schema: closed status
#     vocabulary (never a verdict), the mandatory PI-review disclaimer, and
#     no PHI-shaped identifier survival
#
# Unlike test_server_boot.py (which checks the module-level boot), these tests
# run the REAL compiled agent: every request crosses the entry-point auth, the
# outer trust/input gates, the input_context bridge into the inner graph, all
# five domain nodes, and the output gate.
#
# The app is driven through its real ASGI interface (no TestClient — httpx is
# only a transitive dependency).

import asyncio
import json
import re

import pytest

from src.api import server as server_module  # noqa: F401  (import = boot check)
from src.api.server import app
from src.nodes.post_process_node import _PI_REVIEW_DISCLAIMER

_TOKEN = "pb-invoke-e2e-token"

_INPUT_TEXT = "Run a clinical-trial eligibility pre-screen for the attached patient profile."

# De-identified narrative matching several seeded XYZ-123 criteria, with the
# brain-metastases exclusion explicitly negated.
_CLEAN_PROFILE = (
    "68-year-old adult patient with histologically confirmed Stage II NSCLC, "
    "ECOG performance status 1, no prior chemotherapy, no known brain "
    "metastases, adequate organ function on recent labs."
)

# Same shape, but an exclusion ground (brain metastases) appears UN-negated —
# the pre-screen must flag it for review, never clear it.
_FLAGGED_PROFILE = (
    "68-year-old adult patient with histologically confirmed Stage II NSCLC, "
    "ECOG performance status 1, known brain metastases on the latest imaging, "
    "adequate organ function on recent labs."
)


def _post_invoke(payload: dict, with_auth: bool = True) -> tuple[int, dict]:
    """POST /invoke with a Bearer token through the real ASGI app."""
    body = json.dumps(payload).encode()
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode()),
    ]
    if with_auth:
        headers.append((b"authorization", f"Bearer {_TOKEN}".encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.3"},
        "http_version": "1.1",
        "method": "POST",
        "scheme": "http",
        "path": "/invoke",
        "raw_path": b"/invoke",
        "root_path": "",
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("127.0.0.1", 8000),
    }

    messages = []
    sent = {"body": b""}

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body":
            sent["body"] += message.get("body", b"")

    asyncio.run(app(scope, receive, send))
    start = next(m for m in messages if m["type"] == "http.response.start")
    parsed = json.loads(sent["body"].decode() or "{}")
    return start["status"], parsed


@pytest.fixture(autouse=True)
def token_configured(monkeypatch):
    """Deploy-shaped server environment: INVOKE_AUTH_TOKEN set, caller uses Bearer."""
    monkeypatch.setenv("INVOKE_AUTH_TOKEN", _TOKEN)


def _invoke(input_context: dict, input_text: str = _INPUT_TEXT) -> dict:
    status_code, body = _post_invoke(
        {"input": input_text, "session_id": "pb-invoke-e2e", "input_context": input_context}
    )
    assert status_code == 200, f"expected 200, got {status_code}: {body}"
    return body


class TestInvokeEndToEnd:
    def test_caller_data_via_input_context_produces_a_real_assessment(self):
        """input_context crosses the outer→inner bridge and yields a real,
        cited pre-screen — the input text itself names no trial and no
        criteria, so only the bridged context can have produced this."""
        body = _invoke({"patient_profile": _CLEAN_PROFILE, "trial_id": "XYZ-123"})

        assert body["status"] == "success"
        output = body["output"]
        assert "# Clinical Trial Eligibility Pre-Screen" in output
        assert "APPEARS SATISFIED" in output
        assert "[1]" in output  # numbered citations present
        assert "## Sources" in output
        assert _PI_REVIEW_DISCLAIMER in output
        # Structured payload rides along on SUCCESS.
        assert body["overall_abstain"] is False
        assert len(body["criterion_assessment"]) == 7  # every seeded XYZ-123 criterion
        assert len(body["citations"]) >= 1
        assert body["advisory_notice"]

    def test_unnegated_exclusion_ground_is_flagged_for_review(self):
        body = _invoke({"patient_profile": _FLAGGED_PROFILE, "trial_id": "XYZ-123"})

        assert body["status"] == "success"
        output = body["output"]
        assert "FLAGGED FOR REVIEW" in output, output
        assert "criterion/criteria flagged for review" in output
        flagged = [e for e in body["criterion_assessment"] if e["status"] == "flagged_for_review"]
        assert flagged, body["criterion_assessment"]

    def test_negated_exclusion_ground_is_not_flagged(self):
        """The clean profile explicitly negates the same exclusion ground —
        the negation-aware matcher must not read it as positive evidence."""
        body = _invoke({"patient_profile": _CLEAN_PROFILE, "trial_id": "XYZ-123"})
        assert "FLAGGED FOR REVIEW" not in body["output"]

    def test_sparse_input_abstains_instead_of_assessing(self):
        body = _invoke({}, input_text="short text")

        assert body["status"] == "success"
        assert body["overall_abstain"] is True
        assert body["criterion_assessment"] == []
        assert "too short/sparse" in body["output"]
        assert _PI_REVIEW_DISCLAIMER in body["output"]

    @pytest.mark.parametrize(
        "bad_trial_id",
        [42, True, ["XYZ-123"], {"id": "XYZ-123"}, float("nan"), "not a trial id", "XYZ_123"],
        ids=["int", "bool", "list", "dict", "raw-nan", "free-text", "bad-separator"],
    )
    def test_invalid_trial_id_is_rejected_fail_closed(self, bad_trial_id):
        """A malformed trial_id must produce a validation error, not an
        assessment (raw float('nan') also covers Python json's bare-NaN
        extension reaching the request body)."""
        body = _invoke({"patient_profile": _CLEAN_PROFILE, "trial_id": bad_trial_id})

        assert body["status"] == "success", body
        assert body["output"], body  # the reason reaches the caller
        assert "criterion_assessment" not in body

    def test_invalid_patient_profile_is_rejected_fail_closed(self):
        body = _invoke({"patient_profile": 12345, "trial_id": "XYZ-123"})
        assert body["status"] == "success", body
        assert body["output"], body  # the reason reaches the caller

    def test_rejected_values_are_never_echoed_in_the_response(self):
        marker = "zqxv_marker_never_echoed_9917"
        body = _invoke({"patient_profile": _CLEAN_PROFILE, "trial_id": marker})
        assert body["status"] == "success"
        assert marker not in json.dumps(body)

    def test_oversized_input_context_is_refused_at_the_adapter(self):
        status_code, _ = _post_invoke(
            {
                "input": _INPUT_TEXT,
                "session_id": "pb-invoke-e2e",
                "input_context": {"patient_profile": "x" * 300_000},
            }
        )
        assert status_code == 413

    def test_missing_bearer_token_is_refused(self):
        status_code, body = _post_invoke({"input": _INPUT_TEXT, "session_id": "pb-invoke-e2e"}, with_auth=False)
        assert status_code == 401
        assert body.get("detail") == "Token is invalid or expired."


class TestOutputSchemaScan:
    """The safety schema, scanned on the real rendered output: a closed,
    hedged status vocabulary (never an eligibility verdict), the mandatory
    disclaimer, and no PHI-shaped identifier survival."""

    _VERDICT_RE = re.compile(r"\b(?:eligible|ineligible|enrolled|meets criteria|fails criteria)\b", re.IGNORECASE)
    _ALLOWED_STATUSES = {
        "appears_satisfied",
        "flagged_for_review",
        "uncertain",
        "insufficient_info",
    }

    def test_rendered_output_never_carries_a_verdict(self):
        for profile in (_CLEAN_PROFILE, _FLAGGED_PROFILE):
            body = _invoke({"patient_profile": profile, "trial_id": "XYZ-123"})
            assert not self._VERDICT_RE.search(body["output"]), body["output"]

    def test_structured_statuses_stay_in_the_closed_vocabulary(self):
        body = _invoke({"patient_profile": _FLAGGED_PROFILE, "trial_id": "XYZ-123"})
        statuses = {e["status"] for e in body["criterion_assessment"]}
        assert statuses <= self._ALLOWED_STATUSES, statuses

    def test_phi_shaped_identifiers_do_not_survive_into_the_output(self):
        profile = _CLEAN_PROFILE + " MRN-1234567, SSN 123-45-6789, reachable at patient@example.com."
        body = _invoke({"patient_profile": profile, "trial_id": "XYZ-123"})
        blob = json.dumps(body)
        assert "MRN-1234567" not in blob
        assert "123-45-6789" not in blob
        assert "patient@example.com" not in blob
