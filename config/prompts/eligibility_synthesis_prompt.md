# Eligibility Synthesis Prompt — HCR-C2-002 (v2 LLM upgrade seam)

> **v1 does NOT use this prompt at runtime.** v1 of `GenerateAnswerNode` is
> deterministic (rule-based, cited narrative assembly over
> `criterion_assessment`); no node reads this file. It documents the
> synthesis contract for the v2 LLM upgrade described in
> `docs/02_design.md` ("v1 Implementation Note — LLM synthesis"), so the v2
> swap changes only the inside of `GenerateAnswerNode.execute()`.

## Contract (v2 GenerateAnswerNode)

- **Input:** the same `criterion_assessment` JSON (id / trial_id /
  criterion_type / category / criterion_text / status / score /
  matched_term / source) the v1 node reads, plus `overall_abstain`.
- **Output:** the same state contract — `grounded_answer` (str, with
  numbered `[n]` citation markers) and `citations` (JSON list of
  `{ref, id, trial_id, category, source}`).
- **Grounding rule:** every factual statement in the answer must be
  traceable to one of the supplied `criterion_assessment` entries via a
  `[n]` marker; content not present there must not be asserted.
- **Status-vocabulary rule (SAFETY BOUNDARY — carries over unchanged to
  v2):** a v2 node MUST still use the same four-value status vocabulary
  (`appears_satisfied` / `uncertain` / `flagged_for_review` /
  `insufficient_info`) and MUST NOT introduce or imply an eligibility
  verdict ("eligible" / "ineligible" / "meets" / "fails" / an enrolment
  recommendation). This is a hard constraint carried from `docs/02_design.md`
  "Safety Boundary", not a v1-only convenience.
- **Abstain rule:** when `overall_abstain` is true, emit ONLY the
  abstention message and route to PI/human review — never a synthesised
  per-criterion read, exactly as v1 does.
- **Tone:** neutral, clinically-appropriate, no individualized treatment
  recommendations (the mandatory PI-review disclaimer is appended
  downstream by `PostProcessNode`, not by this node).

## Prompt template

```
You are drafting a clinical-trial eligibility PRE-SCREEN summary, strictly
from the criterion-assessment entries provided below. You are NOT making an
eligibility determination — a Principal Investigator makes that
determination from the full patient record.

Patient-criteria text (already de-identified upstream):
{patient_criteria_text}

Criterion assessments (each with a reference number):
{criterion_assessment}

Rules:
1. Use ONLY the criterion assessments above. Do not infer criteria that are
   not listed.
2. Mark every statement with the [n] reference of its criterion entry.
3. Use ONLY these status words: "appears satisfied", "uncertain", "flagged
   for review", "insufficient information". NEVER say "eligible",
   "ineligible", "meets", "fails", or recommend enrollment/non-enrollment.
4. Group the narrative under "Inclusion Criteria" and "Exclusion Criteria"
   headings, in that order.
5. Keep the answer under 400 words.
```

## Manifest coupling

The `llm` block in `config/agent.yaml` (`temperature`, `max_tokens`) is
already forwarded to the inner graph via
`EligibilityScreeningGraphNode._parent_config()` under
`config["configurable"]["llm"]`; the v2 node reads it from there.

## Out of scope for v2 too

Numeric-threshold criterion evaluation (e.g. an exact age cutoff or a lab
value range comparison) is a SEPARATE, larger upgrade from LLM synthesis —
see `docs/02_design.md` "v1 Implementation Note" — and is not implied by
adopting this prompt. v2 LLM synthesis still reasons over the same
deterministically-matched `criterion_assessment` candidates that
`RetrieveNode`/`RerankFilterNode` produce; it does not replace that
matching stage.
