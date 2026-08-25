"""Rubric-free LLM ranking baseline: one call per JD, "just rank these N
candidates" with no rubric structure — isolates whether app.scoring's
weighted rubric earns its extra cost/latency over a bare LLM ranking. The
TF-IDF baseline (eval/metrics.py) only tests "LLM > grep", which isn't in
question; this one tests "does the rubric matter".

Same PII/error-handling shape as app/scoring/score.py and
app/extraction/extract.py — kept here rather than in app/ since this is
eval-only, never used by the production API/UI.
"""
import pydantic

from app.config import settings

client = None  # set lazily so importing this module doesn't require anthropic creds


def _get_client():
    global client
    if client is None:
        import anthropic
        client = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=60.0)
    return client


class RankingOutput(pydantic.BaseModel):
    ranked_candidate_ids: list[str]


class BaselineLLMError(Exception):
    """Same PII contract as ExtractionError/ScoringError — ids only."""


def rank_via_bare_llm(job_id: str, job_description: str,
                       candidate_summaries: dict[str, str]) -> list[str]:
    ids = list(candidate_summaries.keys())
    listing = "\n\n".join(f"[{cid}]\n{text}" for cid, text in candidate_summaries.items())
    prompt = (
        "Rank these candidates from best to worst fit for the job description below. "
        "Return their ids in ranked_candidate_ids, best first. Use exactly these ids: "
        f"{ids}\n\nJob description:\n{job_description}\n\nCandidates:\n{listing}"
    )
    try:
        response = _get_client().messages.parse(
            model=settings.scoring_model,
            max_tokens=16000,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
            output_format=RankingOutput,
        )
    except pydantic.ValidationError:
        raise BaselineLLMError(f"malformed baseline ranking output for job {job_id}") from None

    if response.stop_reason == "refusal":
        raise BaselineLLMError(f"baseline ranking refused for job {job_id}")
    if response.parsed_output is None:
        raise BaselineLLMError(f"no parsed baseline ranking for job {job_id}")

    result_ids = response.parsed_output.ranked_candidate_ids
    # Model output isn't schema-guaranteed to be exactly `ids` (it could
    # hallucinate/drop one) — filter to known ids, then append any missing
    # ones at the end so every candidate still appears somewhere.
    result_ids = [cid for cid in result_ids if cid in ids]
    result_ids += [cid for cid in ids if cid not in result_ids]
    return result_ids
