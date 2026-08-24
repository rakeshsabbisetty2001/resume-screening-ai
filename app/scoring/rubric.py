"""Fixed weighted rubric criteria for candidate-vs-JD scoring.

Weights sum to 1.0 — checked at import time so a typo fails loud instead of
silently producing a weighted_total that isn't out of 5.
"""
from pydantic import BaseModel, Field

CRITERIA: dict[str, float] = {
    "skills_match": 0.35,
    "experience_fit": 0.30,
    "education_fit": 0.10,
    "role_relevance": 0.25,
}
# `raise`, not `assert` — assertions are stripped under `python -O`, which
# would silently remove this guard in exactly the optimized deploy path
# where a silent weight typo matters most.
if abs(sum(CRITERIA.values()) - 1.0) > 1e-9:
    raise ValueError("rubric weights must sum to 1.0")


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str


class RubricScore(BaseModel):
    skills_match: CriterionScore
    experience_fit: CriterionScore
    education_fit: CriterionScore
    role_relevance: CriterionScore

    def weighted_total(self) -> float:
        total = sum(CRITERIA[name] * getattr(self, name).score for name in CRITERIA)
        # Collapse float noise: weights/scores are multiples of 0.05, so two
        # mathematically-equal totals (e.g. 2.0 vs 1.9999999999999998) can
        # otherwise compare unequal — which would break rank_candidates'
        # documented "stable sort, caller order on ties" contract.
        return round(total, 4)


if set(CRITERIA) != set(RubricScore.model_fields):
    raise ValueError("CRITERIA keys must match RubricScore fields exactly")
