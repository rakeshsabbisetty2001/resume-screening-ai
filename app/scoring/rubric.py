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
assert abs(sum(CRITERIA.values()) - 1.0) < 1e-9, "rubric weights must sum to 1.0"


class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=5)
    rationale: str


class RubricScore(BaseModel):
    skills_match: CriterionScore
    experience_fit: CriterionScore
    education_fit: CriterionScore
    role_relevance: CriterionScore

    def weighted_total(self) -> float:
        return sum(CRITERIA[name] * getattr(self, name).score for name in CRITERIA)
