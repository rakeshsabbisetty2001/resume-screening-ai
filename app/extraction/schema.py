from pydantic import BaseModel, Field


class Education(BaseModel):
    degree: str
    institution: str


class Role(BaseModel):
    title: str
    company: str
    start_date: str = Field(description="YYYY-MM")
    end_date: str = Field(description="YYYY-MM, or the literal string 'present' if still employed")
    bullets: list[str]


class Candidate(BaseModel):
    name: str  # dropped before the scorer sees this object — see app/scoring/score.py
    years_experience: float
    skills: list[str]
    education: list[Education]
    roles: list[Role]
