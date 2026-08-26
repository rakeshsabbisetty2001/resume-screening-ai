"""Synthetic job-description corpus. Hand-written, not LLM-generated — same
rationale as generate_resumes.py: 9 short JDs is less work to write
directly than to prompt-engineer and pay an API call for, and hand-written
means no risk of an LLM silently biasing a JD's wording toward whichever
resumes exist in the corpus.

Writes data/synthetic/jds/<job_id>.txt and manifest.json.
"""
import json
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "jds"

JOBS = [
    ("swe_backend", "software_engineer",
     "Backend Software Engineer\n\n"
     "We're hiring a backend engineer to build and operate services handling "
     "production traffic. Requirements: 3+ years professional experience, "
     "strong Python or Go, experience with REST API design, comfort with "
     "Docker/Kubernetes and CI/CD pipelines, PostgreSQL or similar. Bonus: "
     "AWS experience, unit testing discipline."),
    ("swe_fullstack", "software_engineer",
     "Full-Stack Developer\n\n"
     "Looking for a full-stack developer for a small product team. "
     "Requirements: React experience, a backend language (Python/Java/Go), "
     "REST API design, Git workflow fluency. Bonus: AWS, CI/CD, Kubernetes."),
    ("data_analyst_bi", "data_analyst",
     "Business Intelligence Analyst\n\n"
     "Seeking an analyst to own reporting and dashboarding for a growing "
     "team. Requirements: strong SQL, Tableau or Power BI, comfort building "
     "and maintaining data pipelines. Bonus: Python/R, A/B testing "
     "experience, statistics background."),
    ("data_analyst_growth", "data_analyst",
     "Growth Analyst\n\n"
     "Analyst role focused on experimentation and funnel metrics. "
     "Requirements: SQL, A/B testing methodology, Excel or Python for "
     "analysis. Bonus: dashboarding tools, ETL experience."),
    ("pm_core", "product_manager",
     "Product Manager\n\n"
     "Own the roadmap for a core product area. Requirements: 3+ years PM "
     "experience, user research, roadmapping, stakeholder management, "
     "comfort with SQL for data-informed decisions. Bonus: agile "
     "experience, OKR-setting, wireframing."),
    ("pm_associate", "product_manager",
     "Associate Product Manager\n\n"
     "Early-career PM role, mentored by senior product leadership. "
     "Requirements: some product or analytical experience, comfort running "
     "A/B tests, basic SQL. Bonus: Figma, go-to-market exposure."),
    ("sales_ae", "sales",
     "Account Executive\n\n"
     "Full-cycle sales role owning outbound prospecting through close. "
     "Requirements: 2+ years quota-carrying sales experience, CRM "
     "fluency (Salesforce or HubSpot), consultative selling approach. "
     "Bonus: forecasting experience, account management background."),
    ("sales_bdr", "sales",
     "Business Development Representative\n\n"
     "Entry-level sales role focused on outbound prospecting and pipeline "
     "generation for the AE team. Requirements: cold outreach experience "
     "or strong communication skills, CRM comfort. Bonus: prior quota "
     "attainment, negotiation training."),
    ("rn_medsurg", "registered_nurse",
     "Registered Nurse — Med-Surg\n\n"
     "RN for a medical-surgical unit. Requirements: active RN license, "
     "BLS certification, patient assessment and care planning experience, "
     "EHR/Epic proficiency. Bonus: ACLS certification, wound care "
     "experience, charge nurse experience."),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = []
    for job_id, category, text in JOBS:
        (OUT_DIR / f"{job_id}.txt").write_text(text, encoding="utf-8")
        manifest.append({"job_id": job_id, "category": category})
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} job descriptions to {OUT_DIR}")


if __name__ == "__main__":
    main()
