"""Synthetic resume corpus generator.

ponytail note: the plan originally called for calling the Anthropic API to
generate resumes. Switched to a deterministic, seeded template generator
instead — no API cost, and it's a *better* fit for two of the plan's own
requirements: (1) "no planted quality label" is trivially true (there's no
quality axis to plant — tier only sets experience-length band; a separate
set of randomly-drawn, tier-independent attributes below gives the ranking
eval an actual quality gradient to grade), and (2) the bias eval needs a
resume held byte-identical except one swapped field, which a template can
guarantee and an LLM generation can't.

Opus review round 1 (implementation-level) found the first pass had no real
dates (so the Phase 2 date-ordering/experience-consistency checks had
nothing to validate), a stated-years/role-years mismatch on 37/40 resumes,
cross-category employers, duplicate bullet lines, and no signal beyond
keyword overlap for the ranking eval to grade — this version fixes all five:
years are now *derived* from real generated role dates (can't drift out of
sync by construction), employers are scoped per category, bullets are
sampled without replacement, and four tier-independent quality attributes
(quantified outcomes, tenure pattern, title progression, degree relevance)
are randomly assigned so a rubric judge has something to grade that raw
keyword/TF-IDF overlap can't see.

Writes data/synthetic/resumes/<candidate_id>.txt and manifest.json (id,
category, tier, name — no quality label, so the ranking eval can't recover
a generation-time signal).

Two intentional corpus properties, worth knowing before Phase 2 writes date
checks against this data: every role's end month equals the next role's
start month (contiguous, month-resolution convention — not an overlap
artifact), and every candidate's most recent role ends exactly at AS_OF
(the corpus has no employment-gap or "not currently employed" case to test
extraction against; add one by hand in eval/extraction_dataset.json if that
case needs coverage).
"""
import json
import random
from datetime import date
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic" / "resumes"
SEED = 20260824
AS_OF = date(2026, 8, 1)  # fixed anchor, not date.today() — keeps the corpus reproducible

CATEGORIES = ["software_engineer", "data_analyst", "product_manager", "sales", "registered_nurse"]
TIERS = ["junior", "mid", "senior"]  # experience-length band, not a quality label
TIER_MONTHS = {"junior": (6, 24), "mid": (36, 84), "senior": (96, 180)}

FIRST_NAMES = ["Jordan", "Taylor", "Morgan", "Casey", "Alex", "Sam", "Riley", "Drew",
               "Jamie", "Avery", "Cameron", "Reese", "Skyler", "Quinn", "Harper", "Rowan"]
LAST_NAMES = ["Bennett", "Coleman", "Foster", "Grant", "Hayes", "Irwin", "Kessler",
              "Lambert", "Marsh", "Novak", "Ortiz", "Pruitt", "Quintero", "Rhodes",
              "Sawyer", "Tran"]
# Must stay disjoint from data/name_variants.json's paired names, or the bias
# eval's "same resume, only the name changed" premise breaks — enforced by
# an assertion in main(), not just this comment, so a future edit to either
# list fails loudly instead of silently.
NAME_VARIANTS_PATH = Path(__file__).resolve().parent.parent / "data" / "name_variants.json"

SKILLS = {
    "software_engineer": ["Python", "Java", "Go", "React", "AWS", "Docker", "Kubernetes",
                           "PostgreSQL", "REST API design", "CI/CD", "unit testing", "Git"],
    "data_analyst": ["SQL", "Python", "Tableau", "Excel", "A/B testing", "R", "Power BI",
                      "data pipelines", "statistics", "dashboarding", "ETL", "pandas"],
    "product_manager": ["roadmapping", "user research", "A/B testing", "SQL", "Jira",
                         "stakeholder management", "agile", "wireframing", "OKRs",
                         "competitive analysis", "go-to-market strategy", "Figma"],
    "sales": ["Salesforce", "cold outreach", "pipeline management", "negotiation",
              "CRM administration", "quota attainment", "account management",
              "consultative selling", "forecasting", "prospecting", "closing", "HubSpot"],
    "registered_nurse": ["patient assessment", "EHR/Epic", "medication administration",
                          "IV therapy", "wound care", "BLS/ACLS certification", "triage",
                          "care planning", "patient education", "charting", "vitals monitoring",
                          "infection control"],
}

# Ordered low -> high seniority, used for the title-progression attribute.
TITLE_RANKS = {
    "software_engineer": ["Junior Software Engineer", "Software Engineer", "Senior Software Engineer"],
    "data_analyst": ["Data Analyst I", "Data Analyst II", "Senior Data Analyst"],
    "product_manager": ["Associate Product Manager", "Product Manager", "Senior Product Manager"],
    "sales": ["Sales Development Rep", "Account Executive", "Senior Account Executive"],
    "registered_nurse": ["Staff Nurse", "Clinical Nurse", "Charge Nurse"],
}

# Scoped per category — an earlier pass drew from one global pool and gave
# nurses non-healthcare employers, an uncontrolled confound Opus flagged.
COMPANIES = {
    "software_engineer": ["Northfield Systems", "Meridian Software", "Redstone Technologies",
                           "Vantage Retail Group", "Cedar Ridge Analytics"],
    "data_analyst": ["Cedar Ridge Analytics", "Lakeside Financial", "Vantage Retail Group",
                      "Meridian Software", "Oakton Consulting"],
    "product_manager": ["Vantage Retail Group", "Northfield Systems", "Oakton Consulting",
                         "Meridian Software", "Harbor Logistics"],
    "sales": ["Vantage Retail Group", "Harbor Logistics", "Oakton Consulting",
              "Lakeside Financial", "Northfield Systems"],
    "registered_nurse": ["Brightline Health", "Pinecrest Medical Center", "Harborview Medical Group",
                          "Cedar Grove Hospital", "Lakeshore Health System"],
}

SCHOOLS = ["State University", "Riverbend University", "Central Tech Institute",
           "Lakeview College", "Northgate University"]
DEGREES = {
    "software_engineer": "B.S. Computer Science",
    "data_analyst": "B.S. Statistics",
    "product_manager": "B.A. Business Administration",
    "sales": "B.A. Communications",
    "registered_nurse": "B.S. Nursing",
}

VAGUE_BULLETS = [
    "Led {skill} initiative across a cross-functional team.",
    "Owned day-to-day {skill} work as part of the broader team.",
    "Applied {skill} to resolve a recurring operational bottleneck.",
    "Partnered with stakeholders on {skill}, contributing to a process improvement.",
    "Mentored junior teammates on {skill} best practices.",
]
# Templates paired with a verb direction ("reduce" vs "grow") so a metric
# never lands somewhere backwards ("reduced quarterly output" reads as bad,
# not good) — and metrics are scoped per category, same reasoning as
# COMPANIES: a software engineer's resume citing "patient wait times" would
# be exactly the kind of cross-domain confound Opus flagged for employers.
QUANT_BULLETS_REDUCE = [
    "Reduced {metric} by {pct}% through a {skill} initiative.",
    "Cut {metric} by {pct}% after redesigning the team's approach to {skill}.",
]
QUANT_BULLETS_GROW = [
    "Improved {metric} {pct}% by leading a {skill} project end-to-end.",
    "Grew {metric} {pct}% quarter-over-quarter by applying {skill} consistently.",
]
METRICS_REDUCE = {
    "software_engineer": ["error rate", "response time", "deploy time", "on-call incident count"],
    "data_analyst": ["report turnaround time", "data pipeline failure rate", "manual reporting time"],
    "product_manager": ["feature cycle time", "customer churn", "support ticket volume"],
    "sales": ["sales cycle length", "lead response time", "customer churn"],
    "registered_nurse": ["patient wait times", "medication error rate", "readmission rate"],
}
METRICS_GROW = {
    "software_engineer": ["deployment frequency", "test coverage", "system uptime"],
    "data_analyst": ["dashboard adoption", "report accuracy", "self-serve query volume"],
    "product_manager": ["quarterly active users", "feature adoption", "NPS score"],
    "sales": ["quarterly revenue", "pipeline coverage", "win rate"],
    "registered_nurse": ["patient satisfaction scores", "on-time medication rate", "care plan adherence"],
}


def shift_months(d: date, months: int) -> date:
    total = d.year * 12 + (d.month - 1) - months
    y, m = divmod(total, 12)
    return date(y, m + 1, 1)


def split_months(rng: random.Random, total_months: int, n_roles: int, hopping: bool) -> list[int]:
    # Weighted split, not a "dump the remainder into the last role" walk —
    # the earlier version fixed n_roles by tier and only randomized the
    # first n_roles-1 stints, so "hopping" reliably produced two short
    # stints plus one leftover role holding 79-95% of total tenure: the
    # opposite of what the attribute is supposed to signal. A bounded
    # weight spread keeps every role's share of the total in a realistic
    # range regardless of n_roles or total_months.
    if n_roles == 1:
        return [total_months]
    spread = (0.5, 1.5) if hopping else (0.85, 1.15)
    weights = [rng.uniform(*spread) for _ in range(n_roles)]
    total_w = sum(weights)
    parts = [max(1, round(total_months * w / total_w)) for w in weights]
    parts[-1] += total_months - sum(parts)  # absorb rounding drift, keeps sum exact
    return [max(1, p) for p in parts]


def make_resume(rng: random.Random, category: str, tier: str, name: str) -> str:
    lo, hi = TIER_MONTHS[tier]
    total_months = rng.randint(lo, hi)

    # Quality-relevant attributes, drawn independent of tier — the actual
    # gradient the ranking eval and a rubric judge (but not a bag-of-words
    # baseline) can grade on.
    tenure_pattern = rng.choice(["stable", "hopping"])
    quantified = rng.choice([True, False])
    progressing = rng.choice([True, False])
    # Nurse licensure makes an off-field degree read as a fabricated resume
    # (no nursing credential at all), not "less relevant" — exempt that
    # category so the mismatch draw stays a realistic weak signal elsewhere.
    degree_match = True if category == "registered_nurse" else rng.random() < 0.75

    if tenure_pattern == "hopping":
        # n_roles derived from tenure itself, not the tier-based 1/2/3 count
        # below — fixing role count and only shuffling the split let
        # "hopping" dump most of the tenure into one leftover role (measured
        # 79-95% of total tenure in a single stint), rendering as its
        # opposite. One role per ~9 months, capped at 6; company assignment
        # below cycles the per-category pool (5 entries), so a hopper with
        # 6 roles can repeat an employer — reads as a plausible boomerang,
        # not a bug.
        n_roles = min(6, max(3, total_months // 9))
    else:
        n_roles = 1 if tier == "junior" else (2 if tier == "mid" else 3)

    role_months = split_months(rng, total_months, n_roles, hopping=(tenure_pattern == "hopping"))
    ranks = TITLE_RANKS[category]
    tier_rank_idx = {"junior": 0, "mid": 1, "senior": 2}[tier]

    n_skills = rng.randint(4, 7)
    skills = rng.sample(SKILLS[category], k=min(n_skills, len(SKILLS[category])))
    companies = rng.sample(COMPANIES[category], k=min(n_roles, len(COMPANIES[category])))

    years_display = round(total_months / 12, 1)
    lines = [name, "", f"{years_display} years of experience in {category.replace('_', ' ')} roles.",
             "", "Skills: " + ", ".join(skills), "", "Experience:"]

    # Sample (template, skill) combos once across the *whole* resume, not per
    # role — sampling per role independently let the same combo (and hence
    # the exact same rendered bullet line) recur across different roles.
    n_bullets_per_role = [rng.randint(2, 3) for _ in role_months]
    if quantified:
        combos = ([(t, s, m) for t in QUANT_BULLETS_REDUCE for s in skills
                   for m in METRICS_REDUCE[category]] +
                  [(t, s, m) for t in QUANT_BULLETS_GROW for s in skills
                   for m in METRICS_GROW[category]])
    else:
        combos = [(t, s, None) for t in VAGUE_BULLETS for s in skills]
    rng.shuffle(combos)
    combos_needed = sum(n_bullets_per_role)
    chosen_combos = combos[:combos_needed] if combos_needed <= len(combos) else (
        combos * (combos_needed // len(combos) + 1))[:combos_needed]

    cursor_end = AS_OF
    combo_i = 0
    for i, months in enumerate(role_months):  # role_months[0] = most recent
        start = shift_months(cursor_end, months)
        if progressing:
            rank_idx = max(0, tier_rank_idx - i)  # most recent (i=0) highest, older roles lower
        else:
            rank_idx = tier_rank_idx
        title = ranks[rank_idx]
        company = companies[i % len(companies)]
        lines.append(f"- {title}, {company} ({start.isoformat()[:7]} - {cursor_end.isoformat()[:7]})")

        for _ in range(n_bullets_per_role[i]):
            tmpl, sk, metric = chosen_combos[combo_i]
            combo_i += 1
            bullet = tmpl.format(skill=sk, metric=metric, pct=rng.randint(8, 40))
            lines.append(f"    * {bullet}")
        cursor_end = start

    degree = DEGREES[category] if degree_match else rng.choice(
        [d for c, d in DEGREES.items() if c != category])
    lines += ["", f"Education: {degree}, {rng.choice(SCHOOLS)}"]
    return "\n".join(lines)


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    names = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    rng.shuffle(names)

    variant_names = {n for p in json.loads(NAME_VARIANTS_PATH.read_text())["pairs"]
                      for n in (p["variant_a"], p["variant_b"])}
    assert not (variant_names & set(names)), \
        "corpus name collides with a bias-eval variant name — breaks the swap premise"

    manifest = []
    name_i = 0
    for category in CATEGORIES:
        for tier, count in [("junior", 3), ("mid", 3), ("senior", 2)]:
            for _ in range(count):
                name = names[name_i]
                name_i += 1
                candidate_id = f"{category}_{tier}_{name_i:03d}"
                resume_text = make_resume(rng, category, tier, name)
                (OUT_DIR / f"{candidate_id}.txt").write_text(resume_text, encoding="utf-8")
                manifest.append({"candidate_id": candidate_id, "category": category,
                                  "tier": tier, "name": name})

    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(manifest)} resumes to {OUT_DIR}")


if __name__ == "__main__":
    main()
