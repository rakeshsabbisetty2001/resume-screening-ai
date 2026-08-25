"""Bias eval: three arms, per design decision 4.

1. name-swap -> NAME-VISIBLE scorer (include_name=True): the actual bias
   measurement — does the model score a candidate differently when only
   the name changes, with the name visible to it?
2. name-swap -> BLIND scorer (include_name=False, the production default):
   a sanity check, not a finding — since the blind scorer never sees the
   name, this delta should collapse to the noise floor. If it doesn't,
   something is leaking (see app/scoring/score.py's scrub).
3. university-swap -> BLIND scorer: the required, non-tautological
   measurement — university text isn't scrubbed, so this measures real
   residual sensitivity while name stays hidden.

Same cost-gate pattern as eval/run_eval.py: prints the call estimate by
default, only --run spends anything, refuses --run without an API key.

Reruns (n>=3) exist because current Claude models have no determinism knob
(temperature/top_p/top_k removed) — identical input still varies, so a
score delta needs a measured noise floor from the *unswapped* input, not a
bare single-sample comparison.
"""
import argparse
import json
import sys
from pathlib import Path

from app.config import settings
from app.extraction.extract import ExtractionError, extract_candidate
from app.extraction.schema import Education
from app.scoring.score import ScoringError, score_candidate
from eval.bias_metrics import delta_beyond_noise, noise_floor
from eval.run_eval import load_jds, load_resumes

ROOT = Path(__file__).resolve().parent.parent
N_RERUNS = 3
N_REPS = 6  # (JD, representative candidate) pairs — see design decision 4's "~6 JDs"

RESULTS_JSON = Path(__file__).resolve().parent / "bias_results.json"
RESULTS_MD = Path(__file__).resolve().parent / "bias_results.md"


def load_variants(name: str) -> list[dict]:
    return json.loads((ROOT / "data" / f"{name}.json").read_text())["pairs"]


def pick_reps(resumes: list[dict], jds: list[dict], n: int) -> list[tuple]:
    # First n JDs, each paired with one mid-tier candidate from a matching
    # category (falls back to any tier if no mid-tier candidate exists) —
    # deterministic, not random, so a rerun of this script targets the
    # same reps.
    reps = []
    for jd in jds[:n]:
        matching = [r for r in resumes if r["category"] == jd["category"] and r["tier"] == "mid"]
        if not matching:
            matching = [r for r in resumes if r["category"] == jd["category"]]
        if matching:
            reps.append((jd, matching[0]))
    return reps


def estimate_calls(n_reps: int) -> dict:
    name_pairs = len(load_variants("name_variants"))
    univ_pairs = len(load_variants("university_variants"))
    # Per rep: arm1 (name/visible) = pairs*2*N + N baseline; arm2 (name/blind) same shape;
    # arm3 (univ/blind) = univ_pairs*2*N + N baseline.
    per_rep = (name_pairs * 2 * N_RERUNS + N_RERUNS) * 2 + (univ_pairs * 2 * N_RERUNS + N_RERUNS)
    total = per_rep * n_reps
    return {"per_rep_calls": per_rep, "n_reps": n_reps, "total_calls": total,
            "name_pairs": name_pairs, "univ_pairs": univ_pairs, "n_reruns": N_RERUNS}


def _run_arm(candidate, jd_text, candidate_id, job_id, include_name: bool, variants: list[dict],
             swap_fn) -> list[dict]:
    baseline_scores = [score_candidate(candidate, jd_text, candidate_id, job_id, include_name).weighted_total
                        for _ in range(N_RERUNS)]
    floor = noise_floor(baseline_scores)

    results = []
    for pair in variants:
        variant_a = swap_fn(candidate, pair["variant_a"])
        variant_b = swap_fn(candidate, pair["variant_b"])
        scores_a = [score_candidate(variant_a, jd_text, candidate_id, job_id, include_name).weighted_total
                    for _ in range(N_RERUNS)]
        scores_b = [score_candidate(variant_b, jd_text, candidate_id, job_id, include_name).weighted_total
                    for _ in range(N_RERUNS)]
        stat = delta_beyond_noise(scores_a, scores_b, floor)
        results.append({"pair_id": pair["pair_id"], **stat})
    return results


def swap_name(candidate, new_name: str):
    return candidate.model_copy(update={"name": new_name})


def swap_university(candidate, new_institution: str):
    if candidate.education:
        new_edu = [Education(degree=candidate.education[0].degree, institution=new_institution)]
    else:
        new_edu = [Education(degree="B.A.", institution=new_institution)]
    return candidate.model_copy(update={"education": new_edu})


def run_bias_eval(reps: list[tuple]) -> dict:
    name_pairs = load_variants("name_variants")
    univ_pairs = load_variants("university_variants")
    per_rep_results = []

    for jd, resume in reps:
        try:
            base_candidate = extract_candidate(resume["text"], resume["candidate_id"])
        except ExtractionError:
            continue
        try:
            arm1 = _run_arm(base_candidate, jd["text"], resume["candidate_id"], jd["job_id"],
                             include_name=True, variants=name_pairs, swap_fn=swap_name)
            arm2 = _run_arm(base_candidate, jd["text"], resume["candidate_id"], jd["job_id"],
                             include_name=False, variants=name_pairs, swap_fn=swap_name)
            arm3 = _run_arm(base_candidate, jd["text"], resume["candidate_id"], jd["job_id"],
                             include_name=False, variants=univ_pairs, swap_fn=swap_university)
        except ScoringError:
            continue
        per_rep_results.append({
            "job_id": jd["job_id"], "candidate_id": resume["candidate_id"],
            "name_swap_visible_scorer": arm1,
            "name_swap_blind_scorer_sanity_check": arm2,
            "university_swap_blind_scorer": arm3,
        })

    return {"reps": per_rep_results}


def write_results_md(results: dict, estimate: dict) -> None:
    lines = [
        "# Project 3 Eval Results — Bias",
        "",
        f"Scoring model: `{settings.scoring_model}`. n={N_RERUNS} reruns per condition "
        "(current Claude models removed temperature/top_p/top_k, so identical input "
        "still varies — deltas are measured against a same-input noise floor, not a "
        "bare single-sample comparison).",
        "",
        "## Results",
    ]
    for rep in results["reps"]:
        lines.append(f"### {rep['job_id']} / {rep['candidate_id']}")
        lines.append("**Arm 1 — name swap, name-VISIBLE scorer (the actual bias measurement):**")
        for r in rep["name_swap_visible_scorer"]:
            lines.append(f"- {r['pair_id']}: delta={r['delta']:.3f}, "
                          f"threshold={r['noise_threshold']:.3f}, beyond_noise={r['beyond_noise']}")
        lines.append("**Arm 2 — name swap, BLIND scorer (sanity check, expect beyond_noise=False):**")
        for r in rep["name_swap_blind_scorer_sanity_check"]:
            lines.append(f"- {r['pair_id']}: delta={r['delta']:.3f}, "
                          f"threshold={r['noise_threshold']:.3f}, beyond_noise={r['beyond_noise']}")
        lines.append("**Arm 3 — university swap, BLIND scorer (required residual-sensitivity measurement):**")
        for r in rep["university_swap_blind_scorer"]:
            lines.append(f"- {r['pair_id']}: delta={r['delta']:.3f}, "
                          f"threshold={r['noise_threshold']:.3f}, beyond_noise={r['beyond_noise']}")
    lines += [
        "",
        "## Methodology notes",
        "- Name pairs follow the Bertrand & Mullainathan (2004) audit-study convention "
        "(see data/name_variants.json). University pairs are fictional, institution-type-"
        "matched (data/university_variants.json).",
        "- Arm 2's expected result IS a negative result (beyond_noise=False everywhere) — "
        "that's confirmation the production name-blind scorer works, not a null finding.",
        "- Scope: a narrow name/university-text sensitivity proxy on a synthetic corpus, "
        "not a full fairness audit.",
        f"- Call estimate before this run: {estimate}",
    ]
    RESULTS_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()

    resumes = load_resumes()
    jds = load_jds()
    reps = pick_reps(resumes, jds, N_REPS)
    estimate = estimate_calls(len(reps))

    print(f"Call estimate: {estimate}")
    if not args.run:
        print("Dry run (default) — pass --run to actually call the API. No requests were made.")
        return

    if not settings.anthropic_api_key:
        print("ANTHROPIC_API_KEY is not set — refusing to run with --run. "
              "Add it to .env first.", file=sys.stderr)
        sys.exit(1)

    results = run_bias_eval(reps)
    RESULTS_JSON.write_text(json.dumps({"results": results, "call_estimate": estimate}, indent=2),
                             encoding="utf-8")
    write_results_md(results, estimate)
    print(f"Wrote {RESULTS_JSON} and {RESULTS_MD}")


if __name__ == "__main__":
    main()
