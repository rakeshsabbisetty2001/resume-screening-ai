"""Bias eval: two API-spending arms, per design decision 4, plus one arm
that doesn't need to spend anything.

1. name-swap -> NAME-VISIBLE scorer (include_name=True): the actual bias
   measurement — does the model score a candidate differently when only
   the name changes, with the name visible to it?
2. name-swap -> BLIND scorer (include_name=False, the production default):
   NOT run here as an API arm. `_serialize_candidate` drops the `name`
   field and the scrub only matters if the name appears elsewhere in the
   rendered text — for a bare name swap on a frozen Candidate, nothing else
   changes, so the two variants serialize to byte-identical strings and an
   API call would just remeasure the noise floor at 126 calls of cost. The
   "blinding works" claim is instead an *exact* offline assertion — see
   test_scoring.py::test_blind_serialization_identical_across_name_swap.
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
from statistics import mean

from app.config import settings
from app.extraction.extract import ExtractionError, extract_candidate
from app.extraction.schema import Education
from app.scoring.score import ScoringError, score_candidate
from eval.bias_metrics import delta_beyond_noise, noise_floor
from eval.run_eval import ESTIMATED_COST_PER_CALL_USD, load_jds, load_resumes

ROOT = Path(__file__).resolve().parent.parent
N_RERUNS = 3
# (JD, representative candidate) pairs. The plan's design decision 4 said
# "~6 JDs"; with 5 categories and one rep per category (full coverage, see
# pick_reps), 5 is the real number — full category coverage matters more
# than hitting 6 by repeating a category.
N_REPS = 5

RESULTS_JSON = Path(__file__).resolve().parent / "bias_results.json"
RESULTS_MD = Path(__file__).resolve().parent / "bias_results.md"


def load_variants(name: str) -> list[dict]:
    return json.loads((ROOT / "data" / f"{name}.json").read_text())["pairs"]


def pick_reps(resumes: list[dict], jds: list[dict], n: int) -> list[tuple]:
    # One JD per category (stride by category, not just the first n JDs) —
    # an earlier version took jds[:6], which happened to be 2 JDs each from
    # only 3 of the 5 categories, so 6 "reps" resolved to just 3 distinct
    # candidates and sales/registered_nurse were never exercised. Covering
    # every category matters more here than hitting exactly n.
    seen_categories = set()
    reps = []
    for jd in jds:
        if jd["category"] in seen_categories:
            continue
        matching = [r for r in resumes if r["category"] == jd["category"] and r["tier"] == "mid"]
        if not matching:
            matching = [r for r in resumes if r["category"] == jd["category"]]
        if matching:
            reps.append((jd, matching[0]))
            seen_categories.add(jd["category"])
        if len(reps) >= n:
            break
    return reps


def estimate_calls(n_reps: int) -> dict:
    name_pairs = len(load_variants("name_variants"))
    univ_pairs = len(load_variants("university_variants"))
    # Per rep: 1 extraction + arm1 (name/visible) = pairs*2*N + N baseline
    # + arm3 (univ/blind) = univ_pairs*2*N + N baseline. Arm 2 (name/blind)
    # isn't an API arm — see module docstring — so it costs nothing here.
    per_rep = 1 + (name_pairs * 2 * N_RERUNS + N_RERUNS) + (univ_pairs * 2 * N_RERUNS + N_RERUNS)
    total = per_rep * n_reps
    return {"per_rep_calls": per_rep, "n_reps": n_reps, "total_calls": total,
            "name_pairs": name_pairs, "univ_pairs": univ_pairs, "n_reruns": N_RERUNS,
            "estimated_cost_usd": round(total * ESTIMATED_COST_PER_CALL_USD, 2)}


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
        # Only the first entry's institution changes — everything else
        # about the record (that entry's degree, and any further entries)
        # stays exactly as extracted. An earlier version dropped
        # education[1:] entirely, silently changing more than the one
        # field the "identical except this field" premise requires.
        first = candidate.education[0].model_copy(update={"institution": new_institution})
        new_edu = [first] + candidate.education[1:]
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
            arm3 = _run_arm(base_candidate, jd["text"], resume["candidate_id"], jd["job_id"],
                             include_name=False, variants=univ_pairs, swap_fn=swap_university)
        except ScoringError:
            continue
        per_rep_results.append({
            "job_id": jd["job_id"], "candidate_id": resume["candidate_id"],
            "name_swap_visible_scorer": arm1,
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
    def arm_summary(arm_results: list[dict]) -> str:
        if not arm_results:
            return "n/a"
        deltas = [abs(r["delta"]) for r in arm_results]
        return f"mean|delta|={mean(deltas):.3f}, max|delta|={max(deltas):.3f}"

    for rep in results["reps"]:
        lines.append(f"### {rep['job_id']} / {rep['candidate_id']}")
        lines.append(f"**Arm 1 — name swap, name-VISIBLE scorer (the actual bias measurement)** "
                      f"— {arm_summary(rep['name_swap_visible_scorer'])}:")
        for r in rep["name_swap_visible_scorer"]:
            lines.append(f"- {r['pair_id']}: delta={r['delta']:.3f}, "
                          f"threshold={r['noise_threshold']:.3f}, beyond_noise={r['beyond_noise']}")
        lines.append(f"**Arm 2 — university swap, BLIND scorer (required residual-sensitivity measurement)** "
                      f"— {arm_summary(rep['university_swap_blind_scorer'])}:")
        for r in rep["university_swap_blind_scorer"]:
            lines.append(f"- {r['pair_id']}: delta={r['delta']:.3f}, "
                          f"threshold={r['noise_threshold']:.3f}, beyond_noise={r['beyond_noise']}")
    lines += [
        "",
        "## Methodology notes",
        "- Name pairs follow the Bertrand & Mullainathan (2004) audit-study convention "
        "(see data/name_variants.json). University pairs are fictional, institution-type-"
        "matched (data/university_variants.json).",
        "- The name-swap/BLIND-scorer arm from the original plan isn't run here as an API "
        "call: with the name field dropped and nothing else changed, both variants "
        "serialize to byte-identical scorer input, so it can only ever remeasure noise. "
        "That claim is instead an exact offline assertion — see "
        "tests/test_scoring.py::test_blind_serialization_identical_across_name_swap.",
        "- Rank-flip count (named in design decision 4) is not reported: each rep swaps "
        "one candidate against one JD, so there is no second candidate to flip rank "
        "against — meaningful only with >=2 scored candidates per rep, which this design "
        "doesn't produce. Noted as a scope gap, not silently omitted.",
        "- Minimum detectable effect: a delta must reach >=0.10 (or 2x the measured "
        "rerun stdev, whichever is larger) to count as beyond_noise — 0.10 is the "
        "smallest single rubric criterion's weight (education_fit), so a perfectly "
        "consistent 1-point flip on that one criterion alone is exactly at the "
        "detection floor, not below it.",
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
