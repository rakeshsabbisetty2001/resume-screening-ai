import os
import random

import requests
import streamlit as st

from resume_parsing import ExtractionError, extract_text

# On Streamlit Community Cloud, API_URL MUST be set in the app's Secrets
# (exposed to this process as a regular env var) — the localhost fallback
# only makes sense for local dev. Left unset on Cloud, every request fails
# with an opaque connection error and no hint why; documented as a
# required deploy step in the README, alongside Render's ANTHROPIC_API_KEY.
API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Resume Screening & Ranking", page_icon="📋")
st.title("📋 Resume Screening & Ranking")
st.caption(
    "Paste a job description and upload candidate resumes (.txt/.pdf/.docx) "
    "to get a ranked list with a per-criterion rationale for each score. "
    "Scoring is name-blind by default — the model never sees a candidate's "
    "name, only their skills, experience, and education."
)
st.info(
    "🔒 **Don't paste real personal data.** This is a portfolio demo — "
    "resumes are sent to the Anthropic API for scoring and are never "
    "written to disk by this app, but treat it like any other public demo: "
    "use a synthetic/sample resume, not your own or anyone else's.",
    icon="🔒",
)

job_description = st.text_area("Job description", height=150,
                                placeholder="Paste the job description here...")

uploaded_files = st.file_uploader("Candidate resumes (.txt/.pdf/.docx)",
                                   type=["txt", "pdf", "docx"],
                                   accept_multiple_files=True)

if st.button("Rank candidates", type="primary") and job_description and uploaded_files:
    # This app itself never writes upload bytes to disk — files are only
    # decoded/parsed to str in memory and sent to the API over HTTPS. Note
    # the bytes still live in Streamlit's own session-scoped
    # UploadedFileManager for the duration of the session regardless of
    # what this code does with its local reference to `uploaded_files` —
    # that's Streamlit's mechanism, not something an app-level `= None`
    # rebind changes.
    resumes = []
    for f in uploaded_files:
        try:
            resumes.append({"name": f.name, "text": extract_text(f.name, f.read())})
        except ExtractionError as e:
            st.warning(f"{f.name}: {e} — skipped.")

    results = []
    progress = st.progress(0.0)
    for i, r in enumerate(resumes):
        try:
            try:
                resp = requests.post(f"{API_URL}/score",
                                      json={"resume_text": r["text"], "job_description": job_description},
                                      timeout=100)
            except requests.Timeout:
                st.error(f"{r['name']}: API timed out (likely a free-tier cold start) — try again.")
                continue
            except requests.RequestException as e:
                st.error(f"{r['name']}: could not reach the API ({e}).")
                continue
            if resp.status_code != 200:
                try:
                    detail = resp.json().get("detail", "Something went wrong.")
                except ValueError:
                    detail = f"Something went wrong (HTTP {resp.status_code})."
                st.warning(f"{r['name']}: {detail}")
                continue
            results.append({"file_name": r["name"], **resp.json()})
        finally:
            # Always advance, success or failure — an earlier version only
            # advanced on the success path, so the bar stalled partway
            # through whenever a resume errored.
            progress.progress((i + 1) / len(resumes))
    progress.empty()

    if results:
        # Same tie-break as app/scoring/score.py::rank_candidates — a
        # plain sort here resolves ties in upload order, which is exactly
        # the "whichever tier/candidate the caller happened to list first
        # wins" bias that function's job_id-seeded shuffle exists to kill.
        # Duplicated (not imported) rather than importing app.scoring —
        # that would pull anthropic/pydantic into ui/requirements.txt,
        # which is deliberately scoped to streamlit+requests only (see
        # that file's comment). Keep this in sync with rank_candidates if
        # that function's tie-break logic ever changes.
        job_id = job_description  # ties only need a stable seed per JD, not the API's sha256 job_id
        rng = random.Random(job_id)
        shuffled = results[:]
        rng.shuffle(shuffled)
        ranked = sorted(shuffled, key=lambda x: x["weighted_total"], reverse=True)
        st.subheader("Ranked candidates")
        st.table([{"Rank": i + 1, "File": r["file_name"], "Score": round(r["weighted_total"], 2)}
                   for i, r in enumerate(ranked)])

        for i, r in enumerate(ranked):
            with st.expander(f"#{i + 1} — {r['file_name']} (score: {r['weighted_total']:.2f})"):
                for criterion, detail in r["rubric"].items():
                    st.markdown(f"**{criterion.replace('_', ' ').title()}**: {detail['score']}/5")
                    st.caption(detail["rationale"])
