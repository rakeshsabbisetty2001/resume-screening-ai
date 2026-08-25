import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field
from slowapi.errors import RateLimitExceeded

from app.config import settings
from app.extraction.extract import ExtractionError, extract_candidate
from app.logging_config import configure_logging
from app.middleware.body_limit import BodySizeLimitMiddleware
from app.middleware.rate_limit import API_RATE_LIMIT, _client_ip, limiter
from app.scoring.score import ScoringError, score_candidate

configure_logging()
logger = logging.getLogger("resume_screening.api")

if not settings.anthropic_api_key:
    raise RuntimeError(
        "ANTHROPIC_API_KEY is not set. A container that refuses to boot is far "
        "better than one that 500s on the first real request."
    )

app = FastAPI(title="Resume Screening & Ranking API")
app.state.limiter = limiter
app.add_middleware(BodySizeLimitMiddleware)


@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    logger.warning("rate limit exceeded", extra={"extra_fields": {"client": _client_ip(request)}})
    return JSONResponse(status_code=429, content={"detail": "Too many requests, slow down."},
                         headers={"Retry-After": "60"})


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # FastAPI's default handler echoes the offending `input` value back in
    # the 422 body — for this app that can be the raw resume/JD text a
    # candidate/user submitted. Strip it; keep the field path and message,
    # which is enough to fix a malformed request without echoing content.
    errors = [{k: v for k, v in err.items() if k != "input"} for err in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # logger.exception() logs exc_info, but app/logging_config.py's
    # JsonFormatter only emits the exception TYPE, never a formatted
    # traceback — so this can't leak request-derived text into logs the
    # way a naive formatter would (see logging_config.py's own comment).
    logger.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "Something went wrong."})


class ExtractRequest(BaseModel):
    resume_text: str = Field(min_length=20, max_length=20000)


class ScoreRequest(BaseModel):
    resume_text: str = Field(min_length=20, max_length=20000)
    job_description: str = Field(min_length=20, max_length=10000)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/extract")
@limiter.limit(API_RATE_LIMIT)
def extract(request: Request, body: ExtractRequest):
    # A per-request id, not derived from the submitted text — ExtractionError
    # messages only ever carry this id, never resume content (see
    # app/extraction/extract.py's PII contract).
    request_id = str(uuid.uuid4())
    try:
        return extract_candidate(body.resume_text, request_id)
    except ExtractionError as e:
        logger.warning("extraction failed", extra={"extra_fields": {"request_id": request_id}})
        return JSONResponse(status_code=422, content={"detail": str(e)})


@app.post("/score")
@limiter.limit(API_RATE_LIMIT)
def score(request: Request, body: ScoreRequest):
    request_id = str(uuid.uuid4())
    try:
        candidate = extract_candidate(body.resume_text, request_id)
    except ExtractionError as e:
        logger.warning("extraction failed", extra={"extra_fields": {"request_id": request_id}})
        return JSONResponse(status_code=422, content={"detail": str(e)})
    try:
        # include_name defaults to False — name-blind scoring is the
        # production path (app/scoring/score.py design decision 3).
        return score_candidate(candidate, body.job_description, request_id, request_id)
    except ScoringError as e:
        logger.warning("scoring failed", extra={"extra_fields": {"request_id": request_id}})
        return JSONResponse(status_code=422, content={"detail": str(e)})
