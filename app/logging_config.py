import json
import logging
import sys


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "level": record.levelname,
            "logger": record.name,
            "time": self.formatTime(record),
            "message": record.getMessage(),
        }
        if record.exc_info:
            # Deliberately NOT self.formatException(record.exc_info) here.
            # sec-filings-rag's formatter does that, but this app's exceptions
            # can carry resume/JD text (e.g. a Pydantic ValidationError's
            # message embeds `input_value=...`, and uvicorn logs unhandled
            # request exceptions through this same root handler) — a full
            # traceback would be exactly the PII leak the call-site-discipline
            # comment below claims to avoid. Log the exception type only.
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        if hasattr(record, "extra_fields"):
            payload.update(record.extra_fields)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(logging.INFO)

    # Route uvicorn's own loggers through this handler too. uvicorn's
    # default config gives "uvicorn"/"uvicorn.error"/"uvicorn.access" their
    # own handlers with propagate=False — so an unhandled exception that
    # ServerErrorMiddleware re-raises after app/main.py's catch-all handler
    # returns gets logged by uvicorn's own plain-text formatter, bypassing
    # this module's exc_type-only JsonFormatter entirely. Without this, the
    # PII guardrail above only holds for exceptions logged through
    # application code, not ones that escape past it. Verified with a real
    # traceback containing a name/SSN before adding this — it printed in
    # full via uvicorn's formatter until these loggers were silenced.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logger = logging.getLogger(name)
        logger.handlers = []
        logger.propagate = True


# PII guardrail, mostly by discipline, not by filter — a formatter-level
# scrub can't reliably tell "resume text" from any other string, so
# JsonFormatter only ever emits the exception TYPE (never formatException's
# full traceback, unlike sec-filings-rag's formatter) and configure_logging
# additionally silences uvicorn's own loggers so a re-raised unhandled
# exception can't bypass this module through uvicorn's separate handler.
# The remaining guardrail lives at the call site: application code must
# never put raw resume/JD text into an exception's message or args (raise
# with a candidate/job id only), and must never pass raw text into
# `extra_fields`. Only ids, counts, model names, and latency belong here.
def log_extraction(candidate_id: str, model: str, tokens: int, latency_ms: float,
                    stop_reason: str) -> None:
    logging.getLogger("extraction").info(
        "candidate_extracted",
        extra={"extra_fields": {
            "candidate_id": candidate_id,
            "model": model,
            "tokens": tokens,
            "latency_ms": round(latency_ms, 1),
            "stop_reason": stop_reason,
        }},
    )


def log_scoring(candidate_id: str, job_id: str, model: str, tokens: int,
                 latency_ms: float, total_score: float) -> None:
    logging.getLogger("scoring").info(
        "candidate_scored",
        extra={"extra_fields": {
            "candidate_id": candidate_id,
            "job_id": job_id,
            "model": model,
            "tokens": tokens,
            "latency_ms": round(latency_ms, 1),
            "total_score": total_score,
        }},
    )
