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


# PII guardrail, by discipline not by filter: this formatter still dumps
# exc_info (tracebacks) same as sec-filings-rag's, and a formatter-level
# scrub can't reliably tell "resume text" from any other string. So the
# actual guardrail lives at the call site: application code must never put
# raw resume/JD text into an exception's message or args (raise with a
# candidate/job id only), and must never pass raw text into `extra_fields`.
# Only ids, counts, model names, and latency belong in these logs.
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
