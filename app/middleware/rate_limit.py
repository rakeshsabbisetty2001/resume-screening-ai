"""Ported from sec-filings-rag/app/middleware/rate_limit.py verbatim."""
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import settings


def _client_ip(request: Request) -> str:
    """Key the rate limit on the RIGHTMOST X-Forwarded-For entry, not the
    leftmost. Render (the first real proxy in front of this app) appends
    the true client IP as the last hop, which a client can't forge from
    outside; the leftmost entry is whatever the client itself sent."""
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[-1].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)
API_RATE_LIMIT = f"{settings.rate_limit_per_minute}/minute"
