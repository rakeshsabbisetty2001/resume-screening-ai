"""Ported from sec-filings-rag/app/middleware/rate_limit.py verbatim."""
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.requests import Request

from app.config import settings


def _client_ip(request: Request) -> str:
    """Key the rate limit on the RIGHTMOST X-Forwarded-For entry — but only
    when settings.trust_proxy is on, meaning a real reverse proxy sits in
    front and appends the true client IP as the last hop. Without a proxy,
    X-Forwarded-For is entirely client-supplied and trusting it (rightmost
    or not) lets every request pick its own bucket by forging a new value —
    confirmed empirically before adding this gate. Falls back to the
    connection's actual remote address, which a client can't spoof.

    Rightmost is only correct if the deployed proxy topology has exactly
    one hop appending the real client IP — a claim this codebase can't
    verify without an actual deployment (unlike TRUST_PROXY=false's
    behavior, checked live in a container in Phase 7). If Render (or
    whatever's in front) rewrites XFF to a single value, leftmost and
    rightmost coincide and this works by luck; if there are 2+ appending
    hops, rightmost lands on an internal proxy IP and every real client
    collapses into one shared bucket — a silent availability bug, not a
    security one. Verify the real header shape post-deploy (log it once)
    before trusting this in production; see README's deploying section."""
    if settings.trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[-1].strip()
    return get_remote_address(request)


limiter = Limiter(key_func=_client_ip)
API_RATE_LIMIT = f"{settings.rate_limit_per_minute}/minute"
