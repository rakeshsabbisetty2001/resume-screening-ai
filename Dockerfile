FROM python:3.12-slim AS deps
WORKDIR /build
COPY requirements.txt .
# No vector DB / embedding model here (unlike sec-filings-rag) — extraction
# and scoring are direct Claude structured-output calls, so a plain
# install is correct with no CPU/GPU torch concern and no model-bake step.
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
COPY --from=deps /install /usr/local
COPY app/ app/

# Non-root user — same convention as sec-filings-rag's Dockerfile. No
# COPY-then-chown gotcha here (no committed data directory the app writes
# to at runtime), but running as non-root is a cheap default regardless.
RUN useradd --create-home --shell /bin/bash app
USER app
ENV HOME=/home/app

EXPOSE 8000
# Render (and most PaaS hosts) inject $PORT and expect the app to bind to
# it. Deliberately NOT passing --proxy-headers/--forwarded-allow-ips: an
# earlier version did, on the assumption it would pair with render.yaml's
# TRUST_PROXY=true — checked against uvicorn 0.52.4's actual behavior and
# found the two contradict each other instead of pairing. uvicorn's own
# X-Forwarded-For resolution (--forwarded-allow-ips) takes the LEFTMOST
# entry and rewrites request.client.host; app/middleware/rate_limit.py's
# TRUST_PROXY path reads the header directly and takes the RIGHTMOST —
# so --proxy-headers contributes nothing to rate limiting either way, and
# with TRUST_PROXY off it would silently turn get_remote_address's
# supposedly-unspoofable fallback into a spoofable one (uvicorn would have
# already overwritten request.client.host with the client's own forged
# leftmost value). Leaving proxy-headers off keeps TRUST_PROXY the single
# source of truth for who the header is trusted from. No absolute URLs
# are generated anywhere in this app, so the scheme-detection --proxy-headers
# would otherwise provide isn't needed either.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
