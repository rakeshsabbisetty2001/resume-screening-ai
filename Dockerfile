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
# it. --proxy-headers --forwarded-allow-ips='*' trusts Render's own
# X-Forwarded-For — matches TRUST_PROXY=true set in render.yaml; without
# both together the rate limiter would either trust a spoofable header
# (proxy-headers on, TRUST_PROXY off would be inert) or key every request
# behind Render's proxy IP (TRUST_PROXY on, proxy-headers off would key
# everyone on the same bucket) — see app/middleware/rate_limit.py.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers --forwarded-allow-ips='*'"]
