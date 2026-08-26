from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    anthropic_api_key: str = ""

    @field_validator("anthropic_api_key", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: str) -> str:
        # pydantic-settings' own .env-file parser trims stray whitespace
        # around `KEY=value`, but `docker run --env-file` does NOT - it
        # passes the raw value through, including a leading space if the
        # .env file has one (e.g. "KEY= value"). httpx/httpcore then rejects
        # a header value with a leading space as an "illegal header value",
        # which surfaces as a generic connection error with no indication
        # it's actually a whitespace bug. Stripping here makes the app
        # correct regardless of which env-loading mechanism is in play.
        # (Same bug, same fix, as sec-filings-rag/app/config.py.)
        return v.strip() if isinstance(v, str) else v

    rate_limit_per_minute: int = 10
    # Comfortably above the 100k/20k char field caps in app/main.py plus
    # JSON escaping overhead — was 32,000 back when the UI only accepted
    # pasted plain text; PDF/DOCX extraction produces bigger payloads for
    # the same real resume (see app/main.py's ExtractRequest comment).
    max_body_bytes: int = 300_000
    # False until an actual reverse proxy sits in front (Phase 7 deploy).
    # With no proxy, X-Forwarded-For is entirely client-supplied — trusting
    # it here would let every request pick its own rate-limit bucket via a
    # forged header (verified: 4 requests, 4 different XFF values, all 200
    # under a 2/minute limit). Flip to true only once deployed behind a
    # real proxy that overwrites/appends this header itself.
    trust_proxy: bool = False

    # Pinned model IDs so every number in eval/results.md and
    # eval/bias_results.md stays attributable to a specific model version
    # (sec-filings-rag's judge pin, same convention). Current model IDs
    # carry no date suffix — a dated string here would be wrong, not more
    # precise. Effort/thinking config lives next to the call site in
    # extraction.py / scoring.py, not here.
    extraction_model: str = "claude-sonnet-5"
    scoring_model: str = "claude-sonnet-5"

    # extra="ignore": later phases put PORT/LOG_LEVEL etc. in .env for the
    # platform, and pydantic-settings 2.x defaults to extra="forbid" — an
    # unmapped env key would otherwise crash the app at import with a
    # ValidationError that doesn't name the real cause.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
