import os

# app/main.py refuses to import (raises RuntimeError) with no API key set —
# same fail-fast philosophy as sec-filings-rag. Set a dummy one before any
# test module imports app.main/app.config, so offline tests (which never
# call the real API — every anthropic client call is monkeypatched) can
# still import the app. A distinct, named sentinel (not just "any string")
# so the live smoke tests (test_extraction.py, test_scoring.py) can tell
# "a real key is set" from "conftest filled in a placeholder" and skip
# correctly either way — a bare `if not os.environ.get(...)` check would
# see this dummy value as present and try a real API call with it.
DUMMY_API_KEY = "test-key-for-offline-tests"
os.environ.setdefault("ANTHROPIC_API_KEY", DUMMY_API_KEY)


def has_real_api_key() -> bool:
    key = os.environ.get("ANTHROPIC_API_KEY")
    return bool(key) and key != DUMMY_API_KEY
