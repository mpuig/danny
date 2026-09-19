"""Minimal client for the real Jev API (used for distillation and agreement evals)."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

API_URL = "https://api.typesafe.ai/v1/systemone"
RETRYABLE = (408, 429, 500, 502, 503, 529)


def api_key() -> str:
    if os.environ.get("TYPESAFE_API_KEY"):
        return os.environ["TYPESAFE_API_KEY"]
    env = Path(__file__).resolve().parents[2] / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line.startswith("TYPESAFE_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("TYPESAFE_API_KEY not found in environment or .env")


def ask_jev(key: str, state, question, model: str = "jev-latest") -> dict:
    """One question; returns the answer dict. Retries with exponential backoff
    on transient HTTP errors and socket timeouts."""
    return systemone(key, state, question, model)["answers"]["q"]


def systemone(key: str, state, question, model: str = "jev-latest") -> dict:
    """Full response (answers, reported model id, usage) for one question.
    Pin a versioned `model` (e.g. jev-1.13.0) for reproducible collection;
    the response's own `model` field reports what actually served it."""
    payload = {
        "state": state,
        "model": model,
        "questions": {
            "q": {
                "type": question.type,
                "instructions": question.instructions,
                **({"criteria": question.criteria} if question.criteria else {}),
            }
        },
    }
    req = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            if e.code in RETRYABLE and attempt < 4:
                time.sleep(2**attempt)
                continue
            raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt < 4:
                time.sleep(2**attempt)
                continue
            raise
