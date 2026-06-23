"""Gemini vision backend (free tier).

Uses the Generative Language REST API directly (no extra SDK) with
`responseSchema` for structured JSON output. Returns the same `CoverGuess` as the
Claude backend so the rest of the pipeline is provider-agnostic.

Get a free key at https://aistudio.google.com/apikey
"""

from __future__ import annotations

import base64
import json
import logging
import random
import time

import requests

from ...config import Config
from ..schema import CoverGuess
from .media import media_type

log = logging.getLogger(__name__)

_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
# Gemini's free tier returns these when momentarily overloaded / throttled.
_RETRY_STATUS = {429, 500, 503}
_MAX_ATTEMPTS = 3


def _backoff(attempt: int) -> float:
    return min(6.0, 0.7 * (2 ** attempt)) + random.random() * 0.3


def _generate(model: str, body: dict, cfg: Config, timeout: int) -> dict:
    """POST to a Gemini model with retry on transient overload/throttle."""
    url = f"{_BASE}/{model}:generateContent"
    last = ""
    for attempt in range(_MAX_ATTEMPTS):
        try:
            r = requests.post(url, params={"key": cfg.gemini_api_key}, json=body, timeout=timeout)
            if r.status_code in _RETRY_STATUS:
                last = f"{r.status_code} {r.text[:160]}"
                log.warning("gemini %s overloaded (attempt %d): %s", model, attempt + 1, last)
                if attempt < _MAX_ATTEMPTS - 1:
                    time.sleep(_backoff(attempt))
                continue
            r.raise_for_status()
            return r.json()
        except requests.RequestException as exc:
            last = str(exc)
            if attempt < _MAX_ATTEMPTS - 1:
                time.sleep(_backoff(attempt))
    raise RuntimeError(f"Gemini {model} unavailable after {_MAX_ATTEMPTS} attempts: {last}")


def _generate_with_fallback(body: dict, cfg: Config, timeout: int) -> dict:
    """Try the primary model, then the configured fallback if it stays overloaded."""
    models = [cfg.gemini_model]
    if cfg.gemini_fallback_model and cfg.gemini_fallback_model != cfg.gemini_model:
        models.append(cfg.gemini_fallback_model)
    last_exc: Exception | None = None
    for model in models:
        try:
            return _generate(model, body, cfg, timeout)
        except RuntimeError as exc:
            last_exc = exc
            log.warning("gemini falling back from %s", model)
    raise last_exc or RuntimeError("Gemini unavailable")

# Gemini Schema (OpenAPI subset; types are uppercase enums).
_COVER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "is_book_cover": {"type": "BOOLEAN"},
        "visible_text": {"type": "STRING"},
        "title": {"type": "STRING"},
        "author": {"type": "STRING"},
        "series": {"type": "STRING"},
        "language": {"type": "STRING"},
        "synopsis": {"type": "STRING"},
        "confidence": {"type": "NUMBER"},
        "candidates": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "title": {"type": "STRING"},
                    "author": {"type": "STRING"},
                    "confidence": {"type": "NUMBER"},
                },
                "required": ["title", "author", "confidence"],
            },
        },
        "notes": {"type": "STRING"},
    },
    "required": [
        "is_book_cover",
        "visible_text",
        "title",
        "author",
        "synopsis",
        "confidence",
        "candidates",
        "notes",
    ],
}


def run(image_bytes: bytes, cfg: Config, system_prompt: str, user_prompt: str) -> CoverGuess:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    body = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"inline_data": {"mime_type": media_type(image_bytes), "data": b64}},
                    {"text": user_prompt},
                ],
            }
        ],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": _COVER_SCHEMA,
            "temperature": 0.2,
        },
    }
    data = _generate_with_fallback(body, cfg, timeout=45)

    text = _extract_text(data)
    if not text:
        log.warning("gemini identify: empty response %s", json.dumps(data)[:300])
        return CoverGuess(is_book_cover=True, confidence=0.0, notes="gemini returned no content")

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        log.warning("gemini identify: non-JSON response: %s", text[:300])
        return CoverGuess(is_book_cover=True, confidence=0.0, notes="gemini returned non-JSON")

    return CoverGuess.model_validate(parsed)


def _extract_text(data: dict) -> str:
    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts).strip()


def summarize(title: str, author: str, cfg: Config) -> str:
    prompt = (
        f'In two sentences, neutrally summarize the book "{title}"'
        f"{f' by {author}' if author else ''}. "
        "If you are not confident the book exists, reply with an empty string."
    )
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    try:
        return _extract_text(_generate_with_fallback(body, cfg, timeout=30))
    except (RuntimeError, requests.RequestException) as exc:  # noqa: BLE001 - best-effort
        log.warning("gemini summarize failed: %s", exc)
        return ""
