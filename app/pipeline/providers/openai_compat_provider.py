"""Generic OpenAI-compatible vision backend.

Works with any service that speaks the OpenAI Chat Completions API:
  * OpenRouter   — https://openrouter.ai/api/v1  (free vision models, recommended)
  * GitHub Models — https://models.inference.ai.azure.com
  * Groq, Together, a local llama.cpp/vLLM server, etc.

Set OPENAI_COMPAT_BASE_URL / OPENAI_COMPAT_API_KEY / OPENAI_COMPAT_MODELS. The
model list is tried in order and rotates on quota/overload, so a per-model free
limit on one rolls over to the next. Free models don't all honor structured
output, so we instruct the JSON shape in the prompt and parse leniently.
"""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import time

import requests

from ...config import Config
from ..schema import CoverGuess
from .media import media_type

log = logging.getLogger(__name__)

# Free models are flaky: some 429 instantly, some hang. Don't retry the same model
# (rotation across models is the resilience) and cap how long we'll wait on one
# before moving on, so a single hanging model can't stall the whole scan.
_RETRYABLE_5XX = {500, 502, 503}
_ATTEMPTS_PER_MODEL = 1
_VISION_TIMEOUT = 12
_SUMMARY_TIMEOUT = 12

JSON_HINT = """\
Respond with ONLY a JSON object (no markdown, no code fences) of this exact shape:
{"is_book_cover": true, "visible_text": "", "title": "", "author": "", "series": "",
 "language": "", "synopsis": "", "confidence": 0.0,
 "candidates": [{"title": "", "author": "", "confidence": 0.0}], "notes": ""}
"synopsis" is a 2-3 sentence overview of what the book is about."""


def _backoff(attempt: int) -> float:
    return min(5.0, 0.6 * (2 ** attempt)) + random.random() * 0.3


_discovered_cache: list[str] | None = None


def _resolve_models(cfg: Config) -> list[str]:
    """Explicit OPENAI_COMPAT_MODELS list, or auto-discovered free vision models."""
    explicit = cfg.openai_compat_model_list()
    if explicit and explicit != ["auto"]:
        return explicit
    global _discovered_cache
    if _discovered_cache is None:
        _discovered_cache = _discover_free_vision(cfg)
    return _discovered_cache


def _discover_free_vision(cfg: Config) -> list[str]:
    """Find currently-available free image-capable models (OpenRouter schema)."""
    if "openrouter.ai" not in cfg.openai_compat_base_url:
        return []  # the discovery shape below is OpenRouter-specific
    try:
        r = requests.get(
            f"{cfg.openai_compat_base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {cfg.openai_compat_api_key}"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json().get("data", [])
    except (requests.RequestException, ValueError) as exc:
        log.warning("openrouter model discovery failed: %s", exc)
        return []
    out: list[str] = []
    for m in data:
        mid = m.get("id", "")
        mods = (m.get("architecture", {}) or {}).get("input_modalities") or []
        pr = m.get("pricing", {}) or {}
        free = mid.endswith(":free") or (
            str(pr.get("prompt")) in ("0", "0.0") and str(pr.get("completion")) in ("0", "0.0")
        )
        # Skip non-chat models, and reasoning/omni models that fight strict JSON.
        bad = mid == "openrouter/free" or any(
            t in mid for t in ("safety", "guard", "lyria", "tts", "audio", "embed", "reasoning", "omni")
        )
        if "image" in mods and free and not bad:
            out.append(mid)
    # General instruction-tuned multimodal models first (best at cover reading + JSON).
    out.sort(key=lambda x: (0 if any(k in x for k in ("gemma", "qwen", "llama", "mistral", "pixtral", "nemotron")) else 1, x))
    log.info("discovered %d free vision models on openrouter: %s", len(out), out[:6])
    return out[:6]


def _request(model: str, messages: list[dict], cfg: Config, timeout: int) -> str | None:
    """One model's chat call. Returns content, or None to signal 'try next model'."""
    url = f"{cfg.openai_compat_base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {cfg.openai_compat_api_key}",
        "Content-Type": "application/json",
        # OpenRouter likes these; harmless elsewhere.
        "HTTP-Referer": "https://github.com/lens-book-scanner",
        "X-Title": "Lens Book Scanner",
    }
    for attempt in range(_ATTEMPTS_PER_MODEL):
        try:
            r = requests.post(
                url,
                headers=headers,
                json={"model": model, "messages": messages, "temperature": 0.2},
                timeout=timeout,
            )
            if r.status_code in _RETRYABLE_5XX:
                log.warning("openai-compat %s blip: %s", model, r.text[:120])
                time.sleep(_backoff(attempt))
                continue
            if not r.ok:
                # 429 (rate-limited) / 404 (gone) / other 4xx → next model now.
                log.warning("openai-compat rotating off %s: %s %s", model, r.status_code, r.text[:120])
                return None
            return r.json()["choices"][0]["message"]["content"]
        except (requests.RequestException, KeyError, ValueError) as exc:
            log.warning("openai-compat %s error: %s", model, exc)
            if attempt < _ATTEMPTS_PER_MODEL - 1:
                time.sleep(_backoff(attempt))
    return None


def _balanced_object(text: str) -> str:
    """Extract the first brace-balanced {...} object, respecting strings/escapes."""
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_str = esc = False
    for i in range(start, len(text)):
        c = text[i]
        if esc:
            esc = False
        elif c == "\\":
            esc = True
        elif c == '"':
            in_str = not in_str
        elif not in_str:
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return ""


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text).rstrip("`").strip()
    for candidate in (text, _balanced_object(text)):
        if not candidate:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:  # tolerate trailing commas
                return json.loads(re.sub(r",(\s*[}\]])", r"\1", candidate))
            except json.JSONDecodeError:
                continue
    raise ValueError("non-JSON content")


def run(image_bytes: bytes, cfg: Config, system_prompt: str, user_prompt: str) -> CoverGuess:
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")
    data_url = f"data:{media_type(image_bytes)};base64,{b64}"
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\n{JSON_HINT}"},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]
    models = _resolve_models(cfg)
    if not models:
        raise RuntimeError("No models available. Set OPENAI_COMPAT_MODELS or check the key.")

    last_err = "no models responded"
    for model in models:
        content = _request(model, messages, cfg, timeout=_VISION_TIMEOUT)
        if content is None:
            last_err = f"{model} unavailable"
            continue
        try:
            parsed = _parse_json(content)
        except ValueError:
            last_err = f"{model} returned unparseable JSON"
            log.warning("openai-compat %s: bad JSON, rotating to next model", model)
            continue
        # Free models sometimes omit fields; fill the required ones before validating.
        parsed.setdefault("is_book_cover", True)
        parsed.setdefault("confidence", 0.5)
        parsed.setdefault("candidates", [])
        try:
            return CoverGuess.model_validate(parsed)
        except Exception as exc:  # noqa: BLE001 - schema mismatch → try next model
            last_err = f"{model} schema mismatch: {exc}"
            log.warning("openai-compat %s: %s", model, last_err)
            continue
    raise RuntimeError(f"OpenAI-compatible provider failed: {last_err}")


def summarize(title: str, author: str, cfg: Config) -> str:
    prompt = (
        f'In two sentences, neutrally summarize the book "{title}"'
        f"{f' by {author}' if author else ''}. "
        "If you are not confident the book exists, reply with an empty string."
    )
    for model in _resolve_models(cfg):
        content = _request(model, [{"role": "user", "content": prompt}], cfg, timeout=_SUMMARY_TIMEOUT)
        if content:
            return content.strip()
    return ""
