"""Claude vision backend (paid, highest accuracy).

Uses `messages.parse()` for guaranteed structured output, adaptive thinking, and
configurable effort. The schema is generated from the `CoverGuess` Pydantic model.
"""

from __future__ import annotations

import base64
import logging

import anthropic

from ...config import Config
from ..schema import CoverGuess
from .media import media_type

log = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None
_client_key: str = ""


def _get_client(cfg: Config) -> anthropic.Anthropic:
    global _client, _client_key
    if _client is None or _client_key != cfg.anthropic_api_key:
        _client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        _client_key = cfg.anthropic_api_key
    return _client


def run(image_bytes: bytes, cfg: Config, system_prompt: str, user_prompt: str) -> CoverGuess:
    client = _get_client(cfg)
    b64 = base64.standard_b64encode(image_bytes).decode("ascii")

    # Reading a cover is a perception + knowledge task — it doesn't need extended
    # thinking. Keep it fast (no thinking, low effort, small output) so Claude is a
    # snappy fallback when the free providers are rate-limited.
    response = client.messages.parse(
        model=cfg.model_id,
        max_tokens=2000,
        thinking={"type": "disabled"},
        output_config={"effort": "low"},
        output_format=CoverGuess,
        system=system_prompt,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type(image_bytes),
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": user_prompt},
                ],
            }
        ],
    )

    guess = response.parsed_output
    if guess is None:
        log.warning("claude identify: no structured output (stop_reason=%s)", response.stop_reason)
        return CoverGuess(
            is_book_cover=True,
            confidence=0.0,
            notes=f"model returned no structured output (stop_reason={response.stop_reason})",
        )
    return guess


def summarize(title: str, author: str, cfg: Config) -> str:
    try:
        client = _get_client(cfg)
        msg = client.messages.create(
            model=cfg.model_id,
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f'In two sentences, neutrally summarize the book "{title}"'
                        f"{f' by {author}' if author else ''}. "
                        "If you are not confident the book exists, reply with an empty string."
                    ),
                }
            ],
        )
        return "".join(b.text for b in msg.content if b.type == "text").strip()
    except Exception as exc:  # noqa: BLE001 - summary is best-effort
        log.warning("claude summarize failed: %s", exc)
        return ""
