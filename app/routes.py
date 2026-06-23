"""HTTP surface: /health and /identify."""

from __future__ import annotations

import base64
import binascii
import logging

from flask import Blueprint, current_app, jsonify, request

log = logging.getLogger(__name__)
bp = Blueprint("lens", __name__)


@bp.get("/health")
def health():
    cfg = current_app.config["LENS"]
    provider = cfg.resolved_provider()
    model = {
        "gemini": cfg.gemini_model,
        "openai": ", ".join(cfg.openai_compat_model_list()[:2]),
        "claude": cfg.model_id,
    }.get(provider, cfg.model_id)
    return jsonify(
        {
            "ok": True,
            "provider": provider,
            "model": model,
            "anthropic_key": bool(cfg.anthropic_api_key),
            "gemini_key": bool(cfg.gemini_api_key),
            "openai_compat_key": bool(cfg.openai_compat_api_key),
            "google_books_key": bool(cfg.google_books_api_key),
            "yolo_enabled": cfg.yolo_enabled,
        }
    )


def _read_image_bytes() -> bytes | None:
    """Accept either multipart `image` upload or JSON `image_base64`."""
    if "image" in request.files:
        return request.files["image"].read()

    payload = request.get_json(silent=True) or {}
    b64 = payload.get("image_base64") or payload.get("image")
    if not b64:
        return None
    if "," in b64 and b64.strip().startswith("data:"):
        b64 = b64.split(",", 1)[1]  # strip data: URI prefix
    try:
        return base64.b64decode(b64, validate=True)
    except (binascii.Error, ValueError):
        return None


@bp.post("/identify")
def identify():
    image_bytes = _read_image_bytes()
    if not image_bytes:
        return jsonify({"status": "error", "message": "No image provided."}), 400

    debug = request.args.get("debug") in {"1", "true"} or bool(
        (request.get_json(silent=True) or {}).get("debug")
    )

    # Imported lazily so /health stays up even if heavy deps are misconfigured.
    from .pipeline.orchestrate import identify_book

    cfg = current_app.config["LENS"]
    try:
        cfg.require_provider_key()
        response = identify_book(image_bytes, cfg, debug=debug)
    except Exception as exc:  # noqa: BLE001 - surface a clean error to the client
        log.exception("identify failed")
        return jsonify({"status": "error", "message": str(exc)}), 500

    return jsonify(response.model_dump(exclude_none=False))


@bp.get("/youtube_search")
def youtube_search():
    """Find a YouTube review video for a book (no API key — scrapes search HTML).

    Used by the app's book result modal to embed a "Video Review".
    """
    import re

    import requests

    q = (request.args.get("q") or "").strip()
    if not q:
        return jsonify({"video_id": None, "embed_url": None})
    try:
        r = requests.get(
            "https://www.youtube.com/results",
            params={"search_query": q, "sp": "EgIQAQ%3D%3D"},  # filter: type = Video
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.9",
            },
            timeout=8,
        )
        r.raise_for_status()
        ids = re.findall(r'"videoId":"([\w-]{11})"', r.text)
        vid = ids[0] if ids else None
    except requests.RequestException as exc:
        log.warning("youtube_search failed: %s", exc)
        vid = None

    return jsonify(
        {
            "video_id": vid,
            "embed_url": f"https://www.youtube.com/embed/{vid}" if vid else None,
            "watch_url": f"https://www.youtube.com/watch?v={vid}" if vid else None,
            "search_url": "https://www.youtube.com/results?search_query="
            + requests.utils.quote(q),
        }
    )
