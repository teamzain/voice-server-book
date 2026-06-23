"""Runtime configuration, loaded once from the environment.

Everything tunable about the pipeline lives here so behaviour can be changed
globally (model, effort, confidence thresholds, detection toggles) without
touching per-stage code — there are deliberately no per-book knobs.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


def _flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass(frozen=True)
class Config:
    # Vision provider: "auto" | "claude" | "gemini".
    #   auto = use Gemini (free tier) when a Gemini key is set, else Claude.
    vision_provider: str = field(default_factory=lambda: os.getenv("VISION_PROVIDER", "auto"))

    # Claude (paid, highest accuracy)
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    model_id: str = field(default_factory=lambda: os.getenv("MODEL_ID", "claude-opus-4-8"))
    identify_effort: str = field(default_factory=lambda: os.getenv("IDENTIFY_EFFORT", "high"))

    # Gemini (free tier; get a key at https://aistudio.google.com/apikey)
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    # Tried when the primary model is overloaded (503) after retries.
    gemini_fallback_model: str = field(
        default_factory=lambda: os.getenv("GEMINI_FALLBACK_MODEL", "gemini-2.0-flash")
    )

    # OpenAI-compatible provider — works with OpenRouter (free vision models),
    # GitHub Models, Groq, Together, a local server, etc. Just set the base URL,
    # key, and a comma-separated model list (tried in order, rotating on quota).
    openai_compat_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_COMPAT_API_KEY", ""))
    openai_compat_base_url: str = field(
        default_factory=lambda: os.getenv("OPENAI_COMPAT_BASE_URL", "https://openrouter.ai/api/v1")
    )
    openai_compat_models: str = field(
        default_factory=lambda: os.getenv(
            "OPENAI_COMPAT_MODELS",
            "meta-llama/llama-3.2-11b-vision-instruct:free,google/gemini-2.0-flash-exp:free",
        )
    )

    # Generate a 2-sentence summary with the LLM when the catalog has no description.
    summary_fallback: bool = field(default_factory=lambda: _flag("SUMMARY_FALLBACK", True))

    # Catalog
    google_books_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_BOOKS_API_KEY", ""))
    catalog_timeout_s: float = field(default_factory=lambda: _float("CATALOG_TIMEOUT_S", 8.0))

    # Cover detection
    yolo_enabled: bool = field(default_factory=lambda: _flag("YOLO_BOOK_COVER_ENABLED", False))
    yolo_model: str = field(default_factory=lambda: os.getenv("YOLO_BOOK_MODEL", ""))
    # Largest dimension the cover crop is resized to before the LLM call.
    max_image_dim: int = field(default_factory=lambda: int(os.getenv("MAX_IMAGE_DIM", "1600")))
    # Light CLAHE contrast boost; off by default (the vision model handles lighting well).
    enhance_cover: bool = field(default_factory=lambda: _flag("ENHANCE_COVER", False))

    # Confidence gating (global, not per-book). Above `auto`, return directly.
    # Between `min` and `auto`, return candidates for a quick user confirmation.
    auto_accept_confidence: float = field(default_factory=lambda: _float("AUTO_ACCEPT_CONFIDENCE", 0.78))
    min_match_score: float = field(default_factory=lambda: _float("MIN_MATCH_SCORE", 0.45))

    # Diagnostics
    diagnostics_dir: str = field(default_factory=lambda: os.getenv("DIAGNOSTICS_DIR", "./_diagnostics"))
    diagnostics_save_crops: bool = field(default_factory=lambda: _flag("DIAGNOSTICS_SAVE_CROPS", True))

    # Server
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "3001")))

    def openai_compat_model_list(self) -> list[str]:
        return [m.strip() for m in self.openai_compat_models.split(",") if m.strip()]

    def resolved_provider(self) -> str:
        if self.vision_provider in {"claude", "gemini", "openai"}:
            return self.vision_provider
        # auto: first provider with a key (Gemini → OpenAI-compatible → Claude)
        if self.gemini_api_key:
            return "gemini"
        if self.openai_compat_api_key:
            return "openai"
        return "claude"

    def require_provider_key(self) -> None:
        provider = self.resolved_provider()
        if provider == "gemini" and not self.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to server/.env."
            )
        if provider == "openai" and not self.openai_compat_api_key:
            raise RuntimeError(
                "OPENAI_COMPAT_API_KEY is not set. Get a free OpenRouter key at "
                "https://openrouter.ai/keys and add it to server/.env "
                "(with VISION_PROVIDER=openai)."
            )
        if provider == "claude" and not self.anthropic_api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Copy server/.env.example to server/.env "
                "and add your key (or set GEMINI_API_KEY / OPENAI_COMPAT_API_KEY)."
            )


_config: Config | None = None


def get_config() -> Config:
    global _config
    if _config is None:
        _config = Config()
    return _config
