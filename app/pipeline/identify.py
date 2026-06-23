"""Core recognition: a vision LLM reads a cover → structured CoverGuess.

This replaces OCR-string-parsing as the primary reader. A multimodal model is
robust to angle/blur/glare/occlusion in ways token-level OCR parsing never is,
and it can use world knowledge of real books to recover the true title from
partial or stylized text. OCR and catalog APIs become downstream *validation*,
not the parser. Every accuracy lever here (the prompt, the model, the provider)
is global — it improves recognition for all books at once.

The provider is pluggable: a free Gemini backend or a paid Claude backend, both
returning the same `CoverGuess`. Switch via `VISION_PROVIDER` / which key is set.
"""

from __future__ import annotations

import logging

from ..config import Config
from .providers import claude_provider, gemini_provider, openai_compat_provider
from .schema import CoverGuess

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are an expert at identifying books from a photograph of the cover or spine.
The photo may be angled, blurry, low-light, glare-washed, or partially obscured.

Your job, for ANY book in any language:
1. Read every piece of text legible on the cover, verbatim, into `visible_text`.
2. Determine the book's ACTUAL title and author — not just the largest words.
   Cover typography is decorative: the title may be stylized, split across lines,
   or smaller than a series name, tagline, or imprint. Use your knowledge of real,
   published books to resolve partial or distorted text to the real title (e.g.
   a worn "...REAT GATSB." with "FITZGERALD" below is The Great Gatsby by
   F. Scott Fitzgerald).
3. If several readings are plausible, list them in `candidates`, most likely first.
4. Calibrate `confidence` honestly. If you cannot read enough to be reasonably
   sure, return a LOW confidence and leave `title`/`author` empty rather than
   guessing a book that may not exist. Never invent a title to fill the field.
5. Set `is_book_cover` false if the image is not a book (e.g. a wall, a person).

Do not output ISBNs, descriptions, prices, or store/library UI text as the title —
those are handled separately. Distinguish the author from blurb authors quoted in
review snippets ("...gripping" —SOME OTHER AUTHOR) which are NOT the book's author."""

USER_PROMPT = (
    "Identify this book. Read the cover and return the structured result. "
    "Prefer the real, published title/author over the most visually prominent words."
)


def identify_cover(image_bytes: bytes, cfg: Config) -> tuple[CoverGuess, str]:
    """Run the configured vision provider on a (cropped) cover image.

    Returns (guess, provider_name).
    """
    provider = cfg.resolved_provider()
    if provider == "gemini":
        guess = gemini_provider.run(image_bytes, cfg, SYSTEM_PROMPT, USER_PROMPT)
    elif provider == "openai":
        guess = openai_compat_provider.run(image_bytes, cfg, SYSTEM_PROMPT, USER_PROMPT)
    else:
        guess = claude_provider.run(image_bytes, cfg, SYSTEM_PROMPT, USER_PROMPT)
    return guess.clamped(), provider
