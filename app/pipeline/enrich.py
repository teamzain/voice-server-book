"""Finalize a chosen catalog match into a presentable, enriched BookResult.

The catalog hit already carries most metadata (description, ISBN, thumbnail,
links). This stage fills the one common gap — a missing description — with a
short, clearly-labelled AI summary, using whichever vision provider is active.
"""

from __future__ import annotations

from ..config import Config
from .providers import claude_provider, gemini_provider, openai_compat_provider
from .schema import BookResult, CoverGuess

_SUMMARIZERS = {
    "gemini": gemini_provider,
    "openai": openai_compat_provider,
    "claude": claude_provider,
}


def enrich(book: BookResult, guess: CoverGuess, cfg: Config) -> BookResult:
    if book.description or not cfg.summary_fallback:
        return book

    author = book.authors[0] if book.authors else guess.author
    provider = _SUMMARIZERS.get(cfg.resolved_provider(), claude_provider)
    summary = provider.summarize(book.title, author, cfg).strip()
    if not summary:
        return book

    return book.model_copy(
        update={"description": summary, "description_generated": True}
    )
