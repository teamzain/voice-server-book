"""Typed contracts shared across the pipeline.

Two groups:

* LLM-facing models (`GuessCandidate`, `CoverGuess`) — the structured-output
  schema Claude fills in. Kept flat and constraint-free so the schema the SDK
  sends stays within structured-output limits; ranges are clamped in code.
* Result models (`BookResult`, `IdentifyResponse`, diagnostics) — what the
  server assembles and returns to the app.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ─── LLM-facing (Claude structured output) ───────────────────────────────────

class GuessCandidate(BaseModel):
    title: str = Field(description="A plausible book title read from the cover.")
    author: str = Field(default="", description="Author for this candidate, if visible.")
    confidence: float = Field(description="0..1 confidence this is the correct book.")


class CoverGuess(BaseModel):
    """Claude's read of a single cover image."""

    is_book_cover: bool = Field(
        description="True if the image shows a book cover/spine. False for non-books."
    )
    visible_text: str = Field(
        default="",
        description="All text legible on the cover, verbatim, line by line.",
    )
    title: str = Field(
        default="",
        description="Best guess of the actual book title. Empty string if unsure.",
    )
    author: str = Field(
        default="",
        description="Best guess of the author. Empty string if unsure.",
    )
    series: str = Field(default="", description="Series name, if shown.")
    language: str = Field(default="", description="Primary language of the cover text.")
    confidence: float = Field(
        description="0..1 confidence in the title/author above. Be honest; low when unsure."
    )
    candidates: list[GuessCandidate] = Field(
        default_factory=list,
        description="Alternative title/author readings, most likely first.",
    )
    notes: str = Field(
        default="",
        description="Brief reasoning, e.g. what was occluded, blurred, or ambiguous.",
    )

    def clamped(self) -> "CoverGuess":
        """Return a copy with confidences clamped to [0, 1]."""

        def c(v: float) -> float:
            return max(0.0, min(1.0, float(v)))

        return self.model_copy(
            update={
                "confidence": c(self.confidence),
                "candidates": [
                    cand.model_copy(update={"confidence": c(cand.confidence)})
                    for cand in self.candidates
                ],
            }
        )


# ─── Result models (server → app) ────────────────────────────────────────────

class BookResult(BaseModel):
    """An enriched, catalog-grounded book."""

    title: str
    authors: list[str] = Field(default_factory=list)
    subtitle: str = ""
    description: str = ""
    description_generated: bool = False  # True when the summary is AI-generated
    isbn_10: str = ""
    isbn_13: str = ""
    publisher: str = ""
    published_date: str = ""
    page_count: int | None = None
    categories: list[str] = Field(default_factory=list)
    average_rating: float | None = None
    ratings_count: int | None = None
    thumbnail_url: str = ""
    preview_url: str = ""
    info_url: str = ""
    source: str = ""  # "google_books" | "open_library"
    match_score: float = 0.0  # how well this catalog hit matched the cover (0..1)


class StageRecord(BaseModel):
    name: str
    ms: float
    data: dict = Field(default_factory=dict)


class Diagnostics(BaseModel):
    scan_id: str
    created_at: str
    stages: list[StageRecord] = Field(default_factory=list)


class IdentifyResponse(BaseModel):
    scan_id: str
    status: Literal["matched", "candidates", "not_a_book", "unidentified", "error"]
    result: BookResult | None = None
    candidates: list[BookResult] = Field(default_factory=list)
    cover_guess: CoverGuess | None = None
    message: str = ""
    timings_ms: dict[str, float] = Field(default_factory=dict)
    diagnostics: Diagnostics | None = None
