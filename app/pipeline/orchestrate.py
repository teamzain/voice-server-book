"""End-to-end pipeline: detect → identify → catalog → gate → enrich.

Confidence gating decides how much the user is asked to do:
  * strong, unambiguous match  → return it directly (zero taps)
  * plausible but unsure       → return ranked candidates for a one-tap confirm
  * nothing credible           → "unidentified" (still returns any weak hits)
All thresholds live in `Config` and apply to every book — no per-book rules.
"""

from __future__ import annotations

import logging

from ..config import Config
from .catalog import gather_candidates, title_similarity
from .detect import detect_cover
from .diagnostics import ScanDiagnostics
from .enrich import enrich
from .identify import identify_cover
from .schema import IdentifyResponse

log = logging.getLogger(__name__)

_AMBIGUOUS_GAP = 0.08   # if #1 and #2 are this close, ask the user
_MATCH_FLOOR = 0.70     # a "strong" catalog match


def identify_book(image_bytes: bytes, cfg: Config, debug: bool = False) -> IdentifyResponse:
    diag = ScanDiagnostics(cfg.diagnostics_dir, cfg.diagnostics_save_crops)

    # 1. Detect + crop the cover.
    with diag.stage("detect") as d:
        crop_bytes, det_meta = detect_cover(image_bytes, cfg)
        d.update(det_meta)
    crop_path = diag.save_crop(crop_bytes)
    if crop_path:
        diag.record("crop_saved", path=crop_path)

    # 2. Vision-LLM identification.
    with diag.stage("identify") as d:
        guess, provider = identify_cover(crop_bytes, cfg)
        d.update({"provider": provider, "guess": guess.model_dump()})

    if not guess.is_book_cover:
        return _finalize(
            diag, cfg,
            IdentifyResponse(
                scan_id=diag.scan_id,
                status="not_a_book",
                cover_guess=guess,
                message="That doesn't look like a book cover.",
                timings_ms=diag.timings,
            ),
            debug,
        )

    # 3. Ground against real catalogs.
    with diag.stage("catalog") as d:
        ranked, trace = gather_candidates(guess, cfg)
        d.update({"query_trace": trace, "ranked": [r.model_dump() for r in ranked]})

    # 4. Gate on confidence + match strength.
    status, result, candidates = _gate(guess, ranked, cfg)
    diag.record(
        "gate",
        decision=status,
        guess_confidence=guess.confidence,
        best_match_score=ranked[0].match_score if ranked else None,
    )

    # 5. Enrich the accepted result.
    if status == "matched" and result is not None:
        with diag.stage("enrich"):
            result = enrich(result, guess, cfg)

    message = {
        "matched": "",
        "candidates": "Pick the right book.",
        "unidentified": "Couldn't confidently identify this book — try a clearer shot.",
    }.get(status, "")

    return _finalize(
        diag, cfg,
        IdentifyResponse(
            scan_id=diag.scan_id,
            status=status,
            result=result,
            candidates=candidates,
            cover_guess=guess,
            message=message,
            timings_ms=diag.timings,
        ),
        debug,
    )


def _gate(guess, ranked, cfg: Config):
    if not ranked or ranked[0].match_score < cfg.min_match_score:
        return "unidentified", None, ranked[:5]

    best = ranked[0]
    confident = guess.confidence >= cfg.auto_accept_confidence and best.match_score >= _MATCH_FLOOR

    # Only treat the runner-up as competition if it's a *different work* with a
    # comparable score. Different editions of the same title (novel vs graphic
    # novel, reprints) share a title and should not force a chooser.
    ambiguous = False
    if len(ranked) >= 2:
        second = ranked[1]
        close = (best.match_score - second.match_score) < _AMBIGUOUS_GAP
        different_work = title_similarity(best.title, second.title) < 0.9
        ambiguous = close and different_work

    if confident and not ambiguous:
        # Return the runner-ups too so the app can offer a "Not the right book?"
        # alternative list without another round-trip.
        return "matched", best, ranked[1:6]
    return "candidates", None, ranked[:5]


def _finalize(diag, cfg, response: IdentifyResponse, debug: bool) -> IdentifyResponse:
    response.timings_ms = diag.timings
    diag.persist(
        {
            "status": response.status,
            "result_title": response.result.title if response.result else None,
            "candidate_titles": [c.title for c in response.candidates],
        }
    )
    if debug:
        response.diagnostics = diag.to_model()
    return response
