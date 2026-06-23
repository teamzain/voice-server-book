"""Catalog grounding: turn Claude's read into a real, verified book.

The vision model proposes; the catalog disposes. We search Google Books and
Open Library for the model's title/author (and its alternative candidates), then
score every hit against (a) the proposed title, (b) the proposed author, and
(c) the raw text actually visible on the cover. The best-grounded hit wins.

This is the anti-hallucination gate: the returned book must exist in a real
catalog *and* be consistent with what's physically on the cover. Scoring is
generic fuzzy matching — there are no per-book rules.
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

import requests

from ..config import Config
from .schema import BookResult, CoverGuess

log = logging.getLogger(__name__)

GOOGLE_BOOKS_URL = "https://www.googleapis.com/books/v1/volumes"
OPEN_LIBRARY_URL = "https://openlibrary.org/search.json"

_ARTICLES = {"the", "a", "an"}
_WORD_RE = re.compile(r"[^\w\s]", re.UNICODE)


# ─── text normalization + fuzzy scoring (dependency-free) ────────────────────

def _norm(s: str) -> str:
    s = (s or "").lower().strip()
    s = _WORD_RE.sub(" ", s)
    return re.sub(r"\s+", " ", s).strip()


def _tokens(s: str) -> set[str]:
    return {t for t in _norm(s).split() if t and t not in _ARTICLES}


def _seq_ratio(a: str, b: str) -> float:
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def _token_set_ratio(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    return inter / max(len(ta), len(tb))


def title_similarity(a: str, b: str) -> float:
    return max(_seq_ratio(a, b), _token_set_ratio(a, b))


def author_similarity(query_author: str, hit_authors: list[str]) -> float:
    if not query_author or not hit_authors:
        return 0.0
    # match on last name as well as full string — covers "F. Scott Fitzgerald"
    q_tokens = _tokens(query_author)
    best = 0.0
    for ha in hit_authors:
        full = max(_seq_ratio(query_author, ha), _token_set_ratio(query_author, ha))
        last = 0.0
        h_tokens = _tokens(ha)
        if q_tokens and h_tokens and (q_tokens & h_tokens):
            last = len(q_tokens & h_tokens) / max(len(q_tokens), len(h_tokens))
        best = max(best, full, last)
    return best


def _visible_bonus(hit_title: str, visible_text: str) -> float:
    """Fraction of the hit's title tokens that actually appear on the cover."""
    ht = _tokens(hit_title)
    if not ht:
        return 0.0
    vt = _tokens(visible_text)
    return len(ht & vt) / len(ht)


# ─── catalog calls ───────────────────────────────────────────────────────────

def _google_books(title: str, author: str, cfg: Config) -> list[BookResult]:
    q_parts = []
    if title:
        q_parts.append(f"intitle:{title}")
    if author:
        q_parts.append(f"inauthor:{author}")
    if not q_parts:
        return []
    params = {"q": " ".join(q_parts), "maxResults": 5, "printType": "books"}

    # Try with the key first; the kinseb keys are app-restricted and may 403 from
    # a server, so transparently fall back to a keyless request (Google Books
    # allows anonymous queries at a lower quota).
    attempts: list[dict] = []
    if cfg.google_books_api_key:
        attempts.append({**params, "key": cfg.google_books_api_key})
    attempts.append(params)

    for attempt in attempts:
        try:
            r = requests.get(GOOGLE_BOOKS_URL, params=attempt, timeout=cfg.catalog_timeout_s)
            r.raise_for_status()
            return [_from_google(it) for it in (r.json().get("items", []) or [])]
        except requests.RequestException as exc:
            log.warning("google books query failed (key=%s): %s", "key" in attempt, exc)
            continue
    return []


def _from_google(item: dict) -> BookResult:
    vi = item.get("volumeInfo", {}) or {}
    ids = {x.get("type"): x.get("identifier") for x in vi.get("industryIdentifiers", []) or []}
    images = vi.get("imageLinks", {}) or {}
    thumb = images.get("thumbnail") or images.get("smallThumbnail") or ""
    return BookResult(
        title=vi.get("title", ""),
        subtitle=vi.get("subtitle", ""),
        authors=vi.get("authors", []) or [],
        description=vi.get("description", ""),
        isbn_10=ids.get("ISBN_10", "") or "",
        isbn_13=ids.get("ISBN_13", "") or "",
        publisher=vi.get("publisher", ""),
        published_date=vi.get("publishedDate", ""),
        page_count=vi.get("pageCount"),
        categories=vi.get("categories", []) or [],
        average_rating=vi.get("averageRating"),
        ratings_count=vi.get("ratingsCount"),
        thumbnail_url=thumb.replace("http://", "https://"),
        preview_url=(vi.get("previewLink") or "").replace("http://", "https://"),
        info_url=(vi.get("infoLink") or "").replace("http://", "https://"),
        source="google_books",
    )


def _open_library(title: str, author: str, cfg: Config) -> list[BookResult]:
    params: dict = {"limit": 5}
    if title:
        params["title"] = title
    if author:
        params["author"] = author
    if "title" not in params:
        return []
    params["fields"] = (
        "title,subtitle,author_name,first_publish_year,isbn,cover_i,key,"
        "publisher,number_of_pages_median,subject,ia,public_scan_b"
    )
    try:
        r = requests.get(OPEN_LIBRARY_URL, params=params, timeout=cfg.catalog_timeout_s)
        r.raise_for_status()
        docs = r.json().get("docs", []) or []
    except requests.RequestException as exc:
        log.warning("open library query failed: %s", exc)
        return []
    return [_from_openlibrary(d) for d in docs[:5]]


def _from_openlibrary(doc: dict) -> BookResult:
    isbns = doc.get("isbn", []) or []
    isbn_13 = next((i for i in isbns if len(i) == 13), "")
    isbn_10 = next((i for i in isbns if len(i) == 10), "")
    cover_i = doc.get("cover_i")
    thumb = f"https://covers.openlibrary.org/b/id/{cover_i}-M.jpg" if cover_i else ""
    work_key = doc.get("key", "")  # e.g. /works/OL12345W
    info_url = f"https://openlibrary.org{work_key}" if work_key else ""
    read_url = ""
    if doc.get("public_scan_b") and doc.get("ia"):
        read_url = f"https://archive.org/details/{doc['ia'][0]}"
    publishers = doc.get("publisher", []) or []
    return BookResult(
        title=doc.get("title", ""),
        subtitle=doc.get("subtitle", ""),
        authors=doc.get("author_name", []) or [],
        isbn_10=isbn_10,
        isbn_13=isbn_13,
        publisher=publishers[0] if publishers else "",
        published_date=str(doc.get("first_publish_year") or ""),
        page_count=doc.get("number_of_pages_median"),
        categories=(doc.get("subject", []) or [])[:6],
        thumbnail_url=thumb,
        preview_url=read_url,
        info_url=info_url,
        source="open_library",
    )


# ─── scoring + aggregation ───────────────────────────────────────────────────

def score_hit(hit: BookResult, q_title: str, q_author: str, visible_text: str) -> float:
    t = title_similarity(q_title, hit.title)
    a = author_similarity(q_author, hit.authors)
    v = _visible_bonus(hit.title, visible_text)
    if q_author:
        score = 0.6 * t + 0.25 * a + 0.15 * v
    else:
        score = 0.75 * t + 0.25 * v
    return round(min(1.0, score), 4)


def _dedup_key(b: BookResult) -> tuple[str, str]:
    return (_norm(b.title), _norm(b.authors[0]) if b.authors else "")


def _rank_key(b: BookResult) -> tuple:
    """Order by match score, then by metadata richness so the edition we surface
    has a real description/date/cover. Google Books wins ties over Open Library."""
    return (
        b.match_score,
        1 if b.source == "google_books" else 0,
        1 if b.description else 0,
        1 if b.thumbnail_url else 0,
        1 if b.published_date else 0,
    )


_STRONG_MATCH = 0.70  # above this, trust Google and skip the slower Open Library


def gather_candidates(guess: CoverGuess, cfg: Config) -> tuple[list[BookResult], list[dict]]:
    """Search catalogs for the guess + its candidates, score, dedup, rank.

    Google Books is queried first (fast, comprehensive). Open Library is only
    consulted as a fallback when Google produced no strong match — this keeps the
    common case ~1 round-trip instead of always paying Open Library's latency.

    Returns (ranked BookResults, query trace for diagnostics).
    """
    queries: list[tuple[str, str]] = []
    if guess.title:
        queries.append((guess.title, guess.author))
    for cand in guess.candidates[:3]:
        if cand.title and (cand.title, cand.author) not in queries:
            queries.append((cand.title, cand.author))
    # Last resort: if the model gave no title at all, try the longest visible line.
    if not queries and guess.visible_text:
        longest = max(guess.visible_text.splitlines(), key=len, default="").strip()
        if longest:
            queries.append((longest, ""))

    q_title = guess.title or (queries[0][0] if queries else "")
    q_author = guess.author or (queries[0][1] if queries else "")

    def _score(hits: list[BookResult]) -> list[BookResult]:
        for h in hits:
            h.match_score = score_hit(h, q_title, q_author, guess.visible_text)
        return hits

    hits: list[BookResult] = []
    trace: list[dict] = []

    # Pass 1: Google Books for the guess + alternative readings.
    for qt, qa in queries:
        g = _score(_google_books(qt, qa, cfg))
        trace.append({"source": "google_books", "query": {"title": qt, "author": qa}, "hits": len(g)})
        hits.extend(g)

    # Pass 2: Open Library, only if Google gave us nothing strong.
    best_so_far = max((h.match_score for h in hits), default=0.0)
    if best_so_far < _STRONG_MATCH and queries:
        qt, qa = queries[0]
        o = _score(_open_library(qt, qa, cfg))
        trace.append({"source": "open_library", "query": {"title": qt, "author": qa}, "hits": len(o)})
        hits.extend(o)

    # Dedup, keeping the richer entry per (title, author); Google wins score ties.
    seen: dict[tuple[str, str], BookResult] = {}
    for h in hits:
        key = _dedup_key(h)
        if not key[0]:
            continue
        cur = seen.get(key)
        if cur is None or _rank_key(h) > _rank_key(cur):
            seen[key] = h
    ranked = sorted(seen.values(), key=_rank_key, reverse=True)
    return ranked[:5], trace
