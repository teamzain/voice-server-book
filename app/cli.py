"""Run the full recognition pipeline on a local image, with staged diagnostics.

    python -m app.cli path/to/cover.jpg
    python -m app.cli path/to/cover.jpg --json
"""

from __future__ import annotations

import json
import sys

from .config import get_config
from .pipeline.orchestrate import identify_book


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    flags = {a for a in argv if a.startswith("--")}
    if not args:
        print(__doc__)
        return 2

    path = args[0]
    try:
        image_bytes = open(path, "rb").read()
    except OSError as exc:
        print(f"Could not read {path}: {exc}")
        return 1

    cfg = get_config()
    try:
        cfg.require_provider_key()
    except RuntimeError as exc:
        print(f"Config error: {exc}")
        return 1

    resp = identify_book(image_bytes, cfg, debug=True)

    if "--json" in flags:
        print(json.dumps(resp.model_dump(), indent=2, ensure_ascii=False))
        return 0

    print(f"\nscan {resp.scan_id}  provider={cfg.resolved_provider()}")
    print(f"status: {resp.status.upper()}   {resp.message}")
    if resp.cover_guess:
        g = resp.cover_guess
        print(f"\n[read] title={g.title!r} author={g.author!r} confidence={g.confidence}")
        if g.candidates:
            print("       alt readings:", [(c.title, c.author) for c in g.candidates])

    if resp.result:
        r = resp.result
        print(f"\n[MATCH] {r.title} — {', '.join(r.authors)}  ({r.published_date})")
        print(f"        score={r.match_score} source={r.source} isbn13={r.isbn_13}")
        if r.description:
            tag = " (AI summary)" if r.description_generated else ""
            print(f"        {r.description[:200]}{tag}")
        print(f"        info: {r.info_url}")
    elif resp.candidates:
        print("\n[CANDIDATES]")
        for i, c in enumerate(resp.candidates, 1):
            print(f"  {i}. {c.title} — {', '.join(c.authors)}  (score={c.match_score}, {c.source})")

    print("\n[timings ms]", resp.timings_ms)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
