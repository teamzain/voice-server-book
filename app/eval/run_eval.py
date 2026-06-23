"""Accuracy harness: run the pipeline over a labeled folder and report hit-rate.

    python -m app.eval.run_eval                      # uses app/eval/samples + labels.json
    python -m app.eval.run_eval /path/to/folder      # folder with its own labels.json

`labels.json` maps "<filename>": {"title": ..., "author": ...}. Unlabeled images
still run (prediction shown, not scored). Failures are written to
<diagnostics_dir>/eval_failures.jsonl so problems can be diagnosed *globally* —
the point of the whole exercise is to improve the pipeline for all books, never
to add a rule for one.
"""

from __future__ import annotations

import json
import os
import sys

from ..config import get_config
from ..pipeline.catalog import author_similarity, title_similarity
from ..pipeline.orchestrate import identify_book

_IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
TITLE_OK = 0.85
AUTHOR_OK = 0.6


def _predicted(resp):
    if resp.result:
        return resp.result.title, resp.result.authors
    if resp.candidates:
        return resp.candidates[0].title, resp.candidates[0].authors
    if resp.cover_guess:
        return resp.cover_guess.title, [resp.cover_guess.author] if resp.cover_guess.author else []
    return "", []


def main(argv: list[str]) -> int:
    folder = argv[0] if argv else os.path.join(os.path.dirname(__file__), "samples")
    labels_path = os.path.join(folder, "labels.json")
    if not os.path.exists(labels_path):
        labels_path = os.path.join(os.path.dirname(__file__), "labels.json")
    labels = json.load(open(labels_path, encoding="utf-8")) if os.path.exists(labels_path) else {}

    cfg = get_config()
    try:
        cfg.require_provider_key()
    except RuntimeError as exc:
        print(f"Config error: {exc}")
        return 1

    images = sorted(f for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in _IMG_EXT)
    if not images:
        print(f"No images in {folder}")
        return 1

    failures_path = os.path.join(cfg.diagnostics_dir, "eval_failures.jsonl")
    os.makedirs(cfg.diagnostics_dir, exist_ok=True)
    open(failures_path, "w").close()  # reset

    scored = title_hits = author_hits = both_hits = 0
    print(f"provider={cfg.resolved_provider()}  images={len(images)}  labeled={len(labels)}\n")

    for name in images:
        resp = identify_book(open(os.path.join(folder, name), "rb").read(), cfg, debug=True)
        p_title, p_authors = _predicted(resp)
        label = labels.get(name)

        if not label:
            print(f"  [    ] {name}: {resp.status} -> {p_title!r} {p_authors}")
            continue

        scored += 1
        t_ok = title_similarity(label["title"], p_title) >= TITLE_OK
        a_ok = (not label.get("author")) or author_similarity(label["author"], p_authors) >= AUTHOR_OK
        title_hits += t_ok
        author_hits += a_ok
        both_hits += t_ok and a_ok

        mark = "PASS" if (t_ok and a_ok) else "FAIL"
        print(f"  [{mark}] {name}: {resp.status} -> {p_title!r} {p_authors}")
        if not (t_ok and a_ok):
            print(f"         expected {label['title']!r} / {label.get('author')!r}  (title_ok={t_ok} author_ok={a_ok})")
            with open(failures_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "image": name, "expected": label, "status": resp.status,
                    "predicted_title": p_title, "predicted_authors": p_authors,
                    "cover_guess": resp.cover_guess.model_dump() if resp.cover_guess else None,
                    "candidates": [c.model_dump() for c in resp.candidates],
                }, ensure_ascii=False) + "\n")

    if scored:
        print(f"\n=== accuracy over {scored} labeled ===")
        print(f"  title:        {title_hits}/{scored}  ({100*title_hits/scored:.1f}%)")
        print(f"  author:       {author_hits}/{scored}  ({100*author_hits/scored:.1f}%)")
        print(f"  title+author: {both_hits}/{scored}  ({100*both_hits/scored:.1f}%)")
        if both_hits < scored:
            print(f"  failures -> {failures_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
