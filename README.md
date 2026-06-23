# Lens server

Flask recognition service. Pipeline: **detect → identify → catalog → gate → enrich**.

```
app/
  config.py              all tunables (provider, model, effort, thresholds) — no per-book knobs
  routes.py              GET /health, POST /identify[?debug=1]
  cli.py                 python -m app.cli <image> [--json]
  pipeline/
    schema.py            CoverGuess (LLM output) + BookResult / IdentifyResponse
    detect.py            OpenCV largest-quad crop + perspective de-warp + resize
    identify.py          provider dispatch + the cover-reading prompt
    providers/
      gemini_provider.py free Gemini (REST, responseSchema)
      claude_provider.py paid Claude (messages.parse, adaptive thinking, effort)
    catalog.py           Google Books + Open Library search, fuzzy match scoring
    enrich.py            AI summary fallback when the catalog has no description
    orchestrate.py       runs the stages + confidence gating
    diagnostics.py       timed stage records, JSONL persistence, crop saving
  eval/
    run_eval.py          accuracy harness over a labeled folder
    labels.json          "<file>": {title, author}
    samples/             test covers
```

## Run

```bash
python -m venv .venv                                    # first time only
.venv/Scripts/python -m pip install -r requirements.txt # first time only
cp .env.example .env          # FRESH setup only — don't overwrite a populated .env
.venv/Scripts/python run.py   # http://0.0.0.0:3001  (use the venv Python, not global)
```

> Always invoke the **venv's** Python (`.venv/Scripts/python` on Windows, or
> `source .venv/bin/activate` first). The global `python` doesn't have Flask.

## Test without the app

```bash
.venv/Scripts/python -m app.cli app/eval/samples/great_gatsby.jpg   # diagnostics + result
.venv/Scripts/python -m app.eval.run_eval                           # title/author hit-rate
curl -s localhost:3001/health
```

## Key env vars

| Var                  | Purpose                                              |
| -------------------- | ---------------------------------------------------- |
| `VISION_PROVIDER`    | `auto` \| `gemini` \| `claude`                       |
| `GEMINI_API_KEY`     | free vision provider (https://aistudio.google.com/apikey) |
| `ANTHROPIC_API_KEY`  | paid vision provider                                 |
| `GOOGLE_BOOKS_API_KEY` | optional; keyless fallback if absent/restricted    |
| `AUTO_ACCEPT_CONFIDENCE`, `MIN_MATCH_SCORE` | confidence-gating thresholds  |
| `ENHANCE_COVER`, `MAX_IMAGE_DIM`, `YOLO_BOOK_COVER_ENABLED` | detection tuning |

Per-scan diagnostics (and saved crops) land in `DIAGNOSTICS_DIR` (`./_diagnostics`)
as `scans.jsonl`; eval failures in `eval_failures.jsonl`.
