"""Pluggable vision-LLM backends. Each exposes `run(image_bytes, cfg, system, user)
-> CoverGuess` and an optional `summarize(title, author, cfg) -> str`."""
