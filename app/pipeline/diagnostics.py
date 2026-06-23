"""Per-scan diagnostics: timed stage records, JSONL persistence, crop saving.

The whole point is *global* debuggability — when recognition fails we want to
see exactly what each stage produced (raw LLM output, catalog queries + scores,
the gating decision) so the fix improves the pipeline for every book, never one.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from .schema import Diagnostics, StageRecord


class ScanDiagnostics:
    def __init__(self, diagnostics_dir: str, save_crops: bool) -> None:
        self.scan_id = uuid.uuid4().hex[:12]
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.dir = diagnostics_dir
        self.save_crops = save_crops
        self._stages: list[StageRecord] = []
        self._timings: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        """Time a pipeline stage; the body fills the returned dict with details."""
        data: dict = {}
        start = time.perf_counter()
        try:
            yield data
        finally:
            ms = round((time.perf_counter() - start) * 1000, 1)
            self._timings[name] = ms
            self._stages.append(StageRecord(name=name, ms=ms, data=_safe(data)))

    def record(self, name: str, **data) -> None:
        """Record an instantaneous (untimed) stage."""
        self._stages.append(StageRecord(name=name, ms=0.0, data=_safe(data)))

    @property
    def timings(self) -> dict[str, float]:
        return dict(self._timings)

    def to_model(self) -> Diagnostics:
        return Diagnostics(
            scan_id=self.scan_id, created_at=self.created_at, stages=list(self._stages)
        )

    def save_crop(self, jpeg_bytes: bytes) -> str | None:
        if not self.save_crops:
            return None
        os.makedirs(self.dir, exist_ok=True)
        path = os.path.join(self.dir, f"{self.scan_id}_crop.jpg")
        with open(path, "wb") as f:
            f.write(jpeg_bytes)
        return path

    def persist(self, summary: dict) -> None:
        """Append the full record to a JSONL log for offline analysis / eval."""
        os.makedirs(self.dir, exist_ok=True)
        record = {
            "scan_id": self.scan_id,
            "created_at": self.created_at,
            "timings_ms": self._timings,
            "summary": _safe(summary),
            "stages": [s.model_dump() for s in self._stages],
        }
        with open(os.path.join(self.dir, "scans.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe(obj):
    """Make a value JSON-serializable; never let diagnostics crash a request."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        return json.loads(json.dumps(obj, default=str))
