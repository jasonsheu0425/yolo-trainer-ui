"""Filesystem storage details for versioned training-analysis cache data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from persistence.atomic_writer import atomic_write_json


class TrainingAnalysisStore:
    """Reads, writes, and source-validates analysis cache payloads."""

    filename = "training_analysis.json"

    def cache_path(self, run_folder: str | Path) -> Path:
        return Path(run_folder) / self.filename

    def source_hash(self, results_csv: Path) -> str:
        return hashlib.sha256(results_csv.read_bytes()).hexdigest()

    def read_payload(self, run_folder: str | Path) -> dict[str, Any] | None:
        try:
            payload = json.loads(self.cache_path(run_folder).read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, ValueError, json.JSONDecodeError):
            return None

    def write_payload(self, run_folder: str | Path, payload: dict[str, Any]) -> Path:
        path = self.cache_path(run_folder)
        atomic_write_json(path, payload)
        return path

    def cache_matches_source(self, run_folder: str | Path, payload: dict[str, Any]) -> bool | None:
        """True is verified, False stale, None when source no longer exists."""
        results = Path(run_folder) / "results.csv"
        if not results.is_file():
            return None
        source = payload.get("source")
        if not isinstance(source, dict) or not isinstance(source.get("sha256"), str):
            return False
        return source["sha256"] == self.source_hash(results)
