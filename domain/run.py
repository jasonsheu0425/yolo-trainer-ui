"""Run-folder model shared by discovery, analysis, and navigation contexts."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass(frozen=True)
class RunInfo:
    """Filesystem-backed YOLO run metadata without any UI concerns."""

    path: Path
    name: str
    run_type: str
    modified_at: datetime
    artifacts: dict[str, Path | None]
