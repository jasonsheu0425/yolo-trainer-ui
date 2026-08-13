"""Filesystem run discovery adapter returning typed RunInfo records."""
from __future__ import annotations

from pathlib import Path

from core.results_reader import scan_run_folder, scan_runs
from domain.run import RunInfo


class RunDiscoveryService:
    """Provides read-only YOLO run discovery without a database dependency."""

    def list_runs(self, root: str | Path) -> list[RunInfo]:
        return [
            RunInfo(
                path=Path(item["path"]), name=str(item["name"]), run_type=str(item["type"]),
                modified_at=item["modified"], artifacts=scan_run_folder(item["path"]),
            )
            for item in scan_runs(root)
        ]
