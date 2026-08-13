"""Application boundary for deterministic training result analysis."""
from __future__ import annotations

from pathlib import Path

from core.training_result_analyzer import TrainingAnalysis, analyze_or_load


class AnalysisService:
    """Coordinates analysis cache use without exposing it to UI pages."""

    def load_run(self, run_folder: str | Path, *, force: bool = False) -> TrainingAnalysis:
        return analyze_or_load(Path(run_folder), force=force)
