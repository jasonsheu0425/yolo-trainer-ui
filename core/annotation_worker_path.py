"""Resolve the physical external-runtime worker in source and frozen builds."""
from __future__ import annotations

from pathlib import Path
import sys


def resolve_annotation_worker_path() -> Path:
    candidates: list[Path] = []
    if getattr(sys, "frozen", False):
        bundle = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidates.extend(
            (
                bundle / "runtime_workers" / "annotation_inference_worker.py",
                Path(sys.executable).parent / "_internal" / "runtime_workers"
                / "annotation_inference_worker.py",
                Path(sys.executable).parent / "runtime_workers"
                / "annotation_inference_worker.py",
            )
        )
    candidates.append(
        Path(__file__).resolve().parents[1]
        / "runtime_workers"
        / "annotation_inference_worker.py"
    )
    for path in candidates:
        if path.is_file():
            return path.resolve()
    raise FileNotFoundError("annotation_inference_worker_not_found")
