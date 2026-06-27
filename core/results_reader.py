from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Any

import pandas as pd


SERIES = {
    "loss": ["train/box_loss", "train/cls_loss", "train/dfl_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss"],
    "map": ["metrics/mAP50(B)", "metrics/mAP50-95(B)"],
    "pr": ["metrics/precision(B)", "metrics/recall(B)"],
}

RUN_ARTIFACTS = {
    "best.pt": Path("weights") / "best.pt",
    "last.pt": Path("weights") / "last.pt",
    "results.csv": Path("results.csv"),
    "results.png": Path("results.png"),
    "confusion_matrix.png": Path("confusion_matrix.png"),
}

VALIDATION_ARTIFACTS: dict[str, tuple[str, ...]] = {
    "confusion_matrix.png": ("confusion_matrix.png",),
    "confusion_matrix_normalized.png": ("confusion_matrix_normalized.png",),
    "PR_curve.png": ("PR_curve.png", "BoxPR_curve.png"),
    "P_curve.png": ("P_curve.png", "BoxP_curve.png"),
    "R_curve.png": ("R_curve.png", "BoxR_curve.png"),
    "F1_curve.png": ("F1_curve.png", "BoxF1_curve.png"),
    "val_batch0_labels.jpg": ("val_batch0_labels.jpg",),
    "val_batch0_pred.jpg": ("val_batch0_pred.jpg",),
}

METRIC_ALIASES = {
    "precision": ("metrics/precision(B)", "metrics/precision(M)", "metrics/precision", "precision(B)", "precision"),
    "recall": ("metrics/recall(B)", "metrics/recall(M)", "metrics/recall", "recall(B)", "recall"),
    "map50": ("metrics/mAP50(B)", "metrics/mAP50(M)", "metrics/mAP50", "mAP50(B)", "mAP50"),
    "map50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "metrics/mAP50-95", "mAP50-95(B)", "mAP50-95"),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def scan_run_folder(run_folder: str | Path) -> dict[str, Path | None]:
    """Return standard YOLO artifacts without raising for missing paths."""
    raw = str(run_folder).strip()
    root = Path(raw).expanduser() if raw else None
    if root is None or not root.is_dir():
        return {"run_folder": None, **{name: None for name in RUN_ARTIFACTS}}
    root = root.resolve()
    result: dict[str, Path | None] = {"run_folder": root}
    for name, relative_path in RUN_ARTIFACTS.items():
        candidate = root / relative_path
        result[name] = candidate if candidate.is_file() else None
    return result


def scan_validation_folder(run_folder: str | Path) -> dict[str, Path | None]:
    """Return validation artifacts, including Box-prefixed Ultralytics variants."""
    raw = str(run_folder).strip()
    root = Path(raw).expanduser() if raw else None
    if root is None or not root.is_dir():
        return {"run_folder": None, **{name: None for name in VALIDATION_ARTIFACTS}}
    root = root.resolve()
    result: dict[str, Path | None] = {"run_folder": root}
    for display_name, candidates in VALIDATION_ARTIFACTS.items():
        result[display_name] = next((root / name for name in candidates if (root / name).is_file()), None)
    return result


def read_results(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"results.csv not found: {csv_path}")
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise ValueError("results.csv contains no rows.")
    return frame


def available_series(frame: pd.DataFrame, group: str) -> list[str]:
    return [name for name in SERIES[group] if name in frame.columns]


def extract_metrics(frame: pd.DataFrame) -> dict[str, float | None]:
    """Extract the final metrics row across common Ultralytics column variants."""
    metrics: dict[str, float | None] = {key: None for key in METRIC_ALIASES}
    if frame.empty:
        return metrics
    normalized_columns = {_normalize_column(str(column)): str(column) for column in frame.columns}
    row = frame.iloc[-1]
    for key, aliases in METRIC_ALIASES.items():
        for alias in aliases:
            column = normalized_columns.get(_normalize_column(alias))
            if column is None:
                continue
            try:
                value = float(row[column])
            except (TypeError, ValueError):
                continue
            if pd.notna(value):
                metrics[key] = value
                break
    return metrics


def read_validation_metrics(run_folder: str | Path, log_text: str = "") -> dict[str, Any]:
    """Read metrics from results.csv, falling back to an Ultralytics log summary."""
    metrics: dict[str, Any] = {key: None for key in METRIC_ALIASES}
    root = Path(str(run_folder).strip()).expanduser() if str(run_folder).strip() else None
    results_path = root / "results.csv" if root and root.is_dir() else None
    if results_path and results_path.is_file():
        try:
            metrics.update(extract_metrics(read_results(results_path)))
            metrics.update(message="", source=str(results_path))
            return metrics
        except (OSError, ValueError, pd.errors.ParserError) as exc:
            metrics.update(message=f"Unable to read metrics file: {exc}", source="")
            return metrics
    parsed = parse_validation_log(log_text)
    if any(value is not None for value in parsed.values()):
        metrics.update(parsed)
        metrics.update(message="Metrics file not found. Values parsed from validation log.", source="log")
    else:
        metrics.update(message="Metrics file not found.", source="")
    return metrics


def parse_validation_log(log_text: str) -> dict[str, float | None]:
    """Parse the final 'all' row: precision, recall, mAP50, mAP50-95."""
    empty = {key: None for key in METRIC_ALIASES}
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    for raw_line in reversed(log_text.splitlines()):
        line = ansi_escape.sub("", raw_line).strip()
        if not re.match(r"^all\s+", line, flags=re.IGNORECASE):
            continue
        numbers = re.findall(r"(?<![A-Za-z])[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", line)
        if len(numbers) < 6:
            continue
        try:
            precision, recall, map50, map50_95 = map(float, numbers[-4:])
        except ValueError:
            continue
        return {"precision": precision, "recall": recall, "map50": map50, "map50_95": map50_95}
    return empty


def scan_runs(runs_root: str | Path) -> list[dict[str, Any]]:
    """Scan immediate YOLO run folders and tolerate incomplete or unreadable runs."""
    root = Path(str(runs_root).strip()).expanduser() if str(runs_root).strip() else None
    if root is None or not root.is_dir():
        return []
    runs: list[dict[str, Any]] = []
    try:
        folders = [path for path in root.iterdir() if path.is_dir()]
    except OSError:
        return []
    for folder in folders:
        try:
            artifacts = scan_run_folder(folder)
            metrics = read_validation_metrics(folder)
            run_type = _detect_run_type(folder, artifacts)
            stat = folder.stat()
            runs.append(
                {
                    "name": folder.name,
                    "type": run_type,
                    "path": folder.resolve(),
                    "best": artifacts.get("best.pt") is not None,
                    "last": artifacts.get("last.pt") is not None,
                    "results": artifacts.get("results.csv") is not None,
                    "created": datetime.fromtimestamp(stat.st_ctime),
                    "modified": datetime.fromtimestamp(stat.st_mtime),
                    **{key: metrics.get(key) for key in METRIC_ALIASES},
                }
            )
        except (OSError, ValueError):
            continue
    return sorted(runs, key=lambda item: item["modified"], reverse=True)


def _detect_run_type(folder: Path, artifacts: dict[str, Path | None]) -> str:
    if artifacts.get("best.pt") or artifacts.get("last.pt"):
        return "train"
    if (folder / "confusion_matrix.png").is_file() or any(folder.glob("*PR_curve.png")):
        return "val"
    try:
        has_images = any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in folder.iterdir())
    except OSError:
        has_images = False
    return "predict" if has_images else "unknown"


def _normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())
