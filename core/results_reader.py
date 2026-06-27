from __future__ import annotations

from pathlib import Path

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


def read_results(path: str | Path) -> pd.DataFrame:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"找不到 results.csv：{csv_path}")
    frame = pd.read_csv(csv_path)
    frame.columns = [str(column).strip() for column in frame.columns]
    if frame.empty:
        raise ValueError("results.csv 沒有資料。")
    return frame


def available_series(frame: pd.DataFrame, group: str) -> list[str]:
    return [name for name in SERIES[group] if name in frame.columns]
