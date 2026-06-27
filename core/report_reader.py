from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


REPORT_FIELDS = (
    "image_name",
    "primary_category",
    "all_error_flags",
    "detection_count",
    "ground_truth_count",
    "min_confidence",
    "max_iou",
    "false_negative_count",
    "false_positive_count",
    "class_mismatch_count",
    "low_iou_count",
    "low_confidence_count",
    "copied_to",
    "image_path",
    "prediction_label_path",
    "ground_truth_label_path",
    "notes",
)

CATEGORY_FILTERS = (
    "false_negative",
    "false_positive",
    "class_mismatch",
    "low_iou",
    "low_confidence",
    "no_detection",
    "no_label_file",
    "unknown",
)


def read_hard_cases_report(csv_path: str | Path) -> tuple[list[dict[str, str]], str]:
    """Read a UTF-8 hard-cases report, filling missing fields and returning errors safely."""
    raw = str(csv_path).strip()
    if not raw:
        return [], "Report CSV path is empty."
    try:
        path = Path(raw).expanduser()
        if not path.is_file():
            return [], f"Report CSV not found: {path}"
    except (OSError, ValueError) as exc:
        return [], f"Invalid report CSV path: {exc}"
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                return [], "Report CSV has no header row."
            rows: list[dict[str, str]] = []
            for source_row in reader:
                normalized = {str(key): "" if value is None else str(value) for key, value in source_row.items() if key}
                if not normalized.get("primary_category") and normalized.get("category"):
                    normalized["primary_category"] = normalized["category"]
                for field in REPORT_FIELDS:
                    normalized.setdefault(field, "")
                rows.append(normalized)
            return rows, ""
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], f"Unable to read report CSV: {exc}"


def summarize_report(rows: list[dict[str, Any]]) -> dict[str, int]:
    summary = {"total_rows": len(rows), **{category: 0 for category in CATEGORY_FILTERS}}
    for row in rows:
        category = str(row.get("primary_category", ""))
        if category in CATEGORY_FILTERS:
            summary[category] += 1
    return summary


def filter_report(
    rows: list[dict[str, Any]], category: str = "All", search_text: str = ""
) -> list[dict[str, Any]]:
    requested_category = str(category or "All")
    search = str(search_text or "").strip().casefold()
    searchable_fields = ("image_name", "primary_category", "all_error_flags", "notes")
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if requested_category != "All" and str(row.get("primary_category", "")) != requested_category:
            continue
        if search and not any(search in str(row.get(field, "")).casefold() for field in searchable_fields):
            continue
        filtered.append(row)
    return filtered
