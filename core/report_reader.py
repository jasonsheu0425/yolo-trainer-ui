from __future__ import annotations

import csv
from pathlib import Path
import re
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

FILTER_MODES = (
    "Primary Category Only",
    "Any Error Flag",
    "Primary or Any Flag",
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
            has_error_flags = "all_error_flags" in reader.fieldnames
            rows: list[dict[str, str]] = []
            for source_row in reader:
                normalized = {str(key): "" if value is None else str(value) for key, value in source_row.items() if key}
                if not normalized.get("primary_category") and normalized.get("category"):
                    normalized["primary_category"] = normalized["category"]
                for field in REPORT_FIELDS:
                    normalized.setdefault(field, "")
                normalized["_has_all_error_flags_field"] = "1" if has_error_flags else "0"
                rows.append(normalized)
            return rows, ""
    except (OSError, UnicodeError, csv.Error) as exc:
        return [], f"Unable to read report CSV: {exc}"


def summarize_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary_counts = {category: 0 for category in CATEGORY_FILTERS}
    flag_counts = {category: 0 for category in CATEGORY_FILTERS}
    markers = [str(row.get("_has_all_error_flags_field", "")) for row in rows]
    flags_available = any(marker == "1" for marker in markers) if markers else False
    for row in rows:
        category = str(row.get("primary_category", "")).strip().casefold()
        if category in CATEGORY_FILTERS:
            primary_counts[category] += 1
        for flag in parse_error_flags(row.get("all_error_flags", "")):
            if flag in CATEGORY_FILTERS:
                flag_counts[flag] += 1
    summary: dict[str, Any] = {
        "total_rows": len(rows),
        "primary_counts": primary_counts,
        "flag_counts": flag_counts,
        "flags_available": flags_available,
        **primary_counts,
    }
    return summary


def parse_error_flags(value: Any) -> set[str]:
    """Parse comma, semicolon, or whitespace-delimited flags case-insensitively."""
    return {
        token.casefold()
        for token in re.split(r"[,;\s]+", str(value or "").strip())
        if token.strip()
    }


def filter_report(
    rows: list[dict[str, Any]],
    category: str = "All",
    search_text: str = "",
    filter_mode: str = "Primary or Any Flag",
) -> list[dict[str, Any]]:
    requested_category = str(category or "All").strip().casefold()
    requested_mode = str(filter_mode or "Primary or Any Flag").strip().casefold()
    search = str(search_text or "").strip().casefold()
    searchable_fields = ("image_name", "primary_category", "all_error_flags", "notes")
    filtered: list[dict[str, Any]] = []
    for row in rows:
        if requested_category != "all":
            primary_match = str(row.get("primary_category", "")).strip().casefold() == requested_category
            flag_match = requested_category in parse_error_flags(row.get("all_error_flags", ""))
            if requested_mode == "primary category only":
                category_match = primary_match
            elif requested_mode == "any error flag":
                category_match = flag_match
            else:
                category_match = primary_match or flag_match
            if not category_match:
                continue
        if search and not any(search in str(row.get(field, "")).casefold() for field in searchable_fields):
            continue
        filtered.append(row)
    return filtered
