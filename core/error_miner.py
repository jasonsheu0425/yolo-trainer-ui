from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATEGORIES = ("low_confidence", "no_detection", "no_label_file", "unknown")
REPORT_FIELDS = (
    "image_name",
    "category",
    "image_path",
    "label_path",
    "min_confidence",
    "detection_count",
    "copied_to",
    "notes",
)


def scan_error_cases(
    run_folder: str | Path,
    source_folder: str | Path | None = None,
    labels_folder: str | Path | None = None,
    low_conf_threshold: float = 0.35,
) -> dict[str, Any]:
    """Classify run images using optional YOLO prediction labels."""
    result: dict[str, Any] = {
        "run_folder": "",
        "source_folder": "",
        "labels_folder": "",
        "low_conf_threshold": float(low_conf_threshold),
        "records": [],
        "warnings": [],
        "errors": [],
    }
    run = _existing_directory(run_folder)
    if run is None:
        result["errors"].append(f"Run folder not found: {run_folder}")
        return result
    result["run_folder"] = str(run)

    source = _existing_directory(source_folder)
    external_labels = _existing_directory(labels_folder)
    if source_folder and source is None:
        result["warnings"].append(f"Source folder not found; using run images: {source_folder}")
    if labels_folder and external_labels is None:
        result["warnings"].append(f"Labels folder not found; optional labels ignored: {labels_folder}")
    result["source_folder"] = str(source) if source else ""
    result["labels_folder"] = str(external_labels) if external_labels else ""

    images = _find_images(run)
    if not images:
        result["warnings"].append("No supported images found in the run folder.")
        return result
    source_lookup = _build_image_lookup(source) if source else ({}, {})
    prediction_labels_dir = run / "labels"
    prediction_lookup = _build_file_lookup(prediction_labels_dir, ".txt")
    external_lookup = _build_file_lookup(external_labels, ".txt") if external_labels else {}
    has_prediction_labels = bool(prediction_lookup)
    if not has_prediction_labels:
        result["warnings"].append(
            "No prediction labels found. Enable save_txt and save_conf in Predict for better error mining."
        )

    for image in images:
        try:
            source_image = _source_image_for(image, source_lookup) or image
            prediction_label = prediction_lookup.get(image.stem.lower())
            optional_label = external_lookup.get(image.stem.lower())
            copy_label = prediction_label or optional_label
            category, min_confidence, detection_count, notes = _classify_prediction_label(
                prediction_label, has_prediction_labels, float(low_conf_threshold)
            )
            result["records"].append(
                {
                    "image_name": image.name,
                    "category": category,
                    "image_path": str(source_image),
                    "run_image_path": str(image),
                    "label_path": str(copy_label) if copy_label else "",
                    "prediction_label_path": str(prediction_label) if prediction_label else "",
                    "min_confidence": min_confidence,
                    "detection_count": detection_count,
                    "copied_to": "",
                    "notes": notes,
                }
            )
        except (OSError, UnicodeError, ValueError) as exc:
            result["errors"].append(f"Unable to inspect {image}: {exc}")
    return result


def export_hard_cases(
    scan_result: dict[str, Any],
    output_folder: str | Path,
    *,
    copy_images: bool = True,
    copy_labels_if_found: bool = True,
    create_report_csv: bool = True,
    create_summary_json: bool = True,
) -> dict[str, Any]:
    """Export classified records and reports without raising on individual file failures."""
    result: dict[str, Any] = {
        "output_folder": None,
        "report_csv": None,
        "summary_json": None,
        "records": [dict(record) for record in scan_result.get("records", [])],
        "warnings": list(scan_result.get("warnings", [])),
        "errors": list(scan_result.get("errors", [])),
    }
    raw_output = str(output_folder).strip()
    if not raw_output:
        result["errors"].append("Hard cases output folder is empty.")
        return result
    root = Path(raw_output).expanduser()
    try:
        root.mkdir(parents=True, exist_ok=True)
        for category in CATEGORIES:
            (root / category).mkdir(parents=True, exist_ok=True)
        root = root.resolve()
        result["output_folder"] = root
    except OSError as exc:
        result["errors"].append(f"Unable to create hard cases folder: {exc}")
        return result

    for record in result["records"]:
        category = record.get("category") if record.get("category") in CATEGORIES else "unknown"
        category_folder = root / category
        copied_image: Path | None = None
        if copy_images:
            image = Path(str(record.get("image_path", "")))
            if image.is_file():
                try:
                    copied_image = _unique_destination(category_folder / image.name)
                    shutil.copy2(image, copied_image)
                    record["copied_to"] = str(copied_image)
                except OSError as exc:
                    result["errors"].append(f"Unable to copy image {image}: {exc}")
            else:
                result["errors"].append(f"Image disappeared before export: {image}")
        if copy_labels_if_found and record.get("label_path"):
            label = Path(str(record["label_path"]))
            if label.is_file():
                try:
                    label_name = f"{copied_image.stem}.txt" if copied_image else label.name
                    destination = _unique_destination(category_folder / label_name)
                    shutil.copy2(label, destination)
                except OSError as exc:
                    result["errors"].append(f"Unable to copy label {label}: {exc}")

    if create_report_csv:
        report_path = root / "hard_cases_report.csv"
        error = _write_report_csv(report_path, result["records"])
        if error:
            result["errors"].append(error)
        else:
            result["report_csv"] = report_path

    if create_summary_json:
        summary_path = root / "hard_cases_summary.json"
        summary = build_summary(scan_result, root)
        error = _write_summary_json(summary_path, summary)
        if error:
            result["errors"].append(error)
        else:
            result["summary_json"] = summary_path
    return result


def build_summary(scan_result: dict[str, Any], output_folder: Path) -> dict[str, Any]:
    records = scan_result.get("records", [])
    counts = {category: sum(1 for record in records if record.get("category") == category) for category in CATEGORIES}
    return {
        "run_folder": str(scan_result.get("run_folder", "")),
        "source_folder": str(scan_result.get("source_folder", "")),
        "output_folder": str(output_folder),
        "low_conf_threshold": scan_result.get("low_conf_threshold", 0.35),
        "total_images": len(records),
        "low_confidence_count": counts["low_confidence"],
        "no_detection_count": counts["no_detection"],
        "no_label_file_count": counts["no_label_file"],
        "unknown_count": counts["unknown"],
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }


def read_hard_cases_summary(run_folder: str | Path) -> dict[str, Any] | None:
    """Read a run-local hard cases summary, returning None for missing or invalid data."""
    run = _existing_directory(run_folder)
    if run is None:
        return None
    candidates = (run / "hard_cases_summary.json", run / "hard_cases" / "hard_cases_summary.json")
    for path in candidates:
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else None
        except (OSError, UnicodeError, json.JSONDecodeError):
            return None
    return None


def _find_images(root: Path) -> list[Path]:
    images: list[Path] = []
    try:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            relative_parts = [part.lower() for part in path.relative_to(root).parts[:-1]]
            if "labels" in relative_parts or "hard_cases" in relative_parts:
                continue
            images.append(path.resolve())
    except OSError:
        return images
    return sorted(images)


def _build_image_lookup(root: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for path in _find_images(root):
        by_name.setdefault(path.name.lower(), path)
        by_stem.setdefault(path.stem.lower(), path)
    return by_name, by_stem


def _source_image_for(image: Path, lookup: tuple[dict[str, Path], dict[str, Path]]) -> Path | None:
    by_name, by_stem = lookup
    return by_name.get(image.name.lower()) or by_stem.get(image.stem.lower())


def _build_file_lookup(root: Path | None, suffix: str) -> dict[str, Path]:
    if root is None or not root.is_dir():
        return {}
    lookup: dict[str, Path] = {}
    try:
        for path in root.rglob(f"*{suffix}"):
            if path.is_file():
                lookup.setdefault(path.stem.lower(), path.resolve())
    except OSError:
        return lookup
    return lookup


def _classify_prediction_label(
    label: Path | None, has_prediction_labels: bool, threshold: float
) -> tuple[str, float | None, int, str]:
    if not has_prediction_labels:
        return "unknown", None, 0, "Prediction labels are unavailable."
    if label is None:
        return "no_label_file", None, 0, "No prediction label file for this image."
    lines = [line.strip() for line in label.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if not lines:
        return "no_detection", None, 0, "Prediction label exists but is empty."
    confidences: list[float] = []
    for line in lines:
        fields = line.split()
        if len(fields) == 6:
            try:
                confidences.append(float(fields[5]))
            except ValueError:
                continue
    minimum = min(confidences) if confidences else None
    if minimum is not None and minimum < threshold:
        return "low_confidence", minimum, len(lines), "Minimum confidence is below the configured threshold."
    if minimum is None:
        return "unknown", None, len(lines), "Detections found, but confidence values are unavailable."
    return "unknown", minimum, len(lines), "Detections are above the low-confidence threshold."


def _existing_directory(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    try:
        path = Path(str(value).strip()).expanduser()
        return path.resolve() if path.is_dir() else None
    except (OSError, ValueError):
        return None


def _unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 10000):
        candidate = path.with_name(f"{path.stem}_{index}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"Unable to create a unique destination for {path.name}")


def _write_report_csv(path: Path, records: list[dict[str, Any]]) -> str:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=REPORT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records)
        temporary.replace(path)
        return ""
    except (OSError, UnicodeError, csv.Error) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return f"Unable to write hard_cases_report.csv: {exc}"


def _write_summary_json(path: Path, summary: dict[str, Any]) -> str:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
        return ""
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
        return f"Unable to write hard_cases_summary.json: {exc}"
