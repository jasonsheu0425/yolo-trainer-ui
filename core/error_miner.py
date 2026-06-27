from __future__ import annotations

import csv
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATEGORIES = (
    "false_negative",
    "class_mismatch",
    "false_positive",
    "low_iou",
    "low_confidence",
    "no_detection",
    "no_label_file",
    "unknown",
)
REPORT_FIELDS = (
    "image_name",
    "primary_category",
    "all_error_flags",
    "image_path",
    "prediction_label_path",
    "ground_truth_label_path",
    "detection_count",
    "ground_truth_count",
    "min_confidence",
    "max_iou",
    "matched_count",
    "false_negative_count",
    "false_positive_count",
    "class_mismatch_count",
    "low_iou_count",
    "low_confidence_count",
    "involved_class_ids",
    "involved_class_names",
    "copied_to",
    "notes",
)


def scan_error_cases(
    run_folder: str | Path,
    source_folder: str | Path | None = None,
    labels_folder: str | Path | None = None,
    low_conf_threshold: float = 0.35,
    *,
    ground_truth_labels_folder: str | Path | None = None,
    data_yaml: str | Path | None = None,
    iou_threshold: float = 0.5,
    enable_ground_truth_comparison: bool = False,
) -> dict[str, Any]:
    """Classify run images using confidence-only or ground-truth IoU mining."""
    result: dict[str, Any] = {
        "run_folder": "",
        "source_folder": "",
        "labels_folder": "",
        "prediction_labels_folder": "",
        "ground_truth_labels_folder": "",
        "low_conf_threshold": float(low_conf_threshold),
        "iou_threshold": float(iou_threshold),
        "ground_truth_comparison_enabled": bool(enable_ground_truth_comparison),
        "class_names": {},
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
    ground_truth_labels = _existing_directory(ground_truth_labels_folder)
    if source_folder and source is None:
        result["warnings"].append(f"Source folder not found; using run images: {source_folder}")
    if labels_folder and external_labels is None:
        result["warnings"].append(f"Labels folder not found; optional labels ignored: {labels_folder}")
    result["source_folder"] = str(source) if source else ""
    result["labels_folder"] = str(external_labels) if external_labels else ""
    result["ground_truth_labels_folder"] = str(ground_truth_labels) if ground_truth_labels else ""
    if enable_ground_truth_comparison and ground_truth_labels is None:
        result["errors"].append("Ground-truth comparison is enabled, but the labels folder was not found.")
        return result
    class_names, class_name_warning = load_class_names(data_yaml)
    result["class_names"] = class_names
    if class_name_warning:
        result["warnings"].append(class_name_warning)

    images = _find_images(run)
    if not images:
        result["warnings"].append("No supported images found in the run folder.")
        return result
    source_lookup = _build_image_lookup(source) if source else ({}, {})
    prediction_labels_dir = run / "labels"
    prediction_lookup = _build_file_lookup(prediction_labels_dir, ".txt")
    external_lookup = _build_file_lookup(external_labels, ".txt") if external_labels else {}
    ground_truth_lookup = _build_file_lookup(ground_truth_labels, ".txt") if ground_truth_labels else {}
    result["prediction_labels_folder"] = str(prediction_labels_dir.resolve()) if prediction_labels_dir.is_dir() else ""
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
            ground_truth_label = ground_truth_lookup.get(image.stem.lower())
            if enable_ground_truth_comparison:
                details = compare_prediction_to_ground_truth(
                    prediction_label,
                    ground_truth_label,
                    low_conf_threshold=float(low_conf_threshold),
                    iou_threshold=float(iou_threshold),
                    class_names=class_names,
                    has_prediction_labels=has_prediction_labels,
                )
            else:
                details = _confidence_only_details(
                    prediction_label, has_prediction_labels, float(low_conf_threshold), class_names
                )
            details.update(
                {
                    "image_name": image.name,
                    "image_path": str(source_image),
                    "run_image_path": str(image),
                    "label_path": str(copy_label) if copy_label else "",
                    "prediction_label_path": str(prediction_label) if prediction_label else "",
                    "ground_truth_label_path": str(ground_truth_label) if ground_truth_label else "",
                    "copied_to": "",
                }
            )
            result["records"].append(details)
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
        category = record.get("primary_category") or record.get("category")
        category = category if category in CATEGORIES else "unknown"
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
    return {
        "run_folder": str(scan_result.get("run_folder", "")),
        "source_folder": str(scan_result.get("source_folder", "")),
        "prediction_labels_folder": str(scan_result.get("prediction_labels_folder", "")),
        "ground_truth_labels_folder": str(scan_result.get("ground_truth_labels_folder", "")),
        "output_folder": str(output_folder),
        "iou_threshold": scan_result.get("iou_threshold", 0.5),
        "low_conf_threshold": scan_result.get("low_conf_threshold", 0.35),
        "ground_truth_comparison_enabled": bool(scan_result.get("ground_truth_comparison_enabled", False)),
        "total_images": len(records),
        "total_predictions": sum(int(record.get("detection_count", 0)) for record in records),
        "total_ground_truth_boxes": sum(int(record.get("ground_truth_count", 0)) for record in records),
        "true_positive_count": sum(int(record.get("true_positive_count", 0)) for record in records),
        "false_negative_count": sum(int(record.get("false_negative_count", 0)) for record in records),
        "false_positive_count": sum(int(record.get("false_positive_count", 0)) for record in records),
        "class_mismatch_count": sum(int(record.get("class_mismatch_count", 0)) for record in records),
        "low_iou_count": sum(int(record.get("low_iou_count", 0)) for record in records),
        "low_confidence_count": sum(int(record.get("low_confidence_count", 0)) for record in records),
        "no_detection_count": sum("no_detection" in record.get("all_error_flags", "").split(";") for record in records),
        "no_label_file_count": sum("no_label_file" in record.get("all_error_flags", "").split(";") for record in records),
        "unknown_count": sum("unknown" in record.get("all_error_flags", "").split(";") for record in records),
        "class_names": scan_result.get("class_names", {}),
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


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float] | None:
    """Convert normalized YOLO xywh to xyxy, rejecting invalid boxes."""
    try:
        x_center, y_center, width, height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x_center, y_center, width, height)):
        return None
    if width <= 0 or height <= 0:
        return None
    return (
        x_center - width / 2,
        y_center - height / 2,
        x_center + width / 2,
        y_center + height / 2,
    )


def box_iou(
    box_a: tuple[float, float, float, float] | None,
    box_b: tuple[float, float, float, float] | None,
) -> float:
    """Calculate safe IoU for two xyxy boxes."""
    if box_a is None or box_b is None:
        return 0.0
    try:
        if len(box_a) != 4 or len(box_b) != 4:
            return 0.0
        ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
        bx1, by1, bx2, by2 = (float(value) for value in box_b)
    except (TypeError, ValueError):
        return 0.0
    if not all(math.isfinite(value) for value in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)):
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    if area_a <= 0 or area_b <= 0:
        return 0.0
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = area_a + area_b - intersection
    if intersection <= 0 or union <= 0:
        return 0.0
    return min(1.0, max(0.0, intersection / union))


def read_yolo_label(path: Path | None, *, prediction: bool) -> tuple[list[dict[str, Any]], list[str]]:
    """Read detect-format YOLO boxes and tolerate malformed rows."""
    if path is None or not path.is_file():
        return [], []
    boxes: list[dict[str, Any]] = []
    notes: list[str] = []
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except (OSError, UnicodeError) as exc:
        return [], [f"Unable to read {path.name}: {exc}"]
    for line_number, raw in enumerate(lines, 1):
        fields = raw.split()
        if not fields:
            continue
        expected = {5, 6} if prediction else {5}
        if len(fields) not in expected:
            notes.append(f"Ignored invalid row {line_number} in {path.name}.")
            continue
        try:
            raw_class = float(fields[0])
            coordinates = tuple(float(value) for value in fields[1:5])
            confidence = float(fields[5]) if prediction and len(fields) == 6 else None
        except ValueError:
            notes.append(f"Ignored non-numeric row {line_number} in {path.name}.")
            continue
        if not raw_class.is_integer():
            notes.append(f"Ignored non-integer class at row {line_number} in {path.name}.")
            continue
        converted = xywh_to_xyxy(coordinates)  # type: ignore[arg-type]
        if converted is None or (confidence is not None and not math.isfinite(confidence)):
            notes.append(f"Ignored invalid box at row {line_number} in {path.name}.")
            continue
        boxes.append(
            {
                "class_id": int(raw_class),
                "box": converted,
                "confidence": confidence,
                "index": len(boxes),
            }
        )
    return boxes, notes


def compare_prediction_to_ground_truth(
    prediction_label: Path | None,
    ground_truth_label: Path | None,
    *,
    low_conf_threshold: float,
    iou_threshold: float,
    class_names: dict[int, str] | None = None,
    has_prediction_labels: bool = True,
) -> dict[str, Any]:
    """Greedily match predictions to ground truth and return per-image error details."""
    predictions, prediction_notes = read_yolo_label(prediction_label, prediction=True)
    ground_truth, ground_truth_notes = read_yolo_label(ground_truth_label, prediction=False)
    notes = prediction_notes + ground_truth_notes
    flags: set[str] = set()
    if not has_prediction_labels or prediction_label is None:
        flags.add("no_label_file")
        notes.append("No prediction label file for this image.")
    elif not predictions:
        flags.add("no_detection")
        notes.append("Prediction label is empty or has no valid boxes.")
    if ground_truth_label is None:
        flags.add("unknown")
        notes.append("Ground-truth label file is missing; IoU comparison was skipped for this image.")

    confidences = [box["confidence"] for box in predictions if box["confidence"] is not None]
    low_confidence_count = sum(float(value) < low_conf_threshold for value in confidences)
    if low_confidence_count:
        flags.add("low_confidence")
    true_positive_count = class_mismatch_count = false_positive_count = low_iou_count = matched_count = 0
    max_iou = 0.0
    unmatched_ground_truth = set(range(len(ground_truth)))

    ordered_predictions = sorted(
        enumerate(predictions),
        key=lambda item: (
            item[1]["confidence"] is not None,
            float(item[1]["confidence"]) if item[1]["confidence"] is not None else -1.0,
        ),
        reverse=True,
    )
    if ground_truth_label is not None:
        for _original_index, prediction in ordered_predictions:
            best_index: int | None = None
            best_iou = 0.0
            for ground_truth_index in unmatched_ground_truth:
                candidate_iou = box_iou(prediction["box"], ground_truth[ground_truth_index]["box"])
                if candidate_iou > best_iou:
                    best_iou = candidate_iou
                    best_index = ground_truth_index
            max_iou = max(max_iou, best_iou)
            if best_index is None or best_iou <= 0:
                false_positive_count += 1
                continue
            target = ground_truth[best_index]
            same_class = prediction["class_id"] == target["class_id"]
            if best_iou >= iou_threshold:
                unmatched_ground_truth.remove(best_index)
                matched_count += 1
                if same_class:
                    true_positive_count += 1
                else:
                    class_mismatch_count += 1
            elif same_class:
                unmatched_ground_truth.remove(best_index)
                matched_count += 1
                low_iou_count += 1
            else:
                false_positive_count += 1
        false_negative_count = len(unmatched_ground_truth)
    else:
        false_negative_count = 0

    count_flags = {
        "false_negative": false_negative_count,
        "class_mismatch": class_mismatch_count,
        "false_positive": false_positive_count,
        "low_iou": low_iou_count,
        "low_confidence": low_confidence_count,
    }
    flags.update(name for name, count in count_flags.items() if count)
    if not flags:
        flags.add("unknown")
    involved_ids = sorted({box["class_id"] for box in predictions + ground_truth})
    return _record_details(
        flags,
        predictions=predictions,
        ground_truth=ground_truth,
        min_confidence=min(confidences) if confidences else None,
        max_iou=max_iou,
        matched_count=matched_count,
        true_positive_count=true_positive_count,
        false_negative_count=false_negative_count,
        false_positive_count=false_positive_count,
        class_mismatch_count=class_mismatch_count,
        low_iou_count=low_iou_count,
        low_confidence_count=low_confidence_count,
        involved_ids=involved_ids,
        class_names=class_names or {},
        notes=notes,
    )


def _confidence_only_details(
    label: Path | None,
    has_prediction_labels: bool,
    threshold: float,
    class_names: dict[int, str],
) -> dict[str, Any]:
    predictions, parse_notes = read_yolo_label(label, prediction=True)
    if not has_prediction_labels:
        flags = {"unknown"}
        notes = ["Prediction labels are unavailable."]
    elif label is None:
        flags = {"no_label_file"}
        notes = ["No prediction label file for this image."]
    elif not predictions:
        flags = {"no_detection"}
        notes = ["Prediction label is empty or has no valid boxes."]
    else:
        flags = set()
        notes = []
    notes.extend(parse_notes)
    confidences = [box["confidence"] for box in predictions if box["confidence"] is not None]
    low_confidence_count = sum(float(value) < threshold for value in confidences)
    if low_confidence_count:
        flags.add("low_confidence")
    if predictions and not confidences:
        flags.add("unknown")
        notes.append("Detections found, but confidence values are unavailable.")
    if predictions and confidences and not low_confidence_count:
        flags.add("unknown")
        notes.append("Detections are above the low-confidence threshold.")
    involved_ids = sorted({box["class_id"] for box in predictions})
    return _record_details(
        flags or {"unknown"},
        predictions=predictions,
        ground_truth=[],
        min_confidence=min(confidences) if confidences else None,
        max_iou=None,
        matched_count=0,
        true_positive_count=0,
        false_negative_count=0,
        false_positive_count=0,
        class_mismatch_count=0,
        low_iou_count=0,
        low_confidence_count=low_confidence_count,
        involved_ids=involved_ids,
        class_names=class_names,
        notes=notes,
    )


def _record_details(
    flags: set[str],
    *,
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    min_confidence: float | None,
    max_iou: float | None,
    matched_count: int,
    true_positive_count: int,
    false_negative_count: int,
    false_positive_count: int,
    class_mismatch_count: int,
    low_iou_count: int,
    low_confidence_count: int,
    involved_ids: list[int],
    class_names: dict[int, str],
    notes: list[str],
) -> dict[str, Any]:
    ordered_flags = [category for category in CATEGORIES if category in flags]
    if not ordered_flags:
        ordered_flags = ["unknown"]
    involved_names = [class_names.get(class_id, str(class_id)) for class_id in involved_ids]
    return {
        "category": ordered_flags[0],
        "primary_category": ordered_flags[0],
        "all_error_flags": ";".join(ordered_flags),
        "detection_count": len(predictions),
        "ground_truth_count": len(ground_truth),
        "min_confidence": min_confidence,
        "max_iou": max_iou,
        "matched_count": matched_count,
        "true_positive_count": true_positive_count,
        "false_negative_count": false_negative_count,
        "false_positive_count": false_positive_count,
        "class_mismatch_count": class_mismatch_count,
        "low_iou_count": low_iou_count,
        "low_confidence_count": low_confidence_count,
        "involved_class_ids": ";".join(str(value) for value in involved_ids),
        "involved_class_names": ";".join(involved_names),
        "notes": " ".join(notes),
    }


def load_class_names(data_yaml: str | Path | None) -> tuple[dict[int, str], str]:
    if data_yaml is None or not str(data_yaml).strip():
        return {}, ""
    path = Path(str(data_yaml).strip()).expanduser()
    if not path.is_file():
        return {}, f"Class names YAML not found; class IDs will be used: {path}"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        names = payload.get("names") if isinstance(payload, dict) else None
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}, ""
        if isinstance(names, dict):
            return {int(key): str(value) for key, value in names.items()}, ""
        return {}, "data.yaml has no usable names; class IDs will be used."
    except (OSError, UnicodeError, ValueError, TypeError, yaml.YAMLError) as exc:
        return {}, f"Unable to read class names; class IDs will be used: {exc}"


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
