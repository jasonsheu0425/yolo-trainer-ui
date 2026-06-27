from __future__ import annotations

import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
from typing import Any, Callable

import yaml

from core.report_reader import filter_report, read_hard_cases_report


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")
BUILD_REPORT_FIELDS = (
    "source_type",
    "original_image_path",
    "original_label_path",
    "new_image_path",
    "new_label_path",
    "split",
    "primary_category",
    "all_error_flags",
    "label_source",
    "copied",
    "notes",
)


def preview_dataset_build(
    options: dict[str, Any],
    *,
    progress_callback: Callable[[int, str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Validate inputs and estimate a build without writing files."""
    result: dict[str, Any] = {
        "errors": [],
        "warnings": [],
        "base_images": 0,
        "base_labels": 0,
        "selected_hard_cases": 0,
        "hard_case_split_counts": {split: 0 for split in SPLITS},
        "skipped_samples": 0,
        "missing_labels": 0,
        "output_exists": False,
        "will_overwrite": False,
        "class_names": {},
        "base_root": "",
        "cancelled": False,
    }
    _progress(progress_callback, 0, "Previewing...")
    _log(log_callback, "Scanning base dataset...")
    base, error = load_base_dataset(
        options.get("base_data_yaml", ""),
        cancel_callback=cancel_callback,
    )
    if error == "Operation cancelled.":
        result["cancelled"] = True
        return result
    if error:
        result["errors"].append(error)
        return result
    result["class_names"] = base["names"]
    result["base_root"] = str(base["root"])
    result["base_images"] = sum(len(base["images"][split]) for split in SPLITS)
    result["base_labels"] = sum(
        1 for split in SPLITS for image in base["images"][split] if label_for_image(image).is_file()
    )
    _log(
        log_callback,
        f"Base dataset scan: {result['base_images']} image(s), {result['base_labels']} label(s).",
    )
    missing_base_labels = result["base_images"] - result["base_labels"]
    if missing_base_labels > 0:
        _log(log_callback, f"WARNING: {missing_base_labels} base dataset image(s) have no label file.")
    if _cancelled(cancel_callback):
        result["cancelled"] = True
        return result

    ratios, ratio_error = _validated_ratios(options)
    if ratio_error:
        result["errors"].append(ratio_error)
    output, output_error = _output_path(options.get("output_folder", ""))
    if output_error:
        result["errors"].append(output_error)
    else:
        result["output_exists"] = output.exists()
        result["will_overwrite"] = output.exists() and bool(options.get("overwrite_output", False))
        if _same_path(output, base["root"]):
            result["errors"].append("Output folder cannot be the original dataset root.")

    rows: list[dict[str, str]] = []
    if options.get("include_hard_cases", True):
        _log(log_callback, "Reading and filtering hard-cases report...")
        rows, report_error = read_hard_cases_report(options.get("hard_cases_report", ""))
        if report_error:
            result["errors"].append(report_error)
        else:
            rows = select_hard_case_rows(
                rows,
                set(options.get("selected_categories", [])),
                str(options.get("filter_mode", "Primary or Any Flag")),
            )
    eligible: list[dict[str, str]] = []
    for row in rows:
        image = _first_existing_file(row, ("copied_to", "image_path"))
        label = _first_existing_file(row, ("ground_truth_label_path", "prediction_label_path"))
        if image is None:
            result["skipped_samples"] += 1
            result["warnings"].append(f"Hard-case image not found: {row.get('image_name', '')}")
            continue
        if label is None:
            result["missing_labels"] += 1
            if options.get("skip_without_labels", False):
                result["skipped_samples"] += 1
                continue
        eligible.append(row)
    result["selected_hard_cases"] = len(eligible)
    if ratios:
        result["hard_case_split_counts"] = allocate_split_counts(len(eligible), ratios)
    _log(
        log_callback,
        f"Hard cases selected: {len(eligible)}; skipped: {result['skipped_samples']}; missing labels: {result['missing_labels']}.",
    )
    _progress(progress_callback, 100, "Preview completed")
    return result


def build_dataset(
    options: dict[str, Any],
    *,
    progress_callback: Callable[[int, str], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
    cancel_callback: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Build a new YOLO dataset version without modifying the base dataset."""
    preview = preview_dataset_build(
        options,
        progress_callback=lambda _value, status: _progress(progress_callback, 2, status),
        log_callback=log_callback,
        cancel_callback=cancel_callback,
    )
    result: dict[str, Any] = {
        **preview,
        "output_folder": None,
        "data_yaml": None,
        "report_csv": None,
        "summary_json": None,
        "total_base_images_copied": 0,
        "total_hard_cases_copied": 0,
        "total_empty_labels_created": 0,
        "train_count": 0,
        "val_count": 0,
        "test_count": 0,
        "cancelled": bool(preview.get("cancelled", False)),
    }
    if result["errors"] or result["cancelled"]:
        return result
    base, error = load_base_dataset(options.get("base_data_yaml", ""), cancel_callback=cancel_callback)
    if error == "Operation cancelled.":
        result["cancelled"] = True
        return result
    if error:
        result["errors"].append(error)
        return result
    output, output_error = _output_path(options.get("output_folder", ""))
    if output_error:
        result["errors"].append(output_error)
        return result
    if output.exists():
        if not options.get("overwrite_output", False):
            result["errors"].append("Output folder already exists. Enable overwrite to replace it.")
            return result
        if not _safe_to_replace(output, base["root"]):
            result["errors"].append("Refusing to overwrite an unsafe or original dataset path.")
            return result
        try:
            shutil.rmtree(output)
        except OSError as exc:
            result["errors"].append(f"Unable to clear output folder: {exc}")
            return result
    if _cancelled(cancel_callback):
        result["cancelled"] = True
        return result
    try:
        for split in SPLITS:
            (output / "images" / split).mkdir(parents=True, exist_ok=True)
            (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        output = output.resolve()
        result["output_folder"] = output
        _log(log_callback, f"Output dataset folder created: {output}")
    except OSError as exc:
        result["errors"].append(f"Unable to create output dataset: {exc}")
        return result

    report_rows: list[dict[str, Any]] = []
    base_units = sum(len(base["images"][split]) for split in SPLITS) if (options.get("copy_original_images", True) or options.get("copy_original_labels", True)) else 0
    hard_units = int(preview.get("selected_hard_cases", 0))
    total_units = max(1, base_units + hard_units + 3)
    completed_units = 0
    if options.get("copy_original_images", True) or options.get("copy_original_labels", True):
        _progress(progress_callback, 3, "Copying base dataset...")
        for split in SPLITS:
            for image in base["images"][split]:
                if _cancelled(cancel_callback):
                    result["cancelled"] = True
                    _log(log_callback, "Build cancelled while copying base dataset.")
                    return result
                label = label_for_image(image)
                destination_image, destination_label = destination_pair(output, split, image, "base")
                copied_image = copied_label = False
                notes: list[str] = []
                if options.get("copy_original_images", True):
                    copied_image, copy_error = _copy_file(image, destination_image)
                    if copy_error:
                        result["errors"].append(copy_error)
                    if copied_image:
                        result["total_base_images_copied"] += 1
                if options.get("copy_original_labels", True) and label.is_file():
                    copied_label, copy_error = _copy_file(label, destination_label)
                    if copy_error:
                        result["errors"].append(copy_error)
                elif options.get("copy_original_labels", True):
                    notes.append("Base label missing.")
                report_rows.append(
                    _build_report_row(
                        "base_dataset", image, label if label.is_file() else None,
                        destination_image if copied_image else None,
                        destination_label if copied_label else None,
                        split, "", "", "copied_base_label" if copied_label else "missing",
                        copied_image or copied_label, " ".join(notes),
                    )
                )
                completed_units += 1
                if completed_units % 25 == 0 or completed_units == base_units:
                    _progress(progress_callback, _build_percent(completed_units, total_units), "Copying base dataset...")
                if completed_units % 500 == 0 or completed_units == base_units:
                    _log(log_callback, f"Base dataset copy progress: {completed_units}/{base_units} image pair(s) processed.")

    hard_rows: list[dict[str, str]] = []
    if options.get("include_hard_cases", True):
        loaded_rows, report_error = read_hard_cases_report(options.get("hard_cases_report", ""))
        if report_error:
            result["errors"].append(report_error)
        else:
            hard_rows = select_hard_case_rows(
                loaded_rows,
                set(options.get("selected_categories", [])),
                str(options.get("filter_mode", "Primary or Any Flag")),
            )
    prepared: list[tuple[dict[str, str], Path, Path | None, str]] = []
    for row in hard_rows:
        image = _first_existing_file(row, ("copied_to", "image_path"))
        if image is None:
            continue
        ground_truth = _existing_file(row.get("ground_truth_label_path", ""))
        prediction = _existing_file(row.get("prediction_label_path", ""))
        label = ground_truth or prediction
        label_source = "ground_truth_label" if ground_truth else ("prediction_label" if prediction else "empty_label")
        if label is None and options.get("skip_without_labels", False):
            continue
        prepared.append((row, image, label, label_source))
    random.Random(42).shuffle(prepared)
    ratios, _ratio_error = _validated_ratios(options)
    split_counts = allocate_split_counts(len(prepared), ratios)
    split_sequence = [split for split in SPLITS for _ in range(split_counts[split])]
    _log(log_callback, f"Copying {len(prepared)} selected hard case(s).")
    for (row, image, label, label_source), split in zip(prepared, split_sequence):
        if _cancelled(cancel_callback):
            result["cancelled"] = True
            _log(log_callback, "Build cancelled while copying hard cases.")
            return result
        _progress(progress_callback, _build_percent(completed_units, total_units), "Copying hard cases...")
        destination_image, destination_label = destination_pair(output, split, image, "hardcase")
        copied_image, copy_error = _copy_file(image, destination_image)
        if copy_error:
            result["errors"].append(copy_error)
            continue
        copied_label = False
        notes: list[str] = []
        if options.get("copy_labels_if_found", True) and label is not None:
            copied_label, copy_error = _copy_file(label, destination_label)
            if copy_error:
                result["errors"].append(copy_error)
        elif label is None and not options.get("skip_without_labels", False):
            try:
                destination_label.write_text("", encoding="utf-8")
                copied_label = True
                result["total_empty_labels_created"] += 1
            except OSError as exc:
                result["errors"].append(f"Unable to create empty label {destination_label}: {exc}")
        elif label is not None:
            label_source = "missing"
            notes.append("Label copying disabled.")
        if copied_image:
            result["total_hard_cases_copied"] += 1
        report_rows.append(
            _build_report_row(
                "hard_case", image, label,
                destination_image if copied_image else None,
                destination_label if copied_label else None,
                split, row.get("primary_category", ""), row.get("all_error_flags", ""),
                label_source, copied_image, " ".join(notes),
            )
        )
        completed_units += 1
        hard_completed = completed_units - base_units
        if hard_completed % 25 == 0 or hard_completed == len(prepared):
            _log(log_callback, f"Hard-case copy progress: {hard_completed}/{len(prepared)} sample(s) processed.")

    result["total_missing_labels"] = preview["missing_labels"]
    for split in SPLITS:
        count = len(list((output / "images" / split).iterdir()))
        result[f"{split}_count"] = count
    data_yaml = output / "data.yaml"
    report_csv = output / "dataset_build_report.csv"
    summary_json = output / "dataset_build_summary.json"
    if _cancelled(cancel_callback):
        result["cancelled"] = True
        _log(log_callback, "Build cancelled before writing output files.")
        return result
    _progress(progress_callback, 92, "Writing data.yaml...")
    data_error = _write_data_yaml(data_yaml, output, base["names"])
    if _cancelled(cancel_callback):
        result["cancelled"] = True
        _log(log_callback, "Build cancelled before writing reports.")
        return result
    _progress(progress_callback, 96, "Writing reports...")
    report_error = _write_build_report(report_csv, report_rows)
    summary = _build_summary(options, result, base["names"], data_yaml)
    summary_error = _write_json(summary_json, summary)
    for write_error in (data_error, report_error, summary_error):
        if write_error:
            result["errors"].append(write_error)
    result["data_yaml"] = data_yaml if not data_error else None
    result["report_csv"] = report_csv if not report_error else None
    result["summary_json"] = summary_json if not summary_error else None
    if not result["errors"]:
        _progress(progress_callback, 100, "Completed")
        _log(
            log_callback,
            f"Copy summary: base images={result['total_base_images_copied']}, "
            f"hard cases={result['total_hard_cases_copied']}, empty labels={result['total_empty_labels_created']}.",
        )
        _log(log_callback, f"Dataset build completed: {output}")
    return result


def load_base_dataset(
    data_yaml: str | Path,
    *,
    cancel_callback: Callable[[], bool] | None = None,
) -> tuple[dict[str, Any], str]:
    raw = str(data_yaml).strip()
    if not raw:
        return {}, "Base data.yaml path is empty."
    path = Path(raw).expanduser()
    if not path.is_file():
        return {}, f"Base data.yaml not found: {path}"
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return {}, f"Unable to read base data.yaml: {exc}"
    if not isinstance(payload, dict):
        return {}, "Base data.yaml root must be a mapping."
    names = _normalize_names(payload.get("names"))
    if names is None:
        return {}, "Base data.yaml has no usable names."
    root_value = payload.get("path")
    if root_value in (None, ""):
        root = path.parent.resolve()
    else:
        root_candidate = Path(str(root_value)).expanduser()
        root = root_candidate.resolve() if root_candidate.is_absolute() else (path.parent / root_candidate).resolve()
    images: dict[str, list[Path]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        if _cancelled(cancel_callback):
            return {}, "Operation cancelled."
        entry = payload.get(split)
        if entry is None:
            continue
        entries = entry if isinstance(entry, list) else [entry]
        for item in entries:
            item_text = str(item)
            candidate = Path(item_text).expanduser()
            candidate = candidate if candidate.is_absolute() else root / candidate
            if not candidate.exists() and item_text.replace("\\", "/").startswith("../"):
                fallback = root / item_text.replace("\\", "/")[3:]
                if fallback.exists():
                    candidate = fallback
            if candidate.is_dir():
                try:
                    for index, file in enumerate(candidate.rglob("*")):
                        if index % 100 == 0 and _cancelled(cancel_callback):
                            return {}, "Operation cancelled."
                        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                            images[split].append(file.resolve())
                except OSError as exc:
                    return {}, f"Unable to scan base dataset split {split}: {exc}"
    return {"yaml": path.resolve(), "root": root, "names": names, "images": images}, ""


def select_hard_case_rows(
    rows: list[dict[str, str]], selected_categories: set[str], filter_mode: str
) -> list[dict[str, str]]:
    if not selected_categories:
        return []
    selected: list[dict[str, str]] = []
    for row in rows:
        if any(filter_report([row], category, "", filter_mode) for category in selected_categories):
            selected.append(row)
    return selected


def allocate_split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total <= 0:
        return {split: 0 for split in SPLITS}
    raw = {split: total * ratios[split] for split in SPLITS}
    counts = {split: math.floor(raw[split]) for split in SPLITS}
    remaining = total - sum(counts.values())
    order = sorted(SPLITS, key=lambda split: (raw[split] - counts[split], ratios[split]), reverse=True)
    for split in order:
        if remaining <= 0:
            break
        if ratios[split] > 0:
            counts[split] += 1
            remaining -= 1
    if remaining:
        counts["train"] += remaining
    return counts


def label_for_image(image: Path) -> Path:
    parts = list(image.parts)
    lowered = [part.lower() for part in parts]
    if "images" in lowered:
        index = len(lowered) - 1 - lowered[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def destination_pair(output: Path, split: str, image: Path, prefix: str) -> tuple[Path, Path]:
    image_destination = output / "images" / split / image.name
    label_destination = output / "labels" / split / f"{image.stem}.txt"
    if image_destination.exists() or label_destination.exists():
        digest = hashlib.sha1(str(image.resolve()).casefold().encode("utf-8")).hexdigest()[:8]
        base_stem = f"{prefix}_{image.stem}_{digest}"
        stem = base_stem
        image_destination = image_destination.with_name(f"{stem}{image.suffix.lower()}")
        label_destination = label_destination.with_name(f"{stem}.txt")
        counter = 2
        while image_destination.exists() or label_destination.exists():
            stem = f"{base_stem}_{counter}"
            image_destination = image_destination.with_name(f"{stem}{image.suffix.lower()}")
            label_destination = label_destination.with_name(f"{stem}.txt")
            counter += 1
    return image_destination, label_destination


def _validated_ratios(options: dict[str, Any]) -> tuple[dict[str, float], str]:
    try:
        ratios = {
            "train": float(options.get("train_ratio", 0.8)),
            "val": float(options.get("val_ratio", 0.2)),
            "test": float(options.get("test_ratio", 0.0)),
        }
    except (TypeError, ValueError):
        return {}, "Split ratios must be numbers."
    if any(value < 0 or value > 1 for value in ratios.values()):
        return {}, "Split ratios must be between 0 and 1."
    if not math.isclose(sum(ratios.values()), 1.0, abs_tol=0.001):
        return {}, "Train, val, and test ratios must sum to 1.0."
    return ratios, ""


def _normalize_names(value: Any) -> dict[int, str] | None:
    if isinstance(value, list):
        return {index: str(name) for index, name in enumerate(value)}
    if isinstance(value, dict):
        try:
            return {int(key): str(name) for key, name in value.items()}
        except (TypeError, ValueError):
            return None
    return None


def _first_existing_file(row: dict[str, str], fields: tuple[str, ...]) -> Path | None:
    for field in fields:
        path = _existing_file(row.get(field, ""))
        if path:
            return path
    return None


def _existing_file(value: Any) -> Path | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        path = Path(raw).expanduser()
        return path.resolve() if path.is_file() else None
    except (OSError, ValueError):
        return None


def _output_path(value: Any) -> tuple[Path, str]:
    raw = str(value or "").strip()
    if not raw:
        return Path(), "Output dataset folder is empty."
    try:
        return Path(raw).expanduser().resolve(), ""
    except (OSError, ValueError) as exc:
        return Path(), f"Invalid output dataset folder: {exc}"


def _same_path(left: Path, right: Path) -> bool:
    try:
        return left.resolve() == right.resolve()
    except OSError:
        return False


def _safe_to_replace(output: Path, base_root: Path) -> bool:
    try:
        resolved = output.resolve()
        return resolved != base_root.resolve() and resolved != resolved.parent and len(resolved.parts) >= 3
    except OSError:
        return False


def _copy_file(source: Path, destination: Path) -> tuple[bool, str]:
    try:
        shutil.copy2(source, destination)
        return True, ""
    except OSError as exc:
        return False, f"Unable to copy {source} to {destination}: {exc}"


def _build_report_row(
    source_type: str,
    original_image: Path,
    original_label: Path | None,
    new_image: Path | None,
    new_label: Path | None,
    split: str,
    primary_category: str,
    all_error_flags: str,
    label_source: str,
    copied: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "source_type": source_type,
        "original_image_path": str(original_image),
        "original_label_path": str(original_label) if original_label else "",
        "new_image_path": str(new_image) if new_image else "",
        "new_label_path": str(new_label) if new_label else "",
        "split": split,
        "primary_category": primary_category,
        "all_error_flags": all_error_flags,
        "label_source": label_source,
        "copied": copied,
        "notes": notes,
    }


def _write_data_yaml(path: Path, output: Path, names: dict[int, str]) -> str:
    payload = {
        "path": output.as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": names,
    }
    try:
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return ""
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return f"Unable to write data.yaml: {exc}"


def _write_build_report(path: Path, rows: list[dict[str, Any]]) -> str:
    try:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=BUILD_REPORT_FIELDS, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        return ""
    except (OSError, UnicodeError, csv.Error) as exc:
        return f"Unable to write dataset_build_report.csv: {exc}"


def _build_summary(
    options: dict[str, Any], result: dict[str, Any], class_names: dict[int, str], data_yaml: Path
) -> dict[str, Any]:
    return {
        "base_data_yaml": str(options.get("base_data_yaml", "")),
        "hard_cases_report": str(options.get("hard_cases_report", "")),
        "output_dataset": str(result.get("output_folder") or ""),
        "data_yaml": str(data_yaml),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "class_names": class_names,
        "copy_original_dataset": bool(options.get("copy_original_images", True) or options.get("copy_original_labels", True)),
        "selected_categories": list(options.get("selected_categories", [])),
        "filter_mode": str(options.get("filter_mode", "Primary or Any Flag")),
        "split_ratios": {split: float(options.get(f"{split}_ratio", 0.0)) for split in SPLITS},
        "total_base_images_copied": result.get("total_base_images_copied", 0),
        "total_hard_cases_selected": result.get("selected_hard_cases", 0),
        "total_hard_cases_copied": result.get("total_hard_cases_copied", 0),
        "total_missing_labels": result.get("total_missing_labels", 0),
        "total_empty_labels_created": result.get("total_empty_labels_created", 0),
        "train_count": result.get("train_count", 0),
        "val_count": result.get("val_count", 0),
        "test_count": result.get("test_count", 0),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    try:
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return ""
    except (OSError, UnicodeError, TypeError, ValueError) as exc:
        return f"Unable to write dataset_build_summary.json: {exc}"


def _cancelled(callback: Callable[[], bool] | None) -> bool:
    if callback is None:
        return False
    try:
        return bool(callback())
    except Exception:
        return False


def _progress(callback: Callable[[int, str], None] | None, value: int, status: str) -> None:
    if callback is not None:
        try:
            callback(max(0, min(100, int(value))), str(status))
        except Exception:
            pass


def _log(callback: Callable[[str], None] | None, message: str) -> None:
    if callback is not None:
        try:
            callback(str(message))
        except Exception:
            pass


def _build_percent(completed: int, total: int) -> int:
    return 3 + int(87 * max(0, completed) / max(1, total))
