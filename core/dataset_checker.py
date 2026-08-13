from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_IMAGES_PER_SPLIT = 500
LOW_INSTANCE_WARNING = 20


def load_dataset_manifest(yaml_path: str | Path) -> tuple[dict[str, Any], list[str]]:
    """Resolve classes and available splits with Dataset Check-compatible rules."""
    path = Path(yaml_path).expanduser().resolve()
    errors: list[str] = []
    warnings: list[str] = []
    if not path.is_file():
        return {}, [f"Dataset YAML not found: {path}"]
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        return {}, [f"Unable to read dataset YAML: {exc}"]
    if not isinstance(data, dict):
        return {}, ["Dataset YAML root must be a mapping."]
    names = _normalize_names(data.get("names"), errors)
    if not names:
        return {}, errors or ["Dataset YAML has no usable classes."]
    root = _resolve_root(path.parent, data.get("path"))
    splits: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        if split in data and data[split] not in (None, ""):
            split_errors: list[str] = []
            images = _resolve_images(root, data[split], split_errors, warnings, split)
            if split_errors:
                errors.extend(split_errors)
            elif images:
                splits[split] = images
    if not splits:
        errors.append("Dataset has no readable image split.")
    return {"yaml": path, "root": root, "names": names, "splits": splits, "warnings": warnings}, errors


def check_dataset(yaml_path: str | Path) -> dict[str, Any]:
    result: dict[str, Any] = {"errors": [], "warnings": [], "summary": {}, "class_counts": {}}
    path = Path(yaml_path).expanduser().resolve()
    if not path.is_file():
        result["errors"].append(f"找不到 YAML：{path}")
        return result
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        result["errors"].append(f"無法讀取 YAML：{exc}")
        return result
    if not isinstance(data, dict):
        result["errors"].append("YAML 根節點必須是 mapping。")
        return result

    for key in ("train", "val", "names"):
        if key not in data:
            result["errors"].append(f"缺少必要欄位：{key}")
    if "path" not in data:
        result["warnings"].append("未提供 path；依 Ultralytics 規則使用 YAML 所在資料夾作為 dataset root。")
    names = _normalize_names(data.get("names"), result["errors"])
    if not names or result["errors"]:
        return result

    root = _resolve_root(path.parent, data.get("path"))
    split_images: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        if split not in data:
            continue
        images = _resolve_images(root, data[split], result["errors"], result["warnings"], split)
        split_images[split] = images
        if not images:
            result["warnings"].append(f"{split} 沒有找到支援的影像。")
        elif len(images) > MAX_IMAGES_PER_SPLIT:
            result["warnings"].append(
                f"{split} 有 {len(images)} 張影像；為避免檢查過久，只抽查前 {MAX_IMAGES_PER_SPLIT} 張。"
            )

    counts: Counter[int] = Counter()
    missing = empty = invalid = checked = 0
    for split, images in split_images.items():
        for image in images[:MAX_IMAGES_PER_SPLIT]:
            checked += 1
            label = _label_path(image)
            if not label.is_file():
                missing += 1
                result["warnings"].append(f"缺少標籤：{label}")
                continue
            try:
                lines = label.read_text(encoding="utf-8-sig").splitlines()
            except (OSError, UnicodeError) as exc:
                invalid += 1
                result["errors"].append(f"無法讀取標籤 {label}：{exc}")
                continue
            non_empty = [line.strip() for line in lines if line.strip()]
            if not non_empty:
                empty += 1
                result["warnings"].append(f"空標籤：{label}")
                continue
            for line_number, line in enumerate(non_empty, 1):
                error, class_id = _validate_label_line(line, len(names))
                if error:
                    invalid += 1
                    result["errors"].append(f"{label}:{line_number} — {error}")
                elif class_id is not None:
                    counts[class_id] += 1

    train_set = {item.resolve() for item in split_images.get("train", [])}
    val_set = {item.resolve() for item in split_images.get("val", [])}
    overlap = train_set & val_set
    if overlap:
        result["warnings"].append(f"train 與 val 重複 {len(overlap)} 張影像。")
    for class_id, name in names.items():
        if counts[class_id] < LOW_INSTANCE_WARNING:
            result["warnings"].append(f"類別 {class_id} ({name}) 只有 {counts[class_id]} 個標註，少於 {LOW_INSTANCE_WARNING}。")

    result["class_counts"] = {f"{class_id}: {names[class_id]}": counts[class_id] for class_id in sorted(names)}
    result["summary"] = {
        "dataset_root": str(root),
        "classes": len(names),
        "train_images": len(split_images.get("train", [])),
        "val_images": len(split_images.get("val", [])),
        "checked_images": checked,
        "instances": sum(counts.values()),
        "missing_labels": missing,
        "empty_labels": empty,
        "invalid_lines": invalid,
    }
    return result


def _normalize_names(value: Any, errors: list[str]) -> dict[int, str]:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        return dict(enumerate(value))
    if isinstance(value, dict):
        try:
            normalized = {int(key): str(name) for key, name in value.items()}
        except (TypeError, ValueError):
            errors.append("names 的 key 必須是整數。")
            return {}
        if sorted(normalized) != list(range(len(normalized))):
            errors.append("names 的類別 ID 必須從 0 開始且連續。")
            return {}
        return normalized
    errors.append("names 必須是字串 list 或 ID/name mapping。")
    return {}


def _resolve_root(yaml_dir: Path, value: Any) -> Path:
    if value in (None, ""):
        return yaml_dir.resolve()
    candidate = Path(str(value)).expanduser()
    return candidate.resolve() if candidate.is_absolute() else (yaml_dir / candidate).resolve()


def _resolve_images(root: Path, value: Any, errors: list[str], warnings: list[str], split: str) -> list[Path]:
    values = value if isinstance(value, list) else [value]
    images: list[Path] = []
    for entry in values:
        entry_text = str(entry)
        candidate = Path(entry_text).expanduser()
        candidate = candidate if candidate.is_absolute() else root / candidate
        if not candidate.exists() and entry_text.replace("\\", "/").startswith("../"):
            fallback = root / entry_text.replace("\\", "/")[3:]
            if fallback.exists():
                candidate = fallback
                warnings.append(f"{split} 的 ../ 路徑不存在，已依 Ultralytics 相容規則改用：{candidate.resolve()}")
        if candidate.is_dir():
            images.extend(path for path in candidate.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        elif candidate.is_file() and candidate.suffix.lower() == ".txt":
            try:
                for line in candidate.read_text(encoding="utf-8-sig").splitlines():
                    item = Path(line.strip())
                    item = item if item.is_absolute() else candidate.parent / item
                    if item.suffix.lower() in IMAGE_EXTENSIONS:
                        images.append(item.resolve())
            except (OSError, UnicodeError) as exc:
                errors.append(f"無法讀取 {split} 清單 {candidate}：{exc}")
        elif candidate.is_file() and candidate.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(candidate.resolve())
        else:
            errors.append(f"{split} 路徑不存在或不支援：{candidate}")
    return sorted(set(images))


def _label_path(image: Path) -> Path:
    parts = list(image.parts)
    lower = [part.lower() for part in parts]
    if "images" in lower:
        index = len(lower) - 1 - lower[::-1].index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image.with_suffix(".txt")


def _validate_label_line(line: str, class_count: int) -> tuple[str | None, int | None]:
    fields = line.split()
    if len(fields) != 5:
        return "每行必須有 5 個欄位：class_id x_center y_center width height", None
    try:
        raw_class = float(fields[0])
        values = [float(value) for value in fields[1:]]
    except ValueError:
        return "欄位必須是數值。", None
    if not raw_class.is_integer():
        return "class_id 必須是整數。", None
    class_id = int(raw_class)
    if not 0 <= class_id < class_count:
        return f"class_id {class_id} 超出 names 範圍。", None
    if any(value < 0 or value > 1 for value in values):
        return "座標與寬高必須介於 0 到 1。", None
    if values[2] <= 0 or values[3] <= 0:
        return "width 與 height 必須大於 0。", None
    return None, class_id
