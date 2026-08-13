"""Safe YOLO detection-label loading, validation, atomic save, and repair backup."""
from __future__ import annotations

from datetime import datetime
import hashlib
import math
import os
from pathlib import Path

from domain.annotation import AnnotationDocument, BoundingBox, InvalidLabelLine
from persistence.atomic_writer import atomic_write_text


SAVE_PRECISION = 6


class YoloAnnotationStore:
    """Owns YOLO label-path resolution and label text storage mechanics."""

    def __init__(self, backup_root: Path | None = None) -> None:
        local_data = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
        self.backup_root = backup_root or local_data / "YOLO-Trainer-UI" / "annotation_backups"

    @staticmethod
    def resolve_label_path(image_path: str | Path) -> Path:
        image = Path(image_path)
        parts = list(image.parts)
        lowered = [part.casefold() for part in parts]
        if "images" in lowered:
            index = len(lowered) - 1 - lowered[::-1].index("images")
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
        return image.with_suffix(".txt")

    def load(self, image_path: str | Path, class_count: int) -> AnnotationDocument:
        image = Path(image_path).resolve()
        label = self.resolve_label_path(image)
        if not label.is_file():
            return AnnotationDocument(image, label)
        try:
            lines = label.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            return AnnotationDocument(image, label, invalid_lines=[InvalidLabelLine(0, "", f"read_error:{exc}")], label_existed=True)
        boxes: list[BoundingBox] = []
        invalid: list[InvalidLabelLine] = []
        for number, raw in enumerate(lines, 1):
            if not raw.strip():
                continue
            parsed, error = self.parse_line(raw, class_count)
            if error:
                invalid.append(InvalidLabelLine(number, raw, error))
            elif parsed is not None:
                boxes.append(parsed)
        return AnnotationDocument(image, label, boxes, invalid, label_existed=True)

    @staticmethod
    def parse_line(raw: str, class_count: int) -> tuple[BoundingBox | None, str]:
        fields = raw.split()
        if len(fields) != 5:
            return None, "wrong_field_count"
        try:
            raw_class = float(fields[0])
            values = [float(value) for value in fields[1:]]
        except ValueError:
            return None, "non_numeric"
        if not all(math.isfinite(value) for value in (raw_class, *values)):
            return None, "non_finite"
        if not raw_class.is_integer():
            return None, "class_not_integer"
        box = BoundingBox(int(raw_class), *values)
        if not 0 <= box.class_id < class_count:
            return None, "invalid_class"
        if not box.is_valid(class_count):
            return None, "invalid_geometry"
        return box, ""

    def save(self, document: AnnotationDocument, *, allow_repair: bool = False) -> Path:
        if document.invalid_lines and not allow_repair:
            raise ValueError("repair_confirmation_required")
        if document.invalid_lines and document.label_path.is_file():
            self.backup_original(document.label_path)
        content = "\n".join(self.format_box(box) for box in document.boxes)
        if content:
            content += "\n"
        atomic_write_text(document.label_path, content)
        document.label_existed = True
        document.invalid_lines.clear()
        document.dirty = False
        return document.label_path

    def backup_original(self, label_path: Path) -> Path:
        digest = hashlib.sha256(str(label_path.resolve()).casefold().encode("utf-8")).hexdigest()[:16]
        stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S%f%z")
        target = self.backup_root / digest / f"{stamp}-{label_path.name}"
        atomic_write_text(target, label_path.read_text(encoding="utf-8-sig"))
        return target

    @staticmethod
    def format_box(box: BoundingBox) -> str:
        if not box.is_valid():
            raise ValueError("invalid_bounding_box")
        return f"{box.class_id} {box.x_center:.{SAVE_PRECISION}f} {box.y_center:.{SAVE_PRECISION}f} {box.width:.{SAVE_PRECISION}f} {box.height:.{SAVE_PRECISION}f}"
