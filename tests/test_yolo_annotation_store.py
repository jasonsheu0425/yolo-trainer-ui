from __future__ import annotations

from pathlib import Path

import pytest

from domain.annotation import AnnotationDocument, AnnotationStatus, BoundingBox
from persistence.yolo_annotation_store import YoloAnnotationStore


def make_image(tmp_path: Path, name: str = "images/train/nested/sample.jpg") -> Path:
    image = tmp_path / name
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"image-placeholder")
    return image


def test_valid_empty_and_missing_label_load(tmp_path):
    store = YoloAnnotationStore(tmp_path / "backups")
    image = make_image(tmp_path)
    missing = store.load(image, 2)
    assert missing.status is AnnotationStatus.UNLABELED
    assert missing.label_path == tmp_path / "labels/train/nested/sample.txt"
    missing.label_path.parent.mkdir(parents=True)
    missing.label_path.write_text("", encoding="utf-8")
    assert store.load(image, 2).status is AnnotationStatus.EMPTY
    missing.label_path.write_text("1 0.500000 0.500000 0.250000 0.250000\n", encoding="utf-8")
    loaded = store.load(image, 2)
    assert loaded.status is AnnotationStatus.LABELED and loaded.boxes[0].class_id == 1


@pytest.mark.parametrize("line,error", [
    ("garbage", "wrong_field_count"), ("x 0.5 0.5 0.2 0.2", "non_numeric"),
    ("3 0.5 0.5 0.2 0.2", "invalid_class"), ("0 nan 0.5 0.2 0.2", "non_finite"),
    ("0 inf 0.5 0.2 0.2", "non_finite"), ("0 1.2 0.5 0.2 0.2", "invalid_geometry"),
    ("0 0.5 0.5 0 0.2", "invalid_geometry"),
])
def test_malformed_lines_are_reported(line, error):
    box, actual = YoloAnnotationStore.parse_line(line, 2)
    assert box is None and actual == error


def test_atomic_save_creates_nested_label_and_empty_negative(tmp_path):
    store = YoloAnnotationStore(tmp_path / "backups")
    image = make_image(tmp_path)
    document = AnnotationDocument(image, store.resolve_label_path(image), [BoundingBox(0, .5, .5, .2, .2)], dirty=True)
    path = store.save(document)
    assert path.read_text(encoding="utf-8") == "0 0.500000 0.500000 0.200000 0.200000\n"
    document.boxes.clear()
    document.dirty = True
    store.save(document)
    assert path.is_file() and path.read_text(encoding="utf-8") == ""


def test_malformed_repair_requires_confirmation_and_creates_backup(tmp_path):
    store = YoloAnnotationStore(tmp_path / "backups")
    image = make_image(tmp_path)
    label = store.resolve_label_path(image)
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\ngarbage\n", encoding="utf-8")
    document = store.load(image, 1)
    with pytest.raises(ValueError, match="repair_confirmation_required"):
        store.save(document)
    store.save(document, allow_repair=True)
    backups = list((tmp_path / "backups").rglob("*.txt"))
    assert len(backups) == 1 and "garbage" in backups[0].read_text(encoding="utf-8")
