from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from core.dataset_checker import check_dataset
from domain.annotation import (
    AnnotationSource,
    AnnotationStatus,
    BoundingBox,
    ModelPrediction,
    PixelBox,
)
from persistence.yolo_annotation_store import YoloAnnotationStore
from services.annotation_service import AnnotationService


def make_dataset(tmp_path: Path, count: int = 2) -> Path:
    for split in ("train", "val"):
        folder = tmp_path / "images" / split
        folder.mkdir(parents=True)
        for index in range(count):
            Image.new("RGB", (640, 480), "white").save(folder / f"{index}.jpg")
    yaml = tmp_path / "data.yaml"
    yaml.write_text("path: .\ntrain: images/train\nval: images/val\nnames: [person, vehicle]\n", encoding="utf-8")
    return yaml


def test_open_dataset_classes_splits_and_mutation_history(tmp_path):
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    dataset = service.open_dataset(make_dataset(tmp_path))
    assert dataset.classes == {0: "person", 1: "vehicle"}
    assert set(dataset.splits) == {"train", "val"}
    index = service.create_pixel_box(PixelBox(100, 100, 300, 300), 640, 480, 0)
    original = service.document.boxes[index]
    service.move_box_pixels(index, 10, 20, 640, 480)
    service.change_class(index, 1)
    assert service.document.dirty and service.document.boxes[index].class_id == 1
    assert service.undo() and service.document.boxes[index].class_id == 0
    assert service.undo() and service.document.boxes[index] == original
    assert service.redo()
    service.delete_box(index)
    assert service.undo() and len(service.document.boxes) == 1


def test_untouched_unlabeled_navigation_creates_no_empty_label(tmp_path):
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    service.open_dataset(make_dataset(tmp_path))
    label = service.document.label_path
    service.next_image(autosave=True)
    assert not label.exists()


def test_save_empty_negative_status_and_dataset_check_compatibility(tmp_path):
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    yaml = make_dataset(tmp_path)
    service.open_dataset(yaml)
    service.create_box(BoundingBox(0, .5, .5, .2, .2))
    service.save()
    service.delete_box(0)
    service.save()
    assert service.document.status is AnnotationStatus.EMPTY
    # Label remaining images so Dataset Check's missing-label condition is avoided.
    for image in (*service.dataset.splits["train"], *service.dataset.splits["val"]):
        label = service.store.resolve_label_path(image)
        label.parent.mkdir(parents=True, exist_ok=True)
        if not label.exists():
            label.write_text("", encoding="utf-8")
    assert check_dataset(yaml)["errors"] == []


def test_split_change_and_autosave_off_preserve_dirty_document(tmp_path):
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    service.open_dataset(make_dataset(tmp_path))
    service.create_box(BoundingBox(0, .5, .5, .2, .2))
    assert service.next_image(autosave=False) is None and service.index == 0
    service.discard()
    service.select_split("val")
    assert service.split == "val" and service.index == 0


def test_duplicate_image_stem_collision_is_rejected(tmp_path):
    folder = tmp_path / "images/train"
    folder.mkdir(parents=True)
    Image.new("RGB", (10, 10)).save(folder / "foo.jpg")
    Image.new("RGB", (10, 10)).save(folder / "foo.png")
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        "path: .\ntrain: images/train\nnames: [thing]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate_label_path"):
        AnnotationService().open_dataset(yaml)


def test_malformed_label_requires_repair_and_preserves_original(tmp_path):
    yaml = make_dataset(tmp_path)
    label = tmp_path / "labels" / "train" / "0.txt"
    label.parent.mkdir(parents=True)
    original = "0 0.5 0.5 0.2 0.2\ngarbage\n"
    label.write_text(original, encoding="utf-8")
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    service.open_dataset(yaml)
    assert len(service.document.boxes) == 1
    assert len(service.document.invalid_lines) == 1
    with pytest.raises(ValueError, match="repair_confirmation_required"):
        service.save()
    assert label.read_text(encoding="utf-8") == original
    service.save(repair=True)
    assert "garbage" not in label.read_text(encoding="utf-8")
    assert len(list((tmp_path / "backups").rglob("*.txt"))) == 1


def test_save_permission_failure_is_propagated_without_false_success(
    tmp_path, monkeypatch
):
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    service.open_dataset(make_dataset(tmp_path))
    service.create_box(BoundingBox(0, .5, .5, .2, .2))

    def deny_write(*_args, **_kwargs):
        raise PermissionError("read-only target")

    monkeypatch.setattr(
        "persistence.yolo_annotation_store.atomic_write_text", deny_write
    )
    with pytest.raises(PermissionError, match="read-only"):
        service.save()
    assert service.document.dirty
    assert not service.document.label_path.exists()


def test_large_manifest_discovers_files_without_decoding_images(tmp_path):
    folder = tmp_path / "images" / "train"
    folder.mkdir(parents=True)
    for index in range(5000):
        (folder / f"{index:05}.jpg").write_bytes(b"")
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        "path: .\ntrain: images/train\nnames: [object]\n", encoding="utf-8"
    )
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    service.open_dataset(yaml)
    assert len(service.images) == 5000
    assert service.document.image_path.name == "00000.jpg"


def test_undo_redo_for_create_resize_delete_class_and_paste(tmp_path):
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    service.open_dataset(make_dataset(tmp_path))
    first = service.create_box(BoundingBox(0, .5, .5, .2, .2))
    assert service.undo() and service.document.boxes == []
    assert service.redo() and len(service.document.boxes) == 1

    pasted = service.paste_box(service.document.boxes[first], 640, 480)
    assert pasted == 1 and len(service.document.boxes) == 2
    assert service.undo() and len(service.document.boxes) == 1

    resized = BoundingBox(0, .5, .5, .3, .3)
    service.replace_box(0, resized)
    assert service.undo() and service.document.boxes[0].width == .2
    assert service.redo() and service.document.boxes[0] == resized

    service.change_class(0, 1)
    assert service.undo() and service.document.boxes[0].class_id == 0
    service.delete_box(0)
    assert service.undo() and len(service.document.boxes) == 1


def test_manual_generated_assisted_unknown_transitions_and_prediction_batch_undo(
    tmp_path,
):
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    service.open_dataset(make_dataset(tmp_path))
    assert service.document.source is AnnotationSource.UNKNOWN
    service.create_box(BoundingBox(0, .5, .5, .2, .2))
    assert service.document.source is AnnotationSource.MANUAL
    service.save()
    service.load_image(1)
    predictions = [
        ModelPrediction(BoundingBox(0, .5, .5, .2, .2), .9, "best.pt")
    ]
    service.apply_predictions(predictions)
    assert service.document.source is AnnotationSource.MODEL_GENERATED
    assert service.document.box_metadata[0].confidence == .9
    assert service.undo() and service.document.boxes == []
    assert service.redo() and len(service.document.boxes) == 1
    service.move_box_pixels(0, 10, 10, 640, 480)
    assert service.document.source is AnnotationSource.MODEL_ASSISTED
    assert service.document.box_metadata[0].confidence is None
    service.save()
    service.load_image(1)
    assert service.document.source is AnnotationSource.MODEL_ASSISTED


def test_replace_predictions_is_one_undo_and_existing_unknown_becomes_assisted(
    tmp_path,
):
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    yaml = make_dataset(tmp_path)
    label = tmp_path / "labels" / "train" / "0.txt"
    label.parent.mkdir(parents=True)
    label.write_text("0 0.5 0.5 0.2 0.2\n", encoding="utf-8")
    service.open_dataset(yaml)
    original = service.document.boxes.copy()
    prediction = ModelPrediction(BoundingBox(1, .4, .4, .1, .1), .8, "best.pt")
    service.apply_predictions([prediction], replace=True)
    assert service.document.source is AnnotationSource.MODEL_ASSISTED
    assert service.document.boxes == [prediction.box]
    assert service.undo() and service.document.boxes == original


def test_metadata_failure_does_not_undo_atomic_label_save(tmp_path, monkeypatch):
    service = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    service.open_dataset(make_dataset(tmp_path))

    def deny_metadata(*_args, **_kwargs):
        raise PermissionError("metadata is read-only")

    monkeypatch.setattr(service.metadata_store, "save_image", deny_metadata)
    prediction = ModelPrediction(BoundingBox(0, .5, .5, .2, .2), .9, "best.pt")
    label = service.save_generated_predictions(
        service.document.image_path, [prediction]
    )
    assert label is not None and label.is_file()
    assert "metadata is read-only" in service.last_metadata_warning
