"""Annotation workflow coordinator independent of concrete Qt widgets."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from core.dataset_checker import load_dataset_manifest
from domain.annotation import AnnotationDataset, AnnotationDocument, AnnotationStatus, BoundingBox, PixelBox, xyxy_to_yolo, yolo_to_xyxy
from persistence.yolo_annotation_store import YoloAnnotationStore


@dataclass(frozen=True)
class AnnotationSnapshot:
    boxes: tuple[BoundingBox, ...]
    selected_index: int | None


class AnnotationService:
    """Owns dataset/session state, mutations, history, save, and navigation."""

    def __init__(self, store: YoloAnnotationStore | None = None) -> None:
        self.store = store or YoloAnnotationStore()
        self.dataset: AnnotationDataset | None = None
        self.split = ""
        self.images: tuple[Path, ...] = ()
        self.index = -1
        self.document: AnnotationDocument | None = None
        self.selected_index: int | None = None
        self._undo: list[AnnotationSnapshot] = []
        self._redo: list[AnnotationSnapshot] = []
        self._saved_snapshot: AnnotationSnapshot | None = None

    def open_dataset(self, yaml_path: str | Path, preferred_split: str = "train") -> AnnotationDataset:
        payload, errors = load_dataset_manifest(yaml_path)
        if errors:
            raise ValueError("; ".join(errors))
        collisions = self._stem_collisions(payload["splits"])
        if collisions:
            names = ", ".join(sorted(collisions)[:5])
            raise ValueError(f"duplicate_label_path:{names}")
        self.dataset = AnnotationDataset(
            payload["yaml"], payload["root"], payload["names"],
            {key: tuple(value) for key, value in payload["splits"].items()},
        )
        split = preferred_split if preferred_split in self.dataset.splits else next(iter(self.dataset.splits))
        self.select_split(split)
        return self.dataset

    def select_split(self, split: str) -> None:
        if self.dataset is None or split not in self.dataset.splits:
            raise ValueError("invalid_split")
        self.split = split
        self.images = self.dataset.splits[split]
        self.index = -1
        self.document = None
        if self.images:
            self.load_image(0)

    def load_image(self, index: int) -> AnnotationDocument:
        if self.dataset is None or not 0 <= index < len(self.images):
            raise IndexError("image_index_out_of_range")
        self.index = index
        self.document = self.store.load(self.images[index], len(self.dataset.classes))
        self.selected_index = None
        self._undo.clear()
        self._redo.clear()
        self._saved_snapshot = self._snapshot()
        return self.document

    def next_image(self, autosave: bool = True) -> AnnotationDocument | None:
        if self.document and self.document.dirty:
            if not autosave:
                return None
            self.save()
        return self.load_image(self.index + 1) if self.index + 1 < len(self.images) else None

    def previous_image(self, autosave: bool = True) -> AnnotationDocument | None:
        if self.document and self.document.dirty:
            if not autosave:
                return None
            self.save()
        return self.load_image(self.index - 1) if self.index > 0 else None

    def create_box(self, box: BoundingBox) -> int:
        self._require_document()
        self._validate_box(box)
        self._record()
        self.document.boxes.append(box)  # type: ignore[union-attr]
        self.selected_index = len(self.document.boxes) - 1  # type: ignore[union-attr]
        self._mark_dirty()
        return self.selected_index

    def create_pixel_box(self, pixel: PixelBox, width: int, height: int, class_id: int) -> int:
        return self.create_box(xyxy_to_yolo(pixel, width, height, class_id))

    def replace_box(self, index: int, box: BoundingBox) -> None:
        self._require_box(index)
        self._validate_box(box)
        if self.document.boxes[index] == box:  # type: ignore[union-attr]
            return
        self._record()
        self.document.boxes[index] = box  # type: ignore[union-attr]
        self.selected_index = index
        self._mark_dirty()

    def move_box_pixels(self, index: int, dx: float, dy: float, width: int, height: int) -> None:
        self._require_box(index)
        current = yolo_to_xyxy(self.document.boxes[index], width, height)  # type: ignore[union-attr]
        dx = min(max(dx, -current.x1), width - current.x2)
        dy = min(max(dy, -current.y1), height - current.y2)
        self.replace_box(index, xyxy_to_yolo(PixelBox(current.x1 + dx, current.y1 + dy, current.x2 + dx, current.y2 + dy), width, height, current_class(self.document, index)))

    def delete_box(self, index: int) -> None:
        self._require_box(index)
        self._record()
        del self.document.boxes[index]  # type: ignore[union-attr]
        self.selected_index = None
        self._mark_dirty()

    def change_class(self, index: int, class_id: int) -> None:
        self._require_box(index)
        self.replace_box(index, self.document.boxes[index].with_class(class_id))  # type: ignore[union-attr]

    def paste_box(self, box: BoundingBox, image_width: int, image_height: int, offset: int = 10) -> int:
        self._validate_box(box)
        pixel = yolo_to_xyxy(box, image_width, image_height)
        dx = min(float(offset), image_width - pixel.x2)
        dy = min(float(offset), image_height - pixel.y2)
        return self.create_pixel_box(PixelBox(pixel.x1 + dx, pixel.y1 + dy, pixel.x2 + dx, pixel.y2 + dy), image_width, image_height, box.class_id)

    def save(self, *, repair: bool = False) -> Path:
        self._require_document()
        path = self.store.save(self.document, allow_repair=repair)  # type: ignore[arg-type]
        self._saved_snapshot = self._snapshot()
        return path

    def discard(self) -> AnnotationDocument:
        self._require_document()
        return self.load_image(self.index)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    def image_status(self, image: Path) -> AnnotationStatus:
        if self.document and image == self.document.image_path and self.document.dirty:
            return AnnotationStatus.MODIFIED
        return self.store.load(image, len(self.dataset.classes) if self.dataset else 0).status

    def status_summary(self) -> dict[str, int]:
        counts = Counter(self.image_status(image).value for image in self.images)
        return {status.value: counts[status.value] for status in AnnotationStatus}

    @staticmethod
    def _stem_collisions(splits: dict[str, list[Path]]) -> set[str]:
        destinations: dict[Path, list[Path]] = {}
        for images in splits.values():
            for image in images:
                destinations.setdefault(YoloAnnotationStore.resolve_label_path(image), []).append(image)
        return {str(path) for path, sources in destinations.items() if len(set(sources)) > 1}

    def _record(self) -> None:
        self._undo.append(self._snapshot())
        self._redo.clear()

    def _snapshot(self) -> AnnotationSnapshot:
        boxes = tuple(deepcopy(self.document.boxes)) if self.document else ()
        return AnnotationSnapshot(boxes, self.selected_index)

    def _restore(self, snapshot: AnnotationSnapshot) -> None:
        self._require_document()
        self.document.boxes = list(snapshot.boxes)  # type: ignore[union-attr]
        self.selected_index = snapshot.selected_index
        self.document.dirty = snapshot != self._saved_snapshot  # type: ignore[union-attr]

    def _mark_dirty(self) -> None:
        self.document.dirty = self._snapshot() != self._saved_snapshot  # type: ignore[union-attr]

    def _require_document(self) -> None:
        if self.document is None:
            raise ValueError("no_annotation_document")

    def _require_box(self, index: int) -> None:
        self._require_document()
        if not 0 <= index < len(self.document.boxes):  # type: ignore[union-attr]
            raise IndexError("box_index_out_of_range")

    def _validate_box(self, box: BoundingBox) -> None:
        count = len(self.dataset.classes) if self.dataset else None
        if not box.is_valid(count):
            raise ValueError("invalid_bounding_box")


def current_class(document: AnnotationDocument | None, index: int) -> int:
    if document is None:
        raise ValueError("no_annotation_document")
    return document.boxes[index].class_id
