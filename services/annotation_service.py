"""Annotation workflow coordinator independent of concrete Qt widgets."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.dataset_checker import load_dataset_manifest
from domain.annotation import (
    AnnotationDataset,
    AnnotationDocument,
    AnnotationSource,
    AnnotationStatus,
    BoundingBox,
    BoxMetadata,
    ModelPrediction,
    PixelBox,
    xyxy_to_yolo,
    yolo_to_xyxy,
)
from persistence.annotation_metadata_store import AnnotationMetadataStore, utc_now
from persistence.yolo_annotation_store import YoloAnnotationStore


@dataclass(frozen=True)
class AnnotationSnapshot:
    boxes: tuple[BoundingBox, ...]
    box_metadata: tuple[BoxMetadata, ...]
    source: AnnotationSource
    last_prediction_status: str
    selected_index: int | None


class AnnotationService:
    """Own dataset/session state, mutations, history, provenance, and save."""

    def __init__(
        self,
        store: YoloAnnotationStore | None = None,
        metadata_store: AnnotationMetadataStore | None = None,
    ) -> None:
        self.store = store or YoloAnnotationStore()
        if metadata_store is not None:
            self.metadata_store = metadata_store
        elif store is not None:
            self.metadata_store = AnnotationMetadataStore(
                self.store.backup_root.parent / "annotation_metadata"
            )
        else:
            self.metadata_store = AnnotationMetadataStore()
        self.dataset: AnnotationDataset | None = None
        self.split = ""
        self.images: tuple[Path, ...] = ()
        self.index = -1
        self.document: AnnotationDocument | None = None
        self.selected_index: int | None = None
        self.last_metadata_warning = ""
        self._undo: list[AnnotationSnapshot] = []
        self._redo: list[AnnotationSnapshot] = []
        self._saved_snapshot: AnnotationSnapshot | None = None

    def open_dataset(
        self, yaml_path: str | Path, preferred_split: str = "train"
    ) -> AnnotationDataset:
        payload, errors = load_dataset_manifest(yaml_path)
        if errors:
            raise ValueError("; ".join(errors))
        collisions = self._stem_collisions(payload["splits"])
        if collisions:
            names = ", ".join(sorted(collisions)[:5])
            raise ValueError(f"duplicate_label_path:{names}")
        self.dataset = AnnotationDataset(
            payload["yaml"],
            payload["root"],
            payload["names"],
            {key: tuple(value) for key, value in payload["splits"].items()},
        )
        split = (
            preferred_split
            if preferred_split in self.dataset.splits
            else next(iter(self.dataset.splits))
        )
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
        self.document = self.store.load(
            self.images[index], len(self.dataset.classes)
        )
        self._load_provenance(self.document)
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
        return (
            self.load_image(self.index + 1)
            if self.index + 1 < len(self.images)
            else None
        )

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
        assisted = self._has_model_content()
        self.document.boxes.append(box)  # type: ignore[union-attr]
        self.document.box_metadata.append(  # type: ignore[union-attr]
            BoxMetadata(source=AnnotationSource.MANUAL)
        )
        self.document.source = (  # type: ignore[union-attr]
            AnnotationSource.MODEL_ASSISTED if assisted else AnnotationSource.MANUAL
        )
        self.selected_index = len(self.document.boxes) - 1  # type: ignore[union-attr]
        self._mark_dirty()
        return self.selected_index

    def create_pixel_box(
        self, pixel: PixelBox, width: int, height: int, class_id: int
    ) -> int:
        return self.create_box(xyxy_to_yolo(pixel, width, height, class_id))

    def replace_box(self, index: int, box: BoundingBox) -> None:
        self._require_box(index)
        self._validate_box(box)
        if self.document.boxes[index] == box:  # type: ignore[union-attr]
            return
        self._record()
        self.document.boxes[index] = box  # type: ignore[union-attr]
        self._mark_human_edit(index)
        self.selected_index = index
        self._mark_dirty()

    def move_box_pixels(
        self, index: int, dx: float, dy: float, width: int, height: int
    ) -> None:
        self._require_box(index)
        current = yolo_to_xyxy(
            self.document.boxes[index], width, height  # type: ignore[union-attr]
        )
        dx = min(max(dx, -current.x1), width - current.x2)
        dy = min(max(dy, -current.y1), height - current.y2)
        pixel = PixelBox(
            current.x1 + dx,
            current.y1 + dy,
            current.x2 + dx,
            current.y2 + dy,
        )
        self.replace_box(
            index,
            xyxy_to_yolo(pixel, width, height, current_class(self.document, index)),
        )

    def delete_box(self, index: int) -> None:
        self._require_box(index)
        self._record()
        was_model = self.document.box_metadata[index].source is AnnotationSource.MODEL_GENERATED  # type: ignore[union-attr]
        del self.document.boxes[index]  # type: ignore[union-attr]
        del self.document.box_metadata[index]  # type: ignore[union-attr]
        if was_model or self._has_model_content():
            self.document.source = AnnotationSource.MODEL_ASSISTED  # type: ignore[union-attr]
        elif self.document.source is not AnnotationSource.UNKNOWN:  # type: ignore[union-attr]
            self.document.source = AnnotationSource.MANUAL  # type: ignore[union-attr]
        self.selected_index = None
        self._mark_dirty()

    def change_class(self, index: int, class_id: int) -> None:
        self._require_box(index)
        self.replace_box(
            index,
            self.document.boxes[index].with_class(class_id),  # type: ignore[union-attr]
        )

    def paste_box(
        self,
        box: BoundingBox,
        image_width: int,
        image_height: int,
        offset: int = 10,
    ) -> int:
        self._validate_box(box)
        pixel = yolo_to_xyxy(box, image_width, image_height)
        dx = min(float(offset), image_width - pixel.x2)
        dy = min(float(offset), image_height - pixel.y2)
        return self.create_pixel_box(
            PixelBox(
                pixel.x1 + dx,
                pixel.y1 + dy,
                pixel.x2 + dx,
                pixel.y2 + dy,
            ),
            image_width,
            image_height,
            box.class_id,
        )

    def apply_predictions(
        self,
        predictions: list[ModelPrediction],
        *,
        replace: bool = False,
    ) -> int:
        self._require_document()
        for prediction in predictions:
            if not prediction.is_valid(
                len(self.dataset.classes) if self.dataset else None
            ):
                raise ValueError("invalid_model_prediction")
        self.document.last_prediction_status = (  # type: ignore[union-attr]
            "detections" if predictions else "no_detection"
        )
        if not predictions:
            return 0
        self._record()
        had_boxes = bool(self.document.boxes)  # type: ignore[union-attr]
        if replace:
            self.document.boxes.clear()  # type: ignore[union-attr]
            self.document.box_metadata.clear()  # type: ignore[union-attr]
        self.document.boxes.extend(prediction.box for prediction in predictions)  # type: ignore[union-attr]
        self.document.box_metadata.extend(  # type: ignore[union-attr]
            BoxMetadata(
                source=AnnotationSource.MODEL_GENERATED,
                confidence=prediction.confidence,
                model_identity=prediction.model_identity,
                generated_at=prediction.generated_at,
            )
            for prediction in predictions
        )
        self.document.source = (  # type: ignore[union-attr]
            AnnotationSource.MODEL_ASSISTED
            if had_boxes
            else AnnotationSource.MODEL_GENERATED
        )
        self.selected_index = len(self.document.boxes) - 1  # type: ignore[union-attr]
        self._mark_dirty()
        return len(predictions)

    def save(self, *, repair: bool = False) -> Path:
        self._require_document()
        path = self.store.save(
            self.document, allow_repair=repair  # type: ignore[arg-type]
        )
        self.document.metadata_warning = ""  # type: ignore[union-attr]
        self.last_metadata_warning = ""
        try:
            self._save_provenance(self.document)  # type: ignore[arg-type]
        except (OSError, ValueError, TypeError) as exc:
            self.document.metadata_warning = str(exc)  # type: ignore[union-attr]
            self.last_metadata_warning = str(exc)
        self._saved_snapshot = self._snapshot()
        return path

    def save_generated_predictions(
        self,
        image: Path,
        predictions: list[ModelPrediction],
    ) -> Path | None:
        if self.dataset is None:
            raise ValueError("no_annotation_dataset")
        document = self.store.load(image, len(self.dataset.classes))
        if document.label_existed or document.invalid_lines:
            raise ValueError("existing_label_protected")
        if not predictions:
            self.record_prediction_status(image, "no_detection")
            return None
        for prediction in predictions:
            if not prediction.is_valid(len(self.dataset.classes)):
                raise ValueError("invalid_model_prediction")
        document.boxes = [prediction.box for prediction in predictions]
        document.box_metadata = [
            BoxMetadata(
                source=AnnotationSource.MODEL_GENERATED,
                confidence=prediction.confidence,
                model_identity=prediction.model_identity,
                generated_at=prediction.generated_at,
            )
            for prediction in predictions
        ]
        document.source = AnnotationSource.MODEL_GENERATED
        document.last_prediction_status = "detections"
        document.dirty = True
        path = self.store.save(document)
        self.last_metadata_warning = ""
        try:
            self._save_provenance(document)
        except (OSError, ValueError, TypeError) as exc:
            self.last_metadata_warning = str(exc)
        return path

    def record_prediction_status(self, image: Path, status: str) -> None:
        if self.dataset is None:
            raise ValueError("no_annotation_dataset")
        existing = self.metadata_store.get_image(self.dataset, image)
        existing["last_prediction_status"] = status
        existing["updated_at"] = utc_now()
        self.metadata_store.save_image(self.dataset, image, existing)

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
        if (
            self.document
            and image == self.document.image_path
            and self.document.dirty
        ):
            return AnnotationStatus.MODIFIED
        return self.store.load(
            image, len(self.dataset.classes) if self.dataset else 0
        ).status

    def image_source(self, image: Path) -> AnnotationSource:
        if self.dataset is None:
            return AnnotationSource.UNKNOWN
        record = self.metadata_store.get_image(self.dataset, image)
        try:
            return AnnotationSource(str(record.get("source", "unknown")))
        except ValueError:
            return AnnotationSource.UNKNOWN

    def status_summary(self) -> dict[str, int]:
        counts = Counter(self.image_status(image).value for image in self.images)
        return {status.value: counts[status.value] for status in AnnotationStatus}

    @staticmethod
    def _stem_collisions(splits: dict[str, list[Path]]) -> set[str]:
        destinations: dict[Path, list[Path]] = {}
        for images in splits.values():
            for image in images:
                destinations.setdefault(
                    YoloAnnotationStore.resolve_label_path(image), []
                ).append(image)
        return {
            str(path)
            for path, sources in destinations.items()
            if len(set(sources)) > 1
        }

    def _record(self) -> None:
        self._undo.append(self._snapshot())
        self._redo.clear()

    def _snapshot(self) -> AnnotationSnapshot:
        if self.document is None:
            return AnnotationSnapshot((), (), AnnotationSource.UNKNOWN, "", None)
        return AnnotationSnapshot(
            tuple(deepcopy(self.document.boxes)),
            tuple(deepcopy(self.document.box_metadata)),
            self.document.source,
            self.document.last_prediction_status,
            self.selected_index,
        )

    def _restore(self, snapshot: AnnotationSnapshot) -> None:
        self._require_document()
        self.document.boxes = list(snapshot.boxes)  # type: ignore[union-attr]
        self.document.box_metadata = list(snapshot.box_metadata)  # type: ignore[union-attr]
        self.document.source = snapshot.source  # type: ignore[union-attr]
        self.document.last_prediction_status = snapshot.last_prediction_status  # type: ignore[union-attr]
        self.selected_index = snapshot.selected_index
        self.document.dirty = snapshot != self._saved_snapshot  # type: ignore[union-attr]

    def _mark_dirty(self) -> None:
        self.document.dirty = self._snapshot() != self._saved_snapshot  # type: ignore[union-attr]

    def _mark_human_edit(self, index: int) -> None:
        metadata = self.document.box_metadata[index]  # type: ignore[union-attr]
        was_model = metadata.source is AnnotationSource.MODEL_GENERATED
        self.document.box_metadata[index] = BoxMetadata(  # type: ignore[union-attr]
            source=(
                AnnotationSource.MODEL_ASSISTED
                if was_model
                else AnnotationSource.MANUAL
            ),
            confidence=None,
            model_identity=metadata.model_identity,
            generated_at=metadata.generated_at,
        )
        if was_model or self._has_model_content():
            self.document.source = AnnotationSource.MODEL_ASSISTED  # type: ignore[union-attr]
        else:
            self.document.source = AnnotationSource.MANUAL  # type: ignore[union-attr]

    def _has_model_content(self) -> bool:
        if self.document is None:
            return False
        return self.document.source in {
            AnnotationSource.MODEL_GENERATED,
            AnnotationSource.MODEL_ASSISTED,
        } or any(
            item.source
            in {AnnotationSource.MODEL_GENERATED, AnnotationSource.MODEL_ASSISTED}
            for item in self.document.box_metadata
        )

    def _load_provenance(self, document: AnnotationDocument) -> None:
        document.box_metadata = [BoxMetadata() for _ in document.boxes]
        if self.dataset is None:
            return
        record = self.metadata_store.get_image(self.dataset, document.image_path)
        try:
            document.source = AnnotationSource(str(record.get("source", "unknown")))
        except ValueError:
            document.source = AnnotationSource.UNKNOWN
        document.last_prediction_status = str(
            record.get("last_prediction_status", "")
        )
        saved_boxes = record.get("boxes", [])
        if not isinstance(saved_boxes, list) or len(saved_boxes) != len(document.boxes):
            return
        restored: list[BoxMetadata] = []
        for box, value in zip(document.boxes, saved_boxes):
            if not isinstance(value, dict) or value.get("signature") != self.store.format_box(box):
                return
            try:
                source = AnnotationSource(str(value.get("source", "unknown")))
            except ValueError:
                source = AnnotationSource.UNKNOWN
            confidence = value.get("confidence")
            restored.append(
                BoxMetadata(
                    source=source,
                    confidence=(
                        float(confidence)
                        if isinstance(confidence, (int, float))
                        else None
                    ),
                    model_identity=str(value.get("model_identity", "")),
                    generated_at=str(value.get("generated_at", "")),
                )
            )
        document.box_metadata = restored

    def _save_provenance(self, document: AnnotationDocument) -> None:
        if self.dataset is None:
            return
        record: dict[str, Any] = {
            "source": document.source.value,
            "last_prediction_status": document.last_prediction_status,
            "updated_at": utc_now(),
            "boxes": [
                {
                    "signature": self.store.format_box(box),
                    "source": metadata.source.value,
                    "confidence": metadata.confidence,
                    "model_identity": metadata.model_identity,
                    "generated_at": metadata.generated_at,
                }
                for box, metadata in zip(document.boxes, document.box_metadata)
            ],
        }
        self.metadata_store.save_image(
            self.dataset, document.image_path, record
        )

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
