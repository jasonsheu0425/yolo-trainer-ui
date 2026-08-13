"""UI-independent models and geometry for YOLO detection annotations."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from pathlib import Path


class AnnotationStatus(str, Enum):
    UNLABELED = "unlabeled"
    LABELED = "labeled"
    EMPTY = "empty"
    INVALID = "invalid"
    MODIFIED = "modified"


class AnnotationSource(str, Enum):
    MANUAL = "manual"
    MODEL_GENERATED = "model_generated"
    MODEL_ASSISTED = "model_assisted"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class BoundingBox:
    """Canonical YOLO-normalized detection box."""

    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def is_valid(self, class_count: int | None = None) -> bool:
        values = (self.x_center, self.y_center, self.width, self.height)
        return (
            self.class_id >= 0
            and (class_count is None or self.class_id < class_count)
            and all(math.isfinite(value) for value in values)
            and 0.0 <= self.x_center <= 1.0
            and 0.0 <= self.y_center <= 1.0
            and 0.0 < self.width <= 1.0
            and 0.0 < self.height <= 1.0
            and self.x_center - self.width / 2 >= -1e-9
            and self.x_center + self.width / 2 <= 1.0 + 1e-9
            and self.y_center - self.height / 2 >= -1e-9
            and self.y_center + self.height / 2 <= 1.0 + 1e-9
        )

    def with_class(self, class_id: int) -> "BoundingBox":
        return replace(self, class_id=class_id)


@dataclass(frozen=True)
class ModelPrediction:
    """Validated model proposal kept separate from canonical label geometry."""

    box: BoundingBox
    confidence: float
    model_identity: str = ""
    generated_at: str = ""

    def is_valid(self, class_count: int | None = None) -> bool:
        return (
            self.box.is_valid(class_count)
            and math.isfinite(self.confidence)
            and 0.0 <= self.confidence <= 1.0
        )


@dataclass(frozen=True)
class BoxMetadata:
    source: AnnotationSource = AnnotationSource.UNKNOWN
    confidence: float | None = None
    model_identity: str = ""
    generated_at: str = ""


@dataclass(frozen=True)
class PixelBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


def yolo_to_xyxy(box: BoundingBox, image_width: int, image_height: int) -> PixelBox:
    """Convert normalized YOLO coordinates to clamped image-pixel geometry."""
    _validate_image_size(image_width, image_height)
    half_width = box.width * image_width / 2
    half_height = box.height * image_height / 2
    center_x, center_y = box.x_center * image_width, box.y_center * image_height
    return clamp_xyxy(
        PixelBox(center_x - half_width, center_y - half_height, center_x + half_width, center_y + half_height),
        image_width,
        image_height,
    )


def xyxy_to_yolo(pixel: PixelBox, image_width: int, image_height: int, class_id: int) -> BoundingBox:
    """Convert clamped image-pixel geometry to normalized YOLO coordinates."""
    _validate_image_size(image_width, image_height)
    value = clamp_xyxy(pixel, image_width, image_height)
    if value.width <= 0 or value.height <= 0:
        raise ValueError("bounding_box_has_no_area")
    return BoundingBox(
        class_id=class_id,
        x_center=((value.x1 + value.x2) / 2) / image_width,
        y_center=((value.y1 + value.y2) / 2) / image_height,
        width=value.width / image_width,
        height=value.height / image_height,
    )


def clamp_xyxy(pixel: PixelBox, image_width: int, image_height: int) -> PixelBox:
    _validate_image_size(image_width, image_height)
    x1 = min(max(min(pixel.x1, pixel.x2), 0.0), float(image_width))
    y1 = min(max(min(pixel.y1, pixel.y2), 0.0), float(image_height))
    x2 = min(max(max(pixel.x1, pixel.x2), 0.0), float(image_width))
    y2 = min(max(max(pixel.y1, pixel.y2), 0.0), float(image_height))
    return PixelBox(x1, y1, x2, y2)


@dataclass(frozen=True)
class InvalidLabelLine:
    line_number: int
    raw_text: str
    error_id: str


@dataclass
class AnnotationDocument:
    image_path: Path
    label_path: Path
    boxes: list[BoundingBox] = field(default_factory=list)
    invalid_lines: list[InvalidLabelLine] = field(default_factory=list)
    label_existed: bool = False
    dirty: bool = False
    box_metadata: list[BoxMetadata] = field(default_factory=list)
    source: AnnotationSource = AnnotationSource.UNKNOWN
    last_prediction_status: str = ""
    metadata_warning: str = ""

    @property
    def status(self) -> AnnotationStatus:
        if self.dirty:
            return AnnotationStatus.MODIFIED
        if self.invalid_lines:
            return AnnotationStatus.INVALID
        if not self.label_existed:
            return AnnotationStatus.UNLABELED
        return AnnotationStatus.LABELED if self.boxes else AnnotationStatus.EMPTY


@dataclass(frozen=True)
class AnnotationDataset:
    yaml_path: Path
    root: Path
    classes: dict[int, str]
    splits: dict[str, tuple[Path, ...]]


def _validate_image_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("invalid_image_dimensions")
