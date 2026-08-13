"""Typed training configuration shared by Simple and Advanced workflows."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any


@dataclass(frozen=True)
class TrainingConfig:
    """All user-controlled YOLO train inputs, independent of UI widgets."""

    task: str = "detect"
    model: str = "yolov8n.pt"
    data: str = ""
    imgsz: int = 640
    epochs: int = 100
    batch: int = 16
    device: str = "0"
    workers: int = 8
    project: str = "runs/detect"
    name: str = "train_ui"
    resume: bool = False
    pretrained: bool = True
    cache: bool = False
    patience: int = 50
    advanced: str = ""

    @classmethod
    def from_mapping(cls, values: dict[str, Any], **defaults: Any) -> "TrainingConfig":
        """Create a config while preserving default-compatible omitted fields."""
        fields = {field: values.get(field, defaults.get(field, getattr(cls(), field))) for field in cls.__dataclass_fields__}
        return cls(
            task=str(fields["task"]), model=str(fields["model"]), data=str(fields["data"]),
            imgsz=int(fields["imgsz"]), epochs=int(fields["epochs"]), batch=int(fields["batch"]),
            device=str(fields["device"]), workers=int(fields["workers"]), project=str(fields["project"]),
            name=str(fields["name"]), resume=bool(fields["resume"]), pretrained=bool(fields["pretrained"]),
            cache=bool(fields["cache"]), patience=int(fields["patience"]), advanced=str(fields["advanced"]),
        )

    def with_updates(self, **values: Any) -> "TrainingConfig":
        """Return a new immutable config; callers never mutate widget state as data."""
        return replace(self, **values)
