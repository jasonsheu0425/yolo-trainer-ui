"""Stable, UI-language-independent profiles used by Simple Mode."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TrainingProfile:
    model_profile: str
    training_profile: str
    values: dict[str, int | str]


MODEL_PROFILES: dict[str, dict[str, str]] = {
    "fast": {"model": "yolov8n.pt"},
    "balanced": {"model": "yolov8s.pt"},
    "high_accuracy": {"model": "yolov8m.pt"},
}

# These deliberately reuse the values of the established Train page presets.
TRAINING_PROFILES: dict[str, dict[str, int]] = {
    "quick": {"epochs": 1, "imgsz": 640, "batch": 4, "patience": 50},
    "standard": {"epochs": 100, "imgsz": 640, "batch": 16, "patience": 50},
    "extended": {"epochs": 150, "imgsz": 960, "batch": 8, "patience": 50},
}


def build_simple_profile(model_profile: str, training_profile: str, device: str) -> TrainingProfile:
    """Return one immutable set of values for the existing Train page."""
    model = MODEL_PROFILES.get(model_profile, MODEL_PROFILES["balanced"])
    training = TRAINING_PROFILES.get(training_profile, TRAINING_PROFILES["standard"])
    return TrainingProfile(
        model_profile=model_profile if model_profile in MODEL_PROFILES else "balanced",
        training_profile=training_profile if training_profile in TRAINING_PROFILES else "standard",
        values={**model, **training, "device": device or "0"},
    )


def matches_profile(values: dict[str, object], profile: TrainingProfile) -> bool:
    return all(str(values.get(key)) == str(value) for key, value in profile.values.items())
