from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from persistence.atomic_writer import atomic_write_json


DEFAULT_SETTINGS: dict[str, Any] = {
    "language": "zh_TW",
    "ui_mode": "advanced",
    "python_executable": "",
    "yolo_command": "",
    "runs_folder": "runs/detect",
    "default_model": "yolov8n.pt",
    "default_device": "0",
    "last_run_folder": "",
    "last_annotation_dataset": "",
    "annotation_autosave": True,
    "last_annotation_model": "",
    "annotation_confidence": 0.25,
    "annotation_device": "auto",
}


class ConfigManager:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or _default_settings_path()
        self.settings = self.load()

    def load(self) -> dict[str, Any]:
        values = DEFAULT_SETTINGS.copy()
        try:
            if self.path.exists():
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    values.update(loaded)
        except (OSError, json.JSONDecodeError):
            pass
        self.settings = values
        return values.copy()

    def save(self, values: dict[str, Any]) -> None:
        merged = DEFAULT_SETTINGS.copy()
        merged.update(self.settings)
        merged.update(values)
        atomic_write_json(self.path, merged)
        self.settings = merged

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)


def _default_settings_path() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent / "configs" / "app_settings.json"
    return Path(__file__).resolve().parents[1] / "configs" / "app_settings.json"
