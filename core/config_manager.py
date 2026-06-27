from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any


DEFAULT_SETTINGS: dict[str, Any] = {
    "python_executable": sys.executable,
    "yolo_command": shutil.which("yolo") or "yolo",
    "runs_folder": "runs/detect",
    "default_model": "yolov8n.pt",
    "default_device": "0",
    "last_run_folder": "",
}


class ConfigManager:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or Path(__file__).resolve().parents[1] / "configs" / "app_settings.json"
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
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
        self.settings = merged

    def get(self, key: str, default: Any = None) -> Any:
        return self.settings.get(key, default)
