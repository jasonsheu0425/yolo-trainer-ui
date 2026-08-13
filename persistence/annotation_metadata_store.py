"""App-managed provenance metadata and auto-annotation audit reports."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from domain.annotation import AnnotationDataset
from persistence.atomic_writer import atomic_write_json


METADATA_SCHEMA_VERSION = 1
REPORT_SCHEMA_VERSION = 1


def _local_app_root() -> Path:
    local = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local"))
    return local / "YOLO-Trainer-UI"


def dataset_identity(yaml_path: str | Path, root: str | Path) -> str:
    canonical = f"{Path(yaml_path).resolve()}\n{Path(root).resolve()}".casefold()
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AnnotationMetadataStore:
    def __init__(self, base_root: Path | None = None) -> None:
        self.base_root = base_root or _local_app_root() / "annotation_metadata"

    def path_for(self, dataset: AnnotationDataset) -> Path:
        return self.base_root / f"{dataset_identity(dataset.yaml_path, dataset.root)}.json"

    def image_key(self, dataset: AnnotationDataset, image_path: Path) -> str:
        try:
            return image_path.resolve().relative_to(dataset.root.resolve()).as_posix()
        except ValueError:
            return str(image_path.resolve())

    def read(self, dataset: AnnotationDataset) -> dict[str, Any]:
        path = self.path_for(dataset)
        if not path.is_file():
            return self._empty(dataset)
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict) or value.get("schema_version") != METADATA_SCHEMA_VERSION:
                raise ValueError("unsupported_metadata_schema")
            if not isinstance(value.get("images"), dict):
                raise ValueError("invalid_metadata_images")
            return value
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
            self._quarantine(path)
            return self._empty(dataset)

    def get_image(self, dataset: AnnotationDataset, image_path: Path) -> dict[str, Any]:
        record = self.read(dataset).get("images", {}).get(self.image_key(dataset, image_path), {})
        return dict(record) if isinstance(record, dict) else {}

    def save_image(
        self,
        dataset: AnnotationDataset,
        image_path: Path,
        record: dict[str, Any],
    ) -> Path:
        payload = self.read(dataset)
        payload["images"][self.image_key(dataset, image_path)] = dict(record)
        payload["updated_at"] = utc_now()
        path = self.path_for(dataset)
        atomic_write_json(path, payload)
        return path

    def _empty(self, dataset: AnnotationDataset) -> dict[str, Any]:
        return {
            "schema_version": METADATA_SCHEMA_VERSION,
            "dataset_id": dataset_identity(dataset.yaml_path, dataset.root),
            "dataset_yaml": str(dataset.yaml_path),
            "dataset_root": str(dataset.root),
            "updated_at": utc_now(),
            "images": {},
        }

    @staticmethod
    def _quarantine(path: Path) -> None:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        target = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
        try:
            path.replace(target)
        except OSError:
            pass


class AnnotationReportStore:
    def __init__(self, base_root: Path | None = None) -> None:
        self.base_root = base_root or _local_app_root() / "annotation_reports"

    def save(self, report: dict[str, Any]) -> Path:
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S%f")
        dataset_id = str(report.get("dataset_id", "unknown"))[:16]
        path = self.base_root / dataset_id / f"auto-annotation-{stamp}.json"
        payload = {"schema_version": REPORT_SCHEMA_VERSION, **report}
        atomic_write_json(path, payload)
        return path
