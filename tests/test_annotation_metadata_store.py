from __future__ import annotations

import json
from pathlib import Path

from domain.annotation import AnnotationDataset
from persistence.annotation_metadata_store import (
    AnnotationMetadataStore,
    AnnotationReportStore,
    dataset_identity,
)


def dataset(tmp_path: Path, name: str = "one") -> AnnotationDataset:
    root = tmp_path / name
    root.mkdir()
    yaml = root / "data.yaml"
    yaml.write_text("names: [object]", encoding="utf-8")
    return AnnotationDataset(yaml, root, {0: "object"}, {})


def test_dataset_identity_uses_canonical_paths(tmp_path):
    first = dataset(tmp_path, "first")
    second = dataset(tmp_path, "second")
    assert dataset_identity(first.yaml_path, first.root) != dataset_identity(second.yaml_path, second.root)
    assert dataset_identity(first.yaml_path, first.root) == dataset_identity(first.yaml_path, first.root)


def test_metadata_sources_and_no_detection_are_atomic(tmp_path):
    data = dataset(tmp_path)
    image = data.root / "images" / "one.jpg"
    store = AnnotationMetadataStore(tmp_path / "metadata")
    for source in ("manual", "model_generated", "model_assisted", "unknown"):
        store.save_image(data, image, {"source": source, "last_prediction_status": "no_detection"})
        assert store.get_image(data, image)["source"] == source
    assert store.get_image(data, image)["last_prediction_status"] == "no_detection"
    assert not list((tmp_path / "metadata").rglob("*.tmp"))


def test_corrupt_metadata_is_quarantined_and_labels_are_independent(tmp_path):
    data = dataset(tmp_path)
    store = AnnotationMetadataStore(tmp_path / "metadata")
    path = store.path_for(data)
    path.parent.mkdir(parents=True)
    path.write_text("broken", encoding="utf-8")
    assert store.get_image(data, data.root / "x.jpg") == {}
    assert list(path.parent.glob("*.corrupt-*.json"))


def test_report_store_writes_versioned_audit_json(tmp_path):
    store = AnnotationReportStore(tmp_path / "reports")
    path = store.save({"dataset_id": "abc", "status": "completed", "errors": []})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1 and payload["status"] == "completed"
