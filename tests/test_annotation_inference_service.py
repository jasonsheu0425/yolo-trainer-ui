from __future__ import annotations

import os
from pathlib import Path
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.config_manager import ConfigManager
from core.dataset_checker import check_dataset
from core.runtime_manager import RuntimeManager
from persistence.annotation_metadata_store import (
    AnnotationMetadataStore,
    AnnotationReportStore,
)
from persistence.yolo_annotation_store import YoloAnnotationStore
from services.annotation_inference_service import (
    AnnotationInferenceService,
    InferenceState,
)
from services.annotation_service import AnnotationService
from workers.annotation_inference_controller import InferenceWorkerController


FAKE_WORKER = Path(__file__).parent / "fixtures" / "fake_annotation_worker.py"


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def wait_until(predicate, timeout=5000):
    elapsed = 0
    while not predicate() and elapsed < timeout:
        QTest.qWait(20)
        elapsed += 20
    assert predicate(), "condition timed out"


def setup(tmp_path: Path, image_names=("one.jpg", "two.jpg"), model_name="best.pt"):
    app()
    images = tmp_path / "images" / "train"
    images.mkdir(parents=True)
    for name in image_names:
        Image.new("RGB", (640, 480), "white").save(images / name)
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        "path: .\ntrain: images/train\nval: images/train\nnames: [object]\n",
        encoding="utf-8",
    )
    config = ConfigManager(tmp_path / "settings.json")
    config.save({"python_executable": sys.executable})
    annotations = AnnotationService(
        YoloAnnotationStore(tmp_path / "backups"),
        AnnotationMetadataStore(tmp_path / "metadata"),
    )
    annotations.open_dataset(yaml)
    controller = InferenceWorkerController(FAKE_WORKER)
    inference = AnnotationInferenceService(
        RuntimeManager(config), annotations, controller,
        AnnotationReportStore(tmp_path / "reports"),
    )
    model = tmp_path / model_name
    model.write_bytes(b"fake model")
    return annotations, inference, model


def load(inference, model, device="auto"):
    assert inference.load_model(model, device)
    wait_until(lambda: inference.state in {InferenceState.READY, InferenceState.ERROR})
    assert inference.state is InferenceState.READY


def select_image(annotations, name):
    annotations.load_image(
        next(index for index, path in enumerate(annotations.images) if path.name == name)
    )


def test_runtime_missing_and_worker_start_load_cpu_state(tmp_path):
    app()
    config = ConfigManager(tmp_path / "settings.json")
    annotations = AnnotationService()
    service = AnnotationInferenceService(
        RuntimeManager(config), annotations, InferenceWorkerController(FAKE_WORKER)
    )
    service.runtime._python_candidates = lambda: iter(())  # type: ignore[method-assign]
    model = tmp_path / "best.pt"
    model.write_bytes(b"fake")
    assert not service.load_model(model)
    assert service.state is InferenceState.ERROR

    annotations, service, model = setup(tmp_path / "ready")
    load(service, model, "cpu")
    assert service.actual_device == "cpu"
    assert service.compatibility == "exact_match"
    service.shutdown()
    assert service.state is InferenceState.UNLOADED


def test_persistent_worker_predict_three_images_model_loaded_once(tmp_path):
    annotations, service, model = setup(tmp_path, ("a.jpg", "b.jpg", "c.jpg"))
    results = []
    service.prediction_ready.connect(lambda values, image: results.append((values, image)))
    load(service, model)
    pid = service.controller.process_id
    for index in range(3):
        annotations.load_image(index)
        assert service.predict_current(.25)
        wait_until(lambda: len(results) == index + 1)
        assert service.controller.process_id == pid
        assert results[-1][0][0].confidence == .9
    service.shutdown()


def test_zero_detection_invalid_values_and_image_mismatch_are_safe(tmp_path):
    annotations, service, model = setup(
        tmp_path, ("zero.jpg", "invalid.jpg", "wrong.jpg")
    )
    results, errors = [], []
    service.prediction_ready.connect(lambda values, image: results.append(values))
    service.error.connect(lambda code, message: errors.append(code))
    load(service, model)
    select_image(annotations, "zero.jpg")
    assert service.predict_current(.25)
    wait_until(lambda: len(results) == 1)
    assert results[0] == [] and not annotations.document.label_path.exists()
    select_image(annotations, "invalid.jpg")
    assert service.predict_current(.25)
    wait_until(lambda: "invalid_prediction_values" in errors)
    assert not annotations.document.label_path.exists()
    service._set_state(InferenceState.READY)
    select_image(annotations, "wrong.jpg")
    assert service.predict_current(.25)
    wait_until(lambda: "image_identity_mismatch" in errors)
    service.shutdown()


def test_class_mismatch_requires_session_override_and_count_mismatch_blocks(tmp_path):
    _, service, model = setup(tmp_path, model_name="mismatch.pt")
    load(service, model)
    assert service.compatibility == "id_match_name_mismatch"
    assert not service.model_usable
    service.override_class_mismatch()
    assert service.model_usable
    service.shutdown()


def test_worker_crash_error_restart_unload_and_gpu_mode_mock(tmp_path):
    annotations, service, model = setup(tmp_path)
    load(service, model, "0")
    # Fake worker reports CPU regardless of request; app preserves its actual response.
    assert service.actual_device == "cpu"
    assert service.unload_model()
    wait_until(lambda: service.state is InferenceState.UNLOADED)
    load(service, model, "0")
    crash = annotations.images[0].with_name("crash.jpg")
    Image.new("RGB", (20, 20)).save(crash)
    annotations.images = (crash,)
    annotations.load_image(0)
    assert service.predict_current(.25)
    wait_until(lambda: service.state is InferenceState.ERROR)
    assert service.load_model(model)
    wait_until(lambda: service.state is InferenceState.READY)
    service.shutdown()


def test_timeout_kills_worker_and_allows_restart(tmp_path, monkeypatch):
    import services.annotation_inference_service as module

    monkeypatch.setattr(module, "PREDICTION_TIMEOUT_MS", 50)
    annotations, service, model = setup(tmp_path, ("sleep.jpg",))
    load(service, model)
    assert service.predict_current(.25)
    wait_until(lambda: service.state is InferenceState.ERROR)
    wait_until(lambda: not service.controller.running)


def test_batch_protects_existing_negative_and_no_detection_then_reports(tmp_path):
    annotations, service, model = setup(
        tmp_path,
        (
            "existing.jpg", "negative.jpg", "invalid_label.jpg", "unreadable.jpg",
            "zero.jpg", "new.jpg",
        ),
    )
    (tmp_path / "images" / "train" / "unreadable.jpg").write_bytes(b"broken")
    existing_image = next(path for path in annotations.images if path.name == "existing.jpg")
    negative_image = next(path for path in annotations.images if path.name == "negative.jpg")
    existing = annotations.store.resolve_label_path(existing_image)
    negative = annotations.store.resolve_label_path(negative_image)
    invalid_image = next(
        path for path in annotations.images if path.name == "invalid_label.jpg"
    )
    invalid = annotations.store.resolve_label_path(invalid_image)
    existing.parent.mkdir(parents=True)
    original = b"0 0.500000 0.500000 0.200000 0.200000\n"
    existing.write_bytes(original)
    negative.write_bytes(b"")
    malformed = b"not a valid YOLO label\n"
    invalid.write_bytes(malformed)
    # Refresh current after labels are prepared.
    annotations.load_image(0)
    finished = []
    service.batch_finished.connect(finished.append)
    load(service, model)
    plan = service.batch_plan()
    assert plan["eligible_count"] == 2 and plan["existing_skipped"] == 3
    assert plan["unreadable_skipped"] == 1
    assert service.start_batch(.25)
    wait_until(lambda: bool(finished))
    report = finished[0]
    assert report["created"] == 1 and report["no_detection"] == 1
    assert report["skipped"] == 4
    assert existing.read_bytes() == original and negative.read_bytes() == b""
    assert invalid.read_bytes() == malformed
    zero = annotations.store.resolve_label_path(
        next(path for path in annotations.images if path.name == "zero.jpg")
    )
    created = annotations.store.resolve_label_path(
        next(path for path in annotations.images if path.name == "new.jpg")
    )
    assert not zero.exists() and created.is_file()
    assert Path(report["report_path"]).is_file()
    service.shutdown()


def test_batch_cancel_stops_new_requests_and_keeps_completed(tmp_path):
    annotations, service, model = setup(tmp_path, ("a.jpg", "b.jpg", "c.jpg"))
    progress, finished = [], []
    service.batch_progress.connect(progress.append)
    service.batch_finished.connect(finished.append)
    load(service, model)
    service.batch_progress.connect(
        lambda value: service.cancel_batch() if value["processed"] == 1 else None
    )
    assert service.start_batch(.25)
    wait_until(lambda: bool(finished))
    assert finished[0]["status"] == "cancelled"
    assert finished[0]["processed"] == 1 and finished[0]["remaining"] == 2
    assert sum(path.exists() for path in (
        annotations.store.resolve_label_path(image) for image in annotations.images
    )) == 1
    service.shutdown()


def test_auto_generated_label_passes_dataset_check(tmp_path):
    annotations, service, model = setup(tmp_path, ("new.jpg",))
    finished = []
    service.batch_finished.connect(finished.append)
    load(service, model)
    assert service.start_batch(.25)
    wait_until(lambda: bool(finished))
    assert finished[0]["created"] == 1
    assert check_dataset(annotations.dataset.yaml_path)["errors"] == []
    service.shutdown()
