from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PySide6.QtCore import QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from core.config_manager import ConfigManager
from core.i18n_manager import get_i18n
from core.runtime_manager import RuntimeManager
from domain.annotation import AnnotationSource, BoundingBox, ModelPrediction, PixelBox
from persistence.annotation_metadata_store import AnnotationReportStore
from persistence.yolo_annotation_store import YoloAnnotationStore
from services.annotation_inference_service import AnnotationInferenceService, InferenceState
from services.annotation_service import AnnotationService
from ui.annotation.annotation_page import AnnotationPage
from ui.annotation.bounding_box_item import BoundingBoxItem
from workers.annotation_inference_controller import InferenceWorkerController


FAKE_WORKER = Path(__file__).parent / "fixtures" / "fake_annotation_worker.py"


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def sample_dataset(tmp_path: Path, *, invalid: bool = False) -> Path:
    images = tmp_path / "images" / "train"
    labels = tmp_path / "labels" / "train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for name in ("one.jpg", "two.jpg"):
        Image.new("RGB", (640, 480), "white").save(images / name)
    if invalid:
        (labels / "one.txt").write_text(
            "0 0.5 0.5 0.25 0.25\ngarbage\n", encoding="utf-8"
        )
    yaml = tmp_path / "data.yaml"
    yaml.write_text(
        "path: .\ntrain: images/train\nnames: [person, vehicle]\n",
        encoding="utf-8",
    )
    return yaml


def make_page(tmp_path: Path, *, invalid: bool = False) -> AnnotationPage:
    app()
    store = YoloAnnotationStore(tmp_path / "backups")
    page = AnnotationPage(
        ConfigManager(tmp_path / "settings.json"), AnnotationService(store)
    )
    page.resize(1200, 800)
    page.show()
    page.dataset_picker.set_path(str(sample_dataset(tmp_path, invalid=invalid)))
    page.open_dataset()
    QApplication.processEvents()
    return page


def wait_until(predicate, timeout=5000):
    elapsed = 0
    while not predicate() and elapsed < timeout:
        QTest.qWait(20)
        elapsed += 20
    assert predicate(), "condition timed out"


def assisted_page(tmp_path: Path, image_names=("one.jpg", "two.jpg")):
    app()
    yaml = sample_dataset(tmp_path)
    yaml.write_text(
        "path: .\ntrain: images/train\nnames: [object]\n", encoding="utf-8"
    )
    images = tmp_path / "images" / "train"
    for path in list(images.iterdir()):
        path.unlink()
    for name in image_names:
        Image.new("RGB", (640, 480), "white").save(images / name)
    config = ConfigManager(tmp_path / "settings.json")
    config.save({"python_executable": os.sys.executable})
    annotations = AnnotationService(YoloAnnotationStore(tmp_path / "backups"))
    inference = AnnotationInferenceService(
        RuntimeManager(config), annotations, InferenceWorkerController(FAKE_WORKER),
        AnnotationReportStore(tmp_path / "reports"),
    )
    page = AnnotationPage(config, annotations, inference)
    page.resize(1250, 900)
    page.show()
    page.dataset_picker.set_path(str(yaml))
    page.open_dataset()
    model = tmp_path / "best.pt"
    model.write_bytes(b"fake")
    page.model_picker.set_path(str(model))
    return page


def test_page_construct_empty_open_lists_and_live_language(tmp_path):
    app()
    page = AnnotationPage(
        ConfigManager(tmp_path / "settings.json"), AnnotationService()
    )
    assert page.image_list.count() == 0
    page.dataset_picker.set_path(str(sample_dataset(tmp_path)))
    page.open_dataset()
    assert page.image_list.count() == 2
    assert page.class_list.count() == 2
    get_i18n().set_language("en_US")
    QApplication.processEvents()
    assert page.tool_actions["draw"].text() == "Draw Box"
    assert page.service.document is not None
    get_i18n().set_language("zh_TW")
    QApplication.processEvents()
    assert page.service.document is not None
    page.close()


def test_qttest_draw_save_delete_undo_redo_and_navigation(tmp_path):
    page = make_page(tmp_path)
    page.set_tool("draw")
    start = page.canvas.mapFromScene(100, 100)
    end = page.canvas.mapFromScene(300, 300)
    QTest.mousePress(page.canvas.viewport(), Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(page.canvas.viewport(), end)
    QTest.mouseRelease(page.canvas.viewport(), Qt.MouseButton.LeftButton, pos=end)
    QApplication.processEvents()
    assert len(page.service.document.boxes) == 1
    page.save()
    label = page.service.document.label_path
    fields = label.read_text(encoding="utf-8").split()
    assert fields[0] == "0"
    assert [float(value) for value in fields[1:]] == pytest.approx(
        [0.3125, 0.416667, 0.3125, 0.416667], abs=0.002
    )

    page.service.selected_index = 0
    page.delete_selected()
    assert page.service.document.boxes == []
    page.undo()
    assert len(page.service.document.boxes) == 1
    page.redo()
    assert page.service.document.boxes == []
    page.undo()
    page.next_image()
    assert page.service.index == 1 and label.is_file()
    page.previous_image()
    assert page.service.index == 0
    page.close()


def test_autosave_does_not_create_untouched_empty_label(tmp_path):
    page = make_page(tmp_path)
    first_label = page.service.document.label_path
    page.next_image()
    assert not first_label.exists()
    page.close()


def test_select_move_and_corner_resize_are_persisted(tmp_path):
    page = make_page(tmp_path)
    page.service.create_pixel_box(PixelBox(100, 100, 300, 300), 640, 480, 0)
    page._render_document(select=0)
    page.set_tool("select")
    item = next(
        item
        for item in page.canvas.scene().items()
        if isinstance(item, BoundingBoxItem)
    )
    original = page.service.document.boxes[0]
    center = page.canvas.mapFromScene(item.scene_rect().center())
    moved_center = page.canvas.mapFromScene(
        item.scene_rect().center() + QPointF(30, 20)
    )
    QTest.mousePress(page.canvas.viewport(), Qt.MouseButton.LeftButton, pos=center)
    QTest.mouseMove(page.canvas.viewport(), moved_center)
    QTest.mouseRelease(
        page.canvas.viewport(), Qt.MouseButton.LeftButton, pos=moved_center
    )
    QApplication.processEvents()
    moved = page.service.document.boxes[0]
    assert moved.x_center > original.x_center
    assert moved.y_center > original.y_center

    item = next(
        item
        for item in page.canvas.scene().items()
        if isinstance(item, BoundingBoxItem)
    )
    item.setSelected(True)
    corner = item.scene_rect().bottomRight()
    corner_pos = page.canvas.mapFromScene(corner)
    resized_pos = page.canvas.mapFromScene(corner + QPointF(30, 30))
    QTest.mousePress(page.canvas.viewport(), Qt.MouseButton.LeftButton, pos=corner_pos)
    QTest.mouseMove(page.canvas.viewport(), resized_pos)
    QTest.mouseRelease(
        page.canvas.viewport(), Qt.MouseButton.LeftButton, pos=resized_pos
    )
    QApplication.processEvents()
    resized = page.service.document.boxes[0]
    assert resized.width > moved.width
    assert resized.height > moved.height
    page.canvas.scene().clearSelection()
    QApplication.processEvents()
    assert page.service.selected_index is None
    page.save()
    page.service.load_image(0)
    assert page.service.document.boxes[0].width == pytest.approx(resized.width, abs=1e-6)
    page.close()


def test_invalid_label_is_visible_and_not_silently_overwritten(tmp_path):
    page = make_page(tmp_path, invalid=True)
    document = page.service.document
    assert document is not None and len(document.invalid_lines) == 1
    assert page.issue_label.text()
    assert page.view_issues_button.isVisible()
    assert page.repair_button.isVisible()
    original = document.label_path.read_text(encoding="utf-8")
    page.next_image()
    assert document.label_path.read_text(encoding="utf-8") == original
    page.close()


def test_unreadable_current_image_is_safe(tmp_path):
    yaml = sample_dataset(tmp_path)
    (tmp_path / "images" / "train" / "one.jpg").write_bytes(b"not an image")
    app()
    page = AnnotationPage(
        ConfigManager(tmp_path / "settings.json"), AnnotationService()
    )
    page.dataset_picker.set_path(str(yaml))
    page.open_dataset()
    assert page.canvas.image_size == (0, 0)
    assert page.status_label.text()
    page.close()


def test_model_assistance_panel_ready_predict_confidence_and_language(tmp_path):
    page = assisted_page(tmp_path)
    assert page.confidence.value() == .25
    assert page.load_model_button.isEnabled()
    page.load_model()
    wait_until(lambda: page.inference.state is InferenceState.READY)
    assert page.device_combo.findData("0") == -1
    pid = page.inference.controller.process_id
    assert page.predict_button.isEnabled()
    page.predict_current()
    wait_until(lambda: page.service.document.dirty)
    assert len(page.service.document.boxes) == 1
    assert page.service.document.box_metadata[0].confidence == .9
    assert not page.service.document.label_path.exists()
    get_i18n().set_language("en_US")
    QApplication.processEvents()
    assert page.inference.controller.process_id == pid
    assert page.service.document.dirty
    get_i18n().set_language("zh_TW")
    page.inference.shutdown()
    page.close()


def test_add_replace_prediction_batches_are_single_undo_and_edit_hides_confidence(
    tmp_path,
):
    page = assisted_page(tmp_path)
    page.service.create_box(BoundingBox(0, .3, .3, .1, .1))
    original = page.service.document.boxes.copy()
    predictions = [ModelPrediction(BoundingBox(0, .6, .6, .2, .2), .85, "best.pt")]
    page._apply_prediction_choice(predictions, "add")
    assert len(page.service.document.boxes) == 2
    assert page.service.undo() and page.service.document.boxes == original
    page._apply_prediction_choice(predictions, "replace")
    assert page.service.document.boxes == [predictions[0].box]
    assert page.service.undo() and page.service.document.boxes == original
    page.service.redo()
    page.service.move_box_pixels(0, 5, 5, 640, 480)
    assert page.service.document.source is AnnotationSource.MODEL_ASSISTED
    assert page.service.document.box_metadata[0].confidence is None
    page.close()


def test_model_class_mismatch_shows_both_class_lists(tmp_path):
    page = assisted_page(tmp_path)
    page.model_picker.set_path(str(tmp_path / "mismatch.pt"))
    (tmp_path / "mismatch.pt").write_bytes(b"fake")
    page.load_model()
    wait_until(lambda: page.inference.state is InferenceState.READY)
    text = page.inference_status.text()
    assert "0 object" in text and "0 different" in text
    assert not page.predict_button.isEnabled()
    page.inference.shutdown()
    page.close()


def test_zero_detection_ui_does_not_create_label(tmp_path):
    page = assisted_page(tmp_path, ("zero.jpg",))
    page.load_model()
    wait_until(lambda: page.inference.state is InferenceState.READY)
    page.predict_current()
    wait_until(lambda: page.inference.state is InferenceState.READY)
    assert page.service.document.boxes == []
    assert not page.service.document.label_path.exists()
    assert page.inference_status.text()
    page.inference.shutdown()
    page.close()


def test_batch_progress_cancel_and_runtime_failure_leave_manual_editor_available(
    tmp_path,
):
    page = assisted_page(tmp_path, ("a.jpg", "b.jpg", "c.jpg"))
    page.load_model()
    wait_until(lambda: page.inference.state is InferenceState.READY)
    page.inference.batch_progress.connect(
        lambda value: page.inference.cancel_batch() if value["processed"] == 1 else None
    )
    page.inference.start_batch(.25)
    wait_until(lambda: not page.inference.batch_running)
    assert page.batch_progress.isVisible()
    page.inference.shutdown()
    page.service.load_image(0)
    page._box_created(PixelBox(10, 10, 100, 100))
    assert page.service.document.dirty
    page.close()
