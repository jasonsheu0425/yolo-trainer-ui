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
from domain.annotation import PixelBox
from persistence.yolo_annotation_store import YoloAnnotationStore
from services.annotation_service import AnnotationService
from ui.annotation.annotation_page import AnnotationPage
from ui.annotation.bounding_box_item import BoundingBoxItem


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
