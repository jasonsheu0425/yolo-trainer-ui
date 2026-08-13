from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.config_manager import ConfigManager
from core.i18n_manager import get_i18n
from ui.main_window import MainWindow


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_mode_switch_persistence_and_profile_mapping(tmp_path):
    app()
    settings = ConfigManager(tmp_path / "settings.json")
    settings.save({"ui_mode": "advanced"})
    window = MainWindow()
    # Replace only the test window's config path; no user setting is written.
    window.config = settings
    window.set_ui_mode("simple")
    assert window.navigation.count() == 5
    window.simple_page.model_profile.setCurrentIndex(window.simple_page.model_profile.findData("balanced"))
    window.simple_page.training_profile.setCurrentIndex(window.simple_page.training_profile.findData("standard"))
    window.simple_page._profiles_selected()
    values = window.train_page.training_values()
    assert (values["model"], values["epochs"], values["imgsz"], values["batch"]) == ("yolov8s.pt", 100, 640, 16)
    window.set_ui_mode("advanced")
    assert window.navigation.count() == 11
    assert window.train_page.epochs.value() == 100
    window.close()

def test_language_and_advanced_values_are_preserved():
    app()
    window = MainWindow()
    window.train_page.epochs.setValue(73)
    window.train_page.batch.setValue(11)
    window.train_page.imgsz.setValue(832)
    window.set_ui_mode("simple")
    assert (window.train_page.epochs.value(), window.train_page.batch.value(), window.train_page.imgsz.value()) == (73, 11, 832)
    get_i18n().set_language("en_US")
    QApplication.processEvents()
    assert window.simple_page.start_button.text() == "Start Training"
    get_i18n().set_language("zh_TW")
    window.close()
