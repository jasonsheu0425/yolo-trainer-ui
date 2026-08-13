from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

from core.config_manager import ConfigManager
from core.i18n_manager import get_i18n, tr
from core.runtime_manager import RuntimeManager
from ui.widgets import PageHeader, PathPicker, WheelSafeComboBox, set_tooltip


class SettingsPage(QWidget):
    settings_saved = Signal(dict)

    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.runtime_manager = RuntimeManager(config)
        self.i18n = get_i18n()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(PageHeader("settings.title", "settings.description"))

        self.box = QGroupBox()
        form = QFormLayout(self.box)
        self.form = form
        self.python = PathPicker("", "Python (python.exe);;所有檔案 (*.*)")
        self.yolo = PathPicker("", "執行檔 (yolo.exe);;所有檔案 (*.*)")
        self.runs = PathPicker("", directory=True)
        self.model = QLineEdit()
        self.device = QLineEdit()
        self.language = WheelSafeComboBox()
        self.language.addItem("繁體中文", "zh_TW")
        self.language.addItem("English", "en_US")
        self.language.currentIndexChanged.connect(self._language_changed)
        self.ui_mode = WheelSafeComboBox()
        self.ui_mode.addItem("", "simple")
        self.ui_mode.addItem("", "advanced")
        self.ui_mode.currentIndexChanged.connect(self._mode_changed)
        self.labels = [QLabel() for _ in range(9)]
        for label, widget in zip(self.labels, (self.language, self.ui_mode, self.python, self.yolo, self.runs, self.model, self.device)):
            form.addRow(label, widget)
        self.managed_location = QLabel(str(self.runtime_manager.managed_runtime_folder()))
        self.managed_location.setWordWrap(True)
        self.runtime_status = QLabel("Not checked")
        self.runtime_status.setWordWrap(True)
        form.addRow(self.labels[7], self.managed_location)
        form.addRow(self.labels[8], self.runtime_status)
        set_tooltip(self.python, "tooltip.runtime.python")
        set_tooltip(self.yolo, "tooltip.runtime.yolo")
        layout.addWidget(self.box)

        buttons = QHBoxLayout()
        self.save_button = QPushButton()
        save = self.save_button
        save.setObjectName("primaryButton")
        save.clicked.connect(self.save)
        self.reload_button = QPushButton()
        reload_button = self.reload_button
        reload_button.clicked.connect(self.load)
        self.reset_button = QPushButton()
        reset_button = self.reset_button
        reset_button.clicked.connect(self.reset_runtime_overrides)
        buttons.addStretch()
        buttons.addWidget(reset_button)
        buttons.addWidget(reload_button)
        buttons.addWidget(save)
        layout.addLayout(buttons)
        layout.addStretch()
        self.i18n.language_changed.connect(self._retranslate_ui)
        self._retranslate_ui()
        self.load()

    def _retranslate_ui(self, _locale: str | None = None) -> None:
        self.box.setTitle(tr("settings.application"))
        for label, key in zip(self.labels, ("settings.language", "mode.label", "settings.python", "settings.yolo", "settings.runs", "settings.model", "settings.device", "settings.managed_location", "settings.runtime_status")):
            label.setText(tr(key))
        for index in range(self.language.count()):
            locale = str(self.language.itemData(index))
            self.language.setItemText(index, tr(f"language.{locale}"))
        for index in range(self.ui_mode.count()):
            self.ui_mode.setItemText(index, tr(f"mode.{self.ui_mode.itemData(index)}"))
        self.save_button.setText(tr("settings.save"))
        self.reload_button.setText(tr("settings.reload"))
        self.reset_button.setText(tr("settings.reset"))

    def load(self) -> None:
        values = self.config.load()
        self.python.set_path(str(values["python_executable"]))
        self.yolo.set_path(str(values["yolo_command"]))
        self.runs.set_path(str(values["runs_folder"]))
        self.model.setText(str(values["default_model"]))
        self.device.setText(str(values["default_device"]))
        self.language.blockSignals(True)
        self.language.setCurrentIndex(max(0, self.language.findData(str(values.get("language", "zh_TW")))))
        self.language.blockSignals(False)
        self.ui_mode.blockSignals(True)
        self.ui_mode.setCurrentIndex(max(0, self.ui_mode.findData(str(values.get("ui_mode", "advanced")))))
        self.ui_mode.blockSignals(False)
        self._update_runtime_status()

    def _language_changed(self) -> None:
        locale = str(self.language.currentData())
        self.config.save({"language": locale})
        self.i18n.set_language(locale)
        self.settings_saved.emit({"language": locale})

    def _mode_changed(self) -> None:
        mode = str(self.ui_mode.currentData())
        self.config.save({"ui_mode": mode})
        self.settings_saved.emit({"ui_mode": mode})

    def save(self) -> None:
        values = {
            "language": str(self.language.currentData()),
            "ui_mode": str(self.ui_mode.currentData()),
            "python_executable": self.python.path(),
            "yolo_command": self.yolo.path(),
            "runs_folder": self.runs.path() or "runs/detect",
            "default_model": self.model.text().strip() or "yolov8n.pt",
            "default_device": self.device.text().strip() or "0",
        }
        try:
            self.config.save(values)
        except OSError as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))
            return
        self.settings_saved.emit(values)
        self._update_runtime_status()
        QMessageBox.information(self, tr("settings.title"), tr("settings.save"))

    def reset_runtime_overrides(self) -> None:
        self.python.set_path("")
        self.yolo.set_path("")
        try:
            self.config.save({"python_executable": "", "yolo_command": ""})
        except OSError as exc:
            QMessageBox.critical(self, "Settings Error", str(exc))
            return
        values = {"python_executable": "", "yolo_command": ""}
        self.settings_saved.emit(values)
        self._update_runtime_status()
        QMessageBox.information(self, "Runtime Settings", "Explicit overrides were cleared. The managed environment was not deleted.")

    def _update_runtime_status(self) -> None:
        python = self.runtime_manager.discover_python(validate=False)
        yolo = self.runtime_manager.discover_yolo(validate=False)
        if yolo.get("available"):
            self.runtime_status.setText(f"Ready: {yolo.get('program')} ({yolo.get('source')})")
        elif python.get("available"):
            self.runtime_status.setText("Partial: Python found; YOLO runtime missing")
        else:
            self.runtime_status.setText("Missing: Python and YOLO were not found")
