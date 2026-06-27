from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

from core.config_manager import ConfigManager
from ui.widgets import PageHeader, PathPicker


class SettingsPage(QWidget):
    settings_saved = Signal(dict)

    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(PageHeader("Settings", "設定 YOLO 執行環境與各頁面的預設值。"))

        box = QGroupBox("應用程式設定")
        form = QFormLayout(box)
        self.python = PathPicker("", "Python (python.exe);;所有檔案 (*.*)")
        self.yolo = PathPicker("", "執行檔 (yolo.exe);;所有檔案 (*.*)")
        self.runs = PathPicker("", directory=True)
        self.model = QLineEdit()
        self.device = QLineEdit()
        form.addRow("Python executable", self.python)
        form.addRow("YOLO command", self.yolo)
        form.addRow("Default runs folder", self.runs)
        form.addRow("Default model", self.model)
        form.addRow("Default device", self.device)
        layout.addWidget(box)

        buttons = QHBoxLayout()
        save = QPushButton("Save Settings")
        save.setObjectName("primaryButton")
        save.clicked.connect(self.save)
        reload_button = QPushButton("重新載入")
        reload_button.clicked.connect(self.load)
        buttons.addStretch()
        buttons.addWidget(reload_button)
        buttons.addWidget(save)
        layout.addLayout(buttons)
        layout.addStretch()
        self.load()

    def load(self) -> None:
        values = self.config.load()
        self.python.set_path(str(values["python_executable"]))
        self.yolo.set_path(str(values["yolo_command"]))
        self.runs.set_path(str(values["runs_folder"]))
        self.model.setText(str(values["default_model"]))
        self.device.setText(str(values["default_device"]))

    def save(self) -> None:
        values = {
            "python_executable": self.python.path(),
            "yolo_command": self.yolo.path() or "yolo",
            "runs_folder": self.runs.path() or "runs/detect",
            "default_model": self.model.text().strip() or "yolov8n.pt",
            "default_device": self.device.text().strip() or "0",
        }
        try:
            self.config.save(values)
        except OSError as exc:
            QMessageBox.critical(self, "儲存失敗", str(exc))
            return
        self.settings_saved.emit(values)
        QMessageBox.information(self, "設定", "設定已儲存。")

