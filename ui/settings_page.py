from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox, QPushButton, QVBoxLayout, QWidget

from core.config_manager import ConfigManager
from core.runtime_manager import RuntimeManager
from ui.widgets import PageHeader, PathPicker


class SettingsPage(QWidget):
    settings_saved = Signal(dict)

    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.runtime_manager = RuntimeManager(config)
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
        self.managed_location = QLabel(str(self.runtime_manager.managed_runtime_folder()))
        self.managed_location.setWordWrap(True)
        self.runtime_status = QLabel("Not checked")
        self.runtime_status.setWordWrap(True)
        form.addRow("Managed runtime location", self.managed_location)
        form.addRow("Runtime status", self.runtime_status)
        layout.addWidget(box)

        buttons = QHBoxLayout()
        save = QPushButton("Save Settings")
        save.setObjectName("primaryButton")
        save.clicked.connect(self.save)
        reload_button = QPushButton("重新載入")
        reload_button.clicked.connect(self.load)
        reset_button = QPushButton("Reset to Auto Detect")
        reset_button.clicked.connect(self.reset_runtime_overrides)
        buttons.addStretch()
        buttons.addWidget(reset_button)
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
        self._update_runtime_status()

    def save(self) -> None:
        values = {
            "python_executable": self.python.path(),
            "yolo_command": self.yolo.path(),
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
        self._update_runtime_status()
        QMessageBox.information(self, "Settings", "Settings saved.")

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
