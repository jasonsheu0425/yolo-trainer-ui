from __future__ import annotations

import os
from datetime import datetime

from PySide6.QtCore import Qt, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.runtime_manager import RuntimeManager
from core.runtime_worker import RuntimeWorker
from ui.widgets import PageHeader


class RuntimePage(QWidget):
    runtime_changed = Signal(dict)
    open_settings_requested = Signal()

    FIELD_LABELS = {
        "application_mode": "Application Mode",
        "python_executable": "Python Executable",
        "python_version": "Python Version",
        "python_source": "Python Source",
        "yolo_command": "YOLO Command",
        "yolo_available": "YOLO Available",
        "yolo_version": "YOLO / CLI Version",
        "ultralytics_available": "Ultralytics Available",
        "ultralytics_version": "Ultralytics Version",
        "torch_available": "PyTorch Available",
        "torch_version": "PyTorch Version",
        "cuda_available": "CUDA Available",
        "torch_cuda_version": "Torch CUDA Version",
        "gpu_count": "GPU Count",
        "gpu_name": "GPU Name",
        "managed_runtime_folder": "Managed Runtime Folder",
    }

    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.manager = RuntimeManager(config)
        self._thread: QThread | None = None
        self._worker: RuntimeWorker | None = None
        self.last_diagnostics: dict = {}

        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        page_layout.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.addWidget(PageHeader("Runtime / Environment", "Detect Python, Ultralytics, PyTorch, CUDA, and configure a per-user managed YOLO runtime."))

        self.status = QLabel("Not checked")
        self.status.setWordWrap(True)
        self.status.setProperty("state", "warning")
        layout.addWidget(self.status)

        environment = QGroupBox("Environment Status")
        form = QFormLayout(environment)
        self.values: dict[str, QLabel] = {}
        for key, title in self.FIELD_LABELS.items():
            value = QLabel("Not checked")
            value.setWordWrap(True)
            value.setTextInteractionFlags(value.textInteractionFlags() | Qt.TextInteractionFlag.TextSelectableByMouse)
            form.addRow(title, value)
            self.values[key] = value
        layout.addWidget(environment)

        actions = QGroupBox("Runtime Actions")
        action_layout = QVBoxLayout(actions)
        row = QHBoxLayout()
        self.diagnostics_button = QPushButton("Run Diagnostics")
        self.diagnostics_button.setObjectName("primaryButton")
        self.create_button = QPushButton("Create Managed YOLO Environment")
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        settings_button = QPushButton("Open Settings")
        guide_button = QPushButton("Open PyTorch Installation Guide")
        folder_button = QPushButton("Open Managed Runtime Folder")
        self.diagnostics_button.clicked.connect(self.run_diagnostics)
        self.create_button.clicked.connect(self.create_managed_runtime)
        self.cancel_button.clicked.connect(self.cancel)
        settings_button.clicked.connect(self.open_settings_requested.emit)
        guide_button.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://pytorch.org/get-started/locally/")))
        folder_button.clicked.connect(self.open_managed_folder)
        for button in (self.diagnostics_button, self.create_button, self.cancel_button, settings_button, guide_button, folder_button):
            row.addWidget(button)
        row.addStretch()
        action_layout.addLayout(row)
        self.current_step = QLabel("Idle")
        action_layout.addWidget(self.current_step)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        action_layout.addWidget(self.progress)
        action_layout.addWidget(QLabel("Managed setup installs the standard Ultralytics package. It does not force a specific CUDA wheel; use the PyTorch guide if GPU support is unavailable."))
        layout.addWidget(actions)

        log_box = QGroupBox("Diagnostics / Setup Log")
        log_layout = QVBoxLayout(log_box)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(210)
        self.log.setObjectName("console")
        log_layout.addWidget(self.log)
        layout.addWidget(log_box)
        layout.addStretch()

    def apply_settings(self, values: dict) -> None:
        self.config.settings.update(values)
        if not self._thread:
            self.run_diagnostics()

    def run_diagnostics(self) -> None:
        self._start_worker("diagnostics")

    def create_managed_runtime(self) -> None:
        if self._thread:
            return
        answer = QMessageBox.question(
            self,
            "Create Managed YOLO Environment",
            "Create or update a per-user virtual environment and install Ultralytics? This may download several large packages.",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self._append_log(f"\n=== Managed runtime setup started {datetime.now().isoformat(timespec='seconds')} ===\n")
        self._start_worker("create")

    def _start_worker(self, action: str) -> None:
        if self._thread:
            return
        thread = QThread(self)
        worker = RuntimeWorker(self.manager, action)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(self.progress.setValue)
        worker.step_changed.connect(self.current_step.setText)
        worker.log.connect(self._append_log)
        worker.diagnostics_ready.connect(self._display_diagnostics)
        worker.settings_changed.connect(self._settings_changed)
        worker.finished.connect(self._worker_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._thread_finished)
        self._thread = thread
        self._worker = worker
        self.diagnostics_button.setEnabled(False)
        self.create_button.setEnabled(False)
        self.cancel_button.setEnabled(action == "create")
        thread.start()

    def cancel(self) -> None:
        if self._worker:
            self.cancel_button.setEnabled(False)
            self._worker.cancel()

    def _worker_finished(self, success: bool, message: str) -> None:
        self._append_log(f"{message}\n")
        if not success and message != "Cancelled":
            QMessageBox.warning(self, "Runtime / Environment", message)

    def _thread_finished(self) -> None:
        self._thread = None
        self._worker = None
        self.diagnostics_button.setEnabled(True)
        self.create_button.setEnabled(True)
        self.cancel_button.setEnabled(False)

    def _settings_changed(self, values: dict) -> None:
        self.config.settings.update(values)
        self.runtime_changed.emit(values)

    def _display_diagnostics(self, result: dict) -> None:
        self.last_diagnostics = result.copy()
        for key, label in self.values.items():
            value = result.get(key, "Not found")
            if isinstance(value, bool):
                value = "Yes" if value else "No"
            label.setText(str(value))
        status = str(result.get("status", "Error"))
        readiness = str(result.get("readiness", "Diagnostics failed."))
        self.status.setText(f"{status}: {readiness}")
        self.status.setProperty("state", "ok" if status == "Ready" else "warning")
        self.status.style().unpolish(self.status)
        self.status.style().polish(self.status)
        errors = [result.get("python_error"), result.get("yolo_error"), result.get("package_error")]
        for error in errors:
            if error:
                self._append_log(f"Diagnostic note: {error}\n")
        self.runtime_changed.emit(result)

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def open_managed_folder(self) -> None:
        folder = self.manager.managed_runtime_folder()
        try:
            folder.mkdir(parents=True, exist_ok=True)
            os.startfile(folder)  # type: ignore[attr-defined]
        except OSError as exc:
            QMessageBox.warning(self, "Managed Runtime Folder", str(exc))

    def shutdown(self) -> None:
        if self._worker:
            self._worker.cancel()
        if self._thread:
            self._thread.quit()
            self._thread.wait(60000)
