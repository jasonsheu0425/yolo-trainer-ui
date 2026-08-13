from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import QCheckBox, QFormLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget

from core.config_manager import ConfigManager
from core.exporter_process import ExporterProcess
from core.runtime_manager import RuntimeManager
from ui.widgets import PageHeader, PathPicker, WheelSafeComboBox, WheelSafeSpinBox, bind_text, show_runtime_required


class ExportPage(QWidget):
    runtime_required = Signal()

    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.runtime_manager = RuntimeManager(config)
        self.runner = ExporterProcess(self)
        self.runner.output.connect(self._append_log)
        self.runner.state_changed.connect(self._set_running)
        self.runner.finished.connect(self._finished)
        self.runner.error.connect(lambda message: QMessageBox.critical(self, "Export", message))
        self.output_path: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(PageHeader("export.title", "export.description"))
        self.model = PathPicker("Model .pt", "PyTorch model (*.pt)")
        bind_text(self.model.label, "common.model")
        layout.addWidget(self.model)
        box = QGroupBox()
        bind_text(box, "export.options")
        form = QFormLayout(box)
        self.format = WheelSafeComboBox()
        self.format.addItems(["onnx", "engine", "openvino", "coreml", "tflite"])
        self.opset = WheelSafeSpinBox()
        self.opset.setRange(7, 21)
        self.opset.setValue(12)
        checks = QHBoxLayout()
        self.dynamic = QCheckBox("Dynamic")
        self.simplify = QCheckBox("Simplify")
        self.half = QCheckBox("Half")
        self.int8 = QCheckBox("INT8")
        self.nms = QCheckBox("NMS")
        for widget in (self.dynamic, self.simplify, self.half, self.int8, self.nms):
            checks.addWidget(widget)
        checks.addStretch()
        for text, widget in (("export.format", self.format), ("export.opset", self.opset), ("export.flags", checks)):
            label = QLabel()
            bind_text(label, text)
            form.addRow(label, widget)
        layout.addWidget(box)
        buttons = QHBoxLayout()
        self.export_button = QPushButton()
        bind_text(self.export_button, "export.start")
        self.export_button.setObjectName("primaryButton")
        self.stop_button = QPushButton()
        bind_text(self.stop_button, "export.stop")
        self.stop_button.setEnabled(False)
        self.open_button = QPushButton()
        bind_text(self.open_button, "common.open_folder")
        self.open_button.setEnabled(False)
        self.export_button.clicked.connect(self.export_model)
        self.stop_button.clicked.connect(self.runner.stop)
        self.open_button.clicked.connect(self.open_output)
        buttons.addWidget(self.export_button)
        buttons.addWidget(self.stop_button)
        buttons.addWidget(self.open_button)
        buttons.addStretch()
        layout.addLayout(buttons)
        self.status = QLabel("尚未匯出")
        layout.addWidget(self.status)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("console")
        layout.addWidget(self.log, 1)

    def apply_settings(self, values: dict) -> None:
        self.config.settings.update(values)

    def build_args(self) -> list[str]:
        args = ["export", f"model={self.model.path()}", f"format={self.format.currentText()}"]
        if self.format.currentText() == "onnx":
            args.append(f"opset={self.opset.value()}")
        for name, widget in (("dynamic", self.dynamic), ("simplify", self.simplify), ("half", self.half), ("int8", self.int8), ("nms", self.nms)):
            args.append(f"{name}={widget.isChecked()}")
        return args

    def export_model(self) -> None:
        path = Path(self.model.path())
        if not path.is_file() or path.suffix.lower() != ".pt":
            QMessageBox.warning(self, "Export", "請選擇有效的 .pt 模型。")
            return
        self.output_path = None
        self.open_button.setEnabled(False)
        self.status.setText("匯出中…")
        program = self.runtime_manager.resolve_yolo_command()
        if not program:
            self.status.setText("YOLO runtime not found.")
            if show_runtime_required(self):
                self.runtime_required.emit()
            return
        self.runner.start(program, self.build_args(), path.parent)

    def _set_running(self, running: bool) -> None:
        self.export_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _finished(self, code: int, _status: int) -> None:
        if code != 0:
            self.status.setText(f"匯出失敗（exit code {code}）")
            return
        model = Path(self.model.path())
        suffixes = {"onnx": ".onnx", "engine": ".engine", "coreml": ".mlpackage", "tflite": "_saved_model", "openvino": "_openvino_model"}
        suffix = suffixes[self.format.currentText()]
        candidate = model.with_suffix(suffix) if suffix.startswith(".") else model.parent / f"{model.stem}{suffix}"
        self.output_path = candidate
        self.status.setText(f"完成：{candidate}")
        self.open_button.setEnabled(True)

    def open_output(self) -> None:
        folder = self.output_path if self.output_path and self.output_path.is_dir() else (self.output_path.parent if self.output_path else Path(self.model.path()).parent)
        if folder.exists():
            os.startfile(folder)  # type: ignore[attr-defined]

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()
