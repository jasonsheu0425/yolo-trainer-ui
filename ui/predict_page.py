from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.predictor_process import PredictorProcess
from core.runtime_manager import RuntimeManager
from ui.widgets import PageHeader, PathPicker, WheelSafeComboBox, WheelSafeDoubleSpinBox, WheelSafeSpinBox, bind_combo_items, bind_text, set_tooltip, show_runtime_required


class PredictPage(QWidget):
    runtime_required = Signal()

    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.runtime_manager = RuntimeManager(config)
        self.runner = PredictorProcess(self)
        self.runner.output.connect(self._append_log)
        self.runner.state_changed.connect(self._set_running)
        self.runner.finished.connect(self._finished)
        self.runner.error.connect(self._show_error)
        self.output_folder: Path | None = None
        self._known_output_dirs: set[Path] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(PageHeader("predict.title", "predict.description"))

        inputs = QGroupBox()
        bind_text(inputs, "predict.inputs")
        input_layout = QVBoxLayout(inputs)
        self.model = PathPicker("Model (.pt or .onnx)", "YOLO model (*.pt *.onnx)")
        bind_text(self.model.label, "common.model")
        input_layout.addWidget(self.model)
        source_type_row = QHBoxLayout()
        source_type_label = QLabel()
        bind_text(source_type_label, "predict.source_type")
        source_type_row.addWidget(source_type_label)
        self.source_type = WheelSafeComboBox()
        bind_combo_items(self.source_type, [("predict.single_image", "single_image"), ("predict.image_folder", "image_folder"), ("predict.video", "video")])
        source_type_row.addWidget(self.source_type, 1)
        input_layout.addLayout(source_type_row)
        self.source = PathPicker("Source", "Images (*.jpg *.jpeg *.png *.bmp *.webp)")
        bind_text(self.source.label, "common.source")
        input_layout.addWidget(self.source)
        layout.addWidget(inputs)

        options = QGroupBox()
        bind_text(options, "predict.parameters")
        form = QFormLayout(options)
        self.imgsz = WheelSafeSpinBox()
        self.imgsz.setRange(32, 4096)
        self.imgsz.setValue(640)
        self.conf = self._ratio_spin(0.25)
        self.iou = self._ratio_spin(0.70)
        self.device = QLineEdit(str(config.get("default_device", "0")))
        self.save = QCheckBox()
        bind_text(self.save, "predict.save_outputs")
        self.save.setChecked(True)
        self.save_txt = QCheckBox()
        self.save_conf = QCheckBox()
        bind_text(self.save_txt, "predict.save_txt")
        bind_text(self.save_conf, "predict.save_conf")
        for widget, key in ((self.source, "tooltip.predict.source"), (self.source_type, "tooltip.predict.source"), (self.conf, "tooltip.predict.conf"), (self.iou, "tooltip.predict.iou"), (self.save_txt, "tooltip.predict.save_txt"), (self.save_conf, "tooltip.predict.save_conf")):
            set_tooltip(widget, key)
        label_options = QHBoxLayout()
        label_options.addWidget(self.save_txt)
        label_options.addWidget(self.save_conf)
        label_options.addStretch()
        for key, widget in (("common.image_size", self.imgsz), ("common.confidence", self.conf), ("common.iou", self.iou), ("common.device", self.device), ("common.save", self.save), ("predict.labels", label_options)):
            label = QLabel()
            bind_text(label, key)
            form.addRow(label, widget)
        layout.addWidget(options)

        preview_box = QGroupBox()
        bind_text(preview_box, "common.command_preview")
        preview_layout = QVBoxLayout(preview_box)
        self.preview = QLineEdit()
        self.preview.setReadOnly(True)
        preview_layout.addWidget(self.preview)
        layout.addWidget(preview_box)

        buttons = QHBoxLayout()
        self.start_button = QPushButton()
        bind_text(self.start_button, "predict.start")
        self.start_button.setObjectName("primaryButton")
        self.stop_button = QPushButton()
        bind_text(self.stop_button, "predict.stop")
        self.stop_button.setEnabled(False)
        self.open_button = QPushButton()
        bind_text(self.open_button, "common.open_folder")
        self.open_button.setEnabled(False)
        clear_button = QPushButton()
        bind_text(clear_button, "common.clear_log")
        self.start_button.clicked.connect(self.start_predict)
        self.stop_button.clicked.connect(self.runner.stop)
        self.open_button.clicked.connect(self.open_output_folder)
        clear_button.clicked.connect(self.log_clear)
        for button in (self.start_button, self.stop_button, self.open_button, clear_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)

        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output folder"))
        self.output_value = QLineEdit("Not found")
        self.output_value.setReadOnly(True)
        output_row.addWidget(self.output_value, 1)
        layout.addLayout(output_row)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setObjectName("console")
        layout.addWidget(self.log, 1)

        self.source_type.currentTextChanged.connect(self._source_type_changed)
        self.model.path_changed.connect(self.update_preview)
        self.source.path_changed.connect(self.update_preview)
        for widget in (self.imgsz, self.conf, self.iou):
            widget.valueChanged.connect(self.update_preview)
        self.device.textChanged.connect(self.update_preview)
        self.save.toggled.connect(self.update_preview)
        self.save_txt.toggled.connect(self.update_preview)
        self.save_conf.toggled.connect(self.update_preview)
        self._source_type_changed(self.source_type.currentText())
        self.update_preview()

    @staticmethod
    def _ratio_spin(value: float) -> WheelSafeDoubleSpinBox:
        spin = WheelSafeDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        return spin

    def apply_settings(self, values: dict) -> None:
        self.config.settings.update(values)
        if "default_device" in values:
            self.device.setText(str(values["default_device"]))
        self.update_preview()

    def _source_type_changed(self, source_type: str) -> None:
        source_kind = self.source_type.currentData()
        self.source.directory = source_kind == "image_folder"
        if source_kind == "video":
            self.source.file_filter = "Videos (*.mp4 *.avi *.mov *.mkv *.wmv *.m4v);;All files (*.*)"
        else:
            self.source.file_filter = "Images (*.jpg *.jpeg *.png *.bmp *.webp);;All files (*.*)"
        self.update_preview()

    def _project_folder(self) -> Path:
        project = Path(str(self.config.get("runs_folder", "runs/detect"))).expanduser()
        return project if project.is_absolute() else (Path.cwd() / project).resolve()

    def build_args(self) -> list[str]:
        return [
            "detect",
            "predict",
            f"model={self.model.path()}",
            f"source={self.source.path()}",
            f"imgsz={self.imgsz.value()}",
            f"conf={self.conf.value():.2f}",
            f"iou={self.iou.value():.2f}",
            f"device={self.device.text().strip()}",
            f"save={self.save.isChecked()}",
            f"save_txt={self.save_txt.isChecked()}",
            f"save_conf={self.save_conf.isChecked()}",
            f"project={self._project_folder()}",
            "name=predict_ui",
        ]

    def update_preview(self, *_args) -> None:
        if not hasattr(self, "preview"):
            return
        program = self.runtime_manager.yolo_command_for_preview()
        self.preview.setText(self.runner.preview(program, self.build_args()))

    def start_predict(self) -> None:
        model = Path(self.model.path())
        if not model.is_file() or model.suffix.lower() not in {".pt", ".onnx"}:
            QMessageBox.warning(self, "Predict", "Select a valid .pt or .onnx model.")
            return
        source = Path(self.source.path())
        expects_directory = self.source_type.currentData() == "image_folder"
        if (expects_directory and not source.is_dir()) or (not expects_directory and not source.is_file()):
            QMessageBox.warning(self, "Predict", "Select a valid source for the selected source type.")
            return
        self.output_folder = None
        self.output_value.setText("Not found")
        self.open_button.setEnabled(False)
        self._known_output_dirs = set(self._output_candidates())
        program = self.runtime_manager.resolve_yolo_command()
        if not program:
            if show_runtime_required(self):
                self.runtime_required.emit()
            return
        self.runner.start(program, self.build_args(), Path.cwd())

    def _output_candidates(self) -> list[Path]:
        project = self._project_folder()
        if not project.is_dir():
            return []
        return [path.resolve() for path in project.glob("predict_ui*") if path.is_dir()]

    def _finished(self, code: int, _status: int) -> None:
        self._append_log(f"\nPrediction finished, exit code = {code}\n")
        candidates = self._output_candidates()
        new_candidates = [path for path in candidates if path not in self._known_output_dirs]
        available = new_candidates or candidates
        if code == 0 and available:
            self.output_folder = max(available, key=lambda path: path.stat().st_mtime)
            self.output_value.setText(str(self.output_folder))
            self.open_button.setEnabled(True)
        else:
            self.output_value.setText("Not found")

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def open_output_folder(self) -> None:
        if self.output_folder and self.output_folder.is_dir():
            os.startfile(self.output_folder)  # type: ignore[attr-defined]

    def log_clear(self) -> None:
        self.log.clear()

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _show_error(self, message: str) -> None:
        self._append_log(f"ERROR: {message}\n")
        QMessageBox.critical(self, "Predict process error", message)
