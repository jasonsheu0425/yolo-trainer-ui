from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.results_reader import (
    VALIDATION_ARTIFACTS,
    parse_validation_log,
    persist_validation_metrics,
    read_validation_metrics,
    scan_validation_folder,
)
from core.runtime_manager import RuntimeManager
from core.validator_process import ValidatorProcess
from ui.widgets import PageHeader, PathPicker, WheelSafeComboBox, WheelSafeDoubleSpinBox, WheelSafeSpinBox, bind_text, set_tooltip, show_runtime_required


class ValidatePage(QWidget):
    runtime_required = Signal()

    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.runtime_manager = RuntimeManager(config)
        self.runner = ValidatorProcess(self)
        self.runner.output.connect(self._append_log)
        self.runner.state_changed.connect(self._set_running)
        self.runner.finished.connect(self._finished)
        self.runner.error.connect(self._show_error)
        self.output_folder: Path | None = None
        self._known_output_dirs: set[Path] = set()
        self.validation_artifacts = scan_validation_folder("")

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
        layout.addWidget(PageHeader("validate.title", "validate.description"))

        inputs = QGroupBox()
        bind_text(inputs, "validate.inputs")
        input_layout = QVBoxLayout(inputs)
        self.model = PathPicker("Model (.pt or .onnx)", "YOLO model (*.pt *.onnx)")
        self.data = PathPicker("Dataset YAML", "YAML (*.yaml *.yml)")
        bind_text(self.model.label, "common.model")
        bind_text(self.data.label, "common.dataset_yaml")
        input_layout.addWidget(self.model)
        input_layout.addWidget(self.data)
        layout.addWidget(inputs)

        options = QGroupBox()
        bind_text(options, "validate.parameters")
        grid = QGridLayout(options)
        self.task = WheelSafeComboBox()
        self.task.addItems(["detect", "segment", "pose", "obb", "classify"])
        self.imgsz = self._spin(32, 4096, 640)
        self.batch = self._spin(1, 4096, 8)
        self.device = QLineEdit(str(config.get("default_device", "0")))
        self.split = WheelSafeComboBox()
        self.split.addItems(["val", "test"])
        self.conf = QLineEdit()
        self.conf.setPlaceholderText("Optional")
        self.conf.setValidator(QDoubleValidator(0.0, 1.0, 4, self.conf))
        self.iou = WheelSafeDoubleSpinBox()
        self.iou.setRange(0.0, 1.0)
        self.iou.setDecimals(2)
        self.iou.setSingleStep(0.05)
        self.iou.setValue(0.70)
        self.project = QLineEdit(str(config.get("runs_folder", "runs/detect")))
        self.name = QLineEdit("val_ui")
        for widget, key in ((self.data, "tooltip.validate.dataset"), (self.split, "tooltip.validate.split"), (self.iou, "tooltip.predict.iou")):
            set_tooltip(widget, key)
        fields = [
            ("train.task", self.task), ("common.image_size", self.imgsz), ("common.batch", self.batch),
            ("common.device", self.device), ("validate.split", self.split), ("common.confidence", self.conf),
            ("common.iou", self.iou), ("common.project", self.project), ("common.run_name", self.name),
        ]
        for index, (key, widget) in enumerate(fields):
            row, column = divmod(index, 3)
            label = QLabel()
            bind_text(label, key)
            grid.addWidget(label, row * 2, column)
            grid.addWidget(widget, row * 2 + 1, column)
        checks = QHBoxLayout()
        self.plots = QCheckBox()
        bind_text(self.plots, "validate.plots")
        self.plots.setChecked(True)
        self.save_json = QCheckBox()
        self.save_txt = QCheckBox()
        bind_text(self.save_json, "validate.save_json")
        bind_text(self.save_txt, "validate.save_txt")
        checks.addWidget(self.plots)
        checks.addWidget(self.save_json)
        checks.addWidget(self.save_txt)
        checks.addStretch()
        grid.addLayout(checks, 6, 0, 1, 3)
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
        bind_text(self.start_button, "validate.start")
        self.start_button.setObjectName("primaryButton")
        self.stop_button = QPushButton()
        bind_text(self.stop_button, "validate.stop")
        self.stop_button.setEnabled(False)
        self.open_button = QPushButton()
        bind_text(self.open_button, "common.open_folder")
        self.open_button.setEnabled(False)
        clear_button = QPushButton()
        bind_text(clear_button, "common.clear_log")
        self.start_button.clicked.connect(self.start_validation)
        self.stop_button.clicked.connect(self.runner.stop)
        self.open_button.clicked.connect(self.open_output_folder)
        clear_button.clicked.connect(lambda: self.log.clear())
        for button in (self.start_button, self.stop_button, self.open_button, clear_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(190)
        self.log.setObjectName("console")
        layout.addWidget(self.log)
        layout.addWidget(self._build_metrics_box())
        layout.addWidget(self._build_artifacts_box())

        self.model.path_changed.connect(self.update_preview)
        self.data.path_changed.connect(self.update_preview)
        for widget in (self.task, self.split):
            widget.currentTextChanged.connect(self.update_preview)
        for widget in (self.imgsz, self.batch, self.iou):
            widget.valueChanged.connect(self.update_preview)
        for widget in (self.device, self.conf, self.project, self.name):
            widget.textChanged.connect(self.update_preview)
        for widget in (self.plots, self.save_json, self.save_txt):
            widget.toggled.connect(self.update_preview)
        self.update_preview()

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> WheelSafeSpinBox:
        spin = WheelSafeSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        return spin

    def _build_metrics_box(self) -> QGroupBox:
        box = QGroupBox()
        bind_text(box, "validate.metrics")
        grid = QGridLayout(box)
        self.metric_values: dict[str, QLabel] = {}
        for column, (key, title) in enumerate((("precision", "Precision"), ("recall", "Recall"), ("map50", "mAP50"), ("map50_95", "mAP50-95"))):
            grid.addWidget(QLabel(title), 0, column)
            value = QLabel("Not found")
            value.setObjectName("metricValue")
            grid.addWidget(value, 1, column)
            self.metric_values[key] = value
        grid.addWidget(QLabel("Output folder"), 2, 0)
        self.output_value = QLineEdit("Not found")
        self.output_value.setReadOnly(True)
        grid.addWidget(self.output_value, 2, 1, 1, 3)
        self.metrics_status = QLabel("Metrics file not found.")
        grid.addWidget(self.metrics_status, 3, 0, 1, 4)
        return box

    def _build_artifacts_box(self) -> QGroupBox:
        box = QGroupBox()
        bind_text(box, "validate.artifacts")
        grid = QGridLayout(box)
        self.artifact_values: dict[str, QLineEdit] = {}
        self.artifact_buttons: dict[str, tuple[QPushButton, QPushButton]] = {}
        for row, name in enumerate(VALIDATION_ARTIFACTS):
            grid.addWidget(QLabel(name), row, 0)
            value = QLineEdit("Not found")
            value.setReadOnly(True)
            grid.addWidget(value, row, 1)
            open_file = QPushButton()
            open_folder = QPushButton()
            bind_text(open_file, "common.open_file")
            bind_text(open_folder, "common.open_folder")
            open_file.setEnabled(False)
            open_folder.setEnabled(False)
            open_file.clicked.connect(lambda _checked=False, key=name: self._open_artifact(key, False))
            open_folder.clicked.connect(lambda _checked=False, key=name: self._open_artifact(key, True))
            grid.addWidget(open_file, row, 2)
            grid.addWidget(open_folder, row, 3)
            self.artifact_values[name] = value
            self.artifact_buttons[name] = (open_file, open_folder)
        return box

    def apply_settings(self, values: dict) -> None:
        self.config.settings.update(values)
        if "default_device" in values:
            self.device.setText(str(values["default_device"]))
        if "runs_folder" in values:
            self.project.setText(str(values["runs_folder"]))
        self.update_preview()

    def build_args(self) -> list[str]:
        args = [
            self.task.currentText(),
            "val",
            f"model={self.model.path()}",
            f"data={self.data.path()}",
            f"imgsz={self.imgsz.value()}",
            f"batch={self.batch.value()}",
            f"device={self.device.text().strip()}",
            f"split={self.split.currentText()}",
        ]
        if self.conf.text().strip():
            args.append(f"conf={self.conf.text().strip()}")
        args.extend(
            [
                f"iou={self.iou.value():.2f}",
                f"plots={self.plots.isChecked()}",
                f"save_json={self.save_json.isChecked()}",
                f"save_txt={self.save_txt.isChecked()}",
                f"project={self.project.text().strip()}",
                f"name={self.name.text().strip()}",
            ]
        )
        return args

    def update_preview(self, *_args) -> None:
        if not hasattr(self, "preview"):
            return
        program = self.runtime_manager.yolo_command_for_preview()
        self.preview.setText(self.runner.preview(program, self.build_args()))

    def start_validation(self) -> None:
        model = Path(self.model.path())
        if not model.is_file() or model.suffix.lower() not in {".pt", ".onnx"}:
            QMessageBox.warning(self, "Validation", "Select a valid .pt or .onnx model.")
            return
        data = Path(self.data.path())
        if not data.is_file() or data.suffix.lower() not in {".yaml", ".yml"}:
            QMessageBox.warning(self, "Validation", "Select a valid data.yaml file.")
            return
        if self.conf.text().strip():
            try:
                conf = float(self.conf.text())
            except ValueError:
                QMessageBox.warning(self, "Validation", "Confidence must be blank or a number from 0 to 1.")
                return
            if not 0.0 <= conf <= 1.0:
                QMessageBox.warning(self, "Validation", "Confidence must be blank or a number from 0 to 1.")
                return
        self.output_folder = None
        self.output_value.setText("Not found")
        self.open_button.setEnabled(False)
        self._clear_results()
        self._known_output_dirs = set(self._output_candidates())
        program = self.runtime_manager.resolve_yolo_command()
        if not program:
            if show_runtime_required(self):
                self.runtime_required.emit()
            return
        self.runner.start(program, self.build_args(), Path.cwd())

    def _project_folder(self) -> Path:
        project = Path(self.project.text().strip()).expanduser()
        return project if project.is_absolute() else (Path.cwd() / project).resolve()

    def _output_candidates(self) -> list[Path]:
        project = self._project_folder()
        if not project.is_dir():
            return []
        name = self.name.text().strip()
        try:
            return [path.resolve() for path in project.iterdir() if path.is_dir() and path.name.startswith(name)]
        except OSError:
            return []

    def _finished(self, code: int, _status: int) -> None:
        self._append_log(f"\nValidation finished, exit code = {code}\n")
        candidates = self._output_candidates()
        new_candidates = [path for path in candidates if path not in self._known_output_dirs]
        available = new_candidates or candidates
        if code == 0 and available:
            self.output_folder = max(available, key=lambda path: path.stat().st_mtime)
            self.output_value.setText(str(self.output_folder))
            self.open_button.setEnabled(True)
            self._load_results(self.output_folder)
        else:
            self.metrics_status.setText("Validation failed or output folder was not found.")

    def _load_results(self, folder: Path) -> None:
        log_text = self.log.toPlainText()
        metrics = read_validation_metrics(folder, log_text)
        log_metrics = parse_validation_log(log_text)
        saved_path, save_error = persist_validation_metrics(
            folder,
            log_metrics,
            model=self.model.path(),
            data=self.data.path(),
            split=self.split.currentText(),
            imgsz=self.imgsz.value(),
            batch=self.batch.value(),
            device=self.device.text().strip(),
        )
        if save_error:
            self._append_log(f"ERROR: {save_error}\n")
        self.validation_artifacts = scan_validation_folder(folder)
        for name in VALIDATION_ARTIFACTS:
            path = self.validation_artifacts.get(name)
            self.artifact_values[name].setText(str(path) if path else "Not found")
            for button in self.artifact_buttons[name]:
                button.setEnabled(path is not None)
        for key, label in self.metric_values.items():
            value = metrics.get(key)
            label.setText(f"{value:.4f}" if isinstance(value, (int, float)) else "Not found")
        metrics_message = str(metrics.get("message") or f"Metrics source: {metrics.get('source', '')}")
        save_message = f"Saved: {saved_path}" if saved_path else save_error
        self.metrics_status.setText(f"{metrics_message} {save_message}".strip())

    def _clear_results(self) -> None:
        self.validation_artifacts = scan_validation_folder("")
        for label in self.metric_values.values():
            label.setText("Not found")
        self.metrics_status.setText("Metrics file not found.")
        for name, value in self.artifact_values.items():
            value.setText("Not found")
            for button in self.artifact_buttons[name]:
                button.setEnabled(False)

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def open_output_folder(self) -> None:
        if self.output_folder and self.output_folder.is_dir():
            os.startfile(self.output_folder)  # type: ignore[attr-defined]

    def _open_artifact(self, name: str, folder: bool) -> None:
        path = self.validation_artifacts.get(name)
        if path is None:
            return
        target = path.parent if folder else path
        if target.exists():
            os.startfile(target)  # type: ignore[attr-defined]

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _show_error(self, message: str) -> None:
        self._append_log(f"ERROR: {message}\n")
        QMessageBox.critical(self, "Validation process error", message)
