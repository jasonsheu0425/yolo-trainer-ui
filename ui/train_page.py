from __future__ import annotations

import os
import shlex
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.results_reader import RUN_ARTIFACTS, scan_run_folder
from core.runtime_manager import RuntimeManager
from core.trainer_process import TrainerProcess
from ui.widgets import PageHeader, PathPicker, show_runtime_required


class TrainPage(QWidget):
    dataset_selected = Signal(str)
    results_found = Signal(str)
    runtime_required = Signal()

    PRESETS = {
        "Smoke Test": {"epochs": 1, "imgsz": 640, "batch": 4, "device": "0"},
        "Small Dataset Conservative": {"epochs": 80, "imgsz": 640, "batch": 8, "patience": 30, "cache": False},
        "Standard YOLOv8n": {"model": "yolov8n.pt", "epochs": 100, "imgsz": 640, "batch": 16, "patience": 50},
        "Higher Accuracy YOLOv8s": {"model": "yolov8s.pt", "epochs": 150, "imgsz": 960, "batch": 8, "patience": 50},
    }

    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.runtime_manager = RuntimeManager(config)
        self.runner = TrainerProcess(self)
        self.runner.output.connect(self._append_log)
        self.runner.state_changed.connect(self._set_running)
        self.runner.finished.connect(self._finished)
        self.runner.error.connect(self._show_error)

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
        layout.addWidget(PageHeader("Train", "設定 Ultralytics YOLO 參數，並在不凍結介面的情況下執行訓練。"))

        source_box = QGroupBox("Dataset & Model")
        source_layout = QVBoxLayout(source_box)
        self.dataset = PathPicker("Dataset YAML", "YAML (*.yaml *.yml)")
        self.dataset.path_changed.connect(self._dataset_changed)
        source_layout.addWidget(self.dataset)
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model"))
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolo11n.pt", "yolo11s.pt", "yolo26n.pt", "yolo26s.pt"])
        self.model.currentTextChanged.connect(self.update_preview)
        custom = QPushButton("Custom .pt…")
        custom.clicked.connect(self._select_model)
        model_row.addWidget(self.model, 1)
        model_row.addWidget(custom)
        source_layout.addLayout(model_row)
        layout.addWidget(source_box)

        preset_box = QGroupBox("Training Presets")
        preset_layout = QHBoxLayout(preset_box)
        preset_layout.addWidget(QLabel("Preset"))
        self.preset = QComboBox()
        self.preset.addItem("Select a preset...")
        self.preset.addItems(list(self.PRESETS))
        self.preset.currentTextChanged.connect(self.apply_preset)
        preset_layout.addWidget(self.preset, 1)
        preset_layout.addWidget(QLabel("Selecting a preset only fills the fields; it never starts training."))
        layout.addWidget(preset_box)

        params_box = QGroupBox("Training Parameters")
        grid = QGridLayout(params_box)
        self.task = QComboBox()
        self.task.addItems(["detect", "segment", "classify", "pose", "obb"])
        self.imgsz = self._spin(32, 4096, 640)
        self.epochs = self._spin(1, 100000, 100)
        self.batch = self._spin(-1, 4096, 16)
        self.device = QLineEdit("0")
        self.workers = self._spin(0, 128, 8)
        self.project = QLineEdit("runs/detect")
        self.name = QLineEdit("train_ui")
        self.patience = self._spin(0, 100000, 50)
        self.resume = QCheckBox("Resume")
        self.pretrained = QCheckBox("Pretrained")
        self.pretrained.setChecked(True)
        self.cache = QCheckBox("Cache")
        fields = [
            ("Task", self.task), ("Image size", self.imgsz), ("Epochs", self.epochs),
            ("Batch", self.batch), ("Device", self.device), ("Workers", self.workers),
            ("Project", self.project), ("Run name", self.name), ("Patience", self.patience),
        ]
        for index, (label, widget) in enumerate(fields):
            row, column = divmod(index, 3)
            grid.addWidget(QLabel(label), row * 2, column)
            grid.addWidget(widget, row * 2 + 1, column)
        checks = QHBoxLayout()
        checks.addWidget(self.resume)
        checks.addWidget(self.pretrained)
        checks.addWidget(self.cache)
        checks.addStretch()
        grid.addLayout(checks, 6, 0, 1, 3)
        layout.addWidget(params_box)

        advanced_box = QGroupBox("Advanced Parameters")
        advanced_layout = QVBoxLayout(advanced_box)
        self.advanced = QLineEdit()
        self.advanced.setPlaceholderText("例如：lr0=0.01 optimizer=auto cos_lr=True")
        advanced_layout.addWidget(self.advanced)
        layout.addWidget(advanced_box)

        preview_box = QGroupBox("Command Preview")
        preview_layout = QVBoxLayout(preview_box)
        self.preview = QLineEdit()
        self.preview.setReadOnly(True)
        preview_layout.addWidget(self.preview)
        layout.addWidget(preview_box)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start Training")
        self.start_button.setObjectName("primaryButton")
        self.stop_button = QPushButton("Stop Training")
        self.stop_button.setEnabled(False)
        open_button = QPushButton("Open Runs Folder")
        clear_button = QPushButton("Clear Log")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.runner.stop)
        open_button.clicked.connect(self.open_runs_folder)
        clear_button.clicked.connect(lambda: self.log.clear())
        for widget in (self.start_button, self.stop_button, open_button, clear_button):
            buttons.addWidget(widget)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(220)
        self.log.setObjectName("console")
        layout.addWidget(self.log)
        layout.addWidget(self._create_run_summary())

        self.apply_settings(self.config.settings)
        for widget in (self.task, self.imgsz, self.epochs, self.batch, self.device, self.workers, self.project, self.name, self.patience, self.resume, self.pretrained, self.cache, self.advanced):
            signal = getattr(widget, "textChanged", None) or getattr(widget, "valueChanged", None) or getattr(widget, "currentTextChanged", None) or getattr(widget, "toggled", None)
            if signal:
                signal.connect(self.update_preview)
        self.update_preview()

    @staticmethod
    def _spin(minimum: int, maximum: int, value: int) -> QSpinBox:
        box = QSpinBox()
        box.setRange(minimum, maximum)
        box.setValue(value)
        return box

    def apply_settings(self, values: dict) -> None:
        self.model.setCurrentText(str(values.get("default_model", "yolov8n.pt")))
        self.device.setText(str(values.get("default_device", "0")))
        self.project.setText(str(values.get("runs_folder", "runs/detect")))
        if values.get("last_run_folder"):
            self._update_run_summary(str(values["last_run_folder"]))
        self.update_preview()

    def apply_preset(self, name: str) -> None:
        values = self.PRESETS.get(name)
        if not values:
            return
        if "model" in values:
            self.model.setCurrentText(str(values["model"]))
        if "epochs" in values:
            self.epochs.setValue(int(values["epochs"]))
        if "imgsz" in values:
            self.imgsz.setValue(int(values["imgsz"]))
        if "batch" in values:
            self.batch.setValue(int(values["batch"]))
        if "device" in values:
            self.device.setText(str(values["device"]))
        if "patience" in values:
            self.patience.setValue(int(values["patience"]))
        if "cache" in values:
            self.cache.setChecked(bool(values["cache"]))
        self.update_preview()

    def _dataset_changed(self, value: str) -> None:
        self.dataset_selected.emit(value)
        self.update_preview()

    def _select_model(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇 YOLO model", "", "PyTorch model (*.pt)")
        if path:
            self.model.setCurrentText(path)

    def build_args(self) -> list[str]:
        args = [
            self.task.currentText(), "train",
            f"model={self.model.currentText().strip()}", f"data={self.dataset.path()}",
            f"imgsz={self.imgsz.value()}", f"epochs={self.epochs.value()}",
            f"batch={self.batch.value()}", f"device={self.device.text().strip()}",
            f"workers={self.workers.value()}", f"project={self.project.text().strip()}",
            f"name={self.name.text().strip()}", f"resume={self.resume.isChecked()}",
            f"pretrained={self.pretrained.isChecked()}", f"cache={self.cache.isChecked()}",
            f"patience={self.patience.value()}",
        ]
        if self.advanced.text().strip():
            args.extend(shlex.split(self.advanced.text().strip(), posix=False))
        return args

    def update_preview(self, *_args) -> None:
        if not hasattr(self, "preview"):
            return
        program = self.runtime_manager.yolo_command_for_preview()
        self.preview.setText(self.runner.preview(program, self.build_args()))

    def start_training(self) -> None:
        dataset = Path(self.dataset.path())
        if not dataset.is_file():
            QMessageBox.warning(self, "Train", "請選擇有效的 data.yaml。")
            return
        if not self.model.currentText().strip():
            QMessageBox.warning(self, "Train", "請選擇或輸入 model。")
            return
        try:
            args = self.build_args()
        except ValueError as exc:
            QMessageBox.warning(self, "Advanced Parameters", f"參數格式不正確：{exc}")
            return
        program = self.runtime_manager.resolve_yolo_command()
        if not program:
            if show_runtime_required(self):
                self.runtime_required.emit()
            return
        self.runner.start(program, args, Path.cwd())

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _finished(self, code: int, _status: int) -> None:
        self._append_log(f"\n程序結束，exit code = {code}\n")
        run_dir = self._latest_run_dir()
        if run_dir:
            self._update_run_summary(str(run_dir))
            self.config.save({"last_run_folder": str(run_dir.resolve())})
            artifacts = scan_run_folder(run_dir)
            results = artifacts.get("results.csv")
            if results:
                self.results_found.emit(str(results))

    def _create_run_summary(self) -> QGroupBox:
        box = QGroupBox("Last Run Summary")
        grid = QGridLayout(box)
        grid.addWidget(QLabel("Run folder"), 0, 0)
        self.run_folder_value = QLineEdit("Not found")
        self.run_folder_value.setReadOnly(True)
        grid.addWidget(self.run_folder_value, 0, 1)
        self.run_folder_open = QPushButton("Open Folder")
        self.run_folder_open.setEnabled(False)
        self.run_folder_open.clicked.connect(lambda: self._open_run_artifact("run_folder", True))
        grid.addWidget(self.run_folder_open, 0, 2, 1, 2)
        self.run_file_values: dict[str, QLineEdit] = {}
        self.run_file_buttons: dict[str, tuple[QPushButton, QPushButton]] = {}
        for row, name in enumerate(RUN_ARTIFACTS, 1):
            grid.addWidget(QLabel(name), row, 0)
            value = QLineEdit("Not found")
            value.setReadOnly(True)
            grid.addWidget(value, row, 1)
            open_file = QPushButton("Open File")
            open_folder = QPushButton("Open Folder")
            open_file.clicked.connect(lambda _checked=False, key=name: self._open_run_artifact(key, False))
            open_folder.clicked.connect(lambda _checked=False, key=name: self._open_run_artifact(key, True))
            grid.addWidget(open_file, row, 2)
            grid.addWidget(open_folder, row, 3)
            self.run_file_values[name] = value
            self.run_file_buttons[name] = (open_file, open_folder)
        self.last_run_artifacts = scan_run_folder("")
        return box

    def _update_run_summary(self, run_folder: str) -> None:
        artifacts = scan_run_folder(run_folder)
        self.last_run_artifacts = artifacts
        root = artifacts.get("run_folder")
        self.run_folder_value.setText(str(root) if root else "Not found")
        self.run_folder_open.setEnabled(root is not None)
        for name in RUN_ARTIFACTS:
            path = artifacts.get(name)
            self.run_file_values[name].setText(str(path) if path else "Not found")
            for button in self.run_file_buttons[name]:
                button.setEnabled(path is not None)

    def _open_run_artifact(self, name: str, folder: bool) -> None:
        path = self.last_run_artifacts.get(name)
        if path is None:
            return
        target = path if name == "run_folder" else (path.parent if folder else path)
        if target.exists():
            os.startfile(target)  # type: ignore[attr-defined]

    def _latest_run_dir(self) -> Path | None:
        project = Path(self.project.text().strip()).expanduser()
        if not project.is_absolute():
            project = Path.cwd() / project
        if not project.is_dir():
            return None
        name = self.name.text().strip()
        candidates = [path for path in project.glob(f"{name}*") if path.is_dir()]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None

    def open_runs_folder(self) -> None:
        path = Path(self.project.text().strip()).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        path.mkdir(parents=True, exist_ok=True)
        os.startfile(path)  # type: ignore[attr-defined]

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()

    def _show_error(self, message: str) -> None:
        self._append_log(f"ERROR: {message}\n")
        QMessageBox.critical(self, "程序錯誤", message)
