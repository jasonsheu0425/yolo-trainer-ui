from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
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
from core.dataset_builder import build_dataset, preview_dataset_build
from core.report_reader import CATEGORY_FILTERS, FILTER_MODES
from ui.widgets import PageHeader, PathPicker


class DatasetBuilderPage(QWidget):
    use_dataset_requested = Signal(str)

    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.last_result: dict | None = None

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
        layout.addWidget(PageHeader("Dataset Builder", "Create a new YOLO dataset version from a base dataset and selected hard cases."))

        paths = QGroupBox("Inputs & Output")
        paths_layout = QGridLayout(paths)
        self.base_yaml = PathPicker("Base Dataset data.yaml", "YAML (*.yaml *.yml)")
        self.hard_report = PathPicker("hard_cases_report.csv", "CSV (*.csv)")
        self.hard_folder = PathPicker("Hard cases folder", directory=True)
        self.output_folder = PathPicker("Output dataset folder", directory=True)
        self.output_folder.set_path(str(Path("datasets") / "built" / "war_tycoon_v_next"))
        paths_layout.addWidget(self.base_yaml, 0, 0)
        paths_layout.addWidget(self.hard_report, 0, 1)
        paths_layout.addWidget(self.hard_folder, 1, 0)
        paths_layout.addWidget(self.output_folder, 1, 1)
        layout.addWidget(paths)

        selection = QGroupBox("Hard Case Selection")
        selection_layout = QGridLayout(selection)
        self.category_checks: dict[str, QCheckBox] = {}
        for index, category in enumerate(CATEGORY_FILTERS):
            checkbox = QCheckBox(category)
            checkbox.setChecked(True)
            selection_layout.addWidget(checkbox, index // 4, index % 4)
            self.category_checks[category] = checkbox
        selection_layout.addWidget(QLabel("Filter Mode"), 2, 0)
        self.filter_mode = QComboBox()
        self.filter_mode.addItems(list(FILTER_MODES))
        self.filter_mode.setCurrentText("Primary or Any Flag")
        selection_layout.addWidget(self.filter_mode, 2, 1, 1, 3)
        layout.addWidget(selection)

        split_box = QGroupBox("Sample Split")
        split_layout = QHBoxLayout(split_box)
        self.train_ratio = self._ratio_spin(0.80)
        self.val_ratio = self._ratio_spin(0.20)
        self.test_ratio = self._ratio_spin(0.00)
        for label, spin in (("Train", self.train_ratio), ("Val", self.val_ratio), ("Test", self.test_ratio)):
            split_layout.addWidget(QLabel(label))
            split_layout.addWidget(spin)
        split_layout.addStretch()
        layout.addWidget(split_box)

        options_box = QGroupBox("Build Options")
        options_layout = QGridLayout(options_box)
        self.copy_base_images = self._checked("Copy original dataset images", True)
        self.copy_base_labels = self._checked("Copy original dataset labels", True)
        self.include_hard_cases = self._checked("Include selected hard cases", True)
        self.copy_hard_labels = self._checked("Copy labels if found", True)
        self.skip_without_labels = self._checked("Skip samples without labels", False)
        self.overwrite_output = self._checked("Overwrite output folder", False)
        for index, widget in enumerate((self.copy_base_images, self.copy_base_labels, self.include_hard_cases, self.copy_hard_labels, self.skip_without_labels, self.overwrite_output)):
            options_layout.addWidget(widget, index // 3, index % 3)
        layout.addWidget(options_box)

        buttons = QHBoxLayout()
        preview_button = QPushButton("Preview Build")
        preview_button.setObjectName("primaryButton")
        build_button = QPushButton("Build Dataset")
        self.open_output_button = QPushButton("Open Output Folder")
        self.open_output_button.setEnabled(False)
        clear_button = QPushButton("Clear Log")
        preview_button.clicked.connect(self.preview_build)
        build_button.clicked.connect(self.build)
        self.open_output_button.clicked.connect(lambda: self._open_path("output_folder", folder=True))
        clear_button.clicked.connect(lambda: self.log.clear())
        for button in (preview_button, build_button, self.open_output_button, clear_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.preview_summary = QLabel("Not previewed")
        self.preview_summary.setWordWrap(True)
        layout.addWidget(self.preview_summary)

        results = QGroupBox("Build Results")
        results_layout = QGridLayout(results)
        self.result_values: dict[str, QLineEdit] = {}
        result_fields = (
            ("output_folder", "Output dataset folder"),
            ("data_yaml", "New data.yaml"),
            ("report_csv", "dataset_build_report.csv"),
            ("summary_json", "dataset_build_summary.json"),
        )
        for row, (key, title) in enumerate(result_fields):
            results_layout.addWidget(QLabel(title), row, 0)
            value = QLineEdit("Not found")
            value.setReadOnly(True)
            results_layout.addWidget(value, row, 1)
            open_button = QPushButton("Open")
            open_button.clicked.connect(lambda _checked=False, field=key: self._open_path(field, folder=field == "output_folder"))
            results_layout.addWidget(open_button, row, 2)
            self.result_values[key] = value
        self.result_counts = QLabel("No build results")
        self.result_counts.setWordWrap(True)
        results_layout.addWidget(self.result_counts, len(result_fields), 0, 1, 3)
        self.use_train_button = QPushButton("Use This Dataset in Train Page")
        self.use_train_button.setEnabled(False)
        self.use_train_button.clicked.connect(self.use_in_train)
        results_layout.addWidget(self.use_train_button, len(result_fields) + 1, 0, 1, 3)
        layout.addWidget(results)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(160)
        self.log.setObjectName("console")
        layout.addWidget(self.log)

        self.hard_report.path_changed.connect(self._report_changed)
        self._load_recent_hard_cases()

    @staticmethod
    def _ratio_spin(value: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, 1.0)
        spin.setDecimals(2)
        spin.setSingleStep(0.05)
        spin.setValue(value)
        return spin

    @staticmethod
    def _checked(text: str, checked: bool) -> QCheckBox:
        checkbox = QCheckBox(text)
        checkbox.setChecked(checked)
        return checkbox

    def apply_settings(self, values: dict) -> None:
        self.config.settings.update(values)

    def set_hard_cases(self, report_path: str, folder: str) -> None:
        self.hard_report.set_path(report_path)
        self.hard_folder.set_path(folder)

    def _load_recent_hard_cases(self) -> None:
        runs_root = Path(str(self.config.get("runs_folder", "runs/detect"))).expanduser()
        if not runs_root.is_dir():
            return
        try:
            reports = list(runs_root.glob("*/hard_cases/hard_cases_report.csv"))
            if reports:
                latest = max(reports, key=lambda path: path.stat().st_mtime)
                self.set_hard_cases(str(latest), str(latest.parent))
        except OSError:
            return

    def _report_changed(self, report_path: str) -> None:
        try:
            path = Path(report_path)
            if path.is_file():
                self.hard_folder.set_path(str(path.parent))
        except (OSError, ValueError):
            return

    def _options(self) -> dict:
        return {
            "base_data_yaml": self.base_yaml.path(),
            "hard_cases_report": self.hard_report.path(),
            "hard_cases_folder": self.hard_folder.path(),
            "output_folder": self.output_folder.path(),
            "selected_categories": [category for category, checkbox in self.category_checks.items() if checkbox.isChecked()],
            "filter_mode": self.filter_mode.currentText(),
            "train_ratio": self.train_ratio.value(),
            "val_ratio": self.val_ratio.value(),
            "test_ratio": self.test_ratio.value(),
            "copy_original_images": self.copy_base_images.isChecked(),
            "copy_original_labels": self.copy_base_labels.isChecked(),
            "include_hard_cases": self.include_hard_cases.isChecked(),
            "copy_labels_if_found": self.copy_hard_labels.isChecked(),
            "skip_without_labels": self.skip_without_labels.isChecked(),
            "overwrite_output": self.overwrite_output.isChecked(),
        }

    def preview_build(self) -> None:
        preview = preview_dataset_build(self._options())
        self._display_preview(preview)

    def _display_preview(self, preview: dict) -> None:
        for warning in preview.get("warnings", []):
            self._append_log(f"WARNING: {warning}\n")
        for error in preview.get("errors", []):
            self._append_log(f"ERROR: {error}\n")
        split = preview.get("hard_case_split_counts", {})
        self.preview_summary.setText(
            f"Base images: {preview.get('base_images', 0)} | Base labels: {preview.get('base_labels', 0)} | "
            f"Selected hard cases: {preview.get('selected_hard_cases', 0)} | "
            f"Hard-case split train/val/test: {split.get('train', 0)}/{split.get('val', 0)}/{split.get('test', 0)} | "
            f"Skipped: {preview.get('skipped_samples', 0)} | Missing labels: {preview.get('missing_labels', 0)} | "
            f"Output exists: {preview.get('output_exists', False)} | Will overwrite: {preview.get('will_overwrite', False)}"
        )

    def build(self) -> None:
        result = build_dataset(self._options())
        self.last_result = result
        self._display_preview(result)
        if result.get("errors"):
            QMessageBox.warning(self, "Dataset Builder", "Dataset build did not complete. Check the log for details.")
        for key, value in self.result_values.items():
            path = result.get(key)
            value.setText(str(path) if path else "Not found")
        output = result.get("output_folder")
        data_yaml = result.get("data_yaml")
        self.open_output_button.setEnabled(bool(output and Path(output).is_dir()))
        self.use_train_button.setEnabled(bool(data_yaml and Path(data_yaml).is_file()))
        self.result_counts.setText(
            f"Train: {result.get('train_count', 0)} | Val: {result.get('val_count', 0)} | Test: {result.get('test_count', 0)} | "
            f"Hard cases copied: {result.get('total_hard_cases_copied', 0)} | Missing labels: {result.get('total_missing_labels', 0)} | "
            f"Empty labels created: {result.get('total_empty_labels_created', 0)}"
        )
        if data_yaml:
            self._append_log(f"Dataset build complete: {output}\nNew data.yaml: {data_yaml}\n")

    def use_in_train(self) -> None:
        if not self.last_result or not self.last_result.get("data_yaml"):
            QMessageBox.information(self, "Dataset Builder", "Build a dataset first.")
            return
        self.use_dataset_requested.emit(str(self.last_result["data_yaml"]))

    def _open_path(self, field: str, folder: bool = False) -> None:
        if not self.last_result or not self.last_result.get(field):
            QMessageBox.information(self, "Dataset Builder", "File or folder not found.")
            return
        path = Path(self.last_result[field])
        target = path if folder else path
        if target.exists():
            os.startfile(target)  # type: ignore[attr-defined]
        else:
            QMessageBox.information(self, "Dataset Builder", "File or folder not found.")

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()
