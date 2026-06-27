from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.error_miner import CATEGORIES, export_hard_cases, scan_error_cases
from ui.widgets import PageHeader, PathPicker


class ErrorMiningPage(QWidget):
    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.scan_result: dict | None = None
        self.export_folder: Path | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(PageHeader("Error Mining", "Collect low-confidence and incomplete prediction cases for dataset improvement."))

        paths = QGroupBox("Folders")
        paths_layout = QGridLayout(paths)
        self.run_folder = PathPicker("Predict or validation output folder", directory=True)
        self.source_folder = PathPicker("Original source folder (optional)", directory=True)
        self.labels_folder = PathPicker("YOLO labels folder (optional)", directory=True)
        self.output_folder = PathPicker("Hard cases output folder", directory=True)
        runs_folder = Path(str(config.get("runs_folder", "runs/detect"))).expanduser()
        default_output = runs_folder.parent / "hard_cases"
        self.output_folder.set_path(str(default_output))
        paths_layout.addWidget(self.run_folder, 0, 0)
        paths_layout.addWidget(self.source_folder, 0, 1)
        paths_layout.addWidget(self.labels_folder, 1, 0)
        paths_layout.addWidget(self.output_folder, 1, 1)
        layout.addWidget(paths)

        options = QGroupBox("Mining Options")
        options_layout = QHBoxLayout(options)
        options_layout.addWidget(QLabel("Low confidence threshold"))
        self.low_confidence = QDoubleSpinBox()
        self.low_confidence.setRange(0.0, 1.0)
        self.low_confidence.setDecimals(2)
        self.low_confidence.setSingleStep(0.05)
        self.low_confidence.setValue(0.35)
        options_layout.addWidget(self.low_confidence)
        self.copy_images = QCheckBox("Copy images")
        self.copy_images.setChecked(True)
        self.copy_labels = QCheckBox("Copy labels if found")
        self.copy_labels.setChecked(True)
        self.create_csv = QCheckBox("Create report CSV")
        self.create_csv.setChecked(True)
        self.create_json = QCheckBox("Create summary JSON")
        self.create_json.setChecked(True)
        for widget in (self.copy_images, self.copy_labels, self.create_csv, self.create_json):
            options_layout.addWidget(widget)
        options_layout.addStretch()
        layout.addWidget(options)

        buttons = QHBoxLayout()
        self.scan_button = QPushButton("Scan")
        self.scan_button.setObjectName("primaryButton")
        self.export_button = QPushButton("Export Hard Cases")
        self.export_button.setEnabled(False)
        self.open_button = QPushButton("Open Hard Cases Folder")
        self.open_button.setEnabled(False)
        clear_button = QPushButton("Clear Log")
        self.scan_button.clicked.connect(self.scan)
        self.export_button.clicked.connect(self.export)
        self.open_button.clicked.connect(self.open_hard_cases_folder)
        clear_button.clicked.connect(lambda: self.log.clear())
        for button in (self.scan_button, self.export_button, self.open_button, clear_button):
            buttons.addWidget(button)
        buttons.addStretch()
        layout.addLayout(buttons)

        self.summary = QLabel("Not scanned")
        layout.addWidget(self.summary)
        headers = ["Image", "Category", "Min Confidence", "Detections", "Image Path", "Label Path", "Notes"]
        self.table = QTableWidget(0, len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        self.table.setSortingEnabled(True)
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(150)
        self.log.setObjectName("console")
        layout.addWidget(self.log)

    def apply_settings(self, values: dict) -> None:
        self.config.settings.update(values)

    def scan(self) -> None:
        run = Path(self.run_folder.path())
        if not run.is_dir():
            QMessageBox.warning(self, "Error Mining", "Select a valid predict or validation output folder.")
            return
        source = self.source_folder.path() or None
        labels = self.labels_folder.path() or None
        self._append_log(f"Scanning: {run}\n")
        try:
            self.scan_result = scan_error_cases(
                run,
                source_folder=source,
                labels_folder=labels,
                low_conf_threshold=self.low_confidence.value(),
            )
        except Exception as exc:
            self.scan_result = None
            self._append_log(f"ERROR: Unexpected scan failure: {exc}\n")
            return
        self._show_scan_result()
        self.export_button.setEnabled(self.scan_result is not None)

    def export(self) -> None:
        if self.scan_result is None:
            QMessageBox.information(self, "Error Mining", "Run Scan before exporting hard cases.")
            return
        output = self.output_folder.path()
        if not output:
            QMessageBox.warning(self, "Error Mining", "Select a hard cases output folder.")
            return
        self._append_log(f"Exporting hard cases to: {output}\n")
        try:
            result = export_hard_cases(
                self.scan_result,
                output,
                copy_images=self.copy_images.isChecked(),
                copy_labels_if_found=self.copy_labels.isChecked(),
                create_report_csv=self.create_csv.isChecked(),
                create_summary_json=self.create_json.isChecked(),
            )
        except Exception as exc:
            self._append_log(f"ERROR: Unexpected export failure: {exc}\n")
            return
        self.export_folder = result.get("output_folder")
        self.open_button.setEnabled(bool(self.export_folder and self.export_folder.is_dir()))
        for warning in result.get("warnings", []):
            self._append_log(f"WARNING: {warning}\n")
        for error in result.get("errors", []):
            self._append_log(f"ERROR: {error}\n")
        report = result.get("report_csv") or "Not found"
        summary = result.get("summary_json") or "Not found"
        self._append_log(f"Report CSV: {report}\nSummary JSON: {summary}\n")
        self.summary.setText(f"Exported {len(result.get('records', []))} record(s) to {self.export_folder or 'Not found'}")

    def _show_scan_result(self) -> None:
        if self.scan_result is None:
            return
        records = self.scan_result.get("records", [])
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            confidence = record.get("min_confidence")
            values = [
                record.get("image_name", ""),
                record.get("category", "unknown"),
                f"{confidence:.4f}" if isinstance(confidence, (int, float)) else "Not found",
                record.get("detection_count", 0),
                record.get("image_path", ""),
                record.get("label_path", "") or "Not found",
                record.get("notes", ""),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.setSortingEnabled(True)
        counts = {category: sum(1 for record in records if record.get("category") == category) for category in CATEGORIES}
        self.summary.setText(
            f"Images: {len(records)} | low confidence: {counts['low_confidence']} | "
            f"no detection: {counts['no_detection']} | no label: {counts['no_label_file']} | unknown: {counts['unknown']}"
        )
        for warning in self.scan_result.get("warnings", []):
            self._append_log(f"WARNING: {warning}\n")
        for error in self.scan_result.get("errors", []):
            self._append_log(f"ERROR: {error}\n")

    def open_hard_cases_folder(self) -> None:
        if self.export_folder and self.export_folder.is_dir():
            os.startfile(self.export_folder)  # type: ignore[attr-defined]

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()
