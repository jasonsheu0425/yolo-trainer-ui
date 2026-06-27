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
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.error_miner import export_hard_cases, scan_error_cases
from ui.widgets import PageHeader, PathPicker


class ErrorMiningPage(QWidget):
    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.scan_result: dict | None = None
        self.export_folder: Path | None = None

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
        layout.addWidget(PageHeader("Error Mining", "Collect low-confidence and incomplete prediction cases for dataset improvement."))

        paths = QGroupBox("Folders")
        paths_layout = QGridLayout(paths)
        self.run_folder = PathPicker("Predict or validation output folder", directory=True)
        self.source_folder = PathPicker("Original source folder (optional)", directory=True)
        self.labels_folder = PathPicker("Additional labels folder for copying (optional)", directory=True)
        self.output_folder = PathPicker("Hard cases output folder", directory=True)
        runs_folder = Path(str(config.get("runs_folder", "runs/detect"))).expanduser()
        default_output = runs_folder.parent / "hard_cases"
        self.output_folder.set_path(str(default_output))
        paths_layout.addWidget(self.run_folder, 0, 0)
        paths_layout.addWidget(self.source_folder, 0, 1)
        paths_layout.addWidget(self.labels_folder, 1, 0)
        paths_layout.addWidget(self.output_folder, 1, 1)
        layout.addWidget(paths)

        ground_truth = QGroupBox("Ground Truth Comparison")
        ground_truth_layout = QGridLayout(ground_truth)
        self.ground_truth_labels = PathPicker("Ground Truth Labels Folder", directory=True)
        self.class_names_yaml = PathPicker("Class Names Source (optional data.yaml)", "YAML (*.yaml *.yml)")
        self.enable_ground_truth = QCheckBox("Enable Ground Truth Comparison")
        self.iou_threshold = QDoubleSpinBox()
        self.iou_threshold.setRange(0.0, 1.0)
        self.iou_threshold.setDecimals(2)
        self.iou_threshold.setSingleStep(0.05)
        self.iou_threshold.setValue(0.50)
        ground_truth_layout.addWidget(self.ground_truth_labels, 0, 0)
        ground_truth_layout.addWidget(self.class_names_yaml, 0, 1)
        ground_truth_layout.addWidget(self.enable_ground_truth, 1, 0)
        iou_row = QHBoxLayout()
        iou_row.addWidget(QLabel("IoU Threshold"))
        iou_row.addWidget(self.iou_threshold)
        iou_row.addStretch()
        ground_truth_layout.addLayout(iou_row, 1, 1)
        layout.addWidget(ground_truth)

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
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        headers = [
            "Image", "Primary Category", "All Error Flags", "Predictions", "Ground Truth",
            "Min Confidence", "Max IoU", "Matched", "False Negative", "False Positive",
            "Class Mismatch", "Low IoU", "Image Path", "Prediction Label", "Ground Truth Label", "Notes",
        ]
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
        ground_truth = self.ground_truth_labels.path() or None
        if self.enable_ground_truth.isChecked() and not (ground_truth and Path(ground_truth).is_dir()):
            self.scan_result = None
            self.export_button.setEnabled(False)
            message = "Ground-truth comparison is enabled, but a valid labels folder was not provided."
            self._append_log(f"ERROR: {message}\n")
            QMessageBox.warning(self, "Error Mining", message)
            return
        self._append_log(f"Scanning: {run}\n")
        if not self.enable_ground_truth.isChecked():
            self._append_log("Ground-truth comparison disabled. Using confidence-based mining only.\n")
        try:
            self.scan_result = scan_error_cases(
                run,
                source_folder=source,
                labels_folder=labels,
                low_conf_threshold=self.low_confidence.value(),
                ground_truth_labels_folder=ground_truth,
                data_yaml=self.class_names_yaml.path() or None,
                iou_threshold=self.iou_threshold.value(),
                enable_ground_truth_comparison=self.enable_ground_truth.isChecked(),
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
        self.summary.setText(
            self._summary_text(result.get("records", []))
            + f"\nOutput folder: {self.export_folder or 'Not found'}"
        )

    def _show_scan_result(self) -> None:
        if self.scan_result is None:
            return
        records = self.scan_result.get("records", [])
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(records))
        for row, record in enumerate(records):
            confidence = record.get("min_confidence")
            max_iou = record.get("max_iou")
            values = [
                record.get("image_name", ""),
                record.get("primary_category", record.get("category", "unknown")),
                record.get("all_error_flags", "unknown"),
                record.get("detection_count", 0),
                record.get("ground_truth_count", 0),
                f"{confidence:.4f}" if isinstance(confidence, (int, float)) else "Not found",
                f"{max_iou:.4f}" if isinstance(max_iou, (int, float)) else "Not found",
                record.get("matched_count", 0),
                record.get("false_negative_count", 0),
                record.get("false_positive_count", 0),
                record.get("class_mismatch_count", 0),
                record.get("low_iou_count", 0),
                record.get("image_path", ""),
                record.get("prediction_label_path", "") or "Not found",
                record.get("ground_truth_label_path", "") or "Not found",
                record.get("notes", ""),
            ]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))
        self.table.setSortingEnabled(True)
        self.summary.setText(self._summary_text(records))
        for warning in self.scan_result.get("warnings", []):
            self._append_log(f"WARNING: {warning}\n")
        for error in self.scan_result.get("errors", []):
            self._append_log(f"ERROR: {error}\n")

    def _summary_text(self, records: list[dict]) -> str:
        totals = {
            "predictions": sum(int(record.get("detection_count", 0)) for record in records),
            "ground_truth": sum(int(record.get("ground_truth_count", 0)) for record in records),
            "true_positive": sum(int(record.get("true_positive_count", 0)) for record in records),
            "false_negative": sum(int(record.get("false_negative_count", 0)) for record in records),
            "false_positive": sum(int(record.get("false_positive_count", 0)) for record in records),
            "class_mismatch": sum(int(record.get("class_mismatch_count", 0)) for record in records),
            "low_iou": sum(int(record.get("low_iou_count", 0)) for record in records),
            "low_confidence": sum(int(record.get("low_confidence_count", 0)) for record in records),
        }
        flags = [flag for record in records for flag in str(record.get("all_error_flags", "")).split(";")]
        mode = (
            "Ground-truth IoU comparison enabled."
            if self.enable_ground_truth.isChecked()
            else "Ground-truth comparison disabled. Using confidence-based mining only."
        )
        return (
            f"{mode}\nImages: {len(records)} | predictions: {totals['predictions']} | ground truth: {totals['ground_truth']} | "
            f"TP: {totals['true_positive']} | FN: {totals['false_negative']} | FP: {totals['false_positive']} | "
            f"class mismatch: {totals['class_mismatch']} | low IoU: {totals['low_iou']} | "
            f"low confidence: {totals['low_confidence']} | no detection: {flags.count('no_detection')} | "
            f"no label: {flags.count('no_label_file')} | unknown: {flags.count('unknown')}"
        )

    def open_hard_cases_folder(self) -> None:
        if self.export_folder and self.export_folder.is_dir():
            os.startfile(self.export_folder)  # type: ignore[attr-defined]

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()
