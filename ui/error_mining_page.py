from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
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
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.error_miner import export_hard_cases, scan_error_cases
from core.report_reader import CATEGORY_FILTERS, filter_report, read_hard_cases_report, summarize_report
from ui.widgets import PageHeader, PathPicker


class ImagePreviewLabel(QLabel):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__("Image not selected", parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(420, 240)
        self._source_pixmap: QPixmap | None = None

    def set_image(self, path: Path | None) -> bool:
        if path is None or not path.is_file():
            self._source_pixmap = None
            self.clear()
            self.setText("Image not found")
            return False
        pixmap = QPixmap(str(path))
        if pixmap.isNull():
            self._source_pixmap = None
            self.clear()
            self.setText("Image not found")
            return False
        self._source_pixmap = pixmap
        self._update_scaled_pixmap()
        return True

    def _update_scaled_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        self.setPixmap(
            self._source_pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_scaled_pixmap()


class ErrorMiningPage(QWidget):
    def __init__(self, config: ConfigManager, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.scan_result: dict | None = None
        self.export_folder: Path | None = None
        self.report_rows: list[dict[str, str]] = []
        self.filtered_report_rows: list[dict[str, str]] = []
        self.selected_report_row: dict[str, str] | None = None

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
        layout.addWidget(self._build_report_viewer())
        self.output_folder.path_changed.connect(self._suggest_report_path)

    def apply_settings(self, values: dict) -> None:
        self.config.settings.update(values)

    def _build_report_viewer(self) -> QGroupBox:
        box = QGroupBox("Error Mining Report Viewer")
        layout = QVBoxLayout(box)
        self.report_picker = PathPicker("hard_cases_report.csv", "CSV (*.csv)")
        layout.addWidget(self.report_picker)

        controls = QHBoxLayout()
        load_button = QPushButton("Load Report")
        load_button.setObjectName("primaryButton")
        load_button.clicked.connect(self.load_report)
        controls.addWidget(load_button)
        controls.addWidget(QLabel("Category"))
        self.report_category = QComboBox()
        self.report_category.addItems(["All", *CATEGORY_FILTERS])
        controls.addWidget(self.report_category)
        self.report_search = QLineEdit()
        self.report_search.setPlaceholderText("Search image, category, flags, or notes")
        controls.addWidget(self.report_search, 1)
        layout.addLayout(controls)

        self.report_summary = QLabel("Report not loaded")
        self.report_summary.setWordWrap(True)
        layout.addWidget(self.report_summary)
        self.report_table_fields = (
            "image_name",
            "primary_category",
            "all_error_flags",
            "detection_count",
            "ground_truth_count",
            "min_confidence",
            "max_iou",
            "false_negative_count",
            "false_positive_count",
            "class_mismatch_count",
            "low_iou_count",
            "low_confidence_count",
            "copied_to",
            "notes",
        )
        self.report_table = QTableWidget(0, len(self.report_table_fields))
        self.report_table.setHorizontalHeaderLabels(list(self.report_table_fields))
        self.report_table.setMinimumHeight(240)
        self.report_table.horizontalHeader().setStretchLastSection(True)
        self.report_table.itemSelectionChanged.connect(self._report_row_selected)
        layout.addWidget(self.report_table)

        preview_row = QHBoxLayout()
        self.image_preview = ImagePreviewLabel()
        preview_row.addWidget(self.image_preview, 1)
        self.report_details = QTextEdit()
        self.report_details.setReadOnly(True)
        self.report_details.setMinimumHeight(240)
        preview_row.addWidget(self.report_details, 1)
        layout.addLayout(preview_row)

        actions = QHBoxLayout()
        open_image = QPushButton("Open Image")
        open_image_folder = QPushButton("Open Image Folder")
        open_prediction = QPushButton("Open Prediction Label")
        open_ground_truth = QPushButton("Open Ground Truth Label")
        open_hard_cases = QPushButton("Open Hard Cases Folder")
        open_image.clicked.connect(self.open_report_image)
        open_image_folder.clicked.connect(self.open_report_image_folder)
        open_prediction.clicked.connect(lambda: self.open_report_label("prediction_label_path"))
        open_ground_truth.clicked.connect(lambda: self.open_report_label("ground_truth_label_path"))
        open_hard_cases.clicked.connect(self.open_report_hard_cases_folder)
        for button in (open_image, open_image_folder, open_prediction, open_ground_truth, open_hard_cases):
            actions.addWidget(button)
        actions.addStretch()
        layout.addLayout(actions)

        self.report_category.currentTextChanged.connect(self._apply_report_filter)
        self.report_search.textChanged.connect(self._apply_report_filter)
        return box

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
        if result.get("report_csv"):
            self.report_picker.set_path(str(result["report_csv"]))
            self.load_report()

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

    def _suggest_report_path(self, output_folder: str) -> None:
        try:
            candidate = Path(output_folder) / "hard_cases_report.csv"
            if candidate.is_file():
                self.report_picker.set_path(str(candidate))
        except (OSError, ValueError):
            return

    def load_report(self) -> None:
        rows, error = read_hard_cases_report(self.report_picker.path())
        if error:
            self.report_rows = []
            self.filtered_report_rows = []
            self.selected_report_row = None
            self.report_table.setRowCount(0)
            self.report_summary.setText(error)
            self.image_preview.set_image(None)
            self.report_details.clear()
            self._append_log(f"ERROR: {error}\n")
            return
        self.report_rows = rows
        self._apply_report_filter()
        self._append_log(f"Loaded report: {self.report_picker.path()} ({len(rows)} row(s))\n")

    def _apply_report_filter(self, *_args) -> None:
        self.filtered_report_rows = filter_report(
            self.report_rows,
            self.report_category.currentText(),
            self.report_search.text(),
        )
        self.report_table.setRowCount(len(self.filtered_report_rows))
        for row_index, row in enumerate(self.filtered_report_rows):
            for column, field in enumerate(self.report_table_fields):
                self.report_table.setItem(row_index, column, QTableWidgetItem(str(row.get(field, ""))))
        summary = summarize_report(self.report_rows)
        counts = " | ".join(f"{category}: {summary[category]}" for category in CATEGORY_FILTERS)
        self.report_summary.setText(
            f"Total rows: {summary['total_rows']} | Filtered rows: {len(self.filtered_report_rows)}\n{counts}"
        )
        if self.filtered_report_rows:
            self.report_table.selectRow(0)
        else:
            self.selected_report_row = None
            self.image_preview.set_image(None)
            self.report_details.clear()

    def _report_row_selected(self) -> None:
        row_index = self.report_table.currentRow()
        if not 0 <= row_index < len(self.filtered_report_rows):
            self.selected_report_row = None
            self.image_preview.set_image(None)
            self.report_details.clear()
            return
        self.selected_report_row = self.filtered_report_rows[row_index]
        self.image_preview.set_image(self._selected_image_path())
        fields = (
            "image_name",
            "primary_category",
            "all_error_flags",
            "min_confidence",
            "max_iou",
            "copied_to",
            "image_path",
            "prediction_label_path",
            "ground_truth_label_path",
        )
        self.report_details.setPlainText(
            "\n".join(f"{field}: {self.selected_report_row.get(field, '')}" for field in fields)
        )

    def _selected_image_path(self) -> Path | None:
        if self.selected_report_row is None:
            return None
        for field in ("copied_to", "image_path"):
            path = self._existing_report_path(field)
            if path is not None:
                return path
        return None

    def _existing_report_path(self, field: str) -> Path | None:
        if self.selected_report_row is None:
            return None
        raw = str(self.selected_report_row.get(field, "")).strip()
        if not raw:
            return None
        try:
            path = Path(raw).expanduser()
            return path.resolve() if path.is_file() else None
        except (OSError, ValueError):
            return None

    def open_report_image(self) -> None:
        path = self._selected_image_path()
        if path is None:
            QMessageBox.information(self, "Report Viewer", "Image not found.")
            return
        os.startfile(path)  # type: ignore[attr-defined]

    def open_report_image_folder(self) -> None:
        path = self._selected_image_path()
        if path is None or not path.parent.is_dir():
            QMessageBox.information(self, "Report Viewer", "Image folder not found.")
            return
        os.startfile(path.parent)  # type: ignore[attr-defined]

    def open_report_label(self, field: str) -> None:
        path = self._existing_report_path(field)
        if path is None:
            QMessageBox.information(self, "Report Viewer", "Label not found.")
            return
        os.startfile(path)  # type: ignore[attr-defined]

    def open_report_hard_cases_folder(self) -> None:
        candidates: list[Path] = []
        if self.report_picker.path():
            candidates.append(Path(self.report_picker.path()).expanduser().parent)
        if self.export_folder:
            candidates.append(self.export_folder)
        if self.output_folder.path():
            candidates.append(Path(self.output_folder.path()).expanduser())
        for path in candidates:
            try:
                if path.is_dir():
                    os.startfile(path)  # type: ignore[attr-defined]
                    return
            except (OSError, ValueError):
                continue
        QMessageBox.information(self, "Report Viewer", "Hard cases folder not found.")

    def open_hard_cases_folder(self) -> None:
        if self.export_folder and self.export_folder.is_dir():
            os.startfile(self.export_folder)  # type: ignore[attr-defined]

    def _append_log(self, text: str) -> None:
        cursor = self.log.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        self.log.setTextCursor(cursor)
        self.log.ensureCursorVisible()
