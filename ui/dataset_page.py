from __future__ import annotations

from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.dataset_checker import check_dataset
from ui.widgets import PageHeader, PathPicker, bind_text


class DatasetPage(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(PageHeader("dataset.title", "dataset.description"))
        self.yaml_picker = PathPicker("Dataset YAML", "YAML (*.yaml *.yml)")
        bind_text(self.yaml_picker.label, "common.dataset_yaml")
        layout.addWidget(self.yaml_picker)
        row = QHBoxLayout()
        self.check_button = QPushButton()
        bind_text(self.check_button, "dataset.check")
        self.check_button.setObjectName("primaryButton")
        self.check_button.clicked.connect(self.run_check)
        self.status = QLabel("尚未檢查")
        row.addWidget(self.check_button)
        row.addWidget(self.status)
        row.addStretch()
        layout.addLayout(row)

        self.tabs = QTabWidget()
        self.summary = QTableWidget(0, 2)
        self.summary.setHorizontalHeaderLabels(["項目", "值"])
        self.summary.horizontalHeader().setStretchLastSection(True)
        self.errors = self._text_tab()
        self.warnings = self._text_tab()
        self.class_counts = QTableWidget(0, 2)
        self.class_counts.setHorizontalHeaderLabels(["類別", "Instances"])
        self.class_counts.horizontalHeader().setStretchLastSection(True)
        self.tabs.addTab(self.summary, "Summary")
        self.tabs.addTab(self.errors, "Errors")
        self.tabs.addTab(self.warnings, "Warnings")
        self.tabs.addTab(self.class_counts, "Class Counts")
        layout.addWidget(self.tabs, 1)

    @staticmethod
    def _text_tab() -> QTextEdit:
        edit = QTextEdit()
        edit.setReadOnly(True)
        return edit

    def set_yaml_path(self, path: str) -> None:
        self.yaml_picker.set_path(path)

    def run_check(self) -> None:
        if not self.yaml_picker.path():
            QMessageBox.warning(self, "Dataset Check", "請先選擇 data.yaml。")
            return
        self.check_button.setEnabled(False)
        self.status.setText("檢查中…")
        try:
            result = check_dataset(self.yaml_picker.path())
        except Exception as exc:
            QMessageBox.critical(self, "檢查失敗", str(exc))
            self.status.setText("檢查失敗")
            return
        finally:
            self.check_button.setEnabled(True)
        self._fill_table(self.summary, result["summary"])
        self._fill_table(self.class_counts, result["class_counts"])
        self.errors.setPlainText("\n".join(result["errors"]) or "沒有錯誤。")
        self.warnings.setPlainText("\n".join(result["warnings"]) or "沒有警告。")
        errors = len(result["errors"])
        warnings = len(result["warnings"])
        self.status.setText(f"完成：{errors} errors，{warnings} warnings")
        self.tabs.setCurrentIndex(1 if errors else 0)

    @staticmethod
    def _fill_table(table: QTableWidget, values: dict) -> None:
        table.setRowCount(len(values))
        for row, (key, value) in enumerate(values.items()):
            table.setItem(row, 0, QTableWidgetItem(str(key)))
            table.setItem(row, 1, QTableWidgetItem(str(value)))
