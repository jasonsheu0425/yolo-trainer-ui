from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class PathPicker(QWidget):
    path_changed = Signal(str)

    def __init__(
        self,
        label: str,
        file_filter: str = "所有檔案 (*.*)",
        directory: bool = False,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.file_filter = file_filter
        self.directory = directory
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)
        layout.addWidget(QLabel(label))
        row = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setClearButtonEnabled(True)
        self.button = QPushButton("瀏覽…")
        self.button.clicked.connect(self.browse)
        self.edit.textChanged.connect(self.path_changed)
        row.addWidget(self.edit, 1)
        row.addWidget(self.button)
        layout.addLayout(row)

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, value: str) -> None:
        self.edit.setText(value)

    def browse(self) -> None:
        start = self.path()
        if start and not Path(start).exists():
            start = str(Path(start).parent)
        if self.directory:
            value = QFileDialog.getExistingDirectory(self, "選擇資料夾", start)
        else:
            value, _ = QFileDialog.getOpenFileName(self, "選擇檔案", start, self.file_filter)
        if value:
            self.set_path(value)


class PageHeader(QWidget):
    def __init__(self, title: str, description: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        title_label = QLabel(title)
        title_label.setObjectName("pageTitle")
        description_label = QLabel(description)
        description_label.setObjectName("pageDescription")
        description_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(description_label)

