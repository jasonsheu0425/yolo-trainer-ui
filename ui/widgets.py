from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QPointF, Signal
from PySide6.QtGui import QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QAbstractScrollArea,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QSpinBox,
)

from core.i18n_manager import get_i18n, tr


def _forward_wheel_to_scroll_area(widget: QWidget, event: QWheelEvent) -> None:
    """Keep scroll wheel for the enclosing scroll area, never for a setting.

    Ignoring an event from a child is not consistently propagated on Windows,
    so forward an equivalent event to the nearest scroll viewport explicitly.
    """
    parent = widget.parentWidget()
    while parent is not None and not isinstance(parent, QAbstractScrollArea):
        parent = parent.parentWidget()
    if parent is None:
        event.ignore()
        return
    viewport = parent.viewport()
    global_pos = widget.mapToGlobal(event.position().toPoint())
    forwarded = QWheelEvent(
        QPointF(viewport.mapFromGlobal(global_pos)), event.globalPosition(),
        event.pixelDelta(), event.angleDelta(), event.buttons(), event.modifiers(),
        event.phase(), event.inverted(), event.source(),
    )
    QApplication.sendEvent(viewport, forwarded)
    event.accept()


class WheelSafeSpinBox(QSpinBox):
    """A numeric editor whose wheel gesture scrolls the page instead of values."""
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        _forward_wheel_to_scroll_area(self, event)


class WheelSafeDoubleSpinBox(QDoubleSpinBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        _forward_wheel_to_scroll_area(self, event)


class WheelSafeComboBox(QComboBox):
    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802
        # A visible popup is an intentional choice interaction; keep its native
        # wheel browsing.  Closed comboboxes only forward to page scrolling.
        if self.view().isVisible():
            super().wheelEvent(event)
        else:
            _forward_wheel_to_scroll_area(self, event)


def set_tooltip(widget: QWidget, key: str) -> None:
    """Bind a tooltip to the current app language without page-specific logic."""
    widget.setProperty("i18n_tooltip_key", key)
    widget.setToolTip(tr(key))
    get_i18n().language_changed.connect(lambda _locale: widget.setToolTip(tr(key)))


def bind_text(widget: QWidget, key: str) -> None:
    """Bind QLabel/QPushButton/QGroupBox text to one semantic key."""
    def apply(_locale: str | None = None) -> None:
        text = tr(key)
        if hasattr(widget, "setText"):
            widget.setText(text)  # type: ignore[attr-defined]
        elif hasattr(widget, "setTitle"):
            widget.setTitle(text)  # type: ignore[attr-defined]
    apply()
    get_i18n().language_changed.connect(apply)


def bind_combo_items(combo: QComboBox, items: list[tuple[str, object]]) -> None:
    """Localize display text while retaining immutable internal item data."""
    def apply(_locale: str | None = None) -> None:
        current = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        for key, data in items:
            combo.addItem(tr(key), data)
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)
    apply()
    get_i18n().language_changed.connect(apply)


def refresh_tooltips(root: QWidget) -> None:
    for widget in root.findChildren(QWidget):
        key = widget.property("i18n_tooltip_key")
        if isinstance(key, str):
            widget.setToolTip(tr(key))


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
        self.label = QLabel(label)
        layout.addWidget(self.label)
        row = QHBoxLayout()
        self.edit = QLineEdit()
        self.edit.setClearButtonEnabled(True)
        self.button = QPushButton(tr("common.browse"))
        self.button.clicked.connect(self.browse)
        self.edit.textChanged.connect(self.path_changed)
        row.addWidget(self.edit, 1)
        row.addWidget(self.button)
        layout.addLayout(row)
        get_i18n().language_changed.connect(self._retranslate)

    def _retranslate(self, _locale: str) -> None:
        self.button.setText(tr("common.browse"))

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, value: str) -> None:
        self.edit.setText(value)

    def browse(self) -> None:
        start = self.path()
        if start and not Path(start).exists():
            start = str(Path(start).parent)
        if self.directory:
            value = QFileDialog.getExistingDirectory(self, tr("dialog.choose_folder"), start)
        else:
            value, _ = QFileDialog.getOpenFileName(self, tr("dialog.choose_file"), start, self.file_filter)
        if value:
            self.set_path(value)


class PageHeader(QWidget):
    def __init__(self, title_key: str, description_key: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)
        self.title_key = title_key
        self.description_key = description_key
        self.title_label = QLabel()
        self.title_label.setObjectName("pageTitle")
        self.description_label = QLabel()
        self.description_label.setObjectName("pageDescription")
        self.description_label.setWordWrap(True)
        layout.addWidget(self.title_label)
        layout.addWidget(self.description_label)
        self._retranslate()
        get_i18n().language_changed.connect(self._retranslate)

    def _retranslate(self, _locale: str | None = None) -> None:
        self.title_label.setText(tr(self.title_key))
        self.description_label.setText(tr(self.description_key))


def show_runtime_required(parent: QWidget) -> bool:
    """Show a safe missing-runtime prompt and return whether navigation was requested."""
    message = QMessageBox(parent)
    message.setIcon(QMessageBox.Icon.Warning)
    message.setWindowTitle(tr("runtime.required.title"))
    message.setText(tr("runtime.required.text"))
    open_button = message.addButton(tr("runtime.required.open"), QMessageBox.ButtonRole.AcceptRole)
    message.addButton(QMessageBox.StandardButton.Cancel)
    message.exec()
    return message.clickedButton() is open_button
