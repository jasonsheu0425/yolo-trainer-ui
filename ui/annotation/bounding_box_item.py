"""Interactive scene item that knows pixels and presentation, not YOLO text."""
from __future__ import annotations

from PySide6.QtCore import QObject, QPointF, QRectF, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPen
from PySide6.QtWidgets import QGraphicsItem, QGraphicsRectItem, QStyleOptionGraphicsItem, QWidget


HANDLE_SIZE = 10.0
MIN_BOX_PIXELS = 3.0


class BoundingBoxItem(QObject, QGraphicsRectItem):
    geometry_committed = Signal(int, object)
    selected_box = Signal(int)

    def __init__(
        self,
        index: int,
        rect: QRectF,
        color: QColor,
        scene_bounds: QRectF,
        label: str = "",
    ) -> None:
        QObject.__init__(self)
        QGraphicsRectItem.__init__(self, QRectF(0, 0, rect.width(), rect.height()))
        self.index = index
        self.scene_bounds = scene_bounds
        self.label = label
        self.setPos(rect.topLeft())
        self.setPen(QPen(color, 2.0))
        self.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 25)))
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self._resize_corner: str | None = None
        self._before = self.scene_rect()

    def scene_rect(self) -> QRectF:
        return self.mapRectToScene(self.rect())

    def itemChange(self, change, value):  # type: ignore[override]
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionChange and isinstance(value, QPointF):
            rect = self.rect()
            return QPointF(
                min(max(value.x(), self.scene_bounds.left()), self.scene_bounds.right() - rect.width()),
                min(max(value.y(), self.scene_bounds.top()), self.scene_bounds.bottom() - rect.height()),
            )
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged and bool(value):
            self.selected_box.emit(self.index)
        return super().itemChange(change, value)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        self._before = self.scene_rect()
        self._resize_corner = self._corner_at(event.pos())
        if self._resize_corner:
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if not self._resize_corner:
            super().mouseMoveEvent(event)
            return
        current = self.mapToScene(event.pos())
        old = self.scene_rect()
        left, top, right, bottom = old.left(), old.top(), old.right(), old.bottom()
        if "l" in self._resize_corner:
            left = min(current.x(), right - MIN_BOX_PIXELS)
        if "r" in self._resize_corner:
            right = max(current.x(), left + MIN_BOX_PIXELS)
        if "t" in self._resize_corner:
            top = min(current.y(), bottom - MIN_BOX_PIXELS)
        if "b" in self._resize_corner:
            bottom = max(current.y(), top + MIN_BOX_PIXELS)
        adjusted = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized().intersected(self.scene_bounds)
        if adjusted.width() >= MIN_BOX_PIXELS and adjusted.height() >= MIN_BOX_PIXELS:
            self.prepareGeometryChange()
            self.setPos(adjusted.topLeft())
            self.setRect(0, 0, adjusted.width(), adjusted.height())
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._resize_corner:
            self._resize_corner = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        current = self.scene_rect()
        if current != self._before:
            self.geometry_committed.emit(self.index, current)

    def paint(self, painter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:  # type: ignore[override]
        super().paint(painter, option, widget)
        if self.label:
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            metrics = painter.fontMetrics()
            text_rect = QRectF(metrics.boundingRect(self.label)).adjusted(-4, -2, 4, 2)
            text_rect.moveBottomLeft(self.rect().topLeft())
            painter.fillRect(text_rect, QColor(17, 24, 39, 220))
            painter.setPen(QColor("white"))
            painter.drawText(text_rect, self.label)
        if self.isSelected():
            painter.setBrush(QBrush(QColor("white")))
            painter.setPen(QPen(QColor("#111827"), 1))
            half = HANDLE_SIZE / 2
            for point in self.rect().topLeft(), self.rect().topRight(), self.rect().bottomLeft(), self.rect().bottomRight():
                painter.drawRect(QRectF(point.x() - half, point.y() - half, HANDLE_SIZE, HANDLE_SIZE))

    def _corner_at(self, point: QPointF) -> str | None:
        if not self.isSelected():
            return None
        radius = HANDLE_SIZE * 1.5
        corners = {"lt": self.rect().topLeft(), "rt": self.rect().topRight(), "lb": self.rect().bottomLeft(), "rb": self.rect().bottomRight()}
        return next((name for name, value in corners.items() if (value - point).manhattanLength() <= radius), None)
