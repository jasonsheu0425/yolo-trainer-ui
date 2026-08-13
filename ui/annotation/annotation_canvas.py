"""Zoomable QGraphicsView for image-space bounding-box interaction."""
from __future__ import annotations

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import QGraphicsPixmapItem, QGraphicsScene, QGraphicsView

from domain.annotation import AnnotationSource, BoundingBox, BoxMetadata, PixelBox, yolo_to_xyxy
from ui.annotation.bounding_box_item import BoundingBoxItem, MIN_BOX_PIXELS


class AnnotationCanvas(QGraphicsView):
    box_created = Signal(object)
    box_changed = Signal(int, object)
    selection_changed = Signal(int)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.scene().selectionChanged.connect(self._scene_selection_changed)
        self.setRenderHints(
            self.renderHints()
            | QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.mode = "select"
        self.image_size = (0, 0)
        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._image_path = ""
        self._cached_pixmap = QPixmap()
        self._draw_start: QPointF | None = None
        self._preview = None
        self._middle_pan = False
        self._pan_origin = QPoint()
        self._zoom = 1.0

    def load_image(
        self,
        path: str,
        boxes: list[BoundingBox],
        colors: dict[int, QColor],
        metadata: list[BoxMetadata] | None = None,
        class_names: dict[int, str] | None = None,
    ) -> bool:
        same_image = path == self._image_path and not self._cached_pixmap.isNull()
        pixmap = self._cached_pixmap if same_image else QPixmap(path)
        self.scene().clear()
        if pixmap.isNull():
            self.image_size = (0, 0)
            self._image_path = ""
            self._cached_pixmap = QPixmap()
            return False
        if not same_image:
            self._image_path = path
            self._cached_pixmap = pixmap
        self.image_size = (pixmap.width(), pixmap.height())
        self.scene().setSceneRect(0, 0, pixmap.width(), pixmap.height())
        self._pixmap_item = self.scene().addPixmap(pixmap)
        self._pixmap_item.setZValue(-10)
        for index, box in enumerate(boxes):
            pixel = yolo_to_xyxy(box, *self.image_size)
            details = metadata[index] if metadata and index < len(metadata) else BoxMetadata()
            class_name = (class_names or {}).get(box.class_id, str(box.class_id))
            if details.confidence is not None:
                label = f"{class_name} · {details.confidence:.0%} · AI"
            elif details.source is AnnotationSource.MODEL_ASSISTED:
                label = f"{class_name} · AI+Human"
            else:
                label = class_name
            item = BoundingBoxItem(
                index,
                QRectF(pixel.x1, pixel.y1, pixel.width, pixel.height),
                colors.get(box.class_id, QColor("#ef4444")),
                self.scene().sceneRect(),
                label,
            )
            item.geometry_committed.connect(self._item_changed)
            item.selected_box.connect(self._item_selected)
            self.scene().addItem(item)
        if not same_image:
            self.fit_image()
        return True

    def select_box(self, index: int) -> None:
        self.scene().clearSelection()
        for item in self.scene().items():
            if isinstance(item, BoundingBoxItem) and item.index == index:
                item.setSelected(True)
                break

    def _item_selected(self, index: int) -> None:
        for item in self.scene().selectedItems():
            if isinstance(item, BoundingBoxItem) and item.index != index:
                item.setSelected(False)
        self.selection_changed.emit(index)

    def _scene_selection_changed(self) -> None:
        if not any(
            isinstance(item, BoundingBoxItem)
            for item in self.scene().selectedItems()
        ):
            self.selection_changed.emit(-1)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        drag_mode = (
            QGraphicsView.DragMode.ScrollHandDrag
            if mode == "pan"
            else QGraphicsView.DragMode.NoDrag
        )
        self.setDragMode(drag_mode)

    def fit_image(self) -> None:
        if self.image_size[0]:
            self.fitInView(
                self.scene().sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
            )
            self._zoom = self.transform().m11()

    def actual_size(self) -> None:
        self.resetTransform()
        self._zoom = 1.0

    def wheelEvent(self, event: QWheelEvent) -> None:  # type: ignore[override]
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        target = self._zoom * factor
        if 0.1 <= target <= 8.0:
            self.scale(factor, factor)
            self._zoom = target
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton:
            self._middle_pan = True
            self._pan_origin = event.position().toPoint()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        if (
            self.mode == "draw"
            and event.button() == Qt.MouseButton.LeftButton
            and self.image_size[0]
        ):
            self._draw_start = self._clamped_scene(event.position().toPoint())
            self._preview = self.scene().addRect(
                QRectF(self._draw_start, self._draw_start),
                QPen(QColor("#22c55e"), 2.0),
            )
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if self._middle_pan:
            current = event.position().toPoint()
            delta = current - self._pan_origin
            self._pan_origin = current
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return
        if self._draw_start is not None and self._preview is not None:
            current = self._clamped_scene(event.position().toPoint())
            self._preview.setRect(QRectF(self._draw_start, current).normalized())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == Qt.MouseButton.MiddleButton and self._middle_pan:
            self._middle_pan = False
            self.unsetCursor()
            event.accept()
            return
        if self._draw_start is not None and self._preview is not None:
            rect = self._preview.rect().intersected(self.scene().sceneRect())
            self.scene().removeItem(self._preview)
            self._draw_start = None
            self._preview = None
            if rect.width() >= MIN_BOX_PIXELS and rect.height() >= MIN_BOX_PIXELS:
                self.box_created.emit(
                    PixelBox(rect.left(), rect.top(), rect.right(), rect.bottom())
                )
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def _clamped_scene(self, point: QPoint) -> QPointF:
        value = self.mapToScene(point)
        bounds = self.scene().sceneRect()
        return QPointF(
            min(max(value.x(), bounds.left()), bounds.right()),
            min(max(value.y(), bounds.top()), bounds.bottom()),
        )

    def _item_changed(self, index: int, rect: QRectF) -> None:
        self.box_changed.emit(
            index,
            PixelBox(rect.left(), rect.top(), rect.right(), rect.bottom()),
        )
