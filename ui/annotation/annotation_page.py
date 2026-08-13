"""Annotation Editor page: intent collection and state rendering only."""
from __future__ import annotations

from collections.abc import Callable

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QAbstractSpinBox,
    QApplication,
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.config_manager import ConfigManager
from core.i18n_manager import get_i18n, tr
from core.runtime_manager import RuntimeManager
from domain.annotation import (
    AnnotationSource,
    AnnotationStatus,
    BoundingBox,
    ModelPrediction,
    PixelBox,
    xyxy_to_yolo,
)
from services.annotation_inference_service import (
    AnnotationInferenceService,
    InferenceState,
)
from services.annotation_service import AnnotationService
from ui.annotation.annotation_canvas import AnnotationCanvas
from ui.widgets import (
    PageHeader,
    PathPicker,
    WheelSafeComboBox,
    WheelSafeDoubleSpinBox,
    bind_combo_items,
    bind_text,
    set_tooltip,
)


STATUS_SYMBOLS = {
    AnnotationStatus.UNLABELED: "○",
    AnnotationStatus.LABELED: "●",
    AnnotationStatus.EMPTY: "–",
    AnnotationStatus.INVALID: "!",
    AnnotationStatus.MODIFIED: "*",
}
CLASS_COLORS = (
    "#ef4444", "#22c55e", "#3b82f6", "#f59e0b",
    "#a855f7", "#06b6d4", "#ec4899",
)
TEXT_INPUT_TYPES = (QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QAbstractSpinBox)


class AnnotationPage(QWidget):
    """Presentation for AnnotationService; never parses or writes YOLO labels."""

    runtime_requested = Signal()

    def __init__(
        self,
        config: ConfigManager,
        service: AnnotationService,
        inference: AnnotationInferenceService | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.config = config
        self.service = service
        self.inference = inference or AnnotationInferenceService(
            RuntimeManager(config), service
        )
        self.clipboard_box: BoundingBox | None = None
        self.active_class_id = 0
        self._rendering = False

        outer = QVBoxLayout(self)
        outer.setContentsMargins(24, 20, 24, 20)
        outer.addWidget(PageHeader("annotation.title", "annotation.description"))
        self._build_dataset_row(outer)
        self._build_model_assistance(outer)
        self._build_toolbar(outer)
        self._build_editor(outer)
        self._build_footer(outer)
        self._install_shortcuts()
        self._connect_inference()

        get_i18n().language_changed.connect(self._retranslate)
        self._retranslate()
        self._render_empty()

    def _build_model_assistance(self, outer: QVBoxLayout) -> None:
        self.assistance_group = QGroupBox()
        bind_text(self.assistance_group, "annotation.ai.title")
        layout = QVBoxLayout(self.assistance_group)
        model_row = QHBoxLayout()
        self.model_picker = PathPicker("", "YOLO Detection Model (*.pt)")
        bind_text(self.model_picker.label, "annotation.ai.model")
        self.model_picker.set_path(str(self.config.get("last_annotation_model", "")))
        set_tooltip(self.model_picker, "annotation.ai.tooltip.model")
        self.device_combo = WheelSafeComboBox()
        bind_combo_items(
            self.device_combo,
            [
                ("annotation.ai.device.auto", "auto"),
                ("annotation.ai.device.gpu0", "0"),
                ("annotation.ai.device.cpu", "cpu"),
            ],
        )
        selected_device = str(self.config.get("annotation_device", "auto"))
        self.device_combo.setCurrentIndex(
            max(self.device_combo.findData(selected_device), 0)
        )
        set_tooltip(self.device_combo, "annotation.ai.tooltip.device")
        self.device_label = QLabel()
        bind_text(self.device_label, "annotation.ai.device")
        self.confidence = WheelSafeDoubleSpinBox()
        self.confidence.setRange(0.01, 1.0)
        self.confidence.setSingleStep(0.05)
        self.confidence.setDecimals(2)
        self.confidence.setValue(float(self.config.get("annotation_confidence", 0.25)))
        set_tooltip(self.confidence, "annotation.ai.tooltip.confidence")
        self.confidence_label = QLabel()
        bind_text(self.confidence_label, "annotation.ai.confidence")
        self.load_model_button = QPushButton()
        bind_text(self.load_model_button, "annotation.ai.load")
        self.load_model_button.clicked.connect(lambda: self.load_model())
        model_row.addWidget(self.model_picker, 1)
        model_row.addWidget(self.device_label)
        model_row.addWidget(self.device_combo)
        model_row.addWidget(self.confidence_label)
        model_row.addWidget(self.confidence)
        model_row.addWidget(self.load_model_button)
        layout.addLayout(model_row)

        action_row = QHBoxLayout()
        self.inference_status = QLabel()
        self.inference_status.setWordWrap(True)
        self.predict_button = QPushButton()
        self.auto_annotate_button = QPushButton()
        self.cancel_inference_button = QPushButton()
        self.override_button = QPushButton()
        self.runtime_button = QPushButton()
        bind_text(self.predict_button, "annotation.ai.predict")
        bind_text(self.auto_annotate_button, "annotation.ai.auto_split")
        bind_text(self.cancel_inference_button, "annotation.ai.cancel")
        bind_text(self.override_button, "annotation.ai.override")
        bind_text(self.runtime_button, "annotation.ai.runtime")
        self.predict_button.clicked.connect(lambda: self.predict_current())
        self.auto_annotate_button.clicked.connect(lambda: self.confirm_auto_annotate())
        self.cancel_inference_button.clicked.connect(lambda: self.inference.cancel_batch())
        self.override_button.clicked.connect(lambda: self._override_mismatch())
        self.runtime_button.clicked.connect(self.runtime_requested.emit)
        set_tooltip(self.predict_button, "annotation.ai.tooltip.predict")
        set_tooltip(self.auto_annotate_button, "annotation.ai.tooltip.auto_split")
        for widget in (
            self.predict_button,
            self.auto_annotate_button,
            self.cancel_inference_button,
            self.override_button,
            self.runtime_button,
        ):
            action_row.addWidget(widget)
        action_row.addStretch()
        action_row.addWidget(self.inference_status, 1)
        layout.addLayout(action_row)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.setVisible(False)
        layout.addWidget(self.batch_progress)
        self.inference_log = QPlainTextEdit()
        self.inference_log.setReadOnly(True)
        self.inference_log.setMaximumHeight(90)
        self.inference_log.setVisible(False)
        layout.addWidget(self.inference_log)
        outer.addWidget(self.assistance_group)

    def _connect_inference(self) -> None:
        self.inference.state_changed.connect(self._inference_state_changed)
        self.inference.model_loaded.connect(self._model_loaded)
        self.inference.prediction_ready.connect(self._prediction_ready)
        self.inference.error.connect(self._inference_error)
        self.inference.log.connect(self._append_inference_log)
        self.inference.batch_progress.connect(self._batch_progressed)
        self.inference.batch_finished.connect(self._batch_finished)
        self._inference_state_changed(self.inference.state.value)

    def _build_dataset_row(self, outer: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.dataset_picker = PathPicker("", "YAML (*.yaml *.yml)")
        bind_text(self.dataset_picker.label, "annotation.dataset")
        self.dataset_picker.set_path(str(self.config.get("last_annotation_dataset", "")))
        self.open_button = QPushButton()
        bind_text(self.open_button, "annotation.open")
        self.open_button.setObjectName("primaryButton")
        self.open_button.clicked.connect(lambda: self.open_dataset())
        self.split_combo = WheelSafeComboBox()
        self.split_combo.currentIndexChanged.connect(self._split_changed)
        row.addWidget(self.dataset_picker, 1)
        row.addWidget(self.open_button)
        row.addWidget(self.split_combo)
        outer.addLayout(row)

    def _build_toolbar(self, outer: QVBoxLayout) -> None:
        self.toolbar = QToolBar()
        self.tool_actions: dict[str, QAction] = {}
        for mode in ("select", "draw", "pan"):
            action = self.toolbar.addAction("")
            action.setCheckable(True)
            action.setData(mode)
            action.triggered.connect(
                lambda _checked=False, value=mode: self.set_tool(value)
            )
            self.tool_actions[mode] = action
        self.tool_actions["select"].setChecked(True)
        self.toolbar.addSeparator()
        self.fit_action = self.toolbar.addAction("")
        self.fit_action.triggered.connect(lambda: self.canvas.fit_image())
        self.actual_action = self.toolbar.addAction("100%")
        self.actual_action.triggered.connect(lambda: self.canvas.actual_size())
        outer.addWidget(self.toolbar)

    def _build_editor(self, outer: QVBoxLayout) -> None:
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.image_summary = QLabel()
        left_layout.addWidget(self.image_summary)
        self.image_list = QListWidget()
        self.image_list.currentRowChanged.connect(self._image_selected)
        left_layout.addWidget(self.image_list, 1)

        self.canvas = AnnotationCanvas()
        self.canvas.box_created.connect(self._box_created)
        self.canvas.box_changed.connect(self._box_changed)
        self.canvas.selection_changed.connect(self._box_selected)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        self.class_title = QLabel()
        bind_text(self.class_title, "annotation.classes")
        right_layout.addWidget(self.class_title)
        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self._class_selected)
        right_layout.addWidget(self.class_list, 1)
        self.issue_label = QLabel()
        self.issue_label.setWordWrap(True)
        right_layout.addWidget(self.issue_label)
        issue_actions = QHBoxLayout()
        self.view_issues_button = QPushButton()
        self.repair_button = QPushButton()
        bind_text(self.view_issues_button, "annotation.view_issues")
        bind_text(self.repair_button, "annotation.repair_save")
        self.view_issues_button.clicked.connect(lambda: self._view_issues())
        self.repair_button.clicked.connect(lambda: self.save())
        issue_actions.addWidget(self.view_issues_button)
        issue_actions.addWidget(self.repair_button)
        right_layout.addLayout(issue_actions)

        splitter.addWidget(left)
        splitter.addWidget(self.canvas)
        splitter.addWidget(right)
        splitter.setSizes([240, 800, 220])
        outer.addWidget(splitter, 1)

    def _build_footer(self, outer: QVBoxLayout) -> None:
        row = QHBoxLayout()
        self.previous_button = QPushButton()
        self.save_button = QPushButton()
        self.next_button = QPushButton()
        bind_text(self.previous_button, "annotation.previous")
        bind_text(self.save_button, "annotation.save")
        bind_text(self.next_button, "annotation.next")
        self.previous_button.clicked.connect(lambda: self.previous_image())
        self.save_button.clicked.connect(lambda: self.save())
        self.next_button.clicked.connect(lambda: self.next_image())
        self.autosave = QCheckBox()
        bind_text(self.autosave, "annotation.autosave")
        self.autosave.setChecked(bool(self.config.get("annotation_autosave", True)))
        self.autosave.toggled.connect(
            lambda checked: self.config.save({"annotation_autosave": checked})
        )
        set_tooltip(self.autosave, "annotation.tooltip.autosave")
        self.status_label = QLabel()
        for widget in (
            self.previous_button, self.save_button, self.next_button, self.autosave,
        ):
            row.addWidget(widget)
        row.addStretch()
        row.addWidget(self.status_label)
        outer.addLayout(row)

    def open_dataset(self) -> None:
        if self._inference_busy():
            QMessageBox.warning(
                self, tr("common.warning"), tr("annotation.ai.busy_dataset")
            )
            return
        if not self._resolve_dirty():
            return
        try:
            dataset = self.service.open_dataset(self.dataset_picker.path())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, tr("common.warning"), str(exc))
            return
        self.config.save({"last_annotation_dataset": str(dataset.yaml_path)})
        self.split_combo.blockSignals(True)
        self.split_combo.clear()
        for split in dataset.splits:
            self.split_combo.addItem(tr(f"annotation.split.{split}"), split)
        self.split_combo.setCurrentIndex(self.split_combo.findData(self.service.split))
        self.split_combo.blockSignals(False)
        self.class_list.clear()
        for class_id, name in dataset.classes.items():
            self.class_list.addItem(f"{class_id}  {name}")
        if self.class_list.count():
            self.class_list.setCurrentRow(0)
        self._populate_images()
        self.image_list.setCurrentRow(self.service.index)
        self._render_document()
        self.inference.refresh_class_compatibility()

    def load_model(self) -> None:
        model = self.model_picker.path()
        device = str(self.device_combo.currentData())
        self.config.save(
            {
                "last_annotation_model": model,
                "annotation_device": device,
                "annotation_confidence": self.confidence.value(),
            }
        )
        self.inference.load_model(model, device)

    def predict_current(self) -> None:
        self.config.save({"annotation_confidence": self.confidence.value()})
        self.inference.predict_current(self.confidence.value())

    def _prediction_ready(
        self, predictions: list[ModelPrediction], _image: str
    ) -> None:
        if not predictions:
            self.inference_status.setText(tr("annotation.ai.no_detections"))
            self._inference_state_changed(self.inference.state.value)
            return
        document = self.service.document
        if document is None:
            return
        if not document.boxes:
            self._apply_prediction_choice(predictions, "add")
            return
        dialog = QMessageBox(self)
        dialog.setWindowTitle(tr("annotation.ai.existing_title"))
        dialog.setText(tr("annotation.ai.existing_message"))
        add_button = dialog.addButton(
            tr("annotation.ai.add"), QMessageBox.ButtonRole.AcceptRole
        )
        replace_button = dialog.addButton(
            tr("annotation.ai.replace"), QMessageBox.ButtonRole.DestructiveRole
        )
        dialog.addButton(QMessageBox.StandardButton.Cancel)
        dialog.exec()
        if dialog.clickedButton() is add_button:
            self._apply_prediction_choice(predictions, "add")
        elif dialog.clickedButton() is replace_button:
            self._apply_prediction_choice(predictions, "replace")

    def _apply_prediction_choice(
        self, predictions: list[ModelPrediction], mode: str
    ) -> None:
        try:
            self.service.apply_predictions(predictions, replace=mode == "replace")
        except ValueError as exc:
            QMessageBox.warning(self, tr("common.warning"), str(exc))
            return
        self._render_document(select=self.service.selected_index)
        self.inference_status.setText(
            tr("annotation.ai.predictions_added", count=len(predictions))
        )

    def confirm_auto_annotate(self) -> None:
        if self.service.document and self.service.document.dirty:
            QMessageBox.warning(
                self, tr("common.warning"), tr("annotation.ai.save_before_batch")
            )
            return
        plan = self.inference.batch_plan()
        summary = tr(
            "annotation.ai.confirm_batch",
            model=self.model_picker.path(),
            split=self.service.split,
            confidence=f"{self.confidence.value():.2f}",
            eligible=plan["eligible_count"],
            skipped=plan["existing_skipped"],
        )
        answer = QMessageBox.question(
            self,
            tr("annotation.ai.auto_split"),
            summary,
            QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Ok:
            self.inference.start_batch(self.confidence.value())

    def _override_mismatch(self) -> None:
        answer = QMessageBox.warning(
            self,
            tr("annotation.ai.class_mismatch"),
            tr("annotation.ai.override_warning"),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.inference.override_class_mismatch()
            self._inference_state_changed(self.inference.state.value)

    def _model_loaded(self, details: dict) -> None:
        compatibility = str(details.get("compatibility", "unknown"))
        device = str(details.get("actual_device", ""))
        cuda_available = details.get("cuda_available")
        self._sync_device_options(cuda_available)
        status = tr(
            f"annotation.ai.compatibility.{compatibility}",
            device=device,
        )
        if compatibility in {"id_match_name_mismatch", "class_count_mismatch"}:
            dataset_classes = self.service.dataset.classes if self.service.dataset else {}
            status += "\n" + tr(
                "annotation.ai.dataset_classes",
                classes=self._format_class_names(dataset_classes),
            )
            status += "\n" + tr(
                "annotation.ai.model_classes",
                classes=self._format_class_names(self.inference.model_classes),
            )
        if cuda_available is False:
            status += "\n" + tr("annotation.ai.cpu_warning")
        self.inference_status.setText(status)
        self._inference_state_changed(self.inference.state.value)

    def _sync_device_options(self, cuda_available: object) -> None:
        if cuda_available is not False:
            return
        gpu_index = self.device_combo.findData("0")
        if gpu_index >= 0:
            self.device_combo.removeItem(gpu_index)

    @staticmethod
    def _format_class_names(values: dict[int, str]) -> str:
        return ", ".join(f"{class_id} {name}" for class_id, name in sorted(values.items()))

    def _inference_state_changed(self, state_value: str) -> None:
        state = InferenceState(state_value)
        busy = state in {
            InferenceState.STARTING_WORKER,
            InferenceState.LOADING,
            InferenceState.BUSY,
            InferenceState.STOPPING,
        }
        batch_running = self.inference.batch_running
        compatible = self.inference.model_usable
        self.load_model_button.setEnabled(not busy)
        self.predict_button.setEnabled(
            compatible and self.service.document is not None and not busy
        )
        self.auto_annotate_button.setEnabled(
            compatible and self.service.dataset is not None and not busy
        )
        self.cancel_inference_button.setEnabled(batch_running)
        self.override_button.setVisible(
            self.inference.compatibility in {"id_match_name_mismatch", "unknown"}
            and state is InferenceState.READY
            and not self.inference.compatibility_override
        )
        runtime_missing = not bool(self.inference.runtime_info().get("available"))
        self.runtime_button.setVisible(runtime_missing)
        if state not in {InferenceState.READY, InferenceState.ERROR}:
            self.inference_status.setText(tr(f"annotation.ai.state.{state.value}"))
        self._set_navigation_enabled(not busy)

    def _inference_error(self, code: str, message: str) -> None:
        if code == "cuda_unavailable":
            self._sync_device_options(False)
        self.inference_status.setText(tr(f"annotation.ai.error.{code}"))
        self._append_inference_log(f"{code}: {message}")
        self._inference_state_changed(self.inference.state.value)

    def _append_inference_log(self, text: str) -> None:
        if not text:
            return
        self.inference_log.setVisible(True)
        self.inference_log.appendPlainText(text)
        scrollbar = self.inference_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _batch_progressed(self, progress: dict) -> None:
        eligible = int(progress.get("eligible", 0))
        processed = int(progress.get("processed", 0))
        self.batch_progress.setVisible(True)
        self.batch_progress.setMaximum(max(eligible, 1))
        self.batch_progress.setValue(processed)
        self.batch_progress.setFormat(
            tr(
                "annotation.ai.progress",
                processed=processed,
                eligible=eligible,
                created=progress.get("created", 0),
                no_detection=progress.get("no_detection", 0),
                skipped=progress.get("skipped", 0),
                errors=len(progress.get("errors", [])),
            )
        )
        self._inference_state_changed(self.inference.state.value)

    def _batch_finished(self, report: dict) -> None:
        self._populate_images(keep_row=True)
        if self.service.index >= 0:
            self.service.load_image(self.service.index)
            self._render_document()
        self.inference_status.setText(
            tr(
                "annotation.ai.batch_finished",
                status=report.get("status", ""),
                processed=report.get("processed", 0),
                created=report.get("created", 0),
                no_detection=report.get("no_detection", 0),
                remaining=report.get("remaining", 0),
            )
        )
        self._inference_state_changed(self.inference.state.value)

    def _set_navigation_enabled(self, enabled: bool) -> None:
        self.open_button.setEnabled(enabled)
        self.split_combo.setEnabled(enabled)
        self.image_list.setEnabled(enabled)
        self.previous_button.setEnabled(enabled and self.service.index > 0)
        self.next_button.setEnabled(
            enabled and self.service.index + 1 < len(self.service.images)
        )

    def _inference_busy(self) -> bool:
        return self.inference.state in {
            InferenceState.STARTING_WORKER,
            InferenceState.LOADING,
            InferenceState.BUSY,
            InferenceState.STOPPING,
        }

    def save(self, *, repair: bool = False) -> bool:
        document = self.service.document
        if document is None:
            return True
        if document.invalid_lines and not repair:
            answer = QMessageBox.question(
                self,
                tr("annotation.repair.title"),
                tr(
                    "annotation.repair.message",
                    valid=len(document.boxes),
                    invalid=len(document.invalid_lines),
                ),
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
            repair = True
        elif not document.dirty:
            return True
        try:
            self.service.save(repair=repair)
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, tr("common.error"), str(exc))
            return False
        if document.metadata_warning:
            QMessageBox.warning(
                self,
                tr("common.warning"),
                tr("annotation.metadata_save_warning"),
            )
        self._populate_images(keep_row=True)
        self._render_document()
        return True

    def previous_image(self) -> None:
        if self._inference_busy():
            return
        if self._before_navigation() and self.service.index > 0:
            self.service.load_image(self.service.index - 1)
            self._sync_current()

    def next_image(self) -> None:
        if self._inference_busy():
            return
        if (
            self._before_navigation()
            and self.service.index + 1 < len(self.service.images)
        ):
            self.service.load_image(self.service.index + 1)
            self._sync_current()

    def confirm_close(self) -> bool:
        return self._resolve_dirty()

    def set_tool(self, mode: str) -> None:
        for key, action in self.tool_actions.items():
            action.setChecked(key == mode)
        self.canvas.set_mode(mode)

    def delete_selected(self) -> None:
        if self.service.selected_index is not None:
            self.service.delete_box(self.service.selected_index)
            self._render_document()

    def copy_selected(self) -> None:
        if self.service.document and self.service.selected_index is not None:
            self.clipboard_box = self.service.document.boxes[self.service.selected_index]

    def paste(self) -> None:
        if self.clipboard_box and self.canvas.image_size[0]:
            try:
                index = self.service.paste_box(
                    self.clipboard_box, *self.canvas.image_size
                )
            except ValueError:
                return
            self._render_document(select=index)

    def undo(self) -> None:
        if self.service.undo():
            self._render_document(select=self.service.selected_index)

    def redo(self) -> None:
        if self.service.redo():
            self._render_document(select=self.service.selected_index)

    def _box_created(self, pixel: PixelBox) -> None:
        if self.service.dataset is None:
            return
        try:
            index = self.service.create_pixel_box(
                pixel, *self.canvas.image_size, self.active_class_id
            )
        except ValueError:
            return
        self._render_document(select=index)

    def _box_changed(self, index: int, pixel: PixelBox) -> None:
        document = self.service.document
        if document is None or not 0 <= index < len(document.boxes):
            return
        try:
            box = xyxy_to_yolo(
                pixel, *self.canvas.image_size, document.boxes[index].class_id
            )
            self.service.replace_box(index, box)
        except ValueError:
            return
        self._render_document(select=index)

    def _box_selected(self, index: int) -> None:
        self.service.selected_index = index if index >= 0 else None
        if index < 0:
            return
        document = self.service.document
        if document and 0 <= index < len(document.boxes):
            self._rendering = True
            self.class_list.setCurrentRow(document.boxes[index].class_id)
            self._rendering = False

    def _class_selected(self, row: int) -> None:
        if row < 0:
            return
        self.active_class_id = row
        if self._rendering or self.service.selected_index is None:
            return
        try:
            self.service.change_class(self.service.selected_index, row)
        except (IndexError, ValueError):
            return
        self._render_document(select=self.service.selected_index)

    def _split_changed(self) -> None:
        if self._inference_busy():
            return
        split = self.split_combo.currentData()
        if not split or self.service.dataset is None or split == self.service.split:
            return
        if not self._resolve_dirty():
            self.split_combo.setCurrentIndex(self.split_combo.findData(self.service.split))
            return
        self.service.select_split(str(split))
        self._populate_images()
        self.image_list.setCurrentRow(0)
        self._render_document()

    def _image_selected(self, row: int) -> None:
        if self._inference_busy():
            return
        if row < 0 or row == self.service.index or self.service.dataset is None:
            return
        if not self._resolve_dirty():
            self.image_list.setCurrentRow(self.service.index)
            return
        try:
            self.service.load_image(row)
        except (IndexError, OSError, ValueError) as exc:
            QMessageBox.warning(self, tr("common.warning"), str(exc))
            return
        self._render_document()

    def _before_navigation(self) -> bool:
        document = self.service.document
        if document is None or not document.dirty:
            return True
        return self.save() if self.autosave.isChecked() else self._resolve_dirty()

    def _resolve_dirty(self) -> bool:
        if not self.service.document or not self.service.document.dirty:
            return True
        if self.autosave.isChecked():
            return self.save()
        box = QMessageBox(self)
        box.setWindowTitle(tr("annotation.unsaved.title"))
        box.setText(tr("annotation.unsaved.message"))
        save = box.addButton(tr("annotation.save"), QMessageBox.ButtonRole.AcceptRole)
        discard = box.addButton(
            tr("annotation.discard"), QMessageBox.ButtonRole.DestructiveRole
        )
        box.addButton(QMessageBox.StandardButton.Cancel)
        box.exec()
        if box.clickedButton() is save:
            return self.save()
        if box.clickedButton() is discard:
            self.service.discard()
            self._render_document()
            return True
        return False

    def _populate_images(self, keep_row: bool = False) -> None:
        row = self.image_list.currentRow() if keep_row else self.service.index
        self.image_list.blockSignals(True)
        self.image_list.clear()
        for image in self.service.images:
            status = self.service.image_status(image)
            source = self.service.image_source(image)
            ai = (
                " AI"
                if source is AnnotationSource.MODEL_GENERATED
                else " AI+Human"
                if source is AnnotationSource.MODEL_ASSISTED
                else ""
            )
            item = QListWidgetItem(f"{STATUS_SYMBOLS[status]}{ai}  {image.name}")
            if status is AnnotationStatus.EMPTY:
                item.setToolTip(tr("annotation.tooltip.empty"))
            item.setData(Qt.ItemDataRole.UserRole, str(image))
            self.image_list.addItem(item)
        self.image_list.setCurrentRow(row)
        self.image_list.blockSignals(False)
        self._update_summary()

    def _sync_current(self) -> None:
        self._populate_images()
        self.image_list.setCurrentRow(self.service.index)
        self._render_document()

    def _render_document(self, select: int | None = None) -> None:
        document = self.service.document
        if document is None:
            self._render_empty()
            return
        classes = self.service.dataset.classes if self.service.dataset else {}
        colors = {
            class_id: QColor(CLASS_COLORS[class_id % len(CLASS_COLORS)])
            for class_id in classes
        }
        loaded = self.canvas.load_image(
            str(document.image_path),
            document.boxes,
            colors,
            document.box_metadata,
            classes,
        )
        if select is not None:
            self.canvas.select_box(select)
        self.issue_label.setText(
            tr("annotation.invalid_detail", count=len(document.invalid_lines))
            if document.invalid_lines else ""
        )
        if loaded:
            status = tr("annotation.status." + document.status.value)
            source = tr(f"annotation.source.{document.source.value}")
            self.status_label.setText(
                f"{document.image_path.name} — {status} · {source}"
            )
        else:
            self.status_label.setText(tr("annotation.image_unreadable"))
            current_item = self.image_list.item(self.service.index)
            if current_item is not None:
                current_item.setText(
                    f"!  {document.image_path.name} — {tr('annotation.image_unreadable')}"
                )
        has_issues = bool(document.invalid_lines)
        self.view_issues_button.setVisible(has_issues)
        self.repair_button.setVisible(has_issues)
        self.previous_button.setEnabled(self.service.index > 0)
        self.next_button.setEnabled(self.service.index + 1 < len(self.service.images))
        self.save_button.setEnabled(document.dirty or bool(document.invalid_lines))
        self._inference_state_changed(self.inference.state.value)
        self._update_summary()

    def _render_empty(self) -> None:
        self.image_list.clear()
        self.class_list.clear()
        self.image_summary.setText(tr("annotation.empty"))
        self.status_label.clear()
        self.save_button.setEnabled(False)
        self.previous_button.setEnabled(False)
        self.next_button.setEnabled(False)
        self.view_issues_button.setVisible(False)
        self.repair_button.setVisible(False)

    def _view_issues(self) -> None:
        document = self.service.document
        if document is None or not document.invalid_lines:
            return
        details = "\n".join(
            f"{line.line_number}: {line.error_id} — {line.raw_text}"
            for line in document.invalid_lines
        )
        QMessageBox.warning(self, tr("annotation.view_issues"), details)

    def _update_summary(self) -> None:
        if not self.service.dataset:
            return
        counts = self.service.status_summary()
        self.image_summary.setText(
            tr(
                "annotation.summary",
                total=len(self.service.images),
                labeled=counts["labeled"],
                empty=counts["empty"],
                unlabeled=counts["unlabeled"],
                invalid=counts["invalid"],
                modified=counts["modified"],
            )
        )

    def _install_shortcuts(self) -> None:
        shortcuts: tuple[tuple[str, Callable[[], object], bool], ...] = (
            ("Ctrl+S", lambda: self.save(), False),
            ("Ctrl+Z", self.undo, True),
            ("Ctrl+Y", self.redo, True),
            ("Delete", self.delete_selected, True),
            ("Ctrl+C", self.copy_selected, True),
            ("Ctrl+V", self.paste, True),
            ("A", self.previous_image, True),
            ("Left", self.previous_image, True),
            ("D", self.next_image, True),
            ("Right", self.next_image, True),
        )
        for sequence, callback, text_safe in shortcuts:
            self._add_shortcut(sequence, callback, text_safe=text_safe)
        for number in range(1, 10):
            self._add_shortcut(
                str(number),
                lambda class_id=number - 1: self._select_class_shortcut(class_id),
                text_safe=True,
            )

    def _add_shortcut(
        self,
        sequence: str,
        callback: Callable[[], object],
        *,
        text_safe: bool,
    ) -> None:
        action = QAction(self)
        action.setShortcut(QKeySequence(sequence))
        action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

        def invoke() -> None:
            if text_safe and isinstance(QApplication.focusWidget(), TEXT_INPUT_TYPES):
                return
            callback()

        action.triggered.connect(invoke)
        self.addAction(action)

    def _select_class_shortcut(self, class_id: int) -> None:
        if class_id < self.class_list.count():
            self.class_list.setCurrentRow(class_id)

    def _retranslate(self, _locale: str | None = None) -> None:
        for mode, action in self.tool_actions.items():
            action.setText(tr(f"annotation.tool.{mode}"))
            action.setToolTip(tr(f"annotation.tooltip.{mode}"))
        self.fit_action.setText(tr("annotation.fit"))
        if self.service.dataset:
            for index in range(self.split_combo.count()):
                split = self.split_combo.itemData(index)
                self.split_combo.setItemText(index, tr(f"annotation.split.{split}"))
            self._render_document(select=self.service.selected_index)
        if self.inference.state is InferenceState.READY:
            self._model_loaded(self.inference.model_summary())
        elif self.inference.cuda_available is False:
            self._sync_device_options(False)
