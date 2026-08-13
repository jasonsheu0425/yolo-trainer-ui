"""Guided Simple Mode facade over the existing Dataset Check and Train page."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QScrollArea, QTextEdit, QVBoxLayout, QWidget,
)

from core.config_manager import ConfigManager
from core.dataset_checker import check_dataset
from core.i18n_manager import get_i18n, tr
from core.runtime_manager import RuntimeManager
from core.training_profiles import MODEL_PROFILES, TRAINING_PROFILES, build_simple_profile, matches_profile
from ui.widgets import PageHeader, PathPicker, WheelSafeComboBox, bind_combo_items, bind_text, set_tooltip


class SimpleModePage(QWidget):
    """One-page beginner flow; the TrainPage remains the sole executor."""

    advanced_requested = Signal()
    results_requested = Signal()
    dataset_details_requested = Signal()
    runtime_requested = Signal()

    def __init__(self, config: ConfigManager, train_page, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        self.train_page = train_page
        self.runtime_manager = RuntimeManager(config)
        self.checked_dataset: str | None = None
        self.check_result: dict | None = None

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        outer.addWidget(scroll)
        body = QWidget()
        scroll.setWidget(body)
        layout = QVBoxLayout(body)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.addWidget(PageHeader("simple.title", "simple.description"))

        self.steps = QLabel()
        self.steps.setWordWrap(True)
        self.steps.setObjectName("pageDescription")
        layout.addWidget(self.steps)

        dataset_box = QGroupBox()
        bind_text(dataset_box, "simple.dataset.group")
        dataset_layout = QVBoxLayout(dataset_box)
        self.dataset = PathPicker("", "YAML (*.yaml *.yml)")
        bind_text(self.dataset.label, "simple.dataset.path")
        set_tooltip(self.dataset, "tooltip.simple.dataset")
        self.dataset.set_path(self.train_page.dataset.path())
        self.dataset.path_changed.connect(self._dataset_changed)
        dataset_layout.addWidget(self.dataset)
        dataset_row = QHBoxLayout()
        self.check_button = QPushButton()
        self.details_button = QPushButton()
        bind_text(self.check_button, "simple.dataset.check")
        bind_text(self.details_button, "simple.dataset.details")
        self.check_button.setObjectName("primaryButton")
        self.check_button.clicked.connect(self.run_dataset_check)
        self.details_button.clicked.connect(self.dataset_details_requested.emit)
        dataset_row.addWidget(self.check_button)
        dataset_row.addWidget(self.details_button)
        dataset_row.addStretch()
        dataset_layout.addLayout(dataset_row)
        self.dataset_status = QLabel()
        self.dataset_status.setWordWrap(True)
        dataset_layout.addWidget(self.dataset_status)
        layout.addWidget(dataset_box)

        profile_box = QGroupBox()
        bind_text(profile_box, "simple.profiles.group")
        grid = QGridLayout(profile_box)
        model_label = QLabel()
        training_label = QLabel()
        bind_text(model_label, "simple.model.label")
        bind_text(training_label, "simple.training.label")
        self.model_profile = WheelSafeComboBox()
        self.training_profile = WheelSafeComboBox()
        bind_combo_items(self.model_profile, [("simple.model.fast", "fast"), ("simple.model.balanced", "balanced"), ("simple.model.high_accuracy", "high_accuracy")])
        bind_combo_items(self.training_profile, [("simple.training.quick", "quick"), ("simple.training.standard", "standard"), ("simple.training.extended", "extended")])
        self.model_profile.setCurrentIndex(self.model_profile.findData("balanced"))
        self.training_profile.setCurrentIndex(self.training_profile.findData("standard"))
        set_tooltip(self.model_profile, "tooltip.simple.model_profile")
        set_tooltip(self.training_profile, "tooltip.simple.training_profile")
        self.model_profile.currentIndexChanged.connect(self._profiles_selected)
        self.training_profile.currentIndexChanged.connect(self._profiles_selected)
        grid.addWidget(model_label, 0, 0)
        grid.addWidget(self.model_profile, 1, 0)
        grid.addWidget(training_label, 0, 1)
        grid.addWidget(self.training_profile, 1, 1)
        self.profile_note = QLabel()
        self.profile_note.setWordWrap(True)
        grid.addWidget(self.profile_note, 2, 0, 1, 2)
        layout.addWidget(profile_box)

        ready_box = QGroupBox()
        bind_text(ready_box, "simple.ready.group")
        ready_layout = QVBoxLayout(ready_box)
        self.runtime_status = QLabel()
        self.runtime_status.setWordWrap(True)
        set_tooltip(self.runtime_status, "tooltip.simple.runtime")
        ready_layout.addWidget(self.runtime_status)
        self.actual_settings = QTextEdit()
        self.actual_settings.setReadOnly(True)
        self.actual_settings.setMaximumHeight(112)
        self.actual_settings.setVisible(False)
        self.actual_toggle = QPushButton()
        bind_text(self.actual_toggle, "simple.actual_settings")
        self.actual_toggle.clicked.connect(lambda: self.actual_settings.setVisible(not self.actual_settings.isVisible()))
        ready_layout.addWidget(self.actual_toggle)
        ready_layout.addWidget(self.actual_settings)
        buttons = QHBoxLayout()
        self.start_button = QPushButton()
        self.stop_button = QPushButton()
        self.results_button = QPushButton()
        self.advanced_button = QPushButton()
        bind_text(self.start_button, "simple.start")
        bind_text(self.stop_button, "simple.stop")
        bind_text(self.results_button, "simple.results")
        bind_text(self.advanced_button, "simple.switch_advanced")
        self.start_button.setObjectName("primaryButton")
        self.stop_button.setEnabled(False)
        self.results_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_training)
        self.stop_button.clicked.connect(self.train_page.runner.stop)
        self.results_button.clicked.connect(self.results_requested.emit)
        self.advanced_button.clicked.connect(self.advanced_requested.emit)
        for button in (self.start_button, self.stop_button, self.results_button, self.advanced_button):
            buttons.addWidget(button)
        buttons.addStretch()
        ready_layout.addLayout(buttons)
        self.training_status = QLabel()
        self.training_status.setWordWrap(True)
        ready_layout.addWidget(self.training_status)
        layout.addWidget(ready_box)
        layout.addStretch()

        self.train_page.runner.state_changed.connect(self._training_state)
        self.train_page.runner.finished.connect(self._training_finished)
        self.train_page.runner.error.connect(self._training_error)
        get_i18n().language_changed.connect(self._retranslate)
        self.refresh_from_train()
        self._retranslate()

    def _retranslate(self, _locale: str | None = None) -> None:
        self._refresh_statuses()
        self._refresh_actual_settings()

    def _dataset_changed(self, path: str) -> None:
        self.train_page.dataset.set_path(path)
        if path != self.checked_dataset:
            self.checked_dataset = None
            self.check_result = None
        self._refresh_statuses()

    def run_dataset_check(self) -> None:
        path = Path(self.dataset.path())
        if not path.is_file():
            self.checked_dataset = None
            self.check_result = None
            self._refresh_statuses()
            return
        self.check_button.setEnabled(False)
        try:
            result = check_dataset(str(path))
            self.check_result = result
            self.checked_dataset = str(path)
        except Exception as exc:  # Existing checker errors are surfaced safely.
            self.checked_dataset = None
            self.check_result = None
            QMessageBox.warning(self, tr("common.warning"), str(exc))
        finally:
            self.check_button.setEnabled(True)
            self._refresh_statuses()

    def _profiles_selected(self) -> None:
        # Explicit profile interaction is the only time Simple Mode overwrites
        # advanced fields, preserving arbitrary Advanced Mode configurations.
        profile = build_simple_profile(
            str(self.model_profile.currentData()), str(self.training_profile.currentData()),
            self.train_page.device.text().strip(),
        )
        shared_config = self.train_page.training_config().with_updates(
            data=self.dataset.path(), **profile.values
        )
        self.train_page.apply_training_config(shared_config)
        self._refresh_actual_settings()
        self._refresh_statuses()

    def refresh_from_train(self) -> None:
        if not self.dataset.path() and self.train_page.dataset.path():
            self.dataset.set_path(self.train_page.dataset.path())
        values = self.train_page.training_values()
        matched = any(
            matches_profile(values, build_simple_profile(model, training, str(values.get("device", "0"))))
            for model in MODEL_PROFILES for training in TRAINING_PROFILES
        )
        self.profile_note.setText(tr("simple.profile_custom") if not matched else tr("simple.profile_ready"))
        self._refresh_actual_settings()
        self._refresh_statuses()

    def _runtime_ready(self) -> bool:
        return self.runtime_manager.resolve_yolo_command() is not None

    def _refresh_statuses(self) -> None:
        dataset = Path(self.dataset.path())
        errors = len(self.check_result.get("errors", [])) if self.check_result else 0
        passed = bool(self.checked_dataset and self.checked_dataset == str(dataset) and errors == 0)
        if not dataset.is_file():
            self.dataset_status.setText(tr("simple.dataset.not_selected"))
        elif passed:
            warnings = len(self.check_result.get("warnings", [])) if self.check_result else 0
            self.dataset_status.setText(tr("simple.dataset.passed", warnings=warnings))
        elif self.check_result:
            self.dataset_status.setText(tr("simple.dataset.failed", count=errors))
        else:
            self.dataset_status.setText(tr("simple.dataset.needs_check"))
        runtime_ready = self._runtime_ready()
        self.runtime_status.setText(tr("simple.runtime.ready") if runtime_ready else tr("simple.runtime.missing"))
        self.start_button.setEnabled(passed and runtime_ready and not self.train_page.runner.running)
        self.steps.setText(tr("simple.steps", dataset="✓" if dataset.is_file() else "○", check="✓" if passed else "○", profiles="✓", train="→"))

    def _refresh_actual_settings(self) -> None:
        values = self.train_page.training_values()
        self.actual_settings.setPlainText(
            "\n".join(f"{key}: {values.get(key, '')}" for key in ("model", "epochs", "batch", "imgsz", "device", "patience"))
        )

    def start_training(self) -> None:
        self._profiles_selected()
        self._refresh_statuses()
        if not self.start_button.isEnabled():
            return
        self.training_status.setText(tr("simple.training.running"))
        self.train_page.start_training()

    def _training_state(self, running: bool) -> None:
        self.start_button.setEnabled(not running and self._runtime_ready() and self.checked_dataset == self.dataset.path())
        self.stop_button.setEnabled(running)
        if running:
            self.training_status.setText(tr("simple.training.running"))

    def _training_finished(self, code: int, _status: int) -> None:
        self.stop_button.setEnabled(False)
        has_results = self.train_page.last_run_artifacts.get("results.csv") is not None
        self.results_button.setEnabled(code == 0 and has_results)
        self.training_status.setText(tr("simple.training.completed") if code == 0 else tr("simple.training.failed"))
        self._refresh_statuses()

    def _training_error(self, _message: str) -> None:
        self.training_status.setText(tr("simple.training.failed"))

    def open_output_folder(self) -> None:
        folder = self.train_page.last_run_artifacts.get("run_folder")
        if folder and folder.is_dir():
            os.startfile(folder)  # type: ignore[attr-defined]
