"""UI for the deterministic training-result analyzer."""
from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from core.i18n_manager import get_i18n, tr
from core.results_reader import scan_run_folder
from core.training_result_analyzer import TrainingAnalysis, analyze_or_load
from ui.widgets import PageHeader, PathPicker, bind_text


class _ImagePreview(QLabel):
    """Scaled once per resize from one cached QPixmap, never re-read from disk."""

    def __init__(self) -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumHeight(220)

    def set_image(self, path: Path | None) -> bool:
        image = QPixmap(str(path)) if path and path.is_file() else QPixmap()
        self._pixmap = image if not image.isNull() else None
        self._update_pixmap()
        return self._pixmap is not None

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self._pixmap is None:
            self.setPixmap(QPixmap())
            self.setText(tr("analysis.no_preview"))
            return
        self.setText("")
        self.setPixmap(self._pixmap.scaled(
            self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        ))


class TrainingAnalysisPage(QWidget):
    """Presentation layer over one shared, non-mutating analyzer."""

    review_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.run_folder: Path | None = None
        self.analysis: TrainingAnalysis | None = None
        self.artifacts: dict[str, Path | None] = {}
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
        layout.addWidget(PageHeader("analysis.title", "analysis.description"))

        self.picker = PathPicker("", directory=True)
        bind_text(self.picker.label, "analysis.run_folder")
        layout.addWidget(self.picker)
        controls = QHBoxLayout()
        self.analyze_button = QPushButton()
        self.reanalyze_button = QPushButton()
        self.open_run_button = QPushButton()
        bind_text(self.analyze_button, "analysis.analyze")
        bind_text(self.reanalyze_button, "analysis.reanalyze")
        bind_text(self.open_run_button, "analysis.open_run_folder")
        self.analyze_button.setObjectName("primaryButton")
        self.reanalyze_button.setEnabled(False)
        self.open_run_button.setEnabled(False)
        self.analyze_button.clicked.connect(lambda: self.load_run(self.picker.path()))
        self.reanalyze_button.clicked.connect(lambda: self.load_run(self.picker.path(), force=True))
        self.open_run_button.clicked.connect(self._open_run_folder)
        controls.addWidget(self.analyze_button)
        controls.addWidget(self.reanalyze_button)
        controls.addWidget(self.open_run_button)
        controls.addStretch()
        layout.addLayout(controls)

        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.rating = QLabel("N/A")
        self.rating.setObjectName("metricValue")
        layout.addWidget(self.rating)
        metrics_box = QGroupBox()
        bind_text(metrics_box, "analysis.metrics")
        grid = QGridLayout(metrics_box)
        self.cards: dict[str, QLabel] = {}
        labels = {"precision": "Precision", "recall": "Recall", "map50": "mAP50", "map50_95": "mAP50-95"}
        for column, key in enumerate(labels):
            grid.addWidget(QLabel(labels[key]), 0, column)
            value = QLabel("N/A")
            value.setObjectName("metricValue")
            grid.addWidget(value, 1, column)
            self.cards[key] = value
        layout.addWidget(metrics_box)

        findings_box = QGroupBox()
        bind_text(findings_box, "analysis.findings")
        findings_layout = QVBoxLayout(findings_box)
        self.findings = QTextEdit()
        self.findings.setReadOnly(True)
        self.findings.setMinimumHeight(145)
        findings_layout.addWidget(self.findings)
        layout.addWidget(findings_box)
        self.recommendations_box = QGroupBox()
        bind_text(self.recommendations_box, "analysis.recommendations")
        self.recommendations_layout = QHBoxLayout(self.recommendations_box)
        self.recommendations_layout.addStretch()
        layout.addWidget(self.recommendations_box)

        artifacts_box = QGroupBox()
        bind_text(artifacts_box, "analysis.artifacts")
        artifact_grid = QGridLayout(artifacts_box)
        self.artifact_buttons: dict[str, QPushButton] = {}
        for row, key in enumerate(("results.csv", "results.png", "confusion_matrix_normalized.png", "confusion_matrix.png", "best.pt", "last.pt")):
            artifact_grid.addWidget(QLabel(key), row, 0)
            button = QPushButton()
            bind_text(button, "common.open_file")
            button.setEnabled(False)
            button.clicked.connect(lambda _checked=False, artifact=key: self._open_artifact(artifact))
            artifact_grid.addWidget(button, row, 1)
            self.artifact_buttons[key] = button
        layout.addWidget(artifacts_box)

        self.previews = QTabWidget()
        self.curves_preview = _ImagePreview()
        self.confusion_preview = _ImagePreview()
        self.previews.addTab(self.curves_preview, "")
        self.previews.addTab(self.confusion_preview, "")
        layout.addWidget(self.previews)
        self.per_class = QLabel()
        self.per_class.setWordWrap(True)
        layout.addWidget(self.per_class)
        layout.addStretch()
        get_i18n().language_changed.connect(self._retranslate)
        self._retranslate()
        self._set_empty()

    def load_run(self, folder: str | Path, *, force: bool = False) -> None:
        root = Path(str(folder).strip()).expanduser() if str(folder).strip() else None
        self.run_folder = root.resolve() if root and root.is_dir() else None
        if self.run_folder is None:
            self._set_empty()
            return
        self.picker.set_path(str(self.run_folder))
        self.analysis = analyze_or_load(self.run_folder, force=force)
        self.artifacts = scan_run_folder(self.run_folder)
        normalized = self.run_folder / "confusion_matrix_normalized.png"
        self.artifacts["confusion_matrix_normalized.png"] = normalized if normalized.is_file() else None
        self._refresh_analysis()

    def _set_empty(self) -> None:
        self.analysis = None
        self.artifacts = {}
        self.rating.setText("N/A")
        self.status.setText(tr("analysis.empty"))
        self.findings.clear()
        for card in self.cards.values():
            card.setText("N/A")
        self.reanalyze_button.setEnabled(False)
        self.open_run_button.setEnabled(False)
        for button in self.artifact_buttons.values():
            button.setEnabled(False)
        self.curves_preview.set_image(None)
        self.confusion_preview.set_image(None)
        self.per_class.setText(tr("analysis.no_per_class"))
        self._clear_recommendations()

    def _refresh_analysis(self) -> None:
        if self.analysis is None:
            self._set_empty()
            return
        rating_key = f"analysis.rating.{self.analysis.rating}"
        self.rating.setText(tr("analysis.rating", rating=tr(rating_key)))
        for key, label in self.cards.items():
            value = self.analysis.metrics.get(key)
            label.setText(f"{value:.1%}" if isinstance(value, (float, int)) else "N/A")
        lines: list[str] = []
        for finding in self.analysis.findings:
            lines.append(tr(f"analysis.finding.{finding.finding_id}"))
            if finding.evidence:
                lines.append("  " + ", ".join(f"{key}={value}" for key, value in finding.evidence.items()))
        self.findings.setPlainText("\n".join(lines))
        if self.analysis.persistence_error:
            self.status.setText(tr("analysis.cache_failed", error=self.analysis.persistence_error))
        elif self.analysis.cache_status == "cache_hit":
            self.status.setText(tr("analysis.cache_hit"))
        elif self.analysis.cache_status == "unverified_cache":
            self.status.setText(tr("analysis.cache_unverified"))
        else:
            self.status.setText(tr("analysis.cache_fresh"))
        self.reanalyze_button.setEnabled(self.run_folder is not None and (self.run_folder / "results.csv").is_file())
        self.open_run_button.setEnabled(self.run_folder is not None)
        for key, button in self.artifact_buttons.items():
            button.setEnabled(self.artifacts.get(key) is not None)
        self.curves_preview.set_image(self.artifacts.get("results.png"))
        self.confusion_preview.set_image(
            self.artifacts.get("confusion_matrix_normalized.png") or self.artifacts.get("confusion_matrix.png")
        )
        self.per_class.setText(tr("analysis.no_per_class"))
        self._set_recommendations(self.analysis.recommendations)

    def _clear_recommendations(self) -> None:
        while self.recommendations_layout.count() > 1:
            item = self.recommendations_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

    def _set_recommendations(self, recommendations: list[str]) -> None:
        self._clear_recommendations()
        for recommendation in recommendations:
            button = QPushButton(tr(f"analysis.recommendation.{recommendation}"))
            if recommendation == "review_false_negatives":
                button.clicked.connect(lambda _checked=False: self.review_requested.emit("false_negative"))
            elif recommendation == "review_false_positives":
                button.clicked.connect(lambda _checked=False: self.review_requested.emit("false_positive"))
            elif recommendation == "inspect_low_iou":
                button.clicked.connect(lambda _checked=False: self.review_requested.emit("low_iou"))
            else:
                button.setEnabled(False)
            self.recommendations_layout.insertWidget(self.recommendations_layout.count() - 1, button)

    def _open_run_folder(self) -> None:
        if self.run_folder and self.run_folder.is_dir():
            os.startfile(self.run_folder)  # type: ignore[attr-defined]

    def _open_artifact(self, key: str) -> None:
        path = self.artifacts.get(key)
        if path and path.is_file():
            os.startfile(path)  # type: ignore[attr-defined]

    def _retranslate(self, _locale: str | None = None) -> None:
        self.previews.setTabText(0, tr("analysis.preview_curves"))
        self.previews.setTabText(1, tr("analysis.preview_confusion"))
        if self.analysis is not None:
            self._refresh_analysis()
