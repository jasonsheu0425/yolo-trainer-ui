from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from core.config_manager import ConfigManager
from core.runtime_manager import RuntimeManager
from core.version import APP_NAME, APP_VERSION
from ui.dataset_page import DatasetPage
from ui.dataset_builder_page import DatasetBuilderPage
from ui.error_mining_page import ErrorMiningPage
from ui.export_page import ExportPage
from ui.monitor_page import MonitorPage
from ui.predict_page import PredictPage
from ui.runtime_page import RuntimePage
from ui.settings_page import SettingsPage
from ui.train_page import TrainPage
from ui.validate_page import ValidatePage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1280, 820)
        self.setMinimumSize(1050, 680)
        self.config = ConfigManager()

        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(218)
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(16, 22, 16, 16)
        brand = QLabel("YOLO\nTrainer UI")
        brand.setObjectName("brand")
        side_layout.addWidget(brand)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        labels = ["Train", "Dataset Check", "Dataset Builder", "Validate / Evaluate", "Predict / Test", "Error Mining", "Export", "Monitor / Results", "Runtime / Environment", "Settings"]
        self.runtime_index = labels.index("Runtime / Environment")
        self.settings_index = labels.index("Settings")
        for label in labels:
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(0, 46))
            self.navigation.addItem(item)
        side_layout.addWidget(self.navigation, 1)
        version = QLabel(f"v{APP_VERSION} · Desktop console")
        version.setObjectName("sidebarCaption")
        side_layout.addWidget(version)
        outer.addWidget(sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.runtime_banner = QWidget()
        banner_layout = QHBoxLayout(self.runtime_banner)
        banner_layout.setContentsMargins(18, 9, 18, 9)
        self.runtime_banner_label = QLabel("YOLO runtime is not configured. UI-only features remain available. Open Runtime / Environment to configure training and inference.")
        self.runtime_banner_label.setWordWrap(True)
        banner_button = QLabel('<a href="open">Open Runtime / Environment</a>')
        banner_button.setOpenExternalLinks(False)
        banner_button.linkActivated.connect(lambda _link: self._open_runtime())
        banner_layout.addWidget(self.runtime_banner_label, 1)
        banner_layout.addWidget(banner_button)
        self.runtime_banner.setStyleSheet("background: #fef3c7; color: #92400e;")
        content_layout.addWidget(self.runtime_banner)

        self.stack = QStackedWidget()
        self.train_page = TrainPage(self.config)
        self.dataset_page = DatasetPage()
        self.dataset_builder_page = DatasetBuilderPage(self.config)
        self.validate_page = ValidatePage(self.config)
        self.predict_page = PredictPage(self.config)
        self.error_mining_page = ErrorMiningPage(self.config)
        self.export_page = ExportPage(self.config)
        self.monitor_page = MonitorPage(self.config)
        self.runtime_page = RuntimePage(self.config)
        self.settings_page = SettingsPage(self.config)
        for page in (self.train_page, self.dataset_page, self.dataset_builder_page, self.validate_page, self.predict_page, self.error_mining_page, self.export_page, self.monitor_page, self.runtime_page, self.settings_page):
            self.stack.addWidget(page)
        content_layout.addWidget(self.stack, 1)
        outer.addWidget(content, 1)
        self.navigation.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        self.settings_page.settings_saved.connect(self.train_page.apply_settings)
        self.settings_page.settings_saved.connect(self.export_page.apply_settings)
        self.settings_page.settings_saved.connect(self.predict_page.apply_settings)
        self.settings_page.settings_saved.connect(self.validate_page.apply_settings)
        self.settings_page.settings_saved.connect(self.monitor_page.apply_settings)
        self.settings_page.settings_saved.connect(self.error_mining_page.apply_settings)
        self.settings_page.settings_saved.connect(self.dataset_builder_page.apply_settings)
        self.settings_page.settings_saved.connect(self.runtime_page.apply_settings)
        self.settings_page.settings_saved.connect(self._settings_saved)
        self.runtime_page.runtime_changed.connect(self._runtime_changed)
        self.runtime_page.open_settings_requested.connect(lambda: self.navigation.setCurrentRow(self.settings_index))
        for page in (self.train_page, self.predict_page, self.validate_page, self.export_page):
            page.runtime_required.connect(self._open_runtime)
        self.train_page.dataset_selected.connect(self.dataset_page.set_yaml_path)
        self.train_page.results_found.connect(self.monitor_page.load_results)
        self.error_mining_page.hard_cases_exported.connect(self.dataset_builder_page.set_hard_cases)
        self.dataset_builder_page.use_dataset_requested.connect(self._use_built_dataset)
        self._refresh_runtime_banner()

    def _open_runtime(self) -> None:
        self.navigation.setCurrentRow(self.runtime_index)

    def _settings_saved(self, _values: dict) -> None:
        self._refresh_runtime_banner()

    def _runtime_changed(self, values: dict) -> None:
        if "yolo_command" in values and "application_mode" not in values:
            self.settings_page.load()
            for page in (self.train_page, self.predict_page, self.validate_page, self.export_page):
                page.apply_settings(values)
        available = values.get("yolo_available")
        if isinstance(available, bool):
            self.runtime_banner.setVisible(not available)
        else:
            self._refresh_runtime_banner()

    def _refresh_runtime_banner(self) -> None:
        self.runtime_banner.setVisible(RuntimeManager(self.config).resolve_yolo_command() is None)

    def _use_built_dataset(self, data_yaml: str) -> None:
        self.train_page.dataset.set_path(data_yaml)
        self.navigation.setCurrentRow(0)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        self.dataset_builder_page.shutdown()
        self.runtime_page.shutdown()
        if self.train_page.runner.running:
            self.train_page.runner.stop()
        if self.export_page.runner.running:
            self.export_page.runner.stop()
        if self.predict_page.runner.running:
            self.predict_page.runner.stop()
        if self.validate_page.runner.running:
            self.validate_page.runner.stop()
        event.accept()
