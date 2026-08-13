from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from app.application import ApplicationServices
from app.navigation import NavigationRequest
from core.config_manager import ConfigManager
from core.i18n_manager import get_i18n, tr
from core.version import APP_NAME, APP_VERSION
from ui.dataset_page import DatasetPage
from ui.dataset_builder_page import DatasetBuilderPage
from ui.error_mining_page import ErrorMiningPage
from ui.export_page import ExportPage
from ui.monitor_page import MonitorPage
from ui.predict_page import PredictPage
from ui.runtime_page import RuntimePage
from ui.settings_page import SettingsPage
from ui.simple_mode_page import SimpleModePage
from ui.train_page import TrainPage
from ui.training_analysis_page import TrainingAnalysisPage
from ui.validate_page import ValidatePage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.resize(1280, 820)
        self.setMinimumSize(1050, 680)
        self.config = ConfigManager()
        self.services = ApplicationServices.create(self.config)
        self.i18n = get_i18n()
        self.i18n.set_language(str(self.config.get("language", "zh_TW")), emit=False)

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
        self.brand = QLabel("YOLO\nTrainer UI")
        self.brand.setObjectName("brand")
        side_layout.addWidget(self.brand)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        self.navigation_keys = ["nav.train", "nav.dataset", "nav.builder", "nav.validate", "nav.predict", "nav.mining", "nav.export", "nav.monitor", "nav.analysis", "nav.runtime", "nav.settings"]
        self.runtime_index = self.navigation_keys.index("nav.runtime")
        self.settings_index = self.navigation_keys.index("nav.settings")
        for key in self.navigation_keys:
            item = QListWidgetItem(tr(key))
            item.setSizeHint(QSize(0, 46))
            self.navigation.addItem(item)
        side_layout.addWidget(self.navigation, 1)
        self.version_label = QLabel()
        self.version_label.setObjectName("sidebarCaption")
        side_layout.addWidget(self.version_label)
        outer.addWidget(sidebar)

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        self.runtime_banner = QWidget()
        banner_layout = QHBoxLayout(self.runtime_banner)
        banner_layout.setContentsMargins(18, 9, 18, 9)
        self.runtime_banner_label = QLabel()
        self.runtime_banner_label.setWordWrap(True)
        self.banner_button = QLabel()
        self.banner_button.setOpenExternalLinks(False)
        self.banner_button.linkActivated.connect(lambda _link: self._open_runtime())
        banner_layout.addWidget(self.runtime_banner_label, 1)
        banner_layout.addWidget(self.banner_button)
        self.runtime_banner.setStyleSheet("background: #fef3c7; color: #92400e;")
        content_layout.addWidget(self.runtime_banner)

        self.stack = QStackedWidget()
        self.train_page = TrainPage(self.config, self.services.training)
        self.analysis_page = TrainingAnalysisPage(self.services.analysis)
        self.simple_page = SimpleModePage(self.config, self.train_page)
        self.dataset_page = DatasetPage()
        self.dataset_builder_page = DatasetBuilderPage(self.config)
        self.validate_page = ValidatePage(self.config)
        self.predict_page = PredictPage(self.config)
        self.error_mining_page = ErrorMiningPage(self.config)
        self.export_page = ExportPage(self.config)
        self.monitor_page = MonitorPage(self.config)
        self.runtime_page = RuntimePage(self.config)
        self.settings_page = SettingsPage(self.config)
        for page in (self.simple_page, self.train_page, self.dataset_page, self.dataset_builder_page, self.validate_page, self.predict_page, self.error_mining_page, self.export_page, self.monitor_page, self.analysis_page, self.runtime_page, self.settings_page):
            self.stack.addWidget(page)
        content_layout.addWidget(self.stack, 1)
        outer.addWidget(content, 1)
        self.navigation.currentRowChanged.connect(self._navigation_changed)
        self.services.navigation.requested.connect(self._navigation_requested)
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
        self.simple_page.advanced_requested.connect(lambda: self.set_ui_mode("advanced"))
        self.simple_page.results_requested.connect(lambda: self._open_page("nav.analysis"))
        self.simple_page.dataset_details_requested.connect(lambda: self._open_page("nav.dataset"))
        self.runtime_page.runtime_changed.connect(self._runtime_changed)
        self.runtime_page.open_settings_requested.connect(lambda: self._open_page("nav.settings"))
        for page in (self.train_page, self.predict_page, self.validate_page, self.export_page):
            page.runtime_required.connect(self._open_runtime)
        self.train_page.dataset_selected.connect(self.dataset_page.set_yaml_path)
        self.train_page.results_found.connect(self.monitor_page.load_results)
        self.train_page.results_found.connect(self._prepare_analysis_from_results)
        self.train_page.analysis_requested.connect(self._open_analysis)
        self.monitor_page.analysis_requested.connect(self._open_analysis)
        self.analysis_page.review_requested.connect(self._open_error_mining_review)
        self.error_mining_page.hard_cases_exported.connect(self.dataset_builder_page.set_hard_cases)
        self.dataset_builder_page.use_dataset_requested.connect(self._use_built_dataset)
        self._refresh_runtime_banner()
        self._retranslate_ui()
        self.i18n.language_changed.connect(self._retranslate_ui)
        self.set_ui_mode(str(self.config.get("ui_mode", "advanced")), persist=False)

    def _retranslate_ui(self, _locale: str | None = None) -> None:
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self._rebuild_navigation()
        self.version_label.setText(f"v{APP_VERSION} · {tr('app.desktop_console')}")
        self.runtime_banner_label.setText(tr("main.runtime_missing"))
        self.banner_button.setText(f'<a href="open">{tr("main.open_runtime")}</a>')

    def _open_runtime(self) -> None:
        self._open_page("nav.runtime")

    def _open_analysis(self, run_folder: str) -> None:
        self.analysis_page.load_run(run_folder)
        self._open_page("nav.analysis")

    def _prepare_analysis_from_results(self, results_csv: str) -> None:
        path = Path(results_csv)
        if path.is_file():
            self.analysis_page.picker.set_path(str(path.parent))

    def _open_error_mining_review(self, category: str) -> None:
        if self.ui_mode != "advanced":
            self.set_ui_mode("advanced")
        self.error_mining_page.report_category.setCurrentText(category)
        self._open_page("nav.mining")

    def _settings_saved(self, _values: dict) -> None:
        locale = str(_values.get("language", self.i18n.get_language()))
        self.i18n.set_language(locale)
        if "ui_mode" in _values:
            self.set_ui_mode(str(_values["ui_mode"]), persist=False)
        self._refresh_runtime_banner()

    def set_ui_mode(self, mode: str, *, persist: bool = True) -> None:
        self.ui_mode = mode if mode in {"simple", "advanced"} else "advanced"
        if persist:
            self.config.save({"ui_mode": self.ui_mode})
        self.active_navigation_keys = (["nav.quick_start", "nav.train", "nav.analysis", "nav.runtime", "nav.settings"] if self.ui_mode == "simple" else self.navigation_keys)
        self._rebuild_navigation()
        self.simple_page.refresh_from_train()

    def _navigation_changed(self, row: int) -> None:
        keys = getattr(self, "active_navigation_keys", self.navigation_keys)
        if 0 <= row < len(keys):
            mapping = {"nav.quick_start": 0, "nav.train": 1, "nav.dataset": 2, "nav.builder": 3, "nav.validate": 4, "nav.predict": 5, "nav.mining": 6, "nav.export": 7, "nav.monitor": 8, "nav.analysis": 9, "nav.runtime": 10, "nav.settings": 11}
            self.stack.setCurrentIndex(mapping[keys[row]])

    def _open_page(self, key: str) -> None:
        self.services.navigation.go_to(key.removeprefix("nav."))

    def _navigation_requested(self, request: NavigationRequest) -> None:
        key = f"nav.{request.target}"
        if key in self.active_navigation_keys:
            self.navigation.setCurrentRow(self.active_navigation_keys.index(key))

    def _rebuild_navigation(self) -> None:
        keys = getattr(self, "active_navigation_keys", self.navigation_keys)
        self.navigation.blockSignals(True)
        self.navigation.clear()
        for key in keys:
            item = QListWidgetItem(tr(key))
            item.setSizeHint(QSize(0, 46))
            self.navigation.addItem(item)
        self.navigation.blockSignals(False)
        if keys:
            self.navigation.setCurrentRow(0)

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
        self.runtime_banner.setVisible(self.services.runtime.resolve_yolo_command() is None)

    def _use_built_dataset(self, data_yaml: str) -> None:
        self.train_page.dataset.set_path(data_yaml)
        self._open_page("nav.train")

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
