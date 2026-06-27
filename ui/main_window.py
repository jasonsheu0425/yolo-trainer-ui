from __future__ import annotations

from PySide6.QtCore import QSize
from PySide6.QtWidgets import QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from core.config_manager import ConfigManager
from ui.dataset_page import DatasetPage
from ui.error_mining_page import ErrorMiningPage
from ui.export_page import ExportPage
from ui.monitor_page import MonitorPage
from ui.predict_page import PredictPage
from ui.settings_page import SettingsPage
from ui.train_page import TrainPage
from ui.validate_page import ValidatePage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("YOLO Trainer UI")
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
        labels = ["Train", "Dataset Check", "Validate / Evaluate", "Predict / Test", "Error Mining", "Export", "Monitor / Results", "Settings"]
        for label in labels:
            item = QListWidgetItem(label)
            item.setSizeHint(QSize(0, 46))
            self.navigation.addItem(item)
        side_layout.addWidget(self.navigation, 1)
        version = QLabel("Desktop training console")
        version.setObjectName("sidebarCaption")
        side_layout.addWidget(version)
        outer.addWidget(sidebar)

        self.stack = QStackedWidget()
        self.train_page = TrainPage(self.config)
        self.dataset_page = DatasetPage()
        self.validate_page = ValidatePage(self.config)
        self.predict_page = PredictPage(self.config)
        self.error_mining_page = ErrorMiningPage(self.config)
        self.export_page = ExportPage(self.config)
        self.monitor_page = MonitorPage(self.config)
        self.settings_page = SettingsPage(self.config)
        for page in (self.train_page, self.dataset_page, self.validate_page, self.predict_page, self.error_mining_page, self.export_page, self.monitor_page, self.settings_page):
            self.stack.addWidget(page)
        outer.addWidget(self.stack, 1)
        self.navigation.currentRowChanged.connect(self.stack.setCurrentIndex)
        self.navigation.setCurrentRow(0)
        self.settings_page.settings_saved.connect(self.train_page.apply_settings)
        self.settings_page.settings_saved.connect(self.export_page.apply_settings)
        self.settings_page.settings_saved.connect(self.predict_page.apply_settings)
        self.settings_page.settings_saved.connect(self.validate_page.apply_settings)
        self.settings_page.settings_saved.connect(self.monitor_page.apply_settings)
        self.settings_page.settings_saved.connect(self.error_mining_page.apply_settings)
        self.train_page.dataset_selected.connect(self.dataset_page.set_yaml_path)
        self.train_page.results_found.connect(self.monitor_page.load_results)

    def closeEvent(self, event) -> None:  # type: ignore[override]
        if self.train_page.runner.running:
            self.train_page.runner.stop()
        if self.export_page.runner.running:
            self.export_page.runner.stop()
        if self.predict_page.runner.running:
            self.predict_page.runner.stop()
        if self.validate_page.runner.running:
            self.validate_page.runner.stop()
        event.accept()
