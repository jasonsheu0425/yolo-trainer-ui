from __future__ import annotations

import os
from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton, QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget

from core.config_manager import ConfigManager
from core.gpu_checker import get_gpu_info
from core.results_reader import available_series, read_results, scan_runs
from ui.widgets import PageHeader, PathPicker


class MonitorPage(QWidget):
    def __init__(self, config: ConfigManager | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.config = config
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.addWidget(PageHeader("Monitor / Results", "每 2 秒更新 GPU 狀態，並將 YOLO results.csv 繪製成訓練曲線。"))
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_gpu_tab(), "GPU Monitor")
        self.tabs.addTab(self._build_results_tab(), "Results")
        self.tabs.addTab(self._build_run_browser_tab(), "Run Browser")
        layout.addWidget(self.tabs, 1)
        self.timer = QTimer(self)
        self.timer.setInterval(2000)
        self.timer.timeout.connect(self.refresh_gpu)
        self.timer.start()
        QTimer.singleShot(0, self.refresh_gpu)

    def _build_gpu_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.cuda_warning = QLabel()
        self.cuda_warning.setWordWrap(True)
        layout.addWidget(self.cuda_warning)
        summary = QGroupBox("Runtime")
        grid = QGridLayout(summary)
        self.torch_value = QLabel("—")
        self.cuda_value = QLabel("—")
        self.cuda_version = QLabel("—")
        for col, (title, value) in enumerate((("Torch", self.torch_value), ("CUDA available", self.cuda_value), ("Torch CUDA", self.cuda_version))):
            grid.addWidget(QLabel(title), 0, col)
            value.setObjectName("metricValue")
            grid.addWidget(value, 1, col)
        layout.addWidget(summary)
        self.gpu_table = QTableWidget(0, 9)
        self.gpu_table.setHorizontalHeaderLabels(["#", "GPU", "Util %", "Memory used", "Memory total", "Temp °C", "Power W", "Allocated", "Reserved"])
        self.gpu_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.gpu_table, 1)
        refresh = QPushButton("立即更新")
        refresh.clicked.connect(self.refresh_gpu)
        layout.addWidget(refresh)
        return tab

    def _build_results_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.results_picker = PathPicker("results.csv", "CSV (*.csv)")
        layout.addWidget(self.results_picker)
        row = QHBoxLayout()
        load = QPushButton("Load Results")
        load.setObjectName("primaryButton")
        load.clicked.connect(lambda: self.load_results(self.results_picker.path()))
        self.results_status = QLabel("尚未載入")
        row.addWidget(load)
        row.addWidget(self.results_status)
        row.addStretch()
        layout.addLayout(row)
        self.figure = Figure(figsize=(9, 5), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout.addWidget(self.canvas, 1)
        return tab

    def _build_run_browser_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.runs_picker = PathPicker("Runs root folder", directory=True)
        default_root = str(self.config.get("runs_folder", "runs/detect")) if self.config else "runs/detect"
        self.runs_picker.set_path(default_root)
        layout.addWidget(self.runs_picker)
        row = QHBoxLayout()
        scan_button = QPushButton("Scan Runs")
        scan_button.setObjectName("primaryButton")
        scan_button.clicked.connect(self.scan_run_browser)
        self.runs_status = QLabel("Not scanned")
        row.addWidget(scan_button)
        row.addWidget(self.runs_status)
        row.addStretch()
        layout.addLayout(row)
        headers = [
            "Run Name", "Type", "Path", "best.pt", "last.pt", "results.csv",
            "Hard Cases", "Error Mining", "mAP50", "mAP50-95", "Precision", "Recall",
            "Created / Modified", "Open Folder",
        ]
        self.runs_table = QTableWidget(0, len(headers))
        self.runs_table.setHorizontalHeaderLabels(headers)
        self.runs_table.setSortingEnabled(True)
        self.runs_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.runs_table, 1)
        return tab

    def apply_settings(self, values: dict) -> None:
        if self.config:
            self.config.settings.update(values)
        if "runs_folder" in values:
            self.runs_picker.set_path(str(values["runs_folder"]))

    def scan_run_browser(self) -> None:
        root = self.runs_picker.path()
        runs = scan_runs(root)
        self.runs_table.setSortingEnabled(False)
        self.runs_table.setRowCount(len(runs))
        for row, run in enumerate(runs):
            mining_summary = run.get("hard_cases_summary", {})
            if run.get("hard_cases"):
                mining_hint = (
                    f"low={mining_summary.get('low_confidence_count', 0)}, "
                    f"no_detection={mining_summary.get('no_detection_count', 0)}, "
                    f"no_label={mining_summary.get('no_label_file_count', 0)}"
                )
            elif run["type"] == "predict":
                mining_hint = "Available for Error Mining"
            else:
                mining_hint = "—"
            values = [
                run["name"],
                run["type"],
                str(run["path"]),
                "Yes" if run["best"] else "No",
                "Yes" if run["last"] else "No",
                "Yes" if run["results"] else "No",
                "Yes" if run.get("hard_cases") else "No",
                mining_hint,
                self._format_metric(run.get("map50")),
                self._format_metric(run.get("map50_95")),
                self._format_metric(run.get("precision")),
                self._format_metric(run.get("recall")),
                f"{run['created']:%Y-%m-%d %H:%M} / {run['modified']:%Y-%m-%d %H:%M}",
            ]
            for column, value in enumerate(values):
                self.runs_table.setItem(row, column, QTableWidgetItem(str(value)))
            open_button = QPushButton("Open Folder")
            open_button.clicked.connect(lambda _checked=False, path=run["path"]: self._open_run_folder(path))
            self.runs_table.setCellWidget(row, len(values), open_button)
        self.runs_table.setSortingEnabled(True)
        root_path = Path(root).expanduser() if root else None
        if root_path is None or not root_path.is_dir():
            self.runs_status.setText("Runs root folder not found.")
        else:
            self.runs_status.setText(f"Found {len(runs)} run folder(s).")

    @staticmethod
    def _format_metric(value) -> str:
        return f"{value:.4f}" if isinstance(value, (int, float)) else "Not found"

    @staticmethod
    def _open_run_folder(path: Path) -> None:
        if path.is_dir():
            os.startfile(path)  # type: ignore[attr-defined]

    def refresh_gpu(self) -> None:
        info = get_gpu_info()
        self.torch_value.setText(str(info["torch_version"]))
        self.cuda_value.setText("Yes" if info["cuda_available"] else "No")
        self.cuda_version.setText(str(info["torch_cuda_version"]))
        if info["cuda_available"]:
            self.cuda_warning.setText(f"CUDA 可用，共偵測到 {info['gpu_count']} 張 GPU。")
            self.cuda_warning.setProperty("state", "ok")
        else:
            self.cuda_warning.setText("CUDA is not available. Training may run on CPU and become very slow.")
            self.cuda_warning.setProperty("state", "warning")
        self.cuda_warning.style().unpolish(self.cuda_warning)
        self.cuda_warning.style().polish(self.cuda_warning)
        self.gpu_table.setRowCount(len(info["gpus"]))
        keys = ["index", "name", "utilization_percent", "memory_used_mb", "memory_total_mb", "temperature_c", "power_w", "vram_allocated_mb", "vram_reserved_mb"]
        for row, gpu in enumerate(info["gpus"]):
            for column, key in enumerate(keys):
                value = gpu.get(key, "—")
                if value is None:
                    value = "—"
                self.gpu_table.setItem(row, column, QTableWidgetItem(str(value)))

    def load_results(self, path: str) -> None:
        if not path:
            return
        self.results_picker.set_path(path)
        try:
            frame = read_results(path)
        except Exception as exc:
            QMessageBox.warning(self, "Results", str(exc))
            return
        self.figure.clear()
        groups = [("loss", "Loss"), ("map", "mAP"), ("pr", "Precision / Recall")]
        for index, (group, title) in enumerate(groups, 1):
            axis = self.figure.add_subplot(1, 3, index)
            x = frame["epoch"] if "epoch" in frame.columns else range(len(frame))
            for series in available_series(frame, group):
                axis.plot(x, frame[series], label=series.replace("metrics/", "").replace("train/", "train ").replace("val/", "val "))
            axis.set_title(title)
            axis.set_xlabel("Epoch")
            axis.grid(True, alpha=0.25)
            if axis.lines:
                axis.legend(fontsize=7)
        self.canvas.draw_idle()
        self.results_status.setText(f"{Path(path).name}：{len(frame)} epochs")
        self.tabs.setCurrentIndex(1)
