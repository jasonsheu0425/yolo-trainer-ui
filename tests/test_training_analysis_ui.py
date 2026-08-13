from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import QApplication

from core.i18n_manager import get_i18n
from ui.main_window import MainWindow
from ui.training_analysis_page import TrainingAnalysisPage


def app() -> QApplication:
    return QApplication.instance() or QApplication([])


def make_run(folder):
    folder.mkdir()
    (folder / "weights").mkdir()
    (folder / "weights" / "best.pt").write_bytes(b"test-only")
    (folder / "results.csv").write_text(
        "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)\n"
        "0,0.9,0.7,0.9,0.65\n",
        encoding="utf-8",
    )
    image = QPixmap(32, 16)
    image.fill(QColor("#4f46e5"))
    image.save(str(folder / "results.png"))
    image.save(str(folder / "confusion_matrix.png"))


def test_training_analysis_page_handles_empty_valid_and_cached_runs(tmp_path):
    app()
    get_i18n().set_language("en_US")
    page = TrainingAnalysisPage()
    assert page.findings.toPlainText() == ""
    run = tmp_path / "run"
    make_run(run)
    page.load_run(run)
    assert "Reference" in page.rating.text()
    assert page.cards["precision"].text() == "90.0%"
    assert page.curves_preview.pixmap() is not None
    assert page.confusion_preview.pixmap() is not None
    assert page.artifact_buttons["results.csv"].isEnabled()
    page.load_run(run)
    assert page.analysis and page.analysis.cache_status == "cache_hit"
    page.load_run(tmp_path / "not-a-run")
    assert page.rating.text() == "N/A"
    page.close()


def test_training_analysis_page_live_language_switch_and_reanalyze(tmp_path):
    app()
    page = TrainingAnalysisPage()
    run = tmp_path / "run"
    make_run(run)
    page.load_run(run)
    get_i18n().set_language("zh_TW")
    QApplication.processEvents()
    assert "參考評級" in page.rating.text()
    page.load_run(run, force=True)
    assert page.analysis and page.analysis.cache_status == "fresh"
    get_i18n().set_language("en_US")
    QApplication.processEvents()
    assert page.analyze_button.text() == "Analyze Run"
    get_i18n().set_language("zh_TW")
    page.close()


def test_main_window_train_and_run_browser_analysis_routing(tmp_path):
    app()
    get_i18n().set_language("en_US")
    run = tmp_path / "run"
    make_run(run)
    window = MainWindow()
    window._open_analysis(str(run))
    assert window.stack.currentWidget() is window.analysis_page
    assert window.analysis_page.run_folder == run.resolve()
    window.train_page._update_run_summary(str(run))
    assert window.train_page.analyze_results_button.isEnabled()
    window.monitor_page.runs_picker.set_path(str(tmp_path))
    window.monitor_page.scan_run_browser()
    button = window.monitor_page.runs_table.cellWidget(0, 14)
    assert button is not None and button.isEnabled()
    button.click()
    assert window.stack.currentWidget() is window.analysis_page
    get_i18n().set_language("zh_TW")
    window.close()
