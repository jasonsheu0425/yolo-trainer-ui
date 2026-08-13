from __future__ import annotations

import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from core.version import APP_ID, APP_NAME
from core.i18n_manager import get_i18n
from ui.main_window import MainWindow


STYLE = """
QWidget {
    color: #172033;
    background: #f4f7fb;
    font-size: 13px;
}
#sidebar {
    background: #111827;
}
#brand {
    color: white;
    font-size: 25px;
    font-weight: 700;
    padding: 4px 8px 22px 8px;
    background: transparent;
}
#sidebarCaption {
    color: #8290a8;
    background: transparent;
    padding: 8px;
}
#navigation {
    background: transparent;
    border: none;
    outline: none;
    color: #b9c2d0;
}
#navigation::item {
    border-radius: 8px;
    padding-left: 14px;
    margin: 2px 0;
}
#navigation::item:hover { background: #1f2937; color: white; }
#navigation::item:selected { background: #2563eb; color: white; }
#pageTitle { font-size: 27px; font-weight: 700; color: #111827; }
#pageDescription { color: #64748b; font-size: 14px; }
QGroupBox {
    background: white;
    border: 1px solid #dbe3ef;
    border-radius: 10px;
    margin-top: 12px;
    padding: 14px 12px 12px 12px;
    font-weight: 600;
}
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QLineEdit, QComboBox, QSpinBox, QTextEdit, QTableWidget, QTabWidget::pane {
    background: white;
    border: 1px solid #ccd6e3;
    border-radius: 6px;
}
QLineEdit, QComboBox, QSpinBox { min-height: 31px; padding: 0 8px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QTextEdit:focus { border: 1px solid #2563eb; }
QPushButton {
    background: white;
    border: 1px solid #c8d2df;
    border-radius: 6px;
    min-height: 32px;
    padding: 0 13px;
}
QPushButton:hover { background: #edf3fa; }
QPushButton:disabled { color: #9aa6b5; background: #edf1f5; }
#primaryButton { background: #2563eb; color: white; border-color: #2563eb; font-weight: 600; }
#primaryButton:hover { background: #1d4ed8; }
#console { background: #0b1220; color: #d7e1ef; border: none; font-family: Consolas, monospace; }
QTabBar::tab { background: #e7edf5; padding: 9px 15px; margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px; }
QTabBar::tab:selected { background: white; color: #2563eb; font-weight: 600; }
QHeaderView::section { background: #edf2f7; border: none; border-bottom: 1px solid #ccd6e3; padding: 7px; font-weight: 600; }
#metricValue { font-size: 20px; font-weight: 700; color: #2563eb; }
QLabel[state="ok"] { background: #dcfce7; color: #166534; border-radius: 6px; padding: 9px; }
QLabel[state="warning"] { background: #fef3c7; color: #92400e; border-radius: 6px; padding: 9px; }
"""


def main() -> int:
    app = QApplication(sys.argv)
    # Initialise bundled translation resources before constructing UI widgets.
    get_i18n()
    app.setApplicationName(APP_ID)
    app.setApplicationDisplayName(APP_NAME)
    app.setOrganizationName("")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
