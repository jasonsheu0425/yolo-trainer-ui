from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_non_ui_layers_do_not_depend_on_ui_pages():
    for directory in ("domain", "services", "persistence"):
        for path in (ROOT / directory).glob("*.py"):
            assert "ui" not in imported_roots(path), path


def test_core_algorithms_do_not_import_qt_widgets_or_main_window():
    for path in (ROOT / "core").glob("*.py"):
        imports = imported_roots(path)
        assert "ui" not in imports, path
        source = path.read_text(encoding="utf-8")
        assert "QWidget" not in source and "QMainWindow" not in source, path


def test_no_hardcoded_developer_paths_in_production_python():
    for directory in ("app", "domain", "services", "persistence", "core", "ui"):
        for path in (ROOT / directory).glob("*.py"):
            source = path.read_text(encoding="utf-8").lower()
            assert "e:\\yolo_trainer_ui" not in source and "c:\\users\\jason" not in source, path
