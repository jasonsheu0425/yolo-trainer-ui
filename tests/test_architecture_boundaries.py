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


def test_annotation_canvas_never_writes_label_text():
    for path in (ROOT / "ui" / "annotation").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "write_text(" not in source
        assert "open(" not in source


def test_model_assistance_dependency_boundaries():
    domain_source = (ROOT / "domain" / "annotation.py").read_text(encoding="utf-8")
    metadata_source = (ROOT / "persistence" / "annotation_metadata_store.py").read_text(encoding="utf-8")
    worker_source = (ROOT / "runtime_workers" / "annotation_inference_worker.py").read_text(encoding="utf-8")
    assert "workers" not in imported_roots(ROOT / "domain" / "annotation.py")
    assert "ui" not in domain_source and "ui" not in metadata_source
    assert "PySide6" not in worker_source
    for path in (ROOT / "ui" / "annotation").glob("*.py"):
        imports = imported_roots(path)
        assert "torch" not in imports and "ultralytics" not in imports
    inference_paths = [
        ROOT / "services" / "annotation_inference_service.py",
        ROOT / "workers" / "annotation_inference_controller.py",
        ROOT / "core" / "annotation_worker_path.py",
    ]
    for path in inference_paths:
        imports = imported_roots(path)
        assert "torch" not in imports and "ultralytics" not in imports, path
