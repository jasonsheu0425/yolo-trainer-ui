from __future__ import annotations

import sys

import core.annotation_worker_path as worker_path


def test_source_worker_path_is_a_physical_script():
    resolved = worker_path.resolve_annotation_worker_path()
    assert resolved.is_file()
    assert resolved.name == "annotation_inference_worker.py"


def test_frozen_worker_path_prefers_bundle_resource(tmp_path, monkeypatch):
    bundled = tmp_path / "runtime_workers" / "annotation_inference_worker.py"
    bundled.parent.mkdir()
    bundled.write_text("# physical worker resource\n", encoding="utf-8")
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert worker_path.resolve_annotation_worker_path() == bundled.resolve()
