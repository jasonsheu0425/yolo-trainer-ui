from __future__ import annotations

import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from runtime_workers.annotation_inference_worker import (
    WorkerCommandError,
    handle,
)


class FakeAdapter:
    def __init__(self, detections=None, error: WorkerCommandError | None = None):
        self.detections = detections or []
        self.error = error
        self.loaded = 0
        self.unloaded = 0

    def load(self, _path, _device):
        if self.error:
            raise self.error
        self.loaded += 1
        return {
            "task": "detect",
            "actual_device": "cpu",
            "cuda_available": False,
            "class_names": {"0": "object"},
        }

    def predict(self, _path, _confidence):
        if self.error:
            raise self.error
        return 640, 480, self.detections

    def unload(self):
        self.unloaded += 1


def capture_handle(monkeypatch, adapter, request):
    output = io.BytesIO()
    monkeypatch.setattr(sys, "stdout", type("Output", (), {"buffer": output})())
    keep_running = handle(adapter, request)
    return keep_running, json.loads(output.getvalue())


def test_load_model_success(monkeypatch, tmp_path):
    model = tmp_path / "best.pt"
    model.write_bytes(b"fake")
    adapter = FakeAdapter()
    _, value = capture_handle(monkeypatch, adapter, {
        "command": "load_model", "request_id": "1", "model": str(model), "device": "auto",
    })
    assert value["type"] == "model_loaded" and adapter.loaded == 1
    assert value["cuda_available"] is False


def test_load_model_missing_rejected(monkeypatch, tmp_path):
    with pytest.raises(WorkerCommandError, match="local .pt"):
        capture_handle(monkeypatch, FakeAdapter(), {
            "command": "load_model", "request_id": "1",
            "model": str(tmp_path / "missing.pt"), "device": "cpu",
        })


def test_unsupported_task_propagates(monkeypatch, tmp_path):
    model = tmp_path / "segment.pt"
    model.write_bytes(b"fake")
    with pytest.raises(WorkerCommandError) as raised:
        capture_handle(monkeypatch, FakeAdapter(error=WorkerCommandError("unsupported_model_task", "segment")), {
            "command": "load_model", "request_id": "1", "model": str(model), "device": "cpu",
        })
    assert raised.value.code == "unsupported_model_task"


def test_predict_and_zero_detection(monkeypatch, tmp_path):
    image = tmp_path / "image.jpg"
    image.write_bytes(b"fake")
    detection = {"class_id": 0, "confidence": .9, "x_center": .5,
                 "y_center": .5, "width": .2, "height": .2}
    _, value = capture_handle(monkeypatch, FakeAdapter([detection]), {
        "command": "predict", "request_id": "2", "image": str(image), "confidence": .25,
    })
    assert value["detections"] == [detection]
    _, zero = capture_handle(monkeypatch, FakeAdapter(), {
        "command": "predict", "request_id": "3", "image": str(image), "confidence": .25,
    })
    assert zero["detections"] == []


def test_malformed_image_rejected(monkeypatch, tmp_path):
    with pytest.raises(WorkerCommandError) as raised:
        capture_handle(monkeypatch, FakeAdapter(), {
            "command": "predict", "request_id": "2",
            "image": str(tmp_path / "none.jpg"), "confidence": .25,
        })
    assert raised.value.code == "invalid_image_path"


def test_ping_unload_and_shutdown(monkeypatch):
    adapter = FakeAdapter()
    keep, pong = capture_handle(monkeypatch, adapter, {"command": "ping", "request_id": "1"})
    assert keep and pong["type"] == "pong"
    keep, unloaded = capture_handle(monkeypatch, adapter, {"command": "unload_model", "request_id": "2"})
    assert keep and unloaded["type"] == "model_unloaded"
    keep, shutdown = capture_handle(monkeypatch, adapter, {"command": "shutdown", "request_id": "3"})
    assert not keep and shutdown["type"] == "shutdown_ack"


def test_real_script_hello_invalid_protocol_shutdown_and_stdout_cleanliness(tmp_path):
    script = Path(__file__).parents[1] / "runtime_workers" / "annotation_inference_worker.py"
    process = subprocess.Popen(
        [sys.executable, "-u", str(script)], stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    assert process.stdout and process.stdin
    hello_line = process.stdout.readline()
    hello = json.loads(hello_line)
    assert hello["type"] == "hello" and hello_line.strip().startswith("{")
    process.stdin.write("not-json\n")
    process.stdin.flush()
    error = json.loads(process.stdout.readline())
    assert error["code"] == "malformed_json"
    process.stdin.write(json.dumps({
        "protocol_version": 1, "command": "shutdown", "request_id": "end",
    }) + "\n")
    process.stdin.flush()
    assert json.loads(process.stdout.readline())["type"] == "shutdown_ack"
    assert process.wait(timeout=5) == 0
