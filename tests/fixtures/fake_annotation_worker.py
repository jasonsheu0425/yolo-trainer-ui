"""Deterministic standalone JSONL worker used by QProcess lifecycle tests."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time


VERSION = 1
loaded = False
load_count = 0


def emit(payload):
    payload.setdefault("protocol_version", VERSION)
    print(json.dumps(payload, allow_nan=False), flush=True)


emit({"type": "hello", "worker_version": "fake-0.13"})
for line in sys.stdin:
    try:
        request = json.loads(line)
        request_id = request.get("request_id", "unknown")
        command = request.get("command")
        if command == "load_model":
            model = Path(request.get("model", ""))
            if "crash" in model.name:
                raise SystemExit(23)
            if "unsupported" in model.name:
                emit({"type": "error", "request_id": request_id,
                      "code": "unsupported_model_task", "message": "segment"})
                continue
            load_count += 1
            loaded = True
            names = {"0": "different"} if "mismatch" in model.name else {"0": "object"}
            emit({"type": "model_loaded", "request_id": request_id,
                  "model": str(model), "task": "detect", "actual_device": "cpu",
                  "cuda_available": False, "class_names": names,
                  "load_count": load_count})
        elif command == "predict":
            image = Path(request.get("image", ""))
            if not loaded:
                emit({"type": "error", "request_id": request_id,
                      "code": "model_not_loaded", "message": "not loaded"})
                continue
            if "crash" in image.name:
                raise SystemExit(24)
            if "sleep" in image.name:
                time.sleep(2)
            if "badjson" in image.name:
                print("not-json", flush=True)
                continue
            returned = str(image.with_name("other.jpg")) if "wrong" in image.name else str(image.resolve())
            detections = [] if "zero" in image.name else [{
                "class_id": 0, "confidence": 0.9, "x_center": 0.5,
                "y_center": 0.5, "width": 0.25, "height": 0.25,
            }]
            if "invalid" in image.name:
                detections[0]["width"] = 2.0
            emit({"type": "prediction_result", "request_id": request_id,
                  "image": returned, "image_width": 640, "image_height": 480,
                  "detections": detections, "load_count": load_count})
        elif command == "ping":
            emit({"type": "pong", "request_id": request_id, "load_count": load_count})
        elif command == "unload_model":
            loaded = False
            emit({"type": "model_unloaded", "request_id": request_id})
        elif command == "shutdown":
            emit({"type": "shutdown_ack", "request_id": request_id})
            break
        else:
            emit({"type": "error", "request_id": request_id,
                  "code": "unknown_command", "message": "unknown"})
    except json.JSONDecodeError:
        emit({"type": "error", "request_id": "unknown",
              "code": "malformed_json", "message": "invalid JSON"})
