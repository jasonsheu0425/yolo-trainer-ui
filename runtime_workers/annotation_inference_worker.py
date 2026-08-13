"""Persistent Ultralytics detection worker; stdout is JSONL protocol only."""
from __future__ import annotations

from contextlib import redirect_stdout
import gc
import math
from pathlib import Path
import sys
import traceback
from typing import Any

try:
    from annotation_inference_protocol import (
        PROTOCOL_VERSION, WORKER_VERSION, ProtocolError, decode_message,
        encode_message, error_response, validate_request,
    )
except ImportError:  # Package import used only by source tests.
    from runtime_workers.annotation_inference_protocol import (
        PROTOCOL_VERSION, WORKER_VERSION, ProtocolError, decode_message,
        encode_message, error_response, validate_request,
    )


class UltralyticsAdapter:
    def __init__(self) -> None:
        self.model: Any = None
        self.actual_device = ""

    def load(self, model_path: Path, requested_device: str) -> dict[str, Any]:
        with redirect_stdout(sys.stderr):
            from ultralytics import YOLO
            import torch

            model = YOLO(str(model_path))
            task = str(getattr(model, "task", "") or "unknown")
            if task != "detect":
                raise WorkerCommandError("unsupported_model_task", task)
            if requested_device == "auto":
                selected = "0" if torch.cuda.is_available() else "cpu"
            else:
                selected = requested_device
            if selected == "0" and not torch.cuda.is_available():
                raise WorkerCommandError("cuda_unavailable", "CUDA device 0 is unavailable")
            self.model = model
            self.actual_device = "cuda:0" if selected == "0" else "cpu"
            names = getattr(model, "names", {})
            if isinstance(names, list):
                names = dict(enumerate(names))
            classes = {str(int(key)): str(value) for key, value in dict(names).items()}
            return {
                "task": task,
                "actual_device": self.actual_device,
                "cuda_available": bool(torch.cuda.is_available()),
                "class_names": classes,
            }

    def predict(self, image_path: Path, confidence: float) -> tuple[int, int, list[dict[str, Any]]]:
        if self.model is None:
            raise WorkerCommandError("model_not_loaded", "Load a model first")
        device = "0" if self.actual_device == "cuda:0" else "cpu"
        with redirect_stdout(sys.stderr):
            results = self.model.predict(
                source=str(image_path), conf=confidence, device=device,
                verbose=False, save=False,
            )
        if not results:
            return 0, 0, []
        result = results[0]
        height, width = (int(value) for value in result.orig_shape)
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for xywhn, class_id, conf in zip(boxes.xywhn.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                values = [float(value) for value in xywhn]
                confidence_value = float(conf)
                if not all(math.isfinite(value) for value in (*values, confidence_value)):
                    continue
                detections.append({
                    "class_id": int(class_id), "confidence": confidence_value,
                    "x_center": values[0], "y_center": values[1],
                    "width": values[2], "height": values[3],
                })
        return width, height, detections

    def unload(self) -> None:
        self.model = None
        self.actual_device = ""
        gc.collect()
        try:
            with redirect_stdout(sys.stderr):
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            pass


class WorkerCommandError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def emit(payload: dict[str, Any]) -> None:
    sys.stdout.buffer.write(encode_message(payload))
    sys.stdout.buffer.flush()


def handle(adapter: UltralyticsAdapter, request: dict[str, Any]) -> bool:
    request_id = request["request_id"]
    command = request["command"]
    if command == "load_model":
        model_path = Path(request["model"])
        if not model_path.is_file() or model_path.suffix.casefold() != ".pt":
            raise WorkerCommandError("invalid_model_path", "A local .pt file is required")
        details = adapter.load(model_path.resolve(), request["device"])
        emit({"type": "model_loaded", "protocol_version": PROTOCOL_VERSION,
              "request_id": request_id, "model": str(model_path.resolve()), **details})
    elif command == "predict":
        image_path = Path(request["image"])
        if not image_path.is_file():
            raise WorkerCommandError("invalid_image_path", "Image file was not found")
        width, height, detections = adapter.predict(image_path.resolve(), float(request["confidence"]))
        emit({"type": "prediction_result", "protocol_version": PROTOCOL_VERSION,
              "request_id": request_id, "image": str(image_path.resolve()),
              "image_width": width, "image_height": height, "detections": detections})
    elif command == "unload_model":
        adapter.unload()
        emit({"type": "model_unloaded", "protocol_version": PROTOCOL_VERSION,
              "request_id": request_id})
    elif command == "ping":
        emit({"type": "pong", "protocol_version": PROTOCOL_VERSION,
              "request_id": request_id})
    elif command == "shutdown":
        adapter.unload()
        emit({"type": "shutdown_ack", "protocol_version": PROTOCOL_VERSION,
              "request_id": request_id})
        return False
    return True


def main(adapter: UltralyticsAdapter | None = None) -> int:
    adapter = adapter or UltralyticsAdapter()
    emit({"type": "hello", "protocol_version": PROTOCOL_VERSION,
          "worker_version": WORKER_VERSION})
    for raw in sys.stdin.buffer:
        request_id = "unknown"
        try:
            request = validate_request(decode_message(raw))
            request_id = request["request_id"]
            if not handle(adapter, request):
                return 0
        except (ProtocolError, WorkerCommandError) as exc:
            emit(error_response(request_id, getattr(exc, "code", "invalid_request"), str(exc)))
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            emit(error_response(request_id, "internal_worker_error", str(exc)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
