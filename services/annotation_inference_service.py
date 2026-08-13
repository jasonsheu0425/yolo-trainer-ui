"""App-side model-assisted annotation workflow over a persistent JSONL worker."""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
import math
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, QTimer, Signal
from PySide6.QtGui import QImageReader

from core.runtime_manager import RuntimeManager
from domain.annotation import BoundingBox, ModelPrediction
from persistence.annotation_metadata_store import (
    AnnotationReportStore,
    dataset_identity,
    utc_now,
)
from runtime_workers.annotation_inference_protocol import (
    PROTOCOL_VERSION,
    new_request_id,
)
from services.annotation_service import AnnotationService
from workers.annotation_inference_controller import InferenceWorkerController


MODEL_LOAD_TIMEOUT_MS = 180_000
PREDICTION_TIMEOUT_MS = 120_000


class InferenceState(str, Enum):
    UNLOADED = "unloaded"
    STARTING_WORKER = "starting_worker"
    LOADING = "loading"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPING = "stopping"


class AnnotationInferenceService(QObject):
    state_changed = Signal(str)
    model_loaded = Signal(object)
    prediction_ready = Signal(object, str)
    error = Signal(str, str)
    log = Signal(str)
    batch_progress = Signal(object)
    batch_finished = Signal(object)

    def __init__(
        self,
        runtime: RuntimeManager,
        annotations: AnnotationService,
        controller: InferenceWorkerController | None = None,
        report_store: AnnotationReportStore | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.runtime = runtime
        self.annotations = annotations
        self.controller = controller or InferenceWorkerController()
        self.report_store = report_store or AnnotationReportStore()
        self.state = InferenceState.UNLOADED
        self.model_path: Path | None = None
        self.model_identity: dict[str, Any] = {}
        self.model_classes: dict[int, str] = {}
        self.actual_device = ""
        self.cuda_available: bool | None = None
        self.compatibility = "unknown"
        self.compatibility_override = False
        self._load_request: dict[str, Any] | None = None
        self._pending: dict[str, dict[str, Any]] = {}
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._timed_out)
        self._batch: dict[str, Any] | None = None
        self._batch_cancelled = False
        self.controller.message_received.connect(self._message)
        self.controller.log_received.connect(self.log)
        self.controller.protocol_failed.connect(self._protocol_failed)
        self.controller.exited.connect(self._worker_exited)

    @property
    def model_usable(self) -> bool:
        return self.state is InferenceState.READY and (
            self.compatibility == "exact_match" or self.compatibility_override
        )

    @property
    def batch_running(self) -> bool:
        return self._batch is not None

    def runtime_info(self) -> dict[str, Any]:
        return self.runtime.discover_python(validate=False)

    def load_model(self, model: str | Path, device: str = "auto") -> bool:
        path = Path(model).expanduser()
        if not path.is_file() or path.suffix.casefold() != ".pt":
            self._fail("invalid_model_path", "A trusted local .pt model is required")
            return False
        if device not in {"auto", "0", "cpu"}:
            self._fail("invalid_device", "Unsupported inference device")
            return False
        runtime = self.runtime_info()
        if not runtime.get("available"):
            self._fail("runtime_missing", "YOLO runtime Python was not found")
            return False
        if self.controller.running:
            self.shutdown()
        self.model_path = path.resolve()
        stat = self.model_path.stat()
        self.model_identity = {
            "filename": self.model_path.name,
            "path": str(self.model_path),
            "size": stat.st_size,
            "modified_time": stat.st_mtime,
        }
        request_id = new_request_id()
        self._load_request = {
            "command": "load_model",
            "request_id": request_id,
            "model": str(self.model_path),
            "device": device,
        }
        self.compatibility_override = False
        self._set_state(InferenceState.STARTING_WORKER)
        try:
            self.controller.start(
                str(runtime["program"]), list(runtime.get("prefix_args", []))
            )
        except (OSError, RuntimeError) as exc:
            self._fail("worker_start_failed", str(exc))
            return False
        self._timer.start(MODEL_LOAD_TIMEOUT_MS)
        return True

    def override_class_mismatch(self) -> None:
        if self.compatibility in {"id_match_name_mismatch", "unknown"}:
            self.compatibility_override = True
            self.state_changed.emit(self.state.value)

    def refresh_class_compatibility(self) -> str:
        self.compatibility = self._class_compatibility()
        self.compatibility_override = False
        if self.model_path is not None:
            self.model_loaded.emit(self.model_summary())
        return self.compatibility

    def predict_current(self, confidence: float) -> bool:
        document = self.annotations.document
        if not self.model_usable or document is None:
            self._fail("model_not_ready", "A compatible model is not ready")
            return False
        return self._send_prediction(document.image_path, confidence, "current")

    def batch_plan(self) -> dict[str, Any]:
        existing = 0
        unreadable = 0
        eligible: list[Path] = []
        for image in self.annotations.images:
            label = self.annotations.store.resolve_label_path(image)
            if label.exists():
                existing += 1
            elif not QImageReader(str(image)).canRead():
                unreadable += 1
            else:
                eligible.append(image)
        return {
            "eligible": eligible,
            "eligible_count": len(eligible),
            "existing_skipped": existing,
            "unreadable_skipped": unreadable,
            "total": len(self.annotations.images),
        }

    def start_batch(self, confidence: float) -> bool:
        if not self.model_usable or self.annotations.dataset is None:
            self._fail("model_not_ready", "A compatible model is not ready")
            return False
        if self.annotations.document and self.annotations.document.dirty:
            self._fail("dirty_annotation", "Save or discard the current annotation first")
            return False
        plan = self.batch_plan()
        now = utc_now()
        self._batch = {
            "dataset_id": dataset_identity(
                self.annotations.dataset.yaml_path, self.annotations.dataset.root
            ),
            "model": self.model_identity,
            "split": self.annotations.split,
            "confidence": float(confidence),
            "started_at": now,
            "completed_at": "",
            "status": "running",
            "eligible": plan["eligible_count"],
            "processed": 0,
            "created": 0,
            "no_detection": 0,
            "skipped": plan["existing_skipped"] + plan["unreadable_skipped"],
            "errors": [],
            "outcomes": [],
            "remaining": plan["eligible_count"],
            "queue": list(plan["eligible"]),
        }
        self._batch_cancelled = False
        self._batch_progress()
        if not self._batch["queue"]:
            self._finish_batch("completed")
            return True
        self._predict_next_batch()
        return True

    def cancel_batch(self, *, hard: bool = False) -> None:
        if self._batch is None:
            return
        self._batch_cancelled = True
        if hard:
            self.controller.kill()
            self._finish_batch("cancelled")

    def unload_model(self) -> bool:
        if not self.controller.running:
            self._set_state(InferenceState.UNLOADED)
            return True
        request_id = new_request_id()
        self._pending[request_id] = {"purpose": "unload"}
        self._set_state(InferenceState.STOPPING)
        try:
            self.controller.send(
                {"command": "unload_model", "request_id": request_id}
            )
        except RuntimeError as exc:
            self._fail("worker_not_running", str(exc))
            return False
        self._timer.start(PREDICTION_TIMEOUT_MS)
        return True

    def shutdown(self) -> None:
        if not self.controller.running:
            self._set_state(InferenceState.UNLOADED)
            return
        self._set_state(InferenceState.STOPPING)
        request_id = new_request_id()
        try:
            self.controller.send({"command": "shutdown", "request_id": request_id})
        except RuntimeError:
            pass
        if not self.controller.wait_for_finished(2500):
            self.controller.terminate()
            if not self.controller.wait_for_finished(1000):
                self.controller.kill()
                self.controller.wait_for_finished(1000)
        self._timer.stop()
        self._pending.clear()
        self._set_state(InferenceState.UNLOADED)

    def _send_prediction(self, image: Path, confidence: float, purpose: str) -> bool:
        if not math.isfinite(confidence) or not 0.01 <= confidence <= 1.0:
            self._fail("invalid_confidence", "Confidence must be within 0.01..1.0")
            return False
        request_id = new_request_id()
        expected = str(image.resolve())
        self._pending[request_id] = {
            "purpose": purpose,
            "image": expected,
            "confidence": float(confidence),
        }
        try:
            self.controller.send(
                {
                    "command": "predict",
                    "request_id": request_id,
                    "image": expected,
                    "confidence": float(confidence),
                }
            )
        except RuntimeError as exc:
            self._pending.pop(request_id, None)
            self._fail("worker_not_running", str(exc))
            return False
        self._set_state(InferenceState.BUSY)
        self._timer.start(PREDICTION_TIMEOUT_MS)
        return True

    def _message(self, payload: dict[str, Any]) -> None:
        response_type = payload["type"]
        if response_type == "hello":
            if payload.get("protocol_version") != PROTOCOL_VERSION:
                self._fail("protocol_version_mismatch", "Worker protocol mismatch")
                self.controller.kill()
                return
            if self._load_request is not None:
                self._set_state(InferenceState.LOADING)
                request = self._load_request
                self._pending[request["request_id"]] = {"purpose": "load"}
                self.controller.send(request)
            return
        request_id = str(payload.get("request_id", ""))
        context = self._pending.pop(request_id, None)
        if context is None and response_type != "shutdown_ack":
            self._fail("unexpected_response", "Unknown or stale request_id")
            return
        self._timer.stop()
        if response_type == "model_loaded":
            if payload.get("task") != "detect":
                self._fail("unsupported_model_task", str(payload.get("task", "unknown")))
                return
            self.model_classes = {
                int(key): str(value)
                for key, value in dict(payload.get("class_names", {})).items()
            }
            self.actual_device = str(payload.get("actual_device", ""))
            cuda_available = payload.get("cuda_available")
            self.cuda_available = (
                cuda_available if isinstance(cuda_available, bool) else None
            )
            self.compatibility = self._class_compatibility()
            self._set_state(InferenceState.READY)
            self.model_loaded.emit(self.model_summary())
        elif response_type == "prediction_result":
            self._handle_prediction(payload, context or {})
        elif response_type == "model_unloaded":
            self.model_path = None
            self.model_classes = {}
            self.actual_device = ""
            self.cuda_available = None
            self.compatibility = "unknown"
            self._set_state(InferenceState.UNLOADED)
        elif response_type == "error":
            self._handle_error(payload, context or {})
        elif response_type == "shutdown_ack":
            self.controller.request_shutdown()

    def _handle_prediction(
        self, payload: dict[str, Any], context: dict[str, Any]
    ) -> None:
        expected = str(context.get("image", ""))
        try:
            actual = str(Path(str(payload.get("image", ""))).resolve())
        except OSError:
            actual = ""
        if actual.casefold() != expected.casefold():
            self._fail("image_identity_mismatch", "Worker response image did not match")
            return
        try:
            predictions = self._parse_detections(payload.get("detections", []))
        except ValueError as exc:
            self._fail("invalid_prediction_values", str(exc))
            return
        purpose = context.get("purpose")
        if purpose == "current":
            document = self.annotations.document
            if document is None or str(document.image_path.resolve()).casefold() != expected.casefold():
                self._fail("image_identity_mismatch", "Current image changed")
                return
            self._set_state(InferenceState.READY)
            self.prediction_ready.emit(predictions, expected)
            return
        if purpose == "batch":
            self._consume_batch(Path(expected), predictions)

    def _parse_detections(self, values: Any) -> list[ModelPrediction]:
        if not isinstance(values, list):
            raise ValueError("detections_not_list")
        generated_at = datetime.now(timezone.utc).isoformat()
        identity = str(self.model_identity.get("path", ""))
        predictions: list[ModelPrediction] = []
        class_count = len(self.annotations.dataset.classes) if self.annotations.dataset else 0
        for value in values:
            if not isinstance(value, dict):
                raise ValueError("detection_not_object")
            try:
                box = BoundingBox(
                    int(value["class_id"]), float(value["x_center"]),
                    float(value["y_center"]), float(value["width"]),
                    float(value["height"]),
                )
                prediction = ModelPrediction(
                    box, float(value["confidence"]), identity, generated_at
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("malformed_detection") from exc
            if not prediction.is_valid(class_count):
                raise ValueError("prediction_out_of_range")
            predictions.append(prediction)
        return predictions

    def _handle_error(self, payload: dict[str, Any], context: dict[str, Any]) -> None:
        code = str(payload.get("code", "worker_error"))
        message = str(payload.get("message", code))
        if code == "cuda_unavailable":
            self.cuda_available = False
        if context.get("purpose") == "batch" and self._batch is not None:
            image = self._relative_image(Path(str(context.get("image", ""))))
            self._batch["processed"] += 1
            self._batch["errors"].append({"image": image, "code": code})
            self._batch["outcomes"].append({"image": image, "outcome": "error", "code": code})
            self._batch["remaining"] = len(self._batch["queue"])
            self._batch_progress()
            self._predict_next_batch()
            return
        self._fail(code, message)

    def _consume_batch(self, image: Path, predictions: list[ModelPrediction]) -> None:
        if self._batch is None:
            return
        outcome = "no_detection"
        try:
            path = self.annotations.save_generated_predictions(image, predictions)
            if path is not None:
                self._batch["created"] += 1
                outcome = "created"
                if self.annotations.last_metadata_warning:
                    self._batch["errors"].append(
                        {
                            "image": self._relative_image(image),
                            "code": "metadata_save_failed",
                        }
                    )
            else:
                self._batch["no_detection"] += 1
        except (OSError, ValueError) as exc:
            outcome = "error"
            self._batch["errors"].append(
                {"image": self._relative_image(image), "code": str(exc)}
            )
        self._batch["processed"] += 1
        self._batch["outcomes"].append(
            {"image": self._relative_image(image), "outcome": outcome}
        )
        self._batch["remaining"] = len(self._batch["queue"])
        self._batch_progress()
        self._predict_next_batch()

    def _predict_next_batch(self) -> None:
        if self._batch is None:
            return
        if self._batch_cancelled:
            self._finish_batch("cancelled")
            return
        if not self._batch["queue"]:
            self._finish_batch("completed")
            return
        image = self._batch["queue"].pop(0)
        self._batch["remaining"] = len(self._batch["queue"]) + 1
        self._send_prediction(image, self._batch["confidence"], "batch")

    def _finish_batch(self, status: str) -> None:
        if self._batch is None:
            return
        self._batch["status"] = status
        self._batch["completed_at"] = utc_now()
        self._batch["remaining"] = max(
            self._batch["eligible"] - self._batch["processed"], 0
        )
        report = {key: value for key, value in self._batch.items() if key != "queue"}
        try:
            report["report_path"] = str(self.report_store.save(report))
        except OSError as exc:
            report["report_error"] = str(exc)
        self._batch = None
        if self.controller.running:
            self._set_state(InferenceState.READY)
        self.batch_finished.emit(report)

    def _batch_progress(self) -> None:
        if self._batch is None:
            return
        self.batch_progress.emit(
            {key: value for key, value in self._batch.items() if key != "queue"}
        )

    def _relative_image(self, image: Path) -> str:
        dataset = self.annotations.dataset
        if dataset is None:
            return str(image)
        try:
            return image.resolve().relative_to(dataset.root.resolve()).as_posix()
        except ValueError:
            return str(image.resolve())

    def _class_compatibility(self) -> str:
        if self.annotations.dataset is None or not self.model_classes:
            return "unknown"
        dataset_classes = self.annotations.dataset.classes
        if set(self.model_classes) != set(dataset_classes):
            return "class_count_mismatch"
        if self.model_classes == dataset_classes:
            return "exact_match"
        return "id_match_name_mismatch"

    def model_summary(self) -> dict[str, Any]:
        return {
            "model": str(self.model_path or ""),
            "model_identity": self.model_identity,
            "actual_device": self.actual_device,
            "cuda_available": self.cuda_available,
            "class_names": self.model_classes,
            "compatibility": self.compatibility,
        }

    def _timed_out(self) -> None:
        self._pending.clear()
        self._fail("inference_timeout", "Inference worker timed out")
        self.controller.kill()

    def _protocol_failed(self, code: str, message: str) -> None:
        self._fail(code, message)
        self.controller.kill()

    def _worker_exited(self, exit_code: int, _exit_status: int) -> None:
        if self.state in {InferenceState.STOPPING, InferenceState.UNLOADED}:
            return
        if self._batch is not None:
            self._batch["errors"].append(
                {"image": "", "code": "worker_crashed", "exit_code": exit_code}
            )
            self._finish_batch("failed")
        self._fail("worker_crashed", f"Worker exited with code {exit_code}")

    def _set_state(self, state: InferenceState) -> None:
        self.state = state
        self.state_changed.emit(state.value)

    def _fail(self, code: str, message: str) -> None:
        self._timer.stop()
        self._set_state(InferenceState.ERROR)
        self.error.emit(code, message)
