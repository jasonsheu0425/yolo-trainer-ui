"""Pure-stdlib JSONL protocol shared by the GUI controller and runtime worker."""
from __future__ import annotations

import json
import math
from typing import Any
from uuid import uuid4


PROTOCOL_VERSION = 1
WORKER_VERSION = "0.13.0"
REQUEST_TYPES = frozenset({"load_model", "predict", "unload_model", "ping", "shutdown"})
RESPONSE_TYPES = frozenset({
    "hello", "model_loaded", "prediction_result", "pong", "model_unloaded",
    "error", "shutdown_ack",
})


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def new_request_id() -> str:
    return uuid4().hex


def encode_message(message: dict[str, Any]) -> bytes:
    payload = dict(message)
    payload.setdefault("protocol_version", PROTOCOL_VERSION)
    return (json.dumps(payload, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def decode_message(raw: str | bytes) -> dict[str, Any]:
    try:
        text = raw.decode("utf-8") if isinstance(raw, bytes) else raw
        payload = json.loads(text)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("malformed_json", str(exc)) from exc
    if not isinstance(payload, dict):
        raise ProtocolError("message_not_object", "JSONL message must be an object")
    if payload.get("protocol_version") != PROTOCOL_VERSION:
        raise ProtocolError("protocol_version_mismatch", "Unsupported protocol version")
    return payload


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    command = payload.get("command")
    if command not in REQUEST_TYPES:
        raise ProtocolError("unknown_command", "Unknown worker command")
    request_id = payload.get("request_id")
    if not isinstance(request_id, str) or not request_id:
        raise ProtocolError("invalid_request_id", "request_id is required")
    if command == "load_model":
        if not isinstance(payload.get("model"), str) or not payload["model"]:
            raise ProtocolError("invalid_model_path", "model path is required")
        if payload.get("device") not in {"auto", "0", "cpu"}:
            raise ProtocolError("invalid_device", "device must be auto, 0, or cpu")
    if command == "predict":
        if not isinstance(payload.get("image"), str) or not payload["image"]:
            raise ProtocolError("invalid_image_path", "image path is required")
        confidence = payload.get("confidence")
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ProtocolError("invalid_confidence", "confidence must be numeric")
        if not math.isfinite(float(confidence)) or not 0.01 <= float(confidence) <= 1.0:
            raise ProtocolError("invalid_confidence", "confidence must be within 0.01..1.0")
    return payload


def validate_response(payload: dict[str, Any]) -> dict[str, Any]:
    response_type = payload.get("type")
    if response_type not in RESPONSE_TYPES:
        raise ProtocolError("unknown_response", "Unknown worker response type")
    if response_type != "hello":
        request_id = payload.get("request_id")
        if not isinstance(request_id, str) or not request_id:
            raise ProtocolError("invalid_request_id", "Response request_id is required")
    return payload


def error_response(request_id: str, code: str, message: str) -> dict[str, Any]:
    return {
        "type": "error",
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "code": code,
        "message": message,
    }
