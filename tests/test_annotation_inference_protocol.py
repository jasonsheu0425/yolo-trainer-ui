from __future__ import annotations

import pytest

from runtime_workers.annotation_inference_protocol import (
    PROTOCOL_VERSION,
    ProtocolError,
    decode_message,
    encode_message,
    new_request_id,
    validate_request,
    validate_response,
)


def test_request_round_trip_and_unique_request_ids():
    first, second = new_request_id(), new_request_id()
    assert first != second
    payload = {"command": "ping", "request_id": first}
    assert validate_request(decode_message(encode_message(payload)))["request_id"] == first


@pytest.mark.parametrize(
    ("value", "code"),
    [
        ("not json", "malformed_json"),
        ("[]", "message_not_object"),
        ('{"protocol_version": 999}', "protocol_version_mismatch"),
    ],
)
def test_malformed_messages_are_rejected(value, code):
    with pytest.raises(ProtocolError) as raised:
        decode_message(value)
    assert raised.value.code == code


def test_request_validation_rejects_unknown_missing_id_and_bad_confidence():
    with pytest.raises(ProtocolError, match="Unknown"):
        validate_request({"protocol_version": PROTOCOL_VERSION, "command": "boom"})
    with pytest.raises(ProtocolError, match="request_id"):
        validate_request({"protocol_version": PROTOCOL_VERSION, "command": "ping"})
    with pytest.raises(ProtocolError) as raised:
        validate_request({
            "protocol_version": PROTOCOL_VERSION, "command": "predict",
            "request_id": "1", "image": "a.jpg", "confidence": 2.0,
        })
    assert raised.value.code == "invalid_confidence"


def test_response_validation_requires_known_type_and_request_id():
    with pytest.raises(ProtocolError):
        validate_response({"protocol_version": PROTOCOL_VERSION, "type": "noise"})
    with pytest.raises(ProtocolError):
        validate_response({"protocol_version": PROTOCOL_VERSION, "type": "pong"})


def test_hello_is_versioned_without_request_id():
    value = validate_response({
        "protocol_version": PROTOCOL_VERSION,
        "type": "hello",
        "worker_version": "0.13.0",
    })
    assert value["protocol_version"] == 1
