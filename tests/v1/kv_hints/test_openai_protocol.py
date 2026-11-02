# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import msgspec
import pytest
from pydantic import ValidationError

from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.v1.kv_hints import KvHintsEnvelope, kv_hints_from_wire, kv_hints_to_wire

VALID_ENVELOPE = {
    "protocol_version": "0.1",
    "message_id": "router/message-1",
    "actions": [
        {
            "action_id": "retain-prefix",
            "action_type": "vllm.set_priority",
            "action_version": "1.0",
            "payload": {
                "target": "request_cached",
                "location": "local_g1",
                "claim_id": "router/session-1",
                "revision": 1,
                "value": 10,
                "ttl_seconds": 60,
            },
        }
    ],
}

CHAT_BASE = {
    "model": "m",
    "messages": [{"role": "user", "content": "hi"}],
}
COMPLETION_BASE = {"model": "m", "prompt": "hi"}


def test_wire_round_trip():
    envelope = kv_hints_from_wire(VALID_ENVELOPE)
    assert isinstance(envelope, KvHintsEnvelope)
    assert envelope.message_id == "router/message-1"
    assert kv_hints_to_wire(envelope) == VALID_ENVELOPE
    assert kv_hints_from_wire(None) is None
    assert kv_hints_to_wire(None) is None


def test_wire_rejects_malformed_envelope():
    with pytest.raises(ValueError, match="Invalid kv_hints payload"):
        kv_hints_from_wire({"protocol_version": "0.1"})
    bad_action = {
        "protocol_version": "0.1",
        "message_id": "m",
        "actions": [{"action_id": "a", "payload": {}}],
    }
    with pytest.raises(ValueError, match="Invalid kv_hints payload"):
        kv_hints_from_wire(bad_action)


def test_chat_request_accepts_valid_envelope():
    request = ChatCompletionRequest(**CHAT_BASE, kv_hints=VALID_ENVELOPE)
    assert request.kv_hints == VALID_ENVELOPE


def test_completion_request_accepts_valid_envelope():
    request = CompletionRequest(**COMPLETION_BASE, kv_hints=VALID_ENVELOPE)
    assert request.kv_hints == VALID_ENVELOPE


@pytest.mark.parametrize(
    "model_cls, base",
    [
        (ChatCompletionRequest, CHAT_BASE),
        (CompletionRequest, COMPLETION_BASE),
    ],
)
@pytest.mark.parametrize(
    "kv_hints",
    [
        {"message_id": "m", "actions": []},
        {**VALID_ENVELOPE, "extra_field": 1},
        {**VALID_ENVELOPE, "actions": [{"action_id": "a"}]},
        "not-an-object",
    ],
)
def test_malformed_envelope_rejected_at_api_boundary(model_cls, base, kv_hints):
    with pytest.raises(ValidationError):
        model_cls(**base, kv_hints=kv_hints)


def test_semantically_bad_action_passes_api_boundary():
    """Envelope shape is valid; action payload semantics belong to the engine."""
    envelope = {
        "protocol_version": "0.1",
        "message_id": "m",
        "actions": [
            {
                "action_id": "a",
                "action_type": "vllm.set_priority",
                "action_version": "1.0",
                "payload": {"target": "request_cached", "value": None},
            }
        ],
    }
    request = ChatCompletionRequest(**CHAT_BASE, kv_hints=envelope)
    assert request.kv_hints == envelope


def test_envelope_decodes_to_engine_types():
    envelope = kv_hints_from_wire(VALID_ENVELOPE)
    action = msgspec.convert(envelope.actions[0].payload, type=dict)
    assert action["claim_id"] == "router/session-1"
