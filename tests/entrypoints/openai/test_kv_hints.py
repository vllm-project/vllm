# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.entrypoints.generate.base.serving import GenerateBaseServing
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest
from vllm.exceptions import VLLMValidationError

# A router-issued envelope. Action payloads are validated by the component
# implementing the action, so only the envelope itself is checked here.
KV_FETCH_PAYLOAD = {
    "source_control_endpoint": "tcp://prefiller-0:17771",
    "block_hashes": [1234, 5678],
    "mode": "copy",
}

KV_FETCH_ENVELOPE = {
    "protocol_version": "0.1",
    "message_id": "msg-1",
    "actions": [
        {
            "action_id": "act-1",
            "action_type": "kv.fetch",
            "action_version": "1.0",
            "payload": KV_FETCH_PAYLOAD,
        }
    ],
}

GENERATE_REQUESTS = [
    ChatCompletionRequest(
        model="test-model",
        messages=[{"role": "user", "content": "hi"}],
    ),
    CompletionRequest(model="test-model", prompt="hi"),
]


@pytest.mark.parametrize("openai_request", GENERATE_REQUESTS)
def test_get_kv_hints_parses_envelope(openai_request):
    openai_request.kv_hints = KV_FETCH_ENVELOPE

    envelope = GenerateBaseServing._get_kv_hints(openai_request)

    assert envelope is not None
    assert envelope.protocol_version == "0.1"
    assert envelope.message_id == "msg-1"
    (action,) = envelope.actions
    assert action.action_type == "kv.fetch"
    # The payload is opaque to vLLM and must reach the connector unchanged.
    assert action.payload == KV_FETCH_PAYLOAD


@pytest.mark.parametrize("openai_request", GENERATE_REQUESTS)
def test_get_kv_hints_defaults_to_none(openai_request):
    openai_request.kv_hints = None

    assert GenerateBaseServing._get_kv_hints(openai_request) is None


def test_get_kv_hints_treats_empty_envelope_as_absent():
    request = CompletionRequest(model="test-model", prompt="hi", kv_hints={})

    assert GenerateBaseServing._get_kv_hints(request) is None


def test_get_kv_hints_accepts_envelope_without_actions():
    request = CompletionRequest(
        model="test-model",
        prompt="hi",
        kv_hints={"protocol_version": "0.1", "message_id": "msg-1", "actions": []},
    )

    envelope = GenerateBaseServing._get_kv_hints(request)

    assert envelope is not None
    assert envelope.actions == []


@pytest.mark.parametrize(
    "kv_hints",
    [
        {"message_id": "msg-1", "actions": []},
        {"protocol_version": "0.1", "actions": []},
        {"protocol_version": "0.1", "message_id": "msg-1"},
        {"protocol_version": "0.1", "message_id": "msg-1", "actions": {}},
        {"protocol_version": 1, "message_id": "msg-1", "actions": []},
        {
            "protocol_version": "0.1",
            "message_id": "msg-1",
            "actions": [{"action_id": "act-1"}],
        },
        {
            "protocol_version": "0.1",
            "message_id": "msg-1",
            "actions": [
                {
                    "action_id": "act-1",
                    "action_type": "kv.fetch",
                    "action_version": "1.0",
                    "payload": "not-an-object",
                }
            ],
        },
    ],
)
def test_get_kv_hints_rejects_malformed_envelope(kv_hints):
    """A malformed envelope is a client error, not a silently dropped hint."""
    request = CompletionRequest(model="test-model", prompt="hi", kv_hints=kv_hints)

    with pytest.raises(VLLMValidationError) as excinfo:
        GenerateBaseServing._get_kv_hints(request)

    assert excinfo.value.parameter == "kv_hints"
