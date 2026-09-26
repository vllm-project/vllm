# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from types import SimpleNamespace

import pytest
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
    serialize_message,
    serialize_messages,
)


def test_serialize_message() -> None:
    dict_value = {"a": 1, "b": "2"}
    assert serialize_message(dict_value) == dict_value

    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 1"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_message(msg) == msg_value


def test_serialize_messages() -> None:
    assert serialize_messages(None) is None
    assert serialize_messages([]) is None

    dict_value = {"a": 3, "b": "4"}
    msg_value = {
        "role": "assistant",
        "name": None,
        "content": [{"type": "text", "text": "Test 2"}],
        "channel": "analysis",
    }
    msg = Message.from_dict(msg_value)
    assert serialize_messages([msg, dict_value]) == [msg_value, dict_value]


@pytest.mark.parametrize(
    "value,expected",
    [
        (True, True),
        (False, False),
        (None, False),  # explicit null must resolve to the documented default
    ],
)
def test_responses_response_background_null_resolves_to_default(value, expected):
    """An explicit ``"background": null`` in the request must not fail
    response construction (``ResponsesResponse.background`` is a
    non-optional bool)."""
    request = ResponsesRequest.model_validate(
        {"input": "Hello", "model": "test-model", "background": value}
    )
    sampling_params = request.to_sampling_params(default_max_tokens=16)
    response = ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name="test-model",
        created_time=0,
        output=[],
        status="completed",
    )
    assert response.background is expected


@pytest.mark.parametrize(
    "value,expected",
    [
        ("auto", "auto"),
        ("disabled", "disabled"),
        (None, "disabled"),  # explicit null must resolve to the default
    ],
)
def test_responses_response_truncation_null_resolves_to_default(value, expected):
    """An explicit ``"truncation": null`` in the request must not fail
    response construction (``ResponsesResponse.truncation`` is a
    non-optional literal)."""
    request = ResponsesRequest.model_validate(
        {"input": "Hello", "model": "test-model", "truncation": value}
    )
    sampling_params = request.to_sampling_params(default_max_tokens=16)
    response = ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name="test-model",
        created_time=0,
        output=[],
        status="completed",
    )
    assert response.truncation == expected


@pytest.mark.parametrize(
    "value,expected_truncate",
    [
        ("auto", -1),
        ("disabled", None),
        (None, None),  # explicit null must behave like the default, not "auto"
    ],
)
def test_responses_request_truncation_tok_params(value, expected_truncate):
    request = ResponsesRequest.model_validate(
        {"input": "Hello", "model": "test-model", "truncation": value}
    )
    tok_params = request.build_tok_params(SimpleNamespace(max_model_len=2048))
    assert tok_params.truncate_prompt_tokens == expected_truncate
