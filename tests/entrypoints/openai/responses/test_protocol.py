# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
    serialize_message,
    serialize_messages,
)
from vllm.sampling_params import SamplingParams


def _make_request() -> ResponsesRequest:
    return ResponsesRequest.model_validate({"model": "m", "input": "hi"})


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


def _make_request() -> "ResponsesRequest":
    return ResponsesRequest.model_validate({"model": "m", "input": "hi"})


def test_incomplete_response_defaults_to_max_output_tokens() -> None:
    """Backward-compatible default: incomplete -> max_output_tokens."""
    response = ResponsesResponse.from_request(
        _make_request(),
        sampling_params=SamplingParams(),
        model_name="m",
        created_time=0,
        output=[],
        status="incomplete",
    )
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "max_output_tokens"


def test_completed_response_has_no_incomplete_details() -> None:
    response = ResponsesResponse.from_request(
        _make_request(),
        sampling_params=SamplingParams(),
        model_name="m",
        created_time=0,
        output=[],
        status="completed",
    )
    assert response.incomplete_details is None


def test_incomplete_reason_can_be_specified_explicitly() -> None:
    """Callers may pass the reason explicitly (e.g. derived from
    finish_reason); it must be honored verbatim."""
    response = ResponsesResponse.from_request(
        _make_request(),
        sampling_params=SamplingParams(),
        model_name="m",
        created_time=0,
        output=[],
        status="incomplete",
        incomplete_reason="content_filter",
    )
    assert response.incomplete_details is not None
    assert response.incomplete_details.reason == "content_filter"
