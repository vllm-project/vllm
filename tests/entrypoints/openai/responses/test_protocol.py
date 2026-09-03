# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai_harmony import (
    Message,
)

from vllm.entrypoints.generate.base.protocol import PerRequestMetrics
from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    ResponsesResponse,
    serialize_message,
    serialize_messages,
)
from vllm.sampling_params import SamplingParams


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


def test_response_serializes_per_request_metrics() -> None:
    metrics = PerRequestMetrics(
        time_to_first_token_ms=200.0,
        generation_time_ms=1000.0,
        queue_time_ms=100.0,
        mean_itl_ms=250.0,
        tokens_per_second=4.166666666666667,
    )
    response = ResponsesResponse.from_request(
        request=ResponsesRequest(model="test-model", input="hello"),
        sampling_params=SamplingParams(max_tokens=8),
        model_name="test-model",
        created_time=0,
        output=[],
        status="completed",
        metrics=metrics,
    )

    assert response.model_dump(mode="json", by_alias=True)["metrics"] == {
        "time_to_first_token_ms": 200.0,
        "generation_time_ms": 1000.0,
        "queue_time_ms": 100.0,
        "mean_itl_ms": 250.0,
        "tokens_per_second": 4.166666666666667,
        "speculative_decoding": None,
    }


def test_response_omits_absent_per_request_metrics() -> None:
    response = ResponsesResponse.from_request(
        request=ResponsesRequest(model="test-model", input="hello"),
        sampling_params=SamplingParams(max_tokens=8),
        model_name="test-model",
        created_time=0,
        output=[],
        status="completed",
    )

    assert "metrics" not in response.model_dump(mode="json", by_alias=True)
