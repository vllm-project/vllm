# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai.types.responses import (
    ResponseFunctionToolCall,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponseRawMessageAndToken,
    ResponsesRequest,
    ResponsesResponse,
    _omit_openai_response_non_nullable_nulls,
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


def test_responses_response_omits_non_nullable_null_fields() -> None:
    request = ResponsesRequest(
        model="test-model",
        input="What is the horoscope? She is an Aquarius.",
        tools=[
            {
                "type": "function",
                "name": "get_horoscope",
                "description": "Get today's horoscope for an astrological sign.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "sign": {
                            "type": "string",
                            "description": (
                                "An astrological sign like Taurus or Aquarius"
                            ),
                        }
                    },
                    "required": ["sign"],
                },
            }
        ],
    )
    sampling_params = request.to_sampling_params(default_max_tokens=64)
    response = ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name="test-model",
        created_time=0,
        output=[
            ResponseReasoningItem(
                id="rs_test",
                type="reasoning",
                summary=[],
                content=[
                    ResponseReasoningTextContent(
                        text="Use get_horoscope for Aquarius.",
                        type="reasoning_text",
                    )
                ],
                encrypted_content=None,
                status=None,
            ),
            ResponseFunctionToolCall(
                id="fc_test",
                call_id="call_test",
                name="get_horoscope",
                type="function_call",
                arguments='{"sign":"Aquarius"}',
                namespace=None,
                status=None,
            ),
        ],
        status="completed",
        usage=None,
    )

    data = response.model_dump(mode="json", by_alias=True)

    # These fields are nullable in OpenAI's OpenAPI schema and should remain.
    assert data["error"] is None
    assert data["incomplete_details"] is None
    assert data["instructions"] is None
    assert data["metadata"] is None
    assert data["max_tool_calls"] is None
    assert data["previous_response_id"] is None
    assert data["prompt"] is None
    assert data["reasoning"] is None
    assert data["top_logprobs"] is None

    # These fields are optional but not nullable, so they should be omitted
    # instead of serialized as null.
    assert "text" not in data
    assert "user" not in data
    assert "kv_transfer_params" not in data
    assert "input_messages" not in data
    assert "output_messages" not in data
    assert "status" not in data["output"][0]
    assert "namespace" not in data["output"][1]
    assert "status" not in data["output"][1]
    assert "defer_loading" not in data["tools"][0]

    # strict is explicitly nullable in the OpenAPI function tool schema.
    assert data["tools"][0]["strict"] is None


def test_response_null_omission_is_position_aware() -> None:
    data = _omit_openai_response_non_nullable_nulls(
        {
            "id": "resp_test",
            "object": "response",
            "status": None,
            "text": None,
            "metadata": {
                "status": None,
                "namespace": None,
                "defer_loading": None,
                "text": None,
                "user": None,
            },
            "output": [
                {
                    "type": "function_call",
                    "status": None,
                    "namespace": None,
                    "arguments": "{}",
                    "extra": {"status": None},
                }
            ],
            "tools": [
                {
                    "type": "function",
                    "defer_loading": None,
                    "parameters": {
                        "type": "object",
                        "status": None,
                    },
                }
            ],
        }
    )

    assert data["status"] is None
    assert "text" not in data
    assert data["metadata"] == {
        "status": None,
        "namespace": None,
        "defer_loading": None,
        "text": None,
        "user": None,
    }
    assert "status" not in data["output"][0]
    assert "namespace" not in data["output"][0]
    assert data["output"][0]["extra"]["status"] is None
    assert "defer_loading" not in data["tools"][0]
    assert data["tools"][0]["parameters"]["status"] is None


def test_response_null_omission_handles_streaming_response_events() -> None:
    data = _omit_openai_response_non_nullable_nulls(
        {
            "type": "response.completed",
            "sequence_number": 1,
            "response": {
                "id": "resp_test",
                "object": "response",
                "text": None,
                "output": [{"type": "reasoning", "status": None}],
                "tools": [{"type": "function", "defer_loading": None}],
            },
        }
    )

    assert data["type"] == "response.completed"
    assert data["sequence_number"] == 1
    assert "text" not in data["response"]
    assert "status" not in data["response"]["output"][0]
    assert "defer_loading" not in data["response"]["tools"][0]


def test_response_null_omission_handles_streaming_item_events() -> None:
    data = _omit_openai_response_non_nullable_nulls(
        {
            "type": "response.output_item.done",
            "output_index": 0,
            "item": {
                "type": "function_call",
                "status": None,
                "namespace": None,
                "extra": {"status": None},
            },
        }
    )

    assert data["type"] == "response.output_item.done"
    assert "status" not in data["item"]
    assert "namespace" not in data["item"]
    assert data["item"]["extra"]["status"] is None


def test_response_null_omission_leaves_unrelated_events_unchanged() -> None:
    event = {
        "type": "response.output_text.done",
        "text": None,
        "user": None,
        "status": None,
        "payload": {"defer_loading": None},
    }

    assert _omit_openai_response_non_nullable_nulls(event) == event


def test_responses_response_keeps_populated_vllm_extension_fields() -> None:
    request = ResponsesRequest(model="test-model", input="hello")
    sampling_params = request.to_sampling_params(default_max_tokens=64)
    response = ResponsesResponse.from_request(
        request=request,
        sampling_params=sampling_params,
        model_name="test-model",
        created_time=0,
        output=[],
        status="completed",
        usage=None,
        input_messages=[
            ResponseRawMessageAndToken(
                message="rendered prompt",
                tokens=[1, 2, 3],
            )
        ],
        output_messages=[
            ResponseRawMessageAndToken(
                message="rendered output",
                tokens=[4, 5],
            )
        ],
        kv_transfer_params={"transfer_id": "kv-test"},
    )

    data = response.model_dump(mode="json", by_alias=True)

    assert data["input_messages"][0]["message"] == "rendered prompt"
    assert data["output_messages"][0]["message"] == "rendered output"
    assert data["kv_transfer_params"] == {"transfer_id": "kv-test"}
