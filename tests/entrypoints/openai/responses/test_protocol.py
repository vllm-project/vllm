# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from openai_harmony import (
    Message,
)
from pydantic import ValidationError

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesInputTokensRequest,
    ResponsesRequest,
    serialize_message,
    serialize_messages,
)
from vllm.exceptions import VLLMValidationError


@pytest.mark.parametrize(
    ("payload", "expected_input"),
    [({}, []), ({"input": None}, []), ({"input": ""}, "")],
)
def test_input_token_count_accepts_omitted_or_null_input(
    payload, expected_input
) -> None:
    request = ResponsesInputTokensRequest.model_validate(payload).to_responses_request()
    assert request.input == expected_input
    assert request.max_output_tokens is None
    assert not request.store


def test_input_token_count_forwards_only_declared_fields() -> None:
    request = ResponsesInputTokensRequest.model_validate(
        {
            "input": "Hi",
            "chat_template_kwargs": {"enable_thinking": False},
            "include": ["message.output_text.logprobs"],
            "prompt": {"id": "pmpt_1"},
        }
    ).to_responses_request()
    assert request.chat_template_kwargs == {"enable_thinking": False}
    assert request.include is None


def test_input_token_count_preserves_function_call_items() -> None:
    request = ResponsesInputTokensRequest.model_validate(
        {
            "input": [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city":"Paris"}',
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Sunny",
                },
            ]
        }
    ).to_responses_request()
    assert isinstance(request.input, list)
    assert request.input[0].type == "function_call"
    assert request.input[1]["type"] == "function_call_output"


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


def test_custom_tool_grammar_format_rejected() -> None:
    grammar = {"type": "grammar", "syntax": "lark", "definition": 'start: "pwd"'}
    with pytest.raises(VLLMValidationError) as exc_info:
        ResponsesRequest(
            input="hi", tools=[{"type": "custom", "name": "emit", "format": grammar}]
        )
    assert exc_info.value.parameter == "tools"

    request = ResponsesRequest(
        input="hi",
        tools=[{"type": "custom", "name": "emit", "format": {"type": "text"}}],
    )
    assert request.tools is not None and request.tools[0].type == "custom"


@pytest.mark.parametrize(
    "tools",
    [
        "abc",
        {"type": "custom"},
        [None],
        [{"type": "custom", "name": "x", "format": "text"}],
    ],
)
def test_malformed_tools_rejected_by_field_validation(tools) -> None:
    """Malformed tools bodies fail pydantic field validation (4xx), not an
    AttributeError inside the custom-format validator (500)."""
    with pytest.raises(ValidationError):
        ResponsesRequest(input="hi", tools=tools)
