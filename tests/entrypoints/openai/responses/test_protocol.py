# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from openai_harmony import (
    Message,
)
from pydantic import ValidationError

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
    serialize_message,
    serialize_messages,
)
from vllm.exceptions import VLLMValidationError


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


def test_responses_input_assistant_message_without_type_field() -> None:
    """Regression test for issue #63310.

    Multi-turn ``/v1/responses`` requests that replay a previous assistant turn
    as a message item *without* a ``"type"`` field (with ``output_text`` content
    parts, as clients such as ECA send) were rejected with 239 validation
    errors even though the OpenAI API accepts this shape. The root cause was
    that ``input_item_parsing`` only coerced assistant messages to
    ``ResponseOutputMessage`` when ``type == "message"``; a ``type``-less item
    fell through to ``EasyInputMessageParam``, whose content parts only allow
    ``input_text``.
    """
    body = {
        "model": "m",
        "input": [
            {"role": "user", "content": "My name is Alice."},
            {
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Nice to meet you, Alice."}
                ],
            },
            {"role": "user", "content": "What is my name?"},
        ],
    }

    # Previously raised pydantic.ValidationError (239 errors).
    request = ResponsesRequest.model_validate(body)
    parsed = request.input[1]
    dumped = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
    assert dumped["role"] == "assistant"
    assert dumped["content"][0]["type"] == "output_text"
    assert dumped["content"][0]["text"] == "Nice to meet you, Alice."


def test_responses_input_assistant_message_with_explicit_type_still_valid() -> None:
    """Control: the same assistant item WITH an explicit ``type: message``."""
    body = {
        "model": "m",
        "input": [
            {"role": "user", "content": "My name is Alice."},
            {
                "role": "assistant",
                "type": "message",
                "content": [
                    {"type": "output_text", "text": "Nice to meet you, Alice."}
                ],
            },
        ],
    }
    request = ResponsesRequest.model_validate(body)
    assert len(request.input) == 2


def test_responses_input_assistant_string_content_without_type_still_valid() -> None:
    """String content is a valid ``EasyInputMessageParam`` and must not be
    coerced to ``ResponseOutputMessage`` when the ``type`` field is absent."""
    body = {"model": "m", "input": [{"role": "assistant", "content": "hi"}]}
    request = ResponsesRequest.model_validate(body)
    assert len(request.input) == 1
