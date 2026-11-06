# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest
from openai.types.responses.response_output_message import ResponseOutputMessage
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

pytestmark = pytest.mark.skip_global_cleanup


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


def test_normalize_openresponses_input_item_stamps_type() -> None:
    """A typeless assistant item must keep its role and content.

    ``type`` is optional in the OpenResponses schema, but filling in
    ``id``/``status`` without it makes the item match ``ItemReference`` in the
    input union, which collapses it to ``{"id": ...}`` and silently drops the
    assistant turn from the conversation. Regression test for that.
    """
    item = {
        "role": "assistant",
        "content": [{"type": "output_text", "text": "hi"}],
    }

    normalized = ResponsesRequest._normalize_openresponses_input_item(item)

    assert normalized["type"] == "message"
    assert normalized["role"] == "assistant"
    assert normalized["id"].startswith("msg_")
    assert normalized["status"] == "completed"
    # annotations are filled in for output_text content
    assert normalized["content"][0]["annotations"] == []
    # the caller's dict is not mutated
    assert "type" not in item


def test_normalize_openresponses_input_item_reasoning_type() -> None:
    """An explicit reasoning item keeps its type and gets a summary default."""
    normalized = ResponsesRequest._normalize_openresponses_input_item(
        {"type": "reasoning", "content": [{"type": "reasoning_text", "text": "why"}]}
    )

    assert normalized["type"] == "reasoning"
    assert normalized["id"].startswith("rs_")
    assert normalized["status"] == "completed"
    assert normalized["summary"] == []


def test_input_item_parsing_typeless_assistant_message() -> None:
    """End to end: a typeless assistant item survives as a real message."""
    request = ResponsesRequest.model_validate(
        {
            "input": [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ]
        }
    )

    assert isinstance(request.input, list)
    (parsed,) = request.input
    assert isinstance(parsed, ResponseOutputMessage)
    assert parsed.role == "assistant"
    assert parsed.content[0].text == "hi"


def test_input_item_parsing_leaves_non_assistant_items_alone() -> None:
    """User messages are untouched — normalization only targets assistant items."""
    item = {"role": "user", "content": "hello"}
    request = ResponsesRequest.model_validate({"input": [item]})

    assert request.input == [item]


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
