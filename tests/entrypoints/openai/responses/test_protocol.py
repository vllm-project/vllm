# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
    ResponsesRequest,
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
    data = ResponsesRequest.input_item_parsing(
        {
            "input": [
                {
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "hi"}],
                }
            ]
        }
    )

    (parsed,) = data["input"]
    assert isinstance(parsed, ResponseOutputMessage)
    assert parsed.role == "assistant"
    assert parsed.content[0].text == "hi"


def test_input_item_parsing_leaves_non_assistant_items_alone() -> None:
    """User messages are untouched — normalization only targets assistant items."""
    item = {"role": "user", "content": "hello"}
    data = ResponsesRequest.input_item_parsing({"input": [item]})

    assert data["input"] == [item]
