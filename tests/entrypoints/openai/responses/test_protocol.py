# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from openai_harmony import (
    Message,
)

from vllm.entrypoints.openai.responses.protocol import (
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


def test_chat_message_reasoning_content_serialization() -> None:
    from vllm.entrypoints.openai.chat_completion.protocol import ChatMessage

    msg = ChatMessage(role="assistant", content="Answer", reasoning_content="Thinking process")
    dumped = msg.model_dump()
    assert dumped["role"] == "assistant"
    assert dumped["content"] == "Answer"
    assert dumped["reasoning_content"] == "Thinking process"
    assert "tool_calls" not in dumped

    msg_none = ChatMessage(role="assistant", content="Answer")
    dumped_none = msg_none.model_dump()
    assert "reasoning_content" not in dumped_none

