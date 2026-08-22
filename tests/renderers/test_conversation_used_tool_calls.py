# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.renderers.online_renderer import _conversation_used_tool_calls


def test_no_messages():
    assert not _conversation_used_tool_calls(SimpleNamespace(messages=None))
    assert not _conversation_used_tool_calls(SimpleNamespace(messages=[]))


def test_plain_conversation():
    request = SimpleNamespace(
        messages=[
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
    )
    assert not _conversation_used_tool_calls(request)


def test_prior_assistant_tool_call():
    request = SimpleNamespace(
        messages=[
            {"role": "user", "content": "weather in Paris?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{}"},
                    }
                ],
            },
        ]
    )
    assert _conversation_used_tool_calls(request)


def test_prior_tool_role_message():
    request = SimpleNamespace(
        messages=[
            {"role": "user", "content": "weather in Paris?"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1"}]},
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny"},
            {"role": "user", "content": "and Berlin?"},
        ]
    )
    assert _conversation_used_tool_calls(request)


def test_assistant_message_without_tool_calls_key():
    request = SimpleNamespace(messages=[{"role": "assistant", "content": "just prose"}])
    assert not _conversation_used_tool_calls(request)
