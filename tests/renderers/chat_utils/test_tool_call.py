# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.renderers.chat_utils import ConversationMessage
from vllm.renderers.chat_utils.chat_message import _postprocess_messages


def _assistant_tool_call(arguments, name="write"):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            ],
        }
    ]


def _assistant_tool_calls(calls):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": calls,
        }
    ]


def test_tool_call_arguments_dict_passthrough(caplog):
    messages = _assistant_tool_call({"a": 1})
    _postprocess_messages(messages)
    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {"a": 1}
    assert not caplog.records


def test_tool_call_arguments_valid_object_string(caplog):
    messages = _assistant_tool_call('{"a": 1}')
    _postprocess_messages(messages)
    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {"a": 1}
    assert not caplog.records


def test_tool_call_arguments_malformed_json_small(caplog):
    bad = '{"cmd": "mkdir -p /tmp/mmadtest && cat > /tmp/mmadtest'
    messages = _assistant_tool_call(bad, name="exec")
    _postprocess_messages(messages)
    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {}

    assert len(caplog.records) == 1
    assert "coercing to an empty object" in caplog.records[0].message


def test_tool_call_arguments_malformed_json_large(caplog):
    bad = '{"filepath": "src/main.py", "contents": "' + ("x" * 18000)
    messages = _assistant_tool_call(bad, name="write")
    _postprocess_messages(messages)
    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {}

    assert len(caplog.records) == 1
    assert "coercing to an empty object" in caplog.records[0].message
    assert f"{len(bad)} chars" in caplog.records[0].message


@pytest.mark.parametrize("non_obj", ["[]", "42", "true", '"hello"', "null"])
def test_tool_call_arguments_valid_json_non_object(caplog, non_obj):
    messages = _assistant_tool_call(non_obj, name="bad_tool")
    _postprocess_messages(messages)
    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {}

    if non_obj == "null":
        # null parses to None, so no warning is emitted according to requirements
        assert not caplog.records
    else:
        assert len(caplog.records) == 1
        assert "not a JSON object" in caplog.records[0].message


@pytest.mark.parametrize("missing", [None, ""])
def test_tool_call_arguments_missing_or_empty(caplog, missing):
    messages = _assistant_tool_call(missing)
    if missing is None:
        del messages[0]["tool_calls"][0]["function"]["arguments"]
    _postprocess_messages(messages)
    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {}
    assert not caplog.records


def test_tool_call_arguments_multiple_independent(caplog):
    calls = [
        {
            "id": "call_0",
            "type": "function",
            "function": {"name": "good", "arguments": '{"a": 1}'},
        },
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "bad", "arguments": '{"cmd": "'},
        },
    ]
    messages = _assistant_tool_calls(calls)
    _postprocess_messages(messages)

    tool_calls = messages[0]["tool_calls"]
    assert tool_calls[0]["function"]["arguments"] == {"a": 1}
    assert tool_calls[1]["function"]["arguments"] == {}

    assert len(caplog.records) == 1
    assert "coercing to an empty object" in caplog.records[0].message


def test_postprocess_messages_null_arguments_string():
    """arguments="null" must not reach the chat template as Python None.

    json.loads("null") returns None, which causes Jinja2 templates that call
    tc.arguments.items() to raise 'None' has no attribute 'items'.
    The function should coerce it to {} instead.
    """
    messages: list[ConversationMessage] = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "get_current_time", "arguments": "null"},
                }
            ],
        }
    ]
    _postprocess_messages(messages)
    tool_calls = messages[0]["tool_calls"]
    assert tool_calls is not None
    assert tool_calls[0]["function"]["arguments"] == {}
