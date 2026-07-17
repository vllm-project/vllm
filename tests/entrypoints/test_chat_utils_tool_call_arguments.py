# SPDX-License-Identifier: Apache-2.0
"""Tests for tool-call `arguments` handling in `_postprocess_messages`.

Regression test for #47761: an assistant tool call whose `arguments` string is
not valid JSON (e.g. a docstring with unescaped triple quotes) must not raise
an uncaught `json.JSONDecodeError` and 400 the whole request.

Target path in the repo: tests/entrypoints/test_chat_utils_tool_call_arguments.py
"""
import json

import pytest

from vllm.entrypoints.chat_utils import _postprocess_messages


def _assistant_tool_call(arguments):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_0",
                    "type": "function",
                    "function": {"name": "write", "arguments": arguments},
                }
            ],
        }
    ]


def test_valid_json_arguments_parsed_to_dict():
    messages = _assistant_tool_call('{"path": "a.py", "content": "x"}')
    _postprocess_messages(messages)
    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {"path": "a.py", "content": "x"}


def test_invalid_json_arguments_do_not_raise():
    # A docstring-bearing tool call whose serialized arguments are not valid
    # JSON (unescaped triple quotes). This is the #47761 payload shape.
    bad = '{"path": "tests/test_mcp.py", "content": """\ndocstring\n"""}'

    # Sanity: the payload really is invalid JSON.
    with pytest.raises(json.JSONDecodeError):
        json.loads(bad)

    messages = _assistant_tool_call(bad)
    # Must not raise.
    _postprocess_messages(messages)
    # The original string is preserved instead of crashing the request.
    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert args == bad


def test_missing_arguments_default_to_empty_dict():
    messages = _assistant_tool_call("")
    _postprocess_messages(messages)
    args = messages[0]["tool_calls"][0]["function"]["arguments"]
    assert args == {}
