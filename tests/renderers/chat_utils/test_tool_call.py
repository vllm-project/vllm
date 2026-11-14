# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from copy import deepcopy

import pytest
from transformers import LlamaConfig

from vllm.config import ModelConfig
from vllm.renderers.chat_utils import (
    ConversationMessage,
    parse_chat_messages,
    parse_chat_messages_async,
)
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


@pytest.fixture
def tool_result_model_config(tmp_path):
    LlamaConfig(
        architectures=["LlamaForCausalLM"], max_position_embeddings=128
    ).save_pretrained(tmp_path)
    return ModelConfig(str(tmp_path), max_model_len=128)


def _tool_result_messages(call_ids, result_ids):
    messages = _assistant_tool_calls(
        [
            {
                "id": call_id,
                "type": "function",
                "function": {"name": "echo_payload", "arguments": {}},
            }
            for call_id in call_ids
        ]
    )
    messages.extend(
        {
            "role": "tool",
            "tool_call_id": result_id,
            "content": {"call_A": "7c92e1", "call_B": "3f84d6"}.get(result_id, "other"),
        }
        for result_id in result_ids
    )
    return messages


@pytest.mark.asyncio
@pytest.mark.parametrize("is_async", [False, True])
@pytest.mark.parametrize("content_format", ["string", "openai"])
@pytest.mark.parametrize("result_ids", [("call_A", "call_B"), ("call_B", "call_A")])
async def test_tool_results_follow_call_order(
    tool_result_model_config, is_async, content_format, result_ids
):
    """An ID-less template must see each result in its original call position."""
    messages = _tool_result_messages(("call_A", "call_B"), result_ids)
    original = deepcopy(messages)
    if is_async:
        conversation, _, _ = await parse_chat_messages_async(
            messages, tool_result_model_config, content_format
        )
    else:
        conversation, _, _ = parse_chat_messages(
            messages, tool_result_model_config, content_format
        )
    assert [
        (message["tool_call_id"], message["content"])
        for message in conversation
        if message["role"] == "tool"
    ] == [("call_A", "7c92e1"), ("call_B", "3f84d6")]
    assert messages == original


@pytest.mark.parametrize(
    "call_ids, result_ids, separator",
    [
        (("call_A", "call_B"), ("call_B",), None),
        (("call_A", "call_B"), ("call_B", "call_B"), None),
        (("call_A", "call_A"), ("call_A", "call_A"), None),
        (("call_A", "call_B"), ("call_B", "call_C"), None),
        (("call_A", "call_B"), ("call_B", "call_A", "call_A"), None),
        (("call_A", None), ("call_B", "call_A"), None),
        (("call_A", ""), ("call_B", "call_A"), None),
        (("call_A", "call_B"), (None, "call_A"), None),
        (("call_A", "call_B"), ("", "call_A"), None),
        (("call_A", "call_B"), ("call_B", "call_A"), "user"),
        (("call_A", "call_B"), ("call_B", "call_A"), "assistant"),
    ],
    ids=[
        "partial",
        "duplicate-result",
        "duplicate-call",
        "unknown-result",
        "extra-result",
        "missing-call-id",
        "empty-call-id",
        "missing-result-id",
        "empty-result-id",
        "intervening-user",
        "intervening-assistant",
    ],
)
def test_ambiguous_tool_results_keep_received_order(
    tool_result_model_config, call_ids, result_ids, separator
):
    messages = _tool_result_messages(call_ids, result_ids)
    if separator:
        messages.insert(2, {"role": separator, "content": "intervening turn"})
    expected = deepcopy(messages)
    conversation, _, _ = parse_chat_messages(
        messages, tool_result_model_config, content_format="string"
    )
    assert conversation == expected
    assert messages == expected
