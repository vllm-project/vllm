# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import pytest

from vllm.renderers.tool_call_hints import render_tool_call_hints
from vllm.tokenizers import get_tokenizer

MODEL_NAME = "Qwen/Qwen3-0.6B"

CONVERSATION = [{"role": "user", "content": "What is the weather in Paris?"}]

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather in a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_items",
            "description": "Count things",
            "parameters": {
                "type": "object",
                "properties": {"n": {"type": "integer"}},
                "required": ["n"],
            },
        },
    },
]


@pytest.fixture(scope="module")
def tokenizer():
    return get_tokenizer(MODEL_NAME)


def test_one_hint_per_tool(tokenizer):
    hints = render_tool_call_hints(tokenizer, CONVERSATION, TOOLS)
    assert len(hints) == len(TOOLS)
    for hint, tool in zip(hints, TOOLS):
        assert all(isinstance(token_id, int) for token_id in hint)
        assert tool["function"]["name"] in tokenizer.decode(hint)


def test_hints_do_not_include_the_conversation(tokenizer):
    for hint in render_tool_call_hints(tokenizer, CONVERSATION, TOOLS):
        assert "What is the weather in Paris?" not in tokenizer.decode(hint)


def test_placeholders_follow_the_parameter_types(tokenizer):
    hints = render_tool_call_hints(tokenizer, CONVERSATION, TOOLS)
    assert '"city": "X"' in tokenizer.decode(hints[0])
    assert '"n": 0' in tokenizer.decode(hints[1])


def test_no_tools(tokenizer):
    assert render_tool_call_hints(tokenizer, CONVERSATION, []) == []
    assert render_tool_call_hints(tokenizer, CONVERSATION, None) == []
