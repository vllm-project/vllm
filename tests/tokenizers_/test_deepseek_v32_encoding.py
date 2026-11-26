# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from copy import deepcopy

import pytest

from vllm.tokenizers.deepseek_v32_encoding import encode_messages


@pytest.mark.parametrize("mode", ["chat", "thinking"])
def test_developer_message_with_tool_history(mode):
    messages = [
        {"role": "developer", "content": "Answer in French."},
        {"role": "user", "content": "Check the weather."},
        {
            "role": "assistant",
            "content": "",
            "reasoning": "Check weather first.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {
                        "name": "weather",
                        "arguments": "{}",
                    },
                }
            ],
        },
        {"role": "tool", "content": "Sunny, 25 C."},
        {"role": "user", "content": "Summarize it."},
    ]
    original = deepcopy(messages)

    prompt = encode_messages(messages, thinking_mode=mode)

    assert prompt.count("Answer in French.") == 1
    assert prompt.count("<result>Sunny, 25 C.</result>") == 1
    assert "Summarize it." in prompt
    assert messages == original


@pytest.mark.parametrize("mode", ["chat", "thinking"])
def test_user_suffix_with_developer_in_context(mode):
    context = [
        {"role": "developer", "content": "Answer in French."},
        {"role": "user", "content": "Hello."},
        {"role": "assistant", "content": "Bonjour."},
    ]
    messages = [{"role": "user", "content": "Next question."}]
    original = deepcopy((context, messages))

    prompt = encode_messages(
        messages,
        thinking_mode=mode,
        context=context,
    )

    ending = "<think>" if mode == "thinking" else "</think>"
    assert prompt == "<｜User｜>Next question.<｜Assistant｜>" + ending
    assert (context, messages) == original
