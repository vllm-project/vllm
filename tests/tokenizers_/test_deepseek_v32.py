# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tokenizers.deepseek_v32_encoding import encode_messages

SYSTEM = {"role": "system", "content": "You are helpful."}
USER = {"role": "user", "content": "Hi"}


def test_trailing_system_renders_before_assistant_transition():
    prompt = encode_messages(
        [SYSTEM, USER, {"role": "system", "content": "Be brief."}],
        thinking_mode="thinking",
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜>You are helpful."
        "<｜User｜>HiBe brief.<｜Assistant｜><think>"
    )


def test_system_before_assistant_stays_out_of_the_reply():
    prompt = encode_messages(
        [
            SYSTEM,
            USER,
            {"role": "system", "content": "Note."},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "More"},
        ],
        thinking_mode="thinking",
    )

    assert "<｜User｜>HiNote.<｜Assistant｜>" in prompt
    assert "<｜Assistant｜></think>Hello!" in prompt


@pytest.mark.parametrize(
    "messages",
    [
        [SYSTEM, USER],
        [
            USER,
            {"role": "system", "content": "Note."},
            {"role": "user", "content": "More"},
        ],
    ],
)
def test_unaffected_conversations_are_unchanged(messages):
    prompt = encode_messages(messages, thinking_mode="thinking")

    assert prompt.startswith("<｜begin▁of▁sentence｜>")
    assert prompt.endswith("<think>")


def test_system_turn_after_a_context_boundary_is_kept():
    context = [SYSTEM, USER]

    prompt = encode_messages(
        [{"role": "system", "content": "Later instruction."}],
        thinking_mode="thinking",
        context=context,
    )

    assert "Later instruction." in prompt


def test_folded_system_turn_keeps_its_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]

    prompt = encode_messages(
        [USER, {"role": "system", "content": "Be brief.", "tools": tools}],
        thinking_mode="thinking",
    )

    assert "get_weather" in prompt
    assert prompt.index("Be brief.") < prompt.index("<｜Assistant｜>")
