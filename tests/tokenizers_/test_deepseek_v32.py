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
