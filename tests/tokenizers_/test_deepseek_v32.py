# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from tests.tokenizers_.test_deepseek_v4 import FakeHfTokenizer
from vllm.tokenizers.deepseek_v32 import get_deepseek_v32_tokenizer


def render(messages, **kwargs):
    tokenizer = get_deepseek_v32_tokenizer(FakeHfTokenizer())
    return tokenizer.apply_chat_template(messages, tokenize=False, **kwargs)


def test_add_generation_prompt_false_omits_final_assistant_prefix():
    prompt = render(
        [{"role": "user", "content": "What is 2+2?"}],
        add_generation_prompt=False,
    )

    assert prompt == "<｜begin▁of▁sentence｜><｜User｜>What is 2+2?"


def test_add_generation_prompt_opens_a_new_turn_after_final_assistant():
    prompt = render(
        [
            {"role": "user", "content": "Say hello."},
            {"role": "assistant", "content": "Hello."},
        ],
        add_generation_prompt=True,
    )

    assert prompt.endswith(
        "<｜Assistant｜></think>Hello.<｜end▁of▁sentence｜><｜Assistant｜></think>"
    )


def test_continue_final_message_keeps_assistant_message_open():
    prompt = render(
        [
            {"role": "user", "content": "Complete this sentence."},
            {"role": "assistant", "content": "The answer is "},
        ],
        continue_final_message=True,
    )

    assert prompt.endswith("<｜Assistant｜></think>The answer is ")
    assert "<｜end▁of▁sentence｜>" not in prompt


def test_continue_final_message_requires_final_assistant_message():
    with pytest.raises(ValueError, match="last message is not from the assistant"):
        render(
            [{"role": "user", "content": "What is 2+2?"}],
            continue_final_message=True,
        )
