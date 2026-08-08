# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest

from vllm.tokenizers import deepseek_v4_encoding, deepseek_v32_encoding

ENCODING_MODULES = [deepseek_v32_encoding, deepseek_v4_encoding]


@pytest.mark.parametrize(
    "encoding_module",
    ENCODING_MODULES,
    ids=["deepseek_v32", "deepseek_v4"],
)
def test_encode_messages_scans_last_user_once_per_conversation(
    monkeypatch: pytest.MonkeyPatch,
    encoding_module,
):
    calls = 0
    original_find_last_user_index = encoding_module.find_last_user_index

    def counted_find_last_user_index(messages):
        nonlocal calls
        calls += 1
        return original_find_last_user_index(messages)

    monkeypatch.setattr(
        encoding_module,
        "find_last_user_index",
        counted_find_last_user_index,
    )

    messages = [{"role": "user", "content": "Hello"}]
    messages.extend({"role": "assistant", "content": "Hi"} for _ in range(8))

    encoding_module.encode_messages(messages, thinking_mode="chat")

    assert calls == 1


@pytest.mark.parametrize(
    "encoding_module",
    ENCODING_MODULES,
    ids=["deepseek_v32", "deepseek_v4"],
)
def test_encode_messages_preserves_small_chat_prompt(encoding_module):
    prompt = encoding_module.encode_messages(
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "assistant", "content": "Again"},
        ],
        thinking_mode="chat",
    )

    assert prompt == (
        "<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>"
        "Hi<｜end▁of▁sentence｜>Again<｜end▁of▁sentence｜>"
    )
