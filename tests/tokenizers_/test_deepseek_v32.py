# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tokenizers import deepseek_v32_encoding
from vllm.tokenizers.deepseek_v32 import get_deepseek_v32_tokenizer

MESSAGES = [{"role": "user", "content": "Hello"}]


class FakeHfTokenizer:
    """Minimal stand-in for the underlying HF tokenizer."""

    vocab_size = 100
    name_or_path = "fake-deepseek-v32"

    def get_added_vocab(self) -> dict[str, int]:
        return {"</think>": 100}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        return [len(text)]

    def apply_chat_template(
        self,
        conversation,
        tools=None,
        chat_template=None,
        tokenize=True,
        **kwargs,
    ):
        if tokenize:
            return [len(chat_template or "")]
        return f"custom:{chat_template}:{conversation}:{tools}"


def _tokenizer():
    return get_deepseek_v32_tokenizer(FakeHfTokenizer())


def _builtin_user_prompt(content: str = "Hello") -> str:
    return (
        deepseek_v32_encoding.bos_token
        + deepseek_v32_encoding.user_msg_template.format(content=content)
        + deepseek_v32_encoding.thinking_end_token
    )


def test_deepseek_v32_uses_builtin_encoding_by_default():
    prompt = _tokenizer().apply_chat_template(MESSAGES, tokenize=False)

    assert prompt == _builtin_user_prompt()


def test_deepseek_v32_keeps_builtin_encoding_without_template():
    prompt = _tokenizer().apply_chat_template(
        MESSAGES, chat_template=None, tokenize=False
    )

    assert prompt == _builtin_user_prompt()


def test_deepseek_v32_honors_custom_chat_template():
    rendered = _tokenizer().apply_chat_template(
        conversation=MESSAGES,
        messages=MESSAGES,
        chat_template="CUSTOM",
        tokenize=False,
    )

    assert rendered.startswith("custom:CUSTOM")
    assert _builtin_user_prompt() not in rendered


def test_deepseek_v32_forwards_tools_with_custom_chat_template():
    tools = [{"type": "function", "function": {"name": "get_weather"}}]

    rendered = _tokenizer().apply_chat_template(
        conversation=MESSAGES,
        messages=MESSAGES,
        tools=tools,
        chat_template="CUSTOM",
        tokenize=False,
    )

    assert "get_weather" in rendered


def test_deepseek_v32_tokenizes_with_custom_chat_template():
    token_ids = _tokenizer().apply_chat_template(
        conversation=MESSAGES,
        messages=MESSAGES,
        chat_template="CUSTOM",
    )

    assert token_ids == [len("CUSTOM")]
