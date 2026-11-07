# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tokenizers.deepseek_v32 import get_deepseek_v32_tokenizer


class FakeHfTokenizer:
    def get_added_vocab(self) -> dict[str, int]:
        return {}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        return [len(text)]


def _tokenizer():
    return get_deepseek_v32_tokenizer(FakeHfTokenizer())


_ONE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the weather",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def test_deepseek_v32_tools_render_after_the_system_prompt():
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "user", "content": "Weather?"},
        ],
        tools=_ONE_TOOL,
        tokenize=False,
    )

    assert prompt.count("## Tools") == 1
    assert prompt.index("SYSTEM_MARKER") < prompt.index("## Tools")
    assert "SYSTEM_MARKER\n\n## Tools" in prompt


def test_deepseek_v32_tools_render_without_a_leading_system_message():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Weather?"}],
        tools=_ONE_TOOL,
        tokenize=False,
    )

    assert prompt.count("## Tools") == 1
    assert '"name": "get_weather"' in prompt


def test_deepseek_v32_request_tools_replace_message_tools():
    messages = [
        {
            "role": "system",
            "content": "SYSTEM_MARKER",
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "old_tool",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        },
        {"role": "user", "content": "Weather?"},
    ]

    prompt = _tokenizer().apply_chat_template(
        messages,
        tools=_ONE_TOOL,
        tokenize=False,
    )

    assert prompt.count("## Tools") == 1
    assert "get_weather" in prompt
    assert "old_tool" not in prompt
    assert "tools" not in messages[0]
