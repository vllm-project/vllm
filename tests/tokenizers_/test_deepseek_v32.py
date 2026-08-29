# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tokenizers.deepseek_v32 import get_deepseek_v32_tokenizer


class FakeHfTokenizer:
    vocab_size = 100

    def get_added_vocab(self) -> dict[str, int]:
        return {"</think>": 100}

    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
        **kwargs,
    ) -> list[int]:
        self.last_encode = (text, add_special_tokens, kwargs)
        return [len(text)]


def _tokenizer():
    return get_deepseek_v32_tokenizer(FakeHfTokenizer())


_ONE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "tool_beta",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


def test_deepseek_v32_tools_render_after_the_system_prompt():
    """Tools attach to the existing system message, not a newly inserted one."""
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "system", "content": "SYSTEM_MARKER"},
            {"role": "user", "content": "Weather?"},
        ],
        tools=_ONE_TOOL,
        tokenize=False,
        thinking=True,
    )

    assert prompt.count("## Tools") == 1
    assert prompt.index("SYSTEM_MARKER") < prompt.index("## Tools")
    assert "SYSTEM_MARKER\n\n## Tools" in prompt


def test_deepseek_v32_tools_attach_to_a_leading_developer_message():
    """A leading developer message carries the tools.

    `render_message` emits the tools block before the developer content, so this
    checks attachment and a single rendering rather than ordering.
    """
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "developer", "content": "DEVELOPER_MARKER"},
            {"role": "user", "content": "Weather?"},
        ],
        tools=_ONE_TOOL,
        tokenize=False,
    )

    assert prompt.count("## Tools") == 1
    assert "DEVELOPER_MARKER" in prompt
    assert "tool_beta" in prompt


def test_deepseek_v32_tools_still_rendered_without_a_leading_system_message():
    """With no system/developer message to attach to, one is inserted."""
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Weather?"}],
        tools=_ONE_TOOL,
        tokenize=False,
        thinking=True,
    )

    assert prompt.count("## Tools") == 1
    assert "tool_beta" in prompt


def test_deepseek_v32_request_tools_replace_message_tools():
    """Request-level tools win; rendering both would emit the block twice."""
    prompt = _tokenizer().apply_chat_template(
        [
            {
                "role": "system",
                "content": "SYSTEM_MARKER",
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "tool_alpha",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            {"role": "user", "content": "Weather?"},
        ],
        tools=_ONE_TOOL,
        tokenize=False,
        thinking=True,
    )

    assert prompt.count("## Tools") == 1
    assert "tool_beta" in prompt
    assert "tool_alpha" not in prompt


def test_deepseek_v32_caller_messages_are_not_mutated():
    """Attaching tools must not write into the caller's messages."""
    messages = [
        {"role": "system", "content": "SYSTEM_MARKER"},
        {"role": "user", "content": "Weather?"},
    ]

    _tokenizer().apply_chat_template(
        messages, tools=_ONE_TOOL, tokenize=False, thinking=True
    )

    assert "tools" not in messages[0]
