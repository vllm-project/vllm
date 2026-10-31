# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.tokenizers.deepseek_v32 import get_deepseek_v32_tokenizer


class FakeHfTokenizer:
    """Minimal tokenizer stub that records encode calls."""

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
    """Build a wrapped DeepSeek V3.2 tokenizer from the fake stub."""
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
    """Request-level tools attach to the existing system message, not a new
    one.

    The reference layout is `{system_content}\\n\\n## Tools ...`; inserting a
    fresh system message renders the whole tools block ahead of the system
    prompt.
    """
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


def test_deepseek_v32_tools_render_after_developer_prompt():
    """Request-level tools attach to a leading developer message the same
    way they attach to a system message."""
    prompt = _tokenizer().apply_chat_template(
        [
            {"role": "developer", "content": "DEV_MARKER"},
            {"role": "user", "content": "Weather?"},
        ],
        tools=_ONE_TOOL,
        tokenize=False,
    )

    assert prompt.count("## Tools") == 1
    assert prompt.index("DEV_MARKER") < prompt.index("## Tools")


def test_deepseek_v32_tools_still_rendered_without_a_leading_system_message():
    """With no system/developer message to attach to, one is inserted."""
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Weather?"}],
        tools=_ONE_TOOL,
        tokenize=False,
    )

    assert prompt.count("## Tools") == 1
    assert '"name": "tool_beta"' in prompt


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
                            "name": "OLD_TOOL",
                            "parameters": {"type": "object", "properties": {}},
                        },
                    }
                ],
            },
            {"role": "user", "content": "Weather?"},
        ],
        tools=_ONE_TOOL,
        tokenize=False,
    )

    assert prompt.count("## Tools") == 1
    assert "tool_beta" in prompt
    assert "OLD_TOOL" not in prompt
