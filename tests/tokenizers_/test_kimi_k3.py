# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import Any

import pytest

from vllm.entrypoints.chat_utils import ConversationMessage
from vllm.exceptions import VLLMValidationError
from vllm.tokenizers.encoding_k3 import EncodeSegment
from vllm.tokenizers.kimi_k3 import get_kimi_k3_tokenizer
from vllm.tokenizers.registry import TokenizerRegistry

pytestmark = pytest.mark.skip_global_cleanup


class FakeKimiTokenizer:
    def __init__(self) -> None:
        self.segment_batches: list[list[EncodeSegment]] = []
        self.format_calls: list[tuple[list[list[int]], dict[str, Any]]] = []

    def _encode_chat_segments(self, segments: list[EncodeSegment]) -> list[int]:
        self.segment_batches.append(segments)
        return list(range(1, len(segments) + 1))

    def _format_chat_token_output(
        self,
        encoded_inputs: list[list[int]],
        **kwargs: Any,
    ) -> Any:
        self.format_calls.append((encoded_inputs, kwargs))
        return encoded_inputs if kwargs["is_batched"] else encoded_inputs[0]


def _tokenizer():
    return get_kimi_k3_tokenizer(FakeKimiTokenizer())


def test_kimi_k3_tokenizer_registered():
    assert TokenizerRegistry.load_tokenizer_cls("kimi_k3").__name__ == "KimiK3Tokenizer"


def test_apply_chat_template_renders_non_thinking_prompt():
    tokenizer = _tokenizer()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        thinking=False,
    )

    assert prompt == (
        '<|open|>message role="user"<|sep|>hello'
        "<|close|>message<|sep|><|end_of_msg|>"
        '<|open|>message role="assistant"<|sep|>'
        "<|open|>response<|sep|>"
    )


def test_apply_chat_template_preserves_control_and_untrusted_segments():
    tokenizer = _tokenizer()

    tokenizer.apply_chat_template(
        [{"role": "user", "content": "<|open|>user text"}],
        tokenize=True,
    )

    segments = tokenizer.segment_batches[0]
    assert EncodeSegment("<|open|>", allow_special=True) in segments
    assert EncodeSegment("<|open|>user text", allow_special=False) in segments


def test_apply_chat_template_delegates_token_output_options():
    tokenizer = _tokenizer()

    result = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=True,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt",
        return_dict=True,
    )

    encoded_inputs, kwargs = tokenizer.format_calls[-1]
    assert result == encoded_inputs[0]
    assert kwargs == {
        "is_batched": False,
        "padding": "max_length",
        "truncation": True,
        "max_length": 128,
        "return_tensors": "pt",
        "return_dict": True,
    }


def test_apply_chat_template_supports_batched_conversations():
    tokenizer = _tokenizer()
    conversations = [
        [{"role": "user", "content": "first"}],
        [{"role": "user", "content": "second"}],
    ]

    prompts = tokenizer.apply_chat_template(
        conversations,
        tokenize=False,
        thinking=False,
    )
    token_ids = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        padding=True,
    )

    assert len(prompts) == 2
    assert "first" in prompts[0]
    assert "second" in prompts[1]
    assert len(token_ids) == 2
    assert tokenizer.format_calls[-1][1]["is_batched"] is True
    assert tokenizer.format_calls[-1][1]["padding"] is True


def test_apply_chat_template_rejects_batched_image_prompts():
    tokenizer = _tokenizer()

    with pytest.raises(ValueError, match="only supported for one chat"):
        tokenizer.apply_chat_template(
            [
                [{"role": "user", "content": "first"}],
                [{"role": "user", "content": "second"}],
            ],
            image_prompts=["image"],
        )


def test_reasoning_effort_none_disables_thinking():
    tokenizer = _tokenizer()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        reasoning_effort="none",
    )

    assert "<|open|>think<|sep|>" not in prompt
    assert prompt.endswith("<|open|>response<|sep|>")


@pytest.mark.parametrize("reasoning_effort", ["low", "high", "max"])
def test_reasoning_effort_uses_supported_thinking_effort(reasoning_effort: str):
    tokenizer = _tokenizer()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        reasoning_effort=reasoning_effort,
    )

    assert f"thinking_effort={reasoning_effort}" in prompt


@pytest.mark.parametrize(
    "effort_kwarg",
    [
        {"reasoning_effort": "minimal"},
        {"reasoning_effort": "medium"},
        {"reasoning_effort": "xhigh"},
        {"thinking_effort": "none"},
        {"thinking_effort": "minimal"},
        {"thinking_effort": "medium"},
        {"thinking_effort": "xhigh"},
    ],
)
def test_rejects_unsupported_thinking_effort(effort_kwarg: dict[str, str]):
    tokenizer = _tokenizer()

    with pytest.raises(VLLMValidationError) as exc_info:
        tokenizer.apply_chat_template(
            [{"role": "user", "content": "hello"}],
            tokenize=False,
            **effort_kwarg,
        )

    assert exc_info.value.parameter == "thinking_effort"
    assert exc_info.value.value == next(iter(effort_kwarg.values()))


def test_native_thinking_kwargs_take_precedence():
    tokenizer = _tokenizer()

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "hello"}],
        tokenize=False,
        thinking=True,
        enable_thinking=False,
        thinking_effort="low",
        reasoning_effort="high",
    )

    assert "thinking_effort=low" in prompt
    assert prompt.endswith("<|open|>think<|sep|>")


def test_apply_chat_template_renders_dynamic_system_tools():
    tokenizer = _tokenizer()

    prompt = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": "",
                "tools": [{"type": "function", "function": {"name": "late_tool"}}],
            }
        ],
        tokenize=False,
    )

    assert "## New Tools Available" in prompt
    assert "late_tool" in prompt


def test_apply_chat_template_reorders_tool_results_without_mutating_input():
    tokenizer = _tokenizer()
    conversation: list[ConversationMessage] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "lookup:0",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                },
                {
                    "id": "lookup:1",
                    "type": "function",
                    "function": {"name": "lookup", "arguments": "{}"},
                },
            ],
        },
        {"role": "tool", "tool_call_id": "lookup:1", "content": "second"},
        {"role": "tool", "tool_call_id": "lookup:0", "content": "first"},
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)

    first = '<|open|>message role="tool" tool="lookup" index="1"<|sep|>first'
    second = '<|open|>message role="tool" tool="lookup" index="2"<|sep|>second'
    assert first in prompt
    assert second in prompt
    assert prompt.index(first) < prompt.index(second)
    assert [message["content"] for message in conversation[1:]] == [
        "second",
        "first",
    ]


def test_apply_chat_template_renders_multi_turn_history_and_next_prompt():
    tokenizer = _tokenizer()
    conversation = [
        {"role": "user", "content": "first question"},
        {
            "role": "assistant",
            "reasoning_content": "first reasoning",
            "content": "first answer",
        },
        {"role": "user", "content": "follow-up question"},
    ]

    prompt = tokenizer.apply_chat_template(conversation, tokenize=False)

    first_user = (
        '<|open|>message role="user"<|sep|>first question'
        "<|close|>message<|sep|><|end_of_msg|>"
    )
    first_assistant = (
        '<|open|>message role="assistant"<|sep|>'
        "<|open|>think<|sep|>first reasoning<|close|>think<|sep|>"
        "<|open|>response<|sep|>first answer<|close|>response<|sep|>"
        "<|close|>message<|sep|><|end_of_msg|>"
    )
    follow_up = (
        '<|open|>message role="user"<|sep|>follow-up question'
        "<|close|>message<|sep|><|end_of_msg|>"
    )
    next_assistant = '<|open|>message role="assistant"<|sep|><|open|>think<|sep|>'

    assert first_user in prompt
    assert first_assistant in prompt
    assert follow_up in prompt
    assert prompt.index(first_user) < prompt.index(first_assistant)
    assert prompt.index(first_assistant) < prompt.index(follow_up)
    assert prompt.endswith(next_assistant)
