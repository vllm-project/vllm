# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.entrypoints.chat_utils import parse_chat_messages
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.registry import TokenizerRegistry

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "deepseek_v4"


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
    return get_deepseek_v4_tokenizer(FakeHfTokenizer())


def _model_config():
    return SimpleNamespace(
        multimodal_config=None,
        allowed_local_media_path="",
        allowed_media_domains=None,
        enable_prompt_embeds=False,
    )


def _load_reference_case(case_id: int):
    data = json.loads((FIXTURES_DIR / f"test_input_{case_id}.json").read_text())
    if isinstance(data, dict):
        return data["messages"], data.get("tools")
    return data, None


def _render_reference_case(case_id: int, **kwargs):
    messages, tools = _load_reference_case(case_id)
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )
    return _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tools=tools,
        tokenize=False,
        **kwargs,
    )


def test_deepseek_v4_tokenizer_registered():
    assert TokenizerRegistry.load_tokenizer_cls("deepseek_v4").__name__ == (
        "DeepseekV4Tokenizer"
    )
    assert RENDERER_REGISTRY.load_renderer_cls("deepseek_v4").__name__ == (
        "DeepseekV4Renderer"
    )


def test_deepseek_v4_defaults_to_thinking_with_high_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )
    assert prompt.endswith("<｜Assistant｜><think>")


@pytest.mark.parametrize("kwargs", [{"thinking": True}, {"enable_thinking": True}])
def test_deepseek_v4_enables_thinking_with_compatible_kwargs(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )
    assert prompt.endswith("<｜Assistant｜><think>")


@pytest.mark.parametrize("kwargs", [{"thinking": False}, {"enable_thinking": False}])
def test_deepseek_v4_explicitly_disables_thinking(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


def test_deepseek_v4_uses_v4_tool_prompt_from_request_tools():
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]

    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Weather?"}],
        tools=tools,
        tokenize=False,
    )

    assert "## Tools" in prompt
    assert "<｜DSML｜tool_calls>" in prompt
    assert "</｜DSML｜tool_calls>" in prompt
    assert "function_calls" not in prompt
    assert '"name": "get_weather"' in prompt
    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )
    assert prompt.endswith("<｜User｜>Weather?<｜Assistant｜><think>")


def test_deepseek_v4_renders_parsed_history_tool_arguments():
    messages = [
        {"role": "user", "content": "List the repo"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "str_replace_editor",
                        "arguments": '{"command": "view", "path": "/testbed"}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": "file list",
        },
    ]
    tools = [
        {
            "type": "function",
            "function": {
                "name": "str_replace_editor",
                "description": "Edit files",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "path": {"type": "string"},
                    },
                    "required": ["command", "path"],
                },
            },
        }
    ]
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )

    prompt = _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tools=tools,
        tokenize=False,
    )

    assert '<｜DSML｜parameter name="command" string="true">view' in prompt
    assert '<｜DSML｜parameter name="path" string="true">/testbed' in prompt
    assert 'parameter name="arguments"' not in prompt


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_prefix"),
    [
        ("low", "<｜begin▁of▁sentence｜><｜User｜>Hello"),
        ("high", "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"),
        ("max", "<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum"),
    ],
)
def test_deepseek_v4_renders_0731_reasoning_effort_prompts(
    reasoning_effort, expected_prefix
):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert prompt.endswith("<｜Assistant｜><think>")
    assert prompt.startswith(expected_prefix)


def test_deepseek_v4_none_reasoning_effort_disables_thinking():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="none",
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


@pytest.mark.parametrize(
    ("reasoning_effort", "expected_mode", "expected_effort"),
    [
        ("none", "chat", None),
        ("minimal", "thinking", "low"),
        ("low", "thinking", "low"),
        ("medium", "thinking", "low"),
        ("high", "thinking", "high"),
        ("xhigh", "thinking", "high"),
        ("max", "thinking", "max"),
        ("unexpected", "thinking", "high"),
    ],
)
def test_deepseek_v4_maps_compatible_thinking_reasoning_effort_values(
    monkeypatch: pytest.MonkeyPatch,
    reasoning_effort,
    expected_mode,
    expected_effort,
):
    captured_kwargs = []

    def fake_encode_messages(messages, **kwargs):
        captured_kwargs.append(kwargs)
        return "prompt"

    monkeypatch.setattr(
        "vllm.tokenizers.deepseek_v4.encode_messages",
        fake_encode_messages,
    )

    _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert captured_kwargs[-1]["thinking_mode"] == expected_mode
    assert captured_kwargs[-1]["reasoning_effort"] == expected_effort


def test_deepseek_v4_renders_0731_max_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="max",
    )

    assert prompt.startswith("<｜begin▁of▁sentence｜>Reasoning Effort: Beyond maximum")


def test_deepseek_v4_maps_xhigh_to_high_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="xhigh",
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )


@pytest.mark.parametrize(
    ("case_id", "kwargs"),
    [
        (1, {"thinking": True, "reasoning_effort": "low"}),
        (2, {"thinking": True, "reasoning_effort": "low"}),
        (3, {"thinking": True, "reasoning_effort": "low"}),
        (4, {"thinking": False}),
    ],
)
def test_deepseek_v4_matches_reference_golden_fixtures(case_id, kwargs):
    prompt = _render_reference_case(case_id, **kwargs)

    expected = (FIXTURES_DIR / f"test_output_{case_id}.txt").read_text()
    assert prompt == expected


def test_deepseek_v4_wo_eos_survives_message_parsing():
    """`wo_eos` set on an assistant message must reach the encoder.

    Regression test: previously `parse_chat_messages` stripped the field
    so the encoder always rendered the EOS-terminated template.
    """
    messages = [
        {"role": "user", "content": "tell me a saying"},
        {
            "role": "assistant",
            "content": "An apple a day, keeps",
            "wo_eos": True,
        },
    ]
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )

    assert conversation[-1].get("wo_eos") is True

    prompt = _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tokenize=False,
    )

    # Prefill: must end with the assistant content, NOT with an EOS token.
    assert prompt.endswith("An apple a day, keeps")
    assert "<｜end▁of▁sentence｜>" not in prompt.split("<｜Assistant｜>")[-1]


def test_deepseek_v4_continue_final_message_skips_eos():
    """`continue_final_message=True` must render the final assistant
    message without a trailing EOS token (HF semantics)."""
    messages = [
        {"role": "user", "content": "tell me a saying"},
        {"role": "assistant", "content": "An apple a day, keeps"},
    ]
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )
    prompt = _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tokenize=False,
        continue_final_message=True,
    )

    assert prompt.endswith("An apple a day, keeps")
    assert "<｜end▁of▁sentence｜>" not in prompt.split("<｜Assistant｜>")[-1]


def test_deepseek_v4_continue_final_message_requires_assistant_last():
    """Mirrors HF: error when the last message isn't from the assistant."""
    messages = [{"role": "user", "content": "hi"}]
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )
    with pytest.raises(ValueError, match="continue_final_message"):
        _tokenizer().apply_chat_template(
            conversation=conversation,
            messages=messages,
            tokenize=False,
            continue_final_message=True,
        )


def test_deepseek_v4_default_assistant_message_keeps_eos():
    """Sanity check: without wo_eos / continue_final_message, the last
    assistant message still terminates with EOS."""
    messages = [
        {"role": "user", "content": "tell me a saying"},
        {"role": "assistant", "content": "An apple a day, keeps"},
    ]
    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )
    prompt = _tokenizer().apply_chat_template(
        conversation=conversation,
        messages=messages,
        tokenize=False,
    )
    assert prompt.endswith("<｜end▁of▁sentence｜>")
