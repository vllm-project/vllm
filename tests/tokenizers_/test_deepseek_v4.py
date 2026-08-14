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


def _thinking_prompt_with_reasoning(reasoning):
    """Render a tool-calling conversation whose first assistant turn has `reasoning`."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    assistant = {"role": "assistant", "content": "On it."}
    if reasoning is not None:
        assistant["reasoning"] = reasoning

    return _tokenizer().apply_chat_template(
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Weather?"},
            assistant,
            {"role": "user", "content": "And tomorrow?"},
        ],
        tools=tools,
        tokenize=False,
        reasoning_effort="high",
    )


@pytest.mark.parametrize("reasoning", ["", None])
def test_deepseek_v4_no_empty_thinking_block_for_reasoningless_turn(reasoning):
    """An assistant turn without reasoning is encoded chat-style.

    Declaring tools disables `drop_thinking`, so every prior assistant turn keeps its
    thinking block. When the client did not send reasoning back for a turn there is
    nothing to put in that block, and emitting `<think></think>` produces a sequence
    that appears in none of the reference encoding's expected outputs.
    """
    prompt = _thinking_prompt_with_reasoning(reasoning)

    assert "<think></think>" not in prompt
    assert "<｜Assistant｜></think>On it." in prompt
    # The generation prompt must still open a real thinking block.
    assert prompt.endswith("<｜Assistant｜><think>")


def test_deepseek_v4_keeps_thinking_block_when_reasoning_present():
    """Turns that do carry reasoning are unchanged."""
    prompt = _thinking_prompt_with_reasoning("Let me check the forecast.")

    assert "<｜Assistant｜><think>Let me check the forecast.</think>On it." in prompt
    assert "<think></think>" not in prompt
    assert prompt.endswith("<｜Assistant｜><think>")
