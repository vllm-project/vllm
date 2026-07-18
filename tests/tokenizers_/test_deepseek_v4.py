# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm.entrypoints.chat_utils import parse_chat_messages
from vllm.entrypoints.openai.chat_completion.protocol import (
    ChatCompletionRequest,
    ChatMessage,
)
from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.deepseek_v4 import get_deepseek_v4_tokenizer
from vllm.tokenizers.deepseek_v4_encoding import encode_arguments_to_dsml
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


def test_deepseek_v4_defaults_to_chat_mode():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜></think>")


@pytest.mark.parametrize("kwargs", [{"thinking": True}, {"enable_thinking": True}])
def test_deepseek_v4_enables_thinking_with_compatible_kwargs(kwargs):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        **kwargs,
    )

    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜><think>")


def test_deepseek_v4_honors_official_thinking_request_field():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled"},
        }
    )
    chat_kwargs = request.apply_chat_template_kwargs(
        request.build_chat_params(None, "auto").chat_template_kwargs
    )

    prompt = _tokenizer().apply_chat_template(
        request.messages,
        tokenize=False,
        **chat_kwargs,
    )

    assert chat_kwargs["thinking"] is True
    assert chat_kwargs["enable_thinking"] is True
    assert prompt == ("<｜begin▁of▁sentence｜><｜User｜>Hello<｜Assistant｜><think>")


def test_deepseek_v4_defaults_to_official_thinking_for_openai_request():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    chat_kwargs = request.apply_chat_template_kwargs(
        request.build_chat_params(None, "auto").chat_template_kwargs
    )

    assert chat_kwargs["thinking"] is True
    assert chat_kwargs["enable_thinking"] is True


def test_deepseek_v4_preserves_official_reasoning_content_alias():
    messages = [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "reasoning_content": "because", "content": "A1"},
        {"role": "user", "content": "Q2"},
    ]

    conversation, _, _ = parse_chat_messages(
        messages,
        _model_config(),
        content_format="string",
    )

    assert conversation[1]["reasoning"] == "because"
    assert conversation[1]["reasoning_content"] == "because"


def test_deepseek_v4_response_messages_expose_reasoning_content_alias():
    message = ChatMessage(role="assistant", reasoning="because", content="answer")
    delta = DeltaMessage(reasoning="because")

    assert message.reasoning_content == "because"
    assert delta.reasoning_content == "because"
    assert (
        ChatMessage(
            role="assistant",
            reasoning_content="because",
            content="answer",
        ).reasoning
        == "because"
    )


def test_deepseek_v4_preserves_official_prefix_assistant_message():
    messages = [
        {"role": "user", "content": "Please write quick sort code"},
        {"role": "assistant", "content": "```python\n", "prefix": True},
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

    assert conversation[1]["prefix"] is True
    assert conversation[1]["wo_eos"] is True
    assert prompt.endswith("<｜Assistant｜></think>```python\n")
    assert not prompt.endswith("<｜end▁of▁sentence｜>")


def test_deepseek_v4_thinking_ignores_sampling_controls():
    request = ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled"},
            "temperature": 0.2,
            "top_p": 0.3,
            "top_k": 4,
            "presence_penalty": 1.5,
            "frequency_penalty": 1.25,
        }
    )
    chat_kwargs = request.apply_chat_template_kwargs(
        request.build_chat_params(None, "auto").chat_template_kwargs
    )

    sampling_params = request.to_sampling_params(
        16,
        {},
        chat_template_kwargs=chat_kwargs,
    )

    assert sampling_params.temperature == 1.0
    assert sampling_params.top_p == 1.0
    assert sampling_params.top_k == 0
    assert sampling_params.presence_penalty == 0.0
    assert sampling_params.frequency_penalty == 0.0


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
    assert prompt.endswith("<｜User｜>Weather?<｜Assistant｜></think>")


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
    ("tool_call", "expected_parameter"),
    [
        ({"name": "refresh", "arguments": None}, None),
        ({"name": "refresh"}, None),
        ({"name": "refresh", "arguments": ""}, None),
        (
            {"name": "refresh", "arguments": '{"target": "cache"}'},
            '<｜DSML｜parameter name="target" string="true">cache',
        ),
        (
            {"name": "refresh", "arguments": {"target": "cache"}},
            '<｜DSML｜parameter name="target" string="true">cache',
        ),
    ],
)
def test_deepseek_v4_encodes_empty_history_tool_arguments(
    tool_call, expected_parameter
):
    prompt = encode_arguments_to_dsml(tool_call)

    if expected_parameter is None:
        assert prompt == ""
    else:
        assert expected_parameter in prompt


def test_deepseek_v4_renders_openai_history_tool_call_with_null_arguments():
    messages = [
        {"role": "user", "content": "Refresh state"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "refresh",
                        "arguments": None,
                    },
                }
            ],
        },
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

    assert '<｜DSML｜invoke name="refresh">' in prompt
    assert "<｜DSML｜parameter" not in prompt


@pytest.mark.parametrize("reasoning_effort", ["minimal", "low", "medium", "high"])
def test_deepseek_v4_accepts_openai_reasoning_effort_values(reasoning_effort):
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort=reasoning_effort,
    )

    assert prompt.endswith("<｜Assistant｜><think>")
    assert "Reasoning Effort: Absolute maximum" not in prompt


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
        ("minimal", "thinking", "high"),
        ("low", "thinking", "high"),
        ("medium", "thinking", "high"),
        ("high", "thinking", "high"),
        ("xhigh", "thinking", "max"),
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


def test_deepseek_v4_preserves_reference_max_reasoning_effort():
    prompt = _tokenizer().apply_chat_template(
        [{"role": "user", "content": "Hello"}],
        tokenize=False,
        enable_thinking=True,
        reasoning_effort="max",
    )

    assert prompt.startswith(
        "<｜begin▁of▁sentence｜>Reasoning Effort: Absolute maximum"
    )


def test_deepseek_v4_maps_xhigh_to_reference_max_reasoning_effort():
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
        (1, {"thinking": True}),
        (2, {"thinking": True}),
        (3, {"thinking": True}),
        (4, {}),
    ],
)
def test_deepseek_v4_matches_reference_golden_fixtures(case_id, kwargs):
    prompt = _render_reference_case(case_id, **kwargs)

    expected = (FIXTURES_DIR / f"test_output_{case_id}.txt").read_text()
    assert prompt == expected


@pytest.mark.parametrize(
    "model",
    [
        "deepseek-ai/DeepSeek-V4-Flash",
        "deepseek-ai/DeepSeek-V4-Pro",
    ],
)
def test_deepseek_v4_official_api_defaults_to_thinking_for_v4_family(model):
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    request = ChatCompletionRequest.model_validate(
        {
            "model": model,
            "messages": [{"role": "user", "content": "Hello"}],
        }
    )
    chat_kwargs = request.apply_chat_template_kwargs(
        request.build_chat_params(None, "auto").chat_template_kwargs
    )

    assert chat_kwargs["thinking"] is True
    assert chat_kwargs["enable_thinking"] is True


def test_deepseek_v4_official_api_uses_model_config_for_family_detection():
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    request = ChatCompletionRequest.model_validate(
        {
            "model": "local-ds4-alias",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.2,
        }
    )
    model_config = SimpleNamespace(
        hf_config=SimpleNamespace(model_type="deepseek_v4", architectures=[]),
    )
    chat_kwargs = request.apply_chat_template_kwargs(
        request.build_chat_params(None, "auto").chat_template_kwargs,
        model_config=model_config,
    )

    sampling_params = request.to_sampling_params(
        16,
        {},
        chat_template_kwargs=chat_kwargs,
        model_config=model_config,
    )

    assert chat_kwargs["thinking"] is True
    assert chat_kwargs["enable_thinking"] is True
    assert sampling_params.temperature == 1.0


def test_deepseek_v4_official_api_sampling_override_can_be_disabled():
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    request = ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-ai/DeepSeek-V4-Flash",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled"},
            "deepseek_v4_sampling_override": False,
            "temperature": 0.2,
            "top_p": 0.3,
            "top_k": 4,
            "min_p": 0.05,
            "presence_penalty": 1.5,
            "frequency_penalty": 1.25,
        }
    )
    chat_kwargs = request.apply_chat_template_kwargs(
        request.build_chat_params(None, "auto").chat_template_kwargs
    )

    sampling_params = request.to_sampling_params(
        16,
        {},
        chat_template_kwargs=chat_kwargs,
    )

    assert sampling_params.temperature == 0.2
    assert sampling_params.top_p == 0.3
    assert sampling_params.top_k == 4
    assert sampling_params.min_p == 0.05
    assert sampling_params.presence_penalty == 1.5
    assert sampling_params.frequency_penalty == 1.25


def test_deepseek_v4_official_api_sampling_override_is_v4_only():
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )

    request = ChatCompletionRequest.model_validate(
        {
            "model": "deepseek-ai/DeepSeek-R1",
            "messages": [{"role": "user", "content": "Hello"}],
            "thinking": {"type": "enabled"},
            "temperature": 0.2,
            "top_p": 0.3,
            "top_k": 4,
            "min_p": 0.05,
            "presence_penalty": 1.5,
            "frequency_penalty": 1.25,
        }
    )
    chat_kwargs = request.apply_chat_template_kwargs(
        request.build_chat_params(None, "auto").chat_template_kwargs
    )

    sampling_params = request.to_sampling_params(
        16,
        {},
        chat_template_kwargs=chat_kwargs,
    )

    assert "thinking" not in chat_kwargs
    assert "enable_thinking" not in chat_kwargs
    assert sampling_params.temperature == 0.2
    assert sampling_params.top_p == 0.3
    assert sampling_params.top_k == 4
    assert sampling_params.min_p == 0.05
    assert sampling_params.presence_penalty == 1.5
    assert sampling_params.frequency_penalty == 1.25
