# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.exceptions import VLLMValidationError
from vllm.renderers import ChatParams
from vllm.renderers.kimi_k3 import (
    KimiK3Renderer,
    _merge_k3_media_io_kwargs,
    _preserve_malformed_tool_arguments,
)
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry


class StubTokenizer:
    """Stands in for the model's TikTokenTokenizer.

    Records the kwargs it is called with and returns fixed token ids, so tests
    can assert how the renderer drives ``apply_chat_template`` without the real
    (git-LFS) tiktoken vocabulary.
    """

    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.calls: list[dict[str, Any]] = []
        self.conversations: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, conversation, **kwargs) -> list[int]:
        self.conversations.append(conversation)
        self.calls.append(kwargs)
        return list(self.token_ids)


@dataclass
class MockHFConfig:
    model_type: str = "kimi_k3"


@dataclass
class MockModelConfig:
    runner_type: str = "generate"
    is_multimodal_model: bool = False
    multimodal_config: Any = None
    hf_config: MockHFConfig = field(default_factory=MockHFConfig)
    allowed_local_media_path: str = ""
    allowed_media_domains: Any = None
    enable_prompt_embeds: bool = False
    renderer_num_workers: int = 1


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


def _make_renderer(tokenizer: StubTokenizer) -> KimiK3Renderer:
    config = MockVllmConfig(MockModelConfig(), MockParallelConfig())
    return KimiK3Renderer(config, tokenizer)


def test_kimi_k3_registered():
    assert RENDERER_REGISTRY.load_renderer_cls("kimi_k3").__name__ == "KimiK3Renderer"
    assert (
        TokenizerRegistry.load_tokenizer_cls("kimi_k3").__name__ == "CachedHfTokenizer"
    )


def test_k3_media_io_defaults_preserve_original_mode():
    # Default: K3 keeps the original image mode (no background flattening).
    assert _merge_k3_media_io_kwargs(None) == {"image": {"image_mode": None}}

    # Server-/request-level values take precedence over the K3 default.
    assert _merge_k3_media_io_kwargs({"image": {"image_mode": "RGB"}}) == {
        "image": {"image_mode": "RGB"}
    }

    # Unrelated image kwargs are merged with the default.
    assert _merge_k3_media_io_kwargs(
        {"image": {"rgba_background_color": (0, 0, 0)}}
    ) == {"image": {"image_mode": None, "rgba_background_color": (0, 0, 0)}}


def test_apply_chat_template_forces_tokenize_and_pins_return_dict():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    tools = [{"type": "function", "function": {"name": "search"}}]
    params = ChatParams(
        chat_template_kwargs={"tools": tools, "tokenize": False, "thinking": True}
    )

    token_ids = renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}], params
    )

    assert token_ids == [7, 8, 9]
    kwargs = tokenizer.calls[-1]
    # tokenize is forced on even though the request asked for False, so K3 keeps
    # the special-vs-ordinary token distinction instead of re-tokenizing a string.
    assert kwargs["tokenize"] is True
    # return_dict is pinned False so we always get a flat list of ids.
    assert kwargs["return_dict"] is False
    assert kwargs["tools"] == tools
    assert kwargs["thinking"] is True


def test_apply_chat_template_translates_standard_thinking_kwargs():
    # Standard enable_thinking/reasoning_effort kwargs must be translated
    # to K3's native thinking/thinking_effort.
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "none"}
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["thinking"] is False
    assert "thinking_effort" not in kwargs
    assert "enable_thinking" not in kwargs
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("reasoning_effort", ["low", "high", "max"])
def test_apply_chat_template_translates_supported_reasoning_effort(
    reasoning_effort: str,
):
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(chat_template_kwargs={"reasoning_effort": reasoning_effort})

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["thinking_effort"] == reasoning_effort
    assert "reasoning_effort" not in kwargs


@pytest.mark.parametrize("reasoning_effort", ["minimal", "medium", "xhigh"])
def test_apply_chat_template_rejects_unsupported_reasoning_effort(
    reasoning_effort: str,
):
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(chat_template_kwargs={"reasoning_effort": reasoning_effort})

    with pytest.raises(VLLMValidationError, match="thinking_effort") as exc_info:
        renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert exc_info.value.parameter == "thinking_effort"
    assert exc_info.value.value == reasoning_effort
    assert tokenizer.calls == []


def test_apply_chat_template_validates_canonical_native_thinking_effort():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={
            "thinking_effort": "low",
            "reasoning_effort": "medium",
        }
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert tokenizer.calls[-1]["thinking_effort"] == "low"


@pytest.mark.parametrize("thinking_effort", ["none", "minimal", "medium", "xhigh"])
def test_apply_chat_template_rejects_unsupported_native_thinking_effort(
    thinking_effort: str,
):
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(chat_template_kwargs={"thinking_effort": thinking_effort})

    with pytest.raises(VLLMValidationError, match="thinking_effort") as exc_info:
        renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert exc_info.value.parameter == "thinking_effort"
    assert exc_info.value.value == thinking_effort
    assert tokenizer.calls == []


def test_apply_chat_template_native_k3_kwargs_take_precedence():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={
            "thinking": True,
            "enable_thinking": False,
            "thinking_effort": "low",
            "reasoning_effort": "high",
        }
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["thinking"] is True
    assert kwargs["thinking_effort"] == "low"


def test_apply_chat_template_adds_k3_api_metadata():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    response_format = {"type": "json_object"}
    params = ChatParams(
        tool_choice="required",
        response_format=response_format,
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    kwargs = tokenizer.calls[-1]
    assert kwargs["tool_choice"] == "required"
    assert kwargs["response_format"] == response_format


def test_apply_chat_template_auto_tool_choice_keeps_template_kwarg():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={"tool_choice": "required"},
        tool_choice="auto",
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert tokenizer.calls[-1]["tool_choice"] == "required"


def test_apply_chat_template_omits_tool_choice_without_tools():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)

    renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}], ChatParams(tool_choice=None)
    )

    assert "tool_choice" not in tokenizer.calls[-1]


def test_render_messages_returns_token_prompt():
    renderer = _make_renderer(StubTokenizer([1, 2, 3]))

    conversation, prompt = renderer.render_messages(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [1, 2, 3]}
    assert "multi_modal_data" not in prompt
    assert conversation[0]["role"] == "user"


def test_render_messages_derives_private_xtml_tool_attrs():
    tokenizer = StubTokenizer([1, 2, 3])
    renderer = _make_renderer(tokenizer)

    conversation, _ = renderer.render_messages(
        [
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
            {
                "role": "tool",
                "tool_call_id": "lookup:1",
                "tool": "client-supplied-name",
                "index": 99,
                "content": "second",
            },
            {
                "role": "tool",
                "tool_call_id": "lookup:0",
                "content": "first",
            },
        ],
        ChatParams(),
    )

    assert [message["content"] for message in conversation[1:]] == [
        "first",
        "second",
    ]
    assert conversation[1]["tool"] == "lookup"
    assert conversation[1]["index"] == 1
    assert conversation[2]["tool"] == "lookup"
    assert conversation[2]["index"] == 2
    assert tokenizer.conversations[-1] == conversation


def test_render_messages_ignores_client_supplied_xtml_tool_attrs():
    tokenizer = StubTokenizer([1, 2, 3])
    renderer = _make_renderer(tokenizer)

    conversation, _ = renderer.render_messages(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "lookup:0",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "unknown",
                "tool": "lookup",
                "index": 1,
                "content": "result",
            },
        ],
        ChatParams(),
    )

    assert "tool" not in conversation[1]
    assert "index" not in conversation[1]


@pytest.mark.asyncio
async def test_render_messages_async_returns_token_prompt():
    renderer = _make_renderer(StubTokenizer([4, 5]))

    conversation, prompt = await renderer.render_messages_async(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [4, 5]}
    assert conversation[0]["role"] == "user"


def test_apply_chat_template_strips_null_tool_fields():
    # The serving layer's model_dump() serializes unset tool fields as null;
    # the platform tokenism omits them, so K3 drops them for prompt parity.
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "foo",
                "description": None,
                "strict": None,
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]
    params = ChatParams(chat_template_kwargs={"tools": tools})

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert tokenizer.calls[-1]["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "foo",
                "parameters": {"type": "object", "properties": {}},
            },
        }
    ]


def test_apply_chat_template_keeps_user_schema_content():
    # None values inside user-supplied `parameters` are schema content
    # (e.g. "default": null) and must survive the null stripping.
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    tools = [
        {
            "type": "function",
            "function": {
                "name": "foo",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "string", "default": None}},
                },
            },
        }
    ]
    params = ChatParams(chat_template_kwargs={"tools": tools})

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert tokenizer.calls[-1]["tools"] == tools


def test_preserve_malformed_tool_arguments_helper():
    # Malformed strings are wrapped as JSON string literals; valid strings,
    # dicts and non-list tool_calls pass through untouched.
    messages = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "f", "arguments": '{"x": 1'},
                },
                {
                    "id": "b",
                    "type": "function",
                    "function": {"name": "g", "arguments": '{"x": 1}'},
                },
                {
                    "id": "c",
                    "type": "function",
                    "function": {"name": "h", "arguments": {"x": 1}},
                },
            ],
        },
    ]

    (out,) = _preserve_malformed_tool_arguments(messages)[1:2]
    calls = out["tool_calls"]
    assert json.loads(calls[0]["function"]["arguments"]) == '{"x": 1'
    assert calls[1]["function"]["arguments"] == '{"x": 1}'
    assert calls[2]["function"]["arguments"] == {"x": 1}


def test_preserve_malformed_tool_arguments_whitespace_only():
    # Whitespace-only arguments are normalized to empty arguments instead of
    # failing json.loads downstream (K3 treats them as empty).
    messages = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "a",
                    "type": "function",
                    "function": {"name": "f", "arguments": "  \n "},
                },
            ],
        },
    ]

    (out,) = _preserve_malformed_tool_arguments(messages)
    assert out["tool_calls"][0]["function"]["arguments"] == "{}"


def test_render_messages_preserves_malformed_tool_arguments():
    """Malformed tool-call arguments round-trip to K3's encoding byte-exact.

    parse_chat_messages json.loads string arguments, so the renderer wraps
    unparsable ones as JSON string literals; the loads round-trips them to
    the original string, which K3's encoding renders via its raw-text
    fallback exactly like the platform tokenism.
    """
    tokenizer = StubTokenizer([1, 2])
    renderer = _make_renderer(tokenizer)
    malformed = '{"location":"北京"'

    conversation, prompt = renderer.render_messages(
        [
            {"role": "user", "content": "北京天气怎么样？"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": malformed},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "晴"},
            {"role": "user", "content": "继续"},
        ],
        ChatParams(),
    )

    assert prompt == {"prompt_token_ids": [1, 2]}
    sent = tokenizer.conversations[-1]
    assert sent[1]["tool_calls"][0]["function"]["arguments"] == malformed


@pytest.mark.asyncio
async def test_render_messages_async_preserves_malformed_tool_arguments():
    tokenizer = StubTokenizer([3])
    renderer = _make_renderer(tokenizer)
    malformed = '{"location":"北京"'

    await renderer.render_messages_async(
        [
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": malformed},
                    }
                ],
            },
        ],
        ChatParams(),
    )

    sent = tokenizer.conversations[-1]
    assert sent[0]["tool_calls"][0]["function"]["arguments"] == malformed


def test_render_messages_converts_developer_to_system():
    """developer messages must reach K3's encoding as system messages.

    K3's encoding_k3 only recognizes system/user/assistant/tool and silently
    drops other roles; developer (OpenAI's newer system-role name) is
    converted with tools preserved so it renders as a dynamic tool declare.
    """
    tokenizer = StubTokenizer([1, 2])
    renderer = _make_renderer(tokenizer)
    tools = [{"type": "function", "function": {"name": "search"}}]

    conversation, prompt = renderer.render_messages(
        [
            {"role": "developer", "content": "be terse", "tools": tools},
            {"role": "user", "content": "hi"},
        ],
        ChatParams(),
    )

    assert prompt == {"prompt_token_ids": [1, 2]}
    sent = tokenizer.conversations[-1]
    # A developer message with BOTH content and tools is split in two
    # (declare first, content second): the encoding renders a system
    # message with tools as a tool-declare and would drop the content.
    assert sent[0] == {"role": "system", "tools": tools}
    assert sent[1] == {"role": "system", "content": "be terse"}


def test_render_messages_converts_developer_without_tools_to_single_system():
    tokenizer = StubTokenizer([1, 2])
    renderer = _make_renderer(tokenizer)

    conversation, prompt = renderer.render_messages(
        [{"role": "developer", "content": "be terse"}],
        ChatParams(),
    )

    assert prompt == {"prompt_token_ids": [1, 2]}
    sent = tokenizer.conversations[-1]
    assert sent[0] == {"role": "system", "content": "be terse", "tools": None}
    assert len(sent) == 1


def test_render_messages_converts_developer_tools_without_content_to_single():
    tokenizer = StubTokenizer([1, 2])
    renderer = _make_renderer(tokenizer)
    tools = [{"type": "function", "function": {"name": "search"}}]

    conversation, prompt = renderer.render_messages(
        [{"role": "developer", "tools": tools}],
        ChatParams(),
    )

    assert prompt == {"prompt_token_ids": [1, 2]}
    sent = tokenizer.conversations[-1]
    assert sent[0] == {"role": "system", "content": "", "tools": tools}
    assert len(sent) == 1


@pytest.mark.asyncio
async def test_render_messages_async_converts_developer_to_system():
    tokenizer = StubTokenizer([3])
    renderer = _make_renderer(tokenizer)

    await renderer.render_messages_async(
        [{"role": "developer", "content": "be terse"}], ChatParams()
    )

    sent = tokenizer.conversations[-1]
    assert sent[0]["role"] == "system"
    assert sent[0]["content"] == "be terse"
