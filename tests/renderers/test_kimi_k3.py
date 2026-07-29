# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.renderers import ChatParams
from vllm.renderers.kimi_k3 import KimiK3Renderer, _merge_k3_media_io_kwargs
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.encoding_k3 import EncodeSegment
from vllm.tokenizers.kimi_k3 import get_kimi_k3_tokenizer
from vllm.tokenizers.registry import TokenizerRegistry

pytestmark = pytest.mark.skip_global_cleanup


class StubTokenizer:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.segments: list[tuple[str, bool]] = []

    def _encode_text_piece(
        self, text: str, allow_special_tokens: bool = True
    ) -> list[int]:
        first = not self.segments
        self.segments.append((text, allow_special_tokens))
        return list(self.token_ids) if first else []

    def _encode_chat_segments(self, segments: list[EncodeSegment]) -> list[int]:
        return [
            token_id
            for segment in segments
            for token_id in self._encode_text_piece(
                segment.text,
                allow_special_tokens=segment.allow_special,
            )
        ]

    def _format_chat_token_output(
        self,
        encoded_inputs: list[list[int]],
        **_: Any,
    ) -> list[int]:
        return encoded_inputs[0]

    @property
    def rendered(self) -> str:
        return "".join(text for text, _ in self.segments)


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
    return KimiK3Renderer(config, get_kimi_k3_tokenizer(tokenizer))


def test_kimi_k3_registered():
    assert RENDERER_REGISTRY.load_renderer_cls("kimi_k3").__name__ == "KimiK3Renderer"
    assert TokenizerRegistry.load_tokenizer_cls("kimi_k3").__name__ == "KimiK3Tokenizer"


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


def test_apply_chat_template_renders_and_encodes_local_segments():
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
    assert "search" in tokenizer.rendered
    assert "<|open|>think<|sep|>" in tokenizer.rendered
    assert any(allow_special for _, allow_special in tokenizer.segments)
    assert any(not allow_special for _, allow_special in tokenizer.segments)


def test_apply_chat_template_translates_standard_thinking_kwargs():
    # Standard enable_thinking/reasoning_effort kwargs must be translated
    # to K3's native thinking/thinking_effort.
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={"enable_thinking": False, "reasoning_effort": "none"}
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert "<|open|>think<|sep|>" not in tokenizer.rendered
    assert "<|open|>response<|sep|>" in tokenizer.rendered


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

    assert "thinking_effort=low" in tokenizer.rendered


def test_apply_chat_template_adds_k3_api_metadata():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    response_format = {"type": "json_object"}
    params = ChatParams(
        tool_choice="required",
        response_format=response_format,
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert "tool_choice=required" in tokenizer.rendered
    assert "response_format=json_object" in tokenizer.rendered


def test_apply_chat_template_auto_tool_choice_keeps_template_kwarg():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    params = ChatParams(
        chat_template_kwargs={"tool_choice": "required"},
        tool_choice="auto",
    )

    renderer._apply_chat_template([{"role": "user", "content": "hi"}], params)

    assert "tool_choice=required" in tokenizer.rendered


def test_apply_chat_template_omits_tool_choice_without_tools():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)

    renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}], ChatParams(tool_choice=None)
    )

    assert "tool-choice" not in tokenizer.rendered


def test_render_messages_returns_token_prompt():
    renderer = _make_renderer(StubTokenizer([1, 2, 3]))

    conversation, prompt = renderer.render_messages(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [1, 2, 3]}
    assert "multi_modal_data" not in prompt
    assert conversation[0]["role"] == "user"


def test_render_messages_reorders_tool_results_only_for_rendering():
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
        "second",
        "first",
    ]
    assert 'role="tool" tool="lookup" index="1"' in tokenizer.rendered


def test_apply_chat_template_renders_dynamic_system_tools():
    tokenizer = StubTokenizer([1])
    renderer = _make_renderer(tokenizer)

    renderer._apply_chat_template(
        [
            {
                "role": "system",
                "content": "",
                "tools": [{"type": "function", "function": {"name": "late_tool"}}],
            }
        ],
        ChatParams(),
    )

    assert "## New Tools Available" in tokenizer.rendered
    assert "late_tool" in tokenizer.rendered


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
