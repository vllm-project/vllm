# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.renderers import ChatParams
from vllm.renderers.kimi_k3 import KimiK3Renderer, _merge_k3_media_io_kwargs
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

    def apply_chat_template(self, conversation, **kwargs) -> list[int]:
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
    assert kwargs["thinking_effort"] == "none"
    assert "enable_thinking" not in kwargs
    assert "reasoning_effort" not in kwargs


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


def test_render_messages_returns_token_prompt():
    renderer = _make_renderer(StubTokenizer([1, 2, 3]))

    conversation, prompt = renderer.render_messages(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [1, 2, 3]}
    assert "multi_modal_data" not in prompt
    assert conversation[0]["role"] == "user"


@pytest.mark.asyncio
async def test_render_messages_async_returns_token_prompt():
    renderer = _make_renderer(StubTokenizer([4, 5]))

    conversation, prompt = await renderer.render_messages_async(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [4, 5]}
    assert conversation[0]["role"] == "user"
