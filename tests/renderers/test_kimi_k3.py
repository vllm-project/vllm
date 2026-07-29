# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass, field
from typing import Any

import pytest

from vllm.renderers import ChatParams
from vllm.renderers.kimi_k3 import KimiK3Renderer, _merge_k3_media_io_kwargs
from vllm.renderers.registry import RENDERER_REGISTRY
from vllm.tokenizers.registry import TokenizerRegistry

pytestmark = pytest.mark.skip_global_cleanup


class StubTokenizer:
    def __init__(self, token_ids: list[int]) -> None:
        self.token_ids = token_ids
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def apply_chat_template(
        self,
        conversation: Any,
        **kwargs: Any,
    ) -> list[int]:
        self.calls.append((conversation, kwargs))
        return self.token_ids


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
    assert TokenizerRegistry.load_tokenizer_cls("kimi_k3").__name__ == "KimiK3Tokenizer"


def test_k3_media_io_defaults_preserve_original_mode():
    assert _merge_k3_media_io_kwargs(None) == {"image": {"image_mode": None}}
    assert _merge_k3_media_io_kwargs({"image": {"image_mode": "RGB"}}) == {
        "image": {"image_mode": "RGB"}
    }
    assert _merge_k3_media_io_kwargs(
        {"image": {"rgba_background_color": (0, 0, 0)}}
    ) == {"image": {"image_mode": None, "rgba_background_color": (0, 0, 0)}}


def test_apply_chat_template_forces_flat_token_output():
    tokenizer = StubTokenizer([7, 8, 9])
    renderer = _make_renderer(tokenizer)
    conversation = [{"role": "user", "content": "hi"}]

    token_ids = renderer._apply_chat_template(
        conversation,
        ChatParams(
            chat_template_kwargs={
                "chat_template": "ignored",
                "tokenize": False,
                "return_dict": True,
                "thinking": False,
            }
        ),
    )

    assert token_ids == [7, 8, 9]
    passed_conversation, kwargs = tokenizer.calls[-1]
    assert passed_conversation is conversation
    assert kwargs["tokenize"] is True
    assert kwargs["return_dict"] is False
    assert kwargs["thinking"] is False
    assert "chat_template" not in kwargs


def test_apply_chat_template_passes_api_metadata():
    tokenizer = StubTokenizer([7])
    renderer = _make_renderer(tokenizer)
    response_format = {"type": "json_object"}

    renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}],
        ChatParams(
            tool_choice="required",
            response_format=response_format,
        ),
    )

    kwargs = tokenizer.calls[-1][1]
    assert kwargs["tool_choice"] == "required"
    assert kwargs["response_format"] is response_format


def test_apply_chat_template_preserves_template_tool_choice_for_api_auto():
    tokenizer = StubTokenizer([7])
    renderer = _make_renderer(tokenizer)

    renderer._apply_chat_template(
        [{"role": "user", "content": "hi"}],
        ChatParams(
            chat_template_kwargs={"tool_choice": "required"},
            tool_choice="auto",
        ),
    )

    assert tokenizer.calls[-1][1]["tool_choice"] == "required"


def test_render_messages_returns_token_prompt():
    tokenizer = StubTokenizer([1, 2, 3])
    renderer = _make_renderer(tokenizer)

    conversation, prompt = renderer.render_messages(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [1, 2, 3]}
    assert "multi_modal_data" not in prompt
    assert conversation[0]["role"] == "user"
    assert tokenizer.calls[-1][0] == conversation


@pytest.mark.asyncio
async def test_render_messages_async_returns_token_prompt():
    tokenizer = StubTokenizer([4, 5])
    renderer = _make_renderer(tokenizer)

    conversation, prompt = await renderer.render_messages_async(
        [{"role": "user", "content": "hi"}], ChatParams()
    )

    assert prompt == {"prompt_token_ids": [4, 5]}
    assert conversation[0]["role"] == "user"
    assert tokenizer.calls[-1][0] == conversation
