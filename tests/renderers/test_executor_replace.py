# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from dataclasses import dataclass
from typing import Any

import pytest
import torch

from vllm.renderers.base import _SwappableExecutor
from vllm.renderers.hf import HfRenderer
from vllm.renderers.params import TokenizeParams
from vllm.utils.async_utils import make_async

MODEL_NAME = "openai-community/gpt2"


@dataclass
class MockHFConfig:
    model_type: str = "any"


@dataclass
class MockModelConfig:
    runner_type = "generate"
    model: str = MODEL_NAME
    tokenizer: str = MODEL_NAME
    trust_remote_code: bool = False
    tokenizer_revision = None
    tokenizer_mode = "auto"
    hf_config = MockHFConfig()
    encoder_config: dict[str, Any] | None = None
    enable_prompt_embeds: bool = False
    skip_tokenizer_init: bool = False
    is_encoder_decoder: bool = False
    is_multimodal_model: bool = False
    renderer_num_workers: int = 1
    hidden_size: int = 768
    dtype: torch.dtype = torch.float32

    def get_hidden_size(self) -> int:
        return self.hidden_size


@dataclass
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


@dataclass
class DummyTokenizer:
    truncation_side: str = "left"
    max_chars_per_token: int = 1
    is_fast: bool = False

    def decode(self, tokens: list[int], **kwargs):
        return str(tokens)

    def encode(self, text: str, **kwargs):
        return list(range(len(text)))

    def __call__(self, text: str, **kwargs):
        return {"input_ids": self.encode(text, **kwargs)}


def _build_renderer() -> HfRenderer:
    return HfRenderer(
        MockVllmConfig(MockModelConfig(), parallel_config=MockParallelConfig()),
        tokenizer=DummyTokenizer(),
    )


def test_swappable_executor_keeps_make_async_wrappers_alive():
    pool = _SwappableExecutor(max_workers=1)
    async_add = make_async(lambda x: x + 1, executor=pool)

    async def _run():
        assert await async_add(1) == 2
        old_inner = pool._inner
        pool.replace_inner()

        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            old_inner.submit(lambda: None)

        assert await async_add(40) == 41

    try:
        asyncio.run(_run())
    finally:
        pool.shutdown(wait=False)


def test_replace_executor_does_not_break_tokenize_or_decode():
    renderer = _build_renderer()
    executor = renderer._executor
    old_inner = executor._inner

    async def _run():
        assert await renderer._tokenize_prompt_async(
            {"prompt": "ab"},
            TokenizeParams(max_total_tokens=100),
        )
        renderer._replace_executor()
        assert renderer._executor is executor
        assert executor._inner is not old_inner

        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            old_inner.submit(lambda: None)

        tokenized = await renderer._tokenize_prompt_async(
            {"prompt": "abc"},
            TokenizeParams(max_total_tokens=100),
        )
        assert tokenized["prompt_token_ids"] == [0, 1, 2]
        assert await renderer._async_tokenizer_decode([1, 2]) == "[1, 2]"

    try:
        asyncio.run(_run())
    finally:
        renderer.shutdown()
