# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from dataclasses import dataclass
from threading import Event
from typing import Any

import pytest
import torch

from vllm.exceptions import RetryableRequestError
from vllm.renderers.base import _SwappableExecutor
from vllm.renderers.hf import HfRenderer, _RequestTemplateExecutor
from vllm.renderers.params import TokenizeParams
from vllm.utils.async_utils import make_async

MODEL_NAME = "openai-community/gpt2"


@dataclass(slots=True)
class MockHFConfig:
    model_type: str = "any"


@dataclass(slots=True)
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
    supports_multimodal_inputs: bool = False
    renderer_num_workers: int = 1
    hidden_size: int = 768
    dtype: torch.dtype = torch.float32

    def get_hidden_size(self) -> int:
        return self.hidden_size


@dataclass(slots=True)
class MockParallelConfig:
    _api_process_rank: int = 0


@dataclass(slots=True)
class MockVllmConfig:
    model_config: MockModelConfig
    parallel_config: MockParallelConfig


@dataclass(slots=True)
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


@pytest.mark.asyncio
async def test_request_template_slots_outlive_timeout_and_cancellation():
    executor = _RequestTemplateExecutor(workers=2)
    loop = asyncio.get_running_loop()
    started: asyncio.Queue[None] = asyncio.Queue()
    release = Event()

    def render() -> str:
        loop.call_soon_threadsafe(started.put_nowait, None)
        assert release.wait(timeout=5)
        return "rendered"

    timed = asyncio.create_task(executor.render(render, timeout=0.1))
    cancelled = asyncio.create_task(executor.render(render, timeout=0))
    try:
        for _ in range(2):
            await asyncio.wait_for(started.get(), timeout=5)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        with pytest.raises(TimeoutError):
            await timed
        with pytest.raises(RetryableRequestError):
            await executor.render(lambda: "another render", timeout=0)

        release.set()

        async def render_when_available() -> str:
            while True:
                try:
                    return await executor.render(lambda: "next render", timeout=0)
                except RetryableRequestError:
                    await asyncio.sleep(0)

        assert await asyncio.wait_for(render_when_available(), 5) == "next render"
    finally:
        release.set()
        await asyncio.gather(timed, cancelled, return_exceptions=True)
        executor.shutdown()


def test_replace_executor_does_not_break_tokenize_or_decode():
    renderer = _build_renderer()
    executor = renderer._executor
    old_inner = executor._inner

    async def _run():
        tokenized, _ = await renderer._tokenize_text_prompt_async(
            {"prompt": "ab"},
            TokenizeParams(max_total_tokens=100),
        )
        assert tokenized["prompt_token_ids"] == [0, 1]
        renderer._replace_executor()
        assert renderer._executor is executor
        assert executor._inner is not old_inner

        with pytest.raises(RuntimeError, match="cannot schedule new futures"):
            old_inner.submit(lambda: None)

        tokenized, _ = await renderer._tokenize_text_prompt_async(
            {"prompt": "abc"},
            TokenizeParams(max_total_tokens=100),
        )
        assert tokenized["prompt_token_ids"] == [0, 1, 2]
        assert await renderer._async_tokenizer_decode([1, 2]) == "[1, 2]"

    try:
        asyncio.run(_run())
    finally:
        renderer.shutdown()
