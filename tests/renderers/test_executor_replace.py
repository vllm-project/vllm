# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any

import pytest
import torch

import vllm.renderers.hf as hf_renderer
from vllm import envs
from vllm.renderers.base import _SwappableExecutor
from vllm.renderers.hf import HfRenderer
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


def _patch_render(monkeypatch, timeout_s, render):
    monkeypatch.setattr(
        envs, "VLLM_CHAT_TEMPLATE_RENDER_TIMEOUT", timeout_s, raising=False
    )
    monkeypatch.setattr(hf_renderer, "safe_apply_chat_template", render)


def test_render_timeout_excludes_queue_wait(monkeypatch):
    """A fast render queued behind other pool work (for example tokenization of
    a long prompt) for longer than the timeout must not time out."""
    _patch_render(monkeypatch, 0.5, lambda *args, **kwargs: "rendered")
    renderer = _build_renderer()

    async def _run():
        loop = asyncio.get_running_loop()
        busy = loop.run_in_executor(renderer._executor, time.sleep, 1.0)
        assert await renderer._render_with_timeout(None, None, []) == "rendered"
        await busy

    try:
        asyncio.run(_run())
    finally:
        renderer.shutdown()


def test_slow_render_still_times_out_and_replaces_pool(monkeypatch):
    def slow(*args, **kwargs):
        time.sleep(1.0)
        return "late"

    _patch_render(monkeypatch, 0.3, slow)
    renderer = _build_renderer()
    old_inner = renderer._executor._inner

    async def _run():
        with pytest.raises(TimeoutError, match="timed out"):
            await renderer._render_with_timeout(None, None, [])
        assert renderer._executor._inner is not old_inner

    try:
        asyncio.run(_run())
    finally:
        renderer.shutdown()


def test_render_queued_behind_stuck_render_moves_to_new_pool(monkeypatch):
    stuck_started = threading.Event()
    release = threading.Event()

    def render(model_config, tokenizer, conversation, **kwargs):
        if conversation == ["stuck"]:
            stuck_started.set()
            release.wait(10)
            return "stuck"
        return "fast"

    _patch_render(monkeypatch, 0.3, render)
    renderer = _build_renderer()

    async def _run():
        stuck = asyncio.ensure_future(
            renderer._render_with_timeout(None, None, ["stuck"])
        )
        assert await asyncio.to_thread(stuck_started.wait, 5)
        queued = asyncio.ensure_future(
            renderer._render_with_timeout(None, None, ["fast"])
        )
        with pytest.raises(TimeoutError, match="timed out"):
            await stuck
        assert await asyncio.wait_for(queued, timeout=5) == "fast"

    try:
        asyncio.run(_run())
    finally:
        release.set()
        renderer.shutdown()


def test_cancelled_caller_does_not_render_later(monkeypatch):
    """A render still queued when its caller is cancelled must not run later:
    no caller is left to time it or to replace the pool."""
    rendered: list[list[str]] = []

    def render(model_config, tokenizer, conversation, **kwargs):
        rendered.append(conversation)
        return "rendered"

    _patch_render(monkeypatch, 5.0, render)
    renderer = _build_renderer()
    release = threading.Event()

    async def _run():
        loop = asyncio.get_running_loop()
        busy = loop.run_in_executor(renderer._executor, release.wait, 10)
        cancelled = asyncio.ensure_future(
            renderer._render_with_timeout(None, None, ["cancelled"])
        )
        # One loop step: the task queues its render behind the busy job.
        await asyncio.sleep(0)
        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        release.set()
        await busy
        # The pool has one worker, so this job runs after the queued ones.
        await loop.run_in_executor(renderer._executor, lambda: None)
        assert rendered == []
        assert await renderer._render_with_timeout(None, None, ["later"]) == "rendered"
        assert rendered == [["later"]]

    try:
        asyncio.run(_run())
    finally:
        release.set()
        renderer.shutdown()


def _hold(started: threading.Event, release: threading.Event) -> None:
    """A pool job that reports its start and then holds its worker."""
    started.set()
    release.wait(10)


def test_cancelled_running_render_keeps_its_timeout(monkeypatch):
    """A render that hangs after its caller is cancelled still replaces the
    pool at its deadline, with no other render in the queue."""
    stuck_started = threading.Event()
    release = threading.Event()

    def render(model_config, tokenizer, conversation, **kwargs):
        if conversation == ["stuck"]:
            _hold(stuck_started, release)
        return conversation[0]

    _patch_render(monkeypatch, 0.3, render)
    renderer = _build_renderer()

    async def _run():
        pool_replaced = renderer._pool_replaced
        stuck = asyncio.ensure_future(
            renderer._render_with_timeout(None, None, ["stuck"])
        )
        assert await asyncio.to_thread(stuck_started.wait, 5)
        stuck.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stuck
        await asyncio.wait_for(pool_replaced.wait(), timeout=5)
        new_inner = renderer._executor._inner
        later = renderer._render_with_timeout(None, None, ["later"])
        assert await asyncio.wait_for(later, timeout=5) == "later"
        # The later render ran on the new pool: nothing replaced it again.
        assert renderer._executor._inner is new_inner

    try:
        asyncio.run(_run())
    finally:
        release.set()
        renderer.shutdown()


def test_render_queued_behind_stuck_render_of_cancelled_caller(monkeypatch):
    """A render that is queued behind a render that hangs after its caller was
    cancelled must run and not wait without limit."""
    stuck_started = threading.Event()
    release = threading.Event()

    def render(model_config, tokenizer, conversation, **kwargs):
        if conversation == ["stuck"]:
            _hold(stuck_started, release)
        return conversation[0]

    _patch_render(monkeypatch, 0.3, render)
    renderer = _build_renderer()
    old_inner = renderer._executor._inner

    async def _run():
        stuck = asyncio.ensure_future(
            renderer._render_with_timeout(None, None, ["stuck"])
        )
        assert await asyncio.to_thread(stuck_started.wait, 5)
        stuck.cancel()
        with pytest.raises(asyncio.CancelledError):
            await stuck
        later = renderer._render_with_timeout(None, None, ["later"])
        assert await asyncio.wait_for(later, timeout=5) == "later"
        assert renderer._executor._inner is not old_inner

    try:
        asyncio.run(_run())
    finally:
        release.set()
        renderer.shutdown()


def test_render_replaces_pool_stuck_in_another_job(monkeypatch):
    """Tokenization and decoding share the pool and have no timeout. A render
    queued behind a job that hangs must replace the pool and then run."""
    _patch_render(monkeypatch, 0.3, lambda *args, **kwargs: "rendered")
    renderer = _build_renderer()
    old_inner = renderer._executor._inner
    hung_started = threading.Event()
    release = threading.Event()

    async def _run():
        loop = asyncio.get_running_loop()
        hung = loop.run_in_executor(renderer._executor, _hold, hung_started, release)
        assert await asyncio.to_thread(hung_started.wait, 5)
        queued = renderer._render_with_timeout(None, None, [])
        assert await asyncio.wait_for(queued, timeout=5) == "rendered"
        assert renderer._executor._inner is not old_inner
        release.set()
        await hung

    try:
        asyncio.run(_run())
    finally:
        release.set()
        renderer.shutdown()


def test_render_keeps_pool_whose_queue_moves(monkeypatch):
    """A render that waits longer than the timeout behind jobs that start and
    finish must keep waiting: it neither fails nor replaces the pool."""
    timeout_s = 0.5
    # Each job is shorter than the timeout; together they outlast it.
    job_seconds, num_jobs = 0.2, 5
    _patch_render(monkeypatch, timeout_s, lambda *args, **kwargs: "rendered")
    renderer = _build_renderer()
    executor = renderer._executor
    old_inner = executor._inner

    async def _run():
        loop = asyncio.get_running_loop()
        jobs = [
            loop.run_in_executor(executor, time.sleep, job_seconds)
            for _ in range(num_jobs)
        ]
        start = time.monotonic()
        assert await renderer._render_with_timeout(None, None, []) == "rendered"
        assert time.monotonic() - start > timeout_s
        assert executor._inner is old_inner
        await asyncio.gather(*jobs)

    try:
        asyncio.run(_run())
    finally:
        renderer.shutdown()
