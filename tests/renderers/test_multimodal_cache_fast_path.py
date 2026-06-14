# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import threading
import types
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest

from vllm.multimodal.processing import ProcessorInputs, TimingContext
from vllm.renderers.base import BaseRenderer
from vllm.utils.async_utils import make_async
from vllm.utils.counter import AtomicCounter


class _FakeInfo:
    @staticmethod
    def parse_mm_data(mm_data: dict[str, Any]):
        return mm_data


class _FakeProcessor:
    def __init__(self, started: threading.Event, release: threading.Event) -> None:
        self.info = _FakeInfo()
        self.started = started
        self.release = release

    def try_apply_cached(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        if inputs.mm_data_items["kind"] != "hit":
            return None

        return {
            "prompt_token_ids": inputs.prompt,
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }


    def get_cache_missing_hashes(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        return None

    def try_apply_cached_or_get_missing_hashes(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        mm_input = self.try_apply_cached(inputs, timing_ctx)
        return types.SimpleNamespace(
            mm_input=mm_input,
            missing_hashes=[] if mm_input is not None else None,
        )

    def prefill_cache_items(
        self,
        inputs: ProcessorInputs,
        item_hashes: list[str],
        timing_ctx: TimingContext,
    ) -> None:
        self.apply(inputs, timing_ctx)

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        self.started.set()
        assert self.release.wait(timeout=5)
        return {
            "prompt_token_ids": inputs.prompt,
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }


class _FakeSingleFlightProcessor:
    def __init__(
        self,
        first_started: threading.Event,
        second_started: threading.Event,
        release_first: threading.Event,
        release_second: threading.Event,
    ) -> None:
        self.info = _FakeInfo()
        self.cache = {1, 2, 3}
        self.lock = threading.Lock()
        self.first_started = first_started
        self.second_started = second_started
        self.release_first = release_first
        self.release_second = release_second
        self.apply_calls = 0

    def try_apply_cached(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        images = inputs.mm_data_items["images"]
        with self.lock:
            if not all(image in self.cache for image in images):
                return None

        return {
            "prompt_token_ids": inputs.prompt,
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }

    def get_cache_missing_hashes(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        images = inputs.mm_data_items["images"]
        with self.lock:
            return [
                str(image)
                for image in images
                if image not in self.cache
            ]

    def try_apply_cached_or_get_missing_hashes(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        mm_input = self.try_apply_cached(inputs, timing_ctx)
        return types.SimpleNamespace(
            mm_input=mm_input,
            missing_hashes=(
                [] if mm_input is not None else
                self.get_cache_missing_hashes(inputs, timing_ctx)
            ),
        )

    def prefill_cache_items(
        self,
        inputs: ProcessorInputs,
        item_hashes: list[str],
        timing_ctx: TimingContext,
    ) -> None:
        selected_hashes = {int(item_hash) for item_hash in item_hashes}
        images = inputs.mm_data_items["images"]
        with self.lock:
            missing = [
                image
                for image in images
                if image in selected_hashes and image not in self.cache
            ]

        for image in missing:
            if image == 100:
                self.first_started.set()
                assert self.release_first.wait(timeout=5)
            elif image == 101:
                self.second_started.set()
                assert self.release_second.wait(timeout=5)

        with self.lock:
            self.cache.update(missing)

    def apply(
        self,
        inputs: ProcessorInputs,
        timing_ctx: TimingContext,
    ):
        self.apply_calls += 1
        images = inputs.mm_data_items["images"]
        with self.lock:
            missing = [image for image in images if image not in self.cache]

        for image in missing:
            if image == 100:
                self.first_started.set()
                assert self.release_first.wait(timeout=5)
            elif image == 101:
                self.second_started.set()
                assert self.release_second.wait(timeout=5)

        with self.lock:
            self.cache.update(missing)

        return {
            "prompt_token_ids": inputs.prompt,
            "mm_kwargs": {},
            "mm_hashes": {},
            "mm_placeholders": {},
        }


class _FakeTimingRegistry:
    @staticmethod
    def get(request_id: str):
        return TimingContext(enabled=False)


class _TestRenderer(BaseRenderer):
    def render_messages(self, messages, params):
        raise NotImplementedError


def _make_renderer(processor: _FakeProcessor, max_workers: int = 1):
    renderer: Any = object.__new__(_TestRenderer)
    renderer.api_process_rank = 0
    renderer._mm_req_counter = AtomicCounter()
    renderer._readonly_mm_processor = None
    renderer.mm_processor = processor
    renderer._mm_timing_registry = _FakeTimingRegistry()
    renderer._process_mm_uuids = lambda *args, **kwargs: None
    renderer.update_mm_cache_stats = lambda: None
    renderer.config = type(
        "Config",
        (),
        {
            "cache_config": type(
                "CacheConfig",
                (),
                {"enable_prefix_caching": True},
            )()
        },
    )()
    renderer.model_config = type(
        "ModelConfig",
        (),
        {"multimodal_config": None},
    )()
    renderer.get_mm_processor = lambda: processor
    renderer._mm_cache_inflight_futures = {}
    executor = ThreadPoolExecutor(max_workers=max_workers)
    renderer._process_multimodal_in_executor = make_async(
        renderer._process_multimodal,
        executor=executor,
    )
    renderer._prefill_multimodal_cache_in_executor = make_async(
        renderer._prefill_multimodal_cache,
        executor=executor,
    )
    renderer._test_executor = executor
    return renderer


@pytest.mark.asyncio
async def test_cached_multimodal_request_bypasses_busy_executor():
    started = threading.Event()
    release = threading.Event()
    renderer = _make_renderer(_FakeProcessor(started, release))

    miss_task = asyncio.create_task(
        renderer._process_multimodal_async(
            [1],
            {"kind": "miss"},
            mm_uuids=None,
            mm_processor_kwargs=None,
            tokenization_kwargs=None,
        )
    )

    try:
        assert await asyncio.to_thread(started.wait, 5)

        hit_result = await asyncio.wait_for(
            renderer._process_multimodal_async(
                [2],
                {"kind": "hit"},
                mm_uuids=None,
                mm_processor_kwargs=None,
                tokenization_kwargs=None,
            ),
            timeout=0.5,
        )
        assert hit_result["prompt_token_ids"] == [2]
    finally:
        release.set()
        await miss_task
        renderer._test_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_duplicate_cache_miss_waits_for_inflight_owner():
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    processor = _FakeSingleFlightProcessor(
        first_started,
        second_started,
        release_first,
        release_second,
    )
    renderer = _make_renderer(processor)

    first_task = asyncio.create_task(
        renderer._process_multimodal_async(
            [1],
            {"images": (1, 2, 3, 100)},
            mm_uuids=None,
            mm_processor_kwargs=None,
            tokenization_kwargs=None,
        )
    )

    try:
        assert await asyncio.to_thread(first_started.wait, 5)

        second_task = asyncio.create_task(
            renderer._process_multimodal_async(
                [2],
                {"images": (1, 2, 3, 101)},
                mm_uuids=None,
                mm_processor_kwargs=None,
                tokenization_kwargs=None,
            )
        )
        assert not await asyncio.to_thread(second_started.wait, 0.05)

        duplicate_task = asyncio.create_task(
            renderer._process_multimodal_async(
                [3],
                {"images": (1, 2, 3, 100)},
                mm_uuids=None,
                mm_processor_kwargs=None,
                tokenization_kwargs=None,
            )
        )

        release_first.set()
        duplicate_result = await asyncio.wait_for(duplicate_task, timeout=0.5)
        assert duplicate_result["prompt_token_ids"] == [3]
        assert await asyncio.to_thread(second_started.wait, 5)
        assert not second_task.done()
    finally:
        release_first.set()
        release_second.set()
        await first_task
        if "second_task" in locals():
            await second_task
        renderer._test_executor.shutdown(wait=True)


@pytest.mark.asyncio
async def test_mixed_miss_prefills_owned_hash_while_waiting():
    first_started = threading.Event()
    second_started = threading.Event()
    release_first = threading.Event()
    release_second = threading.Event()
    processor = _FakeSingleFlightProcessor(
        first_started,
        second_started,
        release_first,
        release_second,
    )
    renderer = _make_renderer(processor, max_workers=2)

    first_task = asyncio.create_task(
        renderer._process_multimodal_async(
            [1],
            {"images": (1, 2, 3, 100)},
            mm_uuids=None,
            mm_processor_kwargs=None,
            tokenization_kwargs=None,
        )
    )

    try:
        assert await asyncio.to_thread(first_started.wait, 5)

        mixed_task = asyncio.create_task(
            renderer._process_multimodal_async(
                [2],
                {"images": (1, 2, 100, 101)},
                mm_uuids=None,
                mm_processor_kwargs=None,
                tokenization_kwargs=None,
            )
        )

        assert await asyncio.to_thread(second_started.wait, 5)
        assert not mixed_task.done()

        release_second.set()
        await asyncio.sleep(0)
        assert not mixed_task.done()

        release_first.set()
        mixed_result = await asyncio.wait_for(mixed_task, timeout=0.5)
        assert mixed_result["prompt_token_ids"] == [2]
        assert processor.apply_calls == 1
    finally:
        release_first.set()
        release_second.set()
        await first_task
        if "mixed_task" in locals():
            await mixed_task
        renderer._test_executor.shutdown(wait=True)
