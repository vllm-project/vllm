# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import threading
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor

import pytest

from vllm.utils.async_utils import make_async_batched, merge_async_iterators


async def _mock_async_iterator(idx: int):
    try:
        while True:
            yield f"item from iterator {idx}"
            await asyncio.sleep(0.1)
    except asyncio.CancelledError:
        print(f"iterator {idx} cancelled")


@pytest.mark.asyncio
async def test_merge_async_iterators():
    iterators = [_mock_async_iterator(i) for i in range(3)]
    merged_iterator = merge_async_iterators(*iterators)

    async def stream_output(generator: AsyncIterator[tuple[int, str]]):
        async for idx, output in generator:
            print(f"idx: {idx}, output: {output}")

    task = asyncio.create_task(stream_output(merged_iterator))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for iterator in iterators:
        try:
            await asyncio.wait_for(anext(iterator), 1)
        except StopAsyncIteration:
            # All iterators should be cancelled and print this message.
            print("Iterator was cancelled normally")
        except (Exception, asyncio.CancelledError) as e:
            raise AssertionError() from e


@pytest.mark.asyncio
async def test_merge_async_iterators_single_closes_underlying():
    # The single-iterator fast path must close the underlying generator when
    # the merged generator is closed, matching the multi-iterator path. On the
    # buggy fast path the underlying generator is left running.
    closed = False

    async def gen():
        nonlocal closed
        try:
            while True:
                yield "x"
                await asyncio.sleep(0.01)
        finally:
            closed = True

    merged = merge_async_iterators(gen())
    assert await anext(merged) == (0, "x")
    await merged.aclose()
    assert closed


class _CountingExecutor(ThreadPoolExecutor):
    def __init__(self, max_workers: int):
        super().__init__(max_workers=max_workers)
        self.num_jobs = 0

    def submit(self, fn, /, *args, **kwargs):
        self.num_jobs += 1
        return super().submit(fn, *args, **kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("max_workers", "max_batch_size", "num_jobs"),
    [(1, 256, 1), (1, 32, 4), (4, 256, 4)],
)
async def test_make_async_batched_groups_calls(max_workers, max_batch_size, num_jobs):
    # Calls made in one loop iteration share jobs, split across the workers.
    with _CountingExecutor(max_workers) as executor:
        square = make_async_batched(lambda x: x * x, executor, max_batch_size)
        assert await asyncio.gather(*[square(i) for i in range(100)]) == [
            i * i for i in range(100)
        ]
        assert executor.num_jobs == num_jobs


@pytest.mark.asyncio
async def test_make_async_batched_errors_stay_per_call():
    def check(x: int) -> int:
        if x % 2:
            raise ValueError(x)
        return x

    with ThreadPoolExecutor(max_workers=1) as executor:
        check_async = make_async_batched(check, executor)
        results = await asyncio.gather(
            *[check_async(i) for i in range(4)], return_exceptions=True
        )
    assert results[0::2] == [0, 2]
    assert [type(r) for r in results[1::2]] == [ValueError, ValueError]

    # The executor is shut down.
    with pytest.raises(RuntimeError):
        await check_async(0)


@pytest.mark.asyncio
async def test_make_async_batched_skips_cancelled_calls():
    calls: list[int] = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        record = make_async_batched(calls.append, executor)
        cancelled, kept = record(0), record(1)
        cancelled.cancel()
        await kept
    assert calls == [1]


@pytest.mark.asyncio
async def test_make_async_batched_returns_results_before_the_job_ends():
    release = threading.Event()

    def wait_for_release(block: bool) -> bool:
        if block:
            release.wait(timeout=10)
        return block

    with ThreadPoolExecutor(max_workers=1) as executor:
        call = make_async_batched(wait_for_release, executor)
        first, second = call(False), call(True)  # One job.
        assert await asyncio.wait_for(first, timeout=5) is False
        assert not second.done()
        release.set()
        assert await second is True
