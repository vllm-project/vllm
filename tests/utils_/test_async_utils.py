# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import patch

import pytest

from vllm.utils.async_utils import make_async_with_semaphore, merge_async_iterators


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


@pytest.mark.asyncio
@pytest.mark.parametrize("worker_error", [False, True])
async def test_cancelled_worker_keeps_capacity_until_completion(worker_error):
    started = asyncio.Event()
    release = Event()
    loop = asyncio.get_running_loop()

    def work() -> int:
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        if worker_error:
            raise ValueError("worker failed")
        return 42

    with ThreadPoolExecutor(max_workers=1) as executor:
        run = make_async_with_semaphore(work, executor)
        with patch.object(executor, "submit", wraps=executor.submit) as submit:
            task: asyncio.Task[int] = asyncio.create_task(run())
            try:
                await asyncio.wait_for(started.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                for _ in range(10):
                    waiter: asyncio.Task[int] = asyncio.create_task(run())
                    await asyncio.sleep(0)
                    waiter.cancel()
                    with pytest.raises(asyncio.CancelledError):
                        await waiter
                assert submit.call_count == 1
            finally:
                release.set()

            if worker_error:
                with pytest.raises(ValueError, match="worker failed"):
                    await asyncio.wait_for(run(), 5)
            else:
                assert await asyncio.wait_for(run(), 5) == 42


@pytest.mark.asyncio
async def test_submission_failure_releases_capacity():
    with ThreadPoolExecutor(max_workers=1) as executor:
        run = make_async_with_semaphore(lambda: 42, executor)
        with (
            patch.object(executor, "submit", side_effect=RuntimeError("rejected")),
            pytest.raises(RuntimeError, match="rejected"),
        ):
            await run()
        assert await asyncio.wait_for(run(), 5) == 42
