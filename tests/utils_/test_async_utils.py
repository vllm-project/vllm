# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import AsyncIterator

import pytest

from vllm.utils.async_utils import (
    await_with_cancellation_drain,
    merge_async_iterators,
)


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
@pytest.mark.parametrize("fail", [False, True])
async def test_cancellation_drain_preserves_cancellation_until_work_finishes(fail):
    pending = asyncio.get_running_loop().create_future()
    cancelled = asyncio.Event()
    task = asyncio.create_task(
        await_with_cancellation_drain(pending, on_cancel=cancelled.set)
    )
    await asyncio.sleep(0)
    task.cancel()
    await asyncio.wait_for(cancelled.wait(), timeout=5)
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    assert not pending.cancelled()
    if fail:
        pending.set_exception(RuntimeError("work failed"))
    else:
        pending.set_result("finished")
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=5)


@pytest.mark.asyncio
async def test_cancellation_drain_returns_result_and_propagates_work_errors():
    pending = asyncio.get_running_loop().create_future()
    pending.set_result("finished")
    assert await await_with_cancellation_drain(pending) == "finished"
    failed = asyncio.get_running_loop().create_future()
    failed.set_exception(RuntimeError("work failed"))
    with pytest.raises(RuntimeError, match="work failed"):
        await await_with_cancellation_drain(failed)
