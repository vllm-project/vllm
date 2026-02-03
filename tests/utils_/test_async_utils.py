# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import AsyncIterator

import pytest

from vllm.utils.async_utils import merge_async_iterators


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
