# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from collections.abc import AsyncIterator

import pytest
from transformers.tokenization_utils_base import BatchEncoding

from vllm.utils.async_utils import AsyncMicrobatchTokenizer, merge_async_iterators


class _FakeTokenizer:
    def __init__(self):
        self.encode_calls: list[str] = []
        self.decode_calls: list[list[list[int]]] = []

    def __call__(self, prompts, **kwargs):
        if isinstance(prompts, list):
            self.encode_calls.extend(prompts)
            return {"input_ids": [[len(prompt)] for prompt in prompts]}

        self.encode_calls.append(prompts)
        return BatchEncoding({"input_ids": [len(prompts)]})

    def batch_decode(self, token_ids_list):
        self.decode_calls.append(token_ids_list)
        return [str(token_ids) for token_ids in token_ids_list]


async def _shutdown_microbatch_tokenizer(
    tokenizer: AsyncMicrobatchTokenizer,
) -> None:
    for task in tokenizer._batcher_tasks:
        task.cancel()
    await asyncio.gather(*tokenizer._batcher_tasks, return_exceptions=True)
    tokenizer._executor.shutdown(wait=True)


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
async def test_microbatch_tokenizer_skips_cancelled_encode_requests():
    fake_tokenizer = _FakeTokenizer()
    tokenizer = AsyncMicrobatchTokenizer(
        fake_tokenizer,
        max_batch_size=2,
        batch_wait_timeout_s=0.01,
    )
    queue = tokenizer._get_queue(
        asyncio.get_running_loop(),
        ("encode", True, False, None),
    )

    cancelled_future = asyncio.get_running_loop().create_future()
    active_future = asyncio.get_running_loop().create_future()
    cancelled_future.cancel()

    await queue.put(("cancelled", {}, cancelled_future))
    await queue.put(("active", {}, active_future))

    assert await asyncio.wait_for(active_future, timeout=1) == BatchEncoding(
        {"input_ids": [6]}
    )
    assert fake_tokenizer.encode_calls == ["active"]

    await _shutdown_microbatch_tokenizer(tokenizer)


@pytest.mark.asyncio
async def test_microbatch_tokenizer_skips_fully_cancelled_encode_batch():
    fake_tokenizer = _FakeTokenizer()
    tokenizer = AsyncMicrobatchTokenizer(
        fake_tokenizer,
        max_batch_size=2,
        batch_wait_timeout_s=0.01,
    )
    queue = tokenizer._get_queue(
        asyncio.get_running_loop(),
        ("encode", True, False, None),
    )

    for prompt in ("cancelled-1", "cancelled-2"):
        result_future = asyncio.get_running_loop().create_future()
        result_future.cancel()
        await queue.put((prompt, {}, result_future))

    await asyncio.sleep(0.05)
    assert fake_tokenizer.encode_calls == []

    await _shutdown_microbatch_tokenizer(tokenizer)


@pytest.mark.asyncio
async def test_microbatch_tokenizer_skips_cancelled_decode_requests():
    fake_tokenizer = _FakeTokenizer()
    tokenizer = AsyncMicrobatchTokenizer(
        fake_tokenizer,
        max_batch_size=2,
        batch_wait_timeout_s=0.01,
    )
    queue = tokenizer._get_queue(asyncio.get_running_loop(), ("decode",))

    cancelled_future = asyncio.get_running_loop().create_future()
    active_future = asyncio.get_running_loop().create_future()
    cancelled_future.cancel()

    await queue.put(([1], cancelled_future))
    await queue.put(([2, 3], active_future))

    assert await asyncio.wait_for(active_future, timeout=1) == "[2, 3]"
    assert fake_tokenizer.decode_calls == [[[2, 3]]]

    await _shutdown_microbatch_tokenizer(tokenizer)
