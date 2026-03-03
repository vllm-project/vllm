# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Contains helpers related to asynchronous code.

This is similar in concept to the `asyncio` module.
"""

import asyncio
import contextlib
from asyncio import FIRST_COMPLETED, AbstractEventLoop, Future, Task
from collections.abc import AsyncGenerator, Awaitable, Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, TypeVar

from transformers.tokenization_utils_base import BatchEncoding
from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


class AsyncMicrobatchTokenizer:
    """Asynchronous tokenizer with micro-batching.

    Pulls pending encode/decode requests from a queue and batches them
    up to reduce overhead. A single-thread ThreadPoolExecutor is used
    so the event loop stays responsive.
    """

    def __init__(
        self,
        tokenizer,
        max_batch_size: int = 32,
        batch_wait_timeout_s: float = 0.002,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s

        self._loop = asyncio.get_running_loop()
        self._queues: dict[
            tuple,
            asyncio.Queue[tuple[str, dict, Future] | tuple[list[int], Future]],
        ] = {}
        self._batcher_tasks: list[Task] = []

        # Single-thread executor for blocking tokenizer calls.
        self._executor = ThreadPoolExecutor(max_workers=1)

    # === Public async API ===
    async def __call__(self, prompt, **kwargs) -> BatchEncoding:
        result_future: Future = self._loop.create_future()
        key = self._queue_key("encode", kwargs)
        queue = self._get_queue(self._loop, key)
        await queue.put((prompt, kwargs, result_future))
        return await result_future

    async def encode(self, prompt, **kwargs) -> list[int]:
        return (await self(prompt, **kwargs)).input_ids

    async def decode(self, token_ids, **kwargs) -> str:
        result_future: Future = self._loop.create_future()
        key = self._queue_key("decode", kwargs)
        queue = self._get_queue(self._loop, key)
        await queue.put((token_ids, result_future))
        return await result_future

    # === Internal helpers ===
    def _get_queue(
        self, loop: asyncio.AbstractEventLoop, key: tuple
    ) -> asyncio.Queue[tuple[str, dict, Future] | tuple[list[int], Future]]:
        """Get the request queue for the given operation key, creating a new
        queue and batcher task if needed."""
        queue = self._queues.get(key)
        if queue is None:
            self._queues[key] = queue = asyncio.Queue()
            if key[0] == "encode":
                can_batch = key[1] != "other"
                coro = self._batch_encode_loop(queue, can_batch)
            else:
                assert key[0] == "decode", f"Unknown operation type: {key[0]}."
                coro = self._batch_decode_loop(queue)
            self._batcher_tasks.append(loop.create_task(coro))
        return queue

    async def _batch_encode_loop(self, queue: asyncio.Queue, can_batch: bool):
        """Batch incoming encode requests for efficiency."""
        while True:
            prompt, kwargs, result_future = await queue.get()
            prompts = [prompt]
            kwargs_list = [kwargs]
            result_futures = [result_future]
            deadline = self._loop.time() + self.batch_wait_timeout_s

            while len(prompts) < self.max_batch_size:
                timeout = deadline - self._loop.time()
                if timeout <= 0:
                    break
                try:
                    prompt, kwargs, result_future = await asyncio.wait_for(
                        queue.get(), timeout
                    )
                    prompts.append(prompt)
                    result_futures.append(result_future)
                    if not can_batch:
                        kwargs_list.append(kwargs)
                except asyncio.TimeoutError:
                    break

            try:
                # If every request uses identical kwargs we can run a single
                # batched tokenizer call for a big speed-up.
                if can_batch and len(prompts) > 1:
                    batch_encode_fn = partial(self.tokenizer, prompts, **kwargs)
                    results = await self._loop.run_in_executor(
                        self._executor, batch_encode_fn
                    )

                    for i, fut in enumerate(result_futures):
                        if not fut.done():
                            data = {k: v[i] for k, v in results.items()}
                            fut.set_result(BatchEncoding(data))
                else:
                    encode_fn = lambda prompts=prompts, kwargs=kwargs_list: [
                        self.tokenizer(p, **kw) for p, kw in zip(prompts, kwargs)
                    ]
                    results = await self._loop.run_in_executor(
                        self._executor, encode_fn
                    )

                    for fut, res in zip(result_futures, results):
                        if not fut.done():
                            fut.set_result(res)
            except Exception as e:
                for fut in result_futures:
                    if not fut.done():
                        fut.set_exception(e)

    async def _batch_decode_loop(self, queue: asyncio.Queue):
        """Batch incoming decode requests for efficiency."""
        while True:
            token_ids, result_future = await queue.get()
            token_ids_list = [token_ids]
            result_futures = [result_future]
            deadline = self._loop.time() + self.batch_wait_timeout_s

            while len(token_ids_list) < self.max_batch_size:
                timeout = deadline - self._loop.time()
                if timeout <= 0:
                    break
                try:
                    token_ids, result_future = await asyncio.wait_for(
                        queue.get(), timeout
                    )
                    token_ids_list.append(token_ids)
                    result_futures.append(result_future)
                except asyncio.TimeoutError:
                    break

            try:
                # Perform a single batched decode call for all requests
                results = await self._loop.run_in_executor(
                    self._executor, self.tokenizer.batch_decode, token_ids_list
                )
                for fut, res in zip(result_futures, results):
                    if not fut.done():
                        fut.set_result(res)
            except Exception as e:
                for fut in result_futures:
                    if not fut.done():
                        fut.set_exception(e)

    def _queue_key(self, op: str, kwargs: dict) -> tuple:
        """
        Return a normalized key describing operation + kwargs.

        - `add_special_tokens`: {True/False}
        - `truncation`: {True/False}
          - If `truncation` is False (`max_length` is None),
            returns a key for a can_batch queue.
          - If `truncation` is True and `max_length` is None or equals
            `tokenizer.model_max_length`, returns a key for a can_batch queue.
          - Otherwise, returns a key for a cannot_batch queue.

        Examples:
          - Decode: ("decode",)
          - Encode typical:
            ("encode", add_special_tokens, bool_truncation, max_length_label)
          - Fallback: ("encode", "other")
        """

        if op == "decode":
            return ("decode",)

        add_special_tokens = kwargs.get("add_special_tokens", True)
        truncation = kwargs.get("truncation", False)
        max_length = kwargs.get("max_length")

        if not truncation:
            return "encode", add_special_tokens, False, None

        model_max = getattr(self.tokenizer, "model_max_length", None)
        if max_length is None or (model_max is not None and max_length == model_max):
            return "encode", add_special_tokens, True, "model_max"

        return "encode", "other"

    def __del__(self):
        if (
            (tasks := getattr(self, "_batcher_tasks", None))
            and (loop := getattr(self, "_loop", None))
            and not loop.is_closed()
        ):

            def cancel_tasks():
                for task in tasks:
                    task.cancel()

            loop.call_soon_threadsafe(cancel_tasks)


def cancel_task_threadsafe(task: Task):
    if task and not task.done():
        run_in_loop(task.get_loop(), task.cancel)


def make_async(
    func: Callable[P, T],
    executor: Executor | None = None,
) -> Callable[P, Awaitable[T]]:
    """
    Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Future[T]:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)

    return _async_wrapper


def run_in_loop(loop: AbstractEventLoop, function: Callable, *args):
    if in_loop(loop):
        function(*args)
    elif not loop.is_closed():
        loop.call_soon_threadsafe(function, *args)


def in_loop(event_loop: AbstractEventLoop) -> bool:
    try:
        return asyncio.get_running_loop() == event_loop
    except RuntimeError:
        return False


# A hack to pass mypy
if TYPE_CHECKING:

    def anext(it: AsyncGenerator[T, None]):
        return it.__anext__()


async def merge_async_iterators(
    *iterators: AsyncGenerator[T, None],
) -> AsyncGenerator[tuple[int, T], None]:
    """Merge multiple asynchronous iterators into a single iterator.

    This method handle the case where some iterators finish before others.
    When it yields, it yields a tuple (i, item) where i is the index of the
    iterator that yields the item.
    """
    if len(iterators) == 1:
        # Fast-path single iterator case.
        async for item in iterators[0]:
            yield 0, item
        return

    loop = asyncio.get_running_loop()

    awaits = {loop.create_task(anext(it)): (i, it) for i, it in enumerate(iterators)}
    try:
        while awaits:
            done, _ = await asyncio.wait(awaits.keys(), return_when=FIRST_COMPLETED)
            for d in done:
                pair = awaits.pop(d)
                try:
                    item = await d
                    i, it = pair
                    awaits[loop.create_task(anext(it))] = pair
                    yield i, item
                except StopAsyncIteration:
                    pass
    finally:
        # Cancel any remaining iterators
        for f, (_, it) in awaits.items():
            with contextlib.suppress(BaseException):
                f.cancel()
                await it.aclose()


async def collect_from_async_generator(iterator: AsyncGenerator[T, None]) -> list[T]:
    """Collect all items from an async generator into a list."""
    items = []
    async for item in iterator:
        items.append(item)
    return items
