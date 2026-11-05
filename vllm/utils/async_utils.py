# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contains helpers related to asynchronous code.

This is similar in concept to the `asyncio` module.
"""

import asyncio
import contextlib
from asyncio import FIRST_COMPLETED, AbstractEventLoop, Future, Task
from collections.abc import AsyncGenerator, Awaitable, Callable
from concurrent.futures import Executor, ThreadPoolExecutor
from functools import partial
from typing import TYPE_CHECKING, Any, TypeVar

from typing_extensions import ParamSpec

P = ParamSpec("P")
T = TypeVar("T")


def cancel_task_threadsafe(task: Task):
    if task and not task.done():
        run_in_loop(task.get_loop(), task.cancel)


def make_async(
    func: Callable[P, T],
    executor: Executor | None = None,
) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.
    """

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Future[T]:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        return loop.run_in_executor(executor=executor, func=p_func)

    return _async_wrapper


def make_async_with_semaphore(
    func: Callable[P, T],
    executor: ThreadPoolExecutor,
) -> Callable[P, Awaitable[T]]:
    """Take a blocking function, and run it on in an executor thread.

    This function prevents the blocking function from blocking the
    asyncio event loop.
    The code in this function needs to be thread safe.

    The function is wrapped in a semaphore to limit the number of
    concurrent executions making it easier to cancel tasks before they start.
    """
    semaphore = asyncio.Semaphore(executor._max_workers)

    async def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
        loop = asyncio.get_event_loop()
        p_func = partial(func, *args, **kwargs)
        async with semaphore:
            return await loop.run_in_executor(executor, p_func)

    return _async_wrapper


def make_async_batched(
    func: Callable[P, T],
    executor: Executor,
    max_batch_size: int = 256,
) -> Callable[P, Future[T]]:
    """Like `make_async`, but the calls made in one event loop iteration are
    submitted as one executor job per worker.

    Submitting a job costs the event loop about as much as a small call, such
    as rendering a short prompt. Grouping the calls pays that cost once per job.
    Nothing waits for more calls: a single call is submitted on the next loop
    iteration, and each result is handed back as soon as its call finishes.
    """
    pending: dict[AbstractEventLoop, list[tuple[Future[T], Callable[[], T]]]] = {}

    def _run(loop: AbstractEventLoop, calls: list[tuple[Future[T], Callable[[], T]]]):
        for fut, call in calls:
            if fut.cancelled():
                continue
            try:
                result, exc = call(), None
            except BaseException as e:
                result, exc = None, e
            # One wakeup per job would hold every result back until the last
            # call of the job finishes.
            try:
                loop.call_soon_threadsafe(_set_future_result, fut, result, exc)
            except RuntimeError:  # The loop is closed.
                return

    def _flush(loop: AbstractEventLoop):
        calls = pending.pop(loop)
        num_workers = max(1, getattr(executor, "_max_workers", 1))
        size = min(max_batch_size, (len(calls) + num_workers - 1) // num_workers)
        for start in range(0, len(calls), size):
            chunk = calls[start : start + size]
            try:
                executor.submit(_run, loop, chunk)
            except RuntimeError as e:  # The executor is shut down.
                for fut, _ in chunk:
                    _set_future_result(fut, None, e)

    def _async_wrapper(*args: P.args, **kwargs: P.kwargs) -> Future[T]:
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        calls = pending.get(loop)
        if calls is None:
            calls = pending[loop] = []
            loop.call_soon(_flush, loop)
        calls.append((fut, partial(func, *args, **kwargs)))
        return fut

    return _async_wrapper


def _set_future_result(
    fut: Future[Any], result: Any, exc: BaseException | None
) -> None:
    if fut.done():
        return
    if exc is None:
        fut.set_result(result)
    else:
        fut.set_exception(exc)


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
        iterator: AsyncGenerator[T, None] | None = iterators[0]
        try:
            async for item in iterator:  # type: ignore[union-attr]
                yield 0, item
            iterator = None
        finally:
            if iterator is not None:
                with contextlib.suppress(BaseException):
                    await iterator.aclose()
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
