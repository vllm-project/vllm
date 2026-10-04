# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import Executor, Future
from typing import Any, NamedTuple

from vllm.utils.async_utils import await_with_cancellation_drain

from .base import MediaRef


class MediaDecodeJob(NamedTuple):
    """A submitted decode and its location in the original request."""

    modality: str
    original_index: int
    future: Future[Any]


DecodeError = Callable[[str, int, Exception], Exception]


def submit_media_decodes(
    items: Iterable[tuple[str, int, MediaRef[Any]]],
    executor: Executor,
    jobs: list[MediaDecodeJob],
) -> None:
    """Append submitted work immediately so partial submission can be drained."""
    for modality, index, ref in items:
        if not ref.is_decoded:
            jobs.append(MediaDecodeJob(modality, index, executor.submit(ref.decode)))


def collect_media_decodes(
    jobs: Sequence[MediaDecodeJob], error_factory: DecodeError
) -> None:
    """Drain every decode before reporting the first failure in request order."""
    first_error: tuple[MediaDecodeJob, Exception] | None = None
    for job in jobs:
        try:
            job.future.result()
        except Exception as error:
            if first_error is None:
                first_error = job, error
    if first_error is not None:
        job, cause = first_error
        raise error_factory(job.modality, job.original_index, cause) from cause


async def collect_media_decodes_async(
    jobs: Sequence[MediaDecodeJob], error_factory: DecodeError
) -> None:
    """Drain decodes through caller cancellation without blocking the event loop."""
    if not jobs:
        return
    pending = asyncio.gather(
        *(asyncio.wrap_future(job.future) for job in jobs), return_exceptions=True
    )
    results = await await_with_cancellation_drain(pending)
    for job, result in zip(jobs, results):
        if isinstance(result, Exception):
            raise error_factory(job.modality, job.original_index, result) from result
        if isinstance(result, BaseException):
            raise result
