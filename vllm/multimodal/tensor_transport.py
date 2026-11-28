# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared contracts for externally stored multimodal tensor bundles."""

import asyncio
import contextlib
import hashlib
import weakref
from collections import deque
from collections.abc import AsyncGenerator, AsyncIterator, Callable, Iterable
from concurrent.futures import Executor
from typing import Literal, TypeVar

import torch
from pydantic import BaseModel, ConfigDict, Field
from safetensors.torch import load as load_safetensors

from vllm.exceptions import VLLMUnprocessableEntityError

MM_TENSOR_FORMAT = "safetensors-v1"

_T = TypeVar("_T")


class BaseMMTensorReference(BaseModel):
    """Common metadata for one externally stored tensor bundle."""

    model_config = ConfigDict(extra="forbid")

    format: Literal["safetensors-v1"]
    nbytes: int = Field(gt=0)
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    """Lowercase hex SHA-256 of the object bytes."""


class MMTensorTransportError(VLLMUnprocessableEntityError):
    """Raised when an external multimodal tensor bundle cannot be used."""


def validate_mm_tensor_request_size(
    references: Iterable[BaseMMTensorReference],
    max_nbytes: int,
) -> int:
    """Check and return the total declared size of one request's objects."""
    nbytes = sum(reference.nbytes for reference in references)
    if nbytes > max_nbytes:
        raise ValueError(
            f"Multimodal tensor objects in one request declare {nbytes} bytes, "
            f"exceeding the configured limit of {max_nbytes} bytes"
        )
    return nbytes


class MMTensorReservation:
    """Bytes reserved from an `MMTensorMemoryBudget`, returned by the last holder."""

    def __init__(self, budget: "MMTensorMemoryBudget", nbytes: int) -> None:
        self._budget = budget
        self._nbytes = nbytes
        self._holders = 1

    def hold(self) -> None:
        self._holders += 1

    def release(self) -> None:
        self._holders -= 1
        if self._holders == 0:
            self._budget._release(self._nbytes)


class MMTensorMemoryBudget:
    """Bytes one API server may hold for referenced multimodal tensors.

    Requests wait in arrival order until their reservation fits. All methods
    must be called from the event loop thread.
    """

    def __init__(self, max_nbytes: int) -> None:
        self.max_nbytes = max_nbytes
        self.used_nbytes = 0
        self._waiters: deque[tuple[int, asyncio.Future[None]]] = deque()

    async def reserve(self, nbytes: int) -> MMTensorReservation:
        if nbytes > self.max_nbytes:
            raise ValueError(
                f"Reserving {nbytes} bytes for multimodal tensors exceeds "
                f"max_reserved_bytes ({self.max_nbytes})"
            )
        if not self._waiters and self.used_nbytes + nbytes <= self.max_nbytes:
            self.used_nbytes += nbytes
            return MMTensorReservation(self, nbytes)

        waiter = asyncio.get_running_loop().create_future()
        self._waiters.append((nbytes, waiter))
        try:
            await waiter
        except asyncio.CancelledError:
            if waiter.done() and not waiter.cancelled():
                # Granted just before the cancellation.
                self._release(nbytes)
            else:
                self._wake()
            raise
        return MMTensorReservation(self, nbytes)

    def _release(self, nbytes: int) -> None:
        self.used_nbytes -= nbytes
        self._wake()

    def _wake(self) -> None:
        while self._waiters:
            nbytes, waiter = self._waiters[0]
            if waiter.done():
                self._waiters.popleft()
            elif self.used_nbytes + nbytes <= self.max_nbytes:
                self._waiters.popleft()
                self.used_nbytes += nbytes
                waiter.set_result(None)
            else:
                break


class ReleasingStream(AsyncIterator[_T]):
    """Iterate `stream` and release `reservations` once it's done with.

    That's when the stream ends or fails, when it's closed, or when it's
    dropped without ever being iterated, which an async generator's
    `finally` would miss.
    """

    def __init__(
        self,
        stream: AsyncGenerator[_T, None],
        reservations: Iterable[MMTensorReservation],
    ) -> None:
        self._stream = stream
        self._release = weakref.finalize(
            self,
            _release_soon,
            asyncio.get_running_loop(),
            tuple(reservations),
        )

    def __aiter__(self) -> "ReleasingStream[_T]":
        return self

    async def __anext__(self) -> _T:
        try:
            return await self._stream.__anext__()
        except BaseException:
            self._release()
            raise

    async def aclose(self) -> None:
        try:
            await self._stream.aclose()
        finally:
            self._release()


def _release_soon(
    loop: asyncio.AbstractEventLoop, reservations: tuple[MMTensorReservation, ...]
) -> None:
    # A finalizer can run on any thread; the budget belongs to the loop's.
    def release() -> None:
        for reservation in reservations:
            reservation.release()

    with contextlib.suppress(RuntimeError):  # The loop has closed.
        loop.call_soon_threadsafe(release)


async def run_in_read_slot(
    semaphore: asyncio.Semaphore,
    fn: Callable[[], _T],
    reservation: MMTensorReservation | None = None,
    executor: Executor | None = None,
) -> _T:
    """Run `fn` in `executor` while holding a slot of `semaphore`.

    The slot, and a hold on `reservation`, are released only when `fn`
    returns, even if the caller is cancelled first, because the executor
    thread keeps running.
    """
    await semaphore.acquire()
    try:
        future = asyncio.get_running_loop().run_in_executor(executor, fn)
    except BaseException:
        semaphore.release()
        raise
    if reservation is not None:
        reservation.hold()

    def release(done: asyncio.Future[_T]) -> None:
        semaphore.release()
        if reservation is not None:
            reservation.release()
        if not done.cancelled():
            # Mark the exception retrieved when the caller was cancelled.
            done.exception()

    future.add_done_callback(release)
    return await asyncio.shield(future)


def verify_mm_tensor_digest(payload: bytes, expected_digest: str) -> bool:
    return hashlib.sha256(payload).hexdigest() == expected_digest


def decode_mm_tensor_bundle(
    payload: bytes, *, location: str
) -> dict[str, torch.Tensor]:
    """Decode one SafeTensors bundle whose digest has been verified."""
    try:
        tensors = load_safetensors(payload)
    except Exception as e:
        raise MMTensorTransportError(
            f"Could not decode multimodal tensor object {location} "
            f"as {MM_TENSOR_FORMAT}"
        ) from e
    if not tensors:
        raise MMTensorTransportError(
            f"Multimodal tensor object {location} contains no tensors"
        )
    return tensors
