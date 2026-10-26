# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded snapshots of PP outputs whose sends may outlive the next forward."""

from collections import deque
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch


class SendWork(Protocol):
    def is_completed(self) -> bool: ...

    def wait(self) -> None: ...


@dataclass
class _SendSlot:
    buffers: dict[str, torch.Tensor] = field(default_factory=dict)
    payload: dict[str, Any] = field(default_factory=dict)
    works: Sequence[SendWork] = ()


class PPSendBuffer:
    """Keep independent output storage until the matching send has completed.

    ``acquire`` runs before forward and only waits when all slots are busy.
    ``send`` snapshots on the producer stream, then posts communication on a
    separate stream. Keeping a reference to a graph output alone is unsafe:
    the next replay may overwrite the same allocation.
    """

    def __init__(
        self, capacity: int, device_module: Any = None, *, metadata_handles: int = 0
    ):
        if type(capacity) is not int or capacity < 1:
            raise ValueError("PP send buffer capacity must be a positive integer")
        self._metadata_handles = metadata_handles
        self.capacity = capacity
        self._device_module = device_module
        self._comm_stream = (
            device_module.Stream() if device_module is not None else None
        )
        self._slots = [_SendSlot() for _ in range(capacity)]
        self._free = deque(range(capacity))
        self._pending: deque[int] = deque()
        self._active: int | None = None
        self._reserved = False
        self._failure: Exception | None = None
        self.num_backpressure_waits = 0
        self.num_sends = 0

    @property
    def reserved_bytes(self) -> int:
        return sum(
            t.numel() * t.element_size()
            for slot in self._slots
            for t in slot.buffers.values()
        )

    @property
    def pending_count(self) -> int:
        """Posted sends, including transfers that may already be complete."""
        self._reap_completed()
        return len(self._pending)

    def reserve(self, tensors: Mapping[str, Any]) -> None:
        """Allocate maximum output storage before the KV memory budget is set."""
        if self._pending or self._active is not None:
            raise RuntimeError("Cannot reserve PP output storage while it is in use")
        for slot in self._slots:
            slot.buffers = {
                key: torch.empty_like(value, memory_format=torch.contiguous_format)
                for key, value in tensors.items()
                if isinstance(value, torch.Tensor)
            }
        self._reserved = True

    def _check_failure(self) -> None:
        if self._failure is not None:
            raise RuntimeError(
                "PP send buffer cannot be reused after a communication failure"
            ) from self._failure

    def _finish_oldest(self) -> None:
        index = self._pending[0]
        slot = self._slots[index]
        try:
            # Device work waits order subsequent slot writes on the caller's
            # stream. CPU metadata buffers must also remain alive until done.
            for work in slot.works:
                work.wait()
        except Exception as error:
            self._failure = error
            raise
        slot.works = ()
        slot.payload = {}
        self._pending.popleft()
        self._free.append(index)

    def _reap_completed(self) -> None:
        self._check_failure()
        while self._pending:
            slot = self._slots[self._pending[0]]
            # GroupCoordinator retains Gloo metadata separately. Its Work
            # does not reliably report completion until wait() is called.
            if not all(
                work.is_completed() for work in slot.works[self._metadata_handles :]
            ):
                break
            self._finish_oldest()

    def acquire(self) -> None:
        """Reserve a slot before starting a forward that can produce output."""
        self._check_failure()
        if self._active is not None:
            raise RuntimeError("A PP output slot has already been acquired")
        self._reap_completed()
        if not self._free:
            self.num_backpressure_waits += 1
            self._finish_oldest()
        self._active = self._free.popleft()

    def cancel(self) -> None:
        """Release a reservation when this step produces no PP output."""
        if self._active is not None:
            self._free.appendleft(self._active)
            self._active = None

    @staticmethod
    def _view(buffer: torch.Tensor, value: torch.Tensor) -> torch.Tensor | None:
        if buffer.device != value.device or buffer.dtype != value.dtype:
            return None
        if buffer.shape == value.shape:
            return buffer
        if (
            value.ndim
            and buffer.ndim == value.ndim
            and buffer.shape[1:] == value.shape[1:]
            and buffer.shape[0] >= value.shape[0]
        ):
            return buffer[: value.shape[0]]
        return None

    def send(
        self,
        tensors: Mapping[str, Any],
        send_fn: Callable[[dict[str, Any]], Sequence[SendWork]],
        on_snapshot: Callable[[], None] | None = None,
    ) -> None:
        self._check_failure()
        if self._active is None:
            raise RuntimeError("Acquire a PP output slot before sending")
        index = self._active
        slot = self._slots[index]
        payload: dict[str, Any] = {}
        for key, value in tensors.items():
            if not isinstance(value, torch.Tensor):
                payload[key] = deepcopy(value)
                continue
            buffer = slot.buffers.get(key)
            view = None if buffer is None else self._view(buffer, value)
            if view is None:
                if self._reserved:
                    raise ValueError(
                        f"PP tensor {key!r} exceeds its reserved output storage"
                    )
                slot.buffers[key] = torch.empty_like(
                    value, memory_format=torch.contiguous_format
                )
                view = slot.buffers[key]
            payload[key] = view.copy_(value)

        slot.payload = payload
        if on_snapshot is not None:
            on_snapshot()
        context = nullcontext()
        if self._device_module is not None:
            assert self._comm_stream is not None
            self._comm_stream.wait_stream(self._device_module.current_stream())
            context = self._device_module.stream(self._comm_stream)
        try:
            with context:
                slot.works = tuple(send_fn(payload))
        except Exception as error:
            # A failed post may have issued some transfers. Keep its storage
            # alive and prevent further reuse rather than guessing completion.
            self._failure = error
            raise
        self._pending.append(index)
        self._active = None
        self.num_sends += 1

    def drain(self) -> None:
        """Finish all sends before shutdown, sleep, or releasing the buffers."""
        self._check_failure()
        self.cancel()
        while self._pending:
            self._finish_oldest()
