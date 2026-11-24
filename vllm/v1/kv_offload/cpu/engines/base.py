# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Contract for a copy mechanism used by the CPU offload worker.

A mechanism (torch, plus the NIXL prototype in `engines/nixl.py`) implements
`CopyEngine` and returns `CopyStream` objects. Neither type exposes
stream/event primitives, so a thread- or RDMA-based mechanism fits the same
contract. Platform details -- memory registration, capabilities, and the
configuration of a mechanism -- are supplied by the runtime in `platforms/`.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class CopyBatch:
    """Descriptor storage for one transfer, owned by its copy engine.

    Addresses are always `uint64`, whatever pointer dtype the platform's copy
    op takes, so the planner never needs to know the platform. `payload` is
    engine-private storage; where an engine needs backing storage (the torch
    engine's pinned tensors, say), the three arrays are views into it and it
    must stay alive until the transfer completes. Engines that address memory
    directly (NIXL) leave it None.
    """

    src_ptrs: np.ndarray
    dst_ptrs: np.ndarray
    sizes: np.ndarray
    payload: Any = None

    @property
    def capacity(self) -> int:
        return len(self.src_ptrs)


class CopyStream(ABC):
    """One ordered async transfer on a mechanism's execution context.

    Streams are created by a `CopyEngine`, pooled by the worker, and submitted
    in order. A mechanism that runs transfers on a single FIFO worker may
    implement `wait_for` as a no-op.

    Error contract: a transfer failure is reported as `RuntimeError`. The
    worker records a failed job and continues. Any other exception -- for
    example `TimeoutError` for a request that may still land -- is fatal and
    must not be swallowed.
    """

    @abstractmethod
    def wait_for_compute(self) -> None:
        """Wait for work already queued on the device's compute stream."""

    @abstractmethod
    def wait_for(self, previous: "CopyStream") -> None:
        """Order this transfer after `previous` completes."""

    @abstractmethod
    def submit(self, batch: CopyBatch, num_ops: int) -> None:
        """Run the first `num_ops` copies of `batch`.

        Raises:
            RuntimeError: The device reported a failure.

        """

    @abstractmethod
    def abort(self) -> None:
        """Finish a stream whose `submit` raised.

        Clears any pending device error and makes `done`/`synchronize` usable.

        Raises:
            RuntimeError: The mechanism reports a failure while finishing.

        """

    @abstractmethod
    def done(self) -> bool:
        """Whether the transfer has completed.

        Raises:
            RuntimeError: The transfer failed.

        """

    @abstractmethod
    def synchronize(self) -> None:
        """Block until the transfer completes.

        Raises:
            RuntimeError: The transfer failed.

        """

    @abstractmethod
    def elapsed_seconds(self) -> float:
        """Duration of a completed transfer.

        Raises:
            RuntimeError: The transfer failed.

        """


class CopyEngine(ABC):
    """One copy mechanism, instantiated once per transfer direction.

    Attributes:
        device_to_cpu: True for a device->CPU engine, False for CPU->device.

    """

    device_to_cpu: bool

    @abstractmethod
    def new_stream(self) -> CopyStream:
        """Return a transfer stream for this engine's direction."""

    @abstractmethod
    def new_batch(self, num_ops: int) -> CopyBatch:
        """Allocate descriptor storage for at least `num_ops` copies."""
