# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


class TransferOp(Enum):
    READ = "read"
    WRITE = "write"


class TransferState(Enum):
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class MemoryRegion:
    base_addr: int
    length: int


@dataclass(frozen=True)
class RangeRequest:
    local_addr: int
    remote_addr: int
    length: int


@dataclass(frozen=True)
class TransferEvent:
    state: TransferState
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BatchTransferResult:
    """Terminal outcome covering `num_ranges`, not a count of successful ranges."""

    num_ranges: int
    state: TransferState
    error: str | None = None


class TransferBackend(ABC):
    """Moves KV blocks between two workers synchronously."""

    @property
    @abstractmethod
    def rpc_port(self) -> int:
        """Port peers use to reach this worker's engine."""

    @abstractmethod
    def register_regions(self, regions: Sequence[MemoryRegion]) -> None:
        """Pins local memory for transfers.

        An empty sequence is a no-op. Raises RuntimeError when the engine
        rejects the memory.
        """

    @abstractmethod
    def unregister_regions(self, regions: Sequence[MemoryRegion]) -> None:
        """Releases memory pinned by `register_regions`.

        An empty sequence is a no-op.
        """

    @abstractmethod
    def transfer(
        self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]
    ) -> list[TransferEvent]:
        """Moves every range and returns one terminal event for each.

        The result lines up with `ranges` by position, and an empty request
        returns an empty list. Blocks until every range reached a terminal
        state. Called from worker threads, so an implementation must accept
        concurrent calls.
        """

    @abstractmethod
    def transfer_sync_arrays(
        self,
        peer: str,
        op: TransferOp,
        local_addrs: NDArray[np.uint64],
        remote_addrs: NDArray[np.uint64],
        lengths: NDArray[np.uint64],
    ) -> BatchTransferResult:
        """Moves equal-length, 1D, contiguous native uint64 columns.

        Blocks until all ranges are terminal, holding the input buffers for the
        duration of the call. Callers must not modify them meanwhile. Concurrent
        calls keep their own inputs and results. Empty input completes without
        engine work. COMPLETED means all ranges succeeded; FAILED does not mean
        all failed. Use the first failure in input order when available.

        Returns one batch summary without constructing per-range requests,
        events or status arrays. Invalid inputs and engine exceptions propagate.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Stops the backend.

        The caller stops submitting before calling this, but work already
        queued on its worker threads is not cancelled, so a `transfer` call may
        still be running or start afterwards. An implementation must tolerate
        that.
        """
