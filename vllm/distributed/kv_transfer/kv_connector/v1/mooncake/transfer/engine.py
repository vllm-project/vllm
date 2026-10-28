# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum


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
    range_id: int
    local_addr: int
    remote_addr: int
    length: int


@dataclass(frozen=True)
class TransferEvent:
    range_id: int
    state: TransferState
    transferred_bytes: int | None = None
    error: str | None = None


class TransferBackend(ABC):
    """Moves KV blocks between two workers and reports per-range outcomes."""

    @property
    @abstractmethod
    def hostname(self) -> str:
        """Address peers use to reach this worker."""

    @property
    @abstractmethod
    def rpc_port(self) -> int:
        """Port peers use to reach this worker's engine."""

    @abstractmethod
    def register_regions(self, regions: Sequence[MemoryRegion]) -> None:
        """Pins local memory for transfers. Raises RuntimeError on failure."""

    @abstractmethod
    def unregister_regions(self, regions: Sequence[MemoryRegion]) -> None:
        """Releases memory pinned by `register_regions`."""

    @abstractmethod
    def submit(self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]) -> None:
        """Starts one transfer request. Called from a worker thread and may
        block until the engine finishes, so never call it on the event loop."""

    @abstractmethod
    def poll(self) -> list[TransferEvent]:
        """Returns the terminal events observed since the previous call."""

    @abstractmethod
    def shutdown(self) -> None:
        """Stops the backend. No submit or poll after this returns."""
