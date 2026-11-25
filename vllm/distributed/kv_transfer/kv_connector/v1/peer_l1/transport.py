# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Byte-range reads with explicit completion and DMA quiescence."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from enum import Enum
from typing import ClassVar, Protocol

import torch


class ReadState(Enum):
    PENDING = "pending"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class ReadBatch(Protocol):
    """Expose nonblocking status to the RDMA I/O and model threads."""

    def state(self) -> ReadState:
        """Return immediately; `SUCCEEDED` guarantees no further DMA touches this batch.

        `FAILED` alone gives no such lifetime guarantee.
        """
        ...

    def quiesced(self) -> bool:
        """Prove no future DMA without waiting, from either worker thread.

        Return false on lock contention; synchronize status-handle release
        with the same lock used by status queries.
        """
        ...


class RemoteSession(Protocol):
    """Prepare and post reads on the worker process's dedicated RDMA I/O thread."""

    def prepare(self, ranges: Sequence[tuple[int, int, int]]) -> ReadBatch:
        """Prepare `(local offset, remote offset, bytes)` ranges without posting."""
        ...

    def post(self, batch: ReadBatch) -> None:
        """Post a prepared batch; may raise after partial submission.

        On exception the executor cannot prove which DMA was posted and keeps
        the read's resources and target quarantine until process exit.
        """
        ...


class PeerReadTransport(ABC):
    """Register worker KV memory and open owner-specific read sessions.

    Construction runs during worker startup; open/prepare/post run on its I/O
    thread. Status and drain queries must also be safe on the model thread.
    """

    name: ClassVar[str]
    max_ranges_per_post: ClassVar[int]
    max_range_bytes: ClassVar[int]

    @classmethod
    @abstractmethod
    def available(cls) -> bool: ...

    @abstractmethod
    def __init__(self, storage: torch.UntypedStorage, host: str) -> None:
        """Register all backing storage synchronously or fail worker startup."""
        ...

    @abstractmethod
    def advertisement(self) -> bytes:
        """Export opaque registration metadata for compatible remote workers."""
        ...

    @abstractmethod
    def open(self, advertisement: bytes) -> RemoteSession:
        """Open an owner session on the I/O thread; setup may block on the provider."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Return promptly without revoking published or possibly active memory.

        Remote unilateral reads may outlive every local receive.
        """
        ...


def _mori_transport() -> type[PeerReadTransport]:
    from .transports.mori import MoRITransport

    return MoRITransport


TRANSPORTS: dict[str, Callable[[], type[PeerReadTransport]]] = {"mori": _mori_transport}


def get_transport(name: str | None = None) -> type[PeerReadTransport]:
    if name is not None:
        if name not in TRANSPORTS:
            raise ValueError(
                f"Unknown RDMA transport {name!r}; choose from {list(TRANSPORTS)}"
            )
        cls = TRANSPORTS[name]()
        if not cls.available():
            raise RuntimeError(
                f"RDMA transport {name!r} is unavailable; install its dependencies"
            )
        return cls
    available = []
    for load in dict.fromkeys(TRANSPORTS.values()):
        cls = load()
        if cls.available():
            available.append(cls)
    if len(available) != 1:
        raise RuntimeError(
            "Set rdma_transport to one available implementation; "
            f"available={[cls.name for cls in available]}, "
            f"registered={list(TRANSPORTS)}"
        )
    return available[0]
