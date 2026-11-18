# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Authorization and transfer contracts independent of engine scheduling."""

import hashlib
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Protocol

import numpy as np

from .native import CleanupQueue, Operation, credential_array, key_array, library


def peer_key(identity: bytes) -> bytes:
    """Hash the entire vLLM block identity, including its four-byte group ID."""
    return hashlib.sha256(identity).digest()


@dataclass(frozen=True)
class Grant:
    endpoint: str
    epoch: int
    reader: int
    sequence: int
    rows: tuple[int, ...]
    acquired_before: float
    lease_s: float | None = None


class AccessMode(Enum):
    LOAD_STORE = "load_store"
    RDMA_READ = "rdma_read"


class Transfer(Protocol):
    """Failures may precede quiescence; retain targets and grants until quiesced."""

    def done(self) -> bool: ...

    def failed_requests(self) -> set[str]: ...

    def quiesced(self) -> bool: ...


class Mover(Protocol):
    def submit(
        self, grants: list[Grant], copies: list[tuple[int, int, int]]
    ) -> Transfer: ...


class PeerSession(Protocol):
    """A memory session chooses loading and failure semantics for its caller."""

    access_mode: AccessMode
    cost: Literal["local", "fabric", "rdma"]
    mover: Mover
    mapping: Any

    def submit(self, grant: Grant, copies: list[tuple[int, int, int]]) -> Transfer: ...


class OwnerChannel(Protocol):
    def acquire_exact(self, keys: list[bytes]) -> Grant | None: ...

    def cancel_unsubmitted(self, seq: int) -> None:
        """Cancel authorization only when no read has been submitted."""
        ...

    def release(self, grant: Grant) -> None:
        """Release a grant after the submitted transfer has quiesced."""
        ...


class OwnerBusy(Exception):
    def __init__(self, generation: int, window: int):
        self.generation = generation
        self.window = window
        super().__init__(generation, window)


class OwnerDeferred(OwnerBusy):
    def __init__(self, generation: int, sequence: int, acquired_before: float):
        super().__init__(generation, -1)
        self.sequence = sequence
        self.acquired_before = acquired_before


class NativeOwnerChannel:
    def __init__(
        self,
        endpoint: str,
        epoch: int,
        reader: int,
        cleanup: CleanupQueue,
        timeout_ms: int = 50,
        *,
        lease_s: float | None = None,
    ):
        self.endpoint, self.epoch, self.reader = endpoint, epoch, reader
        self.cleanup, self.timeout_ms = cleanup, timeout_ms
        self.lease_s = lease_s
        self._sequence = 0
        self._lib = library()
        self._credential = credential_array(cleanup.credential)

    def _rpc(
        self, op: Operation, seq: int, keys: list[bytes]
    ) -> tuple[int, np.ndarray]:
        identities = key_array(keys)
        out = np.empty(max(2, len(keys)), dtype=np.int64)
        result = self._lib.l1_owner_rpc(
            self.endpoint.encode(),
            op,
            self.epoch,
            self.reader,
            seq,
            identities.ctypes.data,
            out.ctypes.data,
            len(keys),
            self.timeout_ms,
            self._credential.ctypes.data,
        )
        return result, out

    def acquire_exact(
        self,
        keys: list[bytes],
        *,
        defer: bool = False,
        sequence: int | None = None,
        acquired_before: float | None = None,
    ) -> Grant | None:
        started = time.monotonic() if acquired_before is None else acquired_before
        if sequence is None:
            self._sequence += 1
            sequence = self._sequence
        seq = sequence
        op = (
            (
                Operation.ACQUIRE_DEFERRED_LEASED
                if self.lease_s is not None
                else Operation.ACQUIRE_DEFERRED
            )
            if defer
            else (
                Operation.ACQUIRE_LEASED
                if self.lease_s is not None
                else Operation.ACQUIRE_EXACT
            )
        )
        n, out = self._rpc(op, seq, keys)
        if n == 2 and out[0] <= -3:
            if out[1] == -1:
                raise OwnerDeferred(-3 - int(out[0]), seq, started)
            raise OwnerBusy(-3 - int(out[0]), int(out[1]))
        if n != len(keys):
            if n == -1:
                self.cleanup.enqueue(
                    self.cleanup.create(self.endpoint, self.epoch, self.reader, seq)
                )
            return None
        return Grant(
            self.endpoint,
            self.epoch,
            self.reader,
            seq,
            tuple(int(row) for row in out[:n]),
            started,
            self.lease_s,
        )

    def cancel_unsubmitted(self, seq: int) -> None:
        n, _ = self._rpc(Operation.CANCEL, seq, [])
        if n < 0:
            self.cleanup.enqueue(
                self.cleanup.create(self.endpoint, self.epoch, self.reader, seq)
            )

    def release(self, grant: Grant) -> None:
        self.cleanup.enqueue(
            self.cleanup.create(
                grant.endpoint, grant.epoch, grant.reader, grant.sequence
            )
        )
