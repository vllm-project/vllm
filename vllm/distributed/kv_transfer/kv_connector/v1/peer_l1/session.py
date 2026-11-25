# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Peer block identities, grants, and native owner authorization."""

import hashlib
import time
from dataclasses import dataclass

import numpy as np

from .native import CleanupQueue, Operation, credential_array, key_array, library


def peer_key(identity: bytes) -> bytes:
    """Hash the entire vLLM block identity, including its four-byte group ID."""
    return hashlib.sha256(identity).digest()


@dataclass(frozen=True)
class Grant:
    """Identify source pins requiring explicit release, with an optional RDMA lease."""

    endpoint: str
    epoch: int
    reader: int
    sequence: int
    rows: tuple[int, ...]
    acquired_before: float
    lease_s: float | None = None


def release_grant(cleanup: CleanupQueue, grant: Grant) -> None:
    """Queue source release; the caller must first prove DMA has drained."""
    cleanup.enqueue(
        cleanup.create(
            grant.endpoint,
            grant.epoch,
            grant.reader,
            grant.sequence,
            lease_ms=int((grant.lease_s or 0) * 1000),
        )
    )


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
    """Issue scheduler-thread RPCs for one owner epoch and reader identity.

    Calls are synchronous and bounded by `timeout_ms`; sequence numbers make
    retries and queued cleanup refer to the same grant.
    """

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
        """Pin every key or return no grant; never authorize a partial copy.

        Busy and deferred replies raise the corresponding owner status.
        Deferred retries must reuse sequence and original acquisition time.
        """
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
        # ABI: [-3 - generation, detail]; detail -1 = queued,
        # nonnegative = busy with the advertised immediate window.
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
        """Cancel only before a copy can post; failed RPCs enqueue reliable cleanup."""
        n, _ = self._rpc(Operation.CANCEL, seq, [])
        if n < 0:
            self.cleanup.enqueue(
                self.cleanup.create(self.endpoint, self.epoch, self.reader, seq)
            )
