# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side reservation lifecycle and destination-memory ownership.

Reservations make remote writes idempotent and ensure cancellation or expiry
cannot free a destination while Mooncake may still be writing into it.
"""

from __future__ import annotations

import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum, auto

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    MemoryAllocation,
    ResidentLease,
)


class ConsumerReservationState(Enum):
    """Lifecycle of one Consumer destination reservation."""

    RESERVED = auto()
    WRITING = auto()
    READY = auto()
    TAKEN = auto()
    RESIDENT = auto()
    CANCEL_PENDING = auto()
    EXPIRE_PENDING = auto()
    CANCELLED = auto()
    EXPIRED = auto()


_ALLOWED_TRANSITIONS = {
    ConsumerReservationState.RESERVED: {
        ConsumerReservationState.WRITING,
        ConsumerReservationState.CANCELLED,
    },
    ConsumerReservationState.WRITING: {
        ConsumerReservationState.READY,
        ConsumerReservationState.CANCEL_PENDING,
        ConsumerReservationState.EXPIRE_PENDING,
        ConsumerReservationState.CANCELLED,
    },
    ConsumerReservationState.READY: {
        ConsumerReservationState.TAKEN,
        ConsumerReservationState.CANCELLED,
        ConsumerReservationState.EXPIRED,
    },
    ConsumerReservationState.TAKEN: {ConsumerReservationState.RESIDENT},
    ConsumerReservationState.RESIDENT: set(),
    ConsumerReservationState.CANCEL_PENDING: {ConsumerReservationState.CANCELLED},
    ConsumerReservationState.EXPIRE_PENDING: {
        ConsumerReservationState.CANCELLED,
        ConsumerReservationState.EXPIRED,
    },
    ConsumerReservationState.CANCELLED: set(),
    ConsumerReservationState.EXPIRED: set(),
}

_ACTIVE_STATES = {
    ConsumerReservationState.WRITING,
    ConsumerReservationState.READY,
    ConsumerReservationState.CANCEL_PENDING,
    ConsumerReservationState.EXPIRE_PENDING,
}
_DEFERRED_STATES = {
    ConsumerReservationState.CANCEL_PENDING,
    ConsumerReservationState.EXPIRE_PENDING,
}


@dataclass
class ConsumerReservation:
    """Own the identity and destination allocation of one remote push.

    Attributes:
        transfer_id: Cross-process identity of the transfer.
        mm_hash: Stable identifier of the encoder-cache item.
        reservation_id: Consumer-issued identity required for completion.
        state: Current destination reservation state.
        shape: Expected tensor shape.
        dtype: Unqualified expected ``torch.dtype`` name.
        allocation: Receive-slab allocation while this record owns it.
        lease: Borrowed resident allocation used for a cache hit.
        created_at: Monotonic creation time used for diagnostics.
        expires_at: Deadline for the active record or terminal tombstone.
    """

    transfer_id: str
    mm_hash: str
    reservation_id: str
    state: ConsumerReservationState
    shape: tuple[int, ...] = ()
    dtype: str = ""
    allocation: MemoryAllocation | None = None
    lease: ResidentLease[MemoryAllocation] | None = None
    created_at: float = field(default_factory=time.monotonic)
    expires_at: float = 0


@dataclass(frozen=True)
class CompletionResult:
    """Describe how a completion request affected a reservation.

    Attributes:
        accepted: Whether transfer and reservation identities matched.
        became_ready: Whether the call transitioned WRITING to READY.
        repeated: Whether the reservation was already ready.
        discarded: Whether deferred cancellation consumed the completion.
    """

    accepted: bool
    became_ready: bool = False
    repeated: bool = False
    discarded: bool = False


class CancellationOutcome(Enum):
    """Outcome categories used for control responses and metrics."""

    REJECTED = auto()
    PRE_RESERVED = auto()
    DEFERRED = auto()
    CANCELLED = auto()


class ConsumerReservationManager:
    """Own reservation transitions and destination allocation releases.

    Attributes:
        _memory: Consumer memory pool that owns destination allocations.
        _lease_ttl: Lifetime of active reservations and terminal tombstones.
        _tombstone_limit: Maximum retained terminal cancellation records.
        _records: Active and terminal records keyed by transfer ID.
        _active_ids: Transfer IDs requiring expiry scans.
        _tombstones: Terminal records retained in expiry order.
    """

    def __init__(
        self,
        memory: ConsumerMemoryPool,
        lease_ttl: float,
        tombstone_limit: int,
    ) -> None:
        self._memory = memory
        self._lease_ttl = lease_ttl
        self._tombstone_limit = tombstone_limit
        self._records: dict[str, ConsumerReservation] = {}
        self._active_ids: dict[str, None] = {}
        self._tombstones: OrderedDict[str, None] = OrderedDict()

    def get(self, transfer_id: str) -> ConsumerReservation | None:
        return self._records.get(transfer_id)

    def active_records(self) -> list[ConsumerReservation]:
        return [self._records[transfer_id] for transfer_id in self._active_ids]

    def reserve(
        self,
        transfer_id: str,
        mm_hash: str,
        nbytes: int,
        shape: tuple[int, ...],
        dtype_name: str,
        dtype: torch.dtype,
    ) -> tuple[ConsumerReservation | None, bool, bool, tuple[int, int, int]]:
        with self._memory.lock:
            existing = self._records.get(transfer_id)
            if (
                existing is not None
                and existing.state is ConsumerReservationState.CANCELLED
            ):
                return existing, False, False, (0, 0, 0)
            if (
                existing is not None
                and existing.state is ConsumerReservationState.EXPIRED
            ):
                self._remove(transfer_id)
                existing = None
            if (
                existing is not None
                and existing.state is ConsumerReservationState.EXPIRE_PENDING
            ):
                raise RuntimeError(
                    f"Transfer {transfer_id!r} still has an active writer"
                )
            if existing is not None:
                if (
                    existing.mm_hash != mm_hash
                    or existing.shape != shape
                    or existing.dtype != dtype_name
                ):
                    raise ValueError("conflicting reservation for transfer_id")
                if existing.state not in _ACTIVE_STATES:
                    raise RuntimeError(
                        f"Cannot reserve transfer {transfer_id!r} in "
                        f"{existing.state.name}"
                    )
                if existing.state is ConsumerReservationState.WRITING:
                    existing.expires_at = time.monotonic() + self._lease_ttl
                return existing, False, True, (0, 0, 0)

            lease = self._memory.acquire_cached(mm_hash, shape, dtype)
            now = time.monotonic()
            if lease is not None:
                record = ConsumerReservation(
                    transfer_id,
                    mm_hash,
                    uuid.uuid4().hex,
                    ConsumerReservationState.READY,
                    shape,
                    dtype_name,
                    lease.value,
                    lease,
                    now,
                    now + self._lease_ttl,
                )
                self._insert(record)
                return record, False, False, (0, 0, 0)
            expiry_counts = (0, 0, 0)
            allocation = self._memory.try_allocate(nbytes, shape, dtype)
            if allocation is None:
                expiry_counts = self._expire_locked(time.monotonic())
                allocation = self._memory.try_allocate(nbytes, shape, dtype)
            if allocation is None:
                allocation = self._memory.reclaim_and_allocate(nbytes, shape, dtype)
            if allocation is None:
                return None, False, False, expiry_counts
            record = ConsumerReservation(
                transfer_id,
                mm_hash,
                uuid.uuid4().hex,
                ConsumerReservationState.RESERVED,
                shape,
                dtype_name,
                allocation,
                None,
                now,
                now + self._lease_ttl,
            )
            self._transition(record, ConsumerReservationState.WRITING)
            self._insert(record)
            return record, True, False, expiry_counts

    def status(self, transfer_id: str) -> ConsumerReservation | None:
        with self._memory.lock:
            record = self._records.get(transfer_id)
            if record is None or record.state not in _ACTIVE_STATES:
                return None
            return record

    def complete(self, transfer_id: str, reservation_id: str) -> CompletionResult:
        with self._memory.lock:
            record = self._records.get(transfer_id)
            if record is None or record.reservation_id != reservation_id:
                return CompletionResult(False)
            if record.state is ConsumerReservationState.READY:
                return CompletionResult(True, repeated=True)
            if record.state in _DEFERRED_STATES:
                terminal = (
                    ConsumerReservationState.CANCELLED
                    if record.state is ConsumerReservationState.CANCEL_PENDING
                    else ConsumerReservationState.EXPIRED
                )
                self._terminate(record, terminal)
                return CompletionResult(True, discarded=True)
            if record.state is not ConsumerReservationState.WRITING:
                return CompletionResult(False)
            self._transition(record, ConsumerReservationState.READY)
            record.expires_at = time.monotonic() + self._lease_ttl
            return CompletionResult(True, became_ready=True)

    def cancel(
        self,
        transfer_id: str,
        reservation_id: str,
        abandon: bool = False,
        refresh: bool = False,
    ) -> tuple[CancellationOutcome, int]:
        with self._memory.lock:
            record = self._records.get(transfer_id)
            if (
                record is not None
                and reservation_id
                and record.reservation_id != reservation_id
            ):
                return CancellationOutcome.REJECTED, 0
            if record is None:
                now = time.monotonic()
                record = ConsumerReservation(
                    transfer_id,
                    "",
                    "",
                    ConsumerReservationState.CANCELLED,
                    created_at=now,
                )
                self._insert(record)
                self._set_tombstone_deadline(record)
                dropped = self._reap_tombstones(now)
                return CancellationOutcome.PRE_RESERVED, dropped
            if record.state is ConsumerReservationState.CANCELLED:
                self._set_tombstone_deadline(record)
                dropped = self._reap_tombstones(time.monotonic())
                return CancellationOutcome.PRE_RESERVED, dropped
            if refresh:
                if not abandon or record.state not in {
                    ConsumerReservationState.WRITING,
                    ConsumerReservationState.EXPIRE_PENDING,
                }:
                    return CancellationOutcome.REJECTED, 0
                if record.state is ConsumerReservationState.WRITING:
                    self._transition(record, ConsumerReservationState.EXPIRE_PENDING)
                self._terminate(record, ConsumerReservationState.EXPIRED)
                dropped = self._reap_tombstones(time.monotonic())
                return CancellationOutcome.CANCELLED, dropped
            if record.state in _DEFERRED_STATES and not abandon:
                return CancellationOutcome.DEFERRED, 0
            if record.state is ConsumerReservationState.WRITING and not abandon:
                self._transition(record, ConsumerReservationState.CANCEL_PENDING)
                return CancellationOutcome.DEFERRED, 0
            if (
                record.state not in _ACTIVE_STATES
                and record.state is not ConsumerReservationState.RESERVED
            ):
                return CancellationOutcome.REJECTED, 0
            self._terminate(record, ConsumerReservationState.CANCELLED)
            dropped = self._reap_tombstones(time.monotonic())
            return CancellationOutcome.CANCELLED, dropped

    def take(self, transfer_id: str, mm_hash: str) -> MemoryAllocation:
        with self._memory.lock:
            record = self._records.get(transfer_id)
            if (
                record is None
                or record.state is not ConsumerReservationState.READY
                or record.mm_hash != mm_hash
                or record.allocation is None
            ):
                raise RuntimeError(
                    f"Pushed EC tensor is not ready for mm_hash={mm_hash}"
                )
            self._transition(record, ConsumerReservationState.TAKEN)
            allocation = self._memory.publish(mm_hash, record.allocation, record.lease)
            record.allocation = None
            record.lease = None
            self._transition(record, ConsumerReservationState.RESIDENT)
            self._remove(transfer_id)
            return allocation

    def expire(self) -> tuple[int, int, int]:
        with self._memory.lock:
            return self._expire_locked(time.monotonic())

    def retire_stale(self, encoder_cache: dict[str, torch.Tensor]) -> None:
        with self._memory.lock:
            reserved_hashes = {record.mm_hash for record in self.active_records()}
            self._memory.retire_stale(encoder_cache, reserved_hashes)

    def _expire_locked(self, now: float) -> tuple[int, int, int]:
        expired = 0
        deferred = 0
        for transfer_id in list(self._active_ids):
            record = self._records[transfer_id]
            if record.expires_at > now:
                continue
            if record.state is ConsumerReservationState.READY:
                self._terminate(record, ConsumerReservationState.EXPIRED)
                expired += 1
            elif record.state is ConsumerReservationState.WRITING:
                self._transition(record, ConsumerReservationState.EXPIRE_PENDING)
                deferred += 1
        dropped = self._reap_tombstones(now)
        return expired, deferred, dropped

    def _terminate(
        self, record: ConsumerReservation, state: ConsumerReservationState
    ) -> None:
        self._transition(record, state)
        self._release(record)
        self._set_tombstone_deadline(record)

    def _release(self, record: ConsumerReservation) -> None:
        allocation = record.allocation
        if allocation is None:
            return
        if record.lease is not None:
            self._memory.release_cached(record.lease)
        else:
            self._memory.free(allocation)
        record.allocation = None
        record.lease = None

    def _set_tombstone_deadline(self, record: ConsumerReservation) -> None:
        record.expires_at = time.monotonic() + self._lease_ttl
        self._tombstones[record.transfer_id] = None
        self._tombstones.move_to_end(record.transfer_id)

    def _reap_tombstones(self, now: float) -> int:
        dropped = 0
        while self._tombstones:
            transfer_id = next(iter(self._tombstones))
            record = self._records[transfer_id]
            if (
                record.expires_at > now
                and len(self._tombstones) <= self._tombstone_limit
            ):
                break
            self._remove(transfer_id)
            dropped += 1
        return dropped

    def _insert(self, record: ConsumerReservation) -> None:
        self._records[record.transfer_id] = record
        if record.state in _ACTIVE_STATES:
            self._active_ids[record.transfer_id] = None

    def _remove(self, transfer_id: str) -> None:
        self._records.pop(transfer_id, None)
        self._active_ids.pop(transfer_id, None)
        self._tombstones.pop(transfer_id, None)

    def _transition(
        self, record: ConsumerReservation, state: ConsumerReservationState
    ) -> None:
        if state not in _ALLOWED_TRANSITIONS[record.state]:
            raise RuntimeError(
                f"Cannot transition {record.transfer_id!r} from "
                f"{record.state.name} to {state.name}"
            )
        record.state = state
        if state in _ACTIVE_STATES:
            self._active_ids[record.transfer_id] = None
        else:
            self._active_ids.pop(record.transfer_id, None)
