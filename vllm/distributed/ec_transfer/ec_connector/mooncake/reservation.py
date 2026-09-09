# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Consumer-side reservation lifecycle and destination-memory ownership.

Reservations make remote writes idempotent and ensure cancellation or expiry
cannot free a destination while Mooncake may still be writing into it.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, auto

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.memory import (
    ConsumerMemoryPool,
    MemoryAllocation,
    ResidentLease,
)


class ConsumerReservationState(Enum):
    """Lifecycle of one Consumer destination reservation."""

    WRITING = auto()
    READY = auto()
    CANCEL_PENDING = auto()
    EXPIRE_PENDING = auto()
    CANCELLED = auto()
    EXPIRED = auto()


_ALLOWED_TRANSITIONS = {
    ConsumerReservationState.WRITING: {
        ConsumerReservationState.READY,
        ConsumerReservationState.CANCEL_PENDING,
        ConsumerReservationState.EXPIRE_PENDING,
        ConsumerReservationState.CANCELLED,
    },
    ConsumerReservationState.READY: {
        ConsumerReservationState.CANCELLED,
        ConsumerReservationState.EXPIRED,
    },
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
_WRITER_OWNED_STATES = {
    ConsumerReservationState.WRITING,
    ConsumerReservationState.CANCEL_PENDING,
    ConsumerReservationState.EXPIRE_PENDING,
}


@dataclass
class ConsumerReservation:
    """Own the identity and destination allocation of one remote push."""

    transfer_id: str
    mm_hash: str
    reservation_id: str
    state: ConsumerReservationState
    shape: tuple[int, ...] = ()
    dtype: str = ""
    allocation: MemoryAllocation | None = None
    lease: ResidentLease[MemoryAllocation] | None = None
    expires_at: float = 0
    writer_id: str = ""


class ConsumerReservationManager:
    """Own reservation transitions and destination allocation releases.

    A remote writer owns a ``WRITING`` allocation. Cancellation and expiry are
    therefore deferred until completion, unless refresh explicitly abandons
    that writer before replacing its reservation.
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
        self._writers: dict[str, ConsumerReservation] = {}
        self._followers: dict[str, dict[str, None]] = {}
        self._ready: list[str] = []
        self._condition = threading.Condition(memory.lock)
        self._shutting_down = False

    def reserve(
        self,
        transfer_id: str,
        mm_hash: str,
        nbytes: int,
        shape: tuple[int, ...],
        dtype_name: str,
        dtype: torch.dtype,
    ) -> tuple[ConsumerReservation | None, bool]:
        with self._condition:
            if self._shutting_down:
                raise RuntimeError("Consumer reservation manager is shutting down")
            existing = self._records.get(transfer_id)
            if (
                existing is not None
                and existing.state is ConsumerReservationState.CANCELLED
            ):
                return existing, False
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
                return existing, False

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
                    now + self._lease_ttl,
                )
                self._insert(record)
                return record, False
            writer = self._writers.get(mm_hash)
            if writer is not None and writer.state is ConsumerReservationState.WRITING:
                if writer.shape != shape or writer.dtype != dtype_name:
                    raise ValueError("conflicting in-flight tensor for mm_hash")
                record = ConsumerReservation(
                    transfer_id,
                    mm_hash,
                    uuid.uuid4().hex,
                    ConsumerReservationState.WRITING,
                    shape,
                    dtype_name,
                    writer.allocation,
                    expires_at=now + self._lease_ttl,
                    writer_id=writer.transfer_id,
                )
                self._insert(record)
                self._followers.setdefault(writer.transfer_id, {})[transfer_id] = None
                return record, False
            allocation = self._memory.try_allocate(nbytes, shape, dtype)
            if allocation is None:
                self._expire_locked(time.monotonic())
                allocation = self._memory.try_allocate(nbytes, shape, dtype)
            if allocation is None:
                allocation = self._memory.reclaim_and_allocate(nbytes, shape, dtype)
            if allocation is None:
                return None, False
            record = ConsumerReservation(
                transfer_id,
                mm_hash,
                uuid.uuid4().hex,
                ConsumerReservationState.WRITING,
                shape,
                dtype_name,
                allocation,
                None,
                now + self._lease_ttl,
            )
            self._insert(record)
            self._writers[mm_hash] = record
            return record, True

    def status(self, transfer_id: str) -> ConsumerReservation | None:
        with self._memory.lock:
            record = self._records.get(transfer_id)
            if record is None or record.state not in _ACTIVE_STATES:
                return None
            return record

    def complete(self, transfer_id: str, reservation_id: str) -> tuple[bool, bool]:
        with self._condition:
            try:
                record = self._records.get(transfer_id)
                if record is None or record.reservation_id != reservation_id:
                    return False, False
                if record.state is ConsumerReservationState.READY:
                    return True, False
                if record.writer_id:
                    return False, False
                if record.state in _WRITER_OWNED_STATES:
                    self._publish_followers(record)
                if record.state in _DEFERRED_STATES:
                    terminal = (
                        ConsumerReservationState.CANCELLED
                        if record.state is ConsumerReservationState.CANCEL_PENDING
                        else ConsumerReservationState.EXPIRED
                    )
                    self._terminate(record, terminal)
                    return True, False
                if record.state is not ConsumerReservationState.WRITING:
                    return False, False
                self._transition(record, ConsumerReservationState.READY)
                record.expires_at = time.monotonic() + self._lease_ttl
                return True, True
            finally:
                self._condition.notify_all()

    def _publish_followers(self, writer: ConsumerReservation) -> None:
        if self._writers.get(writer.mm_hash) is writer:
            self._writers.pop(writer.mm_hash)
        followers = self._followers.pop(writer.transfer_id, {})
        if not followers:
            return
        assert writer.allocation is not None
        allocation = self._memory.publish(writer.mm_hash, writer.allocation, pin=False)
        for record in [writer, *(self._records[key] for key in followers)]:
            record.allocation = allocation
            record.lease = self._memory.acquire_cached(
                record.mm_hash, record.shape, allocation.tensor.dtype
            )
            assert record.lease is not None
            if record is not writer:
                record.writer_id = ""
                self._transition(record, ConsumerReservationState.READY)
                record.expires_at = time.monotonic() + self._lease_ttl
                self._ready.append(record.transfer_id)

    def drain_ready(self) -> list[str]:
        with self._memory.lock:
            ready, self._ready = self._ready, []
            return ready

    def begin_shutdown(self) -> None:
        """Stop new reservations and cancel everything without a remote writer."""
        with self._condition:
            if self._shutting_down:
                return
            self._shutting_down = True
            for transfer_id in list(self._active_ids):
                record = self._records[transfer_id]
                if record.writer_id or record.state is ConsumerReservationState.READY:
                    self._terminate(record, ConsumerReservationState.CANCELLED)
                elif record.state is ConsumerReservationState.WRITING:
                    self._defer(record, ConsumerReservationState.CANCEL_PENDING)
            self._condition.notify_all()

    def wait_for_writers(self, timeout: float) -> bool:
        """Wait until no reservation is still owned by a remote writer."""

        def writers_finished() -> bool:
            return not any(
                self._records[transfer_id].state in _WRITER_OWNED_STATES
                for transfer_id in self._active_ids
            )

        with self._condition:
            return self._condition.wait_for(writers_finished, timeout=max(0, timeout))

    def cancel(
        self,
        transfer_id: str,
        reservation_id: str,
        abandon: bool = False,
        refresh: bool = False,
    ) -> bool:
        with self._memory.lock:
            record = self._records.get(transfer_id)
            if (
                record is not None
                and reservation_id
                and record.reservation_id != reservation_id
            ):
                return False
            if record is None:
                record = ConsumerReservation(
                    transfer_id,
                    "",
                    "",
                    ConsumerReservationState.CANCELLED,
                )
                self._insert(record)
                self._set_tombstone_deadline(record)
                self._reap_tombstones(time.monotonic())
                return True
            if record.state is ConsumerReservationState.CANCELLED:
                self._set_tombstone_deadline(record)
                self._reap_tombstones(time.monotonic())
                return True
            if refresh:
                if not abandon or record.state not in {
                    ConsumerReservationState.WRITING,
                    ConsumerReservationState.EXPIRE_PENDING,
                }:
                    return False
                if record.state is ConsumerReservationState.WRITING:
                    self._transition(record, ConsumerReservationState.EXPIRE_PENDING)
                self._terminate(record, ConsumerReservationState.EXPIRED)
                self._condition.notify_all()
                self._reap_tombstones(time.monotonic())
                return True
            if record.writer_id:
                self._terminate(record, ConsumerReservationState.CANCELLED)
                return True
            if record.state in _DEFERRED_STATES and not abandon:
                return True
            if record.state is ConsumerReservationState.WRITING and not abandon:
                self._defer(record, ConsumerReservationState.CANCEL_PENDING)
                return True
            if record.state not in _ACTIVE_STATES:
                return False
            self._terminate(record, ConsumerReservationState.CANCELLED)
            if self._shutting_down:
                self._condition.notify_all()
            self._reap_tombstones(time.monotonic())
            return True

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
            allocation = self._memory.publish(mm_hash, record.allocation, record.lease)
            record.allocation = None
            record.lease = None
            self._remove(transfer_id)
            return allocation

    def expire(self) -> int:
        with self._memory.lock:
            return self._expire_locked(time.monotonic())

    def retire_stale(
        self, encoder_cache: dict[str, torch.Tensor], freed: list[str] | None = None
    ) -> None:
        with self._memory.lock:
            self._memory.retire_stale(encoder_cache, freed=freed)

    def _expire_locked(self, now: float) -> int:
        expired = 0
        for transfer_id in list(self._active_ids):
            record = self._records[transfer_id]
            if record.expires_at > now:
                continue
            if record.state is ConsumerReservationState.READY:
                self._terminate(record, ConsumerReservationState.EXPIRED)
                expired += 1
            elif record.state is ConsumerReservationState.WRITING:
                if record.writer_id:
                    self._terminate(record, ConsumerReservationState.CANCELLED)
                else:
                    self._defer(record, ConsumerReservationState.EXPIRE_PENDING)
        self._reap_tombstones(now)
        return expired

    def _defer(
        self, record: ConsumerReservation, state: ConsumerReservationState
    ) -> None:
        """Keep the destination until the writer completes or abandons it."""
        self._transition(record, state)
        record.expires_at = float("inf")

    def _terminate(
        self, record: ConsumerReservation, state: ConsumerReservationState
    ) -> None:
        self._transition(record, state)
        self._release(record)
        self._set_tombstone_deadline(record)

    def _release(self, record: ConsumerReservation) -> None:
        if record.writer_id:
            followers = self._followers.get(record.writer_id, {})
            followers.pop(record.transfer_id, None)
            record.allocation = None
            return
        if self._writers.get(record.mm_hash) is record:
            self._writers.pop(record.mm_hash)
        for transfer_id in list(self._followers.pop(record.transfer_id, {})):
            self._terminate(
                self._records[transfer_id], ConsumerReservationState.CANCELLED
            )
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
