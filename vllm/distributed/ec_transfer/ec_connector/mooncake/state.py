# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-owned lifecycle and indexes for Consumer-bound transfers."""

from __future__ import annotations

from collections import OrderedDict, deque
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from enum import Enum, auto

from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakeLoadSpec,
)


class SchedulerTransferState(Enum):
    """Lifecycle states visible to the Scheduler.

    The states cover waiting for a Consumer event, dispatching a Worker load,
    retaining a locally reusable tensor, and terminal failure conditions.
    """

    WAITING_EVENT = auto()
    AVAILABLE = auto()
    LOADING = auto()
    READY = auto()
    RESIDENT = auto()
    UNAVAILABLE = auto()
    EXPIRED = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class SchedulerTransfer:
    """Track one transfer as observed by the Scheduler.

    Attributes:
        transfer_id: Cross-process identity of the transfer.
        request_id: Request currently waiting for the transfer.
        mm_hash: Stable identifier of the encoder-cache item.
        state: Current Scheduler lifecycle state.
        spec: Load metadata once the Consumer reports the tensor ready.
        deadline: Expiry time for waiting, available, or terminal records.
        last_error: Last terminal error associated with the transfer.
        notified_requests: Requests already told that the item is unavailable.
    """

    transfer_id: str
    request_id: str
    mm_hash: str
    state: SchedulerTransferState
    spec: ECMooncakeLoadSpec | None
    deadline: float | None
    last_error: str | None = None
    notified_requests: set[str] = field(default_factory=set, repr=False)


class InvalidSchedulerTransferTransition(RuntimeError):
    """Raised when code attempts an unsupported Scheduler state transition."""

    pass


_ALLOWED_TRANSITIONS = {
    SchedulerTransferState.WAITING_EVENT: {
        SchedulerTransferState.AVAILABLE,
        SchedulerTransferState.UNAVAILABLE,
        SchedulerTransferState.CANCELLED,
    },
    SchedulerTransferState.AVAILABLE: {
        SchedulerTransferState.LOADING,
        SchedulerTransferState.EXPIRED,
        SchedulerTransferState.CANCELLED,
    },
    SchedulerTransferState.LOADING: {
        SchedulerTransferState.READY,
        SchedulerTransferState.FAILED,
        SchedulerTransferState.CANCELLED,
    },
    SchedulerTransferState.READY: {
        SchedulerTransferState.RESIDENT,
        SchedulerTransferState.EXPIRED,
    },
    SchedulerTransferState.RESIDENT: {
        SchedulerTransferState.LOADING,
        SchedulerTransferState.EXPIRED,
    },
    SchedulerTransferState.UNAVAILABLE: {SchedulerTransferState.CANCELLED},
    SchedulerTransferState.EXPIRED: {SchedulerTransferState.CANCELLED},
    SchedulerTransferState.FAILED: set(),
    SchedulerTransferState.CANCELLED: set(),
}

_TERMINAL_STATES = {
    SchedulerTransferState.UNAVAILABLE,
    SchedulerTransferState.EXPIRED,
    SchedulerTransferState.FAILED,
    SchedulerTransferState.CANCELLED,
}


class SchedulerTransferTable:
    """Own Scheduler transfer state, lookup indexes, and dispatch queues.

    Attributes:
        _resident_capacity: Maximum bytes represented by resident records.
        _tombstone_ttl: Retention time for terminal records.
        _records: Ordered transfer records keyed by transfer ID.
        _hash_index: Transfer IDs grouped by encoder-cache hash.
        _loads_to_dispatch: Ordered IDs awaiting Worker metadata emission.
        _unavailable_requests: Requests awaiting retryable failure reporting.
    """

    def __init__(self, resident_capacity: int, tombstone_ttl: float) -> None:
        self._resident_capacity = resident_capacity
        self._tombstone_ttl = tombstone_ttl
        self._records: OrderedDict[str, SchedulerTransfer] = OrderedDict()
        self._hash_index: dict[str, deque[str]] = {}
        self._loads_to_dispatch: OrderedDict[str, None] = OrderedDict()
        self._unavailable_requests: set[str] = set()

    def get(self, transfer_id: str) -> SchedulerTransfer | None:
        return self._records.get(transfer_id)

    def records_for_hash(
        self,
        mm_hash: str,
        states: Iterable[SchedulerTransferState],
    ) -> list[SchedulerTransfer]:
        wanted = set(states)
        return [
            record
            for transfer_id in self._hash_index.get(mm_hash, ())
            if (record := self._records.get(transfer_id)) is not None
            and record.state in wanted
        ]

    def first_for_hash(
        self,
        mm_hash: str,
        states: Iterable[SchedulerTransferState],
    ) -> SchedulerTransfer | None:
        return next(iter(self.records_for_hash(mm_hash, states)), None)

    def has_state(self, mm_hash: str, states: Iterable[SchedulerTransferState]) -> bool:
        return self.first_for_hash(mm_hash, states) is not None

    def count(self, state: SchedulerTransferState) -> int:
        return sum(record.state is state for record in self._records.values())

    @property
    def resident_bytes(self) -> int:
        return sum(
            record.spec.nbytes
            for record in self._records.values()
            if record.state is SchedulerTransferState.RESIDENT and record.spec
        )

    def wait_for_event(
        self,
        transfer_id: str,
        request_id: str,
        mm_hash: str,
        deadline: float,
    ) -> SchedulerTransfer:
        record = self._records.get(transfer_id)
        if record is None:
            record = SchedulerTransfer(
                transfer_id=transfer_id,
                request_id=request_id,
                mm_hash=mm_hash,
                state=SchedulerTransferState.WAITING_EVENT,
                spec=None,
                deadline=deadline,
            )
            self._insert(record)
        else:
            self._check_identity(record, mm_hash)
            if not record.request_id:
                record.request_id = request_id
            if record.state in _TERMINAL_STATES and request_id:
                self._notify_unavailable(record, request_id)
        return record

    def observe_ready(
        self, spec: ECMooncakeLoadSpec, deadline: float
    ) -> tuple[SchedulerTransfer, bool]:
        transfer_id = spec.transfer_id or spec.mm_hash
        record = self._records.get(transfer_id)
        if record is None:
            record = SchedulerTransfer(
                transfer_id=transfer_id,
                request_id="",
                mm_hash=spec.mm_hash,
                state=SchedulerTransferState.WAITING_EVENT,
                spec=None,
                deadline=None,
            )
            self._insert(record)
        else:
            self._check_identity(record, spec.mm_hash)
        if record.state is not SchedulerTransferState.WAITING_EVENT:
            return record, False
        record.spec = spec
        record.deadline = deadline
        self._transition(record, SchedulerTransferState.AVAILABLE)
        return record, True

    def touch_available(self, transfer_id: str, deadline: float) -> None:
        record = self._records.get(transfer_id)
        if record is not None and record.state is SchedulerTransferState.AVAILABLE:
            record.deadline = deadline

    def begin_load(
        self,
        mm_hash: str,
        num_token: int,
        transfer_id: str | None = None,
        request_id: str = "",
    ) -> SchedulerTransfer | None:
        record = self._records.get(transfer_id) if transfer_id else None
        if record is not None and (
            record.mm_hash != mm_hash
            or record.state is not SchedulerTransferState.AVAILABLE
        ):
            record = None
        if record is None:
            record = self.first_for_hash(
                mm_hash,
                (
                    SchedulerTransferState.AVAILABLE,
                    SchedulerTransferState.RESIDENT,
                ),
            )
        if record is None or record.spec is None:
            return None
        if not record.request_id:
            record.request_id = request_id
        record.spec = replace(record.spec, num_token=num_token)
        record.deadline = None
        self._transition(record, SchedulerTransferState.LOADING)
        self._loads_to_dispatch[record.transfer_id] = None
        return record

    def take_loads_to_dispatch(self) -> list[SchedulerTransfer]:
        records = [
            record
            for transfer_id in self._loads_to_dispatch
            if (record := self._records.get(transfer_id)) is not None
            and record.state is SchedulerTransferState.LOADING
        ]
        self._loads_to_dispatch.clear()
        return records

    def complete_load(self, mm_hash: str) -> bool:
        record = self.first_for_hash(mm_hash, (SchedulerTransferState.LOADING,))
        if record is None:
            return self.has_state(mm_hash, (SchedulerTransferState.READY,))
        self._transition(record, SchedulerTransferState.READY)
        return True

    def fail_load(self, mm_hash: str, error: str, now: float) -> bool:
        record = self.first_for_hash(mm_hash, (SchedulerTransferState.LOADING,))
        if record is None:
            return self.has_state(mm_hash, (SchedulerTransferState.FAILED,))
        self._transition(record, SchedulerTransferState.FAILED, error, now=now)
        return True

    def release_ready(self, mm_hash: str, now: float) -> None:
        ready = [
            record
            for record in self._records.values()
            if record.mm_hash == mm_hash
            and record.state is SchedulerTransferState.READY
        ]
        if ready:
            canonical = ready[-1]
            for record in self.records_for_hash(
                mm_hash,
                (SchedulerTransferState.READY, SchedulerTransferState.RESIDENT),
            ):
                if record is not canonical:
                    self._transition(record, SchedulerTransferState.EXPIRED, now=now)
            if canonical.spec is None:
                self._transition(canonical, SchedulerTransferState.EXPIRED, now=now)
            else:
                canonical.spec = replace(canonical.spec, num_token=0, local=True)
                self._transition(canonical, SchedulerTransferState.RESIDENT)
        self._evict_residents(now)

    def reclaim(self, mm_hash: str, now: float) -> None:
        for record in self.records_for_hash(
            mm_hash,
            (SchedulerTransferState.READY, SchedulerTransferState.RESIDENT),
        ):
            if record.state is SchedulerTransferState.READY:
                record.spec = None
            else:
                self._transition(record, SchedulerTransferState.EXPIRED, now=now)

    def mark_unavailable(self, transfer_id: str, error: str, now: float) -> None:
        record = self._records[transfer_id]
        self._transition(record, SchedulerTransferState.UNAVAILABLE, error, now=now)
        if record.request_id:
            self._notify_unavailable(record, record.request_id)

    def cancel(
        self,
        transfer_id: str,
        now: float,
        mm_hash: str = "",
        request_id: str = "",
    ) -> bool:
        record = self._records.get(transfer_id)
        if record is None:
            record = SchedulerTransfer(
                transfer_id=transfer_id,
                request_id=request_id,
                mm_hash=mm_hash,
                state=SchedulerTransferState.WAITING_EVENT,
                spec=None,
                deadline=None,
            )
            self._insert(record)
        if record.state in {
            SchedulerTransferState.CANCELLED,
            SchedulerTransferState.READY,
            SchedulerTransferState.RESIDENT,
            SchedulerTransferState.FAILED,
        }:
            return False
        self._transition(record, SchedulerTransferState.CANCELLED, now=now)
        self._loads_to_dispatch.pop(transfer_id, None)
        return True

    def expire(
        self, now: float, terminal_limit: int
    ) -> tuple[list[SchedulerTransfer], int]:
        if terminal_limit < 0:
            raise ValueError("terminal_limit must be non-negative")
        expired = []
        dropped = 0
        for record in list(self._records.values()):
            if record.deadline is None or record.deadline > now:
                continue
            if record.state is SchedulerTransferState.AVAILABLE:
                self._transition(
                    record,
                    SchedulerTransferState.EXPIRED,
                    "lease expired",
                    now=now,
                )
                expired.append(record)
            elif record.state in _TERMINAL_STATES:
                self._remove(record.transfer_id)
                dropped += 1
        terminal_ids = [
            record.transfer_id
            for record in self._records.values()
            if record.state in _TERMINAL_STATES
        ]
        excess = max(0, len(terminal_ids) - terminal_limit)
        for transfer_id in terminal_ids[:excess]:
            self._remove(transfer_id)
            dropped += 1
        return expired, dropped

    def take_unavailable_requests(self) -> set[str]:
        unavailable = self._unavailable_requests
        self._unavailable_requests = set()
        return unavailable

    def _notify_unavailable(self, record: SchedulerTransfer, request_id: str) -> None:
        if request_id not in record.notified_requests:
            record.notified_requests.add(request_id)
            self._unavailable_requests.add(request_id)

    def _insert(self, record: SchedulerTransfer) -> None:
        self._records[record.transfer_id] = record
        if record.mm_hash:
            self._hash_index.setdefault(record.mm_hash, deque()).append(
                record.transfer_id
            )

    @staticmethod
    def _check_identity(record: SchedulerTransfer, mm_hash: str) -> None:
        if record.mm_hash and record.mm_hash != mm_hash:
            raise ValueError(
                f"Transfer {record.transfer_id!r} changed mm_hash from "
                f"{record.mm_hash!r} to {mm_hash!r}"
            )

    def _transition(
        self,
        record: SchedulerTransfer,
        state: SchedulerTransferState,
        error: str | None = None,
        now: float | None = None,
    ) -> None:
        if state not in _ALLOWED_TRANSITIONS[record.state]:
            raise InvalidSchedulerTransferTransition(
                f"Cannot transition {record.transfer_id!r} from "
                f"{record.state.name} to {state.name}"
            )
        if state in _TERMINAL_STATES:
            if now is None:
                raise ValueError("Terminal transition requires a timestamp")
            record.deadline = now + self._tombstone_ttl
        record.state = state
        record.last_error = error
        self._records.move_to_end(record.transfer_id)

    def _evict_residents(self, now: float) -> None:
        while self.resident_bytes > self._resident_capacity:
            record = next(
                record
                for record in self._records.values()
                if record.state is SchedulerTransferState.RESIDENT
            )
            self._transition(record, SchedulerTransferState.EXPIRED, now=now)

    def _remove(self, transfer_id: str) -> None:
        record = self._records.pop(transfer_id, None)
        self._loads_to_dispatch.pop(transfer_id, None)
        if record is None or not record.mm_hash:
            return
        transfer_ids = self._hash_index[record.mm_hash]
        transfer_ids.remove(transfer_id)
        if not transfer_ids:
            self._hash_index.pop(record.mm_hash)
