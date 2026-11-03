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
from vllm.logger import init_logger

logger = init_logger(__name__)


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
    """Track one transfer as observed by the Scheduler."""

    transfer_id: str
    request_id: str
    mm_hash: str
    state: SchedulerTransferState
    spec: ECMooncakeLoadSpec | None
    deadline: float | None
    notified_requests: set[str] = field(default_factory=set, repr=False)


_TERMINAL_STATES = {
    SchedulerTransferState.UNAVAILABLE,
    SchedulerTransferState.EXPIRED,
    SchedulerTransferState.FAILED,
    SchedulerTransferState.CANCELLED,
}


class SchedulerTransferTable:
    """Own Scheduler transfer state, lookup indexes, and dispatch queues.

    The Scheduler is single-threaded, so direct transitions are sufficient;
    the Producer and Consumer managers retain stricter transition matrices
    because callbacks and control requests can race there.
    """

    def __init__(self, resident_capacity: int, tombstone_ttl: float) -> None:
        self._resident_capacity = resident_capacity
        self._tombstone_ttl = tombstone_ttl
        self._records: OrderedDict[str, SchedulerTransfer] = OrderedDict()
        self._active_ids: dict[str, None] = {}
        self._terminal_ids: OrderedDict[str, None] = OrderedDict()
        self._resident_ids: OrderedDict[str, None] = OrderedDict()
        self._resident_bytes = 0
        self._hash_index: dict[str, deque[str]] = {}
        self._loads_to_dispatch: OrderedDict[str, None] = OrderedDict()
        self._unavailable_requests: set[str] = set()
        # Records expired straight out of READY/RESIDENT: the consumer worker
        # still holds their push reservation, and nothing else tells it to let
        # go, so the destination buffer would sit pinned until the lease TTL.
        self._orphaned: list[str] = []

    def get(self, transfer_id: str) -> SchedulerTransfer | None:
        return self._records.get(transfer_id)

    def drain_orphaned(self) -> list[str]:
        """Transfer IDs expired out of READY/RESIDENT since the last drain.

        The caller must release the consumer worker's reservation for each:
        `cancel()` still accepts an EXPIRED record, so the usual cancel path
        applies.
        """
        orphaned = self._orphaned
        self._orphaned = []
        return orphaned

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
        states = tuple(states)
        if len(states) == 1:
            wanted_state = states[0]
            for transfer_id in self._hash_index.get(mm_hash, ()):
                record = self._records.get(transfer_id)
                if record is not None and record.state is wanted_state:
                    return record
            return None
        wanted_states = set(states)
        for transfer_id in self._hash_index.get(mm_hash, ()):
            record = self._records.get(transfer_id)
            if record is not None and record.state in wanted_states:
                return record
        return None

    def has_state(self, mm_hash: str, states: Iterable[SchedulerTransferState]) -> bool:
        return self.first_for_hash(mm_hash, states) is not None

    def wait_for_event(
        self,
        transfer_id: str,
        request_id: str,
        mm_hash: str,
        deadline: float,
    ) -> SchedulerTransfer | None:
        """Track a request waiting for a push.

        Returns None when `transfer_id` already names a different encoding.
        The id comes from the request, so a collision has to fail that one
        request; re-issuing it re-runs the encode under a fresh id.
        """
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
        elif not self._identity_matches(record, mm_hash):
            self._refuse_colliding_id(record, mm_hash, request_id)
            return None
        else:
            if not record.request_id:
                record.request_id = request_id
            if record.state in _TERMINAL_STATES and request_id:
                self._notify_unavailable(record, request_id)
        return record

    def observe_ready(
        self, spec: ECMooncakeLoadSpec, deadline: float
    ) -> tuple[SchedulerTransfer, bool]:
        transfer_id = spec.transfer_id
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
        elif not self._identity_matches(record, spec.mm_hash):
            self._refuse_colliding_id(record, spec.mm_hash, record.request_id)
            return record, False
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
        transfer_id: str | None = None,
        request_id: str = "",
        deadline: float | None = None,
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
        # A dispatched load that never reports back would otherwise hold this
        # hash in LOADING for good, deferring every later request for it.
        record.deadline = deadline
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

    def fail_load(self, mm_hash: str, now: float) -> bool:
        record = self.first_for_hash(mm_hash, (SchedulerTransferState.LOADING,))
        if record is None:
            return self.has_state(mm_hash, (SchedulerTransferState.FAILED,))
        self._transition(record, SchedulerTransferState.FAILED, now=now)
        return True

    def release_ready(self, mm_hash: str, now: float) -> None:
        ready = self.records_for_hash(mm_hash, (SchedulerTransferState.READY,))
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
                canonical.spec = replace(canonical.spec, local=True)
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

    def mark_unavailable(self, transfer_id: str, now: float) -> None:
        record = self._records[transfer_id]
        self._transition(record, SchedulerTransferState.UNAVAILABLE, now=now)
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
        if (
            record is not None
            and mm_hash
            and not self._identity_matches(record, mm_hash)
        ):
            return False
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

    def expire(self, now: float, terminal_limit: int) -> list[SchedulerTransfer]:
        if terminal_limit < 0:
            raise ValueError("terminal_limit must be non-negative")
        expired = []
        for transfer_id in list(self._active_ids):
            record = self._records[transfer_id]
            if record.deadline is None or record.deadline > now:
                continue
            if record.state is SchedulerTransferState.AVAILABLE:
                self._transition(
                    record,
                    SchedulerTransferState.EXPIRED,
                    now=now,
                )
                expired.append(record)
            elif record.state is SchedulerTransferState.LOADING:
                self._transition(
                    record,
                    SchedulerTransferState.UNAVAILABLE,
                    now=now,
                )
                if record.request_id:
                    self._notify_unavailable(record, record.request_id)
                expired.append(record)
        while self._terminal_ids:
            transfer_id = next(iter(self._terminal_ids))
            record = self._records[transfer_id]
            if (
                record.deadline is not None
                and record.deadline > now
                and len(self._terminal_ids) <= terminal_limit
            ):
                break
            self._remove(transfer_id)
        return expired

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
        self._active_ids[record.transfer_id] = None
        if record.mm_hash:
            self._hash_index.setdefault(record.mm_hash, deque()).append(
                record.transfer_id
            )

    @staticmethod
    def _identity_matches(record: SchedulerTransfer, mm_hash: str) -> bool:
        return not record.mm_hash or record.mm_hash == mm_hash

    def _refuse_colliding_id(
        self, record: SchedulerTransfer, mm_hash: str, request_id: str
    ) -> None:
        """Report a transfer id that already names another encoding.

        Transfer ids arrive on the request, so a collision is reachable from
        outside and must not reach the engine as an exception.
        """
        logger.warning(
            "EC Mooncake transfer_id=%s already names mm_hash=%s; refusing "
            "mm_hash=%s for request %s",
            record.transfer_id,
            record.mm_hash[:16],
            mm_hash[:16],
            request_id or "<unknown>",
        )
        if request_id:
            self._unavailable_requests.add(request_id)

    def _transition(
        self,
        record: SchedulerTransfer,
        state: SchedulerTransferState,
        now: float | None = None,
    ) -> None:
        if record.state is SchedulerTransferState.RESIDENT:
            self._resident_ids.pop(record.transfer_id, None)
            if record.spec is not None:
                self._resident_bytes -= record.spec.nbytes
        if state is SchedulerTransferState.RESIDENT:
            self._resident_ids[record.transfer_id] = None
            if record.spec is not None:
                self._resident_bytes += record.spec.nbytes
        if state in _TERMINAL_STATES:
            if now is None:
                raise ValueError("Terminal transition requires a timestamp")
            # Keep a bounded tombstone so late events and repeat request IDs
            # remain idempotent instead of reviving a finished transfer.
            record.deadline = now + self._tombstone_ttl
            self._active_ids.pop(record.transfer_id, None)
            self._terminal_ids[record.transfer_id] = None
            self._terminal_ids.move_to_end(record.transfer_id)
        if state is SchedulerTransferState.EXPIRED and record.state in (
            SchedulerTransferState.READY,
            SchedulerTransferState.RESIDENT,
        ):
            self._orphaned.append(record.transfer_id)
        record.state = state
        self._records.move_to_end(record.transfer_id)

    def _evict_residents(self, now: float) -> None:
        while self._resident_bytes > self._resident_capacity and self._resident_ids:
            record = self._records[next(iter(self._resident_ids))]
            self._transition(record, SchedulerTransferState.EXPIRED, now=now)

    def _remove(self, transfer_id: str) -> None:
        record = self._records.pop(transfer_id, None)
        self._active_ids.pop(transfer_id, None)
        self._terminal_ids.pop(transfer_id, None)
        if transfer_id in self._resident_ids:
            self._resident_ids.pop(transfer_id)
            if record is not None and record.spec is not None:
                self._resident_bytes -= record.spec.nbytes
        self._loads_to_dispatch.pop(transfer_id, None)
        if record is None or not record.mm_hash:
            return
        transfer_ids = self._hash_index[record.mm_hash]
        transfer_ids.remove(transfer_id)
        if not transfer_ids:
            self._hash_index.pop(record.mm_hash)
