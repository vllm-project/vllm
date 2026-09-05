# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-side push lifecycle, futures, and source-tensor ownership.

This module records when a push may advance or release its source.  Worker
orchestration performs the actual control exchanges and Mooncake writes.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import suppress
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import torch

from vllm.distributed.ec_transfer.ec_connector.mooncake.metadata import (
    ECMooncakePushSpec,
)


def _same_destination(
    existing: ECMooncakePushSpec, incoming: ECMooncakePushSpec
) -> bool:
    """Whether two specs describe one push, ignoring which request asked.

    The same encoding requested twice shares a transfer id legitimately; only
    a different payload or destination is a collision.
    """
    return (
        existing.mm_hash == incoming.mm_hash
        and existing.nbytes == incoming.nbytes
        and existing.shape == incoming.shape
        and existing.dtype == incoming.dtype
        and existing.consumer_zmq == incoming.consumer_zmq
    )


class ProducerPushState(Enum):
    """Lifecycle of one Producer push from reservation to terminal state."""

    # The reservation reply and source tensor may arrive in either order.
    WAITING_INPUTS = auto()
    WRITING = auto()
    NOTIFYING = auto()
    DONE = auto()
    CANCEL_PENDING = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class ProducerPushRecord:
    """Collect all asynchronous state for one Producer push."""

    spec: ECMooncakePushSpec
    state: ProducerPushState
    reservation_future: Future[list[dict[str, Any]]]
    reservations: list[dict[str, Any]] = field(default_factory=list)
    shard_futures: list[Future[Any]] = field(default_factory=list)
    source_tensor: torch.Tensor | None = None
    source_event: torch.Event | None = None
    batch_future: Future[None] | None = None
    error: str | None = None


_ALLOWED_TRANSITIONS = {
    ProducerPushState.WAITING_INPUTS: {
        ProducerPushState.WRITING,
        ProducerPushState.CANCEL_PENDING,
        ProducerPushState.FAILED,
    },
    ProducerPushState.WRITING: {
        ProducerPushState.NOTIFYING,
        ProducerPushState.FAILED,
    },
    ProducerPushState.NOTIFYING: {
        ProducerPushState.DONE,
        ProducerPushState.FAILED,
    },
    ProducerPushState.CANCEL_PENDING: {
        ProducerPushState.CANCELLED,
        ProducerPushState.FAILED,
    },
    ProducerPushState.DONE: set(),
    ProducerPushState.CANCELLED: set(),
    ProducerPushState.FAILED: set(),
}

_TERMINAL_STATES = {
    ProducerPushState.DONE,
    ProducerPushState.CANCELLED,
    ProducerPushState.FAILED,
}
_TERMINAL_LIMIT = 1 << 16


class ProducerPushManager:
    """Own Producer push records, transitions, and source tensor leases.

    Terminal records are retained for duplicate-metadata idempotency. Separate
    ordered indexes keep hot polling paths from scanning those tombstones.
    """

    def __init__(self, wake: Callable[[], None] = lambda: None) -> None:
        self._records: OrderedDict[str, ProducerPushRecord] = OrderedDict()
        self._active_ids: OrderedDict[str, None] = OrderedDict()
        self._reapable_terminal_ids: OrderedDict[str, None] = OrderedDict()
        self._unreported_ids: OrderedDict[str, None] = OrderedDict()
        self._batch_ids: OrderedDict[str, None] = OrderedDict()
        self._source_waiters: dict[str, OrderedDict[str, None]] = {}
        self._lock = threading.RLock()
        self._wake = wake

    def reserve(
        self,
        spec: ECMooncakePushSpec,
        submit: Callable[[], Future[list[dict[str, Any]]]],
    ) -> tuple[ProducerPushRecord, bool]:
        with self._lock:
            existing = self._records.get(spec.transfer_id)
            if existing is not None:
                if not _same_destination(existing.spec, spec):
                    raise ValueError(
                        f"Conflicting EC destination for transfer_id={spec.transfer_id}"
                    )
                return existing, False
            record = ProducerPushRecord(
                spec=spec,
                state=ProducerPushState.WAITING_INPUTS,
                reservation_future=submit(),
            )
            self._records[spec.transfer_id] = record
            self._active_ids[spec.transfer_id] = None
            self._source_waiters.setdefault(spec.mm_hash, OrderedDict())[
                spec.transfer_id
            ] = None
            record.reservation_future.add_done_callback(
                lambda future: self._reservation_done(record, future)
            )
            return record, True

    def bind_source(
        self,
        mm_hash: str,
        tensor: torch.Tensor,
        ready_event: torch.Event | None,
    ) -> None:
        with self._lock:
            waiters = self._source_waiters.pop(mm_hash, OrderedDict())
            for transfer_id in waiters:
                record = self._records[transfer_id]
                if (
                    record.source_tensor is not None
                    or record.state is not ProducerPushState.WAITING_INPUTS
                ):
                    continue
                record.source_tensor = tensor
                record.source_event = ready_event
        self._wake()

    def submit_batches(
        self,
        executor: ThreadPoolExecutor,
        run_batch: Callable[[list[ProducerPushRecord]], None],
        *,
        wait: bool = False,
    ) -> bool:
        pending_event = False
        with self._lock:
            grouped: dict[str, list[ProducerPushRecord]] = {}
            for transfer_id in list(self._active_ids):
                record = self._records[transfer_id]
                if (
                    record.source_tensor is not None
                    and record.batch_future is None
                    and record.state is ProducerPushState.WAITING_INPUTS
                ):
                    if not record.reservation_future.done():
                        continue
                    if record.source_event is not None:
                        if wait:
                            record.source_event.synchronize()
                        elif not record.source_event.query():
                            pending_event = True
                            continue
                    grouped.setdefault(record.spec.consumer_zmq, []).append(record)
            batches = list(grouped.values())
            for records in batches:
                future = executor.submit(run_batch, records)
                for record in records:
                    record.batch_future = future
                    self._batch_ids[record.spec.transfer_id] = None
        return pending_event

    def resolve_reservations(self, record: ProducerPushRecord) -> list[dict[str, Any]]:
        results = record.reservation_future.result()
        with self._lock:
            record.reservations = results
            return list(results)

    def finish_reservations(
        self,
        outcomes: list[tuple[ProducerPushRecord, list[dict[str, Any]] | Exception]],
    ) -> None:
        with self._lock:
            for record, result in outcomes:
                if isinstance(result, Exception):
                    record.reservation_future.set_exception(result)
                else:
                    record.reservation_future.set_result(result)

    def _reservation_done(
        self,
        record: ProducerPushRecord,
        future: Future[list[dict[str, Any]]],
    ) -> None:
        try:
            future.result()
        except Exception as exc:
            with self._lock:
                self._set_error(record, exc)
                if (
                    record.state is ProducerPushState.WAITING_INPUTS
                    and record.source_tensor is None
                ):
                    self._transition(record, ProducerPushState.FAILED)
        finally:
            self._wake()

    def settle_all(self, records: list[ProducerPushRecord]) -> None:
        for record in records:
            with suppress(Exception):
                record.reservation_future.result()

    def replace_reservations(
        self,
        record: ProducerPushRecord,
        reservations: list[dict[str, Any]],
    ) -> None:
        with self._lock:
            record.reservations = reservations

    def track_shard_futures(
        self,
        records: list[ProducerPushRecord],
        futures: list[Future[Any]],
    ) -> None:
        with self._lock:
            for record in records:
                record.shard_futures.extend(futures)

    def begin_writing(self, record: ProducerPushRecord) -> None:
        with self._lock:
            if record.source_tensor is None:
                raise RuntimeError(
                    f"Producer push {record.spec.transfer_id!r} has no source tensor"
                )
            self._transition(record, ProducerPushState.WRITING)

    def begin_notifying(self, records: list[ProducerPushRecord]) -> None:
        with self._lock:
            for record in records:
                self._transition(record, ProducerPushState.NOTIFYING)

    def complete(self, records: list[ProducerPushRecord]) -> None:
        with self._lock:
            self._check_sources_releasable(records)
            for record in records:
                self._transition(record, ProducerPushState.DONE)
                self._release_source(record)

    def fail(self, records: list[ProducerPushRecord], error: Exception) -> None:
        with self._lock:
            self._check_sources_releasable(records)
            for record in records:
                if record.state not in _TERMINAL_STATES:
                    self._set_error(record, error)
                    self._transition(record, ProducerPushState.FAILED)
                self._release_source(record)

    def cancel_requests(self, request_ids: set[str] | None) -> list[ProducerPushRecord]:
        """Cancel source-less waiters for requests, or all waiters at shutdown."""
        cancelled = []
        with self._lock:
            for transfer_id in list(self._active_ids):
                record = self._records[transfer_id]
                if (
                    (
                        request_ids is not None
                        and record.spec.request_id not in request_ids
                    )
                    or record.source_tensor is not None
                    or record.batch_future is not None
                ):
                    continue
                if record.state is not ProducerPushState.WAITING_INPUTS:
                    continue
                self._transition(record, ProducerPushState.CANCEL_PENDING)
                cancelled.append(record)
        return cancelled

    def finish_cancel(self, record: ProducerPushRecord) -> None:
        with self._lock:
            if record.state is ProducerPushState.CANCEL_PENDING:
                self._transition(record, ProducerPushState.CANCELLED)

    def submit_cancel(
        self,
        record: ProducerPushRecord,
        executor: ThreadPoolExecutor,
        run_cancel: Callable[[ProducerPushRecord], None],
    ) -> None:
        with self._lock:
            future = executor.submit(run_cancel, record)
            record.batch_future = future
            transfer_id = record.spec.transfer_id
            self._batch_ids[transfer_id] = None
            if record.state in _TERMINAL_STATES:
                self._reapable_terminal_ids.pop(transfer_id, None)

    def poll(self) -> list[tuple[str, str]]:
        failures = []
        with self._lock:
            for transfer_id in list(self._batch_ids):
                record = self._records[transfer_id]
                future = record.batch_future
                assert future is not None
                if not future.done():
                    continue
                try:
                    future.result()
                except Exception as exc:
                    self._set_error(record, exc)
                self._batch_ids.pop(transfer_id)
                self._mark_reapable(record)
            for transfer_id in list(self._unreported_ids):
                if transfer_id in self._batch_ids:
                    continue
                record = self._records[transfer_id]
                assert record.error is not None
                failures.append((record.spec.mm_hash, record.error))
                self._unreported_ids.pop(transfer_id)
                self._mark_reapable(record)
            self._reap_terminals()
        return failures

    @property
    def pending(self) -> bool:
        with self._lock:
            return bool(self._active_ids or self._batch_ids)

    def _release_source(self, record: ProducerPushRecord) -> None:
        record.source_tensor = None
        record.source_event = None

    def _set_error(self, record: ProducerPushRecord, error: BaseException) -> None:
        if record.error is None:
            record.error = str(error)
        if record.state in _TERMINAL_STATES:
            transfer_id = record.spec.transfer_id
            self._unreported_ids[transfer_id] = None
            self._reapable_terminal_ids.pop(transfer_id, None)

    def _reap_terminals(self) -> None:
        while len(self._reapable_terminal_ids) > _TERMINAL_LIMIT:
            transfer_id, _ = self._reapable_terminal_ids.popitem(last=False)
            self._records.pop(transfer_id)

    def _mark_reapable(self, record: ProducerPushRecord) -> None:
        transfer_id = record.spec.transfer_id
        if (
            record.state in _TERMINAL_STATES
            and transfer_id not in self._batch_ids
            and transfer_id not in self._unreported_ids
        ):
            self._reapable_terminal_ids[transfer_id] = None

    def _drop_source_waiter(self, record: ProducerPushRecord) -> None:
        waiters = self._source_waiters.get(record.spec.mm_hash)
        if waiters is None:
            return
        waiters.pop(record.spec.transfer_id, None)
        if not waiters:
            self._source_waiters.pop(record.spec.mm_hash)

    @staticmethod
    def _check_sources_releasable(records: list[ProducerPushRecord]) -> None:
        for record in records:
            futures = [record.reservation_future, *record.shard_futures]
            if record.source_tensor is not None and not all(
                future.done() for future in futures
            ):
                raise RuntimeError(
                    f"Producer push {record.spec.transfer_id!r} released its "
                    "source too early"
                )

    def _transition(self, record: ProducerPushRecord, state: ProducerPushState) -> None:
        if state not in _ALLOWED_TRANSITIONS[record.state]:
            raise RuntimeError(
                f"Cannot transition producer push {record.spec.transfer_id!r} from "
                f"{record.state.name} to {state.name}"
            )
        record.state = state
        if state is not ProducerPushState.WAITING_INPUTS:
            self._drop_source_waiter(record)
        if state in _TERMINAL_STATES:
            transfer_id = record.spec.transfer_id
            self._active_ids.pop(transfer_id, None)
            if record.error is not None:
                self._unreported_ids[transfer_id] = None
            self._mark_reapable(record)
