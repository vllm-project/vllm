# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Producer-side push lifecycle, futures, and source-tensor ownership.

This module records when a push may advance or release its source.  Worker
orchestration performs the actual control exchanges and Mooncake writes.
"""

from __future__ import annotations

import threading
import time
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


class ProducerPushState(Enum):
    """Lifecycle of one Producer push from reservation to terminal state."""

    RESERVING = auto()
    WAITING_SOURCE = auto()
    WRITING = auto()
    NOTIFYING = auto()
    DONE = auto()
    CANCEL_PENDING = auto()
    CANCELLED = auto()
    FAILED = auto()


@dataclass
class ProducerSourceLease:
    """Keep an encoder tensor alive until every remote write is settled.

    Attributes:
        tensor: Source encoder tensor owned by the push.
        ready_event: CUDA event proving that production of the tensor finished.
    """

    tensor: torch.Tensor
    ready_event: torch.Event | None


@dataclass
class ProducerPushRecord:
    """Collect all asynchronous state for one Producer push.

    Attributes:
        spec: Immutable identity and destination metadata for the push.
        state: Current Producer lifecycle state.
        reservation_futures: Futures resolving Consumer shard reservations.
        reservations: Resolved destination descriptors for every shard.
        shard_futures: Data-plane futures that may still read the source.
        source: Source tensor lease once encoder computation has completed.
        batch_future: Transfer or cancellation batch currently owning the push.
        error: First asynchronous error retained for Worker reporting.
        source_at: Time the source became available for queue metrics.
    """

    spec: ECMooncakePushSpec
    state: ProducerPushState
    reservation_futures: list[Future[list[dict[str, Any]]]]
    reservations: list[dict[str, Any]] = field(default_factory=list)
    shard_futures: list[Future[Any]] = field(default_factory=list)
    source: ProducerSourceLease | None = None
    batch_future: Future[None] | None = None
    error: str | None = None
    source_at: float | None = None


_ALLOWED_TRANSITIONS = {
    ProducerPushState.RESERVING: {
        ProducerPushState.WAITING_SOURCE,
        ProducerPushState.CANCEL_PENDING,
        ProducerPushState.FAILED,
    },
    ProducerPushState.WAITING_SOURCE: {
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
    ProducerPushState.CANCEL_PENDING: {ProducerPushState.CANCELLED},
    ProducerPushState.DONE: set(),
    ProducerPushState.CANCELLED: set(),
    ProducerPushState.FAILED: set(),
}

_TERMINAL_STATES = {
    ProducerPushState.DONE,
    ProducerPushState.CANCELLED,
    ProducerPushState.FAILED,
}
_SOURCE_WAIT_STATES = {
    ProducerPushState.RESERVING,
    ProducerPushState.WAITING_SOURCE,
}
_TERMINAL_LIMIT = 1 << 16


class ProducerPushManager:
    """Own Producer push records, transitions, and source tensor leases.

    Attributes:
        _records: All active and retained terminal records by transfer ID.
        _active_ids: Non-terminal transfer IDs in insertion order.
        _reapable_terminal_ids: Terminal records safe to discard.
        _unreported_ids: Failed records awaiting Worker error reporting.
        _batch_ids: Records whose batch future has not been reaped.
        _source_waiters: Transfer IDs waiting for each cache identifier.
        _lock: Reentrant lock protecting lifecycle and ownership changes.
    """

    def __init__(self) -> None:
        self._records: OrderedDict[str, ProducerPushRecord] = OrderedDict()
        self._active_ids: OrderedDict[str, None] = OrderedDict()
        self._reapable_terminal_ids: OrderedDict[str, None] = OrderedDict()
        self._unreported_ids: OrderedDict[str, None] = OrderedDict()
        self._batch_ids: OrderedDict[str, None] = OrderedDict()
        self._source_waiters: dict[str, OrderedDict[str, None]] = {}
        self._lock = threading.RLock()

    def get(self, transfer_id: str) -> ProducerPushRecord | None:
        with self._lock:
            return self._records.get(transfer_id)

    def reserve(
        self,
        spec: ECMooncakePushSpec,
        submit: Callable[[], Future[list[dict[str, Any]]]],
    ) -> tuple[ProducerPushRecord, bool]:
        with self._lock:
            existing = self._records.get(spec.transfer_id)
            if existing is not None:
                if existing.spec != spec:
                    raise ValueError(
                        f"Producer transfer {spec.transfer_id!r} changed identity"
                    )
                return existing, False
            record = ProducerPushRecord(
                spec=spec,
                state=ProducerPushState.RESERVING,
                reservation_futures=[submit()],
            )
            self._records[spec.transfer_id] = record
            self._active_ids[spec.transfer_id] = None
            self._source_waiters.setdefault(spec.mm_hash, OrderedDict())[
                spec.transfer_id
            ] = None
            record.reservation_futures[0].add_done_callback(
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
                if record.source is not None or record.state not in _SOURCE_WAIT_STATES:
                    continue
                record.source = ProducerSourceLease(tensor, ready_event)
                record.source_at = time.monotonic()

    def submit_batches(
        self,
        executor: ThreadPoolExecutor,
        run_batch: Callable[[list[ProducerPushRecord]], None],
        on_submit: Callable[[], None],
    ) -> None:
        with self._lock:
            grouped: dict[str, list[ProducerPushRecord]] = {}
            for transfer_id in list(self._active_ids):
                record = self._records[transfer_id]
                if (
                    record.source is not None
                    and record.batch_future is None
                    and record.state in _SOURCE_WAIT_STATES
                ):
                    grouped.setdefault(record.spec.consumer_zmq, []).append(record)
            batches = list(grouped.values())
            for records in batches:
                on_submit()
                future = executor.submit(run_batch, records)
                for record in records:
                    record.batch_future = future
                    self._batch_ids[record.spec.transfer_id] = None

    def resolve_reservations(self, record: ProducerPushRecord) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        error: Exception | None = None
        for future in record.reservation_futures:
            try:
                results.extend(future.result())
            except Exception as exc:
                if error is None:
                    error = exc
        if error is not None:
            raise error
        with self._lock:
            record.reservations = results
            if record.state is ProducerPushState.RESERVING:
                self._transition(record, ProducerPushState.WAITING_SOURCE)
            return list(results)

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
                    record.state is ProducerPushState.RESERVING
                    and record.source is None
                ):
                    self._transition(record, ProducerPushState.FAILED)
            return
        with self._lock:
            if record.state is ProducerPushState.RESERVING:
                self._transition(record, ProducerPushState.WAITING_SOURCE)

    def settle_all(self, records: list[ProducerPushRecord]) -> None:
        for record in records:
            for future in record.reservation_futures:
                with suppress(Exception):
                    future.result()

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
            if record.source is None:
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

    def cancel_requests(self, request_ids: set[str]) -> list[ProducerPushRecord]:
        cancelled = []
        with self._lock:
            for transfer_id in list(self._active_ids):
                record = self._records[transfer_id]
                if (
                    record.spec.request_id not in request_ids
                    or record.source is not None
                    or record.batch_future is not None
                ):
                    continue
                if record.state not in _SOURCE_WAIT_STATES:
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
        if record.source is not None:
            record.source = None

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
            futures = [*record.reservation_futures, *record.shard_futures]
            if record.source is not None and not all(
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
        if state not in _SOURCE_WAIT_STATES:
            self._drop_source_waiter(record)
        if state in _TERMINAL_STATES:
            transfer_id = record.spec.transfer_id
            self._active_ids.pop(transfer_id, None)
            if record.error is not None:
                self._unreported_ids[transfer_id] = None
            self._mark_reapable(record)
