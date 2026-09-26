# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-owner pull transfer service, independent of CUDA and NIXL bindings.

The backend owns native objects. Messages contain only owned logical metadata.
Native failure is deliberately fatal: uncertain transfers and their pins remain
owned until process recovery, rather than reporting reusable destination blocks.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

JobKey = tuple[str, int]
TransferStatus = Literal["DONE", "PROC", "ERR"]
ReceiveStatus = Literal["done", "cancelled"]

# Unknown DMA state must not turn into off-owner __del__ calls if the worker
# stack unwinds. Production installs a process-fatal handler; embedded tests
# retain the whole owner for their process lifetime instead.
_FAILED_RECEIVERS: list[Any] = []


@dataclass(frozen=True)
class ReadJob:
    key: JobKey
    metadata: Any
    notify_only: bool = False


@dataclass(frozen=True)
class ReceiveTerminal:
    job: ReadJob
    status: ReceiveStatus


@dataclass(frozen=True)
class ReceiveRetired:
    job: ReadJob
    status: ReceiveStatus


@dataclass(frozen=True)
class NotifyOnlyTerminal:
    job: ReadJob
    status: ReceiveStatus = "done"


@dataclass(frozen=True)
class BackendEvent:
    payload: Any


ReceiverResult = ReceiveTerminal | ReceiveRetired | NotifyOnlyTerminal | BackendEvent


class ReceiverFailed(RuntimeError):
    """The owner cannot safely continue; process recovery is required."""


class PullReceiverBackend(Protocol):
    """Methods run exclusively on the receiver thread, including destruction.

    A returned handle must be retained by the backend while it is constructed.
    DONE must prove transport visibility and DMA quiescence. Successful release
    must prove native retirement. Exceptions and ERR have no such guarantee.
    ``tick`` may emit plain-data events; optional telemetry failures should be
    reported there as data, not raised as transport failures.
    """

    def initialize(self) -> None: ...
    def ready(self, job: ReadJob) -> bool: ...
    def transfers(self, job: ReadJob) -> Iterator[Any]: ...
    def post(self, handle: Any) -> TransferStatus: ...
    def poll(self, handle: Any) -> TransferStatus: ...
    def release(self, handle: Any) -> None: ...
    def notify_without_read(self, job: ReadJob) -> None: ...
    def control(self, message: Any) -> None: ...
    def tick(self, active_jobs: tuple[ReadJob, ...]) -> Iterable[Any]: ...
    def retire(self, job: ReadJob) -> None: ...
    def shutdown(self) -> None: ...


@dataclass
class _Reservation:
    job: ReadJob
    cancelled: threading.Event = field(default_factory=threading.Event)
    finalized: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Handle:
    native: Any
    released: bool = False


@dataclass
class _Receive:
    reservation: _Reservation
    transfers: Iterator[Any] | None = None
    handles: list[_Handle] = field(default_factory=list)
    started: bool = False
    submission_complete: bool = False
    terminal: bool = False
    status: ReceiveStatus = "done"


class NixlPullReceiver:
    """Bounded publication and result channels around one native owner.

    The model thread is the sole caller of submit, cancel, finalized and drain.
    Callback threads may publish_control and signal. No publication waits for
    owner progress; queue locks are never held across backend calls. Per-job
    cancellation and finalization use reserved event slots, independently of
    control mailbox capacity. Snapshot publication is single-writer.

    Reservations survive until ReceiveRetired is consumed. Lifetime history is
    bounded without eviction: exhaustion is visible fatal containment, since
    evicting old keys would permit a duplicate to acknowledge recycled blocks.
    """

    def __init__(
        self,
        backend: PullReceiverBackend,
        *,
        max_receives: int = 64,
        max_notify_only: int = 128,
        max_controls: int = 256,
        max_history: int = 100_000,
        poll_interval: float = 0.001,
        idle_interval: float = 0.01,
        poll_budget: int = 32,
        fatal_handler: Callable[[BaseException], None] | None = None,
    ) -> None:
        if (
            min(max_receives, max_notify_only, max_controls, max_history, poll_budget)
            < 1
        ):
            raise ValueError("Receiver capacities must be positive")
        if min(poll_interval, idle_interval) <= 0:
            raise ValueError("Receiver intervals must be positive")
        self.backend = backend
        self._max_receives = max_receives
        self._max_notify_only = max_notify_only
        self._max_history = max_history
        self._poll_interval = poll_interval
        self._idle_interval = idle_interval
        self._poll_budget = poll_budget
        self._jobs: queue.Queue[_Reservation] = queue.Queue(
            max_receives + max_notify_only
        )
        self._controls: queue.Queue[Any] = queue.Queue(max_controls)
        # Each receive reserves both acknowledgment stages; each notification
        # reserves one. Backend events use a separately accounted pool.
        self._results: queue.Queue[ReceiverResult] = queue.Queue(
            2 * max_receives + max_notify_only + max_controls
        )
        self._event_slots = threading.BoundedSemaphore(max_controls)
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._stopped = threading.Event()
        self._initialized = threading.Event()
        self._results_available = threading.Event()
        self._fatal = threading.Event()
        self._failure: BaseException | None = None
        self._fatal_handler = fatal_handler
        self._fatal_handler_error: BaseException | None = None
        self._failure_lock = threading.Lock()
        self._thread = threading.Thread(
            target=self._run, name="nixl-pull-receiver", daemon=True
        )
        # Model-owned ledger. The owner receives reservations by mailbox and
        # never mutates this dictionary or its counters.
        self._reservations: dict[JobKey, _Reservation] = {}
        self._seen: set[JobKey] = set()
        self._early_cancel: set[JobKey] = set()
        self._terminal_observed: set[JobKey] = set()
        self._receive_count = 0
        self._notify_count = 0
        self._snapshot: tuple[int, Any] | None = None
        self._snapshot_applied = -1
        # Owner state deliberately stays attached to self on fatal exit.
        self._active: dict[JobKey, _Receive] = {}
        self._submit_order: deque[JobKey] = deque()
        self._inflight: deque[tuple[_Receive, _Handle]] = deque()

    def start(self) -> None:
        self._thread.start()

    def signal(self) -> None:
        """Wake the owner after a callback publishes backend-ready state."""
        self._wake.set()

    @property
    def initialized(self) -> threading.Event:
        return self._initialized

    @property
    def stopped(self) -> threading.Event:
        return self._stopped

    @property
    def results_available(self) -> threading.Event:
        return self._results_available

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    def check_health(self) -> None:
        if self._fatal.is_set():
            raise ReceiverFailed("Background NIXL receiver failed") from self._failure

    def _fail(self, error: BaseException) -> None:
        # This lock protects only a sticky latch, never native work.
        first_failure = False
        with self._failure_lock:
            if not self._fatal.is_set():
                self._failure = error
                _FAILED_RECEIVERS.append(self)
                self._fatal.set()
                first_failure = True
        self._wake.set()
        self._results_available.set()
        if first_failure and self._fatal_handler is not None:
            try:
                self._fatal_handler(error)
            except BaseException as handler_error:
                # A broken diagnostic handler must not undo quarantine or
                # replace the native failure that required containment.
                self._fatal_handler_error = handler_error

    def _remember(self, key: JobKey) -> bool:
        if key in self._seen or key in self._early_cancel:
            return True
        if len(self._seen) + len(self._early_cancel) >= self._max_history:
            self._fail(ReceiverFailed("Receiver lifetime identity ledger exhausted"))
            return False
        return True

    def submit(self, job: ReadJob) -> bool:
        self.check_health()
        if self._stop.is_set():
            self._fail(ReceiverFailed("Receive published after shutdown"))
            return False
        if job.key in self._seen:
            return False
        if not self._remember(job.key):
            return False
        count, capacity = (
            (self._notify_count, self._max_notify_only)
            if job.notify_only
            else (self._receive_count, self._max_receives)
        )
        if count >= capacity:
            self._fail(ReceiverFailed("Receiver admission reservation exhausted"))
            return False
        reservation = _Reservation(job)
        if job.key in self._early_cancel:
            reservation.cancelled.set()
            self._early_cancel.remove(job.key)
        self._seen.add(job.key)
        self._reservations[job.key] = reservation
        if job.notify_only:
            self._notify_count += 1
        else:
            self._receive_count += 1
        try:
            self._jobs.put_nowait(reservation)
        except queue.Full:
            self._fail(ReceiverFailed("Reserved receive mailbox overflowed"))
            return False
        self._wake.set()
        return True

    def cancel(self, key: JobKey) -> None:
        self.check_health()
        if reservation := self._reservations.get(key):
            reservation.cancelled.set()
        elif key not in self._seen and self._remember(key):
            self._early_cancel.add(key)
        self._wake.set()

    def finalized(self, key: JobKey) -> None:
        self.check_health()
        if reservation := self._reservations.get(key):
            if key not in self._terminal_observed:
                self._fail(ReceiverFailed("Finalization before terminal consumption"))
                return
            reservation.finalized.set()
        elif key not in self._seen:
            self._fail(ReceiverFailed("Finalization for an unknown receive"))
        self._wake.set()

    def publish_control(self, message: Any) -> bool:
        self.check_health()
        # A handshake completion may be necessary to drain accepted work after
        # shutdown was requested. Keep that lane open until the owner exits.
        if self._stopped.is_set():
            self._fail(ReceiverFailed("Control published after receiver stopped"))
            return False
        try:
            self._controls.put_nowait(message)
        except queue.Full:
            self._fail(ReceiverFailed("Receiver control mailbox exhausted"))
            return False
        self._wake.set()
        return True

    def publish_snapshot(self, version: int, message: Any) -> None:
        self.check_health()
        previous = self._snapshot
        if previous is None or version > previous[0]:
            self._snapshot = (version, message)
            self._wake.set()

    def drain_results(self, limit: int | None = None) -> list[ReceiverResult]:
        self.check_health()
        self._results_available.clear()
        results: list[ReceiverResult] = []
        while limit is None or len(results) < limit:
            try:
                result = self._results.get_nowait()
            except queue.Empty:
                break
            if isinstance(result, BackendEvent):
                self._event_slots.release()
            elif isinstance(result, ReceiveTerminal):
                self._terminal_observed.add(result.job.key)
            elif isinstance(result, (ReceiveRetired, NotifyOnlyTerminal)):
                reservation = self._reservations.pop(result.job.key)
                self._terminal_observed.discard(result.job.key)
                if reservation.job.notify_only:
                    self._notify_count -= 1
                else:
                    self._receive_count -= 1
            results.append(result)
        if not self._results.empty():
            self._results_available.set()
        return results

    def shutdown(self, *, wait: bool = False, timeout: float | None = None) -> bool:
        """Request draining; an expired join never permits native destruction.

        The caller must keep consuming terminal results and acknowledging
        finalization while draining. On fatal failure, backend state is retained
        and backend.shutdown is never invoked.
        """
        self._stop.set()
        self._wake.set()
        if wait and self._thread.ident is not None:
            self._thread.join(timeout)
        return self._stopped.is_set() and not self._fatal.is_set()

    def _emit(self, result: ReceiverResult) -> None:
        try:
            self._results.put_nowait(result)
            self._results_available.set()
        except queue.Full as error:
            raise ReceiverFailed("Reserved result mailbox overflowed") from error

    def _service_control(self) -> None:
        for _ in range(self._poll_budget):
            try:
                message = self._controls.get_nowait()
            except queue.Empty:
                break
            self.backend.control(message)
        snapshot = self._snapshot
        if snapshot is not None and snapshot[0] > self._snapshot_applied:
            self.backend.control(snapshot[1])
            self._snapshot_applied = snapshot[0]

    def _accept_jobs(self) -> None:
        for _ in range(self._poll_budget):
            try:
                reservation = self._jobs.get_nowait()
            except queue.Empty:
                break
            key = reservation.job.key
            self._active[key] = _Receive(reservation)
            self._submit_order.append(key)

    def _release(self, handle: _Handle) -> None:
        self.backend.release(handle.native)
        handle.released = True
        # The final native reference must leave on its owner, not a model hook.
        handle.native = None

    def _status(self, state: _Receive, handle: _Handle, status: str) -> None:
        if status == "DONE":
            self._release(handle)
        elif status == "PROC":
            self._inflight.append((state, handle))
        else:
            raise ReceiverFailed(f"Unquiesced native transfer outcome: {status!r}")

    def _poll(self) -> None:
        for _ in range(min(self._poll_budget, len(self._inflight))):
            state, handle = self._inflight.popleft()
            self._status(state, handle, self.backend.poll(handle.native))

    def _advance_submission(self) -> bool:
        # Inspect each waiting job once, but perform at most one preparation and
        # post per loop so controls and existing transfers regain the owner.
        for _ in range(len(self._submit_order)):
            key = self._submit_order.popleft()
            state = self._active[key]
            reservation = state.reservation
            job = reservation.job
            cancelled = reservation.cancelled.is_set() or self._stop.is_set()
            if cancelled:
                state.status = "cancelled"
            if not self.backend.ready(job):
                self._submit_order.append(key)
                continue
            if job.notify_only or (cancelled and not state.started):
                self.backend.notify_without_read(job)
                state.submission_complete = True
                return True
            if state.transfers is None:
                state.transfers = iter(self.backend.transfers(job))
            state.started = True
            try:
                native = next(state.transfers)
            except StopIteration:
                state.submission_complete = True
                return True
            handle = _Handle(native)
            state.handles.append(handle)
            # Keep the acquired handle owned even if posting raises after DMA
            # submission. A failed post must never fall through to __del__.
            self._status(state, handle, self.backend.post(native))
            self._submit_order.append(key)
            return True
        return False

    def _complete(self) -> None:
        for key, state in tuple(self._active.items()):
            if not state.submission_complete or any(
                not handle.released for handle in state.handles
            ):
                continue
            reservation = state.reservation
            job = reservation.job
            if job.notify_only:
                self.backend.retire(job)
                del self._active[key]
                self._emit(NotifyOnlyTerminal(job))
            elif not state.terminal:
                if reservation.cancelled.is_set() or self._stop.is_set():
                    state.status = "cancelled"
                state.terminal = True
                self._emit(ReceiveTerminal(job, state.status))
            elif reservation.finalized.is_set():
                self.backend.retire(job)
                del self._active[key]
                self._emit(ReceiveRetired(job, state.status))

    def _run(self) -> None:
        try:
            self.backend.initialize()
            self._initialized.set()
            while not self._fatal.is_set():
                self._wake.clear()
                self._service_control()
                if self._fatal.is_set():
                    break
                self._accept_jobs()
                for event in self.backend.tick(
                    tuple(state.reservation.job for state in self._active.values())
                ):
                    if not self._event_slots.acquire(blocking=False):
                        raise ReceiverFailed(
                            "Receiver backend result capacity exhausted"
                        )
                    self._emit(BackendEvent(event))
                if self._fatal.is_set():
                    break
                self._poll()
                if self._fatal.is_set():
                    break
                advanced = self._advance_submission()
                if self._fatal.is_set():
                    break
                self._complete()
                if (
                    self._stop.is_set()
                    and not self._fatal.is_set()
                    and not self._active
                    and self._jobs.empty()
                    and self._controls.empty()
                ):
                    self.backend.shutdown()
                    return
                if advanced or not self._jobs.empty() or not self._controls.empty():
                    continue
                timeout = self._poll_interval if self._inflight else self._idle_interval
                self._wake.wait(timeout)
        except BaseException as error:
            self._fail(error)
        finally:
            self._stopped.set()
