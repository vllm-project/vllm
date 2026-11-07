# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import importlib.util
import json
import os
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.backend import (
    MemoryRegion,
    RangeRequest,
    TransferBackend,
    TransferEngineKind,
    TransferEvent,
    TransferOp,
    TransferState,
    format_peer_address,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent_config import (
    build_tent_config,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Status poll interval. TENT settles tasks on its own progress thread, so a
# waiter starts short and backs off to this cap. The cap stays at tens of
# microseconds: a scheduler sleep can overshoot it, and on a small request a
# single millisecond of overshoot costs more than the transfer itself.
POLL_INTERVAL_MIN_S = 0.00002
POLL_INTERVAL_MAX_S = 0.00005

# Neighbouring ranges travel as one task up to this length, so a request of many
# small blocks does not pay one Python object per block.
MAX_MERGED_LENGTH = 1 << 20

# How long `shutdown` waits for canceled transfers to settle.
SHUTDOWN_TIMEOUT_S = 5.0

# How long one submission may stay unsettled. A peer that disappears leaves the
# batch in flight, and the connector cannot release the KV memory behind it until
# the batch settles, so the wait is bounded. `transfer_timeout_s` overrides it
# and 0 waits forever.
DEFAULT_TRANSFER_TIMEOUT_S = 60.0

# Reused `tent.Request` objects retained between submissions. A submission that
# needs more builds them and lets them go, so one huge request does not pin
# memory for the life of the worker.
MAX_POOLED_REQUESTS = 4096


class _EngineFailure(RuntimeError):
    """A TENT call failed, so the ranges it covered have to be reported failed."""


class _TransferAborted(RuntimeError):
    """A transfer ended early without an engine verdict."""


@dataclass
class _Task:
    """One TENT task covering `count` neighbouring `RangeRequest`s."""

    local_addr: int
    remote_addr: int
    length: int
    count: int


@dataclass
class _Submission:
    """A batch and the outcome of every task in it.

    `results` keeps one slot per task so events leave in range order.
    """

    peer: str
    tasks: list[_Task]
    results: list[TransferState | None]
    errors: list[str | None]
    lock: threading.Lock


@dataclass
class _PeerSegment:
    """One peer's segment handle and the batches still using it.

    Every field is read and written under `TentTransferBackend._segment_lock`,
    so a handle cannot be closed between the lookup that returns it and the
    submission that starts writing through it.
    """

    handle: int | None = None
    users: int = 0
    stale: bool = False


class TentTransferBackend(TransferBackend):
    """Wraps TENT behind the backend interface.

    A submission keeps its batch until every task is terminal, which is also
    when the KV memory behind it can be released.
    """

    def __init__(
        self,
        *,
        hostname: str,
        protocol: str,
        device_name: str,
        metrics_port: int,
        transfer_timeout_s: float = DEFAULT_TRANSFER_TIMEOUT_S,
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
        if transfer_timeout_s < 0:
            raise RuntimeError(
                "transfer_timeout_s must be zero or a positive number of seconds, "
                f"got {transfer_timeout_s}."
            )
        self._transfer_timeout_s = transfer_timeout_s
        self._tent = _import_tent_module()
        config = build_tent_config(
            hostname=hostname,
            protocol=protocol,
            device_name=device_name,
            metrics_port=metrics_port,
            overrides=overrides,
        )
        # The engine names its segment after the configured hostname, which an
        # operator may point at an alias peers reach differently.
        advertised_hostname = str(config["rpc_server_hostname"])
        # A name that does not resolve would surface much later as an engine
        # peers cannot reach.
        socket.getaddrinfo(advertised_hostname, None)
        config_path = _write_config_file(config)
        try:
            self._engine = self._tent.TransferEngine(config_path)
        finally:
            os.unlink(config_path)
        if not self._engine.available():
            raise RuntimeError(
                f"The TENT transfer engine is not available over {protocol!r} on "
                f"{hostname!r}."
            )
        rpc_port = int(self._engine.get_rpc_server_port())
        advertised = self._engine.get_segment_name()
        expected = format_peer_address(advertised_hostname, rpc_port)
        if advertised != expected:
            raise RuntimeError(
                f"TENT advertises segment {advertised!r} while peers reach this "
                f"worker as {expected!r}; set rpc_server_hostname through "
                "tent_config."
            )
        self._rpc_port = rpc_port
        # Segment state per peer. A handle is only closed once nothing is in
        # flight over it, and a stale one stays registered until the engine
        # confirms it is closed.
        self._peers: dict[str, _PeerSegment] = {}
        self._segment_lock = threading.Lock()
        self._batches: dict[int, _Submission] = {}
        # Batches a failed submission allocated and the engine would not free.
        # They never reach the registry, so `shutdown` is the one that retries.
        self._pending_free: dict[int, str] = {}
        self._batches_lock = threading.Lock()
        # Guards the closed flag together with the batch registry, so a
        # submission either lands before shutdown takes its snapshot or is
        # refused.
        self._lifecycle_lock = threading.Lock()
        self._closed = threading.Event()
        # Building a tent.Request costs about as much as moving a 32 KiB block,
        # so submitted objects are reused once the engine has copied them.
        self._request_pool: list[Any] = []
        self._request_pool_lock = threading.Lock()

    @property
    def kind(self) -> TransferEngineKind:
        return TransferEngineKind.TENT

    @property
    def rpc_port(self) -> int:
        return self._rpc_port

    def register_regions(self, regions: Sequence[MemoryRegion]) -> None:
        if not regions:
            return
        self._engine.register_local_memory_batch(
            [region.base_addr for region in regions],
            [region.length for region in regions],
        )

    def unregister_regions(self, regions: Sequence[MemoryRegion]) -> None:
        if not regions:
            return
        self._engine.unregister_local_memory_batch(
            [region.base_addr for region in regions],
            [region.length for region in regions],
        )

    def transfer(
        self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]
    ) -> list[TransferEvent]:
        if not ranges:
            return []
        try:
            batch_id = self._submit(peer, op, ranges)
        except (_EngineFailure, _TransferAborted) as error:
            logger.warning("TENT did not start a transfer to %s: %s", peer, error)
            self._forget_segment(peer)
            return _failed(ranges, str(error))
        try:
            events = self._wait(batch_id)
        except (_EngineFailure, _TransferAborted) as error:
            logger.warning("TENT did not settle batch %d: %s", batch_id, error)
            stopped = self._abandon(batch_id)
            self._forget_segment(peer)
            if not stopped:
                # The engine may still be writing through this batch, so the
                # ranges are not failures the caller can retry over the same
                # memory; they stay reserved until shutdown confirms the stop.
                return _unresolved(ranges, str(error))
            return _failed(ranges, str(error))
        if any(event.state is not TransferState.COMPLETED for event in events):
            # The peer may have restarted, so the handle is not reused.
            self._forget_segment(peer)
        return events

    def _submit(self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]) -> int:
        """Registers the batch and submits it as one atomic step.

        Holding the lifecycle lock across both keeps `shutdown` from taking its
        snapshot between them, which would leave the batch running unowned. The
        segment use is counted before the batch exists, so a handle cannot retire
        while this submission is still being prepared; a failed submission gives
        the use back.
        """
        tasks = _merge_ranges(ranges)
        with self._lifecycle_lock:
            if self._closed.is_set():
                raise _TransferAborted("the TENT transfer engine is shutting down.")
            segment = self._acquire_segment(peer)
            try:
                return self._start_batch(peer, op, tasks, segment)
            except BaseException:
                self._release_segment_use(peer)
                raise

    def _start_batch(
        self,
        peer: str,
        op: TransferOp,
        tasks: list[_Task],
        segment: int,
    ) -> int:
        batch_id = int(
            self._engine_call(
                "allocating a transfer batch",
                self._engine.allocate_transfer_batch,
                len(tasks),
            )
        )
        if not batch_id:
            raise _EngineFailure(
                f"TENT could not allocate a batch of {len(tasks)} transfers."
            )
        opcode = (
            self._tent.OpCode.WRITE
            if op is TransferOp.WRITE
            else self._tent.OpCode.READ
        )
        requests = self._take_requests(len(tasks))
        for request, task in zip(requests, tasks, strict=True):
            request.opcode = opcode
            request.source = task.local_addr
            request.target_id = segment
            request.target_offset = task.remote_addr
            request.length = task.length
        try:
            self._engine_call(
                "submitting a transfer batch",
                self._engine.submit_transfer,
                batch_id,
                requests,
            )
        except BaseException:
            self._free_refused_batch(batch_id)
            raise
        finally:
            self._return_requests(requests)
        submission = _Submission(
            peer=peer,
            tasks=tasks,
            results=[None] * len(tasks),
            errors=[None] * len(tasks),
            lock=threading.Lock(),
        )
        with self._batches_lock:
            self._batches[batch_id] = submission
        return batch_id

    def _wait(self, batch_id: int) -> list[TransferEvent]:
        """Waits on the aggregated batch status.

        One status object covers the batch, so the per-task list is read only
        once the batch reports a terminal state. `transfer_timeout_s` bounds the
        wait: the engine retries inside a batch, so without a deadline a peer
        that disappears would hold this thread, and the KV memory behind the
        batch, for good.
        """
        deadline = (
            None
            if self._transfer_timeout_s <= 0
            else time.monotonic() + self._transfer_timeout_s
        )
        delay = POLL_INTERVAL_MIN_S
        while True:
            overall = self._engine_call(
                "reading the batch status",
                self._engine.get_transfer_status_overall,
                batch_id,
            )
            if self._event_state(overall.state) is not None:
                events = self._poll(
                    batch_id,
                    all_completed=(
                        overall.state == self._tent.TransferStatusEnum.COMPLETED
                    ),
                )
                if events:
                    return events
            if self._closed.is_set():
                raise _TransferAborted(
                    "the TENT transfer engine shut down while a transfer was in flight."
                )
            with self._batches_lock:
                known = batch_id in self._batches
            if not known:
                raise _EngineFailure(
                    f"TENT dropped batch {batch_id} before its ranges settled."
                )
            if deadline is not None and time.monotonic() > deadline:
                raise _TransferAborted(
                    f"TENT did not settle batch {batch_id} within "
                    f"{self._transfer_timeout_s}s."
                )
            time.sleep(delay)
            delay = min(delay * 2, POLL_INTERVAL_MAX_S)

    def _poll(
        self, batch_id: int, *, all_completed: bool = False
    ) -> list[TransferEvent]:
        """Records what settled and returns the events once the batch is done.

        An empty list means a task is still running, so events always line up
        with the ranges the caller submitted. `all_completed` skips the
        per-task status list, which costs a Python object per range.
        """
        with self._batches_lock:
            submission = self._batches.get(batch_id)
        if submission is None:
            return []
        events: list[TransferEvent] = []
        already_settled = False
        with submission.lock:
            if all(result is not None for result in submission.results):
                # Every task settled but the batch is still registered: an
                # earlier free failed, so retry it instead of waiting for
                # `shutdown` to be the only one that ever tries again.
                already_settled = True
            else:
                statuses = None
                if not all_completed:
                    statuses = self._engine_call(
                        "reading the per-task status",
                        self._engine.get_transfer_status_list,
                        batch_id,
                    )
                    if len(statuses) != len(submission.tasks):
                        raise _EngineFailure(
                            f"TENT returned {len(statuses)} statuses for "
                            f"{len(submission.tasks)} tasks."
                        )
                pending = False
                settled: list[TransferEvent] = []
                completed_event = TransferEvent(state=TransferState.COMPLETED)
                for index, task in enumerate(submission.tasks):
                    state = submission.results[index]
                    if state is None:
                        if statuses is None:
                            state = TransferState.COMPLETED
                            submission.results[index] = state
                            settled.extend([completed_event] * task.count)
                            continue
                        state = self._event_state(statuses[index].state)
                        if state is None:
                            pending = True
                            continue
                        submission.results[index] = state
                        if state is not TransferState.COMPLETED:
                            submission.errors[index] = (
                                f"TENT reported {statuses[index].state}"
                            )
                    event = TransferEvent(state=state, error=submission.errors[index])
                    settled.extend([event] * task.count)
                if pending:
                    return []
                events = settled
        if already_settled:
            self._retry_free(batch_id, submission)
            return []
        # The ranges have their outcome either way; a batch the engine will not
        # free stays registered for `shutdown` to retry.
        self._retry_free(batch_id, submission)
        return events

    def _free_refused_batch(self, batch_id: int) -> None:
        """Frees a batch the engine refused to accept.

        The submission raised, so this batch never reaches the registry. When
        the engine will not free it either, it is remembered for `shutdown`:
        losing the id here would leak the native batch for good.
        """
        try:
            self._engine_call(
                "freeing a transfer batch", self._engine.free_transfer_batch, batch_id
            )
        except _EngineFailure as error:
            logger.warning(
                "TENT cannot free refused batch %d (%s); shutdown will retry.",
                batch_id,
                error,
            )
            with self._batches_lock:
                self._pending_free[batch_id] = str(error)

    def _retry_free(self, batch_id: int, submission: _Submission) -> bool:
        """Frees a settled batch, logging rather than hiding a problem.

        The ranges already have their outcome, so a batch the engine will not
        free is a leak that `shutdown` retries; it is not a reason to turn a
        finished transfer into a failure.
        """
        try:
            return self._finish(batch_id, submission)
        except _EngineFailure as error:
            logger.warning(
                "TENT cannot free batch %d (%s); shutdown will retry.",
                batch_id,
                error,
            )
            return False

    def _finish(self, batch_id: int, submission: _Submission) -> bool:
        """Frees a settled batch, then forgets it.

        The engine frees first: a batch that stays in the registry can be freed
        by `shutdown`, one that was dropped cannot. The submission lock stops a
        concurrent poll from asking the engine about a batch that was already
        freed.
        """
        with submission.lock, self._batches_lock:
            if batch_id not in self._batches:
                return False
            self._engine_call(
                "freeing a transfer batch", self._engine.free_transfer_batch, batch_id
            )
            self._batches.pop(batch_id, None)
        self._release_segment_use(submission.peer)
        return True

    def shutdown(self) -> None:
        """Cancels what is still in flight, then closes the peer segments.

        A batch cannot be freed while its tasks run, and the KV memory behind it
        cannot be released before that. When a batch is still running after the
        drain, its segment stays open too: closing the handle under an unfinished
        transfer would leave the engine writing through a segment it no longer
        owns.
        """
        with self._lifecycle_lock:
            self._closed.set()
            with self._batches_lock:
                outstanding = {
                    batch_id: [
                        index
                        for index, result in enumerate(submission.results)
                        if result is None
                    ]
                    for batch_id, submission in self._batches.items()
                }
        for batch_id, indices in outstanding.items():
            for index in indices:
                if not self._cancel(batch_id, index):
                    break
        # A refused batch never reaches the registry, so nothing else retries it.
        refused = self._free_refused_batches()
        try:
            # Settled batches stay in the registry when the engine refused to
            # free them, so the drain has to cover every registered batch, not
            # only the ones that were still running.
            remaining = self._settle(list(outstanding), SHUTDOWN_TIMEOUT_S)
        except _EngineFailure as error:
            raise RuntimeError(
                f"TENT could not confirm that {len(outstanding)} batches settled "
                f"({error}), so the KV memory behind them, and the segments that "
                "reach it, stay open."
            ) from error
        if remaining:
            raise RuntimeError(
                f"TENT still holds {len(remaining)} batches {SHUTDOWN_TIMEOUT_S}s "
                "after shutting down, so the KV memory behind them, and the "
                "segments that reach it, stay open."
            )
        with self._segment_lock:
            in_use = sorted(peer for peer, state in self._peers.items() if state.users)
            handles = {
                peer: state.handle
                for peer, state in self._peers.items()
                if state.handle is not None
            }
        if in_use:
            raise RuntimeError(
                f"TENT still counts batches on the segments for {in_use}, so they "
                "stay open."
            )
        failures: list[str] = []
        for peer, segment in handles.items():
            with self._segment_lock:
                state = self._peers.get(peer)
                if state is not None and state.handle == segment:
                    state.handle = None
            if self._close_segment(peer, segment):
                with self._segment_lock:
                    state = self._peers.get(peer)
                    if state is not None and state.handle is None:
                        state.stale = False
                        self._peers.pop(peer, None)
                continue
            # The handle stays registered so a later shutdown can retry it.
            with self._segment_lock:
                state = self._peers.setdefault(peer, _PeerSegment())
                state.handle = segment
            failures.append(peer)
        if failures:
            raise RuntimeError(
                f"TENT could not close the segments for {sorted(failures)}, so "
                "their handles stay registered for the next shutdown."
            )
        if refused:
            raise RuntimeError(
                f"TENT still holds {len(refused)} batches it refused to submit, so "
                "the memory behind them stays reserved."
            )

    def _free_refused_batches(self) -> list[int]:
        """Retries the batches a failed submission left behind.

        Returns the ones the engine still will not free, so `shutdown` can
        report them instead of leaking them silently.
        """
        with self._batches_lock:
            pending = dict(self._pending_free)
        still_refused: list[int] = []
        for batch_id, reason in pending.items():
            try:
                self._engine_call(
                    "freeing a refused transfer batch",
                    self._engine.free_transfer_batch,
                    batch_id,
                )
            except _EngineFailure as error:
                logger.warning(
                    "TENT still cannot free refused batch %d (%s after %s).",
                    batch_id,
                    error,
                    reason,
                )
                still_refused.append(batch_id)
                continue
            with self._batches_lock:
                self._pending_free.pop(batch_id, None)
        return still_refused

    def _cancel(self, batch_id: int, index: int) -> bool:
        """Cancels one task.

        Returns False when TENT refuses, leaving the batch to the drain deadline.
        """
        unsupported = getattr(self._tent, "NotImplementedError", None)
        try:
            self._engine.cancel_transfer(batch_id, index)
        except Exception as error:
            if unsupported is None or not isinstance(error, unsupported):
                raise
            logger.warning(
                "TENT cannot cancel task %d of batch %d: %s", index, batch_id, error
            )
            return False
        return True

    def _settle(self, batch_ids: Sequence[int], timeout_s: float) -> list[int]:
        """Polls the given batches until they leave the registry.

        Returns the batches still registered when the deadline passed, so the
        caller decides whether that is a warning or a failure.
        """
        deadline = time.monotonic() + timeout_s
        pending = list(batch_ids)
        while pending:
            for batch_id in pending:
                self._poll(batch_id)
            with self._batches_lock:
                pending = [
                    batch_id for batch_id in pending if batch_id in self._batches
                ]
            if not pending or time.monotonic() > deadline:
                return pending
            time.sleep(POLL_INTERVAL_MAX_S)
        return pending

    def _abandon(self, batch_id: int) -> bool:
        """Cancels what is left of a failed batch and frees it if it settles.

        Returns whether the batch is gone, which tells the caller whether the
        memory behind it can be handed back. Cleanup never masks the failure the
        caller is about to report, so a cleanup problem is logged and the batch
        stays for `shutdown` to retry.
        """
        with self._batches_lock:
            submission = self._batches.get(batch_id)
        if submission is None:
            return True
        try:
            for index, result in enumerate(submission.results):
                if result is None and not self._cancel(batch_id, index):
                    break
            remaining = self._settle([batch_id], SHUTDOWN_TIMEOUT_S)
        except Exception as error:
            logger.warning("TENT cannot clean up batch %d: %s", batch_id, error)
            return False
        if remaining:
            logger.warning(
                "TENT kept batch %d after a failed transfer; shutdown will retry.",
                batch_id,
            )
            return False
        return True

    def _engine_call(self, action: str, call: Any, *args: Any) -> Any:
        """Runs one engine call, tagging its failure as an engine fault."""
        try:
            return call(*args)
        except Exception as error:
            raise _EngineFailure(f"TENT failed at {action}: {error}") from error

    def _event_state(self, state: Any) -> TransferState | None:
        status = self._tent.TransferStatusEnum
        if state == status.COMPLETED:
            return TransferState.COMPLETED
        if state == status.CANCELED:
            return TransferState.CANCELED
        if state in (status.FAILED, status.TIMEOUT, status.INVALID):
            return TransferState.FAILED
        return None

    def _acquire_segment(self, peer: str) -> int:
        """Returns the peer's handle with one use counted against it.

        The count is taken under the same lock that guards closing, so a failure
        reported by another batch cannot retire this handle between the lookup
        and the submission that is about to write through it.
        """
        with self._segment_lock:
            state = self._peers.setdefault(peer, _PeerSegment())
            if not state.stale:
                if state.handle is None:
                    state.handle = self._open_segment(peer)
                state.users += 1
                return state.handle
            if not state.users:
                handle = self._reopen_segment(peer, state)
                state.users += 1
                return handle
            users = state.users
        # Waiting here would hold the caller's lifecycle lock and stall
        # submissions to other peers, so the request fails and the caller
        # retries or recomputes.
        raise _EngineFailure(
            f"the segment for {peer} is stale and {users} batches still use it."
        )

    def _open_segment(self, peer: str) -> int:
        """Opens a handle for a peer. The caller holds the segment lock."""
        return int(
            self._engine_call(
                f"opening segment {peer}", self._engine.open_segment, peer
            )
        )

    def _reopen_segment(self, peer: str, state: _PeerSegment) -> int:
        """Replaces a stale handle with a fresh one. Caller holds the lock."""
        if state.handle is not None:
            if not self._close_segment(peer, state.handle):
                raise _EngineFailure(
                    f"the segment for {peer} is stale and the engine did not close it."
                )
            state.handle = None
        state.stale = False
        state.handle = self._open_segment(peer)
        return state.handle

    def _forget_segment(self, peer: str) -> None:
        """Marks a peer's handle stale after a failure.

        A peer that restarted keeps its address, so a cached handle would point
        at a segment that no longer exists. The handle is closed only once the
        batches still running over it finish.
        """
        with self._segment_lock:
            state = self._peers.setdefault(peer, _PeerSegment())
            state.stale = True
            if not state.users:
                self._retire_segment(peer, state)

    def _release_segment_use(self, peer: str) -> None:
        """Counts one batch out, closing a handle whose peer was marked stale."""
        with self._segment_lock:
            state = self._peers.get(peer)
            if state is None:
                return
            if state.users > 0:
                state.users -= 1
            if state.users or not state.stale:
                return
            self._retire_segment(peer, state)

    def _retire_segment(self, peer: str, state: _PeerSegment) -> None:
        """Drops a stale handle nothing runs over. Caller holds the lock."""
        if state.handle is not None:
            if not self._close_segment(peer, state.handle):
                # The handle stays registered so the next shutdown retries it.
                return
            state.handle = None
        state.stale = False
        self._peers.pop(peer, None)

    def _close_segment(self, peer: str, segment: int) -> bool:
        """Closes one handle, reporting whether the engine let it go."""
        try:
            self._engine.close_segment(segment)
        except Exception as error:
            logger.warning(
                "TENT could not close the segment for %s: %s; the handle stays "
                "registered for the next shutdown.",
                peer,
                error,
            )
            return False
        return True

    def _take_requests(self, count: int) -> list[Any]:
        """Hands back `count` request objects, reusing pooled ones first."""
        reused: list[Any] = []
        with self._request_pool_lock:
            if count <= len(self._request_pool):
                reused = self._request_pool[-count:]
                del self._request_pool[-count:]
        missing = count - len(reused)
        if not missing:
            return reused
        fresh = [
            self._tent.Request(self._tent.OpCode.WRITE, 0, 0, 0, 0)
            for _ in range(missing)
        ]
        return reused + fresh

    def _return_requests(self, requests: Sequence[Any]) -> None:
        """Keeps submitted objects for the next submission.

        The engine copies every request into its own batch before this runs, so
        mutating them afterwards is safe.
        """
        if not requests:
            return
        with self._request_pool_lock:
            room = MAX_POOLED_REQUESTS - len(self._request_pool)
            if room > 0:
                self._request_pool.extend(requests[:room])


def _merge_ranges(ranges: Sequence[RangeRequest]) -> list[_Task]:
    """Folds contiguous ranges into one task.

    TENT builds one object per task and reports one status per task.
    """
    tasks: list[_Task] = []
    for request in ranges:
        if tasks:
            last = tasks[-1]
            if (
                last.length + request.length <= MAX_MERGED_LENGTH
                and last.local_addr + last.length == request.local_addr
                and last.remote_addr + last.length == request.remote_addr
            ):
                last.length += request.length
                last.count += 1
                continue
        tasks.append(
            _Task(
                local_addr=request.local_addr,
                remote_addr=request.remote_addr,
                length=request.length,
                count=1,
            )
        )
    return tasks


def _import_tent_module() -> Any:
    # An already imported module is returned as is, so callers that loaded the
    # extension themselves keep working.
    module = sys.modules.get("tent")
    if module is not None:
        return module
    if importlib.util.find_spec("tent") is None:
        raise RuntimeError(
            "transfer_engine='tent' requires the 'tent' Python module, which this "
            "Mooncake build does not ship. Build Mooncake with -DUSE_TENT=ON, make "
            "the resulting 'tent' module importable, or select "
            "transfer_engine='te'."
        )
    return importlib.import_module("tent")


def _failed(ranges: Sequence[RangeRequest], error: str) -> list[TransferEvent]:
    """Reports every range of an aborted request as failed."""
    return [TransferEvent(state=TransferState.FAILED, error=error) for _ in ranges]


def _unresolved(ranges: Sequence[RangeRequest], error: str) -> list[TransferEvent]:
    """Reports every range as still possibly being written.

    Used when a batch could not be stopped: the memory behind it must stay
    reserved, so the ranges cannot be reported as failed.
    """
    return [TransferEvent(state=TransferState.IN_FLIGHT, error=error) for _ in ranges]


def _write_config_file(config: Mapping[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="vllm-tent-", suffix=".json", delete=False
    ) as handle:
        json.dump(config, handle)
        return handle.name
