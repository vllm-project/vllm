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

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.engine import (
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

    tasks: list[_Task]
    results: list[TransferState | None]
    errors: list[str | None]
    lock: threading.Lock


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
        overrides: Mapping[str, Any] | None = None,
    ) -> None:
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
        self._segments: dict[str, int] = {}
        self._segment_lock = threading.Lock()
        self._batches: dict[int, _Submission] = {}
        self._batches_lock = threading.Lock()
        # Guards the closed flag together with the batch registry, so a
        # submission either lands before shutdown takes its snapshot or is
        # refused.
        self._lifecycle_lock = threading.Lock()
        self._closed = threading.Event()

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
            return _failed(ranges, str(error))
        try:
            return self._wait(batch_id)
        except _EngineFailure as error:
            logger.warning("TENT failed batch %d: %s", batch_id, error)
            self._abandon(batch_id)
            return _failed(ranges, str(error))
        except _TransferAborted as error:
            logger.warning("TENT abandoned batch %d: %s", batch_id, error)
            self._abandon(batch_id)
            return _failed(ranges, str(error))

    def _submit(self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]) -> int:
        """Registers the batch and submits it as one atomic step.

        Holding the lifecycle lock across both keeps `shutdown` from taking its
        snapshot between them, which would leave the batch running unowned.
        """
        tasks = _merge_ranges(ranges)
        with self._lifecycle_lock:
            if self._closed.is_set():
                raise _TransferAborted("the TENT transfer engine is shutting down.")
            segment = self._segment(peer)
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
            requests = [
                self._tent.Request(
                    opcode,
                    task.local_addr,
                    segment,
                    task.remote_addr,
                    task.length,
                )
                for task in tasks
            ]
            try:
                self._engine_call(
                    "submitting a transfer batch",
                    self._engine.submit_transfer,
                    batch_id,
                    requests,
                )
            except BaseException:
                self._engine.free_transfer_batch(batch_id)
                raise
            self._batches[batch_id] = _Submission(
                tasks=tasks,
                results=[None] * len(tasks),
                errors=[None] * len(tasks),
                lock=threading.Lock(),
            )
            return batch_id

    def _wait(self, batch_id: int) -> list[TransferEvent]:
        """Waits on the aggregated batch status.

        One status object covers the batch, so the per-task list is read only
        once the batch reports a terminal state.
        """
        delay = POLL_INTERVAL_MIN_S
        while True:
            overall = self._engine_call(
                "reading the batch status",
                self._engine.get_transfer_status_overall,
                batch_id,
            )
            if self._event_state(overall.state) is not None:
                events = self._poll(batch_id)
                if events:
                    return events
            if self._closed.is_set():
                raise _TransferAborted(
                    "the TENT transfer engine shut down while a transfer was in flight."
                )
            with self._batches_lock:
                known = batch_id in self._batches
            if not known:
                raise RuntimeError(
                    f"TENT dropped batch {batch_id} before its ranges settled."
                )
            time.sleep(delay)
            delay = min(delay * 2, POLL_INTERVAL_MAX_S)

    def _poll(self, batch_id: int) -> list[TransferEvent]:
        """Records what settled and returns the events once the batch is done.

        An empty list means a task is still running, so events always line up
        with the ranges the caller submitted.
        """
        with self._batches_lock:
            submission = self._batches.get(batch_id)
        if submission is None:
            return []
        with submission.lock:
            if all(result is not None for result in submission.results):
                return []
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
            for index, task in enumerate(submission.tasks):
                state = submission.results[index]
                if state is None:
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
        if not self._finish(batch_id, submission):
            return []
        return events

    def _finish(self, batch_id: int, submission: _Submission) -> bool:
        """Removes a settled batch, freeing it once.

        The submission lock stops a concurrent poll from asking the engine about
        a batch that was already freed.
        """
        with submission.lock, self._batches_lock:
            if self._batches.pop(batch_id, None) is None:
                return False
        self._engine_call(
            "freeing a transfer batch", self._engine.free_transfer_batch, batch_id
        )
        return True

    def shutdown(self) -> None:
        """Cancels what is still in flight, then closes the peer segments.

        A batch cannot be freed while its tasks run, and the KV memory behind it
        cannot be released before that.
        """
        with self._lifecycle_lock:
            self._closed.set()
            with self._batches_lock:
                in_flight = {
                    batch_id: [
                        index
                        for index, result in enumerate(submission.results)
                        if result is None
                    ]
                    for batch_id, submission in self._batches.items()
                }
        for batch_id, outstanding in in_flight.items():
            for index in outstanding:
                if not self._cancel(batch_id, index):
                    break
        try:
            try:
                remaining = self._settle(list(in_flight), SHUTDOWN_TIMEOUT_S)
            except _EngineFailure as error:
                raise RuntimeError(
                    f"TENT could not confirm that {len(in_flight)} batches settled "
                    f"({error}), so the KV memory behind them cannot be released."
                ) from error
            if remaining:
                raise RuntimeError(
                    f"TENT still holds {len(remaining)} batches {SHUTDOWN_TIMEOUT_S}s "
                    "after shutting down, so the KV memory behind them cannot be "
                    "released."
                )
        finally:
            with self._segment_lock:
                for segment in self._segments.values():
                    self._engine.close_segment(segment)
                self._segments.clear()

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

    def _abandon(self, batch_id: int) -> None:
        """Cancels what is left of a failed batch and frees it if it settles.

        Cleanup never masks the failure the caller is about to report, so a
        cleanup problem is logged and the batch stays for `shutdown` to retry.
        """
        with self._batches_lock:
            submission = self._batches.get(batch_id)
        if submission is None:
            return
        try:
            for index, result in enumerate(submission.results):
                if result is None and not self._cancel(batch_id, index):
                    break
            remaining = self._settle([batch_id], SHUTDOWN_TIMEOUT_S)
        except Exception as error:
            logger.warning("TENT cannot clean up batch %d: %s", batch_id, error)
            return
        if remaining:
            logger.warning(
                "TENT kept batch %d after a failed transfer; shutdown will retry.",
                batch_id,
            )

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

    def _segment(self, peer: str) -> int:
        """Opens the peer's segment once, or hands back the open handle."""
        segment = self._segments.get(peer)
        if segment is not None:
            return segment
        with self._segment_lock:
            segment = self._segments.get(peer)
            if segment is None:
                segment = int(
                    self._engine_call(
                        f"opening segment {peer}", self._engine.open_segment, peer
                    )
                )
                self._segments[peer] = segment
        return segment


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


def _write_config_file(config: Mapping[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="vllm-tent-", suffix=".json", delete=False
    ) as handle:
        json.dump(config, handle)
        return handle.name
