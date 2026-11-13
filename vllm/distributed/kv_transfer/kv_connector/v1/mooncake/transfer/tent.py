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

# Neighbouring ranges travel as one request up to this length, so a batch of
# small blocks does not cost one engine task per block.
MAX_MERGED_LENGTH = 1 << 20

# How long one transfer may stay unsettled. The engine keeps the batch alive
# after the deadline, so the wait is bounded and the ranges are reported in
# flight when the engine has not stopped touching them.
DEFAULT_TRANSFER_TIMEOUT_S = 60.0

# How long `shutdown` waits for transfers that are still running.
SHUTDOWN_TIMEOUT_S = 5.0


@dataclass
class _Task:
    """One engine request covering `count` neighbouring ranges."""

    local_addr: int
    remote_addr: int
    length: int
    count: int


class TentTransferBackend(TransferBackend):
    """Moves KV blocks through TENT's batch submission and blocking wait.

    Each transfer submits one batch and then waits for it, so the backend keeps
    no batch state of its own. Transfers to one peer are serialized through a
    per-peer lock because they share that peer's segment; different peers make
    progress in parallel.
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
            raise RuntimeError("The TENT transfer timeout must not be negative.")
        self._timeout_ms = int(transfer_timeout_s * 1000)
        self._tent = _import_tent_module()
        config = build_tent_config(
            hostname=hostname,
            protocol=protocol,
            device_name=device_name,
            metrics_port=metrics_port,
            overrides=overrides,
        )
        advertised_hostname = str(config["rpc_server_hostname"])
        socket.getaddrinfo(advertised_hostname, None)
        config_path = _write_config_file(config)
        try:
            self._engine = self._tent.TransferEngine(config_path)
        finally:
            os.unlink(config_path)
        if not self._engine.available():
            raise RuntimeError("The TENT transfer engine is not available.")
        self._rpc_port = int(self._engine.get_rpc_server_port())
        expected_name = format_peer_address(advertised_hostname, self._rpc_port)
        advertised_name = self._engine.get_segment_name()
        if advertised_name != expected_name:
            raise RuntimeError(
                f"TENT advertises segment {advertised_name!r} instead of "
                f"{expected_name!r}, so peers cannot reach this worker."
            )
        self._opcodes = {
            TransferOp.WRITE: self._tent.OpCode.WRITE,
            TransferOp.READ: self._tent.OpCode.READ,
        }
        self._lock = threading.Lock()
        self._segments: dict[str, Any] = {}
        self._peer_locks: dict[str, threading.Lock] = {}
        self._closed = threading.Event()
        self._stat_calls = 0
        self._stat_ranges = 0
        self._stat_tasks = 0
        self._stat_max_ranges = 0
        self._stat_max_tasks = 0
        self._stat_bytes = 0

    @property
    def kind(self) -> TransferEngineKind:
        return TransferEngineKind.TENT

    @property
    def rpc_port(self) -> int:
        return self._rpc_port

    def register_regions(self, regions: Sequence[MemoryRegion]) -> None:
        if not regions:
            return
        try:
            self._engine.register_local_memory_batch(
                [region.base_addr for region in regions],
                [region.length for region in regions],
            )
        except Exception as error:
            raise RuntimeError(
                f"TENT failed to register KV cache memory: {error}"
            ) from error

    def unregister_regions(self, regions: Sequence[MemoryRegion]) -> None:
        if not regions:
            return
        try:
            self._engine.unregister_local_memory_batch(
                [region.base_addr for region in regions],
                [region.length for region in regions],
            )
        except Exception as error:
            raise RuntimeError(
                f"TENT failed to unregister KV cache memory: {error}"
            ) from error

    def transfer(
        self, peer: str, op: TransferOp, ranges: Sequence[RangeRequest]
    ) -> list[TransferEvent]:
        if not ranges:
            return []
        tasks = _merge_ranges(ranges)
        self._stat_calls += 1
        self._stat_ranges += len(ranges)
        self._stat_tasks += len(tasks)
        self._stat_max_ranges = max(self._stat_max_ranges, len(ranges))
        self._stat_max_tasks = max(self._stat_max_tasks, len(tasks))
        self._stat_bytes += sum(item.length for item in ranges)
        if self._stat_calls % 50 == 0:
            logger.info(
                "TENT stats: calls=%d ranges=%d tasks=%d max_ranges=%d "
                "max_tasks=%d bytes=%d",
                self._stat_calls,
                self._stat_ranges,
                self._stat_tasks,
                self._stat_max_ranges,
                self._stat_max_tasks,
                self._stat_bytes,
            )
        with self._peer_lock(peer):
            try:
                segment = self._open_segment(peer)
            except Exception as error:
                logger.debug("TENT could not open a segment to %s: %s", peer, error)
                return _events(ranges, TransferState.FAILED, str(error))
            requests = [
                self._tent.Request(
                    self._opcodes[op],
                    task.local_addr,
                    segment,
                    task.remote_addr,
                    task.length,
                )
                for task in tasks
            ]
            batch_id = 0
            try:
                batch_id = self._engine.allocate_transfer_batch(len(requests))
                if not batch_id:
                    raise RuntimeError("TENT has no free transfer batch")
                self._engine.submit_transfer(batch_id, requests)
            except Exception as error:
                logger.debug("TENT could not submit transfers to %s: %s", peer, error)
                if batch_id:
                    self._free_batch(batch_id)
                return _events(ranges, TransferState.FAILED, str(error))
            try:
                statuses = self._engine.wait_transfer(batch_id, self._timeout_ms)
            except Exception as error:
                logger.debug("TENT transfer to %s did not settle: %s", peer, error)
                self._close_segment(peer)
                return _events(ranges, TransferState.IN_FLIGHT, str(error))
            finally:
                self._free_batch(batch_id)
            if len(statuses) != len(tasks):
                raise RuntimeError(
                    f"TENT reported {len(statuses)} statuses for {len(tasks)} requests."
                )
            events: list[TransferEvent] = []
            retry = False
            for task, status in zip(tasks, statuses, strict=True):
                event = self._status_event(status)
                retry = retry or event.state in (
                    TransferState.FAILED,
                    TransferState.CANCELED,
                )
                events.extend([event] * task.count)
            if retry:
                self._close_segment(peer)
            return events

    def shutdown(self) -> None:
        self._closed.set()
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        with self._lock:
            peers = list(self._peer_locks)
        for peer in peers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            if not self._peer_lock(peer).acquire(timeout=remaining):
                logger.warning(
                    "TENT transfer to %s is still running at shutdown.", peer
                )
                continue
            try:
                self._close_segment(peer)
            finally:
                self._peer_lock(peer).release()

    def _peer_lock(self, peer: str) -> threading.Lock:
        with self._lock:
            return self._peer_locks.setdefault(peer, threading.Lock())

    def _open_segment(self, peer: str) -> Any:
        with self._lock:
            segment = self._segments.get(peer)
        if segment is not None:
            return segment
        segment = self._engine.open_segment(peer)
        with self._lock:
            self._segments[peer] = segment
        return segment

    def _close_segment(self, peer: str) -> None:
        with self._lock:
            segment = self._segments.pop(peer, None)
        if segment is None:
            return
        try:
            self._engine.close_segment(segment)
        except Exception as error:
            logger.debug("TENT failed to close the segment to %s: %s", peer, error)

    def _free_batch(self, batch_id: int) -> None:
        try:
            self._engine.free_transfer_batch(batch_id)
        except Exception as error:
            logger.debug("TENT failed to free batch %s: %s", batch_id, error)

    def _status_event(self, status: Any) -> TransferEvent:
        states = self._tent.TransferStatusEnum
        state = status.state
        if state == states.COMPLETED:
            return TransferEvent(TransferState.COMPLETED)
        if state == states.CANCELED:
            return TransferEvent(TransferState.CANCELED)
        if state in (states.INVALID, states.FAILED, states.TIMEOUT):
            return TransferEvent(TransferState.FAILED, _state_name(state))
        return TransferEvent(TransferState.IN_FLIGHT, _state_name(state))


def _merge_ranges(ranges: Sequence[RangeRequest]) -> list[_Task]:
    """Folds neighbouring ranges into as few engine requests as possible."""
    tasks: list[_Task] = []
    for request in ranges:
        task = tasks[-1] if tasks else None
        if (
            task is not None
            and task.local_addr + task.length == request.local_addr
            and task.remote_addr + task.length == request.remote_addr
            and task.length + request.length <= MAX_MERGED_LENGTH
        ):
            task.length += request.length
            task.count += 1
        else:
            tasks.append(
                _Task(
                    local_addr=request.local_addr,
                    remote_addr=request.remote_addr,
                    length=request.length,
                    count=1,
                )
            )
    return tasks


def _events(
    ranges: Sequence[RangeRequest], state: TransferState, error: str | None
) -> list[TransferEvent]:
    return [TransferEvent(state, error) for _ in ranges]


def _state_name(state: Any) -> str:
    return getattr(state, "name", str(state))


def _write_config_file(config: Mapping[str, Any]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", prefix="vllm-tent-", suffix=".json", delete=False
    ) as config_file:
        json.dump(config, config_file)
    return config_file.name


def _import_tent_module() -> Any:
    module = sys.modules.get("tent")
    if module is not None:
        return module
    if importlib.util.find_spec("tent") is not None:
        return importlib.import_module("tent")
    raise RuntimeError(
        "transfer_engine='tent' requires the 'tent' Python module, which this "
        "Mooncake build does not ship. Build Mooncake with -DUSE_TENT=ON, make "
        "the resulting 'tent' module importable, or select "
        "transfer_engine='te'."
    )
