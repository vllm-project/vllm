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
from typing import Any

import numpy as np
from numpy.typing import NDArray

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.backend import (
    BatchTransferResult,
    MemoryRegion,
    RangeRequest,
    TransferBackend,
    TransferEvent,
    TransferOp,
    TransferState,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.transfer.tent_config import (
    build_tent_config,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# How long one transfer may stay unsettled. The engine keeps the batch alive
# after the deadline, so the wait is bounded and the ranges are reported in
# flight when the engine has not stopped touching them.
DEFAULT_TRANSFER_TIMEOUT_S = 60.0

# Both errors below report a missing TENT installation, so they share the fix.
_TENT_BUILD_HINT = (
    "Build Mooncake v0.3.14 or later from source with -DUSE_TENT=ON; the "
    "PyPI wheels ship no top-level 'tent' module."
)

# How long `shutdown` waits for transfers that are still running.
SHUTDOWN_TIMEOUT_S = 5.0

POLL_INTERVAL_MIN_S = 0.00002
POLL_INTERVAL_MAX_S = 0.00005


class TentTransferBackend(TransferBackend):
    """Moves KV blocks through TENT's batch submission and blocking wait.

    Each transfer submits one batch and then waits for it, so the backend keeps
    no batch state of its own. The segment to one peer is opened once and shared
    by every transfer to that peer.
    """

    def __init__(
        self,
        *,
        hostname: str,
        protocol: str,
        device_name: str,
        metrics_port: int = 0,
        transfer_timeout_s: float = DEFAULT_TRANSFER_TIMEOUT_S,
        enable_sleep_mode: bool = False,
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
            enable_sleep_mode=enable_sleep_mode,
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
        expected_name = _format_peer_address(advertised_hostname, self._rpc_port)
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
        self._state_names = {
            int(member): member.name
            for member in self._tent.TransferStatusEnum.__members__.values()
        }
        self._completed = int(self._tent.TransferStatusEnum.COMPLETED)
        missing = [
            name
            for name in (
                "allocate_transfer_batch",
                "submit_transfer",
                "get_transfer_status_overall",
                "get_transfer_status_list",
                "free_transfer_batch",
            )
            if not callable(getattr(self._engine, name, None))
        ]
        if missing:
            raise RuntimeError(
                "This Mooncake build does not expose "
                f"{', '.join(missing)}, which the TENT backend needs. "
                f"{_TENT_BUILD_HINT}"
            )
        self._pending = int(self._tent.TransferStatusEnum.PENDING)
        self._timeout_s = float(transfer_timeout_s)
        # Newer builds submit and wait in one call, which keeps the Python side
        # out of the per-range loop below.
        self._sync_arrays = getattr(self._engine, "transfer_sync_arrays", None)
        self._sync_arrays_timed = bool(
            self._sync_arrays is not None
            and "timeout_ms" in (getattr(self._sync_arrays, "__doc__", "") or "")
        )
        self._wait_transfer = getattr(self._engine, "wait_transfer", None)
        self._segment_lock = threading.Lock()
        self._segments: dict[str, Any] = {}
        self._open_locks: dict[str, threading.Lock] = {}
        self._drain = threading.Condition()
        self._closing = False
        self._inflight = 0

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
            # Addresses only: NVLink rewrites sub-allocated buffers to the
            # enclosing cudaMalloc until kvcache-ai/Mooncake#5886, so the
            # per-region sizes would not match the registered entries.
            self._engine.unregister_local_memory_batch(
                [region.base_addr for region in regions]
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
        local_addrs = np.fromiter(
            (item.local_addr for item in ranges), dtype=np.uint64, count=len(ranges)
        )
        remote_addrs = np.fromiter(
            (item.remote_addr for item in ranges), dtype=np.uint64, count=len(ranges)
        )
        lengths = np.fromiter(
            (item.length for item in ranges), dtype=np.uint64, count=len(ranges)
        )
        states = self._submit(peer, op, local_addrs, remote_addrs, lengths)
        events: list[TransferEvent] = []
        for index in range(len(ranges)):
            code = int(states[index])
            if code == self._completed:
                events.append(TransferEvent(TransferState.COMPLETED))
            else:
                events.append(
                    TransferEvent(
                        TransferState.FAILED,
                        f"TENT range {index} ended in {self._state_name(code)}",
                    )
                )
        return events

    def transfer_sync_arrays(
        self,
        peer: str,
        op: TransferOp,
        local_addrs: NDArray[np.uint64],
        remote_addrs: NDArray[np.uint64],
        lengths: NDArray[np.uint64],
    ) -> BatchTransferResult:
        num_ranges = int(local_addrs.size)
        if num_ranges == 0:
            return BatchTransferResult(num_ranges=0, state=TransferState.COMPLETED)
        if remote_addrs.size != num_ranges or lengths.size != num_ranges:
            raise RuntimeError(
                "TENT needs one local address, one remote address and one length "
                f"per range, got {num_ranges}, {remote_addrs.size} and "
                f"{lengths.size}."
            )
        states = self._submit(peer, op, local_addrs, remote_addrs, lengths)
        if int(states.size) != num_ranges:
            return BatchTransferResult(
                num_ranges=num_ranges,
                state=TransferState.FAILED,
                error=(
                    f"TENT reported {int(states.size)} terminal states for "
                    f"{num_ranges} ranges"
                ),
            )
        failed = np.flatnonzero(states != self._completed)
        if failed.size == 0:
            return BatchTransferResult(
                num_ranges=num_ranges, state=TransferState.COMPLETED
            )
        first = int(failed[0])
        return BatchTransferResult(
            num_ranges=num_ranges,
            state=TransferState.FAILED,
            error=(
                f"TENT range {first} ended in "
                f"{self._state_name(int(states[first]))}, "
                f"{int(failed.size)} of {num_ranges} ranges failed"
            ),
        )

    def shutdown(self) -> None:
        deadline = time.monotonic() + SHUTDOWN_TIMEOUT_S
        with self._drain:
            self._closing = True
            while self._inflight:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "TENT still has %d transfers running at shutdown.",
                        self._inflight,
                    )
                    break
                self._drain.wait(remaining)
        with self._segment_lock:
            peers = list(self._segments)
        for peer in peers:
            self._close_segment(peer)

    def _submit(
        self,
        peer: str,
        op: TransferOp,
        local_addrs: NDArray[np.uint64],
        remote_addrs: NDArray[np.uint64],
        lengths: NDArray[np.uint64],
    ) -> NDArray[np.uint8]:
        self._begin_transfer()
        try:
            if self._sync_arrays is not None:
                return self._submit_arrays(peer, op, local_addrs, remote_addrs, lengths)
            return self._submit_lists(peer, op, local_addrs, remote_addrs, lengths)
        finally:
            self._end_transfer()

    def _submit_arrays(
        self,
        peer: str,
        op: TransferOp,
        local_addrs: NDArray[np.uint64],
        remote_addrs: NDArray[np.uint64],
        lengths: NDArray[np.uint64],
    ) -> NDArray[np.uint8]:
        segment = self._segment(peer)
        opcode = self._opcodes[op]
        if self._sync_arrays_timed:
            states = self._engine.transfer_sync_arrays(
                local_addrs,
                remote_addrs,
                lengths,
                segment,
                opcode,
                self._timeout_ms,
            )
        else:
            states = self._engine.transfer_sync_arrays(
                local_addrs, remote_addrs, lengths, segment, opcode
            )
        return np.asarray(states, dtype=np.uint8)

    def _submit_lists(
        self,
        peer: str,
        op: TransferOp,
        local_addrs: NDArray[np.uint64],
        remote_addrs: NDArray[np.uint64],
        lengths: NDArray[np.uint64],
    ) -> NDArray[np.uint8]:
        segment = self._segment(peer)
        count = int(local_addrs.size)
        batch = self._engine.allocate_transfer_batch(count)
        if batch <= 0:
            raise RuntimeError("TENT could not allocate a transfer batch")
        requests = [
            self._tent.Request(
                self._opcodes[op],
                int(local_addrs[index]),
                segment,
                int(remote_addrs[index]),
                int(lengths[index]),
            )
            for index in range(count)
        ]
        try:
            self._engine.submit_transfer(batch, requests)
            if self._wait_transfer is not None:
                statuses = self._wait_transfer(batch, self._timeout_ms)
            else:
                statuses = self._poll_batch(batch)
        finally:
            self._engine.free_transfer_batch(batch)
        return np.asarray([int(item.state) for item in statuses], dtype=np.uint8)

    def _poll_batch(self, batch: Any) -> list[Any]:
        deadline = time.monotonic() + self._timeout_s if self._timeout_s > 0 else None
        delay = POLL_INTERVAL_MIN_S
        while True:
            overall = self._engine.get_transfer_status_overall(batch)
            if int(overall.state) != self._pending:
                break
            if deadline is not None and time.monotonic() >= deadline:
                break
            time.sleep(delay)
            delay = min(delay * 2, POLL_INTERVAL_MAX_S)
        return self._engine.get_transfer_status_list(batch)

    def _begin_transfer(self) -> None:
        with self._drain:
            if self._closing:
                raise RuntimeError("The TENT transfer backend is shutting down.")
            self._inflight += 1

    def _end_transfer(self) -> None:
        with self._drain:
            self._inflight -= 1
            if self._inflight == 0:
                self._drain.notify_all()

    def _segment(self, peer: str) -> Any:
        with self._segment_lock:
            segment = self._segments.get(peer)
            if segment is not None:
                return segment
            open_lock = self._open_locks.setdefault(peer, threading.Lock())
        with open_lock:
            with self._segment_lock:
                segment = self._segments.get(peer)
            if segment is None:
                segment = self._engine.open_segment(peer)
                with self._segment_lock:
                    self._segments[peer] = segment
            return segment

    def _close_segment(self, peer: str) -> None:
        with self._segment_lock:
            segment = self._segments.pop(peer, None)
        if segment is None:
            return
        try:
            self._engine.close_segment(segment)
        except Exception as error:
            logger.debug("TENT failed to close the segment to %s: %s", peer, error)

    def _state_name(self, code: int) -> str:
        return self._state_names.get(code, str(code))


def _format_peer_address(hostname: str, port: int) -> str:
    if ":" in hostname:
        return f"[{hostname}]:{port}"
    return f"{hostname}:{port}"


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
        "transfer_engine='tent' requires the 'tent' Python module. "
        f"{_TENT_BUILD_HINT} Alternatively, select transfer_engine='te'."
    )
