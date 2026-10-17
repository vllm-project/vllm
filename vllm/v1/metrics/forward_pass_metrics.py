# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""FPM v1 wire format and export of completed engine iteration records.

Timing, workload snapshots, and input/time association belong to the generic
iteration-metrics pipeline. This module only maps its completed records and
publishes them from a bounded background queue owned by each EngineCore.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Sequence
from contextlib import suppress
from typing import TYPE_CHECKING

import msgspec
import zmq

from vllm.logger import init_logger
from vllm.v1.metrics.iteration import ITERATION_TIMING_SCOPE

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.metrics.stats import SchedulerIterationDetails

logger = init_logger(__name__)

FPM_VERSION = 1
FPM_MAX_QUEUE_SIZE = 10_000
FPM_TIMING_SCOPE_MODEL_STEP_CUDA = ITERATION_TIMING_SCOPE


class ScheduledRequestMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    """Requests scheduled in one model execution."""

    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    # Population variance of kv_read + scheduled_tokens / 2, in tokens squared.
    var_prefill_length: float = 0.0
    sum_prefill_kv_tokens: int = 0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


class QueuedRequestMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    """Waiting work at the iteration record's queue-snapshot boundary."""

    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
    var_prefill_length: float = 0.0
    num_decode_requests: int = 0
    sum_decode_kv_tokens: int = 0
    var_decode_kv_tokens: float = 0.0


class ForwardPassMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    """Versioned, per-iteration forward-pass metrics wire payload."""

    version: int = FPM_VERSION
    worker_id: str = ""
    dp_rank: int = 0
    counter_id: int = 0
    timing_scope: str = FPM_TIMING_SCOPE_MODEL_STEP_CUDA
    # CUDA-stream interval in seconds; zero denotes engine-confirmed idle.
    wall_time: float = 0.0
    scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()
    queued_requests: QueuedRequestMetrics = QueuedRequestMetrics()


_fpm_encoder = msgspec.msgpack.Encoder()
_fpm_decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)


def encode_forward_pass_metrics(metrics: ForwardPassMetrics) -> bytes:
    return _fpm_encoder.encode(metrics)


def decode_forward_pass_metrics(data: bytes) -> ForwardPassMetrics | None:
    """Decode an FPM payload, ignoring malformed or unsupported versions."""
    try:
        metrics = _fpm_decoder.decode(data)
    except msgspec.DecodeError:
        logger.warning("Ignoring malformed forward-pass metrics payload")
        return None
    if metrics.version != FPM_VERSION:
        logger.warning(
            "Ignoring forward-pass metrics version %d; supported version is %d",
            metrics.version,
            FPM_VERSION,
        )
        return None
    return metrics


def forward_pass_metrics_from_iteration(
    details: SchedulerIterationDetails,
    worker_id: str,
    dp_rank: int,
    sequence: int,
) -> ForwardPassMetrics | None:
    """Map one complete record without recomputing its workload or timing."""
    scheduled = details.scheduled_workload
    queued = details.queued_workload
    if (
        details.elapsed_ms is None
        or details.timing_scope != FPM_TIMING_SCOPE_MODEL_STEP_CUDA
        or scheduled is None
        or queued is None
        or (details.elapsed_ms <= 0 and not details.is_idle)
    ):
        logger.warning_once("Skipping incomplete or untimed forward-pass metrics")
        return None
    return ForwardPassMetrics(
        worker_id=worker_id,
        dp_rank=dp_rank,
        counter_id=sequence,
        wall_time=details.elapsed_ms * 1e-3,
        scheduled_requests=ScheduledRequestMetrics(
            num_prefill_requests=details.num_ctx_requests,
            sum_prefill_tokens=details.num_ctx_tokens,
            var_prefill_length=scheduled.var_prefill_length,
            sum_prefill_kv_tokens=scheduled.sum_prefill_kv_tokens,
            num_decode_requests=details.num_generation_requests,
            sum_decode_kv_tokens=scheduled.sum_decode_kv_tokens,
            var_decode_kv_tokens=scheduled.var_decode_kv_tokens,
        ),
        queued_requests=QueuedRequestMetrics(
            num_prefill_requests=queued.num_prefill_requests,
            sum_prefill_tokens=queued.sum_prefill_tokens,
            var_prefill_length=queued.var_prefill_length,
            num_decode_requests=queued.num_decode_requests,
            sum_decode_kv_tokens=queued.sum_decode_kv_tokens,
            var_decode_kv_tokens=queued.var_decode_kv_tokens,
        ),
    )


class ForwardPassMetricsExporter:
    """Non-blocking exporter; one background thread owns this engine's socket."""

    SHUTDOWN_TIMEOUT_SECONDS = 1.0
    STARTUP_TIMEOUT_SECONDS = 5.0

    def __init__(self, base_port: int, worker_id: str, dp_rank: int) -> None:
        self._base_port = base_port
        self._worker_id = worker_id
        self._dp_rank = dp_rank
        self._queue: queue.Queue[tuple[int, SchedulerIterationDetails] | None] = (
            queue.Queue(maxsize=FPM_MAX_QUEUE_SIZE)
        )
        self._sequence = 0
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: Exception | None = None
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="fpm-zmq-publisher"
        )
        self._thread.start()
        if not self._ready.wait(self.STARTUP_TIMEOUT_SECONDS):
            self.shutdown()
            raise TimeoutError("Timed out binding forward-pass metrics publisher")
        if self._startup_error is not None:
            self.shutdown()
            raise RuntimeError("Failed to bind forward-pass metrics publisher") from (
                self._startup_error
            )

    @classmethod
    def from_vllm_config(
        cls, vllm_config: VllmConfig
    ) -> ForwardPassMetricsExporter | None:
        config = vllm_config.observability_config
        if config.forward_pass_metrics_port == 0:
            return None
        return cls(
            config.forward_pass_metrics_port,
            config.forward_pass_metrics_worker_id or vllm_config.instance_id,
            vllm_config.parallel_config.data_parallel_index,
        )

    def record(self, records: Sequence[SchedulerIterationDetails]) -> None:
        if self._stop.is_set():
            return
        for details in records:
            sequence = self._sequence
            self._sequence += 1
            try:
                self._queue.put_nowait((sequence, details))
            except queue.Full:
                logger.warning_once(
                    "Forward-pass metrics queue is full; dropping samples"
                )

    def shutdown(self) -> None:
        self._stop.set()
        with suppress(queue.Full):
            self._queue.put_nowait(None)
        self._thread.join(timeout=self.SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            logger.warning("Forward-pass metrics publisher did not stop in time")

    def _bind_socket(self) -> zmq.Socket:
        port = self._base_port + self._dp_rank
        if not 1 <= port <= 65535:
            raise ValueError("forward-pass-metrics-port plus DP rank exceeds TCP range")
        endpoint = f"tcp://*:{port}"
        socket = zmq.Context.instance().socket(zmq.PUB)
        try:
            socket.bind(endpoint)
        except Exception:
            socket.close(linger=0)
            raise
        logger.info("Forward-pass metrics publisher bound to %s", endpoint)
        return socket

    def _run(self) -> None:
        socket: zmq.Socket | None = None
        try:
            try:
                socket = self._bind_socket()
            except Exception as exc:
                self._startup_error = exc
                return
            finally:
                self._ready.set()
            while not self._stop.is_set():
                item = self._queue.get()
                if item is None:
                    break
                sequence, details = item
                metrics = forward_pass_metrics_from_iteration(
                    details, self._worker_id, self._dp_rank, sequence
                )
                if metrics is None:
                    continue
                try:
                    socket.send_multipart(
                        (
                            b"",
                            sequence.to_bytes(8, "big"),
                            encode_forward_pass_metrics(metrics),
                        ),
                        flags=zmq.NOBLOCK,
                    )
                except zmq.Again:
                    logger.warning_once("Forward-pass metrics send queue is full")
                except Exception as exc:
                    logger.warning_once("Forward-pass metrics publish failed: %s", exc)
        finally:
            if socket is not None:
                socket.close(linger=0)
