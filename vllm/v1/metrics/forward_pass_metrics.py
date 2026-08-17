# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native per-iteration forward-pass metrics.

This module intentionally owns the FPM wire schema and nearly all supporting
logic: CUDA event timing, scheduler-state aggregation, iteration association,
and background ZMQ publication. The engine, scheduler, and model runners only
need small lifecycle hooks.

``wall_time`` is the CUDA-timeline interval between events recorded immediately
before and after ``execute_model``. It includes GPU work and any GPU-idle gap
while the host prepares/submits work inside that call. It excludes EngineCore
scheduling, output sampling, and metrics serialization/publication.
"""

from __future__ import annotations

import queue
import threading
import time
from collections import OrderedDict, deque
from collections.abc import Callable, Iterable
from contextlib import suppress
from copy import copy
from dataclasses import dataclass
from itertools import count
from typing import TYPE_CHECKING, Any, Protocol

import msgspec
import torch
import zmq

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.interface import SchedulerInterface
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import ModelRunnerOutput

logger = init_logger(__name__)

FPM_VERSION = 1
FPM_HEARTBEAT_INTERVAL_SECONDS = 1.0
FPM_TIMING_SCOPE_EXECUTE_MODEL_CUDA = "execute_model_cuda"


@dataclass(slots=True)
class WelfordAccumulator:
    """Single-pass count, sum, and population variance accumulator."""

    count: int = 0
    total: int = 0
    _mean: float = 0.0
    _m2: float = 0.0

    def add(self, value: int) -> None:
        self.count += 1
        self.total += value
        delta = value - self._mean
        self._mean += delta / self.count
        self._m2 += delta * (value - self._mean)

    @property
    def variance(self) -> float:
        return self._m2 / self.count if self.count else 0.0


class ScheduledRequestMetrics(
    msgspec.Struct,
    frozen=True,  # type: ignore[call-arg]
    gc=False,  # type: ignore[call-arg]
):
    """Requests scheduled in one model execution."""

    num_prefill_requests: int = 0
    sum_prefill_tokens: int = 0
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
    """Requests waiting after one model execution completes."""

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
    timing_scope: str = FPM_TIMING_SCOPE_EXECUTE_MODEL_CUDA
    # CUDA-timeline interval spanning execute_model, in seconds.
    wall_time: float = 0.0
    scheduled_requests: ScheduledRequestMetrics = ScheduledRequestMetrics()
    queued_requests: QueuedRequestMetrics = QueuedRequestMetrics()


_fpm_encoder = msgspec.msgpack.Encoder()
_fpm_decoder = msgspec.msgpack.Decoder(ForwardPassMetrics)


def encode_forward_pass_metrics(metrics: ForwardPassMetrics) -> bytes:
    return _fpm_encoder.encode(metrics)


def decode_forward_pass_metrics(data: bytes) -> ForwardPassMetrics | None:
    """Decode an FPM payload, returning ``None`` for unsupported versions."""

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


@dataclass(slots=True)
class _PendingTiming:
    iteration_id: int
    start_event: Any
    end_event: Any


class ForwardPassMetricsTimer:
    """Non-blocking CUDA event timer used by both GPU model runners.

    Timing events are preallocated and pooled. Hot-path drains only read elapsed
    time after ``query`` reports completion. The engine uses ``wait=True`` only
    on the cold transition to idle so the final iteration cannot be lost.
    """

    def __init__(
        self,
        num_event_pairs: int,
        event_factory: Callable[[], Any] | None = None,
    ) -> None:
        if num_event_pairs <= 0:
            raise ValueError("num_event_pairs must be positive")
        self._event_factory = event_factory or self._make_event
        self._event_pool: queue.SimpleQueue[tuple[Any, Any]] = queue.SimpleQueue()
        for _ in range(num_event_pairs):
            self._event_pool.put((self._event_factory(), self._event_factory()))
        self._pending: deque[_PendingTiming] = deque()
        self._active: _PendingTiming | None = None
        self._pending_lock = threading.Lock()
        self._drain_lock = threading.Lock()

    @staticmethod
    def _make_event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    def start(self, scheduler_output: SchedulerOutput) -> None:
        if self._active is not None:
            raise RuntimeError("A forward-pass timing interval is already active")
        iteration_id = scheduler_output.forward_pass_metrics_iteration_id
        if iteration_id is None or scheduler_output.total_num_scheduled_tokens == 0:
            return
        try:
            start_event, end_event = self._event_pool.get_nowait()
        except queue.Empty:
            # This should only be possible if the configured concurrency bound
            # is violated or completed events are not being drained.
            logger.warning_once(
                "Forward-pass CUDA event pool exhausted; allocating one extra pair"
            )
            start_event, end_event = self._event_factory(), self._event_factory()
        start_event.record()
        self._active = _PendingTiming(iteration_id, start_event, end_event)

    def finish(self) -> None:
        if self._active is None:
            return
        pending = self._active
        self._active = None
        pending.end_event.record()
        with self._pending_lock:
            self._pending.append(pending)

    def cancel(self) -> None:
        # Exception paths are cold. Wait before recycling the recorded start
        # event so a later interval cannot re-record it while it is in flight.
        if self._active is not None:
            self._active.start_event.synchronize()
            self._event_pool.put((self._active.start_event, self._active.end_event))
        self._active = None

    def drain_samples(self, wait: bool = False) -> tuple[tuple[int, float], ...]:
        """Return completed timing samples without blocking unless requested.

        CUDA driver calls run outside ``_pending_lock``. The inference thread
        therefore never waits behind ``query``/``elapsed_time``/``synchronize``.
        """

        ready: list[tuple[int, float]] = []
        with self._drain_lock:
            while True:
                with self._pending_lock:
                    pending = self._pending[0] if self._pending else None
                if pending is None:
                    break
                if wait:
                    pending.end_event.synchronize()
                elif not pending.end_event.query():
                    break

                elapsed_ms = pending.start_event.elapsed_time(pending.end_event)
                with self._pending_lock:
                    completed = self._pending.popleft()
                assert completed is pending
                ready.append((pending.iteration_id, elapsed_ms * 1e-3))
                self._event_pool.put((pending.start_event, pending.end_event))
        return tuple(ready)

    def drain_into(self, output: Any) -> Any:
        """Attach ready timing samples to a model-runner output."""

        if output is None or not hasattr(output, "forward_pass_timing_samples"):
            return output
        ready = self.drain_samples()
        if not ready:
            return output

        # EMPTY_MODEL_RUNNER_OUTPUT is a shared singleton. Copy every output
        # before attaching samples so instrumentation cannot mutate shared state.
        output = copy(output)
        output.forward_pass_timing_samples += ready
        return output


def make_forward_pass_metrics_timer(
    vllm_config: VllmConfig,
    is_output_rank: bool,
) -> ForwardPassMetricsTimer | None:
    """Create a preallocated timer only on the model-output worker.

    The disabled path allocates no CUDA events and model runners only execute
    one predictable Python branch in existing worker methods.
    """

    if (
        vllm_config.observability_config.forward_pass_metrics_port <= 0
        or not is_output_rank
    ):
        return None
    return ForwardPassMetricsTimer(
        num_event_pairs=max(2, vllm_config.max_concurrent_batches + 1)
    )


def is_forward_pass_metrics_output_rank(
    vllm_config: VllmConfig,
    rank: int,
) -> bool:
    """Return whether ``rank`` owns the executor's ModelRunnerOutput."""

    parallel_config = vllm_config.parallel_config
    output_rank = parallel_config.world_size - (
        parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    return rank == output_rank


class _MetricsPublisher(Protocol):
    def publish(self, metrics: ForwardPassMetrics) -> None: ...

    def shutdown(self) -> None: ...


class ZmqForwardPassMetricsPublisher:
    """Bounded, non-blocking FPM publisher with background serialization."""

    SHUTDOWN_TIMEOUT_SECONDS = 1.0
    STARTUP_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        endpoint: str,
        worker_id: str,
        dp_rank: int,
        max_queue_size: int,
    ) -> None:
        self._queue = queue.Queue[ForwardPassMetrics | None](maxsize=max_queue_size)
        self._worker_id = worker_id
        self._dp_rank = dp_rank
        self._sequence = count()
        self._endpoint = endpoint
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._startup_error: Exception | None = None

        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="fpm-zmq-publisher",
        )
        self._thread.start()
        if not self._ready.wait(self.STARTUP_TIMEOUT_SECONDS):
            self._stop.set()
            self._thread.join(timeout=self.SHUTDOWN_TIMEOUT_SECONDS)
            raise TimeoutError(
                f"Timed out binding forward-pass metrics publisher to {endpoint}"
            )
        if self._startup_error is not None:
            raise RuntimeError(
                f"Failed to bind forward-pass metrics publisher to {endpoint}"
            ) from self._startup_error

    def publish(self, metrics: ForwardPassMetrics) -> None:
        if self._stop.is_set():
            return
        try:
            self._queue.put_nowait(metrics)
        except queue.Full:
            # FPM is observability data and must never backpressure inference.
            logger.warning_once("Forward-pass metrics queue is full; dropping samples")

    def shutdown(self) -> None:
        self._stop.set()
        with suppress(queue.Full):
            self._queue.put_nowait(None)
        self._thread.join(timeout=self.SHUTDOWN_TIMEOUT_SECONDS)
        if self._thread.is_alive():
            logger.warning("Forward-pass metrics publisher did not stop in time")

    def _run(self) -> None:
        socket: zmq.Socket[bytes] | None = None
        try:
            context = zmq.Context.instance()
            socket = context.socket(zmq.PUB)
            socket.bind(self._endpoint)
        except Exception as exc:
            self._startup_error = exc
            self._ready.set()
            if socket is not None:
                socket.close(linger=0)
            return

        self._ready.set()
        last_publish = time.monotonic()
        try:
            # Metrics are best-effort observability data. On shutdown, exit
            # instead of draining a potentially large backlog.
            while not self._stop.is_set():
                try:
                    metrics = self._queue.get(
                        timeout=FPM_HEARTBEAT_INTERVAL_SECONDS,
                    )
                    if metrics is None:
                        break
                except queue.Empty:
                    if time.monotonic() - last_publish < FPM_HEARTBEAT_INTERVAL_SECONDS:
                        continue
                    metrics = ForwardPassMetrics(
                        worker_id=self._worker_id,
                        dp_rank=self._dp_rank,
                    )

                try:
                    sequence = next(self._sequence)
                    metrics = msgspec.structs.replace(metrics, counter_id=sequence)
                    socket.send_multipart(
                        (
                            b"",
                            sequence.to_bytes(8, "big"),
                            encode_forward_pass_metrics(metrics),
                        ),
                        flags=zmq.NOBLOCK,
                    )
                    last_publish = time.monotonic()
                except zmq.Again:
                    pass
                except Exception:
                    logger.warning("Forward-pass metrics publish failed", exc_info=True)
        finally:
            socket.close(linger=0)


@dataclass(slots=True)
class _PendingIteration:
    scheduled: ScheduledRequestMetrics
    queued: QueuedRequestMetrics | None = None
    duration_seconds: float | None = None


class ForwardPassMetricsEmitter:
    """Associate worker timing samples with scheduler snapshots and emit FPM."""

    def __init__(
        self,
        worker_id: str,
        dp_rank: int,
        publisher: _MetricsPublisher,
        max_pending_iterations: int = 16,
    ) -> None:
        if max_pending_iterations <= 0:
            raise ValueError("max_pending_iterations must be positive")
        self._worker_id = worker_id
        self._dp_rank = dp_rank
        self._publisher = publisher
        self._iteration_ids = count()
        self._max_pending_iterations = max_pending_iterations
        self._pending: OrderedDict[int, _PendingIteration] = OrderedDict()

    @classmethod
    def from_vllm_config(
        cls,
        vllm_config: VllmConfig,
        scheduler: SchedulerInterface,
    ) -> ForwardPassMetricsEmitter | None:
        config = vllm_config.observability_config
        if config.forward_pass_metrics_port <= 0:
            return None
        # Fail at startup for custom schedulers that do not provide the state
        # contract required to compute FPM snapshots.
        scheduler.get_forward_pass_metrics_request_state()

        dp_rank = getattr(vllm_config.parallel_config, "data_parallel_index", None)
        if dp_rank is None:
            dp_rank = vllm_config.parallel_config.data_parallel_rank or 0
        port = config.forward_pass_metrics_port + dp_rank
        if port > 65535:
            raise ValueError(
                "forward-pass-metrics-port plus data-parallel rank exceeds 65535"
            )
        worker_id = config.forward_pass_metrics_worker_id or vllm_config.instance_id
        publisher = ZmqForwardPassMetricsPublisher(
            endpoint=f"tcp://*:{port}",
            worker_id=worker_id,
            dp_rank=dp_rank,
            max_queue_size=config.forward_pass_metrics_max_queue_size,
        )
        logger.info("Forward-pass metrics publisher bound to tcp://*:%d", port)
        return cls(
            worker_id=worker_id,
            dp_rank=dp_rank,
            publisher=publisher,
            max_pending_iterations=max(16, vllm_config.max_concurrent_batches * 4),
        )

    def begin_iteration(
        self,
        scheduler: SchedulerInterface,
        scheduler_output: SchedulerOutput,
    ) -> None:
        if (
            scheduler_output.total_num_scheduled_tokens == 0
            or not scheduler.should_emit_forward_pass_metrics(scheduler_output)
        ):
            return
        if len(self._pending) >= self._max_pending_iterations:
            stale_iteration_id, _ = self._pending.popitem(last=False)
            logger.warning_once(
                "Forward-pass metrics pending state reached its bound; "
                "dropping stale iterations (first dropped iteration: %d)",
                stale_iteration_id,
            )
        iteration_id = next(self._iteration_ids)
        scheduler_output.forward_pass_metrics_iteration_id = iteration_id
        self._pending[iteration_id] = _PendingIteration(
            scheduled=_extract_scheduled_metrics(scheduler, scheduler_output)
        )

    def complete_iteration(
        self,
        scheduler: SchedulerInterface,
        scheduler_output: SchedulerOutput,
        model_output: ModelRunnerOutput,
    ) -> None:
        self.complete_timing_samples(model_output.forward_pass_timing_samples)

        iteration_id = scheduler_output.forward_pass_metrics_iteration_id
        if iteration_id is not None and (pending := self._pending.get(iteration_id)):
            pending.queued = _extract_queued_metrics(scheduler)
            self._emit_if_ready(iteration_id)

    def complete_timing_samples(self, samples: Iterable[tuple[int, float]]) -> None:
        """Join completed worker timings with their scheduler snapshots."""

        for iteration_id, duration_seconds in samples:
            pending = self._pending.get(iteration_id)
            if pending is None:
                logger.debug(
                    "Ignoring timing for unknown FPM iteration %d",
                    iteration_id,
                )
                continue
            pending.duration_seconds = duration_seconds
            self._emit_if_ready(iteration_id)

    def _emit_if_ready(self, iteration_id: int) -> None:
        pending = self._pending.get(iteration_id)
        if (
            pending is None
            or pending.queued is None
            or pending.duration_seconds is None
        ):
            return
        del self._pending[iteration_id]
        self._publisher.publish(
            ForwardPassMetrics(
                worker_id=self._worker_id,
                dp_rank=self._dp_rank,
                timing_scope=FPM_TIMING_SCOPE_EXECUTE_MODEL_CUDA,
                wall_time=pending.duration_seconds,
                scheduled_requests=pending.scheduled,
                queued_requests=pending.queued,
            )
        )

    def has_pending_timing(self) -> bool:
        return bool(self._pending)

    def shutdown(self) -> None:
        self._publisher.shutdown()


def _extract_scheduled_metrics(
    scheduler: SchedulerInterface,
    output: SchedulerOutput,
) -> ScheduledRequestMetrics:
    num_scheduled = output.num_scheduled_tokens
    requests, _, _ = scheduler.get_forward_pass_metrics_request_state()
    prefill_lengths = WelfordAccumulator()
    decode_kv = WelfordAccumulator()
    num_prefill = 0
    sum_prefill_tokens = 0
    sum_prefill_kv_tokens = 0

    for request_data in output.scheduled_new_reqs:
        num_prefill += 1
        sum_prefill_tokens += num_scheduled.get(request_data.req_id, 0)
        request = requests.get(request_data.req_id)
        prompt_length = (
            request.num_prompt_tokens
            if request is not None
            else len(request_data.prompt_token_ids or ())
        )
        prefill_lengths.add(prompt_length)
        sum_prefill_kv_tokens += request_data.num_computed_tokens

    cached = output.scheduled_cached_reqs
    for index, request_id in enumerate(cached.req_ids):
        if cached.is_context_phase(request_id):
            num_prefill += 1
            sum_prefill_tokens += num_scheduled.get(request_id, 0)
            request = requests.get(request_id)
            prefill_lengths.add(request.num_prompt_tokens if request else 0)
            sum_prefill_kv_tokens += cached.num_computed_tokens[index]
        else:
            decode_kv.add(cached.num_computed_tokens[index])

    return ScheduledRequestMetrics(
        num_prefill_requests=num_prefill,
        sum_prefill_tokens=sum_prefill_tokens,
        var_prefill_length=prefill_lengths.variance,
        sum_prefill_kv_tokens=sum_prefill_kv_tokens,
        num_decode_requests=decode_kv.count,
        sum_decode_kv_tokens=decode_kv.total,
        var_decode_kv_tokens=decode_kv.variance,
    )


def _extract_queued_metrics(scheduler: SchedulerInterface) -> QueuedRequestMetrics:
    from vllm.v1.request import RequestStatus

    prefill = WelfordAccumulator()
    decode_kv = WelfordAccumulator()

    _, waiting, skipped_waiting = scheduler.get_forward_pass_metrics_request_state()
    for request in waiting:
        if request.status == RequestStatus.PREEMPTED:
            decode_kv.add(request.num_computed_tokens)
        else:
            prefill.add(request.num_tokens)

    for request in skipped_waiting:
        if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
            decode_kv.add(request.num_computed_tokens)
        else:
            prefill.add(request.num_tokens)

    return QueuedRequestMetrics(
        num_prefill_requests=prefill.count,
        sum_prefill_tokens=prefill.total,
        var_prefill_length=prefill.variance,
        num_decode_requests=decode_kv.count,
        sum_decode_kv_tokens=decode_kv.total,
        var_decode_kv_tokens=decode_kv.variance,
    )
