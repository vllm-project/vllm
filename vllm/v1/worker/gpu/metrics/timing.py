# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from functools import wraps
from threading import Lock
from typing import Any

import numpy as np
import torch

from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import WorkerTimingStats
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    AsyncModelRunnerOutput,
    ModelRunnerOutput,
)


def finalize_step_timing(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(self: Any, *args, **kwargs) -> Any:
        return self.worker_timing.finish_output(fn(self, *args, **kwargs))

    return wrapper


def drain_timing_samples(fn: Callable) -> Callable:
    @wraps(fn)
    def wrapper(self: Any, *args, **kwargs) -> Any:
        result = fn(self, *args, **kwargs)
        if isinstance(result, ModelRunnerOutput):
            self.worker_timing.cancel_step()
            return self.worker_timing.drain_into(result)
        return result

    return wrapper


@dataclass
class TimingAggregate:
    """Durations accumulated for one phase and batch-size combination."""

    count: int = 0
    total_seconds: float = 0.0

    def observe(self, duration: float) -> None:
        self.count += 1
        self.total_seconds += duration

    @property
    def mean_seconds(self) -> float:
        return self.total_seconds / self.count if self.count else 0.0


@dataclass(frozen=True)
class _StepMetadata:
    """Unpadded batch counts plus the padded target-model token count."""

    num_model_tokens: int
    num_prefill_requests: int
    num_prefill_tokens: int
    num_decode_requests: int
    num_decode_tokens: int


@dataclass
class _Step:
    """Events and metadata for one active or asynchronously pending step."""

    iteration_index: int
    start_event: torch.cuda.Event
    metadata: _StepMetadata | None = None
    proposer_start_event: torch.cuda.Event | None = None
    proposer_end_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None

    def is_ready(self) -> bool:
        return self.end_event is not None and self.end_event.query()


class ModelRunnerTiming:
    """GPUModelRunner timing state, also drained by its async-output thread."""

    def __init__(
        self,
        stream: torch.cuda.Stream,
        event_factory: Callable[[], torch.cuda.Event] | None = None,
    ) -> None:
        self._stream = stream
        self._event_factory = event_factory or self._make_event
        self._event_pool: list[torch.cuda.Event] = []
        self._pending: deque[_Step] = deque()
        self._lock = Lock()
        self._active: _Step | None = None
        self._suspended = False
        self._next_iteration_index = 0

        self.model_times: dict[tuple[str, int], TimingAggregate] = defaultdict(
            TimingAggregate
        )
        self.proposer_times: dict[tuple[str, int], TimingAggregate] = defaultdict(
            TimingAggregate
        )
        self.total_times: dict[tuple[str, int], TimingAggregate] = defaultdict(
            TimingAggregate
        )

    @staticmethod
    def _make_event() -> torch.cuda.Event:
        return torch.cuda.Event(enable_timing=True)

    @property
    def is_active(self) -> bool:
        return self._active is not None

    @contextmanager
    def suspend(self) -> Iterator[None]:
        """Temporarily exclude synthetic model runs from timing."""
        was_suspended = self._suspended
        self._suspended = True
        try:
            yield
        finally:
            self._suspended = was_suspended

    def _record_event(self) -> torch.cuda.Event:
        with self._lock:
            event = (
                self._event_pool.pop() if self._event_pool else self._event_factory()
            )
        event.record(self._stream)
        return event

    def start_step(
        self,
        dummy_run: bool,
        scheduler_output: SchedulerOutput,
    ) -> None:
        if self._active is not None:
            raise RuntimeError("A model-runner timing step is already active")
        if (
            self._suspended
            or dummy_run
            or scheduler_output.total_num_scheduled_tokens == 0
        ):
            return
        self._active = _Step(self._next_iteration_index, self._record_event())
        self._next_iteration_index += 1

    def set_step_metadata(self, input_batch: Any) -> None:
        active = self._active
        if active is None:
            return

        is_prefilling = input_batch.is_prefilling_np
        num_prefill_requests = int(np.count_nonzero(is_prefilling))
        num_prefill_tokens = int(input_batch.num_scheduled_tokens[is_prefilling].sum())
        num_decode_requests = input_batch.num_reqs - num_prefill_requests
        num_decode_tokens = int(input_batch.num_scheduled_tokens[~is_prefilling].sum())
        active.metadata = _StepMetadata(
            num_model_tokens=input_batch.num_tokens_after_padding,
            num_prefill_requests=num_prefill_requests,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_requests=num_decode_requests,
            num_decode_tokens=num_decode_tokens,
        )

    def start_proposer(self) -> None:
        if self._suspended:
            return
        active = self._require_active()
        if active.proposer_start_event is not None:
            raise RuntimeError("The proposer timing interval is already active")
        active.proposer_start_event = self._record_event()

    def end_proposer(self) -> None:
        if self._suspended:
            return
        active = self._require_active()
        if active.proposer_start_event is None:
            raise RuntimeError("The proposer timing interval was not started")
        if active.proposer_end_event is not None:
            raise RuntimeError("The proposer timing interval is already complete")
        active.proposer_end_event = self._record_event()

    def _finish_step(self) -> None:
        active = self._require_active()
        if active.metadata is None:
            raise RuntimeError("Model-runner timing metadata was not set")
        if (active.proposer_start_event is None) != (active.proposer_end_event is None):
            raise RuntimeError("The proposer timing interval is incomplete")

        active.end_event = self._record_event()
        with self._lock:
            self._pending.append(active)
        self._active = None

    def finish_output(
        self,
        result: AsyncModelRunnerOutput | ModelRunnerOutput | None,
    ) -> AsyncModelRunnerOutput | ModelRunnerOutput | None:
        if not self.is_active:
            return result
        if result is None:
            self.cancel_step()
            return None
        self._finish_step()
        if isinstance(result, ModelRunnerOutput):
            return self.drain_into(result)
        return result

    def cancel_step(self) -> None:
        self._active = None

    def drain_into(self, output: ModelRunnerOutput) -> ModelRunnerOutput:
        ready: list[WorkerTimingStats] = []
        with self._lock:
            while self._pending and self._pending[0].is_ready():
                pending = self._pending.popleft()
                timing = self._make_timing_stats(pending)
                ready.append(timing)
                self._observe(timing)
                self._release_events(pending)
        if not ready:
            return output
        if output is EMPTY_MODEL_RUNNER_OUTPUT:
            output = copy(output)
            output.worker_timing_samples = ready
        else:
            output.worker_timing_samples.extend(ready)
        return output

    def _require_active(self) -> _Step:
        if self._active is None:
            raise RuntimeError("No model-runner timing step is active")
        return self._active

    @staticmethod
    def _make_timing_stats(pending: _Step) -> WorkerTimingStats:
        start = pending.start_event
        end = pending.end_event
        metadata = pending.metadata
        assert end is not None and metadata is not None
        proposer_start = pending.proposer_start_event
        proposer_end = pending.proposer_end_event
        proposer_time = None
        if proposer_start is None:
            model_time = start.elapsed_time(end) * 1e-3
        else:
            assert proposer_end is not None
            before_proposer = start.elapsed_time(proposer_start) * 1e-3
            proposer_time = proposer_start.elapsed_time(proposer_end) * 1e-3
            after_proposer = proposer_end.elapsed_time(end) * 1e-3
            model_time = before_proposer + after_proposer
        total_time = model_time + (proposer_time or 0.0)
        return WorkerTimingStats(
            iteration_index=pending.iteration_index,
            phase="prefill" if metadata.num_prefill_requests else "decode",
            num_model_tokens=metadata.num_model_tokens,
            num_requests=(metadata.num_prefill_requests + metadata.num_decode_requests),
            num_prefill_requests=metadata.num_prefill_requests,
            num_prefill_tokens=metadata.num_prefill_tokens,
            num_decode_requests=metadata.num_decode_requests,
            num_decode_tokens=metadata.num_decode_tokens,
            model_time_seconds=model_time,
            proposer_time_seconds=proposer_time,
            total_time_seconds=total_time,
        )

    def _observe(self, timing: WorkerTimingStats) -> None:
        model_key = (timing.phase, timing.num_model_tokens)
        self.model_times[model_key].observe(timing.model_time_seconds)
        self.total_times[model_key].observe(timing.total_time_seconds)
        if timing.proposer_time_seconds is not None:
            proposer_key = (timing.phase, timing.num_requests)
            self.proposer_times[proposer_key].observe(timing.proposer_time_seconds)

    def _release_events(self, pending: _Step) -> None:
        self._event_pool.append(pending.start_event)
        if pending.proposer_start_event is not None:
            self._event_pool.append(pending.proposer_start_event)
        if pending.proposer_end_event is not None:
            self._event_pool.append(pending.proposer_end_event)
        assert pending.end_event is not None
        self._event_pool.append(pending.end_event)
