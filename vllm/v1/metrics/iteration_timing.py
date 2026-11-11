# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Non-blocking GPU timing of engine iterations.

An iteration is timed between two CUDA events recorded on the output worker's
stream: one before `execute_model`, one after `sample_tokens` returns. The
interval covers the model forward pass, sampling and speculative drafting,
including host submission gaps.

The end event is usually still pending when the iteration's own output is
ready, so the finished timing leaves the worker as a record with a later reply:
the next step's output, the next dummy batch, or the flush EngineCore requests
before going idle.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from functools import wraps
from itertools import count
from typing import TYPE_CHECKING

import torch

from vllm.v1.core.sched.output import IterationDetails
from vllm.v1.metrics.stats import SchedulerIterationDetails
from vllm.v1.outputs import AsyncModelRunnerOutput, ModelRunnerOutput

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.outputs import KVConnectorOutput


def _record_event() -> torch.Event:
    event = torch.Event(enable_timing=True)
    event.record()
    return event


@dataclass
class _TimedIteration:
    index: int
    details: IterationDetails
    is_dummy: bool
    start: torch.Event
    end: torch.Event

    def read(self) -> SchedulerIterationDetails:
        """Read the elapsed time. Both events must have completed."""
        details = self.details
        return SchedulerIterationDetails(
            iteration_index=self.index,
            num_ctx_requests=details.num_ctx_requests,
            num_ctx_tokens=details.num_ctx_tokens,
            num_generation_requests=details.num_generation_requests,
            num_generation_tokens=details.num_generation_tokens,
            elapsed_ms=self.start.elapsed_time(self.end),
            num_encoder_inputs=details.num_encoder_inputs,
            num_encoder_output_tokens=details.num_encoder_output_tokens,
            is_dummy=self.is_dummy,
        )


class IterationTimer:
    """Times iterations on the calling thread's CUDA stream.

    Only the worker's main thread uses it, so it needs no lock, and it never
    waits for the GPU: `take_ready` returns what has completed, and `flush`
    hands the rest to the thread that already waits for step outputs.
    """

    def __init__(self) -> None:
        self._indices = count()
        self._started: tuple[IterationDetails, bool, torch.Event] | None = None
        self._pending: deque[_TimedIteration] = deque()

    def start(self, details: IterationDetails | None, is_dummy: bool = False) -> None:
        """Begin an iteration, dropping any that never finished.

        Without a scheduling snapshot the iteration is not timed.
        """
        self._started = (
            None if details is None else (details, is_dummy, _record_event())
        )

    def finish(self) -> None:
        if self._started is None:
            return
        details, is_dummy, start = self._started
        self._started = None
        self._pending.append(
            _TimedIteration(
                next(self._indices), details, is_dummy, start, _record_event()
            )
        )

    def take_ready(self) -> list[SchedulerIterationDetails]:
        """Pop the oldest iterations whose GPU work has completed."""
        records: list[SchedulerIterationDetails] = []
        while self._pending:
            # Timing must never fail a step: a failed read drops that record.
            with suppress(RuntimeError):
                if not self._pending[0].end.query():
                    break
                records.append(self._pending[0].read())
            self._pending.popleft()
        return records

    def flush(self) -> IterationTimingFlush:
        """Hand over every pending iteration, for the engine to wait on."""
        pending, self._pending = self._pending, deque()
        return IterationTimingFlush(pending)


class IterationTimingFlush(AsyncModelRunnerOutput):
    """The iterations still on the GPU when the engine goes idle.

    It resolves wherever a step's output resolves (the output thread with async
    scheduling), and only once the engine has nothing else to run.
    """

    def __init__(self, pending: deque[_TimedIteration]) -> None:
        self._pending = pending
        # Blocking (sleep) event, like AsyncOutput.copy_event. It is recorded
        # after every pending end event on the same stream, so it completes last.
        self._done = torch.cuda.Event(blocking=True)
        self._done.record()

    def get_output(self) -> ModelRunnerOutput:
        self._done.synchronize()
        records: list[SchedulerIterationDetails] = []
        for timed in self._pending:
            with suppress(RuntimeError):
                records.append(timed.read())
        return ModelRunnerOutput(req_ids=[], iteration_records=records)

    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        return None


class _WithIterationRecords(AsyncModelRunnerOutput):
    """A step's output that also carries completed iteration records."""

    def __init__(
        self,
        output: AsyncModelRunnerOutput,
        records: list[SchedulerIterationDetails],
    ) -> None:
        self._output = output
        self._records = records

    def get_output(self) -> ModelRunnerOutput:
        output = self._output.get_output()
        output.iteration_records = self._records
        return output

    def get_kv_connector_output(self) -> KVConnectorOutput | None:
        return self._output.get_kv_connector_output()


def start_iteration_timing(execute_model: Callable) -> Callable:
    """Decorates `Worker.execute_model` to record the iteration's start."""

    @wraps(execute_model)
    def wrapper(self, scheduler_output: SchedulerOutput, *args, **kwargs):
        if self.iteration_timer is not None:
            self.iteration_timer.start(scheduler_output.iteration_details)
        return execute_model(self, scheduler_output, *args, **kwargs)

    return wrapper


def finish_iteration_timing(sample_tokens: Callable) -> Callable:
    """Decorates `Worker.sample_tokens` to record the iteration's end.

    The returned output carries the iterations that have completed by then.
    """

    @wraps(sample_tokens)
    def wrapper(self, *args, **kwargs):
        output = sample_tokens(self, *args, **kwargs)
        timer = self.iteration_timer
        # Only a deferred output ends a timed iteration: sample_tokens returns
        # None when its execute_model failed.
        if timer is None or not isinstance(output, AsyncModelRunnerOutput):
            return output
        timer.finish()
        records = timer.take_ready()
        return _WithIterationRecords(output, records) if records else output

    return wrapper


def time_dummy_batch(execute_dummy_batch: Callable) -> Callable:
    """Decorates `Worker.execute_dummy_batch` to time the dummy iteration.

    Returns the iterations that have completed by then.
    """

    @wraps(execute_dummy_batch)
    def wrapper(self) -> list[SchedulerIterationDetails] | None:
        timer = self.iteration_timer
        if timer is None:
            return execute_dummy_batch(self)
        timer.start(IterationDetails(), is_dummy=True)
        execute_dummy_batch(self)
        timer.finish()
        return timer.take_ready()

    return wrapper


def make_iteration_timer(vllm_config: VllmConfig, rank: int) -> IterationTimer | None:
    """Only the worker whose output the engine reads times iterations."""
    parallel_config = vllm_config.parallel_config
    output_rank = parallel_config.world_size - (
        parallel_config.tensor_parallel_size
        * parallel_config.prefill_context_parallel_size
    )
    if vllm_config.gpu_iteration_timing and rank == output_rank:
        return IterationTimer()
    return None
