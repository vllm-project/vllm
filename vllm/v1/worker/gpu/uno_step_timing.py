# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Launch-gated Uno per-step timing diagnostics.

The tracer is observational. CUDA events measure queued GPU work without
synchronizing the serving thread; a daemon resolves each event after the step
completes and writes one JSON log record. Normal serving creates no tracer,
events, threads, or request counters.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput


logger = init_logger(__name__)


@dataclass
class UnoStepTimingTrace:
    """One real Uno worker step, retaining aggregate timing metadata only."""

    step_id: int
    schedule_wall_ms: float | None
    running_request_count: int
    scheduled_request_count: int
    request_steps: tuple[int, ...]
    capture_cuda_events: bool
    _wall_starts: dict[str, float] = field(default_factory=dict)
    wall_ms: dict[str, float] = field(default_factory=dict)
    _events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = field(
        default_factory=dict
    )
    _completion_event: torch.cuda.Event | None = None
    _finished: bool = False

    def start_wall(self, stage: str) -> None:
        self._wall_starts[stage] = time.perf_counter()

    def end_wall(self, stage: str) -> None:
        start = self._wall_starts.pop(stage, None)
        if start is not None:
            self.wall_ms[stage] = (time.perf_counter() - start) * 1000

    def start_gpu(self, stage: str) -> None:
        if not self.capture_cuda_events:
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        self._events[stage] = (start, end)

    def end_gpu(self, stage: str) -> None:
        events = self._events.get(stage)
        if events is not None:
            events[1].record()

    def start_stage(self, stage: str) -> None:
        self.start_wall(stage)
        self.start_gpu(stage)

    def end_stage(self, stage: str) -> None:
        self.end_gpu(stage)
        self.end_wall(stage)

    def finish(self) -> None:
        if self._finished:
            return
        self._finished = True
        if self.capture_cuda_events:
            self._completion_event = torch.cuda.Event()
            self._completion_event.record()

    def payload(self) -> dict[str, object]:
        gpu_ms = {
            stage: start.elapsed_time(end)
            for stage, (start, end) in self._events.items()
        }
        return {
            "step_id": self.step_id,
            "schedule_wall_ms": self.schedule_wall_ms,
            "prepare_inputs_wall_ms": self.wall_ms.get("prepare_inputs"),
            "model_forward_wall_ms": self.wall_ms.get("model_forward"),
            "sample_wall_ms": self.wall_ms.get("sample"),
            "output_publish_wall_ms": self.wall_ms.get("output_publish"),
            "propose_wall_ms": self.wall_ms.get("propose"),
            "model_forward_gpu_ms": gpu_ms.get("model_forward"),
            "sample_gpu_ms": gpu_ms.get("sample"),
            "output_publish_gpu_ms": gpu_ms.get("output_publish"),
            "propose_gpu_ms": gpu_ms.get("propose"),
            "running_request_count": self.running_request_count,
            "scheduled_request_count": self.scheduled_request_count,
            "request_steps": self.request_steps,
        }


class UnoStepTimingTracer:
    """Track the first three worker steps for each request in this process."""

    def __init__(self, *, capture_cuda_events: bool | None = None) -> None:
        self._request_steps: dict[str, int] = {}
        self._capture_cuda_events = (
            torch.cuda.is_available()
            if capture_cuda_events is None
            else capture_cuda_events
        )
        self._threads: list[threading.Thread] = []
        self._threads_lock = threading.Lock()

    def begin(
        self, scheduler_output: SchedulerOutput, running_count: int
    ) -> UnoStepTimingTrace | None:
        step_id = scheduler_output.debug_uno_step_id
        if step_id is None:
            return None
        request_steps = tuple(
            self._request_steps.setdefault(req_id, 0) + 1
            for req_id in scheduler_output.num_scheduled_tokens
        )
        for req_id, request_step in zip(
            scheduler_output.num_scheduled_tokens, request_steps
        ):
            self._request_steps[req_id] = request_step
        if not any(step <= 3 for step in request_steps):
            return None
        return UnoStepTimingTrace(
            step_id=step_id,
            schedule_wall_ms=scheduler_output.debug_schedule_wall_ms,
            running_request_count=running_count,
            scheduled_request_count=len(scheduler_output.num_scheduled_tokens),
            request_steps=request_steps,
            capture_cuda_events=self._capture_cuda_events,
        )

    def publish(self, trace: UnoStepTimingTrace) -> None:
        trace.finish()
        if trace._completion_event is None:
            self._log(trace)
            return
        thread = threading.Thread(target=self._wait_and_log, args=(trace,), daemon=True)
        with self._threads_lock:
            self._threads.append(thread)
        thread.start()

    def _wait_and_log(self, trace: UnoStepTimingTrace) -> None:
        assert trace._completion_event is not None
        trace._completion_event.synchronize()
        self._log(trace)

    @staticmethod
    def _log(trace: UnoStepTimingTrace) -> None:
        logger.info("UNO_STEP_TIMING %s", json.dumps(trace.payload(), sort_keys=True))

    def flush(self) -> None:
        """Wait for diagnostic writers only during worker shutdown."""
        with self._threads_lock:
            threads, self._threads = self._threads, []
        for thread in threads:
            thread.join()
