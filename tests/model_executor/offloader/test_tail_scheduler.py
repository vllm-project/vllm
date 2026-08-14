# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import threading
from collections import deque
from contextlib import contextmanager
from types import SimpleNamespace

import torch

from vllm.distributed import parallel_state
from vllm.model_executor.offloader import base as offloader_base
from vllm.model_executor.offloader.prefetch_tail_copy import (
    TailCopyScheduler,
    pop_next_ready_tail_copy_job,
    requeue_active_tail_copy_job,
)


class _FakeForkEvent:
    def __init__(self, ready: bool):
        self.ready = ready
        self.query_count = 0

    def query(self) -> bool:
        self.query_count += 1
        return self.ready


def _job(name: str, *, ready: bool):
    return SimpleNamespace(name=name, fork_event=_FakeForkEvent(ready))


def test_tail_scheduler_skips_unready_jobs_without_reordering_pending_work():
    job_a = _job("a", ready=False)
    job_b = _job("b", ready=True)
    jobs = deque([job_a, job_b])

    selected = pop_next_ready_tail_copy_job(jobs)

    assert selected is job_b
    assert list(jobs) == [job_a]
    assert job_a.fork_event.query_count == 1
    assert job_b.fork_event.query_count == 1


def test_tail_scheduler_preserves_jobs_when_none_are_ready():
    job_a = _job("a", ready=False)
    job_b = _job("b", ready=False)
    jobs = deque([job_a, job_b])

    selected = pop_next_ready_tail_copy_job(jobs)

    assert selected is None
    assert list(jobs) == [job_a, job_b]


def test_tail_scheduler_picks_first_ready_job_in_queue_order():
    job_a = _job("a", ready=True)
    job_b = _job("b", ready=True)
    jobs = deque([job_a, job_b])

    selected = pop_next_ready_tail_copy_job(jobs)

    assert selected is job_a
    assert list(jobs) == [job_b]


def test_tail_scheduler_requeues_active_job_before_other_ready_jobs():
    job_a = _job("a", ready=True)
    job_b = _job("b", ready=True)
    jobs = deque([job_b])

    requeue_active_tail_copy_job(jobs, job_a)

    assert list(jobs) == [job_a, job_b]


class _FakeCollectiveEvent:
    def __init__(self, ready=False):
        self.ready = ready
        self.recorded_stream = None

    def record(self, stream) -> None:
        self.recorded_stream = stream

    def query(self) -> bool:
        return self.ready


def _collective_scheduler():
    scheduler = TailCopyScheduler.__new__(TailCopyScheduler)
    scheduler._condition = threading.Condition()
    scheduler._collective_windows_by_stream = {}
    return scheduler


def test_collective_gate_ignores_future_gpu_work():
    scheduler = _collective_scheduler()
    stream = SimpleNamespace(cuda_stream=7)
    start_event = _FakeCollectiveEvent(ready=False)
    done_event = _FakeCollectiveEvent(ready=False)

    scheduler.register_collective_window(stream, start_event, done_event)

    with scheduler._condition:
        assert not scheduler._collective_is_active_locked()
    assert 7 in scheduler._collective_windows_by_stream


def test_collective_gate_pauses_only_between_start_and_done_events():
    scheduler = _collective_scheduler()
    stream = SimpleNamespace(cuda_stream=7)
    start_event = _FakeCollectiveEvent(ready=True)
    done_event = _FakeCollectiveEvent(ready=False)

    scheduler.register_collective_window(stream, start_event, done_event)

    with scheduler._condition:
        assert scheduler._collective_is_active_locked()

    done_event.ready = True
    with scheduler._condition:
        assert not scheduler._collective_is_active_locked()
    assert scheduler._collective_windows_by_stream == {}


def test_collective_gate_observes_active_work_on_each_stream():
    scheduler = _collective_scheduler()
    future_stream = SimpleNamespace(cuda_stream=7)
    active_stream = SimpleNamespace(cuda_stream=8)
    scheduler.register_collective_window(
        future_stream,
        _FakeCollectiveEvent(ready=False),
        _FakeCollectiveEvent(ready=False),
    )
    scheduler.register_collective_window(
        active_stream,
        _FakeCollectiveEvent(ready=True),
        _FakeCollectiveEvent(ready=False),
    )

    with scheduler._condition:
        assert scheduler._collective_is_active_locked()


def test_collective_context_records_start_and_done_on_execution_stream(monkeypatch):
    scheduler = _collective_scheduler()
    stream = SimpleNamespace(cuda_stream=7)
    events = [_FakeCollectiveEvent(), _FakeCollectiveEvent()]
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: stream)
    monkeypatch.setattr(torch.cuda, "Event", lambda: events.pop(0))

    with scheduler.gate_for_collective():
        assert not scheduler._collective_windows_by_stream

    window = scheduler._collective_windows_by_stream[7][0]
    assert window.start_event.recorded_stream is stream
    assert window.done_event.recorded_stream is stream


def test_collective_gate_only_wraps_tp_when_enabled(monkeypatch):
    calls = []

    @contextmanager
    def gate_h2d_for_collective():
        calls.append("enter")
        yield
        calls.append("exit")

    monkeypatch.setattr(
        offloader_base,
        "_instance",
        SimpleNamespace(
            gates_collectives=True,
            gate_h2d_for_collective=gate_h2d_for_collective,
        ),
    )

    with parallel_state._prefetch_collective_context("tp:0"):
        calls.append("tp")
    with parallel_state._prefetch_collective_context("ep:0"):
        calls.append("ep")

    assert calls == ["enter", "tp", "exit", "ep"]


def test_collective_gate_skipped_when_offloader_does_not_gate(monkeypatch):
    """A non-gating offloader must not be entered even on a TP group."""
    calls = []

    @contextmanager
    def gate_h2d_for_collective():
        calls.append("enter")
        yield
        calls.append("exit")

    monkeypatch.setattr(
        offloader_base,
        "_instance",
        SimpleNamespace(
            gates_collectives=False,
            gate_h2d_for_collective=gate_h2d_for_collective,
        ),
    )

    with parallel_state._prefetch_collective_context("tp:0"):
        calls.append("tp")

    assert calls == ["tp"]
