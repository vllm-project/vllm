# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections import deque
from types import SimpleNamespace

from vllm.model_executor.offloader.prefetch_tail_copy import (
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
