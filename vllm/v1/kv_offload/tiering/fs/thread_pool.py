# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Thread pool:
Two queues (load, store) and two sets of threads:
- Load-priority threads: drain the load queue first, then the store queue.
- Store-priority threads: drain the store queue first, then the load queue.
Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId, JobTimeoutStatus

logger = init_logger(__name__)


class JobState:
    """Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_transfer_time",
        "_lock",
        "started",
        "canceled",
    )

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._transfer_time = 0.0
        self._lock = threading.Lock()
        # Guarded by the pool condition, alongside queue membership.
        self.started = False
        self.canceled = False

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(
        self, success: bool, transfer_time: float
    ) -> tuple[bool, bool, float]:
        """Returns if job completed and success flag."""
        with self._lock:
            self._completed += 1
            self._transfer_time += transfer_time
            if not success:
                self._success = False
            return self._completed == self._n_tasks, self._success, self._transfer_time


class DualQueueThreadPool:
    """Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues share
    a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: dict[JobId, tuple[bool, float, float]] = {}
        self._jobs: dict[JobId, JobState] = {}
        self._inflight_jobs = 0  # guarded by _condition

        for i in range(n_read_threads):
            t = threading.Thread(
                target=self._worker,
                args=(True,),
                name=f"{thread_name_prefix}_l{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

        for i in range(n_write_threads):
            t = threading.Thread(
                target=self._worker,
                args=(False,),
                name=f"{thread_name_prefix}_s{i}",
                daemon=True,
            )
            t.start()
            self._threads.append(t)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""
        state = JobState(job_id, n_tasks)
        with self._condition:
            if self._stop:
                raise RuntimeError("Cannot enqueue I/O after shutdown")
            self._jobs[job_id] = state
            self._inflight_jobs += 1
            for fn in tasks:
                self._load_q.append((fn, state))
            self._condition.notify(n_tasks)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Callable],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""
        state = JobState(job_id, n_tasks)
        with self._condition:
            if self._stop:
                raise RuntimeError("Cannot enqueue I/O after shutdown")
            self._jobs[job_id] = state
            self._inflight_jobs += 1
            for fn in tasks:
                self._store_q.append((fn, state))
            self._condition.notify(n_tasks)

    def get_finished(self) -> list[tuple[JobId, bool, float, float]]:
        with self._condition:
            jobs = [(job_id, *result) for job_id, result in self._finished_q.items()]
            for job_id in self._finished_q:
                del self._jobs[job_id]
            self._finished_q.clear()
            return jobs

    def resolve_timeout(self, job_id: JobId) -> JobTimeoutStatus:
        """Check for a completion or cancel a job that has not started."""
        with self._condition:
            if job_id in self._finished_q:
                return JobTimeoutStatus.COMPLETED
            state = self._jobs.get(job_id)
            if state is None or state.started:
                return JobTimeoutStatus.RUNNING
            # Workers skip these tasks before calling them. Keep queue entries
            # until then so cancellation is O(1), even with a stalled pool.
            state.canceled = True
            del self._jobs[job_id]
            self._inflight_jobs -= 1
            self._condition.notify_all()
            return JobTimeoutStatus.CANCELED

    def wait_idle(self) -> None:
        """Block until there are no in-flight jobs.

        After this returns, every task has finished or been canceled before
        starting, so no worker thread is still copying data. Note:
        completed jobs may still be sitting in ``_finished_q`` waiting
        for ``get_finished()`` to drain them.
        """
        with self._condition:
            self._condition.wait_for(lambda: self._inflight_jobs == 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            for queue in (self._load_q, self._store_q):
                while queue:
                    _, state = queue.popleft()
                    if state.canceled:
                        continue
                    finished, _, total_time = state.task_done(False, 0.0)
                    if finished:
                        self._finished_q[state.job_id] = (
                            False,
                            total_time,
                            time.monotonic(),
                        )
                        self._inflight_jobs -= 1
            self._condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
        # Wait for tasks, process from primary queue first, fall back to secondary.
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._load_q or self._store_q
                )
                if self._stop:
                    return
                primary = self._load_q if load_priority else self._store_q
                secondary = self._store_q if load_priority else self._load_q
                task, state = primary.popleft() if primary else secondary.popleft()
                if state.canceled:
                    del task
                    continue
                state.started = True
            try:
                start_time = time.monotonic()
                task()
                transfer_time = time.monotonic() - start_time
                job_finished, success, total_time = state.task_done(True, transfer_time)
            except Exception as exc:
                transfer_time = time.monotonic() - start_time
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success, total_time = state.task_done(
                    False, transfer_time
                )

            if job_finished:
                with self._condition:
                    completed_at = time.monotonic()
                    self._finished_q[state.job_id] = (success, total_time, completed_at)
                    self._inflight_jobs -= 1
                    self._condition.notify_all()
            del task
