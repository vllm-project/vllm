# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import threading
import time
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import Locality, OffloadKey
from vllm.v1.kv_offload.tiering.base import JobId
from vllm.v1.kv_offload.tiering.fs.dispatch import (
    LoadQueue,
    StoreQueue,
    WorkDispatcher,
    make_batches,
)

logger = init_logger(__name__)


@dataclass
class Task:
    """
    I/O Task inputs
    """

    key: OffloadKey
    path: str
    offset: int


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = (
        "_job_id",
        "_n_tasks",
        "_completed",
        "_success",
        "_transfer_start",
        "_transfer_end",
        "_lock",
    )

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._transfer_start = float("inf")
        self._transfer_end = 0.0
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(
        self, batch_size: int, success: bool, start_time: float, end_time: float
    ) -> tuple[bool, bool, float]:
        """Returns (job_finished, success, transfer_time)."""
        with self._lock:
            self._completed += batch_size
            self._transfer_start = min(self._transfer_start, start_time)
            self._transfer_end = max(self._transfer_end, end_time)
            if not success:
                self._success = False
            transfer_time = self._transfer_end - self._transfer_start
            return self._completed == self._n_tasks, self._success, transfer_time


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    - Load-priority threads: drain the load queue first, steal stores when idle.
    - Store-priority threads: drain the store queue first, steal loads when idle.

    All groups share a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        block_size: int,
        locality: Locality,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._condition = threading.Condition(threading.Lock())
        self._idle_condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool, float]] = deque()
        self._inflight_jobs = 0  # guarded by _condition

        assert n_read_threads + n_write_threads > 0, (
            "Threadpool needs atleast on 1 rw thread"
        )

        self._dispatcher = WorkDispatcher(
            locality=locality,
            load_job_q=LoadQueue(block_size),
            store_job_q=StoreQueue(block_size),
            n_read_threads=n_read_threads,
            n_write_threads=n_write_threads,
        )

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

    def _enqueue(
        self,
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
        job_id: JobId,
        tasks: Iterable[Task],
        n_tasks: int,
        is_load: bool,
    ) -> None:
        """Pre-batch tasks outside the lock, then hand off to the scheduler."""
        if n_tasks == 0:
            self._finished_q.append((job_id, True, 0.0))
            return
        state = JobState(job_id, n_tasks)
        task_lst = list(tasks)
        assert len(task_lst) == n_tasks, "Unaccounted tasks"
        # Build batches before acquiring the lock; it is O(n_tasks)
        n_threads = self._dispatcher.n_batch_threads(is_load)
        work_items = make_batches(state, task_lst, make_batch_fn, n_threads)
        with self._condition:
            self._inflight_jobs += 1
            n_wake = self._dispatcher.submit(job_id, work_items, n_tasks, is_load)
            # TODO (varun): Wake threads based on load / store
            self._condition.notify(n_wake)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""

        self._enqueue(make_batch_fn, job_id, tasks, n_tasks=n_tasks, is_load=True)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
        make_batch_fn: Callable[[list[Task]], Callable[[], None]],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""

        self._enqueue(
            make_batch_fn,
            job_id,
            tasks,
            n_tasks=n_tasks,
            is_load=False,
        )

    def get_finished(self) -> list[tuple[JobId, bool, float]]:
        # No lock needed: deque is thread-safe for concurrent append/popleft,
        # and the manager is the sole popper.
        jobs = []
        while self._finished_q:
            jobs.append(self._finished_q.popleft())
        return jobs

    def wait_idle(self) -> None:
        """Block until there are no in-flight jobs.

        After this returns, every submitted job has had its last task
        finish, so no worker thread is still copying data. Note:
        completed jobs may still be sitting in ``_finished_q`` waiting
        for ``get_finished()`` to drain them.
        """
        with self._idle_condition:
            self._idle_condition.wait_for(lambda: self._inflight_jobs == 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._dispatcher.clear()
            # Cancelled tasks will not decrement _inflight_jobs; reset it so a
            # subsequent wait_idle() returns instead of hanging.
            self._inflight_jobs = 0
            self._condition.notify_all()
        with self._idle_condition:
            self._idle_condition.notify_all()
        if wait:
            for t in self._threads:
                t.join()

    def _worker(self, load_priority: bool) -> None:
        # Wait for tasks, drain primary queue first, steal from secondary when idle.
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._stop or self._dispatcher.has_work(load_priority)
                )
                if self._stop:
                    return
                work = self._dispatcher.fetch_work(load_priority)
                if work is None:
                    continue
                fn, batch_size, state = work
            try:
                start_time = time.monotonic()
                fn()
                end_time = time.monotonic()
                job_finished, success, total_time = state.task_done(
                    batch_size, True, start_time, end_time
                )
            except Exception as exc:
                end_time = time.monotonic()
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success, total_time = state.task_done(
                    batch_size, False, start_time, end_time
                )

            if job_finished:
                with self._condition:
                    self._finished_q.append((state.job_id, success, total_time))
                    self._inflight_jobs -= 1
                with self._idle_condition:
                    self._idle_condition.notify_all()
