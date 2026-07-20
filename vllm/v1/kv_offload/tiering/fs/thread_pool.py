# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Thread pool:
    Two queues (load, store) and two sets of threads:
      - Load-priority threads: drain the load queue first, then the store queue.
      - Store-priority threads: drain the store queue first, then the load queue.
    Load jobs are enqueued to the load queue; store jobs to the store queue.
"""

import functools
import threading
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass

from vllm.logger import init_logger
from vllm.v1.kv_offload.tiering.base import JobId
from vllm.v1.kv_offload.tiering.fs.io import batch_load_block, batch_store_block

logger = init_logger(__name__)


@dataclass
class Task:
    """
    I/O Task inputs
    """

    path: str
    view: memoryview
    offset: int
    block_size: int


class JobState:
    """
    Thread-safe completion tracker for a set of per-block I/O tasks.

    Each task calls task_done(success) when it finishes.
    """

    __slots__ = ("_job_id", "_n_tasks", "_completed", "_success", "_lock")

    def __init__(self, job_id: JobId, n_tasks: int) -> None:
        self._job_id: JobId = job_id
        self._n_tasks = n_tasks
        self._completed = 0
        self._success = True
        self._lock = threading.Lock()

    @property
    def job_id(self) -> JobId:
        return self._job_id

    def task_done(self, batch_size: int, success: bool) -> tuple[bool, bool]:
        """Returns if job completed and success flag"""
        with self._lock:
            self._completed += batch_size
            if not success:
                self._success = False
            return self._completed == self._n_tasks, self._success


class DualQueueThreadPool:
    """
    Thread pool with two task queues (load and store) and two thread groups.

    Load-priority threads drain the load queue first, then fall back to the
    store queue.  Store-priority threads do the reverse.  Both queues share
    a single condition variable.
    """

    def __init__(
        self,
        n_read_threads: int,
        n_write_threads: int,
        rw_batch_size: int,
        thread_name_prefix: str = "fs_secondary_tier",
    ) -> None:
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._condition = threading.Condition(threading.Lock())
        self._stop = False
        self._threads: list[threading.Thread] = []
        self._finished_q: deque[tuple[JobId, bool]] = deque()
        self._inflight_jobs = 0  # guarded by _condition

        self.rw_batch_size = rw_batch_size
        assert self.rw_batch_size > 0, (
            f"read/write batch size {self.rw_batch_size} must be greater than 0"
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

    def _batch_tasks(self, tasks: Iterable[Task]) -> Iterator[list[Task]]:
        """Chunk tasks into lists of at most `rw_batch_size` tasks."""
        batch: list[Task] = []
        for task in tasks:
            batch.append(task)
            assert batch[0].view is task.view
            assert batch[0].block_size == task.block_size
            if len(batch) >= self.rw_batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    def _enqueue(
        self,
        queue: deque,
        make_fn: Callable[[list[Task]], Callable[[], None]],
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
    ) -> None:
        """Batch `tasks` and append (fn, state, batch_size) entries to `queue`."""
        if n_tasks == 0:
            self._finished_q.append((job_id, True))
            return
        state = JobState(job_id, n_tasks)
        n_batches = 0
        with self._condition:
            self._inflight_jobs += 1
            for batch in self._batch_tasks(tasks):
                queue.append((make_fn(batch), state, len(batch)))
                n_batches += 1
            self._condition.notify(n_batches)

    def enqueue_load(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
    ) -> None:
        """Enqueue load tasks for a job (high-priority for load-priority threads)."""

        def make_fn(batch: list[Task]) -> Callable[[], None]:
            return functools.partial(
                batch_load_block,
                paths=[t.path for t in batch],
                view=batch[0].view,
                offsets=[t.offset for t in batch],
                block_size=batch[0].block_size,
            )

        self._enqueue(self._load_q, make_fn, job_id, n_tasks, tasks)

    def enqueue_store(
        self,
        job_id: JobId,
        n_tasks: int,
        tasks: Iterable[Task],
    ) -> None:
        """Enqueue store tasks for a job (high-priority for store-priority threads)."""

        def make_fn(batch: list[Task]) -> Callable[[], None]:
            return functools.partial(
                batch_store_block,
                paths=[t.path for t in batch],
                view=batch[0].view,
                offsets=[t.offset for t in batch],
                block_size=batch[0].block_size,
            )

        self._enqueue(self._store_q, make_fn, job_id, n_tasks, tasks)

    def get_finished(self) -> list[tuple[JobId, bool]]:
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
        with self._condition:
            self._condition.wait_for(lambda: self._inflight_jobs == 0)

    def shutdown(self, wait: bool = True) -> None:
        with self._condition:
            self._stop = True
            self._load_q.clear()
            self._store_q.clear()
            # Cancelled tasks will not decrement _inflight_jobs; reset it so a
            # subsequent wait_idle() returns instead of hanging.
            self._inflight_jobs = 0
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
                fn, state, batch_size = (
                    primary.popleft() if primary else secondary.popleft()
                )
            try:
                fn()
                job_finished, success = state.task_done(batch_size, True)
            except Exception as exc:
                logger.error(
                    "Job %s block I/O failed: %s",
                    state.job_id,
                    exc,
                )
                job_finished, success = state.task_done(batch_size, False)

            if job_finished:
                with self._condition:
                    self._finished_q.append((state.job_id, success))
                    self._inflight_jobs -= 1
                    self._condition.notify_all()
