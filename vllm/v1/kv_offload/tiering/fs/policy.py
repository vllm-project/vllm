# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import enum
import heapq
import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Callable, Iterator
from typing import Any

from vllm.v1.kv_offload.base import Locality
from vllm.v1.kv_offload.tiering.base import JobId


class ThreadMode(enum.Enum):
    """Operating mode for a pool worker thread."""

    READ = "read"
    """Load-priority: serve load jobs first, steal store jobs when idle."""

    WRITE = "write"
    """Store-priority: serve store jobs first, steal load jobs when idle."""

    WRITE_EXCL = "write_excl"
    """Store-exclusive: serve store jobs only, never steal from the load queue."""


class JobQueue(ABC):
    def __init__(self, block_size: int):
        self._block_size = block_size

    @abstractmethod
    def put(self, job_id: JobId, num_tasks: int):
        pass

    @abstractmethod
    def get(self) -> JobId | None:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def maybe_has_work(self):
        pass


# put tuples (job_id, num_tasks) as and when they arrive.
# On get(), return job in FCFS order.
class FCFSQueue(JobQueue):
    def __init__(self, block_size: int):
        super().__init__(block_size)
        self.q: deque[JobId] = deque()

    def put(self, job_id: JobId, num_tasks: int):
        self.q.append(job_id)

    def get(self) -> JobId | None:
        return self.q.popleft() if self.q else None

    def clear(self):
        self.q.clear()

    def maybe_has_work(self):
        return bool(self.q)


class SJFHeapQueue(JobQueue):
    """Min-heap queue that returns the shortest job (fewest tasks) first."""

    def __init__(self, block_size: int):
        super().__init__(block_size)
        # heap entries: (num_tasks, job_id)
        self._heap: list[tuple[int, JobId]] = []

    def put(self, job_id: JobId, num_tasks: int):
        heapq.heappush(self._heap, (num_tasks, job_id))

    def get(self) -> JobId | None:
        if not self._heap:
            return None
        _, job_id = heapq.heappop(self._heap)
        return job_id

    def clear(self):
        self._heap.clear()

    def maybe_has_work(self):
        return bool(self._heap)


# put tuples (job_id, num_tasks) as and when they arrive.
# On get(), return the FCFS job in the minimum bucket
class SJFBucketQueue(JobQueue):
    """
    Assign incoming jobs into buckets based on the size of the job.
    When queried, determine if the current workload and multi-modal,
    i.e. could be classified in to short / long jobs, and return a
    job from the short bucket.
    - Advantages of a bucket approach:
        * Maintain FCFS within each bucket. 2 jobs with similar
        num_tasks are executed in FCFS order.
        * Allows adding / removing elements in constant time.
        * It also allows to reason about the multi-modality of
          the current workload. For example,
           - if only one bucket is filled. Then it is uni-modal
           - if contiguous buckets are filled, then it is somewhat bi-modal
           - if 2 non contiguous buckets are filled, the it definitely
             bi-modal.

    Implementation:
        - Bucket boundaries grow exponentially (log₂ of job size in bytes).
          This provides high resolution where it matters most — distinguishing
          small jobs — while coarsely grouping jobs at the large end.
    """

    def __init__(self, block_size: int):
        super().__init__(block_size)
        # bitmask indicating if a bucket has jobs
        self.mask = ctypes.c_uint32(0x00000000)

        self.num_buckets = ctypes.sizeof(self.mask) * 8
        self.q: list[FCFSQueue] = [
            FCFSQueue(block_size) for _ in range(self.num_buckets)
        ]

    def _get_bucket_id(self, job_id: JobId, num_tasks: int):
        assert num_tasks != 0
        mbytes = max(1, num_tasks * self._block_size)
        return min(self.num_buckets - 1, int(math.floor(math.log2(mbytes))))

    def _get_sjf_bucket(self):
        x = self.mask.value
        assert x != 0
        # ~v inverts the bits.
        # + 1 completes the two's complement.
        # & ones (0xFFFFFFFF) forces it to stay within 32-bit unsigned bounds.
        ones = (1 << self.num_buckets) - 1
        pow2 = x & ((~x + 1) & ones)
        return pow2.bit_length() - 1

    def put(self, job_id: JobId, num_tasks: int):
        bid = self._get_bucket_id(job_id, num_tasks)
        self.q[bid].put(job_id, num_tasks)
        self.mask.value |= 1 << bid

    def get(self) -> JobId | None:
        if not self.maybe_has_work():
            return None
        bid = self._get_sjf_bucket()
        job_id = self.q[bid].get()
        assert job_id is not None
        if not self.q[bid].maybe_has_work():
            self.mask.value &= ~(1 << bid)
        return job_id

    def clear(self):
        for x in self.q:
            x.clear()
        self.mask = ctypes.c_uint32(0x00000000)

    def maybe_has_work(self):
        return self.mask.value != 0


class LoadQueue(JobQueue):
    # For loads, we care about both,
    # 1. FCFS - For better TTFT, and
    # 2. SJF  - For clearing CPU KV cache.
    # As a result we alternate between the FCFS and SJF queues
    # for get().

    def __init__(self, block_size: int):
        super().__init__(block_size)
        self.fcfs: FCFSQueue = FCFSQueue(block_size)
        self.sjf: SJFBucketQueue = SJFBucketQueue(block_size)
        # ping-ping index
        self.pp = 0

        # set of jobs in the queues.
        self.jobs: set[JobId] = set()

    def put(self, job_id: JobId, num_tasks: int):
        self.fcfs.put(job_id, num_tasks)
        self.sjf.put(job_id, num_tasks)
        self.jobs.add(job_id)

    def get(self) -> JobId | None:
        def get_job(job_q: JobQueue):
            while (jid := job_q.get()) is not None:
                if jid in self.jobs:
                    break
            return jid

        q, fallback = (self.fcfs, self.sjf) if self.pp == 0 else (self.sjf, self.fcfs)

        if (jid := get_job(q)) is None:
            jid = get_job(fallback)

        if jid is None:
            return None

        self.pp = 1 if self.pp == 0 else 0
        self.jobs.remove(jid)
        return jid

    def maybe_has_work(self):
        return self.fcfs.maybe_has_work() or self.sjf.maybe_has_work()

    def clear(self):
        self.fcfs.clear()
        self.sjf.clear()
        self.pp = 0
        self.jobs.clear()


StoreQueue = SJFBucketQueue


class Scheduler:
    def __init__(
        self,
        locality: Locality,
        load_job_q: JobQueue,
        store_job_q: JobQueue,
        n_read_threads: int,
        n_write_threads: int,
    ):
        self._locality = locality
        self._load_job_q = load_job_q
        self._store_job_q = store_job_q
        self._n_read_threads = n_read_threads
        self._n_write_threads = n_write_threads

        if self._locality == Locality.LOCAL:
            # Assume SSD
            self._n_read_batch_threads = n_read_threads or self.total_threads
            # Limit concurrent SSD writes to 1 thread. When a NAND die is
            # busy with a write, any read to that die stalls until the write
            # completes. More write threads means more dies occupied at once,
            # directly hurting read latency.
            # NOTE: We can make this configurable in the future if needed.
            self._n_write_batch_threads = 1
        else:
            # Assume NFS
            self._n_read_batch_threads = n_read_threads or self.total_threads
            self._n_write_batch_threads = n_write_threads or self.total_threads

        # Read threads may always steal store jobs — stores are short and
        # bounded by the engine forward pass.
        self._read_threads_can_write = True

        # Store threads steal from the load queue in small quanta so they
        # can check back frequently for new store work.  The quanta is the
        # running-mean store job size, updated on every store submission.
        # When no read threads exist, write threads handle full load batches
        # (no quanta restriction).  Stealing is gated on _avg_store_tasks > 0:
        # until the first store job lands there is no calibrated quanta.
        self._write_threads_can_read = True

        # Running mean of store job sizes — used as the steal quanta.
        self._avg_store_tasks: float = 0.0
        self._n_store_jobs: int = 0

        # Work queues that the threads draw work from
        self._load_q: deque = deque()
        self._store_q: deque = deque()
        self._jobs: dict[JobId, Any] = {}

    @property
    def total_threads(self):
        return self._n_read_threads + self._n_write_threads

    def submit(
        self,
        job_id: JobId,
        pre_batched: list[tuple],
        n_tasks: int,
        is_load: bool,
    ) -> int:
        """Register a pre-batched job and return the number of threads to wake.

        pre_batched must be built by make_batches() outside the lock before
        calling submit() under the lock.
        """
        self._jobs[job_id] = pre_batched
        if is_load:
            self._load_job_q.put(job_id, n_tasks)
        else:
            self._store_job_q.put(job_id, n_tasks)
            # Update running mean — used as the steal quanta for write threads.
            self._n_store_jobs += 1
            self._avg_store_tasks += (
                n_tasks - self._avg_store_tasks
            ) / self._n_store_jobs

        # TODO(varun): unfortunate! - wake selectively
        return self._n_read_threads + self._n_write_threads

    def has_work(self, mode: ThreadMode) -> bool:
        has_load_work = bool(self._load_q or self._load_job_q.maybe_has_work())
        has_store_work = bool(self._store_q or self._store_job_q.maybe_has_work())
        if mode is ThreadMode.WRITE_EXCL:
            return has_store_work
        return has_load_work or has_store_work

    def _batch_tasks(
        self,
        tasks: list[Any],
        n_threads: int,
    ) -> Iterator[list[Any]]:
        """
        Batch tasks so that the request's tasks are split evenly across the
        n_threads.
        """
        assert n_threads > 0

        n_tasks = len(tasks)
        q, r = divmod(n_tasks, n_threads)
        batch_sizes = [q + 1 if i < r else q for i in range(n_threads)]
        assert sum(batch_sizes) == n_tasks
        start = 0
        for bs in batch_sizes[: min(n_tasks, n_threads)]:
            yield tasks[start : start + bs]
            start += bs

    def make_batches(
        self,
        state: Any,
        tasks: list[Any],
        make_batch_fn: Callable,
        is_load: bool,
    ) -> list[tuple]:
        """Build work-queue-ready items from raw tasks.

        Must be called outside the condition lock — list-slicing and
        make_batch_fn closure construction are O(n_tasks) and must not
        block other threads waiting on the condition variable.
        """
        n_threads = (
            self._n_read_batch_threads if is_load else self._n_write_batch_threads
        )
        return [
            (make_batch_fn, make_batch_fn(b), b, state)
            for b in self._batch_tasks(tasks, n_threads)
        ]

    def _maybe_populate_work_q(self, work_q: deque, job_q: JobQueue):
        """Move the next job's pre-batched items into work_q.

        O(n_batch_threads) deque appends — no batching or closure
        construction under the lock.
        """
        if work_q:
            return
        job_id = job_q.get()
        if job_id is None:
            return
        work_q.extend(self._jobs.pop(job_id))

    @staticmethod
    def _unpack(item: tuple) -> tuple:
        """Unpack a work-queue item into the (fn, batch_size, state) the
        worker expects."""
        _, fn, tasks, state = item
        return fn, len(tasks), state

    def _steal_from_load_q(self) -> tuple:
        """Pop the head of _load_q and return a quanta-sized work item.

        The quanta is the running-mean store job size.  When no store jobs
        have been observed yet (avg == 0), the full batch is taken.  If the
        batch is larger than the quanta, the remainder is pushed back to the
        front of _load_q for read threads (or future steal calls) to pick up.
        """
        make_batch_fn, fn, tasks, state = self._load_q.popleft()
        quanta = int(self._avg_store_tasks) if self._avg_store_tasks > 0 else len(tasks)
        if len(tasks) <= quanta:
            return fn, len(tasks), state
        # Split: steal the first `quanta` tasks, push remainder back to front.
        steal_fn = make_batch_fn(tasks[:quanta])
        remainder = tasks[quanta:]
        self._load_q.appendleft(
            (make_batch_fn, make_batch_fn(remainder), remainder, state)
        )
        return steal_fn, quanta, state

    def fetch_work(self, mode: ThreadMode):
        if mode is ThreadMode.READ:
            self._maybe_populate_work_q(self._load_q, self._load_job_q)
            if self._load_q:
                return self._unpack(self._load_q.popleft())
            self._maybe_populate_work_q(self._store_q, self._store_job_q)
            if self._store_q:
                return self._unpack(self._store_q.popleft())
            return None
        elif mode is ThreadMode.WRITE:
            self._maybe_populate_work_q(self._store_q, self._store_job_q)
            if self._store_q:
                return self._unpack(self._store_q.popleft())
            self._maybe_populate_work_q(self._load_q, self._load_job_q)
            if self._load_q:
                return self._steal_from_load_q()
            return None
        else:  # WRITE_EXCL
            self._maybe_populate_work_q(self._store_q, self._store_job_q)
            if self._store_q:
                return self._unpack(self._store_q.popleft())
            return None

    def clear(
        self,
    ):
        self._load_job_q.clear()
        self._store_job_q.clear()
        self._load_q.clear()
        self._store_q.clear()
        self._jobs.clear()
        self._avg_store_tasks = 0.0
        self._n_store_jobs = 0
