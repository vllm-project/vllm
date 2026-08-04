# SPDX-License-Identifier: Apache-2.0
"""
Thread pool with affinity routing.

Tasks submitted with the same ``affinity_key`` always execute on the same
worker thread.  Within each worker, tasks execute sequentially in FIFO order.

This is used for GPU-bound request handlers (STORE / RETRIEVE) so that all
operations for a given vLLM instance land on one thread, eliminating the need
for per-instance locks on the shared temporary GPU buffer.
"""

# Standard
from concurrent.futures import Future
import queue
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Sentinel object to signal worker shutdown
_SHUTDOWN = object()


class AffinityThreadPool:
    """Thread pool that routes tasks to workers by affinity key.

    Not thread-safe: ``submit()`` must be called from a single thread (the
    ``MessageQueueServer`` main loop).

    Args:
        max_workers: Number of worker threads.
        thread_name_prefix: Prefix for worker thread names.
    """

    def __init__(
        self,
        max_workers: int,
        thread_name_prefix: str = "affinity",
    ) -> None:
        self._num_workers = max_workers
        self._queues: list[queue.Queue] = [queue.Queue() for _ in range(max_workers)]
        self._threads: list[threading.Thread] = []
        # Maps an affinity_key -> the worker slot (thread index) bound to it.
        self._key_to_slot: dict[int, int] = {}
        self._next_slot = 0
        self._overflow_warned = False
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker,
                args=(self._queues[i],),
                daemon=True,
                name=f"{thread_name_prefix}-{i}",
            )
            t.start()
            self._threads.append(t)

        logger.info(
            "Created AffinityThreadPool '%s' with %d worker slots: up to %d "
            "distinct affinity keys each bind to their own thread before slots "
            "are shared. Compare this against the number of clients expected to "
            "connect to confirm routing.",
            thread_name_prefix,
            max_workers,
            max_workers,
        )

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    @staticmethod
    def _worker(q: queue.Queue) -> None:
        while True:
            item = q.get()
            if item is _SHUTDOWN:
                break
            future, fn, args, kwargs = item
            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except BaseException as exc:
                    future.set_exception(exc)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _slot_for_key(self, affinity_key: int) -> int:
        """Return the worker slot bound to ``affinity_key``, assigning on first use.

        Returns:
            The worker slot (an index in ``[0, _num_workers)``) for the key.
        """
        slot = self._key_to_slot.get(affinity_key)
        if slot is not None:
            return slot

        # First time we see this key -- bind it to the next free slot.
        slot = self._next_slot % self._num_workers
        is_overflow = self._next_slot >= self._num_workers
        self._key_to_slot[affinity_key] = slot
        self._next_slot += 1

        logger.info(
            "AffinityThreadPool: affinity_key=%d assigned to worker "
            "slot %d of %d (thread %s); %d distinct key(s) now bound",
            affinity_key,
            slot,
            self._num_workers,
            self._threads[slot].name,
            len(self._key_to_slot),
        )
        if is_overflow and not self._overflow_warned:
            self._overflow_warned = True
            logger.warning(
                "AffinityThreadPool: affinity_key=%d wrapped onto worker slot "
                "%d (only %d workers), so it shares a thread with an earlier "
                "key and their tasks are serialized. Increase the worker count "
                "to give each client its own thread.",
                affinity_key,
                slot,
                self._num_workers,
            )
        return slot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(self, fn, *args, affinity_key: int = 0, **kwargs) -> Future:
        """Submit *fn* for execution on the worker bound to *affinity_key*.

        Returns a :class:`concurrent.futures.Future`.
        """
        future: Future = Future()
        slot = self._slot_for_key(affinity_key)
        self._queues[slot].put((future, fn, args, kwargs))
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the pool.

        Sends a shutdown sentinel to every worker.  If *wait* is true, blocks
        until all workers have exited.
        """
        for q in self._queues:
            q.put(_SHUTDOWN)
        if wait:
            for t in self._threads:
                t.join()
