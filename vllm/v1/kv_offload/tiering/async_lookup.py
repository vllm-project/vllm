# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AsyncLookupManager: per-tier async lookup manager for secondary tier
existence checks.

Each secondary tier that wants non-blocking lookups composes its own
AsyncLookupManager instance internally.  The manager maintains lookup
state and uses a background thread to execute batch_lookup() calls.

Locking design
--------------
There is no explicit lock.  Thread safety is achieved by ownership:

* _lookup_state and _lookup_batch are owned exclusively by the scheduler
  thread.  query(), flush(), and update_cached_exists() read and write
  them directly.

* _lookup_queue is written by the scheduler (flush → put_nowait, one item
  per step) and read by the background thread (get).  queue.Queue is
  thread-safe.

* _pending_results is written by the background thread (put) and read by
  the scheduler (get_nowait inside drain_results).  queue.SimpleQueue is
  thread-safe by design.

query() accumulates new keys in _lookup_batch without touching the queue.
flush() is called once per step from the tier's on_schedule_end(), posting
the entire batch as a single queue item so the background thread sees one
batch per step.
drain_results() is called before any query() calls in the same step, so
query() is a pure OrderedDict operation.
"""

import queue
import threading
from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Collection

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)

# Status values stored in _lookup_state and returned by query().
_IN_FLIGHT: int = 0  # queued or currently being looked up by the worker
NOT_FOUND: int = -1  # not present in this tier
FOUND: int = 1  # present in this tier


class AsyncLookupManager(ABC):
    """
    Per-tier async lookup manager for secondary tier existence checks.

    Each secondary tier that wants non-blocking lookups composes its own
    AsyncLookupManager instance internally. The manager maintains lookup
    state (cache, queue) and uses a background thread to execute the actual
    batch_lookup() calls.

    Subclasses implement only batch_lookup() — all queue management,
    state tracking, and result delivery is provided by this base class.

    The owning tier delegates its lookup(), on_schedule_end(), and
    get_finished_jobs() to this manager:
      - lookup() → drain_results() + query()
      - on_schedule_end() → flush()
      - get_finished_jobs() → drain_results() + update_cached_exists()
    """

    def __init__(
        self,
        tier_type: str,
        max_results: int = 1_000_000,
    ) -> None:
        self._tier_type = tier_type
        self._max_results = max_results

        # key → status; scheduler-owned, no lock needed.
        self._lookup_state: OrderedDict[OffloadKey, int] = OrderedDict()

        # Accumulates (key, req_context) pairs during query() calls.
        # Flushed as one queue item per step by flush().
        self._lookup_batch: list[tuple[OffloadKey, ReqContext]] = []

        # Scheduler → worker: one full step's batch per item.
        # None is used as a shutdown sentinel.
        self._lookup_queue: queue.SimpleQueue[
            list[tuple[OffloadKey, ReqContext]] | None
        ] = queue.SimpleQueue()

        # Worker → scheduler: completed result batches.
        # Each item is a list of (key, status) pairs.
        # SimpleQueue is explicitly thread-safe for one writer / one reader.
        self._pending_results: queue.SimpleQueue[list[tuple[OffloadKey, int]]] = (
            queue.SimpleQueue()
        )

        self._pending_drain_count: int = 0

        self._thread = threading.Thread(
            target=self._worker,
            name=f"vllm_offloading_lookup_{tier_type}",
            daemon=True,
        )
        self._thread.start()

    @abstractmethod
    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> list[bool | None]:
        """
        Check whether a batch of blocks exist in this tier.

        Called from the worker thread — must be synchronous and must not
        touch the primary tier or scheduler state.

        Returns a list parallel to keys: True if present, False if not
        found, None if the tier is busy (retry later).
        """
        ...

    # ------------------------------------------------------------------
    # Scheduler-thread API
    # ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> int | None:
        """
        Non-blocking lookup called from the scheduler thread.

        drain_results() must have been called earlier in the same step.

        Returns:
            FOUND     — block is present in this tier.
            NOT_FOUND — block is not present in this tier.
            None      — result not yet available; retry next step.
        """
        status = self._lookup_state.get(key, _IN_FLIGHT)
        if status == _IN_FLIGHT:
            if key not in self._lookup_state:
                # New key — buffer for async lookup; flushed by flush().
                self._evict_if_full()
                self._lookup_state[key] = _IN_FLIGHT
                self._lookup_batch.append((key, req_context))
            return None
        return status  # FOUND or NOT_FOUND

    def flush(self) -> None:
        """Post this step's accumulated keys to the worker thread.

        Called once per step from on_schedule_end() after all query() calls
        are done. The worker receives the full batch and processes it during
        the model-execution window, maximising time available before the next
        step's drain_results().  Safe to call with an empty batch (no-op).
        """
        if self._lookup_batch:
            self._lookup_queue.put(self._lookup_batch)
            self._lookup_batch = []
            self._pending_drain_count += 1

    def drain_results(self) -> None:
        """Apply pending worker results to _lookup_state.

        Called once per step from the tier's get_finished_jobs() before
        update_cached_exists() calls.  Applying worker results first ensures
        that a subsequent update_cached_exists() can correctly upgrade a
        worker NOT_FOUND to FOUND when a store completes in the same step.
        """
        if not self._pending_drain_count:
            return
        while True:
            try:
                batch = self._pending_results.get_nowait()
            except queue.Empty:
                break
            self._pending_drain_count -= 1
            for key, status in batch:
                # Only apply if the key is still _IN_FLIGHT.  A FOUND written
                # by update_cached_exists() during the worker's lookup takes
                # precedence and must not be overwritten.
                if self._lookup_state.get(key, _IN_FLIGHT) == _IN_FLIGHT:
                    self._lookup_state[key] = status

    def update_cached_exists(self, keys: Collection[OffloadKey]) -> None:
        """Populate the cache for keys confirmed present by a completed store.

        Called from TieringOffloadingManager._process_finished_jobs() when a
        primary -> secondary store job completes for this tier.  Overwrites
        _IN_FLIGHT or NOT_FOUND entries; leaves existing FOUND entries untouched.
        """
        for key in keys:
            if self._lookup_state.get(key, _IN_FLIGHT) != FOUND:
                if key not in self._lookup_state:
                    self._evict_if_full()
                self._lookup_state[key] = FOUND

    def shutdown(self) -> None:
        """Stop the worker thread."""
        self._lookup_queue.put(None)
        self._thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_full(self) -> None:
        """Evict the oldest resolved entry if at capacity.

        _IN_FLIGHT entries are never evicted — evicting a pending key would
        cause its result to arrive in _pending_results with no matching
        _IN_FLIGHT entry, leading drain_results() to insert a stale result.
        """
        if len(self._lookup_state) < self._max_results:
            return
        evict_key = next(
            (k for k, st in self._lookup_state.items() if st != _IN_FLIGHT),
            None,
        )
        if evict_key is not None:
            del self._lookup_state[evict_key]
            return
        logger.warning(
            "async_lookup cache full (%d entries): all entries _IN_FLIGHT, "
            "skipping eviction",
            self._max_results,
        )

    def _worker(self) -> None:
        while True:
            pending = self._lookup_queue.get()
            if pending is None:
                break

            # Group by req_id.
            batches: dict[str, tuple[ReqContext, list[OffloadKey]]] = {}
            for key, req_context in pending:
                req_id = req_context.req_id
                if req_id not in batches:
                    batches[req_id] = (req_context, [])
                batches[req_id][1].append(key)

            if not batches:
                continue

            results: list[tuple[OffloadKey, int]] = []
            for req_context, keys in batches.values():
                try:
                    hits = self.batch_lookup(keys, req_context)
                except Exception as exc:
                    logger.warning(
                        "batch_lookup failed on tier %s for %d keys: %s",
                        self._tier_type,
                        len(keys),
                        exc,
                    )
                    hits = [False] * len(keys)

                for key, hit in zip(keys, hits):
                    if hit is True:
                        results.append((key, FOUND))
                    elif hit is False:
                        results.append((key, NOT_FOUND))
                    # hit is None → stays _IN_FLIGHT (not added to results)

            # Post the entire batch as one item — no lock needed.
            if results:
                self._pending_results.put(results)
