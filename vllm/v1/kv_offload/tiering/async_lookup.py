# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AsyncLookupWorker: per-tier background worker for secondary tier existence
checks.

Each secondary tier gets its own AsyncLookupWorker instance and background
thread, so all tiers process lookups concurrently.  The scheduler fans out
query() calls to every tier worker simultaneously, then sweeps results in
tier-priority order to respect tier precedence.

Locking design
--------------
There is no explicit lock.  Thread safety is achieved by ownership:

* _lookup_state and _lookup_batch are owned exclusively by the scheduler
  thread.  query(), flush(), and update_cached_exists() read and write
  them directly.

* _lookup_queue is written by the scheduler (flush → put_nowait, one item
  per step) and read by the worker (get).  queue.Queue is thread-safe.

* _pending_results is written by the worker (put) and read by the
  scheduler (get_nowait inside drain_results).  queue.SimpleQueue is
  thread-safe by design.

query() accumulates new keys in _lookup_batch without touching the queue.
flush() is called once per step from on_schedule_end(), posting the entire
batch as a single queue item so the worker sees one batch per step.
drain_results() is called once per step from _process_finished_jobs()
before any query() calls, so query() is a pure OrderedDict operation.
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


def _slot(key: OffloadKey) -> int:
    """Map an OffloadKey to an int slot using its first 8 bytes.

    OffloadKey is itself a strong hash (block_hash + group_idx), so the
    first 8 bytes give a 64-bit identifier without storing the full bytes
    object.  Collision probability at 1M entries is ~3e-8 — negligible.
    """
    return int.from_bytes(key[:8], "big")


class AsyncLookupWorker(ABC):
    """
    Per-tier background worker for secondary tier existence checks.

    Each secondary tier has its own AsyncLookupWorker instance, giving
    each tier its own background thread so all tiers can process lookups
    concurrently.

    Subclasses implement only batch_lookup() — all queue management,
    state tracking, and result delivery is provided by this base class.

    The scheduler fans out query() calls to all tier workers simultaneously,
    then sweeps results in tier-priority order: the first FOUND wins, and
    a None from any lower-priority tier blocks acting on a higher-priority
    FOUND until that tier resolves.

    drain_results() must be called once per step from
    TieringOffloadingManager._process_finished_jobs() before any query()
    calls so that query() is a pure dict operation.
    """

    def __init__(
        self,
        tier_idx: int,
        max_results: int = 1_000_000,
    ) -> None:
        self._tier_idx = tier_idx
        self._max_results = max_results

        # slot → status; scheduler-owned, no lock needed.
        self._lookup_state: OrderedDict[int, int] = OrderedDict()

        # Accumulates (key, req_context) pairs during query() calls.
        # Flushed as one queue item per step by flush().
        self._lookup_batch: list[tuple[OffloadKey, ReqContext]] = []

        # Scheduler → worker: one full step's batch per item.
        self._lookup_queue: queue.Queue[list[tuple[OffloadKey, ReqContext]]] = (
            queue.Queue()
        )

        # Worker → scheduler: completed result batches.
        # Each item is a list of (slot, status) pairs.
        # SimpleQueue is explicitly thread-safe for one writer / one reader.
        self._pending_results: queue.SimpleQueue[list[tuple[int, int]]] = (
            queue.SimpleQueue()
        )

        self._shutdown_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name=f"vllm_offloading_lookup_tier{tier_idx}",
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

    def query(self, key: OffloadKey, req_context: ReqContext) -> int | None:
        """
        Non-blocking lookup called from the scheduler thread.

        drain_results() must have been called earlier in the same step.

        Returns:
            FOUND     — block is present in this tier.
            NOT_FOUND — block is not present in this tier.
            None      — result not yet available; retry next step.
        """
        slot = _slot(key)
        status = self._lookup_state.get(slot, _IN_FLIGHT)
        if status == _IN_FLIGHT:
            if slot not in self._lookup_state:
                # New key — buffer for async lookup; flushed by flush().
                self._evict_if_full()
                self._lookup_state[slot] = _IN_FLIGHT
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
            self._lookup_queue.put_nowait(self._lookup_batch)
            self._lookup_batch = []

    def drain_results(self) -> None:
        """Apply pending worker results to _lookup_state.

        Called once per step from TieringOffloadingManager._process_finished_jobs()
        before update_cached_exists() calls, so query() needs no queue
        interaction.  Applying worker results first ensures that a subsequent
        update_cached_exists() can correctly upgrade a worker NOT_FOUND to
        FOUND when a store completes in the same step.
        """
        while True:
            try:
                batch = self._pending_results.get_nowait()
            except queue.Empty:
                break
            for slot, status in batch:
                # Only apply if the slot is still _IN_FLIGHT.  A FOUND written
                # by update_cached_exists() during the worker's lookup takes
                # precedence and must not be overwritten.
                if self._lookup_state.get(slot, _IN_FLIGHT) == _IN_FLIGHT:
                    self._lookup_state[slot] = status

    def update_cached_exists(self, keys: Collection[OffloadKey]) -> None:
        """Populate the cache for keys confirmed present by a completed store.

        Called from TieringOffloadingManager._process_finished_jobs() when a
        primary -> secondary store job completes for this tier.  Overwrites
        _IN_FLIGHT or NOT_FOUND entries; leaves existing FOUND entries untouched.
        """
        for key in keys:
            slot = _slot(key)
            if self._lookup_state.get(slot, _IN_FLIGHT) != FOUND:
                if slot not in self._lookup_state:
                    self._evict_if_full()
                self._lookup_state[slot] = FOUND

    def shutdown(self) -> None:
        """Stop the worker thread."""
        self._shutdown_event.set()
        self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_full(self) -> None:
        """Evict the oldest resolved entry if at capacity.

        _IN_FLIGHT entries are never evicted — evicting a pending slot would
        cause its result to arrive in _pending_results with no matching
        _IN_FLIGHT entry, leading drain_results() to insert a stale result.
        """
        if len(self._lookup_state) < self._max_results:
            return
        evict_slot = next(
            (s for s, st in self._lookup_state.items() if st != _IN_FLIGHT),
            None,
        )
        if evict_slot is not None:
            del self._lookup_state[evict_slot]
            logger.warning(
                "async_lookup cache full (%d entries): evicted slot %d",
                self._max_results,
                evict_slot,
            )
            return
        # Every slot is _IN_FLIGHT — allow the dict to exceed max_results
        # temporarily rather than evict a pending lookup.
        logger.warning(
            "async_lookup cache full (%d entries): all slots _IN_FLIGHT, "
            "skipping eviction",
            self._max_results,
        )

    def _worker(self) -> None:
        while not self._shutdown_event.is_set():
            # Block until flush() posts a batch.
            try:
                pending = self._lookup_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._shutdown_event.is_set():
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

            results: list[tuple[int, int]] = []  # (slot, status)
            for req_context, keys in batches.values():
                try:
                    hits = self.batch_lookup(keys, req_context)
                except Exception as exc:
                    logger.warning(
                        "batch_lookup failed on tier %d for %d keys: %s",
                        self._tier_idx,
                        len(keys),
                        exc,
                    )
                    hits = [False] * len(keys)

                for key, hit in zip(keys, hits):
                    if hit is True:
                        results.append((_slot(key), FOUND))
                    elif hit is False:
                        results.append((_slot(key), NOT_FOUND))
                    # hit is None → stays _IN_FLIGHT (not added to results)

            # Post the entire batch as one item — no lock needed.
            if results:
                self._pending_results.put(results)
