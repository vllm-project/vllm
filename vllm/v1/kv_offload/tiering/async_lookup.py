# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AsyncLookupWorker: background worker for secondary tier existence checks.

Queue management lives here; actual lookup is delegated to each tier's
batch_lookup() implementation.
"""

import queue
import threading
from collections import OrderedDict
from collections.abc import Collection

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext
from vllm.v1.kv_offload.tiering.base import SecondaryTierManager

logger = init_logger(__name__)

# Sentinel for a slot that is queued or currently being looked up.
_IN_FLIGHT = object()

# Sentinel placed in the queue by notify_end_of_step() to guarantee the
# worker wakes and drains any trailing keys added late in a step.
_END_OF_STEP = object()

# Result constants stored in _async_results.
NOT_FOUND: int = -1
FOUND: int = 1


def _slot(key: OffloadKey) -> int:
    """Map an OffloadKey to an int slot using its first 8 bytes.

    OffloadKey is itself a strong hash (block_hash + group_idx), so the
    first 8 bytes give a 64-bit identifier without storing the full bytes
    object.  Collision probability at 1M entries is ~3e-8 — negligible.
    """
    return int.from_bytes(key[:8], "big")


class AsyncLookupWorker:
    """
    Background worker that performs secondary tier existence checks
    asynchronously, keeping the scheduler thread non-blocking.

    The scheduler thread calls query() which either returns a cached result
    or queues the key and returns None.  The worker is triggered once per
    engine step by notify_end_of_step() (called from take_events()), at
    which point all keys for that step are already queued.  The worker
    drains the queue, groups keys by req_id, and calls
    tier.batch_lookup() on each tier in order.

    Results are stored in a bounded FIFO OrderedDict.  On store completion,
    update_cached_exists() proactively populates the cache so subsequent
    lookups for stored keys return immediately.
    """

    def __init__(
        self,
        tiers: list[SecondaryTierManager],
        max_results: int = 1_000_000,
    ) -> None:
        self._tiers = tiers
        self._max_results = max_results
        self._lock = threading.Lock()
        # slot → _IN_FLIGHT | NOT_FOUND | FOUND
        self._async_results: OrderedDict[int, object] = OrderedDict()
        # slot → tier_idx, populated only when FOUND
        self._result_tier: dict[int, int] = {}
        self._queue: queue.Queue[tuple[OffloadKey, ReqContext] | object] = queue.Queue()
        self._shutdown_event = threading.Event()
        self._thread = threading.Thread(
            target=self._worker,
            name="vllm_offloading_lookup",
            daemon=True,
        )
        self._thread.start()

    def query(self, key: OffloadKey, req_context: ReqContext) -> int | None:
        """
        Non-blocking lookup called from the scheduler thread.

        Returns:
            tier_idx  — block found in that tier (consume and initiate promotion).
            NOT_FOUND — block not present in any tier.
            None      — result not yet available; retry next step.
        """
        with self._lock:
            slot = _slot(key)
            if slot in self._async_results:
                state = self._async_results[slot]
                if state is _IN_FLIGHT:
                    return None
                if state == FOUND:
                    return self._result_tier[slot]
                return NOT_FOUND  # state == NOT_FOUND
            # New key — queue for async lookup.
            self._evict_if_full()
            self._async_results[slot] = _IN_FLIGHT
            self._queue.put_nowait((key, req_context))
        return None

    def notify_end_of_step(self) -> None:
        """Signal that all lookup() calls for this engine step are done.

        Called from TieringOffloadingManager.take_events().  Places a
        sentinel in the queue so the worker wakes and drains any trailing
        keys that arrived late in the step.
        """
        self._queue.put_nowait(_END_OF_STEP)

    def update_cached_exists(
        self, keys: Collection[OffloadKey], tier_idx: int
    ) -> None:
        """Populate the cache for keys confirmed present by a completed store.

        Called from TieringOffloadingManager._process_finished_jobs() when a
        primary→secondary store job completes.  Overwrites _IN_FLIGHT or
        NOT_FOUND entries; leaves existing FOUND entries untouched.
        """
        with self._lock:
            for key in keys:
                slot = _slot(key)
                if slot in self._async_results:
                    current = self._async_results[slot]
                    if current is _IN_FLIGHT or current == NOT_FOUND:
                        self._async_results[slot] = FOUND
                        self._result_tier[slot] = tier_idx
                    # else: valid FOUND already present — leave it
                else:
                    self._evict_if_full()
                    self._async_results[slot] = FOUND
                    self._result_tier[slot] = tier_idx

    def shutdown(self) -> None:
        """Stop the worker thread."""
        self._shutdown_event.set()
        self._queue.put_nowait(_END_OF_STEP)  # unblock queue.get() if sleeping
        self._thread.join(timeout=5.0)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_if_full(self) -> None:
        """Evict the oldest resolved entry if at capacity. Must be called under _lock.

        _IN_FLIGHT entries are never evicted — evicting a pending slot would
        cause its result to be silently discarded and the key re-queued
        indefinitely, producing a livelock under high load.
        """
        if len(self._async_results) < self._max_results:
            return
        for slot, state in self._async_results.items():
            if state is not _IN_FLIGHT:
                del self._async_results[slot]
                self._result_tier.pop(slot, None)
                logger.warning(
                    "async_lookup cache full (%d entries): evicted slot %d state=%s",
                    self._max_results,
                    slot,
                    state,
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
            # Block until the first item arrives (key or end-of-step sentinel).
            try:
                item = self._queue.get(timeout=1.0)
            except queue.Empty:
                continue

            if self._shutdown_event.is_set():
                break

            # Drain the rest of the queue and group keys by req_id.
            # Sentinels (_END_OF_STEP) are discarded.
            batches: dict[str, tuple[ReqContext, list[OffloadKey]]] = {}

            def _add(item: object) -> None:
                if item is not _END_OF_STEP:
                    key, req_context = item  # type: ignore[misc]
                    req_id = req_context.req_id
                    if req_id not in batches:
                        batches[req_id] = (req_context, [])
                    batches[req_id][1].append(key)

            _add(item)
            while True:
                try:
                    _add(self._queue.get_nowait())
                except queue.Empty:
                    break

            if not batches:
                continue

            # For each request group, iterate tiers until all keys are resolved.
            results: list[tuple[int, int, int]] = []  # (slot, result, tier_idx)
            for req_context, keys in batches.values():
                unresolved = list(keys)
                found: dict[int, int] = {}   # slot → tier_idx
                not_found: list[int] = []
                # Slots where at least one tier returned None (busy); these
                # must remain _IN_FLIGHT so the scheduler retries next step.
                busy_slots: set[int] = set()

                for tier_idx, tier in enumerate(self._tiers):
                    if not unresolved:
                        break
                    try:
                        tier_results = tier.batch_lookup(unresolved, req_context)
                    except Exception as exc:
                        logger.warning(
                            "batch_lookup failed on tier %d for %d keys: %s",
                            tier_idx, len(unresolved), exc,
                        )
                        tier_results = [False] * len(unresolved)

                    still_unresolved = []
                    for key, hit in zip(unresolved, tier_results):
                        if hit is True:
                            found[_slot(key)] = tier_idx
                        elif hit is None:
                            busy_slots.add(_slot(key))
                            still_unresolved.append(key)
                        else:  # False — not in this tier, try the next
                            still_unresolved.append(key)
                    unresolved = still_unresolved

                # Keys not found in any tier: only mark NOT_FOUND if no tier
                # was busy for that key; busy keys stay _IN_FLIGHT for retry.
                for key in unresolved:
                    slot = _slot(key)
                    if slot not in busy_slots:
                        not_found.append(slot)

                for slot, tier_idx in found.items():
                    results.append((slot, FOUND, tier_idx))
                for slot in not_found:
                    results.append((slot, NOT_FOUND, -1))

            # Write results under lock; skip slots no longer _IN_FLIGHT
            # (e.g. overwritten by update_cached_exists during the lookup).
            with self._lock:
                for slot, result, tier_idx in results:
                    if self._async_results.get(slot) is _IN_FLIGHT:
                        self._async_results[slot] = result
                        if result == FOUND:
                            self._result_tier[slot] = tier_idx
