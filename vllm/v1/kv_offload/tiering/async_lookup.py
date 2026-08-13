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
  thread.  lookup(), flush(), and cleanup() read and write them directly.

* _lookup_queue is written by the scheduler (flush → put_nowait, one item
  per step) and read by the background thread (get).  queue.Queue is
  thread-safe.

* _pending_results is written by the background thread (put) and read by
  the scheduler (get_nowait inside drain_results).  queue.SimpleQueue is
  thread-safe by design.

lookup() accumulates new keys in _lookup_batch without touching the queue.
flush() is called once per step from the tier's on_schedule_end(), posting
the entire batch as a single queue item so the background thread sees one
batch per step.
drain_results() is called before any lookup() calls in the same step, so
lookup() is a pure OrderedDict operation.
"""

import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)


@dataclass(slots=True)
class LookupState:
    generation: int
    result: bool | None = None  # True (found), False (not found), None
    request_context_ids: set[int] = field(default_factory=set)


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
    on_request_finished() to this manager:
      - lookup() → drain_results() + lookup state check
      - on_schedule_end() → flush()
      - on_request_finished() → cleanup()
    """

    def __init__(
        self,
        tier_type: str,
    ) -> None:
        self._tier_type = tier_type

        # key → LookupState; scheduler-owned, no lock needed.
        self._lookup_state: dict[OffloadKey, LookupState] = {}
        # ReqContext identity → keys looked up by that request incarnation.
        self._req_keys: dict[int, set[OffloadKey]] = {}

        self._next_generation = 0

        # Accumulates (key, req_context, generation) tuples during lookup().
        # Flushed as one queue item per step by flush().
        self._lookup_batch: list[tuple[OffloadKey, ReqContext, int]] = []

        # Scheduler → worker: one full step's batch per item.
        # None is used as a shutdown sentinel.
        self._lookup_queue: queue.SimpleQueue[
            list[tuple[OffloadKey, ReqContext, int]] | None
        ] = queue.SimpleQueue()

        # Worker → scheduler: completed result batches.
        # Each item is a list of (key, found) pairs.
        # SimpleQueue is explicitly thread-safe for one writer / one reader.
        self._pending_results: queue.SimpleQueue[list[tuple[OffloadKey, int, bool]]] = (
            queue.SimpleQueue()
        )
        self._need_to_drain: bool = False

        self._thread = threading.Thread(
            target=self._worker,
            name=f"vllm_offloading_lookup_{tier_type}",
            daemon=True,
        )
        self._thread.start()

    @abstractmethod
    def batch_lookup(
        self, keys: list[OffloadKey], req_context: ReqContext
    ) -> Iterable[bool]:
        """
        Check whether a batch of blocks exist in this tier.

        Called from the worker thread — must be synchronous and must not
        touch the primary tier or scheduler state.

        Returns a list parallel to keys: True if present, False if not.
        """
        ...

    # ------------------------------------------------------------------
    # Scheduler-thread API
    # ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """
        Non-blocking lookup called from the scheduler thread.

        Returns:
            True  — block is present in this tier.
            False — block is not present in this tier.
            None  — result not yet available; retry next step.
        """
        if self._need_to_drain:
            self.drain_results()
            self._need_to_drain = False
        context_id = id(req_context)
        state = self._lookup_state.get(key)
        if state is None:
            generation = self._next_generation
            self._next_generation += 1
            state = LookupState(generation=generation)
            self._lookup_state[key] = state
            self._lookup_batch.append((key, req_context, generation))
        state.request_context_ids.add(context_id)
        self._req_keys.setdefault(context_id, set()).add(key)
        return state.result

    def flush(self) -> None:
        """Post this step's accumulated keys to the worker thread.

        Called once per step from on_schedule_end() after all lookup() calls
        are done. The worker receives the full batch and processes it during
        the model-execution window, maximising time available before the next
        step's drain_results().  Safe to call with an empty batch (no-op).
        """
        self._need_to_drain = True
        if self._lookup_batch:
            self._lookup_queue.put(self._lookup_batch)
            self._lookup_batch = []

    def drain_results(self) -> None:
        """Apply pending worker results to _lookup_state.

        Called from lookup() before checking state.
        """
        while True:
            try:
                batch = self._pending_results.get_nowait()
            except queue.Empty:
                break
            for key, generation, result in batch:
                state = self._lookup_state.get(key)
                if state is not None and state.generation == generation:
                    # A key is enqueued for probing exactly once, so a decided
                    # verdict must never receive a second result. Enforcing it
                    # keeps a late/duplicate result from resurrecting a stale
                    # True and reopening the failed-load livelock.
                    assert state.result is None, (
                        "cached key received a second lookup result; the "
                        "enqueue-once invariant is broken and could reopen the "
                        "failed-load livelock"
                    )
                    state.result = result

    def mark_miss(self, keys: Collection[OffloadKey]) -> None:
        """Force the cached verdict for ``keys`` to False after a failed load, so
        the scheduler stops re-issuing the doomed promotion (livelock, #49176).
        Keys with no cached entry are skipped."""
        for key in keys:
            state = self._lookup_state.get(key)
            if state is not None:
                state.result = False

    def cleanup(self, req_context: ReqContext) -> None:
        """Remove entries no longer needed by any active request.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        context_id = id(req_context)
        for key in self._req_keys.pop(context_id, ()):
            state = self._lookup_state[key]
            state.request_context_ids.discard(context_id)
            if not state.request_context_ids:
                del self._lookup_state[key]

    def shutdown(self) -> None:
        """Stop the worker thread."""
        self._lookup_queue.put(None)  # unblock _worker from _lookup_queue.get()
        self._thread.join()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            pending = self._lookup_queue.get()
            if pending is None:
                break

            # Group by request incarnation, not the reusable request ID.
            batches: dict[int, tuple[ReqContext, list[OffloadKey], list[int]]] = {}
            for key, req_context, generation in pending:
                context_id = id(req_context)
                if context_id not in batches:
                    batches[context_id] = (req_context, [], [])
                batches[context_id][1].append(key)
                batches[context_id][2].append(generation)

            if not batches:
                continue

            results: list[tuple[OffloadKey, int, bool]] = []
            for req_context, keys, generations in batches.values():
                try:
                    hits = self.batch_lookup(keys, req_context)
                except Exception as exc:
                    logger.warning(
                        "batch_lookup failed on tier %s for %d keys: %s",
                        self._tier_type,
                        len(keys),
                        exc,
                    )
                    hits = (False for _ in keys)

                for key, generation, hit in zip(keys, generations, hits):
                    results.append((key, generation, hit))

            # Post the entire batch as one item — no lock needed.
            if results:
                self._pending_results.put(results)
