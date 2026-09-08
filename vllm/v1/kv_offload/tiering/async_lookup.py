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
Results are drained on the first lookup after each flush, at flush(), and
after worker shutdown. In-flight lookups with no remaining request references
are retained until their results are drained, allowing new requests to share
the same probe.
"""

import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field
from enum import Enum, auto

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)


class LookupPhase(Enum):
    """Lifecycle phase of a lookup probe."""

    PENDING = auto()  # Accumulated in _lookup_batch, but not yet submitted.
    IN_FLIGHT = auto()  # Submitted to the worker, but not yet resolved.
    RESOLVED = auto()  # The worker result has been applied to the state.


@dataclass(slots=True)
class LookupState:
    generation: int
    phase: LookupPhase = LookupPhase.PENDING
    result: bool | None = None  # Populated when phase is RESOLVED.
    request_ids: set[str] = field(default_factory=set)  # requests asking for the lookup


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
        # req_id → keys looked up by that request (reverse index for cleanup).
        self._req_keys: dict[str, set[OffloadKey]] = {}
        # Results from an older state must not be applied after cleanup removes
        # and recreates a key.
        self._next_generation = 0

        # Accumulates (key, req_context, generation) triples during lookup().
        # Flushed as one queue item per step by flush().
        self._lookup_batch: list[tuple[OffloadKey, ReqContext, int]] = []

        # Scheduler → worker: one full step's batch per item.
        # None is used as a shutdown sentinel.
        self._lookup_queue: queue.SimpleQueue[
            list[tuple[OffloadKey, ReqContext, int]] | None
        ] = queue.SimpleQueue()

        # Worker → scheduler: completed result batches.
        # Each item is a list of (key, generation, found) triples.
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
        req_id = req_context.req_id
        state = self._lookup_state.get(key)
        if state is None:
            state = LookupState(generation=self._next_generation)
            self._next_generation += 1
            self._lookup_state[key] = state
            self._lookup_batch.append((key, req_context, state.generation))
        state.request_ids.add(req_id)
        self._req_keys.setdefault(req_id, set()).add(key)
        return state.result

    def flush(self) -> None:
        """Post this step's accumulated keys to the worker thread.

        Called once per step from on_schedule_end() after all lookup() calls
        are done. The worker receives the full batch and processes it during
        the model-execution window, maximising time available before the next
        step's drain_results(). Also drains completed lookups when there
        are no new keys to submit.
        """
        self.drain_results()
        self._need_to_drain = True
        batch = self._lookup_batch
        self._lookup_batch = []
        in_flight_batch = []
        for key, req_context, generation in batch:
            state = self._lookup_state.get(key)
            if state is None or state.generation != generation:
                continue
            assert state.phase is LookupPhase.PENDING
            state.phase = LookupPhase.IN_FLIGHT
            in_flight_batch.append((key, req_context, generation))
        if in_flight_batch:
            self._lookup_queue.put(in_flight_batch)

    def drain_results(self) -> None:
        """Apply pending worker results to _lookup_state.

        Called from lookup(), flush(), and shutdown() on the scheduler thread.
        """
        while True:
            try:
                batch = self._pending_results.get_nowait()
            except queue.Empty:
                break
            for key, generation, result in batch:
                state = self._lookup_state.get(key)
                if state is None or state.generation != generation:
                    continue
                if not state.request_ids:
                    del self._lookup_state[key]
                    continue
                assert state.phase is LookupPhase.IN_FLIGHT
                # Each lookup generation is enqueued exactly once. A matching
                # generation must not receive a second result; stale
                # generations were discarded above.
                assert state.result is None, (
                    "cached key received a second lookup result; the "
                    "enqueue-once invariant is broken and could reopen the "
                    "failed-load livelock"
                )
                state.result = result
                state.phase = LookupPhase.RESOLVED

    def mark_miss(self, keys: Collection[OffloadKey]) -> None:
        """Force the cached verdict for ``keys`` to False after a failed load, so
        the scheduler stops re-issuing the doomed promotion (livelock, #49176).
        Keys with no cached entry are skipped."""
        for key in keys:
            state = self._lookup_state.get(key)
            if state is not None:
                state.result = False
                state.phase = LookupPhase.RESOLVED

    def cleanup(self, req_id: str) -> None:
        """Release request references, retaining in-flight lookups.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        for key in self._req_keys.pop(req_id, ()):
            state = self._lookup_state[key]
            state.request_ids.discard(req_id)
            if not state.request_ids and state.phase is not LookupPhase.IN_FLIGHT:
                del self._lookup_state[key]

    def shutdown(self) -> None:
        """Stop the worker thread and drain completed lookups."""
        self._lookup_queue.put(None)  # unblock _worker from _lookup_queue.get()
        self._thread.join()
        self.drain_results()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _worker(self) -> None:
        while True:
            pending = self._lookup_queue.get()
            if pending is None:
                break

            # Group by req_id.
            batches: dict[str, tuple[ReqContext, list[tuple[OffloadKey, int]]]] = {}
            for key, req_context, generation in pending:
                req_id = req_context.req_id
                if req_id not in batches:
                    batches[req_id] = (req_context, [])
                batches[req_id][1].append((key, generation))

            if not batches:
                continue

            results: list[tuple[OffloadKey, int, bool]] = []
            for req_context, entries in batches.values():
                keys = [key for key, _ in entries]
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

                for (key, generation), hit in zip(entries, hits):
                    results.append((key, generation, hit))

            # Post the entire batch as one item — no lock needed.
            if results:
                self._pending_results.put(results)
