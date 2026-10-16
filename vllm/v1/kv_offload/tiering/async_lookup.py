# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AsyncLookupManager: per-tier async lookup manager for secondary tier
existence checks.

Each secondary tier that wants non-blocking lookups composes its own
AsyncLookupManager instance internally.  The manager maintains lookup
state and uses a background thread to execute batch_lookup() calls.

Locking design
--------------
There is no explicit lock.  Thread safety is achieved by ownership:

* _lookup_state and _lookup_batch are owned by whichever thread holds the
  OffloadingManager lock.  lookup(), flush(), and cleanup() read and write
  them directly.  That is usually the scheduler thread, but a tier reached
  through ParentManager -- a peer lookup fanning out from another tier, say --
  can arrive on the tiering manager's tier-poll thread instead.  Both hold
  the manager lock, so the accesses stay serialized.

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


@dataclass(frozen=True, slots=True)
class LookupToken:
    """Lookup key plus the request and resolved generation that observed it."""

    key: OffloadKey
    req_id: str
    generation: int


class LookupSubmissionStatus(Enum):
    """Lifecycle of the shared lookup submission."""

    PENDING = auto()  # Accumulated in _lookup_batch, but not yet submitted.
    IN_FLIGHT = auto()  # Submitted to the worker, but not yet resolved.
    RESOLVED = auto()  # The worker result has been published to requests.


class LookupResultStatus(Enum):
    """Result observed by one request for a lookup."""

    PENDING = auto()
    HIT = auto()
    MISS = auto()
    MARKED_MISS = auto()


@dataclass(slots=True)
class LookupState:
    generation: int
    # Submission status is shared by all requests that reuse this probe.
    submission_status: LookupSubmissionStatus = LookupSubmissionStatus.PENDING
    # The mapping is also the ownership index: every key is a request holding
    # this state, and its value is that request's result.
    request_statuses: dict[str, LookupResultStatus] = field(default_factory=dict)


class AsyncLookupManager(ABC):
    """Per-tier async lookup manager for secondary tier existence checks.

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
        """Check whether a batch of blocks exist in this tier.

        Called from the worker thread — must be synchronous and must not
        touch the primary tier or scheduler state.

        Returns a list parallel to keys: True if present, False if not.
        """
        ...

    # ------------------------------------------------------------------
    # Scheduler-thread API
    # ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """Non-blocking lookup called from the scheduler thread.

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
        self._req_keys.setdefault(req_id, set()).add(key)
        if req_id not in state.request_statuses:
            if state.submission_status is LookupSubmissionStatus.RESOLVED:
                # A MARKED_MISS belongs to the request that experienced the
                # failed promotion. Use another request's probe result when
                # one is available, so it cannot leak to a new request.
                request_status = next(
                    (
                        status
                        for status in state.request_statuses.values()
                        if status
                        in (
                            LookupResultStatus.HIT,
                            LookupResultStatus.MISS,
                        )
                    ),
                    LookupResultStatus.MISS,
                )
            else:
                request_status = LookupResultStatus.PENDING
            state.request_statuses[req_id] = request_status
        request_status = state.request_statuses[req_id]
        if request_status is LookupResultStatus.PENDING:
            return None
        return request_status is LookupResultStatus.HIT

    def lookup_token(self, key: OffloadKey, req_id: str) -> LookupToken:
        """Return the resolved lookup generation held by ``req_id``."""
        state = self._lookup_state.get(key)
        assert state is not None and req_id in state.request_statuses, (
            f"request {req_id!r} holds no lookup for key {key!r}"
        )
        assert state.request_statuses[req_id] in (
            LookupResultStatus.HIT,
            LookupResultStatus.MISS,
            LookupResultStatus.MARKED_MISS,
        ), f"lookup for key {key!r} is unresolved"
        return LookupToken(key=key, req_id=req_id, generation=state.generation)

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
            assert state.submission_status is LookupSubmissionStatus.PENDING
            state.submission_status = LookupSubmissionStatus.IN_FLIGHT
            for req_id in state.request_statuses:
                assert state.request_statuses[req_id] is LookupResultStatus.PENDING
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
                if not state.request_statuses:
                    del self._lookup_state[key]
                    continue
                assert state.submission_status is LookupSubmissionStatus.IN_FLIGHT
                # Each lookup generation is enqueued exactly once. A matching
                # generation must not receive a second result; stale
                # generations were discarded above.
                result_status = (
                    LookupResultStatus.HIT if result else LookupResultStatus.MISS
                )
                state.submission_status = LookupSubmissionStatus.RESOLVED
                for req_id in state.request_statuses:
                    if state.request_statuses[req_id] is LookupResultStatus.PENDING:
                        state.request_statuses[req_id] = result_status

    def mark_miss(self, tokens: Collection[LookupToken]) -> None:
        """Mark a failed promotion only for the request that started it."""
        for token in tokens:
            state = self._lookup_state.get(token.key)
            if state is None or state.generation != token.generation:
                continue
            if state.submission_status is not LookupSubmissionStatus.RESOLVED:
                continue
            if state.request_statuses.get(token.req_id) not in (
                LookupResultStatus.HIT,
                LookupResultStatus.MISS,
                LookupResultStatus.MARKED_MISS,
            ):
                continue
            state.request_statuses[token.req_id] = LookupResultStatus.MARKED_MISS

    def cleanup(self, req_id: str) -> None:
        """Release request references, retaining in-flight lookups.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        for key in self._req_keys.pop(req_id, ()):
            state = self._lookup_state.get(key)
            if state is None:
                continue
            state.request_statuses.pop(req_id, None)
            if (
                not state.request_statuses
                and state.submission_status is not LookupSubmissionStatus.IN_FLIGHT
            ):
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
