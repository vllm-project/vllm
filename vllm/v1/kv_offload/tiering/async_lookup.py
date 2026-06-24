# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
AsyncLookupManager: per-tier async lookup manager for secondary tier
existence checks.

Each secondary tier that wants non-blocking lookups composes its own
AsyncLookupManager instance internally.  The manager maintains lookup
state while a concrete runtime strategy (thread/process) executes lookups.

Locking design
--------------
There is no explicit lock.  Thread safety is achieved by ownership:

* _lookup_state and _lookup_batch are owned exclusively by the scheduler
  thread.  lookup(), flush(), and cleanup() read and write them directly.

* _lookup_queue is written by the scheduler (flush → put_nowait, one item
  per step) and read by the background worker (get).  Queue primitives are
  thread/process-safe.

* _pending_results is written by the background worker (put) and read by
  the scheduler (get_nowait inside drain_results).  queue.SimpleQueue is
  thread-safe by design and multiprocessing.Queue is process-safe.

lookup() accumulates new keys in _lookup_batch without touching the queue.
flush() is called once per step from the tier's on_schedule_end(), posting
the entire batch as a single queue item so the background worker sees one
batch per step.
drain_results() is called before any lookup() calls in the same step, so
lookup() is a pure OrderedDict operation.
"""

import queue
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import cast

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import OffloadKey, ReqContext

logger = init_logger(__name__)


@dataclass(slots=True)
class LookupState:
    result: bool | None = None  # True (found), False (not found), None
    request_ids: set[str] = field(default_factory=set)  # requests asking for the lookup


LookupBatch = list[tuple[OffloadKey, ReqContext]]
LookupResults = list[tuple[OffloadKey, bool]]


class BaseAsyncLookupManager(ABC):
    """
    Per-tier async lookup manager for secondary tier existence checks.

    Each secondary tier that wants non-blocking lookups composes its own
    AsyncLookupManager instance internally. The manager maintains lookup
    state (cache, queue) and lets subclasses decide the worker runtime
    strategy (thread vs process).

    Subclasses provide queue transport and worker lifecycle.

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

        # Accumulates (key, req_context) pairs during lookup() calls.
        # Flushed as one queue item per step by flush().
        self._lookup_batch: list[tuple[OffloadKey, ReqContext]] = []

        self._need_to_drain: bool = False

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
            state = LookupState()
            self._lookup_state[key] = state
            self._lookup_batch.append((key, req_context))
        state.request_ids.add(req_id)
        self._req_keys.setdefault(req_id, set()).add(key)
        return state.result

    def flush(self) -> None:
        """Post this step's accumulated keys to the worker.

        Called once per step from on_schedule_end() after all lookup() calls
        are done. The worker receives the full batch and processes it during
        the model-execution window, maximising time available before the next
        step's drain_results().  Safe to call with an empty batch (no-op).
        """
        self._need_to_drain = True
        if self._lookup_batch:
            self.send_lookup_batch(self._lookup_batch)
            self._lookup_batch = []

    def drain_results(self) -> None:
        """Apply pending worker results to _lookup_state.

        Called from lookup() before checking state.
        """
        while True:
            batch = self.try_recv_result_batch()
            if batch is None:
                break
            for key, result in batch:
                state = self._lookup_state.get(key)
                if state is not None:
                    state.result = result

    def cleanup(self, req_id: str) -> None:
        """Remove entries no longer needed by any active request.

        Called from the tier's on_request_finished(). Uses the reverse
        index to visit only keys associated with this request.
        """
        for key in self._req_keys.pop(req_id, ()):
            state = self._lookup_state[key]
            state.request_ids.discard(req_id)
            if not state.request_ids:
                del self._lookup_state[key]

    @abstractmethod
    def send_lookup_batch(self, batch: LookupBatch) -> None: ...

    @abstractmethod
    def try_recv_result_batch(self) -> LookupResults | None: ...

    @abstractmethod
    def shutdown(self) -> None:
        """Stop the background worker."""
        ...

    @staticmethod
    def execute_lookup_batch(
        pending: LookupBatch,
        tier_type: str,
        batch_lookup: Callable[[list[OffloadKey], ReqContext], Iterable[bool]],
    ) -> LookupResults:
        """Resolve one queued lookup batch grouped by request id."""
        batches: dict[str, tuple[ReqContext, list[OffloadKey]]] = {}
        for key, req_context in pending:
            req_id = req_context.req_id
            if req_id not in batches:
                batches[req_id] = (req_context, [])
            batches[req_id][1].append(key)

        if not batches:
            return []

        results: LookupResults = []
        for req_context, keys in batches.values():
            try:
                hits = batch_lookup(keys, req_context)
            except Exception as exc:
                logger.warning(
                    "batch_lookup failed on tier %s for %d keys: %s",
                    tier_type,
                    len(keys),
                    exc,
                )
                hits = (False for _ in keys)

            for key, hit in zip(keys, hits):
                results.append((key, hit))

        return results


class AsyncLookupManager(BaseAsyncLookupManager, ABC):
    """Thread-backed async lookup manager."""

    def __init__(self, tier_type: str) -> None:
        super().__init__(tier_type=tier_type)
        # Scheduler → worker: one full step's batch per item.
        # None is used as a shutdown sentinel.
        self._lookup_queue: queue.SimpleQueue[LookupBatch | None] = queue.SimpleQueue()

        # Worker → scheduler: completed result batches.
        # SimpleQueue is explicitly thread-safe for one writer / one reader.
        self._pending_results: queue.SimpleQueue[LookupResults] = queue.SimpleQueue()

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
        """Check whether a batch of blocks exist in this tier."""
        ...

    @override
    def shutdown(self) -> None:
        self._lookup_queue.put(None)  # unblock _worker from _lookup_queue.get()
        self._thread.join()

    @override
    def send_lookup_batch(self, batch: LookupBatch) -> None:
        self._lookup_queue.put(batch)

    @override
    def try_recv_result_batch(self) -> LookupResults | None:
        try:
            return cast(LookupResults, self._pending_results.get_nowait())
        except queue.Empty:
            return None

    def _worker(self) -> None:
        while True:
            pending = self._lookup_queue.get()
            if pending is None:
                break
            results = self.execute_lookup_batch(
                pending,
                self._tier_type,
                self.batch_lookup,
            )
            if results:
                self._pending_results.put(results)
