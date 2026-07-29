# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""StepTracker — step-counting delay with first-finish fast-path.

Manages a collection of (mm_hash, request_id) entries that need to be
processed after a configurable number of engine steps have elapsed.
Additionally, if the originating request finishes before the step-count
expires, the entry is processed immediately (the GPU transfer is
guaranteed complete by request finish time).

Used by ECCPUScheduler: one instance for readiness marking, one for
unpins.
"""

from collections import deque
from dataclasses import dataclass


@dataclass(slots=True)
class _PendingEntry:
    mm_hash: str
    request_id: str
    processed: bool = False


class StepTracker:
    """Track pending operations with step-count delay and first-finish.

    Each ``add(mm_hash, request_id)`` registers an operation that will be
    returned from ``step()`` either when:
      (a) ``max_concurrent_batches`` steps elapse (deque expiry), OR
      (b) ``request_id`` appears in ``finished_req_ids`` (first-finish).

    Each add is processed exactly once — never both (a) and (b).
    """

    def __init__(self, max_concurrent_batches: int) -> None:
        # One slot per past step; oldest at the right. When the deque is
        # full the rightmost slot is popped and its entries expire.
        self._slots: deque[list[_PendingEntry]] = deque(maxlen=max_concurrent_batches)
        # Reverse index: request_id → entries across all slots + _current.
        # Enables O(1) lookup for the first-finish fast-path.
        self._req_index: dict[str, list[_PendingEntry]] = {}
        # Entries added this step (via add()), committed to a new slot at
        # the start of the next step() call.
        self._current: list[_PendingEntry] = []

    def add(self, mm_hash: str, request_id: str) -> None:
        """Register an mm_hash needing processing after a delay."""
        entry = _PendingEntry(mm_hash=mm_hash, request_id=request_id)
        self._current.append(entry)
        self._req_index.setdefault(request_id, []).append(entry)

    def step(self, finished_req_ids: set[str]) -> list[str]:
        """Advance one step. Return mm_hashes ready for processing.

        Returns:
            List of mm_hashes to process. May contain duplicates if the
            same mm_hash was added multiple times (from different requests).
        """
        result: list[str] = []

        # Phase 1: first-finish fast-path.
        for req_id in finished_req_ids:
            entries = self._req_index.pop(req_id, None)
            if entries is None:
                continue
            for entry in entries:
                if not entry.processed:
                    entry.processed = True
                    result.append(entry.mm_hash)

        # Phase 2: step-count expiry.
        if len(self._slots) == self._slots.maxlen:
            expired_slot = self._slots.pop()
            for entry in expired_slot:
                if not entry.processed:
                    entry.processed = True
                    result.append(entry.mm_hash)
            self._cleanup_expired(expired_slot)

        # Phase 3: commit current entries to new slot.
        self._slots.appendleft(self._current)
        self._current = []

        return result

    def drain_all(self) -> list[str]:
        """Return all unprocessed entries and clear internal state."""
        result: list[str] = []
        for entry in self._current:
            if not entry.processed:
                entry.processed = True
                result.append(entry.mm_hash)
        self._current = []

        for slot in self._slots:
            for entry in slot:
                if not entry.processed:
                    entry.processed = True
                    result.append(entry.mm_hash)
        self._slots.clear()
        self._req_index.clear()
        return result

    def _cleanup_expired(self, slot: list[_PendingEntry]) -> None:
        """Prune reverse index for entries in an expired slot."""
        for entry in slot:
            entries_for_req = self._req_index.get(entry.request_id)
            if entries_for_req is None:
                # Already removed by fast-path (pop in step()).
                continue
            entries_for_req.remove(entry)
            if not entries_for_req:
                del self._req_index[entry.request_id]
