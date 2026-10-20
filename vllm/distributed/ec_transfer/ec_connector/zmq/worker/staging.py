# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded staging area for embeddings received over ZMQ.

The receive thread parks an embedding here; the model runner picks it up in
`start_load_caches` and copies it into the GPU encoder cache. Entries are
therefore short-lived, but a request that never gets scheduled (aborted client,
producer pushing for a request the consumer already dropped) would leak one, so
entries also expire.
"""

import threading
import time

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


class EmbeddingStaging:
    """Thread-safe `mm_hash` -> CPU embedding map with a byte budget.

    All public methods are safe to call from the receive thread and the model
    runner thread concurrently.
    """

    def __init__(self, capacity_bytes: int, ttl_s: float) -> None:
        self._capacity_bytes = capacity_bytes
        self._ttl_s = ttl_s
        self._lock = threading.Lock()
        self._entries: dict[str, torch.Tensor] = {}
        # Arrival time per entry, for expiry.
        self._arrived_at: dict[str, float] = {}
        # Hashes staged since the last `drain_arrivals`, reported to the
        # scheduler so it knows the item is available on this rank.
        self._arrivals: list[str] = []
        self._used_bytes = 0
        self._num_dropped = 0

    @property
    def used_bytes(self) -> int:
        with self._lock:
            return self._used_bytes

    def try_put(self, mm_hash: str, embedding: torch.Tensor) -> bool:
        """Stage an embedding.

        A duplicate push for a hash that is already staged is accepted but not
        re-reported: the scheduler counts arrivals per rank, so reporting twice
        from one rank could make an item look ready everywhere.

        Returns:
            False if the budget is exhausted; the caller should retry after
            entries are consumed.
        """
        nbytes = embedding.numel() * embedding.element_size()
        with self._lock:
            if mm_hash in self._entries:
                return True
            self._expire_locked()
            if self._used_bytes + nbytes > self._capacity_bytes:
                self._num_dropped += 1
                return False
            self._entries[mm_hash] = embedding
            self._arrived_at[mm_hash] = time.monotonic()
            self._arrivals.append(mm_hash)
            self._used_bytes += nbytes
            return True

    def pop(self, mm_hash: str) -> torch.Tensor | None:
        """Take an embedding out of staging, or None if it is not there."""
        with self._lock:
            embedding = self._entries.pop(mm_hash, None)
            if embedding is None:
                return None
            self._arrived_at.pop(mm_hash, None)
            self._used_bytes -= embedding.numel() * embedding.element_size()
            return embedding

    def drain_arrivals(self) -> dict[str, int]:
        """Return and clear the arrivals staged since the last call.

        The count is per hash so the scheduler can add up the ranks that have
        reported it.
        """
        with self._lock:
            arrivals = self._arrivals
            self._arrivals = []
        counts: dict[str, int] = {}
        for mm_hash in arrivals:
            counts[mm_hash] = counts.get(mm_hash, 0) + 1
        return counts

    def expire(self) -> None:
        """Drop entries older than the TTL."""
        with self._lock:
            self._expire_locked()

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._arrived_at.clear()
            self._arrivals.clear()
            self._used_bytes = 0

    def _expire_locked(self) -> None:
        if not self._arrived_at:
            return
        cutoff = time.monotonic() - self._ttl_s
        expired = [h for h, at in self._arrived_at.items() if at < cutoff]
        for mm_hash in expired:
            embedding = self._entries.pop(mm_hash)
            del self._arrived_at[mm_hash]
            self._used_bytes -= embedding.numel() * embedding.element_size()
        if expired:
            logger.warning(
                "EC ZMQ: dropped %d staged embedding(s) after %.0fs without a "
                "matching request: %s",
                len(expired),
                self._ttl_s,
                expired,
            )
