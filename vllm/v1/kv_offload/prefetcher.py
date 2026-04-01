# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Predictive prefetching for KV cache offloading.

Predicts which CPU-offloaded blocks will be needed next and triggers
asynchronous CPU->GPU transfers before the blocks are actually requested.
This hides transfer latency by overlapping data movement with computation.

Supported prediction strategies:
- Sequential: predict next N blocks following the current access
- Strided: detect strided access patterns and predict accordingly
- Frequency-based: prefetch blocks with high historical access frequency
"""
import time
from collections import defaultdict
from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_utils import BlockHash


@dataclass
class PrefetchRequest:
    """A request to prefetch a block from CPU to GPU."""
    block_hash: BlockHash
    priority: float
    predicted_at: float = field(default_factory=time.monotonic)


@dataclass
class PrefetchStats:
    """Statistics for prefetch accuracy tracking."""
    total_prefetches: int = 0
    useful_prefetches: int = 0  # prefetched block was actually accessed
    wasted_prefetches: int = 0  # prefetched block was evicted before use
    latency_saved_ms: float = 0.0  # estimated latency savings

    @property
    def accuracy(self) -> float:
        return (
            self.useful_prefetches / self.total_prefetches
            if self.total_prefetches > 0
            else 0.0
        )


class SequentialPrefetcher:
    """
    Predicts next blocks based on sequential and strided access patterns.

    Tracks per-request access sequences. When a block is accessed, the
    prefetcher predicts the next `lookahead` blocks in the sequence will
    also be accessed soon and returns them as prefetch candidates.

    For auto-regressive generation, tokens are generated sequentially,
    so KV blocks within a request tend to be accessed in order. This
    prefetcher exploits that pattern.
    """

    def __init__(
        self,
        lookahead: int = 2,
        max_pending: int = 8,
        cooldown_seconds: float = 0.1,
    ):
        # Number of blocks ahead to prefetch
        self.lookahead: int = lookahead
        # Maximum number of concurrent pending prefetches
        self.max_pending: int = max_pending
        # Minimum time between prefetch requests for the same block
        self.cooldown: float = cooldown_seconds

        # Per-request access history: request_id -> list of block_hashes
        self._request_sequences: defaultdict[str, list[BlockHash]] = (
            defaultdict(list)
        )
        # block_hash -> last prefetch time (for cooldown)
        self._last_prefetch_time: dict[BlockHash, float] = {}
        # Currently pending prefetches (not yet confirmed loaded)
        self._pending: set[BlockHash] = set()
        # Blocks that have been prefetched (for accuracy tracking)
        self._prefetched: set[BlockHash] = set()

        self.stats: PrefetchStats = PrefetchStats()

    def record_access(
        self,
        request_id: str,
        block_hashes: list[BlockHash],
    ) -> None:
        """
        Record that a request accessed the given blocks.

        Args:
            request_id: unique request identifier.
            block_hashes: blocks accessed in order.
        """
        seq = self._request_sequences[request_id]
        for bh in block_hashes:
            if not seq or seq[-1] != bh:
                seq.append(bh)
            # Track useful prefetch
            if bh in self._prefetched:
                self.stats.useful_prefetches += 1
                self._prefetched.discard(bh)
                self._pending.discard(bh)

    def predict(
        self,
        request_id: str,
        current_blocks: list[BlockHash],
        offloaded_blocks: set[BlockHash],
    ) -> list[PrefetchRequest]:
        """
        Predict which offloaded blocks should be prefetched.

        Args:
            request_id: the request making the access.
            current_blocks: blocks currently being accessed.
            offloaded_blocks: set of block hashes currently on CPU.

        Returns:
            List of PrefetchRequests, ordered by priority (highest first).
        """
        now = time.monotonic()
        seq = self._request_sequences.get(request_id, [])
        candidates: list[PrefetchRequest] = []

        if not current_blocks:
            return candidates

        # Find position of current access in the request's history
        last_block = current_blocks[-1]
        try:
            pos = seq.index(last_block)
        except ValueError:
            pos = len(seq) - 1

        # Look ahead in the sequence for blocks to prefetch
        for offset in range(1, self.lookahead + 1):
            target_pos = pos + offset
            if target_pos >= len(seq):
                break

            target_hash = seq[target_pos]

            # Only prefetch if block is on CPU and not already pending
            if target_hash not in offloaded_blocks:
                continue
            if target_hash in self._pending:
                continue

            # Check cooldown
            last_time = self._last_prefetch_time.get(target_hash, 0.0)
            if now - last_time < self.cooldown:
                continue

            # Respect max_pending limit
            if len(self._pending) >= self.max_pending:
                break

            priority = 1.0 / offset  # closer blocks have higher priority
            candidates.append(
                PrefetchRequest(
                    block_hash=target_hash,
                    priority=priority,
                    predicted_at=now,
                )
            )
            self._pending.add(target_hash)
            self._prefetched.add(target_hash)
            self._last_prefetch_time[target_hash] = now
            self.stats.total_prefetches += 1

        return sorted(candidates, key=lambda r: r.priority, reverse=True)

    def complete_prefetch(self, block_hash: BlockHash) -> None:
        """Mark a prefetch as completed (block is now on GPU)."""
        self._pending.discard(block_hash)

    def cancel_prefetch(self, block_hash: BlockHash) -> None:
        """Mark a prefetch as cancelled/wasted."""
        self._pending.discard(block_hash)
        if block_hash in self._prefetched:
            self._prefetched.discard(block_hash)
            self.stats.wasted_prefetches += 1

    def remove_request(self, request_id: str) -> None:
        """Clean up state when a request completes."""
        self._request_sequences.pop(request_id, None)

    def get_stats(self) -> dict:
        return {
            "total_prefetches": self.stats.total_prefetches,
            "useful_prefetches": self.stats.useful_prefetches,
            "wasted_prefetches": self.stats.wasted_prefetches,
            "accuracy": self.stats.accuracy,
            "pending_count": len(self._pending),
            "tracked_requests": len(self._request_sequences),
        }


class FrequencyPrefetcher:
    """
    Prefetches blocks based on historical access frequency.

    Maintains a frequency table of block accesses. When blocks need
    to be loaded, it also prefetches the most frequently accessed
    blocks that are currently offloaded.

    Useful for shared-prefix workloads where the same system prompt
    blocks are accessed by many requests.
    """

    def __init__(
        self,
        top_k: int = 4,
        min_frequency: int = 3,
        max_pending: int = 8,
    ):
        self.top_k: int = top_k
        self.min_frequency: int = min_frequency
        self.max_pending: int = max_pending

        self._frequency: defaultdict[BlockHash, int] = defaultdict(int)
        self._pending: set[BlockHash] = set()
        self._prefetched: set[BlockHash] = set()
        self.stats: PrefetchStats = PrefetchStats()

    def record_access(self, block_hash: BlockHash) -> None:
        """Record a block access for frequency tracking."""
        self._frequency[block_hash] += 1
        if block_hash in self._prefetched:
            self.stats.useful_prefetches += 1
            self._prefetched.discard(block_hash)
            self._pending.discard(block_hash)

    def predict(
        self,
        offloaded_blocks: set[BlockHash],
    ) -> list[PrefetchRequest]:
        """
        Return top-K most frequently accessed blocks that are offloaded.

        Args:
            offloaded_blocks: set of block hashes currently on CPU.

        Returns:
            List of PrefetchRequests for high-frequency offloaded blocks.
        """
        now = time.monotonic()
        candidates: list[tuple[int, BlockHash]] = []

        for bh in offloaded_blocks:
            freq = self._frequency.get(bh, 0)
            if freq >= self.min_frequency and bh not in self._pending:
                candidates.append((freq, bh))

        # Sort by frequency descending
        candidates.sort(key=lambda x: x[0], reverse=True)

        requests: list[PrefetchRequest] = []
        for freq, bh in candidates[: self.top_k]:
            if len(self._pending) >= self.max_pending:
                break
            requests.append(
                PrefetchRequest(
                    block_hash=bh,
                    priority=float(freq),
                    predicted_at=now,
                )
            )
            self._pending.add(bh)
            self._prefetched.add(bh)
            self.stats.total_prefetches += 1

        return requests

    def complete_prefetch(self, block_hash: BlockHash) -> None:
        self._pending.discard(block_hash)

    def cancel_prefetch(self, block_hash: BlockHash) -> None:
        self._pending.discard(block_hash)
        if block_hash in self._prefetched:
            self._prefetched.discard(block_hash)
            self.stats.wasted_prefetches += 1
