# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Hybrid eviction manager combining attention scores, recency, and frequency.

The composite eviction score for each block is:

    score = alpha * norm_attention + beta * norm_recency + gamma * norm_frequency

where alpha + beta + gamma = 1.0. Blocks with the lowest composite score
are evicted first. This allows workload-aware tuning: long-context tasks
benefit from higher alpha (attention), conversational tasks from higher
beta (recency), and shared-prefix tasks from higher gamma (frequency).
"""
import time
from collections import OrderedDict
from collections.abc import Iterable
from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.kv_offload.abstract import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    PrepareStoreOutput,
)
from vllm.v1.kv_offload.backend import Backend, BlockStatus


@dataclass
class HybridBlockMetadata:
    """Per-block metadata for the hybrid eviction policy."""
    status: BlockStatus
    cumulative_attention_score: float = 0.0
    access_count: int = 0
    last_access_time: float = field(default_factory=time.monotonic)
    creation_time: float = field(default_factory=time.monotonic)


class HybridOffloadingManager(OffloadingManager):
    """
    An OffloadingManager that uses a weighted combination of attention
    scores, recency, and access frequency to decide eviction order.

    The three weights (alpha, beta, gamma) control the eviction policy:
        - alpha: weight for attention score (model-aware importance)
        - beta:  weight for recency (time since last access)
        - gamma: weight for frequency (total access count)

    When all attention scores are zero (no score updates received),
    the policy degrades to a recency+frequency hybrid, which itself
    degrades to approximate LRU when frequency is also uniform.
    """

    def __init__(
        self,
        backend: Backend,
        enable_events: bool = False,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        score_decay: float = 0.95,
    ):
        self.backend: Backend = backend
        self.blocks: OrderedDict[BlockHash, HybridBlockMetadata] = (
            OrderedDict()
        )
        self.events: list[OffloadingEvent] | None = (
            [] if enable_events else None
        )

        # Validate and store weights
        total = alpha + beta + gamma
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {total:.6f} "
                f"(alpha={alpha}, beta={beta}, gamma={gamma})"
            )
        self.alpha: float = alpha
        self.beta: float = beta
        self.gamma: float = gamma
        self.score_decay: float = score_decay

    def update_attention_scores(
        self, scores: dict[BlockHash, float]
    ) -> None:
        """Update cumulative attention scores for blocks."""
        for block_hash, score in scores.items():
            meta = self.blocks.get(block_hash)
            if meta is not None:
                meta.cumulative_attention_score += score

    def apply_score_decay(self) -> None:
        """Apply exponential decay to all attention scores."""
        for meta in self.blocks.values():
            meta.cumulative_attention_score *= self.score_decay

    def _compute_eviction_score(
        self,
        meta: HybridBlockMetadata,
        now: float,
        max_attention: float,
        max_frequency: int,
        max_age: float,
    ) -> float:
        """
        Compute composite eviction score. Higher = more valuable (keep).

        All components are normalized to [0, 1] before weighting.
        """
        # Normalize attention score
        norm_attention = (
            meta.cumulative_attention_score / max_attention
            if max_attention > 0
            else 0.0
        )

        # Normalize recency: recent blocks get higher scores
        age = now - meta.last_access_time
        norm_recency = 1.0 - (age / max_age) if max_age > 0 else 1.0

        # Normalize frequency
        norm_frequency = (
            meta.access_count / max_frequency
            if max_frequency > 0
            else 0.0
        )

        return (
            self.alpha * norm_attention
            + self.beta * norm_recency
            + self.gamma * norm_frequency
        )

    def lookup(self, block_hashes: Iterable[BlockHash]) -> int | None:
        hit_count = 0
        for block_hash in block_hashes:
            meta = self.blocks.get(block_hash)
            if meta is None or not meta.status.is_ready:
                break
            hit_count += 1
        return hit_count

    def prepare_load(
        self, block_hashes: Iterable[BlockHash]
    ) -> LoadStoreSpec:
        blocks = []
        for block_hash in block_hashes:
            meta = self.blocks[block_hash]
            assert meta.status.is_ready
            meta.status.ref_cnt += 1
            blocks.append(meta.status)
        return self.backend.get_load_store_spec(block_hashes, blocks)

    def touch(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in reversed(list(block_hashes)):
            meta = self.blocks.get(block_hash)
            if meta is not None:
                meta.access_count += 1
                meta.last_access_time = time.monotonic()
                self.blocks.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in block_hashes:
            meta = self.blocks[block_hash]
            assert meta.status.ref_cnt > 0
            meta.status.ref_cnt -= 1

    def _select_eviction_candidates(self, count: int) -> list[BlockHash]:
        """Select blocks with the lowest composite eviction scores."""
        now = time.monotonic()

        # Compute normalization bounds from evictable blocks only
        evictable: list[tuple[BlockHash, HybridBlockMetadata]] = [
            (bh, meta)
            for bh, meta in self.blocks.items()
            if meta.status.ref_cnt == 0
        ]

        if len(evictable) < count:
            return [bh for bh, _ in evictable]

        max_attention = max(
            (m.cumulative_attention_score for _, m in evictable), default=0.0
        )
        max_frequency = max(
            (m.access_count for _, m in evictable), default=0
        )
        max_age = max(
            (now - m.last_access_time for _, m in evictable), default=0.0
        )

        # Score each block
        scored: list[tuple[float, BlockHash]] = []
        for bh, meta in evictable:
            score = self._compute_eviction_score(
                meta, now, max_attention, max_frequency, max_age
            )
            scored.append((score, bh))

        # Sort ascending: lowest composite score evicted first
        scored.sort(key=lambda x: x[0])
        return [bh for _, bh in scored[:count]]

    def prepare_store(
        self, block_hashes: Iterable[BlockHash]
    ) -> PrepareStoreOutput | None:
        block_hashes_to_store = [
            bh for bh in block_hashes if bh not in self.blocks
        ]

        if not block_hashes_to_store:
            return PrepareStoreOutput(
                block_hashes_to_store=[],
                store_spec=self.backend.get_load_store_spec([], []),
                block_hashes_evicted=[],
            )

        num_to_evict = (
            len(block_hashes_to_store) - self.backend.get_num_free_blocks()
        )

        to_evict: list[BlockHash] = []
        if num_to_evict > 0:
            self.apply_score_decay()
            to_evict = self._select_eviction_candidates(num_to_evict)
            if len(to_evict) < num_to_evict:
                return None

        for block_hash in to_evict:
            meta = self.blocks.pop(block_hash)
            self.backend.free(meta.status)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=to_evict,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=True,
                )
            )

        statuses = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(statuses) == len(block_hashes_to_store)

        now = time.monotonic()
        for block_hash, status in zip(block_hashes_to_store, statuses):
            self.blocks[block_hash] = HybridBlockMetadata(
                status=status,
                cumulative_attention_score=0.0,
                access_count=0,
                last_access_time=now,
                creation_time=now,
            )

        store_spec = self.backend.get_load_store_spec(
            block_hashes_to_store, statuses
        )
        return PrepareStoreOutput(
            block_hashes_to_store=block_hashes_to_store,
            store_spec=store_spec,
            block_hashes_evicted=to_evict,
        )

    def complete_store(
        self,
        block_hashes: Iterable[BlockHash],
        success: bool = True,
    ) -> None:
        stored_block_hashes: list[BlockHash] = []
        if success:
            for block_hash in block_hashes:
                meta = self.blocks.get(block_hash)
                if meta is not None and not meta.status.is_ready:
                    meta.status.ref_cnt = 0
                    stored_block_hashes.append(block_hash)
        else:
            for block_hash in block_hashes:
                meta = self.blocks.get(block_hash)
                if meta is not None and not meta.status.is_ready:
                    self.backend.free(meta.status)
                    del self.blocks[block_hash]

        if stored_block_hashes and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    block_hashes=stored_block_hashes,
                    block_size=self.backend.block_size,
                    medium=self.backend.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> dict:
        """Return current manager statistics for instrumentation."""
        total = len(self.blocks)
        ready = sum(1 for m in self.blocks.values() if m.status.is_ready)
        avg_score = (
            sum(m.cumulative_attention_score for m in self.blocks.values())
            / total
            if total > 0
            else 0.0
        )
        avg_freq = (
            sum(m.access_count for m in self.blocks.values()) / total
            if total > 0
            else 0.0
        )
        return {
            "total_blocks": total,
            "ready_blocks": ready,
            "avg_attention_score": avg_score,
            "avg_access_count": avg_freq,
            "weights": {
                "alpha": self.alpha,
                "beta": self.beta,
                "gamma": self.gamma,
            },
            "free_backend_blocks": self.backend.get_num_free_blocks(),
        }
