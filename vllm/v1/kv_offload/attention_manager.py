# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Attention-weighted eviction manager for KV cache offloading.

Evicts blocks with the lowest cumulative attention scores, rather than
using purely recency-based (LRU) or frequency-based policies. This
allows the system to retain KV blocks that are most relevant to the
model's current attention patterns, improving hit rates for workloads
with non-uniform access distributions.
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
class BlockMetadata:
    """Metadata tracked per offloaded block for attention-aware eviction."""
    status: BlockStatus
    cumulative_attention_score: float = 0.0
    access_count: int = 0
    last_access_time: float = field(default_factory=time.monotonic)


class AttentionWeightedOffloadingManager(OffloadingManager):
    """
    An OffloadingManager that evicts blocks with the lowest cumulative
    attention scores.

    When attention scores are not available (score == 0), this degrades
    gracefully to access-count-based eviction, which itself degrades to
    FIFO ordering for blocks that have never been touched.

    Attention scores are fed externally via `update_attention_scores()`.
    """

    def __init__(
        self,
        backend: Backend,
        enable_events: bool = False,
        score_decay: float = 0.95,
    ):
        self.backend: Backend = backend
        # block_hash -> BlockMetadata (insertion-ordered)
        self.blocks: OrderedDict[BlockHash, BlockMetadata] = OrderedDict()
        self.events: list[OffloadingEvent] | None = (
            [] if enable_events else None
        )
        # Exponential decay applied to scores each eviction round
        self.score_decay: float = score_decay

    def update_attention_scores(
        self, scores: dict[BlockHash, float]
    ) -> None:
        """
        Update cumulative attention scores for blocks.

        Called by the scheduler after receiving attention score data
        from the model runner. Scores are additive (accumulated over
        time) with exponential decay applied during eviction selection.

        Args:
            scores: mapping from block hash to attention score delta.
        """
        for block_hash, score in scores.items():
            meta = self.blocks.get(block_hash)
            if meta is not None:
                meta.cumulative_attention_score += score

    def apply_score_decay(self) -> None:
        """Apply exponential decay to all attention scores."""
        for meta in self.blocks.values():
            meta.cumulative_attention_score *= self.score_decay

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
                # Move to end (MRU position) as recency tiebreaker
                self.blocks.move_to_end(block_hash)

    def complete_load(self, block_hashes: Iterable[BlockHash]) -> None:
        for block_hash in block_hashes:
            meta = self.blocks[block_hash]
            assert meta.status.ref_cnt > 0
            meta.status.ref_cnt -= 1

    def _select_eviction_candidates(self, count: int) -> list[BlockHash]:
        """
        Select `count` blocks for eviction, preferring those with the
        lowest attention scores. Blocks with ref_cnt > 0 are skipped.

        Eviction priority (ascending, evict first):
            1. Lowest cumulative_attention_score
            2. Lowest access_count (tiebreaker)
            3. Oldest insertion order (final tiebreaker via OrderedDict)
        """
        candidates: list[tuple[float, int, BlockHash]] = []
        for block_hash, meta in self.blocks.items():
            if meta.status.ref_cnt == 0:
                candidates.append((
                    meta.cumulative_attention_score,
                    meta.access_count,
                    block_hash,
                ))

        # Sort ascending: lowest score evicted first
        candidates.sort(key=lambda x: (x[0], x[1]))
        return [bh for _, _, bh in candidates[:count]]

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
            # Apply decay before selecting eviction candidates
            self.apply_score_decay()
            to_evict = self._select_eviction_candidates(num_to_evict)
            if len(to_evict) < num_to_evict:
                return None

        # Evict selected blocks
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

        # Allocate new blocks
        statuses = self.backend.allocate_blocks(block_hashes_to_store)
        assert len(statuses) == len(block_hashes_to_store)

        now = time.monotonic()
        for block_hash, status in zip(block_hashes_to_store, statuses):
            self.blocks[block_hash] = BlockMetadata(
                status=status,
                cumulative_attention_score=0.0,
                access_count=0,
                last_access_time=now,
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
        return {
            "total_blocks": total,
            "ready_blocks": ready,
            "avg_attention_score": avg_score,
            "free_backend_blocks": self.backend.get_num_free_blocks(),
        }
