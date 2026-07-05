# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-HUST project
"""Knorm KV cache manager — evicts blocks based on key L2 norms."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import TYPE_CHECKING

from vllm.knorm.config import KnormConfig
from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import KVCacheBlock
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheSpec

# ---------------------------------------------------------------------------
# Global bridge: model runner (attention) → KV cache manager
# ---------------------------------------------------------------------------
_pending_block_scores: dict[str, list[tuple[int, float]]] = {}


def submit_block_scores(scores: dict[str, list[tuple[int, float]]]) -> None:
    """Store block-level importance scores from the model runner."""
    _pending_block_scores.update(scores)


def drain_block_scores() -> dict[str, list[tuple[int, float]]]:
    """Read and clear pending scores. Called by manager each scheduling step."""
    result = dict(_pending_block_scores)
    _pending_block_scores.clear()
    return result


# ---------------------------------------------------------------------------
# KnormFullAttentionManager
# ---------------------------------------------------------------------------


class KnormFullAttentionManager(FullAttentionManager):
    """Full attention manager with Knorm-based KV cache compression.

    Overrides ``get_num_skipped_tokens`` and ``remove_skipped_blocks``
    to evict blocks from the prefix based on compression ratio.

    When ``compression_ratio == 1.0`` or ``enabled == False``, behaves
    identically to :class:`FullAttentionManager`.
    """

    def __init__(self, kv_cache_spec: KVCacheSpec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        self._config = KnormConfig()
        # Per-request: block_index_in_request → importance_score
        # Lower score = lower norm = higher importance = should be kept.
        self._block_scores: dict[str, dict[int, float]] = defaultdict(dict)

    # ------------------------------------------------------------------
    # Public API for receiving norm data
    # ------------------------------------------------------------------

    def update_block_scores(self, scores: dict[str, list[tuple[int, float]]]) -> None:
        """Ingest importance scores. Lower score = more important."""
        for req_id, block_scores in scores.items():
            req_dict = self._block_scores[req_id]
            for block_idx, score in block_scores:
                if block_idx in req_dict:
                    req_dict[block_idx] = min(req_dict[block_idx], score)
                else:
                    req_dict[block_idx] = score

    # ------------------------------------------------------------------
    # Overrides — core eviction logic
    # ------------------------------------------------------------------

    def get_num_skipped_tokens(self, num_computed_tokens: int) -> int:
        """Return tokens to evict from the prefix based on compression ratio."""
        if not self._config.is_active:
            return 0

        total_blocks_for_tokens = cdiv(num_computed_tokens, self.block_size)
        warmup_blocks = max(1, self._config.warmup_tokens // self.block_size)
        if total_blocks_for_tokens <= warmup_blocks:
            return 0

        target_keep_blocks = max(
            warmup_blocks,
            math.ceil(total_blocks_for_tokens * self._config.compression_ratio),
        )
        evict_blocks = total_blocks_for_tokens - target_keep_blocks
        if evict_blocks <= 0:
            return 0
        return evict_blocks * self.block_size

    def remove_skipped_blocks(
        self,
        request_id: str,
        total_computed_tokens: int,
        num_prompt_tokens: int | None = None,
    ) -> None:
        """Remove the *least important* blocks from the prefix.

        Drains the global score buffer, then evicts blocks with the
        highest key L2 norm (lowest importance), preserving the warmup
        region.  Blocks without scores (newly allocated, not yet observed
        in a forward pass) are evicted last.
        """
        # ── 1. Drain and ingest pending scores ──
        scores = drain_block_scores()
        if scores:
            self.update_block_scores(scores)

        # ── 2. Determine how many tokens (blocks) to evict ──
        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            return

        blocks = self.req_to_blocks[request_id]
        block_size = self.block_size
        num_skipped_blocks = min(
            num_skipped_tokens // block_size,
            len(blocks),
        )
        warmup_blocks = max(1, self._config.warmup_tokens // block_size)
        target_evict = num_skipped_blocks - warmup_blocks
        if target_evict <= 0:
            return

        # ── 3. Collect candidate blocks from the eviction range ──
        # Exclude warmup blocks and already-null blocks.
        req_scores = self._block_scores.get(request_id, {})
        scored: list[tuple[int, KVCacheBlock, float]] = []  # (idx, block, score)
        unscored: list[tuple[int, KVCacheBlock]] = []  # (idx, block)

        for i in range(warmup_blocks, num_skipped_blocks):
            if i >= len(blocks):
                continue
            block = blocks[i]
            if block is self._null_block:
                continue  # skip individually; no early break
            s = req_scores.get(i)
            if s is not None:
                scored.append((i, block, s))
            else:
                unscored.append((i, block))

        # ── 4. Sort: highest score (highest norm = least important) first ──
        scored.sort(key=lambda x: x[2], reverse=True)

        # ── 5. Merge and truncate to the target count ──
        eviction_candidates: list[tuple[int, KVCacheBlock]] = [
            (idx, blk) for idx, blk, _ in scored
        ]
        eviction_candidates.extend(unscored)
        num_to_evict = min(target_evict, len(eviction_candidates))
        if num_to_evict <= 0:
            return
        to_evict = eviction_candidates[:num_to_evict]

        # ── 6. Separate cached vs. uncached, replace with null_block ──
        removed_cached: list[KVCacheBlock] = []
        removed_uncached: list[KVCacheBlock] = []

        for idx, block in to_evict:
            if block.block_hash is not None:
                removed_cached.append(block)
            else:
                removed_uncached.append(block)
            blocks[idx] = self._null_block

        # ── 7. Free blocks ──
        # Cached blocks go to the back of the free queue (LRU — give
        # prefix cache a chance to hit).  Uncached blocks go to the
        # front (immediate reuse).
        if removed_cached:
            self.block_pool.free_blocks(removed_cached)
        if removed_uncached:
            self.block_pool.free_blocks(removed_uncached, prepend=True)

        # ── 8. Clean up score records for evicted indices ──
        if request_id in self._block_scores:
            for idx, _ in to_evict:
                self._block_scores[request_id].pop(idx, None)

    def free(self, request_id: str) -> None:
        """Free blocks and clean up score records."""
        super().free(request_id)
        self._block_scores.pop(request_id, None)
