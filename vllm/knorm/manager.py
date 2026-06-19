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
        self, request_id: str, total_computed_tokens: int
    ) -> None:
        """Remove blocks from the prefix.

        Drains global score buffer, then frees the oldest blocks from
        the prefix, preserving warmup region.
        """
        scores = drain_block_scores()
        if scores:
            self.update_block_scores(scores)

        num_skipped_tokens = self.get_num_skipped_tokens(total_computed_tokens)
        if num_skipped_tokens <= 0:
            return

        blocks = self.req_to_blocks[request_id]
        block_size = self.block_size
        num_skipped_blocks = min(
            num_skipped_tokens // block_size,
            len(blocks),
        )
        warmup_blocks = max(0, self._config.warmup_tokens // block_size)

        removed_cached: list[KVCacheBlock] = []
        removed_uncached: list[KVCacheBlock] = []

        for i in range(num_skipped_blocks - 1, -1, -1):
            if i >= len(blocks):
                continue
            if blocks[i] is self._null_block:
                break
            if i < warmup_blocks:
                break
            if blocks[i].block_hash is None:
                removed_uncached.append(blocks[i])
            else:
                removed_cached.append(blocks[i])
            blocks[i] = self._null_block

        if removed_cached:
            self.block_pool.free_blocks(removed_cached)
        if removed_uncached:
            self.block_pool.free_blocks(removed_uncached, prepend=True)

        if request_id in self._block_scores:
            for i in range(num_skipped_blocks):
                self._block_scores[request_id].pop(i, None)

    def free(self, request_id: str) -> None:
        """Free blocks and clean up score records."""
        super().free(request_id)
        self._block_scores.pop(request_id, None)
