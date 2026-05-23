# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prefix-cache-aware routing helpers for internal data-parallel LB.

This module runs on the frontend side. It only chooses a DP engine and may
attach a block-aligned prefix boundary hint for the scheduler. Actual KV cache
hits are still confirmed by the engine-local prefix cache or KV connectors.
"""

from __future__ import annotations

import sys
from collections import OrderedDict, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from vllm.config import ParallelConfig
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    compute_block_hashes_from_components,
    ensure_none_hash_initialized,
)
from vllm.v1.engine import EngineCoreRequest

RankSelector = Callable[[Sequence[Sequence[int]]], int]


@dataclass(frozen=True)
class DPPrefixCacheRouteDecision:
    rank: int
    reason: str
    miss_rerouted: bool = False
    dp_prefix_cache_prefix_len: int | None = None


@dataclass
class _PrefixStats:
    hits: int = 0
    rank_hits: defaultdict[int, int] = field(
        default_factory=lambda: defaultdict(int)
    )

    def record(self, rank: int) -> None:
        self.hits += 1
        self.rank_hits[rank] += 1


class DPPrefixCacheRouter:
    """Adaptive DP rank router using vLLM block-level prompt hashes."""

    def __init__(
        self,
        parallel_config: ParallelConfig,
        *,
        hash_block_size: int | None,
        branch_block_size: int | None = None,
        hash_algo: str,
        n_ranks: int,
        start_index: int = 0,
    ) -> None:
        self.n_ranks = n_ranks
        # Hashes are computed at the engine-resolved hash granularity, while
        # scheduler branch hints must be aligned to the scheduler's cache block
        # boundary.
        self.hash_block_size = max(hash_block_size or 1, 1)
        self.branch_block_size = max(branch_block_size or self.hash_block_size, 1)
        self.start_index = start_index
        self.shallow_depth = parallel_config.data_parallel_prefix_cache_lb_shallow_depth
        self.deep_depth = parallel_config.data_parallel_prefix_cache_lb_deep_depth
        self.warm_threshold = (
            parallel_config.data_parallel_prefix_cache_lb_warm_threshold
        )
        self.load_imbalance_ratio = (
            parallel_config.data_parallel_prefix_cache_lb_load_imbalance_ratio
        )
        self.max_prefixes = parallel_config.data_parallel_prefix_cache_lb_max_prefixes
        self.caching_hash_fn = get_hash_fn_by_name(hash_algo)
        ensure_none_hash_initialized(self.caching_hash_fn)

        self.session_rank: OrderedDict[BlockHash, int] = OrderedDict()
        self.prefix_stats: OrderedDict[tuple[BlockHash, ...], _PrefixStats] = (
            OrderedDict()
        )

    def route(
        self,
        request: EngineCoreRequest,
        counts: Sequence[Sequence[int]],
        fallback_selector: RankSelector,
    ) -> DPPrefixCacheRouteDecision:
        full_signatures = self._prompt_signatures(request)
        if not full_signatures:
            return DPPrefixCacheRouteDecision(fallback_selector(counts), "no-prefix")

        overlap_depth, overlap_rank = self._longest_known_prefix(
            full_signatures, counts
        )

        session_key = self._session_key(full_signatures)
        rank = self._get_session_rank(session_key)
        reason = "sticky" if rank is not None else "hash"

        if rank is None and overlap_rank is not None:
            rank = overlap_rank
            reason = "overlap-prefix"
        elif rank is None:
            rank = self._hash_rank(session_key)

        miss_rerouted = False
        if self._would_miss(rank, full_signatures):
            balanced = self._pick_balanced_miss_rank(rank, counts, fallback_selector)
            if balanced != rank:
                rank = balanced
                reason = "miss-reroute"
                miss_rerouted = True

        dp_prefix_cache_prefix_len = None
        if rank == overlap_rank and 0 < overlap_depth < len(full_signatures):
            dp_prefix_cache_prefix_len = self._branch_prefix_len(overlap_depth)

        self._record_session(session_key, rank)
        self._record_assignment(rank, full_signatures)
        return DPPrefixCacheRouteDecision(
            rank,
            reason,
            miss_rerouted,
            dp_prefix_cache_prefix_len,
        )

    def _prompt_signatures(self, request: EngineCoreRequest) -> list[BlockHash]:
        all_token_ids = self._all_token_ids(request)
        if not all_token_ids:
            return []
        return compute_block_hashes_from_components(
            all_token_ids=all_token_ids,
            existing_block_hashes=[],
            mm_features=request.mm_features or [],
            lora_request=request.lora_request,
            cache_salt=request.cache_salt,
            prompt_embeds=request.prompt_embeds,
            prompt_embeds_per_block_hashes={},
            block_size=self.hash_block_size,
            caching_hash_fn=self.caching_hash_fn,
        )

    @staticmethod
    def _all_token_ids(request: EngineCoreRequest) -> Sequence[int]:
        if request.prompt_token_ids is not None:
            return request.prompt_token_ids
        if request.prompt_embeds is None:
            return []
        return [0] * len(request.prompt_embeds)

    def _session_key(self, signatures: list[BlockHash]) -> BlockHash:
        depth = min(len(signatures), self.shallow_depth)
        return signatures[depth - 1]

    def _get_session_rank(self, session_key: BlockHash) -> int | None:
        rank = self.session_rank.get(session_key)
        if rank is None:
            return None
        if rank >= self.n_ranks:
            del self.session_rank[session_key]
            return None
        self.session_rank.move_to_end(session_key)
        return rank

    def _hash_rank(self, key: BlockHash) -> int:
        return (int.from_bytes(key[:8], byteorder="big") + self.start_index) % (
            self.n_ranks
        )

    def _branch_prefix_len(self, depth: int) -> int | None:
        prefix_len = depth * self.hash_block_size
        prefix_len -= prefix_len % self.branch_block_size
        return prefix_len or None

    def _would_miss(self, rank: int, signatures: list[BlockHash]) -> bool:
        if rank >= self.n_ranks:
            return True
        max_depth = min(len(signatures), self.deep_depth)
        for depth in range(max_depth, 0, -1):
            prefix = tuple(signatures[:depth])
            stats = self.prefix_stats.get(prefix)
            if stats is None:
                continue
            self.prefix_stats.move_to_end(prefix)
            if stats.rank_hits.get(rank, 0) > 0:
                return False
        return True

    def _longest_known_prefix(
        self,
        signatures: list[BlockHash],
        counts: Sequence[Sequence[int]],
    ) -> tuple[int, int | None]:
        max_depth = min(len(signatures), self.deep_depth)
        min_depth = min(self.shallow_depth, max_depth)
        if max_depth == 0:
            return 0, None
        for depth in range(max_depth, min_depth - 1, -1):
            prefix = tuple(signatures[:depth])
            stats = self.prefix_stats.get(prefix)
            if stats is None or stats.hits < self.warm_threshold:
                continue
            self.prefix_stats.move_to_end(prefix)
            owners = [
                rank
                for rank, hits in stats.rank_hits.items()
                if rank < self.n_ranks and hits > 0
            ]
            if owners:
                rank = min(owners, key=lambda r: (self._load_score(counts, r), r))
                return depth, rank
        return 0, None

    def _pick_balanced_miss_rank(
        self,
        current_rank: int,
        counts: Sequence[Sequence[int]],
        fallback_selector: RankSelector,
    ) -> int:
        balanced = fallback_selector(counts)
        if balanced == current_rank:
            return current_rank
        current_score = self._load_score(counts, current_rank)
        balanced_score = self._load_score(counts, balanced)
        if current_score >= max(3, int(balanced_score * self.load_imbalance_ratio)):
            return balanced
        return current_rank

    def _record_session(self, session_key: BlockHash, rank: int) -> None:
        if session_key in self.session_rank:
            self.session_rank[session_key] = rank
            self.session_rank.move_to_end(session_key)
            return
        self.session_rank[session_key] = rank
        while len(self.session_rank) > self.max_prefixes:
            self.session_rank.popitem(last=False)

    def _record_assignment(self, rank: int, signatures: list[BlockHash]) -> None:
        max_depth = min(len(signatures), self.deep_depth)
        for depth in range(1, max_depth + 1):
            prefix = tuple(signatures[:depth])
            stats = self.prefix_stats.get(prefix)
            if stats is None:
                stats = _PrefixStats()
                self.prefix_stats[prefix] = stats
            else:
                self.prefix_stats.move_to_end(prefix)
            stats.record(rank)
            while len(self.prefix_stats) > self.max_prefixes:
                self.prefix_stats.popitem(last=False)

    @staticmethod
    def _load_score(counts: Sequence[Sequence[int]], rank: int) -> int:
        if rank >= len(counts):
            return sys.maxsize
        waiting, running = counts[rank]
        return waiting * 4 + running
