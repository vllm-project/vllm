# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prefix-aware routing helpers for distributed vLLM deployments.

The global scheduler stores prefix-cache block hashes reported by each vLLM
node and routes a new request to the node with the longest cached prefix.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from itertools import count
from typing import TypeAlias

from vllm.distributed.kv_events import (
    AllBlocksCleared,
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
    KVEventBatch,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    ExternalBlockHash,
    maybe_convert_block_hash,
)

PrefixBlockHash: TypeAlias = ExternalBlockHash


@dataclass
class PrefixRouteDecision:
    """Routing decision returned by ``GlobalPrefixScheduler``."""

    node_id: str
    matched_tokens: int
    data_parallel_rank: int | None = None


@dataclass
class PrefixCacheSnapshot:
    """Full prefix-cache view exported by one vLLM node.

    ``group_hashes`` stores one hash set per KV-cache group. Standard full
    attention models usually have one group. Hybrid attention models may have
    multiple groups with different block sizes.
    """

    node_id: str
    hash_block_size: int
    group_block_sizes: dict[int, int]
    group_hashes: dict[int, set[PrefixBlockHash]]
    data_parallel_rank: int | None = None


# Backward-compatible alias for earlier local code that used this name.
PrefixCacheSingleNode = PrefixCacheSnapshot


@dataclass
class NodePrefixCacheState:
    """Prefix-cache index for one vLLM node."""

    node_id: str
    hash_block_size: int
    data_parallel_rank: int | None = None
    group_block_sizes: dict[int, int] = field(default_factory=dict)
    group_hashes: dict[int, set[PrefixBlockHash]] = field(
        default_factory=lambda: defaultdict(set)
    )

    @classmethod
    def from_snapshot(cls, snapshot: PrefixCacheSnapshot) -> "NodePrefixCacheState":
        return cls(
            node_id=snapshot.node_id,
            data_parallel_rank=snapshot.data_parallel_rank,
            hash_block_size=snapshot.hash_block_size,
            group_block_sizes=dict(snapshot.group_block_sizes),
            group_hashes=defaultdict(
                set,
                {
                    group_id: set(hashes)
                    for group_id, hashes in snapshot.group_hashes.items()
                },
            ),
        )

    @classmethod
    def from_singleNode(cls, snapshot: PrefixCacheSnapshot) -> "NodePrefixCacheState":
        return cls.from_snapshot(snapshot)

    def apply_snapshot(self, snapshot: PrefixCacheSnapshot) -> None:
        if snapshot.node_id != self.node_id:
            raise ValueError(
                f"snapshot for node {snapshot.node_id!r} cannot update "
                f"state for node {self.node_id!r}"
            )
        self.data_parallel_rank = snapshot.data_parallel_rank
        self.hash_block_size = snapshot.hash_block_size
        self.group_block_sizes = dict(snapshot.group_block_sizes)
        self.group_hashes = defaultdict(
            set,
            {
                group_id: set(hashes)
                for group_id, hashes in snapshot.group_hashes.items()
            },
        )

    def apply_singleNode(self, snapshot: PrefixCacheSnapshot) -> None:
        self.apply_snapshot(snapshot)

    def apply_events(self, events: Iterable[KVCacheEvent]) -> None:
        """Apply prefix-cache deltas emitted by a vLLM node."""
        for event in events:
            if isinstance(event, BlockStored):
                group_idx = 0 if event.group_idx is None else event.group_idx
                self.group_block_sizes[group_idx] = event.block_size
                self.group_hashes[group_idx].update(event.block_hashes)
            elif isinstance(event, BlockRemoved):
                group_idx = 0 if event.group_idx is None else event.group_idx
                hashes = self.group_hashes.get(group_idx)
                if hashes is not None:
                    hashes.difference_update(event.block_hashes)
            elif isinstance(event, AllBlocksCleared):
                self.group_hashes.clear()

    def longest_prefix_match(
        self,
        block_hashes: Sequence[BlockHash],
        prompt_num_tokens: int,
        max_cache_hit_length: int | None = None,
    ) -> int:
        """Return the longest cached prefix length in tokens for this node."""
        if not block_hashes or not self.group_hashes:
            return 0

        max_length = prompt_num_tokens - 1
        if max_cache_hit_length is not None:
            max_length = min(max_length, max_cache_hit_length)
        if max_length <= 0:
            return 0

        group_hits = [
            self._longest_group_match(
                block_hashes=block_hashes,
                hashes=hashes,
                block_size=self.group_block_sizes.get(gid, self.hash_block_size),
                max_cache_hit_length=max_length,
            )
            for gid, hashes in self.group_hashes.items()
            if hashes
        ]
        return min(group_hits, default=0)

    def _longest_group_match(
        self,
        block_hashes: Sequence[BlockHash],
        hashes: set[PrefixBlockHash],
        block_size: int,
        max_cache_hit_length: int,
    ) -> int:
        if block_size <= 0 or block_size % self.hash_block_size != 0:
            return 0

        scale = block_size // self.hash_block_size
        max_blocks = min(
            max_cache_hit_length // block_size,
            len(block_hashes) // scale,
        )

        matched_blocks = 0
        for block_idx in range(max_blocks):
            cache_key = self._cache_key_for_group_block(
                block_hashes, block_idx, scale
            )
            if cache_key not in hashes:
                break
            matched_blocks += 1
        return matched_blocks * block_size

    def _cache_key_for_group_block(
        self, block_hashes: Sequence[BlockHash], block_idx: int, scale: int
    ) -> PrefixBlockHash:
        if scale == 1:
            return maybe_convert_block_hash(block_hashes[block_idx])

        start = block_idx * scale
        end = start + scale
        return maybe_convert_block_hash(BlockHash(b"".join(block_hashes[start:end])))


class GlobalPrefixScheduler:
    """In-memory longest-prefix-first router for vLLM nodes."""

    def __init__(self) -> None:
        self._nodes: dict[str, NodePrefixCacheState] = {}
        self._tie_breaker = count()

    def register_node(
        self,
        node_id: str,
        *,
        hash_block_size: int,
        data_parallel_rank: int | None = None,
        group_block_sizes: Mapping[int, int] | None = None,
    ) -> NodePrefixCacheState:
        state = NodePrefixCacheState(
            node_id=node_id,
            hash_block_size=hash_block_size,
            data_parallel_rank=data_parallel_rank,
            group_block_sizes=dict(group_block_sizes or {}),
        )
        self._nodes[node_id] = state
        return state

    def update_snapshot(self, snapshot: PrefixCacheSnapshot) -> None:
        state = self._nodes.get(snapshot.node_id)
        if state is None:
            self._nodes[snapshot.node_id] = NodePrefixCacheState.from_snapshot(
                snapshot
            )
        else:
            state.apply_snapshot(snapshot)

    def update_singleNode(self, snapshot: PrefixCacheSnapshot) -> None:
        self.update_snapshot(snapshot)

    def apply_events(self, node_id: str, events: Iterable[KVCacheEvent]) -> None:
        self._nodes[node_id].apply_events(events)

    def apply_event_batch(self, node_id: str, batch: KVEventBatch) -> None:
        state = self._nodes[node_id]
        if batch.data_parallel_rank is not None:
            state.data_parallel_rank = batch.data_parallel_rank
        state.apply_events(batch.events)

    def remove_node(self, node_id: str) -> None:
        self._nodes.pop(node_id, None)

    def choose_node(
        self,
        block_hashes: Sequence[BlockHash],
        prompt_num_tokens: int,
        *,
        candidate_node_ids: Iterable[str] | None = None,
        max_cache_hit_length: int | None = None,
    ) -> PrefixRouteDecision | None:
        """Choose the node with the longest prefix-cache hit.

        Ties are broken round-robin so equally good nodes still receive traffic.
        """
        node_ids = list(candidate_node_ids) if candidate_node_ids is not None else None
        states = (
            [self._nodes[node_id] for node_id in node_ids if node_id in self._nodes]
            if node_ids is not None
            else list(self._nodes.values())
        )
        if not states:
            return None

        scored: list[tuple[int, NodePrefixCacheState]] = []
        for state in states:
            matched_tokens = state.longest_prefix_match(
                block_hashes=block_hashes,
                prompt_num_tokens=prompt_num_tokens,
                max_cache_hit_length=max_cache_hit_length,
            )
            scored.append((matched_tokens, state))

        best_match = max(matched_tokens for matched_tokens, _ in scored)
        tied_states = [
            state for matched_tokens, state in scored if matched_tokens == best_match
        ]
        state = tied_states[next(self._tie_breaker) % len(tied_states)]
        return PrefixRouteDecision(
            node_id=state.node_id,
            data_parallel_rank=state.data_parallel_rank,
            matched_tokens=best_match,
        )
