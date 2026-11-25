# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run core prefix matching over local and external block residency."""

from collections.abc import Callable
from dataclasses import replace

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import KVCacheConfig

BlockPresence = Callable[[BlockHash, int], bool]


class _PresenceBlockPool(BlockPool):
    def __init__(self, hash_block_size: int, num_hash_lookahead_tokens: int):
        super().__init__(
            1,
            enable_caching=True,
            hash_block_size=hash_block_size,
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
        )
        self.presence: BlockPresence | None = None
        self.hit_blocks: dict[tuple[BlockHash, int], KVCacheBlock] = {}

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        assert self.presence is not None
        if not all(self.presence(block_hash, group) for group in kv_cache_group_ids):
            return None
        result = []
        for group in kv_cache_group_ids:
            key = (block_hash, group)
            block = self.hit_blocks.get(key)
            if block is None:
                block = KVCacheBlock(
                    len(self.hit_blocks) + 1,
                    block_hash=make_block_hash_with_group_id(block_hash, group),
                )
                self.hit_blocks[key] = block
            result.append(block)
        return result


class ExternalKVCacheLookup:
    """Apply the core coordinator to a request-scoped residency provider.

    Group indices are the original configuration indices, including groups
    that opt out of prefix caching. Local presence must be checked block by
    block: sliding windows need not retain the entire local token prefix.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        *,
        max_model_len: int,
        max_in_flight_tokens: int,
        scheduler_block_size: int,
        hash_block_size: int,
        dcp_world_size: int = 1,
        num_hash_lookahead_tokens: int = 0,
    ) -> None:
        self._pool = _PresenceBlockPool(hash_block_size, num_hash_lookahead_tokens)
        self._coordinator = get_kv_cache_coordinator(
            replace(kv_cache_config, num_blocks=1),
            max_model_len=max_model_len,
            max_in_flight_tokens=max_in_flight_tokens,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=dcp_world_size,
            scheduler_block_size=scheduler_block_size,
            hash_block_size=hash_block_size,
            num_hash_lookahead_tokens=num_hash_lookahead_tokens,
            allow_partial_hash_hits=False,
        )
        self._coordinator.block_pool = self._pool
        for manager in self._coordinator.single_type_managers:
            manager.block_pool = self._pool
            manager._null_block = self._pool.null_block

    def find_longest_cache_hit(
        self,
        block_hashes: list[BlockHash],
        max_length: int,
        presence: BlockPresence,
        *,
        num_request_tokens: int | None = None,
    ) -> tuple[int, tuple[tuple[BlockHash | None, ...], ...]]:
        """Return the absolute hit length and required hashes in table order."""
        self._pool.presence = presence
        try:
            blocks, length, _ = self._coordinator.find_longest_cache_hit(
                block_hashes, max_length, num_request_tokens
            )
            return length, tuple(
                tuple(
                    None
                    if b.is_null or b.block_hash is None
                    else get_block_hash(b.block_hash)
                    for b in group
                )
                for group in blocks
            )
        finally:
            self._pool.presence = None
            self._pool.hit_blocks.clear()
