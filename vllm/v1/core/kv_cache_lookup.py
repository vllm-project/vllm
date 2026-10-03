# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only prefix lookup and positional cache hits across storage pools."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator


class CacheLookup(Protocol):
    hash_block_size: int
    null_block: KVCacheBlock

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None: ...


class JointCacheLookup:
    """Prefer GPU independently for each group, memoizing physical lookups."""

    def __init__(self, gpu: BlockPool, cpu: BlockPool):
        self.gpu = gpu
        self.cpu = cpu
        self.hash_block_size = gpu.hash_block_size
        self.null_block = gpu.null_block
        self._hits: dict[tuple[BlockHash, int], KVCacheBlock | None] = {}
        self.cpu_objects: set[int] = set()

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        result = []
        for group in kv_cache_group_ids:
            key = (block_hash, group)
            if key not in self._hits:
                hit = self.gpu.get_cached_block(block_hash, [group])
                if hit is None:
                    hit = self.cpu.get_cached_block(block_hash, [group])
                    if hit is not None:
                        self.cpu_objects.add(id(hit[0]))
                self._hits[key] = hit[0] if hit is not None else None
            block = self._hits[key]
            if block is None:
                return None
            result.append(block)
        return result


@dataclass(frozen=True, slots=True)
class CacheHitBlock:
    block: KVCacheBlock
    is_cpu: bool = False


@dataclass(kw_only=True)
class JointCacheHit:
    """A reconciled prefix; null blocks denote attention padding, not loads.

    ``num_gpu_prefix_tokens`` is a boundary, not the total GPU reuse count.
    Scheduling uses the boundary and its extension; allocation uses positions.
    Admission accepts the complete match or defers the request.
    """

    blocks: tuple[list[CacheHitBlock], ...]
    num_computed_tokens: int
    num_gpu_prefix_tokens: int
    shared_prefix_boundary: int
    num_cpu_tokens: int
    lookup: JointCacheLookup
    pinned: bool = False

    @classmethod
    def find(
        cls,
        coordinator: "KVCacheCoordinator",
        lookup: JointCacheLookup,
        block_hashes: list[BlockHash],
        max_length: int,
        block_size: int,
    ) -> "JointCacheHit":
        blocks, length, uncached = coordinator.find_longest_cache_hit(
            block_hashes, max_length, lookup=lookup
        )
        while length % block_size and any(
            id(block) in lookup.cpu_objects for group in blocks for block in group
        ):
            blocks, length, uncached = coordinator.find_longest_cache_hit(
                block_hashes,
                length // block_size * block_size,
                lookup=lookup,
            )
        hits = tuple(
            [CacheHitBlock(block, id(block) in lookup.cpu_objects) for block in group]
            for group in blocks
        )
        gpu_prefix = length
        cpu_units: set[int] = set()
        hash_size = lookup.hash_block_size
        for group, manager in zip(hits, coordinator.single_type_managers, strict=True):
            group_size = manager.block_size
            for index, hit in enumerate(group):
                if hit.is_cpu:
                    gpu_prefix = min(gpu_prefix, index * group_size)
                    cpu_units.update(
                        range(
                            index * group_size // hash_size,
                            min((index + 1) * group_size, length) // hash_size,
                        )
                    )
        return cls(
            blocks=hits,
            num_computed_tokens=length,
            num_gpu_prefix_tokens=(
                gpu_prefix // block_size * block_size if cpu_units else length
            ),
            num_cpu_tokens=len(cpu_units) * hash_size,
            shared_prefix_boundary=length + uncached if uncached else 0,
            lookup=lookup,
        )

    @property
    def num_gpu_tokens(self) -> int:
        return self.num_computed_tokens - self.num_cpu_tokens

    @property
    def needs_load(self) -> bool:
        return self.num_cpu_tokens > 0

    def pin(self) -> None:
        assert not self.pinned
        for pool, is_cpu in ((self.lookup.gpu, False), (self.lookup.cpu, True)):
            pool.touch(
                [
                    hit.block
                    for group in self.blocks
                    for hit in group
                    if not hit.block.is_null and hit.is_cpu == is_cpu
                ]
            )
        self.pinned = True

    def release(self) -> None:
        if not self.pinned:
            return
        for pool, is_cpu in ((self.lookup.gpu, False), (self.lookup.cpu, True)):
            pool.free_blocks(
                hit.block
                for group in self.blocks
                for hit in group
                if not hit.block.is_null and hit.is_cpu == is_cpu
            )
        self.pinned = False

    def allocation_blocks(self) -> tuple[list[KVCacheBlock], ...]:
        """Use unowned placeholders only for the allocator's capacity check."""
        return tuple(
            [KVCacheBlock(block_id=-1) if hit.is_cpu else hit.block for hit in group]
            for group in self.blocks
        )
