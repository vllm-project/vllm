# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only prefix lookup and positional cache hits across storage pools."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator


def iter_true_runs(flags: Iterable[bool], start: int = 0) -> Iterator[tuple[int, int]]:
    """Yield contiguous `[run_start, run_end)` spans where `flags` are True."""
    run_start: int | None = None
    index = start
    for flag in flags:
        if flag and run_start is None:
            run_start = index
        elif not flag and run_start is not None:
            yield run_start, index
            run_start = None
        index += 1
    if run_start is not None:
        yield run_start, index


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


@dataclass(kw_only=True, slots=True)
class JointCacheHit:
    """A reconciled prefix; null blocks denote attention padding, not loads.

    Admission accepts the complete match or defers the request.
    """

    blocks: tuple[list[CacheHitBlock], ...]
    num_computed_tokens: int
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
        scheduler_block_size: int,
        num_request_tokens: int | None = None,
        *,
        allow_partial_cpu_hits: bool = False,
    ) -> "JointCacheHit":
        def _is_cpu(block: KVCacheBlock) -> bool:
            return block.pool is not None and block.pool() is lookup.cpu

        blocks, length, uncached = coordinator.find_longest_cache_hit(
            block_hashes, max_length, num_request_tokens, lookup=lookup
        )
        while (
            not allow_partial_cpu_hits
            and length % scheduler_block_size
            and any(_is_cpu(block) for group in blocks for block in group)
        ):
            blocks, length, uncached = coordinator.find_longest_cache_hit(
                block_hashes,
                length // scheduler_block_size * scheduler_block_size,
                num_request_tokens,
                lookup=lookup,
            )
        hits = tuple(
            [CacheHitBlock(block, _is_cpu(block)) for block in group]
            for group in blocks
        )
        cpu_hash_blocks: set[int] = set()
        hash_block_size = lookup.hash_block_size
        for group, manager in zip(hits, coordinator.single_type_managers, strict=True):
            for index, hit in enumerate(group):
                if hit.is_cpu:
                    start = index * manager.block_size
                    end = min(start + manager.block_size, length)
                    cpu_hash_blocks.update(
                        range(start // hash_block_size, end // hash_block_size)
                    )
        return cls(
            blocks=hits,
            num_computed_tokens=length,
            num_cpu_tokens=len(cpu_hash_blocks) * hash_block_size,
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
