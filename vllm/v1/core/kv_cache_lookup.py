# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Read-only prefix lookup and positional cache hits across storage pools."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import groupby
from typing import TYPE_CHECKING, Protocol, Self

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
)

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator

# Read-only sentinel block used during admission capacity checks for CPU load
# positions (`block_id < 0`, `ref_cnt == 0`, `not is_null`). Never inserted
# into any block pool.
_CPU_LOAD_PLACEHOLDER = KVCacheBlock(block_id=-1)


def iter_true_runs(flags: Iterable[bool], start: int = 0) -> Iterator[tuple[int, int]]:
    """Yield contiguous `[run_start, run_end)` spans where `flags` are True."""
    index = start
    for flag, run in groupby(flags):
        run_len = sum(1 for _ in run)
        if flag:
            yield index, index + run_len
        index += run_len


class CacheLookup(Protocol):
    hash_block_size: int
    null_block: KVCacheBlock

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None: ...


class JointCacheLookup:
    """Prefer GPU independently for each group, memoizing physical lookups."""

    __slots__ = ("gpu", "cpu", "hash_block_size", "null_block", "_hits")

    def __init__(self, gpu: BlockPool, cpu: BlockPool):
        self.gpu = gpu
        self.cpu = cpu
        self.hash_block_size = gpu.hash_block_size
        self.null_block = gpu.null_block
        self._hits: dict[tuple[BlockHash, int], KVCacheBlock | None] = {}

    def get_cached_block(
        self, block_hash: BlockHash, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        result: list[KVCacheBlock] = []
        hits = self._hits
        for group in kv_cache_group_ids:
            key = (block_hash, group)
            if key in hits:
                block = hits[key]
            else:
                found = self.gpu.get_cached_block(
                    block_hash, [group]
                ) or self.cpu.get_cached_block(block_hash, [group])
                block = found[0] if found else None
                hits[key] = block
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
    ) -> Self:
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

    def _partition_blocks(self) -> tuple[list[KVCacheBlock], list[KVCacheBlock]]:
        by_pool: tuple[list[KVCacheBlock], list[KVCacheBlock]] = ([], [])
        for group in self.blocks:
            for hit in group:
                if not hit.block.is_null:
                    by_pool[hit.is_cpu].append(hit.block)
        return by_pool

    def pin(self) -> None:
        assert not self.pinned
        gpu_blocks, cpu_blocks = self._partition_blocks()
        if gpu_blocks:
            self.lookup.gpu.touch(gpu_blocks)
        if cpu_blocks:
            self.lookup.cpu.touch(cpu_blocks)
        self.pinned = True

    def release(self) -> None:
        if not self.pinned:
            return
        gpu_blocks, cpu_blocks = self._partition_blocks()
        if gpu_blocks:
            self.lookup.gpu.free_blocks(gpu_blocks)
        if cpu_blocks:
            self.lookup.cpu.free_blocks(cpu_blocks)
        self.pinned = False

    def allocation_blocks(self) -> tuple[list[KVCacheBlock], ...]:
        """Use unowned placeholders only for the allocator's capacity check."""
        return tuple(
            [_CPU_LOAD_PLACEHOLDER if hit.is_cpu else hit.block for hit in group]
            for group in self.blocks
        )
