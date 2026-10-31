# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU-primary ownership boundary for encoder-cache entries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    CacheEntry,
    EmbeddingCache,
)

class _BlockStorage(Protocol):
    def data_ptr(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


class _ECPrimaryRegion(Protocol):
    num_blocks: int
    block_size_bytes: int
    blocks: _BlockStorage

    def cleanup(self) -> None: ...


class ECPrimaryTierManager:
    """Own the CPU shared region and its entry allocation lifecycle."""

    def __init__(self, region: _ECPrimaryRegion) -> None:
        self._region = region
        self._cache = EmbeddingCache(region.num_blocks)

    @property
    def cache(self) -> EmbeddingCache:
        """Return the allocator for existing CPU session integrations."""
        return self._cache

    @property
    def num_blocks(self) -> int:
        return self._region.num_blocks

    @property
    def block_size_bytes(self) -> int:
        return self._region.block_size_bytes

    @property
    def total_size_bytes(self) -> int:
        return self.num_blocks * self.block_size_bytes

    @property
    def base_ptr(self) -> int:
        return self._region.blocks.data_ptr()

    def get(self, key: str) -> CacheEntry | None:
        return self._cache.get(key)

    def alloc(
        self,
        key: str,
        n_blocks: int,
        shape: tuple[int, int] | None = None,
    ) -> CacheEntry | None:
        return self._cache.alloc(key, n_blocks, shape)

    def mark_ready(self, key: str) -> None:
        self._cache.mark_ready(key)

    def pin(self, key: str) -> None:
        self._cache.pin(key)

    def pin_if_ready(self, key: str) -> CacheEntry | None:
        return self._cache.pin_if_ready(key)

    def unpin(self, key: str) -> None:
        self._cache.unpin(key)

    def discard(self, key: str) -> None:
        self._cache.discard(key)

    def has_held_entries(self) -> bool:
        return self._cache.has_held_entries()

    def block_views(self, block_ids: Sequence[int]) -> tuple[memoryview, ...]:
        """Return zero-copy byte views for complete allocated blocks."""
        return tuple(
            memoryview(self._region.blocks[block_id].numpy()).cast("B")
            for block_id in block_ids
        )

    def shutdown(self) -> None:
        self._region.cleanup()
