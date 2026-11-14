# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Borrow the CPU scheduler's existing cache and storage for tiering."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler.embedding_cache import (
    CacheEntry,
    EmbeddingCache,
)


class _BlockStorage(Protocol):
    def numpy(self) -> Any: ...


class _ECPrimaryRegion(Protocol):
    block_size_bytes: int
    blocks: _BlockStorage


class ECCPUPrimaryAdapter:
    """Expose existing CPU allocations without owning storage or its lifecycle."""

    def __init__(self, region: _ECPrimaryRegion, cache: EmbeddingCache) -> None:
        self._region = region
        self._cache = cache

    @property
    def block_size_bytes(self) -> int:
        return self._region.block_size_bytes

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

    def unpin(self, key: str) -> None:
        self._cache.unpin(key)

    def discard(self, key: str) -> None:
        self._cache.discard(key)

    def block_views(self, block_ids: Sequence[int]) -> tuple[memoryview, ...]:
        """Return zero-copy views of consecutive runs, preserving block order."""
        if not block_ids:
            return ()

        storage = memoryview(self._region.blocks.numpy()).cast("B")
        block_size = self.block_size_bytes
        views: list[memoryview] = []
        first = previous = block_ids[0]
        for block_id in block_ids[1:]:
            if block_id != previous + 1:
                views.append(storage[first * block_size : (previous + 1) * block_size])
                first = block_id
            previous = block_id
        views.append(storage[first * block_size : (previous + 1) * block_size])
        return tuple(views)
