# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext


class ChunkStatus(ctypes.Structure):
    """
    Offloading status for a single chunk of KV data.
    Holds the following information:

    ref_cnt - the current number of transfers using this chunk as a source.
        A value of -1 indicates the chunk is not yet ready to be read.
    chunk_id - index of the physical CPU buffer slot.
    """

    _fields_ = [("ref_cnt", ctypes.c_int32), ("chunk_id", ctypes.c_int64)]

    def __init__(self, chunk_id: int):
        super().__init__()
        # initialize chunk as "not ready" (ref_cnt = -1)
        self.ref_cnt = -1
        self.chunk_id = chunk_id

    @property
    def is_ready(self) -> bool:
        """
        Returns whether the chunk is ready to be read.
        """
        return self.ref_cnt >= 0


class CachePolicy(ABC):
    """
    Encapsulates both chunk organization (data structures) and replacement
    decisions (which chunk to evict). LRU and ARC differ in both dimensions —
    ARC's ghost lists and target_t1_size live at the intersection of storage
    and eviction, so they cannot be separated cleanly.
    """

    def __init__(self, cache_capacity: int) -> None:
        self.cache_capacity = cache_capacity

    @abstractmethod
    def get(self, key: OffloadKey) -> ChunkStatus | None:
        """Find chunk in data structures. Returns None if not present."""

    @abstractmethod
    def insert(self, key: OffloadKey, chunk: ChunkStatus) -> None:
        """Add a newly allocated chunk. For ARC: also removes from ghost lists."""

    @abstractmethod
    def remove(self, key: OffloadKey) -> None:
        """Remove a chunk (used to clean up after a failed store)."""

    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        """
        Mark chunks as recently used.

        Args:
            keys: Chunks to mark as recently used.
            req_context: Per-request context for the request touching these chunks.
        """

    @abstractmethod
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, ChunkStatus]] | None:
        """
        Evict exactly n chunks, skipping any in protected.

        Returns a list of (key, chunk) for the evicted chunks,
        or None if n evictions cannot be satisfied. The operation is atomic:
        if None is returned, no state changes are made.

        For ARC: ghost list cleanup (trimming to cache_capacity) is performed
        at the end of a successful eviction.
        """

    @abstractmethod
    def clear(self) -> None:
        """
        Remove ALL chunks regardless of ref_cnt.

        Ghost lists and adaptive state are also reset.
        """

    def mark_evictable(self, key: OffloadKey) -> None:
        """Called when a chunk's ref_cnt transitions to 0."""
        return

    def mark_non_evictable(self, key: OffloadKey) -> None:
        """Called when a chunk's ref_cnt transitions from 0."""
        return
