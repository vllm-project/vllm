# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable

from vllm.v1.kv_offload.abstract import OffloadKey


class BlockStatus(ctypes.Structure):
    """
    Offloading status for a single block of KV data.
    Holds the following information:

    ref_cnt - the current number of transfers using this block as a source.
        A value of -1 indicates the block is not yet ready to be read.
    block_id - index of the physical CPU buffer slot.
    """

    _fields_ = [("ref_cnt", ctypes.c_int32), ("block_id", ctypes.c_int64)]

    def __init__(self, block_id: int):
        super().__init__()
        # initialize block as "not ready" (ref_cnt = -1)
        self.ref_cnt = -1
        self.block_id = block_id

    @property
    def is_ready(self) -> bool:
        """
        Returns whether the block is ready to be read.
        """
        return self.ref_cnt >= 0


class CachePolicy(ABC):
    """
    Encapsulates both block organization (data structures) and replacement
    decisions (which block to evict). LRU and ARC differ in both dimensions —
    ARC's ghost lists and target_t1_size live at the intersection of storage
    and eviction, so they cannot be separated cleanly.
    """

    @abstractmethod
    def __init__(self, cache_capacity: int) -> None: ...

    @abstractmethod
    def get(self, key: OffloadKey) -> BlockStatus | None:
        """Find block in data structures. Returns None if not present."""

    @abstractmethod
    def insert(self, key: OffloadKey, block: BlockStatus) -> None:
        """Add a newly allocated block. For ARC: also removes from ghost lists."""

    @abstractmethod
    def remove(self, key: OffloadKey) -> None:
        """Remove a block (used to clean up after a failed store)."""

    @abstractmethod
    def touch(self, keys: Iterable[OffloadKey]) -> None:
        """Mark blocks as recently used."""

    @abstractmethod
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """
        Evict exactly n blocks, skipping any in protected.

        Returns a list of (key, block) for the evicted blocks,
        or None if n evictions cannot be satisfied. The operation is atomic:
        if None is returned, no state changes are made.

        For ARC: ghost list cleanup (trimming to cache_capacity) is performed
        at the end of a successful eviction.
        """
