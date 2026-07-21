# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable

from vllm.v1.kv_offload.base import OffloadKey, ReqContext


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
    def touch(self, keys: Iterable[OffloadKey], req_context: ReqContext) -> None:
        """
        Mark blocks as recently used.

        Args:
            keys: Blocks to mark as recently used.
            req_context: Per-request context for the request touching these blocks.
        """

    @abstractmethod
    def evict(
        self, n: int, protected: set[OffloadKey]
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """
        Evict exactly ``n`` blocks.

        Legacy extension point — external policy subclasses override this
        method.  Built-in policies (LRU, ARC) override both ``evict`` and
        ``evict_until``.
        """

    def evict_until(
        self,
        can_fit: Callable[[list[tuple[OffloadKey, BlockStatus]]], bool],
        protected: set[OffloadKey],
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """
        Yield eviction candidates in exact policy order, calling ``can_fit``
        after each candidate.

        The default implementation raises :class:`NotImplementedError` because
        variable-size compact eviction requires concrete policy support.
        Built-in policies (LRU, ARC) override this method.

        Protected keys and entries with a non-zero ``ref_cnt`` are never
        selected.

        Args:
            can_fit:  Predicate called with the current (key, block) list
                      after each candidate is added.  Return True to commit
                      the collected prefix.
            protected:  Keys that must never be evicted.

        Returns:
            The committed list of (key, block) pairs, or None if the
            predicate never accepted and eviction is impossible.
        """
        raise NotImplementedError(
            "variable-size compact eviction requires policy support; "
            "override evict_until"
        )

    @abstractmethod
    def clear(self) -> None:
        """
        Remove ALL blocks regardless of ref_cnt.

        Ghost lists and adaptive state are also reset.
        """

    def mark_evictable(self, key: OffloadKey) -> None:
        """Called when a block's ref_cnt transitions to 0."""
        return

    def mark_non_evictable(self, key: OffloadKey) -> None:
        """Called when a block's ref_cnt transitions from 0."""
        return
