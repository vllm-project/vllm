# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import ctypes
from abc import ABC, abstractmethod
from collections.abc import Iterable

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

    def __init__(self, cache_capacity: int) -> None:
        self.cache_capacity = cache_capacity

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
        self,
        n: int,
        protected: set[OffloadKey],
        req_context: ReqContext,
        num_blocks_in_cache: int,
    ) -> list[tuple[OffloadKey, BlockStatus]] | None:
        """
        Evict exactly n blocks, skipping any in protected.

        Returns a list of (key, block) for the evicted blocks,
        or None if n evictions cannot be satisfied. The operation is atomic:
        if None is returned, no state changes are made.

        ``req_context`` and ``num_blocks_in_cache`` describe the store
        batch that triggered this eviction: the request making the store,
        and the number of blocks in the batch that are already resident in
        the cache (equivalently, the prefix position at which the
        not-yet-stored keys begin within its offload keys). Ignored by
        LRU/ARC; SAE uses them to compute its admission gate baseline and
        detect intra-request session continuation.

        For ARC: ghost list cleanup (trimming to cache_capacity) is performed
        at the end of a successful eviction.
        """

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

    def record_lookup(self, key: OffloadKey, req_context: ReqContext) -> None:
        """Signal that this key was inspected as part of a genuine
        request-driven lookup (i.e. the scheduler asking whether it can
        skip recomputation), not as part of an internal existence check
        (e.g. prepare_store's "already stored?" filter, prepare_load's
        ref-count bumps).

        ``req_context`` identifies the request making the lookup so policies
        can bound merge windows per-request (rather than per-scheduler-flow).

        Default is a no-op. SAE overrides this to feed its session-merge
        pointer, so only scheduler lookups can bias session continuation —
        matching the reference algorithm's separation of `lookup` vs.
        `prepare_store` at the manager level."""
        return

    def on_request_finished(self, req_context: ReqContext) -> None:
        """Called by the manager when a request has finished, so policies
        that hold per-request state (SAE's merge pointer) can drop it.
        Default is a no-op."""
        return

    def open_session(self, req_context: ReqContext, num_blocks_in_cache: int) -> None:
        """Called by the manager immediately before the ``insert`` loop of
        a prepare_store batch. Policies that group inserts into sessions
        (SAE) decide here whether this batch continues an existing session
        (merges) or opens a new one. ``req_context`` and
        ``num_blocks_in_cache`` are the same values passed to the
        preceding ``evict``. Default is a no-op."""
        return

    def close_session(self) -> None:
        """Called by the manager immediately after the ``insert`` loop of
        a prepare_store batch. SAE seals its open session here (truncating
        float-accumulated ``hits`` to int). Default is a no-op."""
        return
