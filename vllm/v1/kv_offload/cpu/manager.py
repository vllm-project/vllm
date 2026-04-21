# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Sequence
from typing import Literal

from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy (LRU or ARC).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
    ):
        self.medium: str = CPULoadStoreSpec.medium()
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        block = self._policy.get(key)
        return block is not None and block.is_ready

    def prepare_load(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    def touch(self, keys: Sequence[OffloadKey]) -> None:
        self._policy.touch(keys)

    def complete_load(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1

    def prepare_store(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None
            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    def complete_store(self, keys: Iterable[OffloadKey], success: bool = True) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._policy.remove(key)
                    self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()


# -----------------------------------------------------------------------------
# FilterReusedOffloadingManager — reuse-frequency gating for CPU offload stores
# -----------------------------------------------------------------------------


class FilterReusedOffloadingManager(OffloadingManager):
    """An :class:`OffloadingManager` decorator that skips storing blocks
    whose reuse frequency is below *store_threshold*.

    All methods are delegated to the *backing* manager.  Two methods are
    intercepted:

    * ``prepare_store`` — filters out keys that have not yet
    * ``lookup`` — records the visited key in an internal LRU
      counter, then delegates to the backing manager.
      crossed the threshold *before* calling the backing
      ``prepare_store``.

    Args:
        backing: The underlying ``OffloadingManager`` to delegate to.
        store_threshold: A block must be seen at least this many times in
            ``lookup()`` before it is eligible for offloading.  Must be >= 2
            (a value of 1 would be equivalent to no filtering).
        max_tracker_size: Maximum entries in the internal tracker's LRU table.
    """

    def __init__(
        self,
        backing: OffloadingManager,
        store_threshold: int = 2,
        max_tracker_size: int = 64_000,
    ):
        if store_threshold < 2:
            raise ValueError(
                "FilterReusedOffloadingManager store_threshold must be >= 2, "
                f"got {store_threshold}"
            )
        if max_tracker_size < 1:
            raise ValueError(
                "FilterReusedOffloadingManager max_tracker_size must be >= 1, "
                f"got {max_tracker_size}"
            )
        self._backing = backing
        self.store_threshold = store_threshold
        self.max_tracker_size = max_tracker_size
        # Ordered so we can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] = OrderedDict()

    # ------------------------------------------------------------------
    # Intercepted methods
    # ------------------------------------------------------------------

    def lookup(self, key: OffloadKey, req_context: ReqContext) -> bool | None:
        """Record the key, then delegate lookup to backing manager."""
        if key in self.counts:
            self.counts.move_to_end(key)
            self.counts[key] += 1
        else:
            if len(self.counts) >= self.max_tracker_size:
                self.counts.popitem(last=False)  # evict LRU
            self.counts[key] = 1
        return self._backing.lookup(key, req_context)

    def prepare_store(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """Filter out blocks below threshold, then delegate to backing.

        Filtering is evaluated *before* calling the backing manager's
        ``prepare_store`` so that blocks that would be skipped do not
        consume any CPU offload capacity.
        """
        keys = list(keys)
        eligible = [
            key for key in keys if self.counts.get(key, 0) >= self.store_threshold
        ]

        # Passing an empty list is intentional and safe — CPUOffloadingManager
        # handles it correctly, returning a PrepareStoreOutput with empty lists.
        # Delegate to the backing manager with only the eligible keys.
        return self._backing.prepare_store(eligible, req_context)

    # ------------------------------------------------------------------
    # Delegated methods
    # ------------------------------------------------------------------

    def prepare_load(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        return self._backing.prepare_load(keys, req_context)

    def touch(self, keys: Iterable[OffloadKey]) -> None:
        return self._backing.touch(keys)

    def complete_load(self, keys: Iterable[OffloadKey]) -> None:
        return self._backing.complete_load(keys)

    def complete_store(self, keys: Iterable[OffloadKey], success: bool = True) -> None:
        return self._backing.complete_store(keys, success)

    def take_events(self) -> Iterable[OffloadingEvent]:
        return self._backing.take_events()
