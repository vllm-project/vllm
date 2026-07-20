# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Collection, Iterable
from typing import Literal

from typing_extensions import override

from vllm.distributed.kv_events import MEDIUM_CPU
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.config import CompactGroupSliceConfig
from vllm.v1.kv_offload.cpu.common import (
    CompactCPUAddress,
    CompactCPUAddressSpan,
    CompactCPULoadStoreSpec,
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.fixed_page_allocator import (
    FixedPageAllocator,
    PageAllocation,
)
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}

# Policies that support compact variable-size eviction via evict_until.
_COMPACT_SUPPORTED_POLICIES = frozenset({"lru", "arc"})


# ---------------------------------------------------------------------------
# Per-group logical payload charge for compact layout
# ---------------------------------------------------------------------------


def _build_group_payload_bytes(
    group_slice_configs: tuple[CompactGroupSliceConfig, ...],
    blocks_per_chunk: int,
) -> dict[int, int]:
    """Map group index to logical compact payload bytes per chunk."""
    lookup: dict[int, int] = {}
    for cfg in group_slice_configs:
        payload = cfg.compact_real_bytes_per_rank * blocks_per_chunk
        if payload <= 0:
            raise ValueError(
                f"compact group {cfg.group_idx} has non-positive payload {payload}"
            )
        if cfg.group_idx in lookup:
            raise ValueError(f"duplicate compact group index {cfg.group_idx}")
        lookup[cfg.group_idx] = payload
    return lookup


# ---------------------------------------------------------------------------
# CPUOffloadingManager
# ---------------------------------------------------------------------------


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy (LRU or ARC).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.

    When *compact layout* is enabled (all of *compact_group_slice_configs*,
    *blocks_per_chunk*, *compact_cpu_budget_bytes*, and *compact_page_size*
    are provided), the manager uses a global :class:`FixedPageAllocator` for
    all groups so that byte offsets share a single coordinate space.
    Prepare-store allocates pages, builds :class:`CompactCPUAddress` with
    complete ``physical_spans``, and returns a :class:`CompactCPULoadStoreSpec`.
    The same address is preserved through pending → committed for load.
    The legacy block-pool path is unchanged when compact is disabled.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
        # Compact layout arguments (optional -- all four must be set together)
        compact_group_slice_configs: tuple[CompactGroupSliceConfig, ...] | None = None,
        blocks_per_chunk: int | None = None,
        compact_cpu_budget_bytes: int | None = None,
        compact_page_size: int | None = None,
    ):
        self.medium: str = MEDIUM_CPU
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
        # Track the number of blocks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_blocks: int = 0
        # Track blocks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_blocks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

        # ---- Compact layout setup ----
        compact_args = [
            compact_group_slice_configs,
            blocks_per_chunk,
            compact_cpu_budget_bytes,
            compact_page_size,
        ]
        compact_enabled = all(arg is not None for arg in compact_args)
        if not compact_enabled and any(arg is not None for arg in compact_args):
            raise ValueError(
                "compact layout requires all four compact args to be non-None: "
                "compact_group_slice_configs, blocks_per_chunk, "
                "compact_cpu_budget_bytes, compact_page_size"
            )
        if compact_enabled:
            assert compact_group_slice_configs is not None
            assert blocks_per_chunk is not None
            assert compact_cpu_budget_bytes is not None
            assert compact_page_size is not None

            if cache_policy not in _COMPACT_SUPPORTED_POLICIES:
                raise ValueError(
                    f"compact layout policy {cache_policy!r} is unsupported; "
                    f"must be one of {sorted(_COMPACT_SUPPORTED_POLICIES)}"
                )

            self._compact_enabled = True
            self._compact_group_slice_configs = compact_group_slice_configs
            self._blocks_per_chunk = blocks_per_chunk
            self._compact_allocator: FixedPageAllocator = FixedPageAllocator(
                compact_cpu_budget_bytes,
                compact_page_size,
            )
            self._group_payload_bytes: dict[int, int] = _build_group_payload_bytes(
                compact_group_slice_configs, blocks_per_chunk
            )
            # Per-key tracking for compact layout
            self._compact_pending: dict[OffloadKey, PageAllocation] = {}
            self._compact_allocations: dict[OffloadKey, PageAllocation] = {}
        else:
            self._compact_enabled = False
            self._compact_group_slice_configs = None
            self._blocks_per_chunk = None
            self._compact_allocator = None
            self._group_payload_bytes = {}
            self._compact_pending = {}
            self._compact_allocations = {}

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

    # --- compact helpers ---

    def _compact_payload_bytes(self, key: OffloadKey) -> int:
        """Return the logical payload bytes for *key* in compact layout."""
        group_idx = int.from_bytes(key[-4:], "big", signed=False)
        try:
            return self._group_payload_bytes[group_idx]
        except KeyError:
            raise ValueError(
                f"unknown compact group index {group_idx} for key {key.hex()!r}; "
                f"configured groups: {sorted(self._group_payload_bytes)}"
            ) from None

    def _compact_build_address(
        self, group_idx: int, logical_length: int, alloc: PageAllocation
    ) -> CompactCPUAddress:
        """Build a ``CompactCPUAddress`` from a ``PageAllocation``."""
        spans = self._compact_allocator.page_spans(alloc)
        return CompactCPUAddress(
            byte_offset=spans[0][0],
            logical_length=logical_length,
            allocated_length=alloc.allocated_length,
            group_idx=group_idx,
            spans=tuple(
                CompactCPUAddressSpan(
                    byte_offset=span[0],
                    logical_length=span[1],
                    allocated_length=span[2],
                )
                for span in spans
            ),
        )

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        if self._compact_enabled:
            if key in self._compact_allocations:
                return LookupResult.HIT
            if key in self._compact_pending:
                return LookupResult.HIT_PENDING
            return LookupResult.MISS
        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        if self._compact_enabled:
            addresses: list[CompactCPUAddress] = []
            for key in keys:
                # Committed allocations only; pending keys fail loud.
                alloc = self._compact_allocations.get(key)
                assert alloc is not None, (
                    f"Block {key!r} not found in compact layout (not committed)"
                )
                block = self._policy.get(key)
                assert block is not None, f"Block {key!r} not found in cache"
                assert block.is_ready, (
                    f"Block {key!r} is not ready for reading (pending)"
                )
                if block.ref_cnt == 0:
                    self._policy.mark_non_evictable(key)
                    self._num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                    assert self._num_evictable_cache_blocks >= 0
                block.ref_cnt += 1
                group_idx = int.from_bytes(key[-4:], "big", signed=False)
                logical_length = self._compact_payload_bytes(key)
                addresses.append(
                    self._compact_build_address(group_idx, logical_length, alloc)
                )
            return CompactCPULoadStoreSpec(addresses)
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            if block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        if self._compact_enabled:
            for key in keys:
                block = self._policy.get(key)
                assert block is not None, f"Block {key!r} not found"
                assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
                block.ref_cnt -= 1
                if block.ref_cnt == 0:
                    self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                    self._policy.mark_evictable(key)
            return
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        if self.counts is not None:
            num_keys = len(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)

        if self._compact_enabled:
            return self._compact_prepare_store(list(keys))

        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                return None
            # There is a still a chance for eviction failure as some of the
            # idle blocks might be in the protected list.

            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None

            # cache-policy removes only idle blocks.
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0

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
        self._num_write_pending_blocks += len(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    def _compact_prepare_store(
        self, keys: list[OffloadKey]
    ) -> PrepareStoreOutput | None:
        """Compact layout prepare_store implementation."""
        # Filter out keys already stored in compact layout.
        keys_to_store = [
            k
            for k in keys
            if k not in self._compact_allocations and k not in self._compact_pending
        ]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=CompactCPULoadStoreSpec([]),
                evicted_keys=[],
            )

        allocator = self._compact_allocator
        assert allocator is not None
        assert self._compact_group_slice_configs is not None
        assert self._blocks_per_chunk is not None

        # --- Compute pages required for this batch, per key ---
        # (key, group_idx, payload_bytes)
        key_payloads: list[tuple[OffloadKey, int, int]] = []
        for key in keys_to_store:
            group_idx = int.from_bytes(key[-4:], "big", signed=False)
            payload_bytes = self._compact_payload_bytes(key)
            key_payloads.append((key, group_idx, payload_bytes))

        sizes = [p[2] for p in key_payloads]

        # --- Simulate batch allocation atomically ---
        # Check if we need eviction by simulating with no frees first.
        if not allocator.simulate_batch_allocation(sizes):
            # Not enough free pages; try eviction using evict_until.
            protected = set(keys)

            # Compute freeable pages without mutating state (read-only simulation).
            def _compute_freeable_pages(
                candidates: list[tuple[OffloadKey, BlockStatus]],
            ) -> list[PageAllocation]:
                result: list[PageAllocation] = []
                seen: set[OffloadKey] = set()
                for k, _ in candidates:
                    if k in seen:
                        continue
                    seen.add(k)
                    alloc = self._compact_allocations.get(k)
                    if alloc is not None:
                        result.append(alloc)
                return result

            def can_fit(candidates: list[tuple[OffloadKey, BlockStatus]]) -> bool:
                freeable = _compute_freeable_pages(candidates)
                return allocator.simulate_batch_allocation(sizes, freeable)

            evicted = self._policy.evict_until(can_fit, protected)
            if evicted is None:
                return None  # eviction failed, no mutation

            # Commit evictions: free pages from allocator.
            # evict_until already removed the keys from policy data structures.
            to_evict: list[OffloadKey] = []
            for evicted_key, _ in evicted:
                alloc = self._compact_allocations.pop(evicted_key, None)
                self._compact_pending.pop(evicted_key, None)
                if alloc is not None:
                    allocator.free(alloc)
                    self._num_evictable_cache_blocks -= 1
                    assert self._num_evictable_cache_blocks >= 0
                to_evict.append(evicted_key)

            if to_evict and self.events is not None:
                self.events.append(
                    OffloadingEvent(
                        keys=to_evict,
                        medium=self.medium,
                        removed=True,
                    )
                )
        else:
            to_evict = []

        # --- Allocate pages ---
        # Allocate each key's pages. If any allocation fails mid-batch,
        # roll back all preceding allocations.
        allocated_allocations: list[tuple[OffloadKey, PageAllocation, int]] = []
        success = True
        for key, group_idx, payload_bytes in key_payloads:
            alloc = allocator.allocate(payload_bytes)
            if alloc is None:
                success = False
                break
            allocated_allocations.append((key, alloc, group_idx))

        if not success:
            # Partial allocation rollback.
            for _, alloc, _ in allocated_allocations:
                allocator.free(alloc)
            # Re-insert evicted keys back? No -- eviction was already committed
            # because evict_until committed them. The caller (scheduler) receives
            # None and should retry. The evicted keys are gone.
            # Return None to signal failure.
            return None

        # --- Register pending allocations and build CompactCPUAddress list ---
        compact_addresses: list[CompactCPUAddress] = []
        for key, alloc, group_idx in allocated_allocations:
            self._compact_pending[key] = alloc
            # Insert into LRU/ARC policy with dummy block so evict_until
            # can track this key for future eviction ordering.
            self._policy.insert(key, BlockStatus(block_id=0))
            # Build compact address with physical_spans from the allocator.
            logical_length = self._compact_payload_bytes(key)
            address = self._compact_build_address(group_idx, logical_length, alloc)
            compact_addresses.append(address)

        assert len(compact_addresses) == len(keys_to_store)
        self._num_write_pending_blocks += len(keys_to_store)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=CompactCPULoadStoreSpec(compact_addresses),
            evicted_keys=to_evict if success else [],
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        if self._compact_enabled:
            return self._compact_complete_store(list(keys), success)
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
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

    def _compact_complete_store(self, keys: list[OffloadKey], success: bool) -> None:
        """Compact layout complete_store implementation."""
        stored_keys: list[OffloadKey] = []
        if success:
            for key in keys:
                alloc = self._compact_pending.pop(key, None)
                if alloc is not None:
                    # Move pending -> committed. Same address, same alloc.
                    self._compact_allocations[key] = alloc
                    stored_keys.append(key)
                    # Update the policy dummy block: mark as ready (ref_cnt 0)
                    # and evictable so the policy can evict it later.
                    block = self._policy.get(key)
                    if block is not None and not block.is_ready:
                        block.ref_cnt = 0
                        self._num_write_pending_blocks -= 1
                        self._num_evictable_cache_blocks += 1
                        self._policy.mark_evictable(key)
        else:
            for key in keys:
                alloc = self._compact_pending.pop(key, None)
                if alloc is not None:
                    # Free pages. The key was tracked in the policy for
                    # eviction ordering; remove it since the store failed.
                    self._compact_allocator.free(alloc)
                    self._compact_allocations.pop(key, None)
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def reset_cache(self) -> None:
        if self._compact_enabled:
            # Reset compact layout state.
            self._compact_allocator.reset()
            self._compact_pending.clear()
            self._compact_allocations.clear()
            self._policy.clear()
            self._num_evictable_cache_blocks = 0
            self._num_write_pending_blocks = 0
            return
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0

        self._free_list.clear()
        self._num_allocated_blocks = 0

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = (
            self._num_allocated_blocks
            - len(self._free_list)
            - self._num_evictable_cache_blocks
        )
        usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = (
            self._num_write_pending_blocks / self._num_blocks
            if self._num_blocks > 0
            else 0.0
        )
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats
