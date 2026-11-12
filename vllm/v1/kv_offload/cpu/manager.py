# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import time
from collections import OrderedDict
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_utils import BlockHash, maybe_convert_block_hash
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_block_hash,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from vllm.v1.kv_offload.ttl import OffloadTTLTracker, TTLEntry

logger = init_logger(__name__)


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        self.medium: Medium = Medium.CPU
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
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

        # Prefix-cache TTL of offloaded blocks, mirroring the GPU prefix cache.
        self._ttl_tracker = OffloadTTLTracker()
        # req_id -> offloaded blocks the request uses (TTL is paused)
        self._req_ttl_entries: dict[str, dict[OffloadKey, TTLEntry]] = {}
        # blocks with an in-flight store (TTL is paused until complete_store)
        self._store_ttl_entries: dict[OffloadKey, TTLEntry] = {}
        # blocks with in-flight loads (TTL is paused until complete_load)
        self._load_ttl_entries: dict[OffloadKey, list[TTLEntry]] = {}

    def _acquire_req_ttl(
        self, keys: Iterable[OffloadKey], req_context: ReqContext
    ) -> None:
        held = self._req_ttl_entries.get(req_context.req_id)
        tracker = self._ttl_tracker
        seen: set[OffloadKey] = set()
        to_acquire: list[OffloadKey] = []
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            entry = tracker.get(key)
            if entry is not None and (held is None or held.get(key) is not entry):
                to_acquire.append(key)
        if not to_acquire:
            return
        acquired = tracker.acquire(to_acquire, req_context.prompt_cache_ttl_s)
        if held is not None:
            held.update(acquired)
        else:
            tracker.release(acquired, time.monotonic())

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

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._req_ttl_entries.setdefault(req_context.req_id, {})
        return RequestOffloadingContext()

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        acquired = self._req_ttl_entries.pop(req_context.req_id, None)
        if acquired:
            self._ttl_tracker.release(list(acquired.items()), time.monotonic())

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
        self._acquire_req_ttl(keys, req_context)
        for key, entry in self._ttl_tracker.acquire(
            keys, req_context.prompt_cache_ttl_s
        ):
            self._load_ttl_entries.setdefault(key, []).append(entry)
        return self._get_load_store_spec(keys, blocks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)
        self._acquire_req_ttl(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        to_release: list[tuple[OffloadKey, TTLEntry]] = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)
            entries = self._load_ttl_entries.get(key)
            if entries:
                to_release.append((key, entries.pop()))
                if not entries:
                    del self._load_ttl_entries[key]
        if to_release:
            self._ttl_tracker.release(to_release, time.monotonic())

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        # Blocks of this range that are already offloaded are in use by the
        # request, even if storing the rest fails below.
        self._acquire_req_ttl(keys, req_context)
        if self.counts is not None:
            num_keys = len(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)
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

        if to_evict:
            self._ttl_tracker.remove(to_evict)
            if self.events is not None:
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

        self._ttl_tracker.add(keys_to_store)
        self._acquire_req_ttl(keys_to_store, req_context)
        for key, entry in self._ttl_tracker.acquire(
            keys_to_store, req_context.prompt_cache_ttl_s
        ):
            self._store_ttl_entries[key] = entry

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            to_release: list[tuple[OffloadKey, TTLEntry]] = []
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
                    if (entry := self._store_ttl_entries.pop(key, None)) is not None:
                        to_release.append((key, entry))
            if to_release:
                self._ttl_tracker.release(to_release, time.monotonic())
        else:
            removed_keys: list[OffloadKey] = []
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)
                    self._free_block(block)
                    self._store_ttl_entries.pop(key, None)
                    removed_keys.append(key)
            if removed_keys:
                self._ttl_tracker.remove(removed_keys)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def evict(self, keys: Iterable[OffloadKey]) -> list[OffloadKey]:
        evicted: list[OffloadKey] = []
        for key in keys:
            block = self._policy.get(key)
            if block is None or block.ref_cnt != 0:
                continue
            self._policy.remove(key)
            self._free_block(block)
            self._num_evictable_cache_blocks -= 1
            assert self._num_evictable_cache_blocks >= 0
            evicted.append(key)

        if evicted:
            self._ttl_tracker.remove(evicted)
            if self.events is not None:
                self.events.append(
                    OffloadingEvent(
                        keys=evicted,
                        medium=self.medium,
                        removed=True,
                    )
                )
        return evicted

    @override
    def evict_expired_blocks(self) -> list[OffloadKey]:
        now = time.monotonic()
        expired = self._ttl_tracker.pop_expired(now)
        if not expired:
            return []
        evicted = self.evict(expired)
        if len(evicted) < len(expired):
            evicted_set = set(evicted)
            self._ttl_tracker.requeue(
                [key for key in expired if key not in evicted_set], now
            )
        if evicted:
            logged_hashes = [
                maybe_convert_block_hash(
                    BlockHash(get_offload_block_hash(k) if len(k) > 4 else bytes(k))
                )
                for k in evicted
            ]
            logger.info(
                "Evicted %d expired blocks from the CPU offload cache (TTL): %s",
                len(evicted),
                logged_hashes,
            )
        return evicted

    @override
    def reset_cache(self) -> None:
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

        self._ttl_tracker.clear()
        for held in self._req_ttl_entries.values():
            held.clear()
        self._store_ttl_entries.clear()
        self._load_ttl_entries.clear()

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
