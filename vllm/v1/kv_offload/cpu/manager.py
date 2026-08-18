# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Collection, Iterable

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
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
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkSlotStatus
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy, resolved by name via
    CachePolicyFactory (built in: "lru", "arc"; external policies can either
    register their own or be loaded out-of-tree via cache_policy_module_path).

    The manager owns all shared logic: ref-counting, event emission,
    chunk pool management, and the prepare_store/complete_store skeletons.
    Policy-specific chunk organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_chunks: int,
        cache_policy: str = "lru",
        cache_policy_module_path: str | None = None,
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        self.medium: Medium = Medium.CPU
        self._num_chunks: int = num_chunks
        self._num_allocated_chunks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_chunks)
        # Track chunks in the cache that are evictable, i.e. ref_cnt 0.
        self._num_evictable_cache_chunks: int = 0
        # Track chunks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_chunks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of chunk references. Ordered so we can evict the LRU entry
        # in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    # --- chunk pool ---

    def _get_num_free_chunks(self) -> int:
        return len(self._free_list) + self._num_chunks - self._num_allocated_chunks

    def _allocate_chunks(self, keys: list[OffloadKey]) -> list[ChunkSlotStatus]:
        num_fresh = min(len(keys), self._num_chunks - self._num_allocated_chunks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh chunks
        chunks: list[ChunkSlotStatus] = []
        for _ in range(num_fresh):
            chunks.append(ChunkSlotStatus(self._num_allocated_chunks))
            self._num_allocated_chunks += 1
        # allocate reused chunks
        for _ in range(num_reused):
            chunks.append(ChunkSlotStatus(self._free_list.pop()))
        return chunks

    def _free_chunk(self, chunk: ChunkSlotStatus) -> None:
        self._free_list.append(chunk.slot_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        chunks: Iterable[ChunkSlotStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([chunk.slot_id for chunk in chunks])

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
        chunk = self._policy.get(key)
        if chunk is None:
            return LookupResult.MISS
        if not chunk.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        chunks = []
        for key in keys:
            chunk = self._policy.get(key)
            assert chunk is not None, f"Chunk {key!r} not found in cache"
            assert chunk.is_ready, f"Chunk {key!r} is not ready for reading"
            if chunk.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_chunks -= 1
                assert self._num_evictable_cache_chunks >= 0
            chunk.ref_cnt += 1
            chunks.append(chunk)
        return self._get_load_store_spec(keys, chunks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            chunk = self._policy.get(key)
            assert chunk is not None, f"Chunk {key!r} not found"
            assert chunk.ref_cnt > 0, f"Chunk {key!r} ref_cnt is already 0"
            chunk.ref_cnt -= 1
            if chunk.ref_cnt == 0:
                self._num_evictable_cache_chunks += 1
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
        # Filter out chunks that are already stored.
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        num_chunks_to_evict = len(keys_to_store) - self._get_num_free_chunks()

        to_evict: list[OffloadKey] = []
        if num_chunks_to_evict > 0:
            if num_chunks_to_evict > self._num_evictable_cache_chunks:
                # Eviction will fail.
                return None

            # There is still a chance for eviction failure as some of the
            # idle chunks might be in the protected list.
            # Chunks from the original input are excluded from eviction:
            # a chunk already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_chunks_to_evict, protected)
            if evicted is None:
                return None

            # cache-policy removes only idle chunks.
            self._num_evictable_cache_chunks -= len(evicted)
            assert self._num_evictable_cache_chunks >= 0

            for key, chunk in evicted:
                self._free_chunk(chunk)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        chunks = self._allocate_chunks(keys_to_store)
        assert len(chunks) == len(keys_to_store), (
            "Chunk pool did not allocate the expected number of chunks"
        )

        for key, chunk in zip(keys_to_store, chunks):
            self._policy.insert(key, chunk)
        self._num_write_pending_chunks += len(keys_to_store)

        # build store specs for allocated chunks
        store_spec = self._get_load_store_spec(keys_to_store, chunks)

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
            for key in keys:
                chunk = self._policy.get(key)
                if chunk is not None and not chunk.is_ready:
                    chunk.ref_cnt = 0
                    self._num_write_pending_chunks -= 1
                    self._num_evictable_cache_chunks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                chunk = self._policy.get(key)
                if chunk is not None and not chunk.is_ready:
                    self._num_write_pending_chunks -= 1
                    self._policy.remove(key)
                    self._free_chunk(chunk)

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
        # Clear ALL chunks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused slot IDs.
        self._policy.clear()
        self._num_evictable_cache_chunks = 0
        self._num_write_pending_chunks = 0
        self._free_list.clear()
        self._num_allocated_chunks = 0

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = (
            self._num_allocated_chunks
            - len(self._free_list)
            - self._num_evictable_cache_chunks
        )
        usage = num_used / self._num_chunks if self._num_chunks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = (
            self._num_write_pending_chunks / self._num_chunks
            if self._num_chunks > 0
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
