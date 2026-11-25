# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import math
import time
from collections import OrderedDict
from collections.abc import Collection, Iterable, Sequence
from dataclasses import dataclass, field, replace

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    ConfigInfoMapping,
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    StoreTrigger,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingInfo,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.pin import (
    PIN_ACTION_TYPE,
    PinRequest,
    parse_pin_hints,
)
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy, ChunkStatus
from vllm.v1.kv_offload.cpu.policies.factory import CachePolicyFactory
from vllm.v1.metrics.cache_hit_source import CacheHitSource


def _prefix_sort_key(key: OffloadKey, req_context: ReqContext) -> tuple[bool, int]:
    """Order keys by their token position, unknown positions last."""
    position = req_context.get_offload_key_position(key)
    return (position is None, position if position is not None else 0)


@dataclass(slots=True)
class _RequestCacheAccess:
    """Cache keys observed by one request, grouped in prefix order."""

    owner: object
    cache_generation: int
    key_groups: dict[int, list[OffloadKey]] = field(default_factory=dict)
    seen_keys: set[OffloadKey] = field(default_factory=set)
    inserted_keys: set[OffloadKey] = field(default_factory=set)
    reused_keys: set[OffloadKey] = field(default_factory=set)
    store_miss_keys: set[OffloadKey] = field(default_factory=set)
    # Keys this request has counted toward store_threshold. Each counts once
    # per request: a retry of a failed offer, or a re-offer after preemption,
    # is not another use of the content.
    counted_keys: set[OffloadKey] = field(default_factory=set)
    finished: bool = False


@dataclass(slots=True)
class _PinState:
    """The pin requests of one request, parsed once per cache generation."""

    owner: object
    cache_generation: int
    requests: tuple[PinRequest, ...]


class CPUOffloadingManager(OffloadingManager):
    """An OffloadingManager with a pluggable CachePolicy, resolved by name via
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
        pin_capacity_ratio: float = 0.5,
        info: CPUOffloadingInfo | None = None,
    ):
        if not 0 < pin_capacity_ratio <= 1:
            raise ValueError("pin_capacity_ratio must be in (0, 1]")
        self.medium: Medium = Medium.CPU
        self._num_chunks: int = num_chunks
        self._config_info: ConfigInfoMapping = (
            info.as_config_info() if info is not None else {}
        )
        self._num_allocated_chunks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = CachePolicyFactory.get_cache_policy_cls(
            cache_policy, cache_policy_module_path
        )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_chunks)
        # Track the number of chunks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_chunks: int = 0
        # Track chunks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_chunks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []
        self._cache_generation = 0

        # Number of chunk references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

        # OffloadKey -> time.monotonic() deadline. Pinned chunks are excluded
        # from eviction until their deadline lapses.
        self._pins: dict[OffloadKey, float] = {}
        # At most this many chunks may be pinned at once. The rest of the tier
        # stays evictable, so an accurate pin cannot starve normal traffic.
        self._max_pinned_chunks: int = max(1, int(num_chunks * pin_capacity_ratio))
        # (action_type, outcome) -> count, drained by get_stats().
        self._hint_outcome_counts: dict[tuple[str, str], int] = {}

    # --- chunk pool ---

    def _get_num_free_chunks(self) -> int:
        return len(self._free_list) + self._num_chunks - self._num_allocated_chunks

    def _allocate_chunks(self, keys: list[OffloadKey]) -> list[ChunkStatus]:
        num_fresh = min(len(keys), self._num_chunks - self._num_allocated_chunks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh chunks
        chunks: list[ChunkStatus] = []
        for _ in range(num_fresh):
            chunks.append(ChunkStatus(self._num_allocated_chunks))
            self._num_allocated_chunks += 1

        # allocate reused chunks
        for _ in range(num_reused):
            chunks.append(ChunkStatus(self._free_list.pop()))
        return chunks

    def _free_chunk(self, chunk: ChunkStatus) -> None:
        self._free_list.append(chunk.chunk_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        chunks: Iterable[ChunkStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([chunk.chunk_id for chunk in chunks])

    def _record_accesses(
        self, keys: Collection[OffloadKey], counted: Collection[OffloadKey]
    ) -> None:
        """Record an offer without evicting its tracked candidates.

        Keys in `counted` were counted by an earlier offer of the same request:
        they stay tracked but don't count again.
        """
        assert self.counts is not None
        protected: set[OffloadKey] = set()
        for key in keys:
            if key in self.counts:
                self.counts.move_to_end(key)
                if key not in counted:
                    self.counts[key] += 1
                protected.add(key)

        num_unprotected = len(self.counts) - len(protected)
        for key in keys:
            if key in self.counts:
                continue
            if len(self.counts) >= self.max_tracker_size:
                if num_unprotected == 0:
                    continue
                self.counts.popitem(last=False)
                num_unprotected -= 1
            self.counts[key] = 1

    def _count_hint_outcome(self, count: int, outcome: str) -> None:
        key = (PIN_ACTION_TYPE, outcome)
        self._hint_outcome_counts[key] = self._hint_outcome_counts.get(key, 0) + count

    def _get_pin_state(self, req_context: ReqContext) -> _PinState:
        state = req_context.get_state(_PinState)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
        ):
            hints = parse_pin_hints(req_context.kv_hints)
            if hints.pins:
                self._count_hint_outcome(len(hints.pins), "applied")
            if hints.unsupported:
                self._count_hint_outcome(hints.unsupported, "unsupported")
            if hints.rejected:
                self._count_hint_outcome(hints.rejected, "rejected")
            state = _PinState(
                owner=self,
                cache_generation=self._cache_generation,
                requests=hints.pins,
            )
            req_context.set_state(state)
        return state

    def _pin_keys(
        self, keys: Iterable[OffloadKey], req_context: ReqContext, now: float
    ) -> None:
        """Pin a request's chunks, prefix-first, within the tier's pin cap."""
        state = self._get_pin_state(req_context)
        if not state.requests:
            return
        deadline = now + max(request.ttl_seconds for request in state.requests)
        # Lapsed pins must not hold cap space for new pins.
        self._unexpired_pins(now)
        ordered = sorted(keys, key=lambda key: _prefix_sort_key(key, req_context))
        for key in ordered:
            pinned_until = self._pins.get(key)
            if pinned_until is not None:
                # Escalating an existing pin is always allowed.
                self._pins[key] = max(pinned_until, deadline)
            elif len(self._pins) >= self._max_pinned_chunks:
                self._count_hint_outcome(1, "clipped")
            else:
                self._pins[key] = deadline

    def _unexpired_pins(self, now: float) -> set[OffloadKey]:
        """Return the live pins, dropping the ones whose TTL has lapsed."""
        if not self._pins:
            return set()
        expired = [key for key, deadline in self._pins.items() if deadline <= now]
        for key in expired:
            del self._pins[key]
        return set(self._pins)

    def _get_request_cache_access(self, req_context: ReqContext) -> _RequestCacheAccess:
        state = req_context.get_state(_RequestCacheAccess)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
        ):
            state = _RequestCacheAccess(
                owner=self, cache_generation=self._cache_generation
            )
            req_context.set_state(state)
        return state

    def _record_request_cache_access(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        inserted_keys: Iterable[OffloadKey] = (),
        reused_keys: Iterable[OffloadKey] = (),
    ) -> None:
        state = self._get_request_cache_access(req_context)
        new_keys = []
        for key in keys:
            if key in state.seen_keys:
                continue
            assert not state.finished, (
                "New cache keys observed after request finalization"
            )
            group_idx = get_offload_group_idx(key)
            state.key_groups.setdefault(group_idx, []).append(key)
            state.seen_keys.add(key)
            new_keys.append(key)
        if new_keys:
            self._policy.on_request_access(new_keys, req_context)
        state.inserted_keys.update(inserted_keys)
        # Re-reading a chunk inserted by this request is an internal transfer
        # (for example, a tiering cascade), not a second cache access.
        state.reused_keys.update(
            key for key in reused_keys if key not in state.inserted_keys
        )

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        self._get_request_cache_access(req_context)
        self._get_pin_state(req_context)
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        chunk = self._policy.get(key)
        if chunk is None:
            return LookupResult.MISS
        if not chunk.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def get_load_source(
        self, key: OffloadKey, req_context: ReqContext
    ) -> CacheHitSource:
        return CacheHitSource.HOST

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        return self._prepare_load(keys, req_context, record_access=True)

    def _prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        record_access: bool,
    ) -> LoadStoreSpec:
        chunks = []
        for key in keys:
            chunk = self._policy.get(key)
            assert chunk is not None, f"Chunk {key!r} not found in cache"
            assert chunk.is_ready, f"Chunk {key!r} is not ready for reading"
            if chunk.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_chunks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_chunks >= 0
            chunk.ref_cnt += 1
            chunks.append(chunk)
        if record_access:
            self._record_request_cache_access(keys, req_context, reused_keys=keys)
        return self._get_load_store_spec(keys, chunks)

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
                self._num_evictable_cache_chunks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        trigger: StoreTrigger,
    ) -> PrepareStoreOutput | None:
        return self._prepare_store(keys, req_context, allow_partial=True)

    def _prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        allow_partial: bool = False,
    ) -> PrepareStoreOutput | None:
        keys = list(keys)
        state = self._get_request_cache_access(req_context)
        if self.counts is not None:
            self._record_accesses(keys, state.counted_keys)
        # Partition the original offer before admission filtering. Ready
        # resident chunks are request reuses even if their tracker entry was
        # aged out; the threshold applies only to new store candidates.
        keys_to_store: list[OffloadKey] = []
        ready_existing_keys: list[OffloadKey] = []
        evictable_existing_keys: set[OffloadKey] = set()
        for key in keys:
            chunk = self._policy.get(key)
            if chunk is None:
                keys_to_store.append(key)
            else:
                if chunk.is_ready:
                    ready_existing_keys.append(key)
                if chunk.ref_cnt == 0:
                    evictable_existing_keys.add(key)

        if self.counts is not None:
            admitted_keys: list[OffloadKey] = []
            for key in keys_to_store:
                if self.counts.get(key, 0) >= self.store_threshold:
                    admitted_keys.append(key)
                elif key not in state.counted_keys:
                    self.stores_skipped_in_current_batch += 1
            keys_to_store = admitted_keys
            state.counted_keys.update(keys)

        new_store_misses = [
            key for key in keys_to_store if key not in state.store_miss_keys
        ]
        if new_store_misses:
            # ARC learns from B1/B2 before insert() removes the ghost entry.
            # Deduplication makes this one policy observation per request.
            self._policy.on_store_miss(new_store_misses, req_context)
            state.store_miss_keys.update(new_store_misses)

        recorded_keys = set(keys_to_store)
        recorded_keys.update(ready_existing_keys)
        # Record once before any allocation-related early return. Store
        # candidates are classified as insertions only after allocation succeeds.
        self._record_request_cache_access(
            (key for key in keys if key in recorded_keys),
            req_context,
            reused_keys=ready_existing_keys,
        )

        if not keys_to_store:
            self._pin_keys(ready_existing_keys, req_context, time.monotonic())
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        keys_to_retry: list[OffloadKey] = []
        if allow_partial:
            # Offered chunks are not evicted, so they are not room for the offer.
            num_storable_keys = (
                self._get_num_free_chunks()
                + self._num_evictable_cache_chunks
                - len(evictable_existing_keys)
            )
            if len(keys_to_store) > num_storable_keys:
                # Store the lowest positions, so that the groups stay aligned.
                order = {
                    key: (
                        math.inf
                        if (position := req_context.get_offload_key_position(key))
                        is None
                        else position,
                        offer_idx,
                    )
                    for offer_idx, key in enumerate(keys)
                }
                keys_by_position = sorted(keys_to_store, key=order.__getitem__)
                # Retry each group from its first key that does not fit, resident
                # keys included, as the connector offers all of them again.
                group_cuts: dict[int, tuple[float, int]] = {}
                for key in keys_by_position[num_storable_keys:]:
                    group_cuts.setdefault(get_offload_group_idx(key), order[key])
                keys_to_retry = [
                    key
                    for key in keys
                    if (cut := group_cuts.get(get_offload_group_idx(key))) is not None
                    and order[key] >= cut
                ]
                retried_keys = set(keys_to_retry)
                keys_to_store = [k for k in keys_to_store if k not in retried_keys]
                if not keys_to_store:
                    self._pin_keys(ready_existing_keys, req_context, time.monotonic())
                    return PrepareStoreOutput(
                        keys_to_store=[],
                        store_spec=self._get_load_store_spec([], []),
                        evicted_keys=[],
                        keys_to_retry=keys_to_retry,
                    )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        num_chunks_to_evict = len(keys_to_store) - self._get_num_free_chunks()

        to_evict: list[OffloadKey] = []
        if num_chunks_to_evict > 0:
            if num_chunks_to_evict > self._num_evictable_cache_chunks:
                # Eviction will fail.
                return None
            # There is a still a chance for eviction failure as some of the
            # idle chunks might be in the protected list.

            # Chunks from the original input are excluded from eviction candidates:
            # a chunk that was already stored must remain in the cache after this call.
            # Live pins are excluded as well; a pin the tier accepted is honored
            # rather than silently broken, so the store fails instead.
            now = time.monotonic()
            protected = set(keys) | self._unexpired_pins(now)
            evicted = self._policy.evict(num_chunks_to_evict, protected)
            if evicted is None:
                return None

            # cache-policy removes only idle chunks.
            self._num_evictable_cache_chunks -= len(evicted)
            assert self._num_evictable_cache_chunks >= 0

            for key, chunk in evicted:
                self._pins.pop(key, None)
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
            self._policy.insert(key, chunk, req_context)
        self._num_write_pending_chunks += len(keys_to_store)
        state.inserted_keys.update(keys_to_store)
        self._pin_keys(
            (*keys_to_store, *ready_existing_keys), req_context, time.monotonic()
        )

        # build store specs for allocated chunks
        store_spec = self._get_load_store_spec(keys_to_store, chunks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
            keys_to_retry=keys_to_retry,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        *,
        trigger: StoreTrigger,
        success: bool = True,
    ) -> None:
        self._complete_store(keys, req_context, success)

    def _complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []
        removed_keys: list[OffloadKey] = []

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
                    self._pins.pop(key, None)
                    self._free_chunk(chunk)
                    removed_keys.append(key)

        if removed_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=removed_keys,
                    medium=self.medium,
                    removed=True,
                    metadata_only=True,
                )
            )

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                    removal_expected=True,
                    req_context=req_context,
                )
            )

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        state = req_context.get_state(_RequestCacheAccess)
        if (
            state is None
            or state.owner is not self
            or state.cache_generation != self._cache_generation
            or state.finished
        ):
            return
        state.finished = True
        key_groups = []
        for group_idx in sorted(state.key_groups):
            keys = state.key_groups[group_idx]
            positions = {
                key: position
                for key in keys
                if (position := req_context.get_offload_key_position(key)) is not None
            }
            if len(positions) == len(keys):
                keys = sorted(keys, key=positions.__getitem__)
            key_groups.append(tuple(keys))
        self._policy.on_request_finished(
            tuple(key_groups),
            state.inserted_keys - state.reused_keys,
            state.reused_keys,
            req_context,
        )

    @override
    def reset_cache(self) -> None:
        # Clear ALL chunks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload chunk IDs.
        self._policy.clear()
        self._cache_generation += 1
        self._num_evictable_cache_chunks = 0
        self._num_write_pending_chunks = 0
        self._pins.clear()

        self._free_list.clear()
        self._num_allocated_chunks = 0

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is None:
            return
        # Detach the batch up front so a partially consumed drain cannot
        # replay cleanup against a key a retry has since re-allocated.
        events, self.events = self.events, []
        for event in events:
            if event.metadata_only:
                # A retry may already own this key before cleanup is drained.
                keys = [key for key in event.keys if self._policy.get(key) is None]
                if not keys:
                    continue
                event = replace(event, keys=keys)
            yield event

    @override
    def config_info(self) -> Sequence[ConfigInfoMapping]:
        """Report the CPU cache facts of the one cache this manager holds."""
        return [self._config_info]

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

        for (action_type, outcome), count in self._hint_outcome_counts.items():
            stats.increase_counter(
                CPUOffloadingMetrics.HINT_ACTIONS,
                count,
                labelvalues=(action_type, outcome),
            )
        self._hint_outcome_counts.clear()

        return stats
