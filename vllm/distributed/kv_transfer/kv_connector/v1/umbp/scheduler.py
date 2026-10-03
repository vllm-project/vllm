# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scheduler-side UMBP connector behavior."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    dcp_world_size_for_kv_cache_spec,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import (
    TRANSFER_NOT_ATTEMPTED,
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockTransferPlan,
    KVLayoutPlanner,
    LoadSpec,
    RankTopology,
    RequestTracker,
    StoreEventResult,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from .runtime import UMBPSchedulerHandle

logger = init_logger(__name__)


@lru_cache(maxsize=8192)
def _rank_keys(
    codec: BlockIdentityCodec,
    topology: RankTopology,
    block_hash: bytes,
    group_id: int,
) -> tuple[str, ...]:
    """Reuse immutable object identities, not runtime residency."""
    return codec.keys_for_topology(block_hash, topology, (group_id,))


@lru_cache(maxsize=8192)
def _decode_lazy_block_hash(
    codec: BlockIdentityCodec, block_hash_with_group: BlockHashWithGroupId
) -> tuple[int, bytes, str]:
    """Cache immutable identities, never GPU blocks or runtime residency."""
    group_id = get_group_id(block_hash_with_group)
    block_hash = get_block_hash(block_hash_with_group)
    return group_id, block_hash, codec.key(block_hash, group_id)


@dataclass
class _StoreEvent:
    plans: tuple[BlockTransferPlan, ...]
    result: StoreEventResult = field(default_factory=StoreEventResult)


@dataclass(frozen=True)
class _LookupContext:
    request: Request
    local_tokens: int
    num_tokens: int
    hashes: tuple[bytes, ...]
    objects: dict[int, list[tuple[bytes, tuple[str, ...]]]]
    keys: list[str]


@dataclass(frozen=True)
class _PendingLookup:
    context: _LookupContext
    future: Future[Sequence[bool]]


class _LookupBlockPool(BlockPool):
    """Request-scoped residency view for the core hit-window algorithms."""

    def __init__(self, hash_block_size: int) -> None:
        super().__init__(
            num_gpu_blocks=1,
            enable_caching=True,
            hash_block_size=hash_block_size,
            enable_kv_cache_events=False,
        )
        self.hits: list[set[bytes]] = []
        self.hit_blocks: dict[bytes, KVCacheBlock] = {}

    def get_cached_block(
        self, block_hash: bytes, kv_cache_group_ids: list[int]
    ) -> list[KVCacheBlock] | None:
        if any(block_hash not in self.hits[group] for group in kv_cache_group_ids):
            return None
        block = self.hit_blocks.get(block_hash)
        if block is None:
            block = KVCacheBlock(
                block_id=len(self.hit_blocks) + 1,
                _block_hash=make_block_hash_with_group_id(
                    BlockHash(block_hash), kv_cache_group_ids[0]
                ),
            )
            self.hit_blocks[block_hash] = block
        return [block] * len(kv_cache_group_ids)

    def clear_lookup(self) -> None:
        self.hits = []
        self.hit_blocks.clear()


class UMBPStoreConnectorScheduler:
    """Own vLLM prefix matching while the runtime owns lookup transport."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        runtime: UMBPSchedulerHandle,
        codec: BlockIdentityCodec,
        topology: RankTopology | None = None,
        *,
        layerwise_store: bool = False,
    ) -> None:
        self.block_size, resolved_hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.kv_cache_config = kv_cache_config
        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.group_block_sizes = {
            group_id: group.kv_cache_spec.block_size
            * dcp_world_size_for_kv_cache_spec(group.kv_cache_spec, dcp_size)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        }
        self.runtime = runtime
        self.layerwise_store = layerwise_store
        self.enable_kv_cache_events = bool(
            vllm_config.kv_events_config
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        self.codec = codec
        self.layout = KVLayoutPlanner.from_kv_cache_config(kv_cache_config)
        self.topology = topology or RankTopology()
        transfer_config = vllm_config.kv_transfer_config
        assert transfer_config is not None
        extra = transfer_config.kv_connector_extra_config
        self.load_async = bool(extra.get("load_async", True))
        if not self.load_async and any(
            isinstance(group.kv_cache_spec, MambaSpec)
            for group in kv_cache_config.kv_cache_groups
        ):
            # vLLM copies a restored Mamba state into the running slot while
            # preparing inputs, before a synchronous load can have landed.
            logger.warning(
                "UMBP loads asynchronously for models with Mamba-style layers; "
                "load_async=false is ignored"
            )
            self.load_async = True
        self.enable_lookup = bool(extra.get("enable_lookup", True))
        self.lookup_async = bool(extra.get("lookup_async", False))
        self.save_decode_cache = bool(extra.get("save_decode_cache", False))
        self.lazy_offload = bool(extra.get("lazy_offload", False))
        self.enable_partial_hash_hits = bool(
            extra.get("enable_partial_hash_hits", False)
        )
        self.hash_block_size = int(
            extra.get("hash_block_size", resolved_hash_block_size)
        )
        if self.hash_block_size <= 0 or self.block_size % self.hash_block_size:
            raise ValueError(
                "UMBP hash_block_size must be a positive divisor of block_size"
            )
        lookup_groups = [
            kv_cache_config.kv_cache_groups[group_id]
            for group_id in kv_cache_config.prefix_cacheable_group_ids
        ]
        model_config = getattr(vllm_config, "model_config", None)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        use_eagle = bool(
            speculative_config is not None and speculative_config.use_eagle_block_drop()
        )
        max_model_len = getattr(
            model_config,
            "max_model_len",
            kv_cache_config.num_blocks * self.block_size,
        )
        lookup_config = replace(
            kv_cache_config,
            kv_cache_groups=lookup_groups,
            num_blocks=1,
        )
        self._lookup_coordinator = (
            get_kv_cache_coordinator(
                kv_cache_config=lookup_config,
                max_model_len=max_model_len,
                max_in_flight_tokens=getattr(
                    vllm_config, "max_in_flight_tokens", max_model_len
                ),
                use_eagle=use_eagle,
                enable_caching=True,
                enable_kv_cache_events=False,
                dcp_world_size=dcp_size,
                pcp_world_size=getattr(
                    vllm_config.parallel_config,
                    "prefill_context_parallel_size",
                    1,
                ),
                scheduler_block_size=self.block_size,
                hash_block_size=self.hash_block_size,
                num_prefill_lookahead=getattr(
                    getattr(vllm_config, "scheduler_config", None),
                    "num_lookahead_slots",
                    0,
                ),
                allow_partial_hash_hits=self.enable_partial_hash_hits,
            )
            if len(lookup_groups) > 1
            else None
        )
        self._lookup_pool = _LookupBlockPool(self.hash_block_size)
        if self._lookup_coordinator is not None:
            self._lookup_coordinator.block_pool = self._lookup_pool
            for manager in self._lookup_coordinator.single_type_managers:
                manager.block_pool = self._lookup_pool
                manager._null_block = self._lookup_pool.null_block
        self._pending_loads: dict[str, list[BlockTransferPlan] | BlockLoadBatch] = {}
        self._load_specs: dict[str, LoadSpec] = {}
        self._lookup_units = {
            group_id: (
                self.hash_block_size
                if self.enable_partial_hash_hits
                and (
                    self._lookup_coordinator is None
                    or self._lookup_coordinator.enable_partial_hash_hits
                )
                else self.group_block_sizes[group_id]
            )
            for group_id in kv_cache_config.prefix_cacheable_group_ids
        }
        self._pending_lookups: dict[str, _PendingLookup] = {}
        self._lookup_executor = ThreadPoolExecutor(
            max_workers=int(extra.get("lookup_workers", 2)),
            thread_name_prefix="umbp-lookup",
        )
        # A distributed lookup keeps reporting objects of a node that died
        # until the master expires it. Treating a failed object as a miss for
        # a while stops a rescheduled request from retrying the same failing
        # load every step. Local pools forget such objects at once.
        quarantine_ms = extra.get(
            "load_failure_quarantine_ms",
            30000 if extra.get("mode") == "distributed" else 0,
        )
        if type(quarantine_ms) is not int or quarantine_ms < 0:
            raise ValueError(
                "load_failure_quarantine_ms must be a non-negative integer"
            )
        self._load_failure_quarantine_s = quarantine_ms / 1000
        self._quarantined_keys: dict[str, float] = {}
        self._requests: dict[str, Request] = {}
        self._request_trackers: dict[str, RequestTracker] = {}
        self._next_generation = 0
        self._pending_finished_stores: dict[str, list[BlockTransferPlan]] = {}
        self._gpu_block_pool: BlockPool | None = None
        self._store_events: dict[int, _StoreEvent] = {}
        self._num_workers = getattr(vllm_config.parallel_config, "world_size", 1)
        self._lazy_scan_pending = False
        self._lazy_prefix_chains_by_block: dict[
            int,
            tuple[tuple[tuple[int, str, int, bytes], ...], int],
        ] = {}
        configured_lazy_target = extra.get("lazy_offload_max_blocks")
        self._lazy_target_blocks = (
            int(configured_lazy_target)
            if configured_lazy_target is not None
            else self._estimate_lazy_target_blocks(
                kv_cache_config,
                getattr(
                    getattr(vllm_config, "scheduler_config", None),
                    "max_num_batched_tokens",
                    self.block_size,
                ),
                dcp_size,
            )
        )
        self._store_event_counter = 0

    @staticmethod
    def _estimate_lazy_target_blocks(
        kv_cache_config: KVCacheConfig,
        max_num_batched_tokens: int,
        dcp_size: int,
    ) -> int:
        target = 0
        prefix_group_ids = set(kv_cache_config.prefix_cacheable_group_ids)
        for group_id, group in enumerate(kv_cache_config.kv_cache_groups):
            if group_id not in prefix_group_ids:
                continue
            spec = group.kv_cache_spec
            block_size = spec.block_size * dcp_world_size_for_kv_cache_spec(
                spec, dcp_size
            )
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, block_size)
        return 2 * target

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if not self.enable_lookup:
            return 0, False
        hashes = tuple(request.block_hashes)
        pending = self._pending_lookups.get(request.request_id)
        if pending is not None:
            context = pending.context
            if (
                context.request is not request
                or context.local_tokens != num_computed_tokens
                or context.num_tokens != request.num_tokens
                or context.hashes != hashes
            ):
                self._pending_lookups.pop(request.request_id)
                pending.future.cancel()
                pending = None
        align = (
            self.hash_block_size if self.enable_partial_hash_hits else self.block_size
        )
        if not hashes or request.num_tokens < align:
            return 0, False
        if num_computed_tokens % self.block_size != 0:
            return 0, False
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        # vLLM must execute at least the final prompt token. Returning a hit
        # for the entire prompt leaves the scheduler with no new token to run.
        max_external_token = request.num_tokens - 1
        if max_external_token <= num_computed_tokens:
            self._load_specs.pop(request.request_id, None)
            return 0, False
        if pending is not None and not pending.future.done():
            return None, False
        if pending is None:
            context = self._build_lookup_context(request, num_computed_tokens, hashes)
            if self.lookup_async:
                self._pending_lookups[request.request_id] = _PendingLookup(
                    context,
                    self._lookup_executor.submit(self.runtime.lookup, context.keys),
                )
                return None, False
        else:
            context = self._pending_lookups.pop(request.request_id).context
        try:
            hits = list(
                pending.future.result()
                if pending is not None
                else self.runtime.lookup(context.keys)
            )
        except Exception as exc:
            logger.debug("UMBP lookup failed request=%s: %s", request.request_id, exc)
            return 0, False
        keys = context.keys
        logical_objects_by_group = context.objects
        if len(hits) != len(keys):
            logger.debug(
                "UMBP lookup returned an invalid result length request=%s",
                request.request_id,
            )
            return 0, False
        hits = self._drop_quarantined_hits(keys, hits)
        per_rank = self.topology.rank_count
        offset = 0
        logical_hits_by_group: dict[int, list[bool]] = {}
        for group_id in group_ids:
            group_size = len(logical_objects_by_group[group_id]) * per_rank
            group_hits = hits[offset : offset + group_size]
            offset += group_size
            logical_hits_by_group[group_id] = (
                group_hits
                if per_rank == 1
                else [
                    all(group_hits[index : index + per_rank])
                    for index in range(0, len(group_hits), per_rank)
                ]
            )
        if self._lookup_coordinator is None:
            group_id = group_ids[0]
            group_hits = logical_hits_by_group[group_id]
            units_per_block = (
                self.group_block_sizes[group_id] // self._lookup_units[group_id]
            )
            matched_units = 0
            for index in range(units_per_block - 1, len(group_hits), units_per_block):
                if not group_hits[index]:
                    break
                matched_units = index + 1
            # Interior hashes are sparse: only the saved tail needs to exist.
            for index in range(
                min(matched_units + units_per_block - 1, len(group_hits)) - 1,
                matched_units - 1,
                -1,
            ):
                if group_hits[index]:
                    matched_units = index + 1
                    break
            need_to_load = matched_units * self._lookup_units[group_id]
            block_hashes_by_group: tuple[tuple[bytes | None, ...], ...] = ()
        else:
            need_to_load, block_hashes_by_group = self._coordinate_external_hits(
                list(context.hashes),
                num_computed_tokens,
                max_external_token - num_computed_tokens,
                logical_objects_by_group,
                logical_hits_by_group,
            )
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "UMBP lookup result request=%s keys=%s group_hits=%s load_tokens=%d",
                request.request_id,
                keys,
                logical_hits_by_group,
                need_to_load,
            )
        matched_tokens = num_computed_tokens + need_to_load
        if need_to_load <= 0:
            return 0, False
        self._load_specs[request.request_id] = LoadSpec(
            local_tokens=num_computed_tokens,
            external_tokens=matched_tokens,
            block_hashes_by_group=block_hashes_by_group,
        )
        return need_to_load, self.load_async

    def _build_lookup_context(
        self, request: Request, local_tokens: int, hashes: tuple[bytes, ...]
    ) -> _LookupContext:
        objects: dict[int, list[tuple[bytes, tuple[str, ...]]]] = {}
        keys: list[str] = []
        for group_id, unit in self._lookup_units.items():
            group_objects = []
            for token_end in range(local_tokens + unit, request.num_tokens, unit):
                block_hash = self._object_hash_at_token_end(hashes, token_end)
                rank_keys = _rank_keys(self.codec, self.topology, block_hash, group_id)
                group_objects.append((block_hash, rank_keys))
                keys.extend(rank_keys)
            objects[group_id] = group_objects
        return _LookupContext(
            request, local_tokens, request.num_tokens, hashes, objects, keys
        )

    def _coordinate_external_hits(
        self,
        request_hashes: list[bytes],
        local_tokens: int,
        max_external_tokens: int,
        logical_objects_by_group: dict[int, list[tuple[bytes, tuple[str, ...]]]],
        logical_hits_by_group: dict[int, list[bool]],
    ) -> tuple[int, tuple[tuple[bytes | None, ...], ...]]:
        """Apply the core KV cache coordinator to runtime lookup results."""
        coordinator = self._lookup_coordinator
        assert coordinator is not None
        pool = self._lookup_pool
        try:
            pool.hits = [
                {
                    block_hash
                    for (block_hash, _), hit in zip(
                        logical_objects_by_group[group_id],
                        logical_hits_by_group[group_id],
                        strict=True,
                    )
                    if hit
                }
                for group_id in self.kv_cache_config.prefix_cacheable_group_ids
            ]
            skipped_hashes = local_tokens // self.hash_block_size
            hit_blocks, hit_length, _ = coordinator.find_longest_cache_hit(
                cast(list[BlockHash], request_hashes)[skipped_hashes:],
                max_external_tokens,
            )
            block_hashes_by_group = tuple(
                tuple(
                    None
                    if block.is_null or block.block_hash is None
                    else get_block_hash(block.block_hash)
                    for block in group_blocks
                )
                for group_blocks in hit_blocks
            )
            return hit_length, block_hashes_by_group
        finally:
            self._reset_lookup_coordinator()

    def _reset_lookup_coordinator(self) -> None:
        self._lookup_pool.clear_lookup()

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        self._requests[request.request_id] = request
        block_groups = blocks.get_block_ids(
            group_ids=self.kv_cache_config.prefix_cacheable_group_ids
        )
        if not block_groups:
            return
        tracker = self._tracker_for_request(request.request_id)
        tracker.update_blocks(tuple(list(group) for group in block_groups))
        if num_external_tokens <= 0:
            self._pending_loads.pop(request.request_id, None)
            self._load_specs.pop(request.request_id, None)
            return
        spec = self._load_specs.get(request.request_id)
        if spec is not None:
            tracker.load_spec = spec
            if num_external_tokens != spec.num_tokens_to_load:
                return
        plans = self._load_plans_for_external_tokens(
            request,
            tracker,
            block_groups,
            num_external_tokens,
        )
        self._pending_loads[request.request_id] = plans

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = UMBPConnectorMetadata(async_load=self.load_async)
        for request_id in scheduler_output.finished_req_ids:
            self._pending_loads.pop(request_id, None)
            self._load_specs.pop(request_id, None)
            self._request_trackers.pop(request_id, None)
            self._requests.pop(request_id, None)
            pending = self._pending_lookups.pop(request_id, None)
            if pending is not None:
                pending.future.cancel()
        for request in scheduler_output.scheduled_new_reqs:
            load_plans = self._pending_loads.pop(request.req_id, [])
            self._load_specs.pop(request.req_id, None)
            if load_plans:
                meta.load_requests[request.req_id] = load_plans
            else:
                tracker = self._tracker_for_request(request.req_id)
                total_tokens = (
                    request.num_computed_tokens
                    + scheduler_output.num_scheduled_tokens[request.req_id]
                )
                tracked_request = self._requests.get(request.req_id)
                selected_block_ids = tuple(
                    request.block_ids[group_id]
                    for group_id in (self.kv_cache_config.prefix_cacheable_group_ids)
                )
                store_plans = (
                    []
                    if self.lazy_offload or tracked_request is None
                    else self._store_plans(
                        tracked_request,
                        tracker,
                        total_tokens,
                        block_ids_override=selected_block_ids,
                    )
                )
                logger.debug(
                    "UMBP store planning request=%s tokens=%d plans=%d keys=%s",
                    request.req_id,
                    total_tokens,
                    len(store_plans),
                    tuple(plan.key for plan in store_plans),
                )
                meta.store_plans.extend(store_plans)
                if store_plans:
                    meta.store_requests[request.req_id] = store_plans

        cached_reqs = scheduler_output.scheduled_cached_reqs
        resumed_ids: set[str] = set(getattr(cached_reqs, "resumed_req_ids", ()))
        for request_id in cached_reqs.req_ids:
            load_plans = self._pending_loads.pop(request_id, [])
            self._load_specs.pop(request_id, None)
            if load_plans:
                meta.load_requests[request_id] = load_plans
            elif not self.lazy_offload:
                cached_request = self._requests.get(request_id)
                cached_tracker = self._request_trackers.get(request_id)
                if cached_request is not None and cached_tracker is not None:
                    request_index = cached_reqs.req_ids.index(request_id)
                    num_computed_tokens = cached_reqs.num_computed_tokens[request_index]
                    is_prefill = num_computed_tokens < cached_request.num_tokens
                    if not is_prefill and not self.save_decode_cache:
                        continue
                    new_block_ids = cached_reqs.new_block_ids[request_index]
                    if new_block_ids:
                        selected = tuple(
                            new_block_ids[group_id]
                            for group_id in (
                                self.kv_cache_config.prefix_cacheable_group_ids
                            )
                        )
                        if request_id in resumed_ids:
                            cached_tracker.replace_blocks(selected)
                        else:
                            cached_tracker.update_blocks(selected)
                    total_tokens = (
                        num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[request_id]
                    )
                    if not self.save_decode_cache:
                        total_tokens = min(total_tokens, cached_request.num_tokens)
                    store_plans = self._store_plans(
                        cached_request,
                        cached_tracker,
                        total_tokens,
                        block_ids_override=cached_tracker.block_ids,
                    )
                    meta.store_plans.extend(store_plans)
                    if store_plans:
                        meta.store_requests[request_id] = store_plans

        # Async loads are admitted without scheduling model tokens, so those
        # requests are absent from both scheduled request collections above.
        # Forward their plans explicitly so workers can start the transfer and
        # eventually move the request out of WAITING_FOR_REMOTE_KVS.
        for request_id, load_plans in list(self._pending_loads.items()):
            self._pending_loads.pop(request_id, None)
            self._load_specs.pop(request_id, None)
            if load_plans:
                meta.load_requests[request_id] = load_plans

        for request_id in scheduler_output.preempted_req_ids or set():
            meta.preempted_request_ids.add(request_id)
            plans = self._pending_loads.pop(request_id, [])
            meta.preempted_block_ids.update(plan.block_id for plan in plans)
            self._load_specs.pop(request_id, None)
            if preempted_tracker := self._request_trackers.get(request_id):
                preempted_tracker.reset()

        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        if block_state is not None and block_state.boundary_state_offloads:
            self._add_boundary_state_plans(
                block_state.boundary_state_offloads,
                meta,
                set(scheduler_output.finished_req_ids),
                set(scheduler_output.preempted_req_ids or ()),
            )

        finished_stores = self._pending_finished_stores
        self._pending_finished_stores = {}
        for plans in finished_stores.values():
            meta.store_plans.extend(plans)
        if self.lazy_offload:
            meta.store_plans.extend(self._prepare_lazy_store_plans())
            self._lazy_scan_pending = False
        event_plans = tuple(meta.store_plans)
        if event_plans:
            event = self._store_event_counter
            self._store_event_counter += 1
            meta.store_event = event
            self._store_events[event] = _StoreEvent(event_plans)
            if self._gpu_block_pool is not None:
                self._gpu_block_pool.touch(
                    [
                        self._gpu_block_pool.blocks[block_id]
                        for block_id in dict.fromkeys(
                            plan.block_id for plan in event_plans
                        )
                    ]
                )
                # Transfer ownership from request-finish pins to the event.
                for plans in finished_stores.values():
                    self._gpu_block_pool.free_blocks(
                        self._gpu_block_pool.blocks[block_id]
                        for block_id in dict.fromkeys(plan.block_id for plan in plans)
                    )
        if self.layerwise_store:
            self._group_layer_plans(meta)
        return meta

    def _prepare_lazy_store_plans(self) -> list[BlockTransferPlan]:
        pool = self._gpu_block_pool
        if pool is None or self._lazy_target_blocks <= 0:
            return []
        candidates: list[tuple[KVCacheBlock, str, int, bytes]] = []
        seen_keys: set[str] = set()
        expanded_chains: dict[int, int] = {}
        visited_by_group: dict[int, int] = {}
        unhashed = 0
        generation = self._next_generation
        self._next_generation += 1
        in_flight_keys = {
            plan.key for event in self._store_events.values() for plan in event.plans
        }
        aliases = getattr(pool, "cached_block_hashes_by_block", {})
        prefix_groups = self.kv_cache_config.prefix_cacheable_group_ids

        def add_candidate(
            block: KVCacheBlock,
            key: str,
            group_id: int,
            block_hash: bytes,
        ) -> None:
            if key in seen_keys or key in in_flight_keys:
                return
            seen_keys.add(key)
            candidates.append((block, key, group_id, block_hash))

        for covered, block in enumerate(pool.free_block_queue.iter_blocks_after(None)):
            if covered >= self._lazy_target_blocks:
                break
            if block.is_null or block.block_hash is None:
                unhashed += 1
                continue
            block_hashes = (
                block.block_hash,
                *aliases.get(block.block_id, ()),
            )
            for block_hash_with_group in block_hashes:
                group_id, block_hash, key = _decode_lazy_block_hash(
                    self.codec, block_hash_with_group
                )
                if group_id not in prefix_groups:
                    continue
                visited_by_group[group_id] = visited_by_group.get(group_id, 0) + 1
                add_candidate(block, key, group_id, block_hash)

            chain_entry = self._lazy_prefix_chains_by_block.get(block.block_id)
            if chain_entry is None:
                continue
            chain, chain_index = chain_entry
            # Chain snapshots are shared by their blocks and immutable during a scan.
            start = expanded_chains.get(id(chain), 0)
            if chain_index < start:
                continue
            expanded_chains[id(chain)] = chain_index + 1
            for block_id, key, group_id, block_hash in chain[start : chain_index + 1]:
                if key in seen_keys or key in in_flight_keys:
                    continue
                prefix_block = pool.blocks[block_id]
                expected_hash = make_block_hash_with_group_id(
                    BlockHash(block_hash), group_id
                )
                if prefix_block.is_null or (
                    expected_hash != prefix_block.block_hash
                    and expected_hash not in aliases.get(block_id, ())
                ):
                    continue
                add_candidate(prefix_block, key, group_id, block_hash)

        logger.debug(
            "UMBP lazy scan target=%d free=%d unhashed=%d keys_by_group=%s "
            "candidates=%d",
            self._lazy_target_blocks,
            getattr(pool.free_block_queue, "num_free_blocks", -1),
            unhashed,
            tuple(sorted(visited_by_group.items())),
            len(candidates),
        )
        if not candidates:
            return []

        hits = self.runtime.lookup([key for _, key, _, _ in candidates])
        if len(hits) != len(candidates):
            raise RuntimeError("UMBP lazy lookup returned an invalid result count")

        plans: list[BlockTransferPlan] = []
        for (block, key, group_id, block_hash), exists in zip(
            candidates, hits, strict=True
        ):
            if exists:
                continue
            plans.append(
                BlockTransferPlan(
                    key=key,
                    block_id=block.block_id,
                    generation=generation,
                    group_id=group_id,
                    block_hash=block_hash,
                    block_size=self.group_block_sizes[group_id],
                )
            )
        return plans

    def _add_boundary_state_plans(
        self,
        offloads: dict[str, list[tuple[int, int, int]]],
        meta: UMBPConnectorMetadata,
        finished: set[str],
        preempted: set[str],
    ) -> None:
        """Store exact Mamba/hybrid boundary blocks handed off by core."""
        prefix_groups = set(self.kv_cache_config.prefix_cacheable_group_ids)
        for request_id, entries in offloads.items():
            if request_id in finished or request_id in preempted:
                continue
            request = self._requests.get(request_id)
            tracker = self._request_trackers.get(request_id)
            if request is None or tracker is None:
                continue
            plans: list[BlockTransferPlan] = []
            for group_id, block_id, boundary_tokens in entries:
                if group_id not in prefix_groups or block_id == NULL_BLOCK_ID:
                    continue
                if boundary_tokens <= 0:
                    continue
                hash_index = boundary_tokens // self.hash_block_size - 1
                if not 0 <= hash_index < len(request.block_hashes):
                    continue
                block_hash = request.block_hashes[hash_index]
                plans.append(
                    BlockTransferPlan(
                        key=self.codec.key(block_hash, group_id),
                        block_id=block_id,
                        request_id=request_id,
                        generation=tracker.generation,
                        token_offset=max(
                            0, boundary_tokens - self.group_block_sizes[group_id]
                        ),
                        group_id=group_id,
                        block_hash=block_hash,
                        parent_block_hash=(
                            request.block_hashes[hash_index - 1]
                            if self.enable_kv_cache_events and hash_index > 0
                            else None
                        ),
                        block_size=self.group_block_sizes[group_id],
                    )
                )
            if plans:
                meta.store_plans.extend(plans)
                meta.store_requests.setdefault(request_id, []).extend(plans)

    def _group_layer_plans(self, meta: UMBPConnectorMetadata) -> None:
        """Expose layer ownership without changing bulk plan semantics."""
        for plan in meta.store_plans:
            layer_names = {item.layer_name for item in plan.ranges} or {
                region.layer_name
                for region in self.layout.regions
                if plan.group_id is None or region.group_id == plan.group_id
            }
            for layer_name in layer_names:
                meta.store_plans_by_layer.setdefault(layer_name, []).append(plan)

    def _tracker_for_request(self, request_id: str) -> RequestTracker:
        tracker = self._request_trackers.get(request_id)
        if tracker is None:
            tracker = RequestTracker(generation=self._next_generation)
            self._next_generation += 1
            self._request_trackers[request_id] = tracker
        return tracker

    def _load_plans_for_external_tokens(
        self,
        request: Any,
        tracker: RequestTracker,
        block_groups: tuple[list[int], ...],
        num_external_tokens: int,
    ) -> BlockLoadBatch:
        """Batch the logical objects selected by the external hit window."""
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        spec = tracker.load_spec
        local_tokens = spec.local_tokens if spec is not None else 0
        end_tokens = local_tokens + num_external_tokens
        keys: list[str] = []
        destinations: list[int] = []
        groups: list[int] = []
        if spec is not None and spec.block_hashes_by_group:
            for group_index, group_id in enumerate(group_ids):
                group_block_size = self.group_block_sizes[group_id]
                start_block = local_tokens // group_block_size
                for relative_index, block_hash in enumerate(
                    spec.block_hashes_by_group[group_index]
                ):
                    if block_hash is None:
                        continue
                    block_index = start_block + relative_index
                    block_id = block_groups[group_index][block_index]
                    if block_id == NULL_BLOCK_ID:
                        raise RuntimeError(
                            "UMBP external hit mapped a runtime object to a null "
                            f"destination block for group {group_id}"
                        )
                    keys.append(self.codec.key(block_hash, group_id=group_id))
                    destinations.append(block_id)
                    groups.append(group_id)
            return BlockLoadBatch(
                keys, destinations, groups, request.request_id, tracker.generation
            )
        for group_index, group_id in enumerate(group_ids):
            group_block_size = self.group_block_sizes[group_id]
            start_block = local_tokens // group_block_size
            num_full_blocks = end_tokens // group_block_size
            for block_index in range(start_block, num_full_blocks):
                token_end = (block_index + 1) * group_block_size
                keys.append(
                    self.codec.key(
                        self._object_hash_at_token_end(request.block_hashes, token_end),
                        group_id=group_id,
                    )
                )
                destinations.append(block_groups[group_index][block_index])
                groups.append(group_id)
        partial_tokens = end_tokens % self.block_size
        if partial_tokens and self.enable_partial_hash_hits:
            hash_index = end_tokens // self.hash_block_size - 1
            for group_index, group_id in enumerate(group_ids):
                block_index = end_tokens // self.group_block_sizes[group_id]
                group_block_ids = block_groups[group_index]
                if (
                    block_index >= len(group_block_ids)
                    or group_block_ids[block_index] == NULL_BLOCK_ID
                ):
                    raise RuntimeError("UMBP partial hit has no destination block")
                keys.append(self.codec.key(request.block_hashes[hash_index], group_id))
                destinations.append(group_block_ids[block_index])
                groups.append(group_id)
        return BlockLoadBatch(
            keys, destinations, groups, request.request_id, tracker.generation
        )

    def _object_hash_at_token_end(
        self, hashes: Sequence[bytes], token_end: int
    ) -> bytes:
        hash_index = token_end // self.hash_block_size - 1
        if hash_index < 0 or hash_index >= len(hashes):
            raise ValueError(
                f"request has {len(hashes)} hashes, needs index {hash_index}"
            )
        return hashes[hash_index]

    def _store_plans(
        self,
        request: Any,
        tracker: RequestTracker,
        token_count: int,
        block_ids_override: tuple[list[int], ...] | None = None,
    ) -> list[BlockTransferPlan]:
        """Describe full blocks produced by a scheduled prefill."""
        request_id = getattr(request, "request_id", None) or request.req_id
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        previous_saved_tokens = (
            tracker.retry_from_tokens
            if tracker.retry_from_tokens is not None
            else tracker.saved_tokens
        )
        save_to = tracker.mark_saved(token_count, self.block_size)
        plans: list[BlockTransferPlan] = []
        block_groups = (
            block_ids_override
            if block_ids_override is not None
            else tuple(request.block_ids[group_id] for group_id in group_ids)
        )
        block_ranges = {
            group_id: range(
                previous_saved_tokens // self.group_block_sizes[group_id],
                min(save_to // self.group_block_sizes[group_id], len(block_ids)),
            )
            for group_id, block_ids in zip(group_ids, block_groups)
        }
        resident_blocks = (
            self._restored_store_blocks(
                request, block_ranges, tracker.load_spec.external_tokens, block_groups
            )
            if tracker.load_spec is not None
            else {}
        )
        for group_id, block_ids in zip(group_ids, block_groups):
            group_block_size = self.group_block_sizes[group_id]
            resident = resident_blocks.get(group_id, ())
            for index in block_ranges[group_id]:
                if block_ids[index] == NULL_BLOCK_ID or index in resident:
                    continue
                token_end = (index + 1) * group_block_size
                block_hash = self._object_hash_at_token_end(
                    request.block_hashes, token_end
                )
                plans.append(
                    BlockTransferPlan(
                        key=self.codec.key(block_hash, group_id=group_id),
                        block_id=block_ids[index],
                        request_id=request_id,
                        generation=tracker.generation,
                        token_offset=index * group_block_size,
                        group_id=group_id,
                        block_hash=block_hash,
                        parent_block_hash=(
                            self._object_hash_at_token_end(
                                request.block_hashes, token_end - group_block_size
                            )
                            if self.enable_kv_cache_events and index > 0
                            else None
                        ),
                        token_ids=(
                            tuple(
                                request.prompt_token_ids[
                                    index * group_block_size : token_end
                                ]
                            )
                            if self.enable_kv_cache_events and request.prompt_token_ids
                            else ()
                        ),
                        block_size=group_block_size,
                    )
                )
        return plans

    def _restored_store_blocks(
        self,
        request: Any,
        block_ranges: dict[int, range],
        restored_tokens: int,
        block_groups: tuple[list[int], ...],
    ) -> dict[int, set[int]]:
        """Query residency before constructing plans for restored blocks."""
        candidates = {
            group_id: [
                index
                for index in range(
                    indices.start,
                    min(
                        indices.stop,
                        restored_tokens // self.group_block_sizes[group_id],
                    ),
                )
                if block_ids[index] != NULL_BLOCK_ID
            ]
            for (group_id, indices), block_ids in zip(
                block_ranges.items(), block_groups, strict=True
            )
        }
        keys = [
            key
            for group_id, indices in candidates.items()
            for index in indices
            for key in _rank_keys(
                self.codec,
                self.topology,
                self._object_hash_at_token_end(
                    request.block_hashes, (index + 1) * self.group_block_sizes[group_id]
                ),
                group_id,
            )
        ]
        if not keys:
            return {}
        try:
            hits = list(self.runtime.lookup(keys))
        except Exception as exc:
            logger.debug("UMBP store residency check failed: %s", exc)
            return {}
        if len(hits) != len(keys):
            return {}
        ranks = self.topology.rank_count
        offset = 0
        resident = {}
        for group_id, indices in candidates.items():
            resident[group_id] = {
                index
                for position, index in enumerate(indices)
                if all(
                    hits[offset + position * ranks : offset + (position + 1) * ranks]
                )
            }
            offset += len(indices) * ranks
        return resident

    def request_finished(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._pending_loads.pop(request.request_id, None)
        self._load_specs.pop(request.request_id, None)
        self._requests.pop(request.request_id, None)
        pending = self._pending_lookups.pop(request.request_id, None)
        if pending is not None:
            pending.future.cancel()
        if self.lazy_offload:
            self._register_lazy_prefix_chains(request, block_ids)
            self._request_trackers.pop(request.request_id, None)
            self._lazy_scan_pending = True
        return False, None

    def _register_lazy_prefix_chains(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
    ) -> None:
        """Remember full-attention chains so an at-risk tail stores its prefix."""
        prompt_tokens = getattr(
            request,
            "num_prompt_tokens",
            getattr(request, "num_tokens", 0),
        )
        computed_tokens = getattr(request, "num_computed_tokens", prompt_tokens)
        save_to = min(prompt_tokens, computed_tokens)
        hashes = list(getattr(request, "block_hashes", ()))
        if save_to <= 0 or not hashes:
            return

        for group_id in self.kv_cache_config.prefix_cacheable_group_ids:
            spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            if not isinstance(spec, FullAttentionSpec) or group_id >= len(block_ids):
                continue
            group_block_size = self.group_block_sizes[group_id]
            num_blocks = min(save_to // group_block_size, len(block_ids[group_id]))
            chain: list[tuple[int, str, int, bytes]] = []
            for index, block_id in enumerate(block_ids[group_id][:num_blocks]):
                if block_id == NULL_BLOCK_ID:
                    continue
                token_end = (index + 1) * group_block_size
                block_hash = self._object_hash_at_token_end(hashes, token_end)
                chain.append(
                    (
                        block_id,
                        self.codec.key(block_hash, group_id),
                        group_id,
                        block_hash,
                    )
                )
            frozen_chain = tuple(chain)
            for index, (block_id, _, _, _) in enumerate(frozen_chain):
                self._lazy_prefix_chains_by_block[block_id] = (frozen_chain, index)

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Pin finished boundary pages without delaying core request cleanup."""
        pool = self._gpu_block_pool
        if not partial_tail_offloads or pool is None:
            return False
        if request.request_id in self._pending_finished_stores:
            return False
        prefix_groups = set(self.kv_cache_config.prefix_cacheable_group_ids)
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None:
            return False
        boundaries = {entry[2] for entry in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError("partial tail entries must share a boundary")
        boundary = boundaries.pop()
        if (
            boundary <= 0
            or boundary % self.hash_block_size != 0
            or boundary != request.num_computed_tokens
            or bool(getattr(request, "num_in_flight_tokens", 0))
        ):
            return False
        hash_index = boundary // self.hash_block_size - 1
        if hash_index >= len(request.block_hashes):
            return False
        sources = {
            group_id: block_id for group_id, block_id, _ in partial_tail_offloads
        }
        if not sources.keys() <= prefix_groups:
            return False
        # Core hands off Mamba state pages. Include the matching FA tail page
        # so every group can resolve the same fine-grained prefix boundary.
        for group_id in prefix_groups:
            spec = self.kv_cache_config.kv_cache_groups[group_id].kv_cache_spec
            if (
                isinstance(spec, FullAttentionSpec)
                and boundary % self.group_block_sizes[group_id]
            ):
                index = boundary // self.group_block_sizes[group_id]
                if index >= len(block_ids[group_id]):
                    return False
                sources.setdefault(group_id, block_ids[group_id][index])
        plans: list[BlockTransferPlan] = []
        for group_id, block_id in sources.items():
            if block_id == NULL_BLOCK_ID or not 0 <= block_id < len(pool.blocks):
                return False
            if pool.blocks[block_id].is_null:
                return False
            group_block_size = self.group_block_sizes[group_id]
            plans.append(
                BlockTransferPlan(
                    request_id=request.request_id,
                    generation=tracker.generation,
                    block_id=block_id,
                    group_id=group_id,
                    token_offset=(boundary - 1) // group_block_size * group_block_size,
                    key=self.codec.key(request.block_hashes[hash_index], group_id),
                    block_hash=request.block_hashes[hash_index],
                    block_size=group_block_size,
                )
            )
        # A Mamba page is a state snapshot, not a token-addressable byte array.
        # Store full pages for both state and attention, matching restore.
        pool.touch(
            [pool.blocks[block_id] for block_id in dict.fromkeys(sources.values())]
        )
        self._pending_finished_stores[request.request_id] = plans
        return False

    def _drop_quarantined_hits(self, keys: list[str], hits: list[bool]) -> list[bool]:
        if not self._quarantined_keys:
            return hits
        now = time.monotonic()
        self._quarantined_keys = {
            key: until for key, until in self._quarantined_keys.items() if until > now
        }
        return [
            hit and key not in self._quarantined_keys
            for key, hit in zip(keys, hits, strict=True)
        ]

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        metadata = output.kv_connector_worker_meta
        if not isinstance(metadata, UMBPConnectorWorkerMetadata):
            return
        # A later store of the same key does not lift the quarantine: a pool
        # that still lists the dead copy reports that store as already done.
        if self._load_failure_quarantine_s > 0 and metadata.failed_loads:
            until = time.monotonic() + self._load_failure_quarantine_s
            for key, error in metadata.failed_loads.items():
                # A load that was never attempted says nothing about the object.
                if not error.startswith(TRANSFER_NOT_ATTEMPTED):
                    self._quarantined_keys[key] = until
        pool = self._gpu_block_pool
        for event_id, result in metadata.store_events.items():
            event = self._store_events.get(event_id)
            if event is None:
                continue
            event.result.merge(result)
            if event.result.completed_workers < self._num_workers:
                continue
            del self._store_events[event_id]
            if pool is not None:
                block_ids = dict.fromkeys(plan.block_id for plan in event.plans)
                pool.free_blocks(
                    pool.blocks[block_id] for block_id in reversed(block_ids)
                )
            request_plans: dict[str, list[BlockTransferPlan]] = {}
            for plan in event.plans:
                if plan.request_id is not None:
                    request_plans.setdefault(plan.request_id, []).append(plan)
            for request_id, plans in request_plans.items():
                tracker = self._request_trackers.get(request_id)
                if tracker is None:
                    continue
                current = [
                    plan for plan in plans if plan.generation == tracker.generation
                ]
                failed = [
                    plan.token_offset
                    for plan in current
                    if (plan.key, plan.generation) in event.result.failed_tokens
                ]
                if failed:
                    tracker.record_store_failure(min(failed))
                elif tracker.retry_from_tokens is not None and any(
                    plan.token_offset
                    <= tracker.retry_from_tokens
                    < plan.token_offset + plan.block_size
                    for plan in current
                ):
                    tracker.clear_store_retry()

    def has_pending_push_work(self) -> bool:
        return (
            self._lazy_scan_pending
            or bool(self._pending_finished_stores)
            or bool(self._store_events)
        )

    def reset_store(self) -> bool:
        if self._store_events or self._pending_finished_stores:
            return False
        for pending in self._pending_lookups.values():
            pending.future.cancel()
        self._pending_lookups.clear()
        self._quarantined_keys.clear()
        self._load_specs.clear()
        self._pending_loads.clear()
        return self.runtime.clear()

    def close(self) -> None:
        self._lookup_executor.shutdown(wait=True, cancel_futures=True)
        self._pending_lookups.clear()
        self.runtime.close()
