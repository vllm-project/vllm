# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scheduler-side UMBP connector behavior."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlock,
    dcp_world_size_for_kv_cache_spec,
    get_block_hash,
    get_group_id,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import (
    BlockIdentityCodec,
    BlockTransferPlan,
    KVLayoutPlanner,
    LoadSpec,
    LookupState,
    PartialTailPlan,
    RankCompletenessPolicy,
    RankTopology,
    RequestTracker,
    TPShardMapping,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from .runtime import UMBPSchedulerHandle

logger = init_logger(__name__)


class UMBPStoreConnectorScheduler:
    """Own vLLM prefix matching while the runtime owns lookup transport."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        runtime: UMBPSchedulerHandle,
        codec: BlockIdentityCodec,
        topology: RankTopology | None = None,
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
        self.codec = codec
        self.layout = KVLayoutPlanner.from_kv_cache_config(kv_cache_config)
        self.topology = topology or RankTopology()
        self.completeness = RankCompletenessPolicy(self.topology)
        self.tp_shard_mapping = self._build_local_tp_mapping()
        self.layout_descriptor = self.layout.describe(
            self.topology,
            tp_shard_mapping=self.tp_shard_mapping,
        )
        transfer_config = vllm_config.kv_transfer_config
        assert transfer_config is not None
        extra = transfer_config.kv_connector_extra_config
        self.load_async = bool(extra.get("load_async", True))
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
        self._pending_loads: dict[str, list[BlockTransferPlan]] = {}
        self._load_specs: dict[str, LoadSpec] = {}
        self._lookup_states: dict[str, LookupState] = {}
        self._lookup_futures: dict[str, Future[list[bool]]] = {}
        self._lookup_executor = ThreadPoolExecutor(
            max_workers=int(extra.get("lookup_workers", 2)),
            thread_name_prefix="umbp-lookup",
        )
        self._requests: dict[str, Request] = {}
        self._request_trackers: dict[str, RequestTracker] = {}
        self._next_generation = 0
        self._pending_stores: list[BlockTransferPlan] = []
        self._pending_partial_tails: dict[str, list[PartialTailPlan]] = {}
        self._gpu_block_pool: BlockPool | None = None
        self._pinned_store_blocks: dict[tuple[str, int], list[int]] = {}
        self._store_plan_requests: dict[tuple[str, int], tuple[str, int]] = {}
        self._num_workers = getattr(vllm_config.parallel_config, "world_size", 1)
        self._lazy_scan_pending = False
        self._lazy_cursor: KVCacheBlock | None = None
        self._lazy_max_blocks = int(extra.get("lazy_offload_max_blocks", 64))
        self._store_event_counter = 0
        self._store_event_tokens: dict[int, set[tuple[str, int]]] = {}
        self._store_event_pending_counts: dict[int, int] = {}

    def _build_local_tp_mapping(self) -> TPShardMapping | None:
        """Build the identity mapping for homogeneous local TP."""
        heads = [
            getattr(group.kv_cache_spec, "num_kv_heads", None)
            for group in self.kv_cache_config.prefix_cacheable_groups
        ]
        if not heads or heads[0] is None or any(head != heads[0] for head in heads):
            return None
        return TPShardMapping.build(
            self.topology.tp_size,
            self.topology.tp_size,
            self.topology.tp_rank,
            heads[0],
        )

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._gpu_block_pool = gpu_block_pool

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if not self.enable_lookup:
            return 0, False
        lookup_state = self._lookup_states.setdefault(
            request.request_id, LookupState(request.request_id)
        )
        hashes = list(request.block_hashes)
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
        endpoints_by_group = {
            group_id: range(
                num_computed_tokens + self.group_block_sizes[group_id],
                max_external_token + 1,
                self.group_block_sizes[group_id],
            )
            for group_id in group_ids
        }
        keys_by_group = {
            group_id: [
                key
                for token_end in endpoints
                for key in self.codec.keys_for_topology(
                    self._object_hash_at_token_end(hashes, token_end),
                    self.topology,
                    (group_id,),
                )
            ]
            for group_id, endpoints in endpoints_by_group.items()
        }
        keys = [key for group_keys in keys_by_group.values() for key in group_keys]
        if self.lookup_async:
            future = self._lookup_futures.get(request.request_id)
            if future is None:
                future = self._lookup_executor.submit(
                    lambda: list(self.runtime.lookup(keys))
                )
                self._lookup_futures[request.request_id] = future
                return None, False
            if not future.done():
                return None, False
            self._lookup_futures.pop(request.request_id, None)
            try:
                hits = future.result()
            except Exception as exc:
                lookup_state.fail(str(exc))
                return 0, False
        else:
            try:
                hits = list(self.runtime.lookup(keys))
            except Exception as exc:
                lookup_state.fail(str(exc))
                return 0, False
        if len(hits) != len(keys):
            lookup_state.fail("lookup returned an invalid result length")
            return 0, False
        logger.debug(
            "UMBP lookup request=%s groups=%s keys=%d hits=%d lookup_keys=%s",
            request.request_id,
            tuple((group_id, len(keys_by_group[group_id])) for group_id in group_ids),
            len(keys),
            sum(hits),
            tuple(keys),
        )
        per_rank = self.completeness.required_rank_count
        if per_rank == 0 or len(hits) != len(keys):
            lookup_state.fail("lookup returned an invalid result length")
            return 0, False
        offset = 0
        matched_tokens_by_group: list[int] = []
        for group_id in group_ids:
            group_hits = hits[offset : offset + len(keys_by_group[group_id])]
            offset += len(group_hits)
            group_units = 0
            for index in range(0, len(group_hits), per_rank):
                if not all(group_hits[index : index + per_rank]):
                    break
                group_units += 1
            matched_tokens_by_group.append(
                group_units * self.group_block_sizes[group_id]
            )
        need_to_load = min(matched_tokens_by_group, default=0)
        need_to_load = need_to_load // self.block_size * self.block_size
        need_to_load = min(need_to_load, max_external_token - num_computed_tokens)
        logger.info(
            "UMBP lookup result request=%s group_tokens=%s load_tokens=%d",
            request.request_id,
            tuple(matched_tokens_by_group),
            need_to_load,
        )
        matched_tokens = num_computed_tokens + need_to_load
        lookup_state.complete(matched_tokens)
        if need_to_load <= 0:
            return 0, False
        self._load_specs[request.request_id] = LoadSpec(
            local_tokens=num_computed_tokens,
            external_tokens=matched_tokens,
        )
        return need_to_load, self.load_async

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
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None:
            tracker = RequestTracker(
                request_id=request.request_id,
                generation=self._next_generation,
            )
            self._next_generation += 1
            self._request_trackers[request.request_id] = tracker
        tracker.update_blocks(tuple(list(group) for group in block_groups))
        if num_external_tokens <= 0:
            self._pending_loads.pop(request.request_id, None)
            self._load_specs.pop(request.request_id, None)
            return
        spec = self._load_specs.get(request.request_id)
        if spec is not None:
            spec.can_load = num_external_tokens == spec.num_tokens_to_load
            tracker.load_spec = spec
            if not spec.can_load:
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
            self._lookup_states.pop(request_id, None)
            future = self._lookup_futures.pop(request_id, None)
            if future is not None:
                future.cancel()
        for request in scheduler_output.scheduled_new_reqs:
            load_plans = self._pending_loads.pop(request.req_id, [])
            self._load_specs.pop(request.req_id, None)
            meta.load_plans.extend(load_plans)
            if load_plans:
                meta.load_requests[request.req_id] = load_plans
            else:
                tracker = self._tracker_for_request(request.req_id)
                tracker.save_mode = "lazy" if self.lazy_offload else "eager"
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
            if request.req_id in self._lookup_states:
                meta.lookup_states[request.req_id] = self._lookup_states[request.req_id]

        cached_reqs = scheduler_output.scheduled_cached_reqs
        resumed_ids: set[str] = set(getattr(cached_reqs, "resumed_req_ids", ()))
        for request_id in cached_reqs.req_ids:
            load_plans = self._pending_loads.pop(request_id, [])
            self._load_specs.pop(request_id, None)
            if load_plans:
                meta.load_plans.extend(load_plans)
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
                    cached_tracker.token_len = total_tokens
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
            meta.load_plans.extend(load_plans)
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

        for partial_plans in self._pending_partial_tails.values():
            meta.partial_tail_plans.extend(partial_plans)
        self._pending_partial_tails.clear()
        if self.lazy_offload and self._lazy_scan_pending:
            lazy_plans = self._prepare_lazy_store_plans()
            self._pending_stores.extend(lazy_plans)
            self._lazy_scan_pending = False
        meta.deferred_store_requests.update(
            plan.request_id
            for plan in self._pending_stores
            if plan.request_id is not None
        )
        meta.store_plans.extend(self._pending_stores)
        self._pending_stores.clear()
        self._reference_store_blocks(meta)
        if self.lazy_offload and meta.store_plans:
            event = self._store_event_counter
            self._store_event_counter += 1
            meta.store_event = event
            self._store_event_tokens[event] = {
                (plan.key, plan.generation) for plan in meta.store_plans
            }
        self._group_layer_plans(meta)
        return meta

    def _prepare_lazy_store_plans(self) -> list[BlockTransferPlan]:
        pool = self._gpu_block_pool
        if pool is None or self._lazy_max_blocks <= 0:
            return []
        if self._lazy_cursor is not None and self._lazy_cursor.ref_cnt > 0:
            self._lazy_cursor = None
        plans: list[BlockTransferPlan] = []
        generation = self._next_generation
        self._next_generation += 1
        for block in pool.free_block_queue.iter_blocks_after(self._lazy_cursor):
            self._lazy_cursor = block
            if block.is_null or block.block_hash is None:
                continue
            group_id = get_group_id(block.block_hash)
            block_hash = get_block_hash(block.block_hash)
            key = self.codec.key(block_hash, group_id)
            if any(token[0] == key for token in self._pinned_store_blocks):
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
            if len(plans) >= self._lazy_max_blocks:
                break
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
                        group_id=group_id,
                        block_hash=block_hash,
                        parent_block_hash=(
                            request.block_hashes[hash_index - 1]
                            if hash_index > 0
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
        for plan in meta.load_plans:
            layer_names = {item.layer_name for item in plan.ranges} or {
                region.layer_name
                for region in self.layout.regions
                if plan.group_id is None or region.group_id == plan.group_id
            }
            for layer_name in layer_names:
                meta.load_plans_by_layer.setdefault(layer_name, []).append(plan)
        for plan in meta.store_plans:
            layer_names = {item.layer_name for item in plan.ranges} or {
                region.layer_name
                for region in self.layout.regions
                if plan.group_id is None or region.group_id == plan.group_id
            }
            for layer_name in layer_names:
                meta.store_plans_by_layer.setdefault(layer_name, []).append(plan)

    def _reference_store_blocks(self, metadata: UMBPConnectorMetadata) -> None:
        if self._gpu_block_pool is None:
            return
        new_block_ids: list[int] = []
        for plan in metadata.store_plans:
            token = (plan.key, plan.generation)
            blocks = self._pinned_store_blocks.setdefault(token, [])
            if plan.request_id is not None:
                self._store_plan_requests.setdefault(
                    token, (plan.request_id, plan.block_id * self.block_size)
                )
            if plan.block_id not in blocks:
                blocks.append(plan.block_id)
                new_block_ids.append(plan.block_id)
        if new_block_ids:
            self._gpu_block_pool.touch(
                [self._gpu_block_pool.blocks[block_id] for block_id in new_block_ids]
            )

    def _tracker_for_request(self, request_id: str) -> RequestTracker:
        tracker = self._request_trackers.get(request_id)
        if tracker is None:
            tracker = RequestTracker(
                request_id=request_id,
                generation=self._next_generation,
            )
            self._next_generation += 1
            self._request_trackers[request_id] = tracker
        return tracker

    def _load_plans_for_external_tokens(
        self,
        request: Any,
        tracker: RequestTracker,
        block_groups: tuple[list[int], ...],
        num_external_tokens: int,
    ) -> list[BlockTransferPlan]:
        """Build full and partial-prefix load plans for one external hit."""
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        spec = tracker.load_spec
        local_tokens = spec.local_tokens if spec is not None else 0
        end_tokens = local_tokens + num_external_tokens
        plans: list[BlockTransferPlan] = []
        for group_index, group_id in enumerate(group_ids):
            group_block_size = self.group_block_sizes[group_id]
            start_block = local_tokens // group_block_size
            num_full_blocks = end_tokens // group_block_size
            for block_index in range(start_block, num_full_blocks):
                token_end = (block_index + 1) * group_block_size
                plans.append(
                    BlockTransferPlan(
                        key=self.codec.key(
                            self._object_hash_at_token_end(
                                request.block_hashes, token_end
                            ),
                            group_id=group_id,
                        ),
                        block_id=block_groups[group_index][block_index],
                        request_id=request.request_id,
                        generation=tracker.generation,
                        group_id=group_id,
                    )
                )
        partial_tokens = end_tokens % self.block_size
        if partial_tokens and self.enable_partial_hash_hits:
            block_index = start_block + num_full_blocks
            hash_index = end_tokens // self.hash_block_size - 1
            token_start = 0
            token_end = partial_tokens
            for group_index, group_id in enumerate(group_ids):
                group_block_ids = block_groups[group_index]
                if block_index >= len(group_block_ids):
                    continue
                plans.append(
                    BlockTransferPlan(
                        key=self.codec.key(
                            request.block_hashes[hash_index],
                            group_id=group_id,
                        ),
                        block_id=group_block_ids[block_index],
                        request_id=request.request_id,
                        generation=tracker.generation,
                        group_id=group_id,
                        token_start=token_start,
                        token_end=token_end,
                    )
                )
        return plans

    def _object_hash(self, hashes: list[bytes], block_index: int) -> bytes:
        scale = self.block_size // self.hash_block_size
        hash_index = block_index * scale + scale - 1
        if hash_index >= len(hashes):
            raise ValueError(
                f"request has {len(hashes)} hashes, needs index {hash_index}"
            )
        return hashes[hash_index]

    def _object_hash_at_token_end(self, hashes: list[bytes], token_end: int) -> bytes:
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
        for group_id in group_ids:
            block_index = group_ids.index(group_id)
            block_ids = block_groups[block_index]
            group_block_size = self.group_block_sizes[group_id]
            start_block = previous_saved_tokens // group_block_size
            num_blocks = save_to // group_block_size
            for index, block_id in enumerate(
                block_ids[start_block:num_blocks], start=start_block
            ):
                token_end = (index + 1) * group_block_size
                block_hash = self._object_hash_at_token_end(
                    request.block_hashes, token_end
                )
                plans.append(
                    BlockTransferPlan(
                        key=self.codec.key(block_hash, group_id=group_id),
                        block_id=block_id,
                        request_id=request_id,
                        generation=tracker.generation,
                        group_id=group_id,
                        block_hash=block_hash,
                        parent_block_hash=(
                            self._object_hash_at_token_end(
                                request.block_hashes, token_end - group_block_size
                            )
                            if index > 0 and token_end > group_block_size
                            else None
                        ),
                        token_ids=tuple(getattr(request, "prompt_token_ids", []))[
                            index * group_block_size : token_end
                        ],
                        block_size=group_block_size,
                    )
                )
        return plans

    def request_finished(
        self, request: Request, block_ids: tuple[list[int], ...]
    ) -> tuple[bool, dict[str, Any] | None]:
        self._pending_loads.pop(request.request_id, None)
        self._load_specs.pop(request.request_id, None)
        self._requests.pop(request.request_id, None)
        future = self._lookup_futures.pop(request.request_id, None)
        if future is not None:
            future.cancel()
        if self.lazy_offload:
            self._request_trackers.pop(request.request_id, None)
            self._lazy_scan_pending = True
        return False, None

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Queue partial-tail stores for one or more prefix-cacheable groups."""
        del block_ids
        if not partial_tail_offloads:
            return False
        prefix_groups = set(self.kv_cache_config.prefix_cacheable_group_ids)
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None:
            return False
        boundaries = {entry[2] for entry in partial_tail_offloads}
        if len(boundaries) != 1:
            raise ValueError("partial tail entries must share a boundary")
        boundary = boundaries.pop()
        if boundary <= 0 or boundary % self.hash_block_size == 0:
            return False
        hash_index = boundary // self.hash_block_size - 1
        if hash_index >= len(request.block_hashes):
            return False
        start_token = (boundary - 1) // self.block_size * self.block_size
        plans: list[PartialTailPlan] = []
        for entry_group_id, block_id, entry_boundary in partial_tail_offloads:
            if entry_group_id not in prefix_groups:
                return False
            if entry_boundary != boundary:
                return False
            group_block_size = self.group_block_sizes[entry_group_id]
            plans.append(
                PartialTailPlan(
                    request_id=request.request_id,
                    generation=tracker.generation,
                    block_id=block_id,
                    group_id=entry_group_id,
                    start_token=start_token,
                    end_token=boundary,
                    key=self.codec.key(
                        request.block_hashes[hash_index], entry_group_id
                    ),
                    block_size=group_block_size,
                )
            )
        self._pending_partial_tails[request.request_id] = plans
        return True

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        metadata = output.kv_connector_worker_meta
        if not isinstance(metadata, UMBPConnectorWorkerMetadata):
            return
        pool = self._gpu_block_pool
        terminal_counts = dict(metadata.completed_store_tokens)
        for token, count in metadata.failed_store_tokens.items():
            terminal_counts[token] = terminal_counts.get(token, 0) + count
        failed_tokens = set(metadata.failed_store_tokens)
        event_tokens: set[tuple[str, int]] = set()
        terminal_events = dict(metadata.completed_store_events)
        for event, count in metadata.failed_store_events.items():
            terminal_events[event] = terminal_events.get(event, 0) + count
        for event, count in terminal_events.items():
            total = self._store_event_pending_counts.get(event, 0) + count
            if total < self._num_workers:
                self._store_event_pending_counts[event] = total
                continue
            self._store_event_pending_counts.pop(event, None)
            tokens = self._store_event_tokens.pop(event, set())
            event_tokens.update(tokens)
            for token in tokens:
                block_ids = self._pinned_store_blocks.pop(token, None)
                if pool is not None and block_ids is not None:
                    if (
                        self._lazy_cursor is not None
                        and self._lazy_cursor.block_id in block_ids
                    ):
                        self._lazy_cursor = None
                    pool.free_blocks(
                        pool.blocks[block_id] for block_id in reversed(block_ids)
                    )
        for token, count in terminal_counts.items():
            if token in event_tokens:
                continue
            if count < self._num_workers:
                continue
            block_ids = self._pinned_store_blocks.pop(token, None)
            request_info = self._store_plan_requests.pop(token, None)
            if pool is not None and block_ids is not None:
                pool.free_blocks(
                    pool.blocks[block_id] for block_id in reversed(block_ids)
                )
            if request_info is not None:
                request_id, start_tokens = request_info
                tracker = self._request_trackers.get(request_id)
                if tracker is not None:
                    if token in failed_tokens:
                        tracker.record_store_failure(start_tokens)
                    else:
                        tracker.clear_store_retry()

    def has_pending_push_work(self) -> bool:
        return self._lazy_scan_pending or bool(self._pinned_store_blocks)

    def reset_store(self) -> bool:
        for future in self._lookup_futures.values():
            future.cancel()
        self._lookup_futures.clear()
        self._lookup_states.clear()
        self._load_specs.clear()
        self._pending_loads.clear()
        return self.runtime.clear()

    def close(self) -> None:
        self._lookup_executor.shutdown(wait=True, cancel_futures=True)
        self.runtime.close()
