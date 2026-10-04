# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scheduler-side UMBP connector behavior."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    KVCacheBlock,
    get_block_hash,
    make_block_hash_with_group_id,
    resolve_dcp_kv_block_size,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    SparseCacheRole,
    iter_layer_specs,
)
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import (
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockTransferPlan,
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
    ) -> None:
        self.block_size, resolved_hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.kv_cache_config = kv_cache_config
        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self.group_block_sizes = {
            group_id: resolve_dcp_kv_block_size(group.kv_cache_spec, dcp_size)
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
        }
        # Mamba "align" block tables are not append-only: a position may hold
        # the live state the next step writes, or a speculative block core will
        # relocate, so positional stores would read or pin the wrong block.
        # Core hands off each committed boundary state instead
        # (_add_boundary_state_plans).
        self._boundary_handoff_groups = frozenset(
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if isinstance(group.kv_cache_spec, MambaSpec)
            and getattr(group.kv_cache_spec, "mamba_cache_mode", None) == "align"
        )
        self.runtime = runtime
        self.enable_kv_cache_events = bool(
            vllm_config.kv_events_config
            and vllm_config.kv_events_config.enable_kv_cache_events
        )
        self.codec = codec
        self.topology = topology or RankTopology()
        transfer_config = vllm_config.kv_transfer_config
        assert transfer_config is not None
        extra = transfer_config.kv_connector_extra_config
        self.load_async = bool(extra.get("load_async", True))
        if not self.load_async and (
            len(kv_cache_config.kv_cache_groups) > 1
            or any(
                isinstance(spec, MambaSpec)
                or getattr(spec, "cache_role", None) == SparseCacheRole.INDEXER
                for group in kv_cache_config.kv_cache_groups
                for spec in iter_layer_specs(group.kv_cache_spec)
            )
        ):
            # Core accepts only request-level load failures for several cache
            # groups, and acts on them only while the request waits for an
            # asynchronous load. Mamba-style and sparse-attention indexer
            # layers also read restored state before any wait_for_layer_load.
            logger.warning(
                "UMBP loads asynchronously for models with several KV cache "
                "groups, Mamba-style or sparse-attention indexer layers; "
                "load_async=false is ignored"
            )
            self.load_async = True
        self.enable_lookup = bool(extra.get("enable_lookup", True))
        self.lookup_async = bool(extra.get("lookup_async", False))
        self.save_decode_cache = bool(extra.get("save_decode_cache", False))
        # A kv_consumer only restores from the pool, as with other store
        # connectors; it never offloads its own KV.
        self.store_enabled = getattr(transfer_config, "kv_role", None) != "kv_consumer"
        self.enable_partial_hash_hits = bool(
            extra.get("enable_partial_hash_hits", False)
        )
        # Request.block_hashes are computed at this size; keys index into them.
        self.hash_block_size = resolved_hash_block_size
        lookup_groups = [
            kv_cache_config.kv_cache_groups[group_id]
            for group_id in kv_cache_config.prefix_cacheable_group_ids
        ]
        model_config = getattr(vllm_config, "model_config", None)
        speculative_config = getattr(vllm_config, "speculative_config", None)
        use_eagle = bool(
            speculative_config is not None and speculative_config.use_eagle_block_drop()
        )
        self.use_eagle = use_eagle
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
                max_in_flight_tokens=vllm_config.max_in_flight_tokens,
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
                # As core's scheduler builds its coordinator.
                num_prefill_lookahead=vllm_config.num_prefill_lookahead_tokens,
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
        self._requests: dict[str, Request] = {}
        self._request_trackers: dict[str, RequestTracker] = {}
        self._pending_finished_stores: dict[str, list[BlockTransferPlan]] = {}
        self._gpu_block_pool: BlockPool | None = None
        self._store_events: dict[int, _StoreEvent] = {}
        self._num_workers = getattr(vllm_config.parallel_config, "world_size", 1)
        self._store_event_counter = 0

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
            if self.use_eagle:
                # As core's coordinator: the drafter recomputes the last block.
                matched_units = max(0, matched_units - units_per_block)
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
        # Core passes the request's whole table on every admission. A request
        # whose load failed is recomputed into newly allocated blocks.
        tracker.replace_blocks(tuple(list(group) for group in block_groups))
        if num_external_tokens <= 0:
            self._pending_loads.pop(request.request_id, None)
            self._load_specs.pop(request.request_id, None)
            return
        spec = self._load_specs.get(request.request_id)
        if spec is not None:
            tracker.load_spec = spec
            # A request admitted without its load would wait for it forever.
            assert num_external_tokens == spec.num_tokens_to_load, (
                f"UMBP matched {spec.num_tokens_to_load} external tokens for "
                f"request {request.request_id} but {num_external_tokens} were "
                "allocated"
            )
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
                    if not self.store_enabled or tracked_request is None
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

        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        resumed_ids: set[str] = set(getattr(cached_reqs, "resumed_req_ids", ()))
        for request_id in cached_reqs.req_ids:
            load_plans = self._pending_loads.pop(request_id, [])
            self._load_specs.pop(request_id, None)
            if load_plans:
                meta.load_requests[request_id] = load_plans
            elif self.store_enabled:
                cached_request = self._requests.get(request_id)
                cached_tracker = self._request_trackers.get(request_id)
                if cached_request is not None and cached_tracker is not None:
                    request_index = cached_reqs.req_ids.index(request_id)
                    num_computed_tokens = cached_reqs.num_computed_tokens[request_index]
                    # num_tokens counts sampled tokens, so a decode step is still
                    # short of it; only the prompt bounds the prefill.
                    is_prefill = num_computed_tokens < cached_request.num_prompt_tokens
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
                    # The appended table keeps blocks core has since freed, such
                    # as sliding-window blocks out of the window, which a store
                    # would read; core's current table does not.
                    current = (
                        block_state.get_block_ids(request_id)
                        if block_state is not None
                        else None
                    )
                    if current is not None:
                        cached_tracker.block_ids = tuple(
                            list(current[group_id])
                            for group_id in (
                                self.kv_cache_config.prefix_cacheable_group_ids
                            )
                        )
                    total_tokens = (
                        num_computed_tokens
                        + scheduler_output.num_scheduled_tokens[request_id]
                    )
                    if not self.save_decode_cache:
                        total_tokens = min(
                            total_tokens, cached_request.num_prompt_tokens
                        )
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
            self._load_specs.pop(request_id, None)
            if preempted_tracker := self._request_trackers.get(request_id):
                preempted_tracker.reset()

        if (
            self.store_enabled
            and block_state is not None
            and block_state.boundary_state_offloads
        ):
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
        return meta

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
                logger.debug(
                    "UMBP boundary offloads dropped request=%s entries=%d: %s",
                    request_id,
                    len(entries),
                    "finished" if request_id in finished else "preempted",
                )
                continue
            request = self._requests.get(request_id)
            tracker = self._request_trackers.get(request_id)
            if request is None or tracker is None:
                logger.debug(
                    "UMBP boundary offloads dropped request=%s entries=%d: untracked",
                    request_id,
                    len(entries),
                )
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
            logger.debug(
                "UMBP boundary offloads request=%s offered=%d stored=%d boundaries=%s",
                request_id,
                len(entries),
                len(plans),
                sorted({entry[2] for entry in entries}),
            )
            if plans:
                meta.store_plans.extend(plans)
                meta.store_requests.setdefault(request_id, []).extend(plans)

    def _tracker_for_request(self, request_id: str) -> RequestTracker:
        return self._request_trackers.setdefault(request_id, RequestTracker())

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
            return BlockLoadBatch(keys, destinations, groups, request.request_id)
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
        return BlockLoadBatch(keys, destinations, groups, request.request_id)

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
        # Scheduled speculative tokens are not yet verified and have no block
        # hash; a block is storable only once its hash exists.
        token_count = min(token_count, len(request.block_hashes) * self.hash_block_size)
        previous_saved_tokens = tracker.saved_tokens
        # Hash-aligned rather than block-aligned: a group with smaller blocks
        # than the largest must not wait for that group's block to fill.
        save_to = tracker.mark_saved(token_count, self.hash_block_size)
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
            if group_id not in self._boundary_handoff_groups
        }
        resident_blocks = (
            self._restored_store_blocks(
                request, block_ranges, tracker.load_spec.external_tokens, block_groups
            )
            if tracker.load_spec is not None
            else {}
        )
        for group_id, block_ids in zip(group_ids, block_groups):
            if group_id not in block_ranges:
                continue
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
        # block_ranges covers only positionally stored groups; block_groups
        # follows prefix_cacheable_group_ids.
        blocks_of = dict(
            zip(
                self.kv_cache_config.prefix_cacheable_group_ids,
                block_groups,
                strict=True,
            )
        )
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
                if blocks_of[group_id][index] != NULL_BLOCK_ID
            ]
            for group_id, indices in block_ranges.items()
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
        return False, None

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Pin finished boundary pages without delaying core request cleanup."""
        pool = self._gpu_block_pool
        if not self.store_enabled or not partial_tail_offloads or pool is None:
            return False
        if request.request_id in self._pending_finished_stores:
            return False
        prefix_groups = set(self.kv_cache_config.prefix_cacheable_group_ids)
        tracker = self._request_trackers.get(request.request_id)
        if tracker is None:
            return False
        boundaries = {entry[2] for entry in partial_tail_offloads}
        if len(boundaries) != 1:
            logger.warning("UMBP partial tail entries span several boundaries")
            return False
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
                    block_id=block_id,
                    group_id=group_id,
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

    def update_connector_output(self, output: KVConnectorOutput) -> None:
        metadata = output.kv_connector_worker_meta
        if not isinstance(metadata, UMBPConnectorWorkerMetadata):
            return
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

    def has_pending_push_work(self) -> bool:
        return bool(self._pending_finished_stores) or bool(self._store_events)

    def reset_store(self) -> bool:
        if self._store_events or self._pending_finished_stores:
            return False
        for pending in self._pending_lookups.values():
            pending.future.cancel()
        self._pending_lookups.clear()
        self._load_specs.clear()
        self._pending_loads.clear()
        return self.runtime.clear()

    def close(self) -> None:
        self._lookup_executor.shutdown(wait=True, cancel_futures=True)
        self._pending_lookups.clear()
        self.runtime.close()
