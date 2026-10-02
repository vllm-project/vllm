# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scheduler-side UMBP connector behavior."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    resolve_dcp_kv_block_size,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
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
        self.runtime = runtime
        self.codec = codec
        self.topology = topology or RankTopology()
        transfer_config = vllm_config.kv_transfer_config
        assert transfer_config is not None
        extra = transfer_config.kv_connector_extra_config
        self.load_async = bool(extra.get("load_async", True))
        self.enable_lookup = bool(extra.get("enable_lookup", True))
        self.lookup_async = bool(extra.get("lookup_async", False))
        self.save_decode_cache = bool(extra.get("save_decode_cache", False))
        # A kv_consumer only restores from the pool, as with other store
        # connectors; it never offloads its own KV.
        self.store_enabled = getattr(transfer_config, "kv_role", None) != "kv_consumer"
        # Request.block_hashes are computed at this size; keys index into them.
        self.hash_block_size = resolved_hash_block_size
        speculative_config = getattr(vllm_config, "speculative_config", None)
        self.use_eagle = bool(
            speculative_config is not None and speculative_config.use_eagle_block_drop()
        )
        self._pending_loads: dict[str, list[BlockTransferPlan] | BlockLoadBatch] = {}
        self._load_specs: dict[str, LoadSpec] = {}
        self._lookup_units = {
            group_id: self.group_block_sizes[group_id]
            for group_id in kv_cache_config.prefix_cacheable_group_ids
        }
        self._pending_lookups: dict[str, _PendingLookup] = {}
        self._lookup_executor = ThreadPoolExecutor(
            max_workers=int(extra.get("lookup_workers", 2)),
            thread_name_prefix="umbp-lookup",
        )
        self._requests: dict[str, Request] = {}
        self._request_trackers: dict[str, RequestTracker] = {}
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
        if not hashes or request.num_tokens < self.block_size:
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
        group_id = group_ids[0]
        matched_blocks = 0
        for hit in logical_hits_by_group[group_id]:
            if not hit:
                break
            matched_blocks += 1
        if self.use_eagle:
            # As core's coordinator: the drafter recomputes the last block.
            matched_blocks = max(0, matched_blocks - 1)
        need_to_load = matched_blocks * self._lookup_units[group_id]
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
        return meta

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
        return bool(self._store_events)

    def reset_store(self) -> bool:
        if self._store_events:
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
