# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scheduler-side UMBP connector behavior."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import (
    BlockIdentityCodec,
    BlockTransferPlan,
    KVLayoutPlanner,
    LoadSpec,
    PartialTailPlan,
    RankCompletenessPolicy,
    RankTopology,
    RequestTracker,
    LookupState,
    TPShardMapping,
    UMBPConnectorMetadata,
    UMBPConnectorWorkerMetadata,
)
from .runtime import UMBPSchedulerHandle


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
        self.block_size = vllm_config.cache_config.block_size
        self.kv_cache_config = kv_cache_config
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
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        self.load_async = bool(extra.get("load_async", True))
        self.enable_lookup = bool(extra.get("enable_lookup", True))
        self.lookup_async = bool(extra.get("lookup_async", False))
        self.save_decode_cache = bool(extra.get("save_decode_cache", False))
        self.lazy_offload = bool(extra.get("lazy_offload", False))
        if extra.get("enable_partial_hash_hits", False):
            raise ValueError(
                "UMBP partial hash hits require a runtime tail-key protocol"
            )
        self.enable_partial_hash_hits = False
        self.hash_block_size = int(extra.get("hash_block_size", self.block_size))
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
        self._store_plan_requests: dict[
            tuple[str, int], tuple[str, int]
        ] = {}
        self._num_workers = getattr(vllm_config.parallel_config, "world_size", 1)

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
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.block_size
        )
        if not hashes or request.num_tokens < align:
            return 0, False
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        scale = self.block_size // self.hash_block_size
        lookup_hashes = (
            hashes
            if self.enable_partial_hash_hits
            else [
                hashes[index * scale + scale - 1]
                for index in range(len(hashes) // scale)
            ]
        )
        keys = [
            key
            for block_hash in lookup_hashes
            for key in self.codec.keys_for_topology(
                block_hash, self.topology, group_ids
            )
        ]
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
        per_block = self.completeness.required_rank_count * len(group_ids)
        if per_block == 0 or len(hits) != len(keys):
            lookup_state.fail("lookup returned an invalid result length")
            return 0, False
        matched_units = 0
        for offset in range(0, len(hits), per_block):
            if not all(hits[offset : offset + per_block]):
                break
            matched_units += 1
        unit_size = (
            self.hash_block_size
            if self.enable_partial_hash_hits
            else self.block_size
        )
        matched_tokens = matched_units * unit_size
        matched_tokens = min(matched_tokens, request.num_tokens)
        lookup_state.complete(matched_tokens)
        need_to_load = max(matched_tokens - num_computed_tokens, 0)
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
        num_blocks = num_external_tokens // self.block_size
        spec = self._load_specs.get(request.request_id)
        if spec is not None:
            spec.can_load = (
                num_external_tokens == spec.num_tokens_to_load
            )
            tracker.load_spec = spec
            if not spec.can_load:
                return
        start_hash = request.num_tokens - num_external_tokens
        plans = [
            BlockTransferPlan(
                key=self.codec.key(
                    self._object_hash(
                        request.block_hashes,
                        start_hash // self.block_size + index,
                    ),
                    group_id=group_id,
                ),
                block_id=block_groups[group_index][-num_blocks + index],
                request_id=request.request_id,
                generation=tracker.generation,
            )
            for index in range(num_blocks)
            for group_index, group_id in enumerate(
                self.kv_cache_config.prefix_cacheable_group_ids
            )
        ]
        self._pending_loads[request.request_id] = plans

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        meta = UMBPConnectorMetadata()
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
                store_plans = (
                    []
                    if self.lazy_offload
                    else self._store_plans(request, tracker, total_tokens)
                )
                meta.store_plans.extend(store_plans)
                if store_plans:
                    meta.store_requests[request.req_id] = store_plans
            if request.req_id in self._lookup_states:
                meta.lookup_states[request.req_id] = self._lookup_states[
                    request.req_id
                ]

        cached_reqs = scheduler_output.scheduled_cached_reqs
        for request_id in cached_reqs.req_ids:
            load_plans = self._pending_loads.pop(request_id, [])
            self._load_specs.pop(request_id, None)
            if load_plans:
                meta.load_plans.extend(load_plans)
                meta.load_requests[request_id] = load_plans
            elif self.save_decode_cache and not self.lazy_offload:
                request = self._requests.get(request_id)
                tracker = self._request_trackers.get(request_id)
                if request is not None and tracker is not None:
                    request_index = cached_reqs.req_ids.index(request_id)
                    new_block_ids = cached_reqs.new_block_ids[request_index]
                    if new_block_ids:
                        tracker.update_blocks(
                            tuple(
                                new_block_ids[group_id]
                                for group_id in self.kv_cache_config.prefix_cacheable_group_ids
                            )
                        )
                    total_tokens = (
                        cached_reqs.num_computed_tokens[request_index]
                        + scheduler_output.num_scheduled_tokens[request_id]
                    )
                    tracker.token_len = total_tokens
                    store_plans = self._store_plans(
                        request,
                        tracker,
                        total_tokens,
                        block_ids_override=tracker.block_ids,
                    )
                    meta.store_plans.extend(store_plans)
                    if store_plans:
                        meta.store_requests[request_id] = store_plans

        for request_id in scheduler_output.preempted_req_ids or set():
            meta.preempted_request_ids.add(request_id)
            plans = self._pending_loads.pop(request_id, [])
            meta.preempted_block_ids.update(plan.block_id for plan in plans)
            self._load_specs.pop(request_id, None)
            if tracker := self._request_trackers.get(request_id):
                tracker.reset()

        for plans in self._pending_partial_tails.values():
            meta.partial_tail_plans.extend(plans)
        self._pending_partial_tails.clear()
        meta.store_plans.extend(self._pending_stores)
        self._pending_stores.clear()
        self._reference_store_blocks(meta)
        self._group_layer_plans(meta)
        return meta

    @staticmethod
    def _group_layer_plans(meta: UMBPConnectorMetadata) -> None:
        """Expose layer ownership without changing bulk plan semantics."""
        for plan in meta.load_plans:
            for item in plan.ranges:
                meta.load_plans_by_layer.setdefault(item.layer_name, []).append(
                    plan
                )
        for plan in meta.store_plans:
            for item in plan.ranges:
                meta.store_plans_by_layer.setdefault(item.layer_name, []).append(
                    plan
                )

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
                [
                    self._gpu_block_pool.blocks[block_id]
                    for block_id in new_block_ids
                ]
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

    def _object_hash(self, hashes: list[bytes], block_index: int) -> bytes:
        scale = self.block_size // self.hash_block_size
        hash_index = block_index * scale + scale - 1
        if hash_index >= len(hashes):
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
        start_block = previous_saved_tokens // self.block_size
        num_blocks = save_to // self.block_size
        plans: list[BlockTransferPlan] = []
        block_groups = (
            block_ids_override
            if block_ids_override is not None
            else tuple(request.block_ids[group_id] for group_id in group_ids)
        )
        for group_id in group_ids:
            block_index = group_ids.index(group_id)
            block_ids = block_groups[block_index]
            for index, block_id in enumerate(
                block_ids[start_block:num_blocks], start=start_block
            ):
                if index >= len(request.block_hashes):
                    break
                plans.append(
                    BlockTransferPlan(
                        key=self.codec.key(
                            self._object_hash(request.block_hashes, index),
                            group_id=group_id,
                        ),
                        block_id=block_id,
                        request_id=request_id,
                        generation=tracker.generation,
                        group_id=group_id,
                        block_hash=self._object_hash(
                            request.block_hashes, index
                        ),
                        parent_block_hash=(
                            self._object_hash(request.block_hashes, index - 1)
                            if index > 0
                            else None
                        ),
                        token_ids=tuple(
                            getattr(request, "prompt_token_ids", [])
                        )[
                            index * self.block_size : (index + 1)
                            * self.block_size
                        ],
                        block_size=self.block_size,
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
            tracker = self._tracker_for_request(request.request_id)
            tracker.save_mode = "lazy"
            group_ids = self.kv_cache_config.prefix_cacheable_group_ids
            selected_block_ids = tuple(block_ids[group_id] for group_id in group_ids)
            token_count = getattr(
                request,
                "num_computed_tokens",
                getattr(request, "num_tokens", 0),
            )
            plans = self._store_plans(
                request,
                tracker,
                token_count,
                block_ids_override=selected_block_ids,
            )
            self._pending_stores.extend(plans)
            if plans:
                pending_meta = UMBPConnectorMetadata(store_plans=plans)
                self._reference_store_blocks(pending_meta)
        return False, None

    def register_finished_partial_tail(
        self,
        request: Request,
        block_ids: tuple[list[int], ...],
        partial_tail_offloads: list[tuple[int, int, int]],
    ) -> bool:
        """Queue a single full-attention partial tail for the next step."""
        del block_ids
        if not partial_tail_offloads:
            return False
        if len(self.kv_cache_config.prefix_cacheable_group_ids) != 1:
            return False
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
        group_id = self.kv_cache_config.prefix_cacheable_group_ids[0]
        start_token = (boundary - 1) // self.block_size * self.block_size
        plans: list[PartialTailPlan] = []
        for entry_group_id, block_id, _ in partial_tail_offloads:
            if entry_group_id != group_id:
                return False
            plans.append(
                PartialTailPlan(
                    request_id=request.request_id,
                    generation=tracker.generation,
                    block_id=block_id,
                    group_id=group_id,
                    start_token=start_token,
                    end_token=boundary,
                    key=self.codec.key(request.block_hashes[hash_index], group_id),
                    block_size=self.block_size,
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
        for token, count in terminal_counts.items():
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
        return bool(self._pinned_store_blocks)

    def close(self) -> None:
        self._lookup_executor.shutdown(wait=True, cancel_futures=True)
        self.runtime.close()
