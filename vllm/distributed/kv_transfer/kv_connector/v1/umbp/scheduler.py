# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared scheduler-side UMBP connector behavior."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from functools import lru_cache
from itertools import islice
from typing import Any, cast

from vllm.config import VllmConfig
from vllm.distributed.kv_events import BlockRemoved, BlockStored, KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_coordinator import (
    get_kv_cache_coordinator,
    replay_boundaries,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    get_block_hash,
    get_group_id,
    make_block_hash_with_group_id,
    maybe_convert_block_hash,
    resolve_dcp_kv_block_size,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
    SparseCacheRole,
    UniformTypeKVCacheSpecs,
    iter_layer_specs,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request

from .data import (
    BlockIdentityCodec,
    BlockLoadBatch,
    BlockStoreBatch,
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


@lru_cache(maxsize=8192)
def _decode_lazy_block_hash(
    codec: BlockIdentityCodec, block_hash_with_group: BlockHashWithGroupId
) -> tuple[int, bytes, str]:
    """Decode an immutable block identity for lazy candidate selection."""
    group_id = get_group_id(block_hash_with_group)
    block_hash = get_block_hash(block_hash_with_group)
    return group_id, block_hash, codec.key(block_hash, group_id)


def _unwrap_store_spec(spec: KVCacheSpec) -> KVCacheSpec:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return next(iter(spec.kv_cache_specs.values()))
    return spec


@dataclass
class _StoreEvent:
    plan_groups: tuple[Sequence[BlockTransferPlan], ...]
    result: StoreEventResult = field(default_factory=StoreEventResult)

    def iter_plans(self):
        for plans in self.plan_groups:
            yield from plans

    def iter_keys(self):
        for plans in self.plan_groups:
            if isinstance(plans, BlockStoreBatch):
                yield from plans.keys
            else:
                yield from (plan.key for plan in plans)

    def block_ids(self):
        for plans in self.plan_groups:
            if isinstance(plans, BlockStoreBatch):
                yield from plans.block_ids
            else:
                yield from (plan.block_id for plan in plans)


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


@dataclass
class _EventObject:
    serial: int
    plan: BlockTransferPlan | None
    published: bool = False


@dataclass(frozen=True)
class _EventLookup:
    objects: tuple[tuple[tuple[int, bytes], int], ...]
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
    """Own vLLM prefix matching while the runtime owns lookup transport.

    Without a runtime and codec the scheduler neither looks up nor stores
    until ``bind_namespace`` supplies them.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        runtime: UMBPSchedulerHandle | None,
        codec: BlockIdentityCodec | None,
        topology: RankTopology | None = None,
    ) -> None:
        self.block_size, resolved_hash_block_size = resolve_kv_cache_block_sizes(
            kv_cache_config, vllm_config
        )
        self.kv_cache_config = kv_cache_config
        dcp_size = vllm_config.parallel_config.decode_context_parallel_size
        self._dcp_size = dcp_size
        self._retention_interval = kv_cache_config.prefix_cache_retention_interval
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
        self.lazy_offload = self.store_enabled and bool(
            extra.get("lazy_offload", False)
        )
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
        explicit_eagle_groups = {
            group_id
            for group_id, group in enumerate(kv_cache_config.kv_cache_groups)
            if group.is_eagle_group
        }
        self._eagle_store_groups = (
            explicit_eagle_groups
            if explicit_eagle_groups or not use_eagle
            else set(kv_cache_config.prefix_cacheable_group_ids)
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
            if lookup_groups
            and (
                len(lookup_groups) > 1
                or not isinstance(
                    _unwrap_store_spec(lookup_groups[0].kv_cache_spec),
                    FullAttentionSpec,
                )
            )
            else None
        )
        self._lookup_pool = _LookupBlockPool(self.hash_block_size)
        self._local_cache_lookup: Callable[[int, bytes], bool] | None = None
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
        self._kv_events: list[KVCacheEvent] = []
        self._event_objects: dict[tuple[int, bytes], _EventObject] = {}
        self._pending_event_objects: dict[tuple[int, bytes], int] = {}
        self._event_lookup: _EventLookup | None = None
        self._event_serial = 0
        self._next_event_reconcile = 0.0
        self._event_max_objects = extra.get("kv_event_max_objects", 65536)
        if type(self._event_max_objects) is not int or self._event_max_objects <= 0:
            raise ValueError("kv_event_max_objects must be a positive integer")
        self._event_check_batch_size = 256
        self._event_publications_suppressed = 0
        self._num_workers = getattr(vllm_config.parallel_config, "world_size", 1)
        self._lazy_scan_pending = False
        self._lazy_prefix_chains_by_block: dict[
            int,
            tuple[tuple[tuple[int, str, int, bytes], ...], int],
        ] = {}
        configured_lazy_target = extra.get("lazy_offload_max_blocks")
        if configured_lazy_target is not None and (
            type(configured_lazy_target) is not int or configured_lazy_target < 0
        ):
            raise ValueError("lazy_offload_max_blocks must be a non-negative integer")
        self._lazy_target_blocks = (
            configured_lazy_target
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
            block_size = resolve_dcp_kv_block_size(spec, dcp_size)
            if isinstance(spec, MambaSpec):
                target += 2
            elif isinstance(spec, SlidingWindowSpec):
                target += cdiv(spec.sliding_window, block_size) + 1
            else:
                target += cdiv(max_num_batched_tokens, block_size)
        return 2 * target

    def bind_gpu_block_pool(self, gpu_block_pool: BlockPool) -> None:
        self._gpu_block_pool = gpu_block_pool

    def bind_local_cache_lookup(self, lookup: Callable[[int, bytes], bool]) -> None:
        self._local_cache_lookup = lookup

    def _has_local_block(self, group_id: int, block_hash: bytes) -> bool:
        if self._local_cache_lookup is not None:
            return self._local_cache_lookup(group_id, block_hash)
        pool = self._gpu_block_pool
        return pool is not None and bool(
            pool.get_cached_block(cast(BlockHash, block_hash), [group_id])
        )

    def bind_namespace(
        self, runtime: UMBPSchedulerHandle, codec: BlockIdentityCodec
    ) -> None:
        self.runtime = runtime
        self.codec = codec

    def get_num_new_matched_tokens(
        self, request: Request, num_computed_tokens: int
    ) -> tuple[int | None, bool]:
        if not self.enable_lookup:
            return 0, False
        if self.runtime is None:
            logger.warning_once(
                "UMBP lookup is disabled until workers report their namespace"
            )
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
        if not group_ids:
            return 0, False
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
        load_async = self.load_async
        if block_hashes_by_group and not any(
            block_hash is not None
            for group_id, group_hashes in zip(
                group_ids, block_hashes_by_group, strict=True
            )
            for index, block_hash in enumerate(group_hashes)
            if index >= num_computed_tokens // self.group_block_sizes[group_id]
        ):
            load_async = False
        return need_to_load, load_async

    def _build_lookup_context(
        self, request: Request, local_tokens: int, hashes: tuple[bytes, ...]
    ) -> _LookupContext:
        assert self.codec is not None
        objects: dict[int, list[tuple[bytes, tuple[str, ...]]]] = {}
        keys: list[str] = []
        lookup_limit = request.num_tokens - 1
        if self.use_eagle and self._lookup_coordinator is not None:
            lookup_limit = min(request.num_tokens, len(hashes) * self.hash_block_size)
        for group_id, unit in self._lookup_units.items():
            group_objects = []
            for token_end in range(local_tokens + unit, lookup_limit + 1, unit):
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
            if local_tokens:
                for group_index, group_id in enumerate(
                    self.kv_cache_config.prefix_cacheable_group_ids
                ):
                    for token_end in range(
                        self._lookup_units[group_id],
                        local_tokens + 1,
                        self._lookup_units[group_id],
                    ):
                        block_hash = self._object_hash_at_token_end(
                            request_hashes, token_end
                        )
                        if self._has_local_block(group_id, block_hash):
                            pool.hits[group_index].add(block_hash)
            hit_blocks, hit_length, _ = coordinator.find_longest_cache_hit(
                cast(list[BlockHash], request_hashes),
                local_tokens + max_external_tokens,
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
            return max(0, hit_length - local_tokens), block_hashes_by_group
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
        reserved_store_keys = {
            key for event in self._store_events.values() for key in event.iter_keys()
        }
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
                    None
                    if (
                        not self.store_enabled
                        or self.lazy_offload
                        or tracked_request is None
                    )
                    else self._store_plans(
                        tracked_request,
                        tracker,
                        total_tokens,
                        block_ids_override=selected_block_ids,
                        skip_keys=reserved_store_keys,
                    )
                )
                logger.debug(
                    "UMBP store planning request=%s tokens=%d plans=%d keys=%s",
                    request.req_id,
                    total_tokens,
                    len(store_plans) if store_plans is not None else 0,
                    tuple(store_plans.keys) if store_plans is not None else (),
                )
                if store_plans:
                    meta.store_requests[request.req_id] = store_plans
                    reserved_store_keys.update(store_plans.keys)

        block_state = getattr(scheduler_output, "kv_connector_block_state", None)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        resumed_ids: set[str] = set(getattr(cached_reqs, "resumed_req_ids", ()))
        for request_id in cached_reqs.req_ids:
            load_plans = self._pending_loads.pop(request_id, [])
            self._load_specs.pop(request_id, None)
            if load_plans:
                meta.load_requests[request_id] = load_plans
            elif self.store_enabled and not self.lazy_offload:
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
                        skip_keys=reserved_store_keys,
                    )
                    if store_plans:
                        meta.store_requests[request_id] = store_plans
                        reserved_store_keys.update(store_plans.keys)

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
        if self.lazy_offload:
            lazy_plans, lookup_succeeded = self._prepare_lazy_store_plans()
            meta.store_plans.extend(lazy_plans)
            if lookup_succeeded:
                self._lazy_scan_pending = False
        event_plan_groups = tuple(meta.store_plan_groups())
        if event_plan_groups:
            event = self._store_event_counter
            self._store_event_counter += 1
            meta.store_event = event
            self._store_events[event] = _StoreEvent(event_plan_groups)
            if self._gpu_block_pool is not None:
                self._gpu_block_pool.touch(
                    [
                        self._gpu_block_pool.blocks[block_id]
                        for block_id in dict.fromkeys(
                            block_id
                            for plans in event_plan_groups
                            for block_id in (
                                plans.block_ids
                                if isinstance(plans, BlockStoreBatch)
                                else (plan.block_id for plan in plans)
                            )
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

    def _prepare_lazy_store_plans(self) -> tuple[list[BlockTransferPlan], bool]:
        pool = self._gpu_block_pool
        if (
            pool is None
            or self.runtime is None
            or self.codec is None
            or self._lazy_target_blocks <= 0
        ):
            return [], True
        candidates: list[tuple[KVCacheBlock, str, int, bytes]] = []
        seen_keys: set[str] = set()
        expanded_chains: dict[int, int] = {}
        visited_by_group: dict[int, int] = {}
        unhashed = 0
        in_flight_keys = {
            key for event in self._store_events.values() for key in event.iter_keys()
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
            _, _, tail_group_id, tail_hash = chain[chain_index]
            expected_tail_hash = make_block_hash_with_group_id(
                BlockHash(tail_hash), tail_group_id
            )
            if (
                expected_tail_hash != block.block_hash
                and expected_tail_hash not in aliases.get(block.block_id, ())
            ):
                # Block IDs are recycled. Do not expand a prefix chain whose
                # tail identity no longer matches the free-queue block.
                del self._lazy_prefix_chains_by_block[block.block_id]
                continue
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
            return [], True

        try:
            keys = [
                rank_key
                for _, _, group_id, block_hash in candidates
                for rank_key in _rank_keys(
                    self.codec, self.topology, block_hash, group_id
                )
            ]
            hits = list(self.runtime.lookup(keys))
        except Exception as exc:
            logger.warning("UMBP lazy residency lookup failed: %s", exc)
            return [], False
        if len(hits) != len(keys):
            logger.warning(
                "UMBP lazy lookup returned %d results for %d candidates",
                len(hits),
                len(keys),
            )
            return [], False

        plans: list[BlockTransferPlan] = []
        rank_count = self.topology.rank_count
        for index, (block, key, group_id, block_hash) in enumerate(candidates):
            if all(hits[index * rank_count : (index + 1) * rank_count]):
                continue
            plans.append(
                BlockTransferPlan(
                    key=key,
                    block_id=block.block_id,
                    group_id=group_id,
                    block_hash=block_hash,
                    block_size=self.group_block_sizes[group_id],
                )
            )
        return plans, True

    def _add_boundary_state_plans(
        self,
        offloads: dict[str, list[tuple[int, int, int]]],
        meta: UMBPConnectorMetadata,
        finished: set[str],
        preempted: set[str],
    ) -> None:
        """Store exact Mamba/hybrid boundary blocks handed off by core."""
        if self.codec is None:
            return
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
        assert self.codec is not None
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
                for block_index, block_hash in enumerate(
                    spec.block_hashes_by_group[group_index]
                ):
                    if block_index < start_block or block_hash is None:
                        continue
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
        skip_keys: set[str] | None = None,
    ) -> BlockStoreBatch:
        """Describe full blocks produced by a scheduled prefill."""
        request_id = getattr(request, "request_id", None) or request.req_id
        if self.codec is None:
            return BlockStoreBatch([], [], [], [], [], [], [], request_id)
        group_ids = self.kv_cache_config.prefix_cacheable_group_ids
        # Scheduled speculative tokens are not yet verified and have no block
        # hash; a block is storable only once its hash exists.
        token_count = min(token_count, len(request.block_hashes) * self.hash_block_size)
        previous_saved_tokens = tracker.saved_tokens
        # Hash-aligned rather than block-aligned: a group with smaller blocks
        # than the largest must not wait for that group's block to fill.
        save_to = tracker.mark_saved(token_count, self.hash_block_size)
        keys: list[str] = []
        source_block_ids: list[int] = []
        plan_group_ids: list[int] = []
        block_hashes: list[bytes] = []
        parent_block_hashes: list[bytes | None] = []
        token_ids_by_block: list[tuple[int, ...]] = []
        block_sizes: list[int] = []
        block_groups = (
            block_ids_override
            if block_ids_override is not None
            else tuple(request.block_ids[group_id] for group_id in group_ids)
        )
        candidate_ranges = {
            group_id: range(
                previous_saved_tokens // self.group_block_sizes[group_id],
                min(save_to // self.group_block_sizes[group_id], len(block_ids)),
            )
            for group_id, block_ids in zip(group_ids, block_groups)
            if group_id not in self._boundary_handoff_groups
        }
        block_ranges = {
            group_id: tuple(
                index
                for relative_index, index in enumerate(indices)
                if mask is None or mask[relative_index]
            )
            for group_id, indices in candidate_ranges.items()
            for mask in (self._store_mask(request, group_id, indices),)
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
                key = self.codec.key(block_hash, group_id=group_id)
                if skip_keys is not None and key in skip_keys:
                    continue
                keys.append(key)
                source_block_ids.append(block_ids[index])
                plan_group_ids.append(group_id)
                block_hashes.append(block_hash)
                parent_block_hashes.append(
                    self._object_hash_at_token_end(
                        request.block_hashes, token_end - group_block_size
                    )
                    if self.enable_kv_cache_events and index > 0
                    else None
                )
                token_ids_by_block.append(
                    tuple(request.all_token_ids[index * group_block_size : token_end])
                    if self.enable_kv_cache_events
                    else ()
                )
                block_sizes.append(group_block_size)
        return BlockStoreBatch(
            keys,
            source_block_ids,
            plan_group_ids,
            block_hashes,
            parent_block_hashes,
            token_ids_by_block,
            block_sizes,
            request_id,
        )

    def _restored_store_blocks(
        self,
        request: Any,
        block_ranges: Mapping[int, Sequence[int]],
        restored_tokens: int,
        block_groups: tuple[list[int], ...],
    ) -> dict[int, set[int]]:
        """Query residency before constructing plans for restored blocks."""
        assert self.runtime is not None and self.codec is not None
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
                for index in indices
                if index < restored_tokens // self.group_block_sizes[group_id]
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

    def _store_mask(
        self,
        request: Any,
        group_id: int,
        indices: range,
    ) -> list[bool] | None:
        """Keep only blocks that core's cache semantics can later reuse."""
        if not indices:
            return []
        group = self.kv_cache_config.kv_cache_groups[group_id]
        spec = _unwrap_store_spec(group.kv_cache_spec)
        manager_cls = KVCacheSpecRegistry.get_manager_class(spec)
        if manager_cls is None:
            return None
        num_prompt_tokens = getattr(request, "num_prompt_tokens", None)
        reachable_boundaries = (
            ()
            if num_prompt_tokens is None
            else replay_boundaries(
                num_prompt_tokens, self.block_size, use_eagle=self.use_eagle
            )
        )
        mask = manager_cls.reachable_block_mask(
            start_block=indices.start,
            end_block=indices.stop,
            alignment_tokens=self.block_size,
            kv_cache_spec=spec,
            use_eagle=group_id in self._eagle_store_groups,
            retention_interval=self._retention_interval,
            reachable_boundaries=reachable_boundaries,
            dcp_world_size=self._dcp_size,
        )
        if mask is not None and len(mask) != len(indices):
            raise RuntimeError(
                f"UMBP store mask for group {group_id} returned {len(mask)} "
                f"entries for {len(indices)} blocks"
            )
        return mask

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
        if self.codec is None:
            return
        prompt_tokens = request.num_prompt_tokens
        save_to = min(prompt_tokens, request.num_computed_tokens)
        hashes = list(request.block_hashes)
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
        if (
            not self.store_enabled
            or self.codec is None
            or not partial_tail_offloads
            or pool is None
        ):
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
                block_ids = dict.fromkeys(event.block_ids())
                pool.free_blocks(
                    pool.blocks[block_id] for block_id in reversed(block_ids)
                )
            if self.enable_kv_cache_events:
                self._queue_event_objects(event)
        self._reconcile_kv_events()

    def _queue_event_objects(self, event: _StoreEvent) -> None:
        seen: set[tuple[int, bytes]] = set()
        for plan in event.iter_plans():
            if plan.group_id is None or plan.block_hash is None:
                continue
            identity = (plan.group_id, plan.block_hash)
            if identity in seen or identity in event.result.failed_objects:
                continue
            seen.add(identity)
            previous = self._event_objects.get(identity)
            if previous is None and len(self._event_objects) >= self._event_max_objects:
                self._event_publications_suppressed += 1
                logger.warning_once(
                    "UMBP KV event tracking limit reached; suppressing new "
                    "BlockStored events until tracked objects are evicted"
                )
                continue
            self._event_serial += 1
            self._event_objects[identity] = _EventObject(
                self._event_serial, plan, previous.published if previous else False
            )
            self._pending_event_objects[identity] = self._event_serial

    def invalidate_kv_events(self, events: Sequence[KVCacheEvent]) -> None:
        removed = {
            (event.group_idx, block_hash)
            for event in events
            if isinstance(event, BlockRemoved)
            for block_hash in event.block_hashes
        }
        if not removed:
            return
        for identity in list(self._event_objects):
            group_id, block_hash = identity
            if (
                group_id,
                maybe_convert_block_hash(cast(BlockHash, block_hash)),
            ) in removed:
                del self._event_objects[identity]
                self._pending_event_objects.pop(identity, None)

    def _reconcile_kv_events(self) -> None:
        if (
            not self.enable_kv_cache_events
            or self.runtime is None
            or self.codec is None
        ):
            return
        pending = self._event_lookup
        if pending is not None:
            if not pending.future.done():
                return
            self._event_lookup = None
            try:
                hits = list(pending.future.result())
                if len(hits) != len(pending.objects) * self.topology.rank_count:
                    raise ValueError("invalid UMBP event residency result length")
            except Exception as exc:
                logger.debug("UMBP event residency check failed: %s", exc)
                hits = None
            for index, (identity, serial) in enumerate(pending.objects):
                record = self._event_objects.get(identity)
                if record is None or record.serial != serial:
                    continue
                # Rotate completed checks so a bounded batch visits every object.
                del self._event_objects[identity]
                self._pending_event_objects.pop(identity, None)
                if hits is None:
                    if record.published:
                        self._event_objects[identity] = record
                    continue
                ranks = self.topology.rank_count
                resident = all(hits[index * ranks : (index + 1) * ranks])
                if not resident:
                    if record.published:
                        group_id, block_hash = identity
                        self._kv_events.append(
                            BlockRemoved(
                                block_hashes=[
                                    maybe_convert_block_hash(
                                        cast(BlockHash, block_hash)
                                    )
                                ],
                                medium="CPU",
                                group_idx=group_id,
                                locality="LOCAL",
                            )
                        )
                    continue
                if not record.published:
                    plan = record.plan
                    assert plan is not None and plan.block_hash is not None
                    self._kv_events.append(
                        BlockStored(
                            block_hashes=[
                                maybe_convert_block_hash(
                                    cast(BlockHash, plan.block_hash)
                                )
                            ],
                            parent_block_hash=(
                                maybe_convert_block_hash(
                                    cast(BlockHash, plan.parent_block_hash)
                                )
                                if plan.parent_block_hash is not None
                                else None
                            ),
                            token_ids=list(plan.token_ids),
                            block_size=plan.block_size,
                            lora_id=None,
                            medium=plan.medium,
                            lora_name=None,
                            group_idx=plan.group_id,
                            locality="LOCAL",
                        )
                    )
                record.published = True
                record.plan = None
                self._event_objects[identity] = record
        if not self._event_objects:
            return
        now = time.monotonic()
        candidates = []
        if now >= self._next_event_reconcile:
            candidates = list(
                islice(
                    (
                        (identity, record.serial)
                        for identity, record in self._event_objects.items()
                        if record.published
                    ),
                    max(1, self._event_check_batch_size // 2),
                )
            )
            self._next_event_reconcile = now + 1.0
        selected = {identity for identity, _ in candidates}
        candidates.extend(
            islice(
                (
                    (identity, serial)
                    for identity, serial in self._pending_event_objects.items()
                    if identity not in selected
                ),
                self._event_check_batch_size - len(candidates),
            )
        )
        if not candidates:
            return
        keys = [
            key
            for (group_id, block_hash), _ in candidates
            for key in _rank_keys(self.codec, self.topology, block_hash, group_id)
        ]
        self._event_lookup = _EventLookup(
            tuple(candidates),
            self._lookup_executor.submit(self.runtime.lookup, keys),
        )

    def take_events(self) -> list[KVCacheEvent]:
        self._reconcile_kv_events()
        events = self._kv_events
        self._kv_events = []
        return events

    def has_pending_push_work(self) -> bool:
        return (
            self._lazy_scan_pending
            or bool(self._pending_finished_stores)
            or bool(self._store_events)
            or self._event_lookup is not None
            or bool(self._pending_event_objects)
        )

    def reset_store(self) -> bool:
        if self._store_events or self._pending_finished_stores:
            return False
        for pending in self._pending_lookups.values():
            pending.future.cancel()
        self._pending_lookups.clear()
        self._load_specs.clear()
        self._pending_loads.clear()
        self._lazy_scan_pending = False
        self._lazy_prefix_chains_by_block.clear()
        cleared = True if self.runtime is None else self.runtime.clear()
        if cleared:
            self._event_objects.clear()
            self._pending_event_objects.clear()
            if self._event_lookup is not None:
                self._event_lookup.future.cancel()
                self._event_lookup = None
        return cleared

    def close(self) -> None:
        self._lookup_executor.shutdown(wait=True, cancel_futures=True)
        self._pending_lookups.clear()
        if self.runtime is not None:
            self.runtime.close()
