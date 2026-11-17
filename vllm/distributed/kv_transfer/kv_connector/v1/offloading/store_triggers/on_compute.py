# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Iterable, Mapping
from itertools import chain
from math import lcm

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.state import (
    RequestOffloadState,
    SchedulerOffloadConfig,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.base import (  # noqa: E501
    StoreJobSink,
    StoreTriggerHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.candidates import (  # noqa: E501
    select_store_candidates,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadKey,
    OffloadPolicy,
    StoreTrigger,
)

logger = init_logger(__name__)

# Steps a blocked request skips while PUT admission state is unchanged, before
# re-planning anyway (bounds both scan cost and retry latency).
_PUT_RETRY_MAX_SKIPS = 15


class OnComputeStoreHandler(StoreTriggerHandler):
    """Offers each chunk when its KV data is computed (StoreTrigger.ON_COMPUTE)."""

    trigger = StoreTrigger.ON_COMPUTE

    def __init__(
        self,
        config: SchedulerOffloadConfig,
        events_tracker: OffloadingEventsTracker,
        put_batch_chunks: int = 0,
    ):
        super().__init__(config, events_tracker)
        # req_id -> per group, the index of the next chunk to offer.
        self._next_stored_chunk_idxs: dict[ReqId, list[int]] = {}
        # Opt-in PUT batching: offers are clipped to whole aligned units.
        self._put_batch_chunks = put_batch_chunks
        self._put_alignment = lcm(
            config.alignment_tokens or 1,
            *(g.tokens_per_chunk for g in config.kv_group_configs),
        )
        self._put_retry_after: ReqId | None = None
        self._put_retry_gate: dict[ReqId, tuple[tuple[int, ...], int]] = {}

    @override
    def on_load_scheduled(
        self,
        req_status: RequestOffloadState,
        num_cached_tokens: int,
    ) -> None:
        # Skip prefix-hit chunks for block-level policy; for
        # request-level, the next stored chunk stays at 0 so all
        # chunks (including hits) are offloaded.
        if req_status.offloading_context.policy != OffloadPolicy.CHUNK_LEVEL:
            return
        self._next_stored_chunk_idxs[req_status.req.request_id] = [
            cdiv(num_cached_tokens, group_config.tokens_per_chunk)
            for group_config in self._config.kv_group_configs
        ]

    @override
    def on_request_removed(self, req_id: ReqId) -> None:
        self._next_stored_chunk_idxs.pop(req_id, None)
        self._forget_put_retry(req_id)

    def _forget_put_retry(self, req_id: ReqId) -> None:
        self._put_retry_gate.pop(req_id, None)
        if self._put_retry_after == req_id:
            self._put_retry_after = None

    @override
    def on_request_restarted(self, req_id: ReqId) -> None:
        # Offer every chunk again.
        self._next_stored_chunk_idxs.pop(req_id, None)
        self._forget_put_retry(req_id)

    @override
    def reset(self) -> None:
        # Active requests store again from chunk 0.
        self._next_stored_chunk_idxs.clear()
        self._put_retry_gate.clear()
        self._put_retry_after = None

    @override
    def first_live_chunk_idx(
        self, req_status: RequestOffloadState, config_group_idx: int
    ) -> int | None:
        next_idxs = self._next_stored_chunk_idxs.get(req_status.req.request_id)
        return 0 if next_idxs is None else next_idxs[config_group_idx]

    def _next_idxs(self, req_status: RequestOffloadState) -> list[int]:
        """The next chunk to offer in each group, created at 0."""
        req_id = req_status.req.request_id
        next_idxs = self._next_stored_chunk_idxs.get(req_id)
        if next_idxs is None:
            next_idxs = [0] * len(req_status.group_states)
            self._next_stored_chunk_idxs[req_id] = next_idxs
        return next_idxs

    def _advance_stored_idx(
        self, req_status: RequestOffloadState, num_offloadable_tokens: int
    ) -> None:
        # max(): at the prefill->decode transition of a chunk-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored chunks.
        apply_eagle_drop = not self._config.num_hash_lookahead_tokens
        next_idxs = self._next_idxs(req_status)
        for config_group_idx, (group_config, group_state) in enumerate(
            zip(self._config.kv_group_configs, req_status.group_states)
        ):
            next_idxs[config_group_idx] = max(
                next_idxs[config_group_idx],
                req_status.storable_chunks(
                    group_config,
                    group_state,
                    num_offloadable_tokens,
                    apply_eagle_drop,
                ),
            )

    def _clip_put_frontier(
        self,
        ranges: list[tuple[int, int]],
        key_ends: list[tuple[int, OffloadKey, int, int]],
        budget: int,
        needs_credit: set[OffloadKey],
        final: bool = False,
    ) -> tuple[list[tuple[int, int]], dict[OffloadKey, tuple[int, int]]]:
        """Select whole aligned multi-group units, never a flattened prefix.

        Eligibility/masks were computed against the ORIGINAL store horizon.
        Clipping must not make an intermediate frontier look like a final SWA
        tail. A minimum unit may exceed the nominal batch size only if all its
        keys fit real credits (which are bounded by the PUT staging limit).
        `final` (a finished request, whose blocks are freed after this
        step) waives the batch-size cap; `budget` is then the allocator's
        admissible count, so the offer never exceeds what prepare_store can
        accept (it is all-or-nothing).
        Only `needs_credit` keys count against the budget and batch size:
        resident keys (e.g. shared prefixes) allocate nothing.
        """
        alignment = self._put_alignment
        units: dict[int, set[OffloadKey]] = {}
        for end, key, _, _ in key_ends:
            units.setdefault(cdiv(end, alignment) * alignment, set()).add(key)
        chosen: set[OffloadKey] = set()
        charged: set[OffloadKey] = set()
        frontier = 0
        for end, keys in sorted(units.items()):
            size = len(charged | (keys & needs_credit))
            if size > budget or (
                charged and not final and size > self._put_batch_chunks
            ):
                break
            chosen.update(keys)
            charged.update(keys & needs_credit)
            frontier = end
        if chosen and len(chosen) == len({key for _, key, _, _ in key_ends}):
            # All eligible keys fit: resolve legitimately skipped trailing
            # ranges as the original scheduler did, including SWA decode gaps.
            frontier = max(
                end * group.tokens_per_chunk
                for group, (_, end) in zip(self._config.kv_group_configs, ranges)
            )
        clipped = [
            (start, max(start, min(end, frontier // group.tokens_per_chunk)))
            for group, (start, end) in zip(self._config.kv_group_configs, ranges)
        ]
        # Preserve an eligible occurrence, not just key membership: a duplicate
        # at an earlier position may have an unavailable or unreachable source.
        # This mapping is bounded by the admitted frontier/credit limit, even
        # though eligibility planning still enumerates the remaining tail.
        sources: dict[OffloadKey, tuple[int, int]] = {}
        for end, key, group_idx, chunk_idx in key_ends:
            if end <= frontier:
                sources.setdefault(key, (group_idx, chunk_idx))
        # Insertion order preserves CPU/GPU transfer-spec correspondence.
        return clipped, sources

    @override
    def build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, RequestOffloadState],
        sink: StoreJobSink,
    ) -> None:
        blocks_per_chunk = self._config.blocks_per_chunk
        put = bool(self._put_batch_chunks)
        req_ids: Iterable[ReqId] = chain(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.finished_req_ids or (),
        )
        if put:
            for req_id in scheduler_output.preempted_req_ids or ():
                self._forget_put_retry(req_id)
            req_ids = list(dict.fromkeys(req_ids))
            if self._put_retry_after in req_ids:
                start = req_ids.index(self._put_retry_after) + 1
                req_ids = req_ids[start:] + req_ids[:start]
            # Finished requests get one store pass before their blocks are
            # freed, so they claim primary capacity ahead of running ones.
            finished = scheduler_output.finished_req_ids or set()
            req_ids.sort(key=lambda r: r not in finished)
            # Computed once per step and refreshed only after a store is
            # prepared: gated or blocked requests leave it unchanged.
            admission_state = sink.put_admission_state()
        for req_id in req_ids:
            req_status = req_states.get(req_id)
            if req_status is None:
                continue
            req = req_status.req
            if put:
                if req_id in (scheduler_output.preempted_req_ids or ()):
                    continue
                budget = admission_state[0]
                previous, retries = self._put_retry_gate.get(req_id, ((), 0))
                if (
                    previous == admission_state
                    and retries < _PUT_RETRY_MAX_SKIPS
                    and not req.is_finished()
                ):
                    self._put_retry_gate[req_id] = (previous, retries + 1)
                    continue
                self._put_retry_gate.pop(req_id, None)
                if req.is_finished():
                    # Last chance: blocks are freed after this step, so the
                    # allocator, not the advisory credit, bounds this store.
                    budget = sink.num_admissible_chunks()
            candidates = select_store_candidates(
                self._config,
                scheduler_output,
                req_status,
                self._next_idxs(req_status),
                put_batching=put,
            )
            num_offloadable_tokens = candidates.num_offloadable_tokens
            new_offload_keys = candidates.keys
            group_store_ranges = candidates.group_ranges

            store_sources: dict[OffloadKey, tuple[int, int]] = {}
            if put and new_offload_keys:
                group_store_ranges, store_sources = self._clip_put_frontier(
                    group_store_ranges,
                    candidates.key_ends,
                    budget,
                    sink.keys_needing_put_credit(
                        new_offload_keys, req_status.req_context
                    ),
                    req.is_finished(),
                )
                new_offload_keys = list(store_sources)
                if not new_offload_keys:
                    self._put_retry_gate[req_id] = (admission_state, 0)
                    continue

            if not new_offload_keys:
                self._advance_stored_idx(req_status, num_offloadable_tokens)
                continue

            deferrable = put and not req.is_finished()
            store_output = sink.prepare_store(
                new_offload_keys, req_status.req_context, expected_failure=deferrable
            )
            if put:
                admission_state = sink.put_admission_state()
            if store_output is None:
                if deferrable:
                    # Expected under batching: retried once capacity changes.
                    self._put_retry_gate[req_id] = (admission_state, 0)
                    logger.debug("Request %s: PUT batch deferred", req_id)
                else:
                    logger.warning("Request %s: cannot store chunks", req_id)
                continue

            if put:
                self._put_retry_after = req_id

            if not store_output.keys_to_store:
                if put:
                    next_idxs = self._next_idxs(req_status)
                    for i, (_, end) in enumerate(group_store_ranges):
                        next_idxs[i] = max(next_idxs[i], end)
                else:
                    self._advance_stored_idx(req_status, num_offloadable_tokens)
                continue

            ordered_keys_to_store = tuple(dict.fromkeys(store_output.keys_to_store))
            keys_to_store = set(ordered_keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            fenced_block_ids: list[int] = []
            deferred_fence_block_ids: list[int] = []
            for group_config, group_state, store_range in zip(
                self._config.kv_group_configs,
                req_status.group_states,
                group_store_ranges,
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )
                start_chunk_idx, num_chunks = store_range
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_chunk_idx:num_chunks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    chunk_idx = start_chunk_idx + idx
                    if put and store_sources.get(offload_key) != (
                        group_config.group_idx,
                        chunk_idx,
                    ):
                        continue

                    self._events_tracker.record_store(
                        req, group_config, chunk_idx, offload_key
                    )

                    gpu_block_idx = chunk_idx * blocks_per_chunk
                    if start_gpu_block_idx is None:
                        start_gpu_block_idx = gpu_block_idx
                    for i in range(blocks_per_chunk):
                        block_id = block_ids[gpu_block_idx + i]
                        # Keep null pages inside the chunk as placeholders so
                        # later blocks retain their logical offsets in the CPU
                        # chunk. The null GPU block is zeroed and can be copied
                        # safely; it must not be fenced as a live allocation.
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if block_id == 0:
                            continue
                        if is_sliding_window:
                            fenced_block_ids.append(block_id)
                        else:
                            deferred_fence_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)

            next_idxs = self._next_idxs(req_status)
            for config_group_idx, (_, num_chunks) in enumerate(group_store_ranges):
                next_idxs[config_group_idx] = max(
                    next_idxs[config_group_idx], num_chunks
                )

            # the non-sliding window blocks will be watched only
            # when the request finishes
            sink.add_job(
                req_id,
                ordered_keys_to_store,
                GPULoadStoreSpec(
                    src_block_ids,
                    group_sizes=group_sizes,
                    block_indices=block_indices,
                ),
                store_output.store_spec,
                fenced_block_ids=fenced_block_ids or None,
                deferred_fence_block_ids=deferred_fence_block_ids,
            )
