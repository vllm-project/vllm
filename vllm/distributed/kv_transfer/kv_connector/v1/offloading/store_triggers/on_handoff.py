# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    ReqId,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.state import (
    RequestOffloadState,
    resolve_partial_tail_sources,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.base import (  # noqa: E501
    StoreJobSink,
    StoreTriggerHandler,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import get_reusable_prefix_length
from vllm.v1.kv_offload.base import GPULoadStoreSpec, StoreTrigger


class OnHandoffStoreHandler(StoreTriggerHandler):
    """Stores the state blocks that the KV cache manager hands off
    (StoreTrigger.ON_HANDOFF).

    A recurrent source is handed off before reuse or request cleanup.
    Offer its state to the manager on the step that receives the hand-off.
    """

    trigger = StoreTrigger.ON_HANDOFF

    @override
    def first_live_chunk_idx(
        self, req_status: RequestOffloadState, config_group_idx: int
    ) -> int | None:
        # The aligned boundary and CoW sources come from the scheduler output.
        # The partial tail reads the request's block ids only of full-attention
        # groups: supports_partial_tail excludes sliding-window groups that are
        # not CoW sources, and the scheduler zeroes only sliding-window groups.
        return None

    @override
    def build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, RequestOffloadState],
        sink: StoreJobSink,
    ) -> None:
        block_state = scheduler_output.kv_connector_block_state
        handoffs = block_state.boundary_state_offloads if block_state else None
        if handoffs:
            self._build_aligned_boundary_store_jobs(handoffs, req_states, sink)
        if not self._config.supports_partial_tail:
            return
        if handoffs:
            self._build_partial_tail_store_jobs(handoffs, req_states, sink)
        for req_id in scheduler_output.finished_req_ids:
            req_status = req_states.get(req_id)
            if (
                req_status is not None
                and (tail := req_status.finished_partial_tail) is not None
            ):
                self._store_partial_tail(req_status, tail.boundary, tail.sources, sink)

    def _build_aligned_boundary_store_jobs(
        self,
        handoffs: dict[str, list[tuple[int, int, int]]],
        req_states: Mapping[ReqId, RequestOffloadState],
        sink: StoreJobSink,
    ) -> None:
        num_groups = len(self._config.kv_group_configs)
        config_group_idx_by_group = {
            config.group_idx: config_group_idx
            for config_group_idx, config in enumerate(self._config.kv_group_configs)
        }
        for req_id, entries in handoffs.items():
            req_status = req_states.get(req_id)
            if req_status is None:
                continue
            req = req_status.req
            max_boundary = req_status.num_offloadable_tokens(req.num_tokens)
            for group_idx, block_id, boundary in entries:
                config_group_idx = config_group_idx_by_group.get(group_idx)
                if config_group_idx is None:
                    continue
                group_config = self._config.kv_group_configs[config_group_idx]
                if (
                    block_id == 0
                    or boundary > max_boundary
                    or boundary % group_config.tokens_per_chunk != 0
                ):
                    continue

                key = req_status.register_boundary_key(group_idx, boundary)
                store_output = sink.prepare_store([key], req_status.req_context)
                if store_output is None:
                    continue
                if store_output.keys_to_retry:
                    # A hand-off is not offered again.
                    sink.record_allocation_failure()
                if not store_output.keys_to_store:
                    continue

                group_sizes = [0] * num_groups
                group_sizes[config_group_idx] = 1
                block_indices = [0] * num_groups
                block_indices[config_group_idx] = (
                    boundary // group_config.tokens_per_block - 1
                )
                sink.add_job(
                    req_id,
                    [key],
                    GPULoadStoreSpec(
                        [block_id],
                        group_sizes=group_sizes,
                        block_indices=block_indices,
                    ),
                    store_output.store_spec,
                    fenced_block_ids=[block_id],
                )
                self._events_tracker.record_store(
                    req,
                    group_config,
                    boundary // group_config.tokens_per_chunk - 1,
                    key,
                )

    def _build_partial_tail_store_jobs(
        self,
        handoffs: dict[str, list[tuple[int, int, int]]],
        req_states: Mapping[ReqId, RequestOffloadState],
        sink: StoreJobSink,
    ) -> None:
        cache_hit_alignment = self._config.cache_hit_alignment_tokens
        assert cache_hit_alignment is not None
        for req_id, entries in handoffs.items():
            req_status = req_states.get(req_id)
            if req_status is None:
                continue
            req = req_status.req
            boundary = get_reusable_prefix_length(
                req.num_prompt_tokens,
                num_hash_lookahead_tokens=self._config.num_hash_lookahead_tokens,
                alignment_tokens=cache_hit_alignment,
            )
            if (
                not 0 < boundary <= req_status.num_offloadable_tokens(req.num_tokens)
                or boundary % self._config.tokens_per_hash != 0
                or boundary % self._config.partial_tail_block_size == 0
            ):
                continue
            recurrent_sources = {
                group_id: block_id
                for group_id, block_id, token_end in entries
                if token_end == boundary
            }
            if not recurrent_sources:
                continue
            group_block_ids = {
                group.group_idx: state.block_ids
                for group, state in zip(
                    self._config.kv_group_configs, req_status.group_states
                )
            }
            sources = resolve_partial_tail_sources(
                self._config, boundary, recurrent_sources, group_block_ids
            )
            if sources is not None:
                self._store_partial_tail(req_status, boundary, sources, sink)

    def _store_partial_tail(
        self,
        req_status: RequestOffloadState,
        boundary: int,
        sources: list[tuple[int, int]],
        sink: StoreJobSink,
    ) -> None:
        groups = self._config.kv_group_configs
        keys = [
            req_status.register_boundary_key(group.group_idx, boundary)
            for group in groups
        ]
        store_output = sink.prepare_store(keys, req_status.req_context)
        if store_output is None:
            return
        if store_output.keys_to_retry:
            # A hand-off is not offered again.
            sink.record_allocation_failure()
        if not store_output.keys_to_store:
            return

        group_by_key = {key: idx for idx, key in enumerate(keys)}
        group_sizes = [0] * len(groups)
        block_indices = [0] * len(groups)
        source_blocks = []
        for key in store_output.keys_to_store:
            idx = group_by_key[key]
            block_id, block_index = sources[idx]
            group_sizes[idx] = 1
            block_indices[idx] = block_index
            source_blocks.append(block_id)
            self._events_tracker.record_partial_store(
                req_status.req, groups[idx], boundary, key
            )

        sink.add_job(
            req_status.req.request_id,
            store_output.keys_to_store,
            GPULoadStoreSpec(
                source_blocks, group_sizes=group_sizes, block_indices=block_indices
            ),
            store_output.store_spec,
            fenced_block_ids=list(dict.fromkeys(source_blocks)),
        )
