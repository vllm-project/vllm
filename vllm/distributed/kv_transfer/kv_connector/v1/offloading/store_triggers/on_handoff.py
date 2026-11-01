# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping

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
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import GPULoadStoreSpec, StoreTrigger


class OnHandoffStoreHandler(StoreTriggerHandler):
    """Stores the state blocks that the KV cache manager hands off
    (StoreTrigger.ON_HANDOFF).

    The hand-off arrives when a recurrent state block is about to be reused.
    Its source is short-lived, so every store mode stores it at the same step.
    """

    trigger = StoreTrigger.ON_HANDOFF

    def __init__(
        self,
        config: SchedulerOffloadConfig,
        events_tracker: OffloadingEventsTracker,
    ):
        super().__init__(config, events_tracker)
        self._cow_source_groups = frozenset(
            group_config.group_idx
            for group_config in config.kv_group_configs
            if group_config.requires_cow_source
        )

    @override
    def first_live_chunk_idx(
        self, req_status: RequestOffloadState, config_idx: int
    ) -> int | None:
        # The source blocks come from the scheduler output, not from the
        # request's block list.
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
        if not handoffs:
            return
        self._build_aligned_boundary_store_jobs(handoffs, req_states, sink)
        if self._config.supports_partial_tail:
            self._build_partial_tail_store_jobs(
                scheduler_output, handoffs, req_states, sink
            )

    def _build_aligned_boundary_store_jobs(
        self,
        handoffs: dict[str, list[tuple[int, int, int]]],
        req_states: Mapping[ReqId, RequestOffloadState],
        sink: StoreJobSink,
    ) -> None:
        num_groups = len(self._config.kv_group_configs)
        config_idx_by_group = {
            config.group_idx: idx
            for idx, config in enumerate(self._config.kv_group_configs)
        }
        for req_id, entries in handoffs.items():
            req_status = req_states.get(req_id)
            if req_status is None:
                continue
            req = req_status.req
            max_boundary = req_status.num_offloadable_tokens(req.num_tokens)
            for group_idx, block_id, boundary in entries:
                config_idx = config_idx_by_group.get(group_idx)
                if config_idx is None:
                    continue
                group_config = self._config.kv_group_configs[config_idx]
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
                if not store_output.keys_to_store:
                    continue

                group_sizes = [0] * num_groups
                group_sizes[config_idx] = 1
                block_indices = [0] * num_groups
                block_indices[config_idx] = (
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
        scheduler_output: SchedulerOutput,
        handoffs: dict[str, list[tuple[int, int, int]]],
        req_states: Mapping[ReqId, RequestOffloadState],
        sink: StoreJobSink,
    ) -> None:
        partial_tail_block_size = self._config.partial_tail_block_size
        max_eagle_proof_margin = self._config.max_eagle_proof_margin
        for req_id, entries in handoffs.items():
            entries = [
                entry for entry in entries if entry[2] % partial_tail_block_size != 0
            ]
            if not entries:
                continue
            req_status = req_states.get(req_id)
            assert req_status is not None
            if max_eagle_proof_margin and not req_status.loaded_from_offload:
                # Requests that did not load from the offload tier keep their
                # EAGLE tail in the GPU prefix cache; skip the store.
                continue
            boundaries = {boundary for _, _, boundary in entries}
            assert len(boundaries) == 1
            boundary = boundaries.pop()
            req = req_status.req
            group_states = {
                group.group_idx: state
                for group, state in zip(
                    self._config.kv_group_configs, req_status.group_states
                )
            }
            max_boundary = min(
                req.num_prompt_tokens,
                req_status.max_offload_tokens or req.num_prompt_tokens,
            )
            assert boundary > 0
            assert boundary % self._config.tokens_per_hash == 0
            assert boundary <= max_boundary

            if max_eagle_proof_margin:
                # The CoW hand-off arrives on the step that computes the prompt
                # tokens past the boundary, and it is not offered again. Workers
                # submit a store at the next step's start, after this step's
                # forward, so count the prompt tokens this step computes.
                prompt_tokens_after_step = min(
                    req.num_computed_tokens
                    + scheduler_output.num_scheduled_tokens.get(req_id, 0),
                    req.num_prompt_tokens,
                )
                if boundary + max_eagle_proof_margin > prompt_tokens_after_step:
                    # The EAGLE proof is not computed by the end of this step.
                    continue

            cow_blocks = {group_idx: block_id for group_idx, block_id, _ in entries}
            assert self._cow_source_groups.issubset(cow_blocks)

            assert boundary % partial_tail_block_size != 0
            block_idx = boundary // partial_tail_block_size
            if any(
                group.group_idx not in self._cow_source_groups
                and block_idx >= len(group_states[group.group_idx].block_ids)
                for group in self._config.kv_group_configs
            ):
                continue
            key_tokens = [
                self._config.partial_tail_key_tokens(config_idx, boundary)
                for config_idx in range(len(self._config.kv_group_configs))
            ]
            keys = [
                req_status.register_boundary_key(group.group_idx, tokens)
                for group, tokens in zip(self._config.kv_group_configs, key_tokens)
            ]
            block_ids = [
                cow_blocks[group.group_idx]
                if group.group_idx in self._cow_source_groups
                else group_states[group.group_idx].block_ids[block_idx]
                for group in self._config.kv_group_configs
            ]
            assert all(block_id != 0 for block_id in block_ids)

            store_output = sink.prepare_store(keys, req_status.req_context)
            if store_output is None:
                continue
            if not store_output.keys_to_store:
                continue

            for group_config, tokens, key in zip(
                self._config.kv_group_configs, key_tokens, keys
            ):
                if key in store_output.keys_to_store:
                    self._events_tracker.record_partial_store(
                        req, group_config, tokens, key
                    )

            group_by_key = {key: idx for idx, key in enumerate(keys)}
            accepted_groups = [group_by_key[key] for key in store_output.keys_to_store]
            group_sizes = [0] * len(self._config.kv_group_configs)
            block_indices = [0] * len(self._config.kv_group_configs)
            for group_idx in accepted_groups:
                group_sizes[group_idx] = 1
                block_indices[group_idx] = block_idx
            source_blocks = [block_ids[group_idx] for group_idx in accepted_groups]

            sink.add_job(
                req_id,
                store_output.keys_to_store,
                GPULoadStoreSpec(
                    source_blocks,
                    group_sizes=group_sizes,
                    block_indices=block_indices,
                ),
                store_output.store_spec,
                fenced_block_ids=source_blocks,
            )
