# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from itertools import chain
from typing import TYPE_CHECKING

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    ReqId,
    TransferJob,
    TransferJobStatus,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventsTracker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.base import (  # noqa: E501
    StoreJobsOutput,
    StoreTriggerHandler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.common import (  # noqa: E501
    select_store_candidates,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadingManager,
    OffloadPolicy,
    StoreTrigger,
)

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestOffloadState,
        SchedulerOffloadConfig,
    )

logger = init_logger(__name__)


class OnComputeStoreHandler(StoreTriggerHandler):
    """Offers each chunk when its KV data is computed (StoreTrigger.ON_COMPUTE)."""

    def __init__(
        self,
        config: "SchedulerOffloadConfig",
        manager: OffloadingManager,
        events_tracker: OffloadingEventsTracker,
    ):
        super().__init__(config, manager, events_tracker)
        self._cow_source_groups = frozenset(
            group_config.group_idx
            for group_config in config.kv_group_configs
            if group_config.requires_cow_source
        )

    @override
    def build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, "RequestOffloadState"],
    ) -> StoreJobsOutput:
        output = self._build_partial_tail_store_jobs(scheduler_output, req_states)
        store_output = self._build_store_jobs(scheduler_output, req_states)
        output.jobs.extend(store_output.jobs)
        output.num_allocation_failures += store_output.num_allocation_failures
        return output

    @override
    def on_load_scheduled(
        self, req_status: "RequestOffloadState", num_cached_tokens: int
    ) -> None:
        # Skip prefix-hit chunks for block-level policy; for
        # request-level, next_stored_chunk_idx stays at 0 so all
        # chunks (including hits) are offloaded.
        if req_status.offloading_context.policy != OffloadPolicy.CHUNK_LEVEL:
            return
        for group_config, group_state in zip(
            self._config.kv_group_configs, req_status.group_states
        ):
            group_state.next_stored_chunk_idx = cdiv(
                num_cached_tokens, group_config.tokens_per_chunk
            )

    @override
    def reset(self, req_states: Mapping[ReqId, "RequestOffloadState"]) -> None:
        # Active requests store again from chunk 0.
        for req_status in req_states.values():
            for group_state in req_status.group_states:
                group_state.next_stored_chunk_idx = 0

    def _advance_stored_idx(
        self, req_status: "RequestOffloadState", num_offloadable_tokens: int
    ) -> None:
        # max(): at the prefill->decode transition of a chunk-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored chunks.
        for group_config, group_state in zip(
            self._config.kv_group_configs, req_status.group_states
        ):
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                ),
            )

    def _build_aligned_boundary_store_jobs(
        self,
        handoffs: dict[str, list[tuple[int, int, int]]],
        req_states: Mapping[ReqId, "RequestOffloadState"],
    ) -> StoreJobsOutput:
        output = StoreJobsOutput()
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

                key = req_status.make_boundary_key(group_idx, boundary)
                store_output = self._manager.prepare_store(
                    [key], req_status.req_context, StoreTrigger.ON_COMPUTE
                )
                if store_output is None:
                    output.num_allocation_failures += 1
                    continue
                if not store_output.keys_to_store:
                    continue

                group_sizes = [0] * num_groups
                group_sizes[config_idx] = 1
                block_indices = [0] * num_groups
                block_indices[config_idx] = (
                    boundary // group_config.tokens_per_block - 1
                )
                job_status = TransferJobStatus(
                    req_id=req_id,
                    pending_count=self._config.num_workers,
                    keys={key},
                    is_store=True,
                    fenced_block_ids=[block_id],
                )
                job = TransferJob(
                    req_id=req_id,
                    src_spec=GPULoadStoreSpec(
                        [block_id],
                        group_sizes=group_sizes,
                        block_indices=block_indices,
                    ),
                    dst_spec=store_output.store_spec,
                )
                output.jobs.append((job, job_status))
                self._events_tracker.record_store(
                    req,
                    group_config,
                    boundary // group_config.tokens_per_chunk - 1,
                    key,
                )
        return output

    def _build_partial_tail_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, "RequestOffloadState"],
    ) -> StoreJobsOutput:
        block_state = scheduler_output.kv_connector_block_state
        handoffs = block_state.boundary_state_offloads if block_state else None
        if not handoffs:
            return StoreJobsOutput()

        output = self._build_aligned_boundary_store_jobs(handoffs, req_states)
        if not self._config.supports_partial_tail:
            return output

        partial_tail_block_size = self._config.partial_tail_block_size
        for req_id, entries in handoffs.items():
            entries = [
                entry for entry in entries if entry[2] % partial_tail_block_size != 0
            ]
            if not entries:
                continue
            req_status = req_states.get(req_id)
            assert req_status is not None
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
            keys = [
                req_status.make_boundary_key(group.group_idx, boundary)
                for group in self._config.kv_group_configs
            ]
            block_ids = [
                cow_blocks[group.group_idx]
                if group.group_idx in self._cow_source_groups
                else group_states[group.group_idx].block_ids[block_idx]
                for group in self._config.kv_group_configs
            ]
            assert all(block_id != 0 for block_id in block_ids)

            store_output = self._manager.prepare_store(
                keys, req_status.req_context, StoreTrigger.ON_COMPUTE
            )
            if store_output is None:
                output.num_allocation_failures += 1
                continue
            if not store_output.keys_to_store:
                continue

            for group_config, key in zip(self._config.kv_group_configs, keys):
                if key in store_output.keys_to_store:
                    self._events_tracker.record_partial_store(
                        req, group_config, boundary, key
                    )

            group_by_key = {key: idx for idx, key in enumerate(keys)}
            accepted_groups = [group_by_key[key] for key in store_output.keys_to_store]
            group_sizes = [0] * len(self._config.kv_group_configs)
            block_indices = [0] * len(self._config.kv_group_configs)
            for group_idx in accepted_groups:
                group_sizes[group_idx] = 1
                block_indices[group_idx] = block_idx
            source_blocks = [block_ids[group_idx] for group_idx in accepted_groups]

            job_status = TransferJobStatus(
                req_id=req_id,
                pending_count=self._config.num_workers,
                keys=set(store_output.keys_to_store),
                is_store=True,
                fenced_block_ids=source_blocks,
            )
            job = TransferJob(
                req_id=req_id,
                src_spec=GPULoadStoreSpec(
                    source_blocks,
                    group_sizes=group_sizes,
                    block_indices=block_indices,
                ),
                dst_spec=store_output.store_spec,
            )
            output.jobs.append((job, job_status))

        return output

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, "RequestOffloadState"],
    ) -> StoreJobsOutput:
        blocks_per_chunk = self._config.blocks_per_chunk
        output = StoreJobsOutput()
        for req_id in chain(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.finished_req_ids or (),
        ):
            req_status = req_states.get(req_id)
            if req_status is None:
                continue
            req = req_status.req
            candidates = select_store_candidates(
                self._config,
                scheduler_output,
                req_status,
                [g.next_stored_chunk_idx for g in req_status.group_states],
            )
            num_offloadable_tokens = candidates.num_offloadable_tokens
            new_offload_keys = candidates.keys
            group_store_ranges = candidates.group_ranges

            if not new_offload_keys:
                self._advance_stored_idx(req_status, num_offloadable_tokens)
                continue

            store_output = self._manager.prepare_store(
                new_offload_keys, req_status.req_context, StoreTrigger.ON_COMPUTE
            )
            if store_output is None:
                output.num_allocation_failures += 1
                logger.warning("Request %s: cannot store chunks", req_id)
                continue

            if not store_output.keys_to_store:
                self._advance_stored_idx(req_status, num_offloadable_tokens)
                continue

            keys_to_store = set(store_output.keys_to_store)

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

                    self._events_tracker.record_store(
                        req, group_config, chunk_idx, offload_key
                    )

                    gpu_block_idx = chunk_idx * blocks_per_chunk
                    for i in range(blocks_per_chunk):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            fenced_block_ids.append(block_id)
                        else:
                            deferred_fence_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_chunk_idx = max(
                    group_state.next_stored_chunk_idx, num_chunks
                )

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            # the non-sliding window blocks will be watched only
            # when the request finishes
            job_status = TransferJobStatus(
                req_id=req_id,
                pending_count=self._config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                deferred_fence_block_ids=deferred_fence_block_ids,
                fenced_block_ids=fenced_block_ids or None,
            )
            job = TransferJob(req_id=req_id, src_spec=src_spec, dst_spec=dst_spec)
            output.jobs.append((job, job_status))

            logger.debug(
                "Request %s offloading %s chunks upto %d tokens",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
            )

        return output
