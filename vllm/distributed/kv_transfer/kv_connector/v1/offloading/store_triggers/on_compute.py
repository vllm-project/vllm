# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping
from itertools import chain

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
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.boundary import (  # noqa: E501
    build_aligned_boundary_store_jobs,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.store_triggers.candidates import (  # noqa: E501
    final_swa_tail_start_chunk,
    select_store_candidates,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import round_down
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    OffloadPolicy,
    StoreTrigger,
)
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)


class OnComputeStoreHandler(StoreTriggerHandler):
    """Stores computed chunks and retained recurrent states at request finish."""

    trigger = StoreTrigger.ON_COMPUTE

    def __init__(
        self,
        config: SchedulerOffloadConfig,
        events_tracker: OffloadingEventsTracker,
    ):
        super().__init__(config, events_tracker)
        # req_id -> per group, the index of the next chunk to offer.
        self._next_stored_chunk_idxs: dict[ReqId, list[int]] = {}

    @override
    def on_load_scheduled(
        self,
        req_status: RequestOffloadState,
        num_cached_tokens: int,
    ) -> None:
        # Skip prefix-hit chunks for block-level policy; for
        # request-level, the next stored chunk stays at 0 so all
        # chunks (including hits) are offloaded.
        # Round down: after a partial-tail hit the last chunk is only
        # checkpointed up to the hit, so it must still be offered.
        if req_status.offloading_context.policy != OffloadPolicy.CHUNK_LEVEL:
            return
        self._next_stored_chunk_idxs[req_status.req.request_id] = [
            num_cached_tokens // group_config.tokens_per_chunk
            for group_config in self._config.kv_group_configs
        ]

    @override
    def on_request_removed(self, req_id: ReqId) -> None:
        self._next_stored_chunk_idxs.pop(req_id, None)

    @override
    def on_request_restarted(self, req_id: ReqId) -> None:
        # Offer every chunk again.
        self._next_stored_chunk_idxs.pop(req_id, None)

    @override
    def reset(self) -> None:
        # Active requests store again from chunk 0.
        self._next_stored_chunk_idxs.clear()

    @override
    def first_live_chunk_idx(
        self, req_status: RequestOffloadState, config_group_idx: int
    ) -> int | None:
        next_idxs = self._next_stored_chunk_idxs.get(req_status.req.request_id)
        if next_idxs is None:
            return 0
        # The final SWA tail at finish can start below the stored progress.
        tail_start = final_swa_tail_start_chunk(
            self._config, req_status, config_group_idx
        )
        if tail_start is None:
            return next_idxs[config_group_idx]
        return min(next_idxs[config_group_idx], tail_start)

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
        next_idxs = self._next_idxs(req_status)
        for config_group_idx, (group_config, group_state) in enumerate(
            zip(self._config.kv_group_configs, req_status.group_states)
        ):
            # max(): the finish-time clamp can lower num_offloadable_tokens, and
            # the index must not move back past chunks already stored.
            next_idxs[config_group_idx] = max(
                next_idxs[config_group_idx],
                req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                ),
            )

    @override
    def build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
        req_states: Mapping[ReqId, RequestOffloadState],
        sink: StoreJobSink,
    ) -> None:
        blocks_per_chunk = self._config.blocks_per_chunk
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
                self._next_idxs(req_status),
            )
            num_offloadable_tokens = candidates.num_offloadable_tokens
            new_offload_keys = candidates.keys
            group_store_ranges = candidates.group_ranges

            if not new_offload_keys:
                self._advance_stored_idx(req_status, num_offloadable_tokens)
                continue

            next_idxs = self._next_idxs(req_status)
            store_output = sink.prepare_store(new_offload_keys, req_status.req_context)
            if store_output is None:
                logger.warning("Request %s: cannot store chunks", req_id)
                continue

            if store_output.keys_to_retry:
                # Each group's range ends at its first chunk to retry, which is
                # offered again on the next step.
                key_chunks = dict(zip(new_offload_keys, candidates.key_chunks))
                for key in store_output.keys_to_retry:
                    assert key in key_chunks, "A key to retry was not offered"
                    config_group_idx, chunk_idx = key_chunks[key]
                    start_chunk_idx, end_chunk_idx = group_store_ranges[
                        config_group_idx
                    ]
                    group_store_ranges[config_group_idx] = (
                        start_chunk_idx,
                        min(end_chunk_idx, chunk_idx),
                    )
                retried_keys = set(store_output.keys_to_retry)
                assert all(
                    (key in retried_keys)
                    == (chunk_idx >= group_store_ranges[config_group_idx][1])
                    for key, (config_group_idx, chunk_idx) in key_chunks.items()
                ), "The keys to retry are not each group's offered keys from its cut"
                assert retried_keys.isdisjoint(store_output.keys_to_store), (
                    "A key is both stored and retried"
                )
                if req.is_finished():
                    logger.warning("Request %s: cannot store chunks", req_id)
                    sink.record_allocation_failure()

            if not store_output.keys_to_store:
                for config_group_idx, (_, num_chunks) in enumerate(group_store_ranges):
                    next_idxs[config_group_idx] = max(
                        next_idxs[config_group_idx], num_chunks
                    )
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

        finished_decode_states = {}
        for req_id in scheduler_output.finished_req_ids or ():
            req_status = req_states.get(req_id)
            if req_status is not None:
                states = self._collect_finished_decode_states(req_status)
                if states:
                    finished_decode_states[req_id] = states
        if finished_decode_states:
            build_aligned_boundary_store_jobs(
                self._config,
                self._events_tracker,
                finished_decode_states,
                req_states,
                sink,
            )

    def _collect_finished_decode_states(
        self, req_status: RequestOffloadState
    ) -> list[tuple[int, int, int]]:
        block_ids = req_status.finished_decode_block_ids
        if block_ids is None:
            return []

        request = req_status.req
        if (
            request.num_in_flight_tokens
            or request.num_preemptions
            or request.resumable
            or request.status
            not in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
                RequestStatus.FINISHED_REPETITION,
            )
        ):
            return []

        confirmed_tokens = request.num_computed_tokens
        max_boundary = req_status.num_offloadable_tokens(confirmed_tokens)
        if max_boundary <= request.num_prompt_tokens:
            return []

        states = []
        for group in self._config.kv_group_configs:
            if not group.requires_cow_source:
                continue
            boundary = round_down(confirmed_tokens, group.tokens_per_block)
            if (
                boundary <= request.num_prompt_tokens
                or boundary == confirmed_tokens
                or boundary > max_boundary
                or boundary % group.tokens_per_chunk
            ):
                continue
            group_blocks = block_ids.get(group.group_idx, ())
            block_idx = boundary // group.tokens_per_block - 1
            if block_idx < len(group_blocks) and (block_id := group_blocks[block_idx]):
                states.append((group.group_idx, block_id, boundary))
        return states
