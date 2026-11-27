# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Mapping

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
)
from vllm.v1.kv_offload.base import GPULoadStoreSpec


def build_aligned_boundary_store_jobs(
    config: SchedulerOffloadConfig,
    events_tracker: OffloadingEventsTracker,
    boundaries: Mapping[ReqId, list[tuple[int, int, int]]],
    req_states: Mapping[ReqId, RequestOffloadState],
    sink: StoreJobSink,
) -> None:
    """Build store jobs for recurrent states at aligned token boundaries."""
    num_groups = len(config.kv_group_configs)
    config_group_idx_by_group = {
        group.group_idx: config_group_idx
        for config_group_idx, group in enumerate(config.kv_group_configs)
    }
    for req_id, entries in boundaries.items():
        req_status = req_states.get(req_id)
        if req_status is None:
            continue
        req = req_status.req
        max_boundary = req_status.num_offloadable_tokens(req.num_tokens)
        for group_idx, block_id, boundary in entries:
            config_group_idx = config_group_idx_by_group.get(group_idx)
            if config_group_idx is None:
                continue
            group_config = config.kv_group_configs[config_group_idx]
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
                # Boundary-state sources are offered only once.
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
            events_tracker.record_store(
                req,
                group_config,
                boundary // group_config.tokens_per_chunk - 1,
                key,
            )
