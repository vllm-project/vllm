# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from contextlib import suppress
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig


def invalidate_new_blocks(
    vllm_config: "VllmConfig",
    kv_cache_config: "KVCacheConfig",
    scheduler_output: "SchedulerOutput",
) -> None:
    if vllm_config.attention_config.hisparse_config is None:
        return

    block_ids = [
        block_id
        for request in scheduler_output.scheduled_new_reqs
        for block_id in request.block_ids[0]
    ]
    for new_block_ids in scheduler_output.scheduled_cached_reqs.new_block_ids:
        if new_block_ids is not None:
            block_ids.extend(new_block_ids[0])
    if not block_ids:
        return

    from vllm.v1.attention.backends.mla.hisparse import invalidate_blocks

    invalidate_blocks(
        block_ids,
        kv_cache_config.kv_cache_groups[0].kv_cache_spec.block_size,
    )


def prepare_batch(
    vllm_config: "VllmConfig", request_state_indices: torch.Tensor
) -> None:
    if vllm_config.attention_config.hisparse_config is None:
        return
    from vllm.v1.attention.backends.mla.hisparse import set_request_state_indices

    set_request_state_indices(request_state_indices)


def shutdown() -> None:
    from vllm.v1.attention.backends.mla.hisparse import release_pinned_state

    if release_pinned_state():
        with suppress(RuntimeError):
            torch._C._host_emptyCache()
