# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared sizing helpers for SimpleCPUOffloadConnector."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.distributed as dist

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


def sync_num_offload_blocks_across_workers(num_offload_blocks: int) -> int:
    """All-reduce MIN so every rank allocates the same offload pool size."""
    if not dist.is_initialized():
        return num_offload_blocks

    from vllm.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if world_group.world_size <= 1:
        return num_offload_blocks

    blocks_tensor = torch.tensor([num_offload_blocks], dtype=torch.int64, device="cpu")
    dist.all_reduce(blocks_tensor, group=world_group.cpu_group, op=dist.ReduceOp.MIN)
    synced = int(blocks_tensor.item())
    if synced != num_offload_blocks:
        logger.info(
            "SimpleCPUOffload: aligned num_offload_blocks from %d to %d "
            "across %d workers",
            num_offload_blocks,
            synced,
            world_group.world_size,
        )
    return synced


def gpu_total_bytes(gpu_config: KVCacheConfig) -> int:
    assert len(gpu_config.kv_cache_tensors) > 0
    # Every KVCacheTensor describes placement within the same backing allocation,
    # so its size is the total GPU KV cache size.
    return gpu_config.kv_cache_tensors[0].size


def compute_num_offload_blocks_from_config(
    gpu_config: KVCacheConfig,
    capacity_bytes: int,
) -> int:
    """Conservative config-based estimate before live tensors are registered."""
    gpu_total = gpu_total_bytes(gpu_config)
    return max(1, gpu_config.num_blocks * capacity_bytes // gpu_total)


def compute_num_offload_blocks_from_configs(
    kv_cache_configs: list[KVCacheConfig],
    capacity_bytes: int,
) -> int:
    """Return the min offload block count implied by all worker configs."""
    assert kv_cache_configs
    return min(
        compute_num_offload_blocks_from_config(cfg, capacity_bytes)
        for cfg in kv_cache_configs
    )
