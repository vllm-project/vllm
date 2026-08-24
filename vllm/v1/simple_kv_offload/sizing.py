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


def repr_kv_cache_tensor(value: torch.Tensor | list[torch.Tensor]) -> torch.Tensor:
    assert isinstance(value, torch.Tensor | list)
    return value if isinstance(value, torch.Tensor) else value[0]


def build_unique_gpu_block_views(
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    num_blocks: int,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Build [num_blocks, block_bytes] views used for offload copy sizing."""
    unique_gpu_caches: dict[str, torch.Tensor] = {}
    seen: set[tuple[torch.device, int]] = set()
    for name, value in kv_caches.items():
        tensor = repr_kv_cache_tensor(value)
        storage = tensor.untyped_storage()
        key = (tensor.device, storage.data_ptr())
        if key in seen:
            continue
        seen.add(key)

        physical_per_block, remainder = divmod(tensor.shape[0], num_blocks)
        assert remainder == 0, (
            f"KV cache {name!r} has {tensor.shape[0]} physical blocks, which "
            f"is not divisible by {num_blocks} scheduler blocks"
        )
        block_bytes = tensor.stride(0) * tensor.element_size() * physical_per_block
        raw = torch.empty(0, dtype=torch.int8, device=device).set_(storage)
        regions = raw.view(-1, num_blocks, block_bytes)
        for idx, region in enumerate(regions):
            key_name = name if len(regions) == 1 else f"{name}.{idx}"
            unique_gpu_caches[key_name] = region
    return unique_gpu_caches


def total_bytes_per_block_from_views(
    unique_gpu_caches: dict[str, torch.Tensor],
) -> int:
    per_tensor_bpb = [
        t.stride(0) * t.element_size() for t in unique_gpu_caches.values()
    ]
    return sum(per_tensor_bpb)


def compute_total_bytes_per_block_from_kv_caches(
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
    num_blocks: int,
    device: torch.device,
) -> int:
    """Compute per-block offload bytes from live GPU KV tensors."""
    unique_gpu_caches = build_unique_gpu_block_views(kv_caches, num_blocks, device)
    return total_bytes_per_block_from_views(unique_gpu_caches)


def local_num_offload_blocks(capacity_bytes: int, total_bytes_per_block: int) -> int:
    assert total_bytes_per_block > 0
    return max(1, capacity_bytes // total_bytes_per_block)


def sync_num_offload_blocks_across_workers(num_offload_blocks: int) -> int:
    """All-reduce MIN so every rank allocates the same offload pool size."""
    from vllm.distributed.parallel_state import get_world_group

    world_group = get_world_group()
    if world_group.world_size <= 1:
        return num_offload_blocks

    blocks_tensor = torch.tensor(
        [num_offload_blocks], dtype=torch.int64, device="cpu"
    )
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
    tensors = gpu_config.kv_cache_tensors
    sizes = {t.size for t in tensors}
    if len(sizes) == 1:
        # All KVCacheTensor entries describe regions in one backing allocation.
        return tensors[0].size
    # Fallback when tensors describe disjoint allocations.
    return sum(t.size for t in tensors)


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
