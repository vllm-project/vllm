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
    seen_ptrs: dict[int, tuple[str, torch.Tensor]] = {}
    for name, value in kv_caches.items():
        tensor = repr_kv_cache_tensor(value)
        ptr = tensor.untyped_storage().data_ptr()
        if ptr not in seen_ptrs:
            seen_ptrs[ptr] = (name, tensor)

    unique_gpu_caches: dict[str, torch.Tensor] = {}
    for name, tensor in seen_ptrs.values():
        storage = tensor.untyped_storage()
        raw = torch.empty(0, dtype=torch.int8, device=device).set_(
            storage, 0, (storage.nbytes(),)
        )
        el = tensor.element_size()
        page_size_bytes = storage.nbytes() // num_blocks
        outer_dims = [
            d for d in range(tensor.ndim) if tensor.stride(d) * el > page_size_bytes
        ]
        if not outer_dims:
            unique_gpu_caches[name] = raw.view(num_blocks, -1)
        else:
            seg_stride = tensor.stride(outer_dims[0]) * el
            for idx in range(tensor.shape[outer_dims[0]]):
                offset = idx * seg_stride
                chunk = raw[offset : offset + seg_stride]
                unique_gpu_caches[f"{name}.{idx}"] = chunk.view(num_blocks, -1)
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
    is_packed = any(t.block_stride for t in gpu_config.kv_cache_tensors)
    assert not is_packed or all(t.block_stride for t in gpu_config.kv_cache_tensors)
    if is_packed:
        return gpu_config.kv_cache_tensors[0].size
    return sum(t.size for t in gpu_config.kv_cache_tensors)


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
