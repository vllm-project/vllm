# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared sizing helpers for SimpleCPUOffloadConnector."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig


def offload_capacity_blocks(
    capacity_bytes: int,
    bytes_per_block: int,
    shared_bytes_per_block: int = 0,
    world_size: int = 1,
) -> int:
    """Blocks that fit when each of `world_size` ranks has `capacity_bytes`
    and `shared_bytes_per_block` of every block are stored once for all ranks."""
    private_bytes_per_block = bytes_per_block - shared_bytes_per_block
    return (world_size * capacity_bytes) // (
        world_size * private_bytes_per_block + shared_bytes_per_block
    )


def num_offload_blocks(
    kv_cache_config: KVCacheConfig,
    capacity_bytes: int,
    bytes_per_block: int,
    shared_bytes_per_block: int = 0,
    world_size: int = 1,
) -> int:
    """How many blocks of the pipeline stage with the largest blocks fit in
    `capacity_bytes`; without `pipeline_stages`, blocks of `bytes_per_block`."""
    largest = max(
        (stage.bytes_per_block for stage in kv_cache_config.pipeline_stages),
        default=bytes_per_block,
    )
    return max(
        1,
        offload_capacity_blocks(
            capacity_bytes, largest, shared_bytes_per_block, world_size
        ),
    )


def rank_shared_layer_bytes(
    kv_cache_config: KVCacheConfig, vllm_config: VllmConfig
) -> dict[str, int]:
    """Bytes per block of each layer whose KV every rank holds identically.

    Applies to tensor parallelism alone on one node with the multiprocessing
    executor. Cache groups can overlay each other in memory, so a layer is
    shared only if its regions hold nothing but its own KV.
    """
    from vllm.v1.kv_cache_interface import AttentionSpec

    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.tensor_parallel_size == 1
        or parallel_config.world_size != parallel_config.tensor_parallel_size
        or parallel_config.prefill_context_parallel_size > 1
        or parallel_config.decode_context_parallel_size > 1
        or parallel_config.distributed_executor_backend != "mp"
        or parallel_config.nnodes_within_dp != 1
    ):
        return {}
    num_blocks = kv_cache_config.num_blocks
    tensors = kv_cache_config.kv_cache_tensors
    if any(
        t.host_resident or t.layer_stride < num_blocks * t.block_stride for t in tensors
    ):
        return {}
    replicated = {
        name
        for group in kv_cache_config.kv_cache_groups
        for name, spec in group.kv_cache_specs.items()
        if isinstance(spec, AttentionSpec) and spec.max_tp_shards == 1
    }
    spans = [(t.offset, t.offset + len(t.layers) * t.layer_stride) for t in tensors]
    shared: dict[str, int] = {}
    for tensor, (start, end) in zip(tensors, spans):
        overlapping = sum(s < end and start < e for s, e in spans)
        if overlapping == 1 and replicated.issuperset(tensor.layers):
            shared.update(
                dict.fromkeys(tensor.layers, tensor.layer_stride // num_blocks)
            )
    return shared


def gpu_total_bytes(gpu_config: KVCacheConfig) -> int:
    assert len(gpu_config.kv_cache_tensors) > 0
    # Every KVCacheTensor describes placement within the same backing allocation,
    # so its size is the total GPU KV cache size.
    return gpu_config.kv_cache_tensors[0].size
