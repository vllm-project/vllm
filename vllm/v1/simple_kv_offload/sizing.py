# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared sizing helpers for SimpleCPUOffloadConnector."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig


def num_offload_blocks(
    kv_cache_config: KVCacheConfig, capacity_bytes: int, bytes_per_block: int
) -> int:
    """The offload pool size in blocks, the same on every worker and the
    scheduler: as many blocks as fit in `capacity_bytes` on the pipeline stage
    with the largest blocks, or with `bytes_per_block`, this worker's, when the
    stages are unknown.
    """
    largest = max(
        (stage.bytes_per_block for stage in kv_cache_config.pipeline_stages),
        default=bytes_per_block,
    )
    assert bytes_per_block <= largest, (
        f"Blocks of {bytes_per_block} bytes exceed the largest pipeline "
        f"stage's {largest}"
    )
    return max(1, capacity_bytes // largest)


def gpu_total_bytes(gpu_config: KVCacheConfig) -> int:
    assert len(gpu_config.kv_cache_tensors) > 0
    # Every KVCacheTensor describes placement within the same backing allocation,
    # so its size is the total GPU KV cache size.
    return gpu_config.kv_cache_tensors[0].size
