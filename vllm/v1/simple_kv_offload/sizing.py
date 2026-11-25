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
    """How many blocks of the pipeline stage with the largest blocks fit in
    `capacity_bytes`; without `pipeline_stages`, blocks of `bytes_per_block`."""
    largest = max(
        (stage.bytes_per_block for stage in kv_cache_config.pipeline_stages),
        default=bytes_per_block,
    )
    return max(1, capacity_bytes // largest)


def gpu_total_bytes(gpu_config: KVCacheConfig) -> int:
    assert len(gpu_config.kv_cache_tensors) > 0
    # Every KVCacheTensor describes placement within the same backing allocation,
    # so its size is the total GPU KV cache size.
    return gpu_config.kv_cache_tensors[0].size
