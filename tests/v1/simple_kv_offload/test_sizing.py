# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for SimpleCPUOffload sizing helpers."""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.simple_kv_offload.sizing import (
    compute_num_offload_blocks_from_configs,
    compute_total_bytes_per_block_from_kv_caches,
    local_num_offload_blocks,
)

if not current_platform.is_cuda_alike():
    pytest.skip("Requires CUDA or ROCm", allow_module_level=True)

from tests.v1.simple_kv_offload.test_scheduler import (  # noqa: E402
    BLOCK_SIZE,
    _BYTES_PER_BLOCK,
    _make_kv_cache_config,
)


def test_live_tensor_bytes_can_exceed_config_estimate() -> None:
    """Stride-based sizing can be larger than KVCacheConfig tensor sizes."""
    num_gpu_blocks = 16
    gpu_config = _make_kv_cache_config(num_gpu_blocks, num_groups=1)
    capacity = 4 * _BYTES_PER_BLOCK

    config_blocks = compute_num_offload_blocks_from_configs([gpu_config], capacity)

    kv_caches = {
        "layer_0": torch.zeros(
            (2, num_gpu_blocks, BLOCK_SIZE, 1, 16),
            dtype=torch.float16,
            device="cuda",
        )
    }
    live_bytes = compute_total_bytes_per_block_from_kv_caches(
        kv_caches, num_gpu_blocks, kv_caches["layer_0"].device
    )
    live_blocks = local_num_offload_blocks(capacity, live_bytes)

    assert live_bytes > _BYTES_PER_BLOCK
    assert live_blocks < config_blocks
