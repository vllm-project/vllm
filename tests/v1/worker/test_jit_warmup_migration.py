# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the registry reference kernel against runtime dispatch."""

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

from vllm.v1.worker.block_table import ComputeSlotMappingKernel


@pytest.mark.parametrize(
    ("kv_cache_block_size", "blocks_per_kv_block", "block_size", "block_size_rep"),
    [
        (256, 1, 256, 16),
        (256, 4, 64, 16),
        (64, 1, 64, 16),
        (8, 1, 8, 2),
        (4, 1, 4, 2),
    ],
)
def test_compute_slot_mapping_warmup_matches_runtime_specializations(
    kv_cache_block_size: int,
    blocks_per_kv_block: int,
    block_size: int,
    block_size_rep: int,
) -> None:
    kernel = ComputeSlotMappingKernel()
    kwargs = dict(
        kv_cache_block_size=kv_cache_block_size,
        blocks_per_kv_block=blocks_per_kv_block,
        total_cp_world_size=1,
        total_cp_rank=0,
        cp_kv_cache_interleave_size=1,
        block_table_stride=32768,
        block_size=block_size,
    )
    expected = kernel.CompileKey(
        kv_cache_block_size=kv_cache_block_size,
        blocks_per_kv_block=blocks_per_kv_block,
        total_cp_world_size=1,
        total_cp_rank=0,
        cp_kv_cache_interleave_size=1,
        block_table_stride=16,
        block_size=block_size_rep,
    )

    assert kernel.dispatch(**kwargs) == expected
    assert kernel.get_warmup_keys(**kwargs) == [expected]
