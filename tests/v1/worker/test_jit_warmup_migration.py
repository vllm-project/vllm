# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Validate the registry reference kernel against runtime dispatch."""

import pytest

from vllm.platforms import current_platform

if not current_platform.is_cuda_alike():
    pytest.skip("NVIDIA dispatch tests require CUDA", allow_module_level=True)

from vllm.v1.worker.block_table import ComputeSlotMappingKernel


@pytest.mark.parametrize("kv_cache_block_size", [256, 64, 8, 4])
def test_compute_slot_mapping_warmup_matches_runtime_specializations(
    kv_cache_block_size: int,
) -> None:
    kernel = ComputeSlotMappingKernel()
    kwargs = dict(
        kv_cache_block_size=kv_cache_block_size,
        total_cp_world_size=1,
        total_cp_rank=0,
        cp_kv_cache_interleave_size=1,
        block_table_stride=32768,
    )
    expected = kernel.CompileKey(
        kv_cache_block_size=kv_cache_block_size,
        total_cp_world_size=1,
        total_cp_rank=0,
        cp_kv_cache_interleave_size=1,
        block_table_stride=16,
    )

    assert kernel.dispatch(**kwargs) == expected
    assert kernel.get_warmup_keys(**kwargs) == [expected]
