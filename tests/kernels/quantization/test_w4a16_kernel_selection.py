#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for W4A16 kernel selection logic (ROCm, CUDA sm_120).

Run `pytest tests/kernels/quantization/test_w4a16_kernel_selection.py`.
"""

import pytest
import torch

from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types

if current_platform.is_rocm():
    from vllm.platforms.rocm import on_gfx1x, on_gfx1100
else:
    on_gfx1100 = on_gfx1x = lambda: False  # noqa: E731

# Group sizes the HIP skinny path of RDNAHybridW4A16LinearKernel instantiates.
_HYBRID_GROUP_SIZES = (32, 64, 128)


def _expected_rocm_kernel(weight_type, group_size: int) -> str:
    """Mirror the ROCm priority order in `_POSSIBLE_KERNELS[ROCM]`:
    RDNA3 (gfx1100, symmetric uint4b8) -> Hybrid (gfx11/gfx12) -> Triton.
    """
    if on_gfx1100() and weight_type == scalar_types.uint4b8:
        return "RDNA3W4A16LinearKernel"
    if on_gfx1x() and group_size in _HYBRID_GROUP_SIZES:
        return "RDNAHybridW4A16LinearKernel"
    return "TritonW4A16LinearKernel"


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_choose_mp_linear_kernel_uint4b8():
    # int4 weights, 16-bit activations (CT W4A16 typical config).
    K, N = 1024, 256
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,  # symmetric int4 (bias=8)
        act_type=torch.float16,
        group_size=128,
        zero_points=False,
    )

    kernel_type = choose_mp_linear_kernel(config)
    assert kernel_type.__name__ == _expected_rocm_kernel(scalar_types.uint4b8, 128)


@pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only")
def test_choose_mp_linear_kernel_uint4_asymmetric():
    # Asymmetric int4 weights (explicit zero points).
    K, N = 512, 512
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,  # asymmetric int4 (explicit zeros)
        act_type=torch.bfloat16,
        group_size=64,
        zero_points=True,
    )

    kernel_type = choose_mp_linear_kernel(config)
    assert kernel_type.__name__ == _expected_rocm_kernel(scalar_types.uint4, 64)


def _cuda_config(act_type, k: int, group_size: int) -> MPLinearLayerConfig:
    return MPLinearLayerConfig(
        full_weight_shape=(k, 4096),
        partition_weight_shape=(k, 4096),
        weight_type=scalar_types.uint4b8,
        act_type=act_type,
        group_size=group_size,
        zero_points=False,
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_choose_mp_linear_kernel_sm120_fp16_exllama(group_size: int):
    # On sm_120, fp16 W4A16 with K <= 2048 dispatches to Exllama: measured
    # faster than Marlin at every M in [1, 512] on RTX 5060 Laptop (#66232).
    config = _cuda_config(torch.float16, 1024, group_size)
    kernel_type = choose_mp_linear_kernel(config, compute_capability=120)
    assert kernel_type.__name__ == "ExllamaLinearKernel"


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
def test_choose_mp_linear_kernel_sm120_reorder_scope():
    # The reorder is scoped: K above the measured window, bf16, and other
    # capabilities all keep the Marlin-first order.
    cases = [
        (_cuda_config(torch.float16, 4096, 32), 120),
        (_cuda_config(torch.bfloat16, 1024, 32), 120),
        (_cuda_config(torch.float16, 1024, 32), 89),
    ]
    for config, cc in cases:
        kernel_type = choose_mp_linear_kernel(config, compute_capability=cc)
        assert kernel_type.__name__ == "MarlinLinearKernel"
