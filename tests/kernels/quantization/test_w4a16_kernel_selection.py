#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for W4A16 kernel selection logic (ROCm).

Run `pytest tests/kernels/quantization/test_w4a16_kernel_selection.py`.
"""

import torch

from tests.utils import requires_platform
from vllm.model_executor.kernels.linear import (
    MPLinearLayerConfig,
    choose_mp_linear_kernel,
)
from vllm.scalar_type import scalar_types


@requires_platform("rocm")
def test_choose_mp_linear_kernel_picks_triton_w4a16_for_uint4b8():
    # int4 weights, 16-bit activations (CT W4A16 typical config).
    K, N = 1024, 256
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4b8,  # symmetric int4 (bias=8)
        act_type=torch.float16,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )

    kernel_type = choose_mp_linear_kernel(config)
    assert kernel_type.__name__ == "TritonW4A16LinearKernel"


@requires_platform("rocm")
def test_choose_mp_linear_kernel_picks_triton_w4a16_for_uint4_asymmetric():
    # Asymmetric int4 weights should also be supported (explicit zero points).
    K, N = 512, 512
    config = MPLinearLayerConfig(
        full_weight_shape=(K, N),
        partition_weight_shape=(K, N),
        weight_type=scalar_types.uint4,  # asymmetric int4 (explicit zeros)
        act_type=torch.bfloat16,
        group_size=64,
        zero_points=True,
        has_g_idx=False,
    )

    kernel_type = choose_mp_linear_kernel(config)
    assert kernel_type.__name__ == "TritonW4A16LinearKernel"
