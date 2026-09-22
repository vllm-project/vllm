#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kernel-selection / gating tests for the ROCm RDNA3 W4A16 GPTQ kernel.

Verifies that ``choose_mp_linear_kernel`` resolves a supported W4A16 GPTQ
config to ``RDNA3W4A16LinearKernel`` on gfx1100 (it is registered ahead of
``TritonW4A16LinearKernel`` in the ROCm priority list), and that
``RDNA3W4A16LinearKernel.can_implement`` rejects the configs it does not
support so selection falls through to the next kernel.

Run `pytest tests/kernels/quantization/test_rdna3_w4a16_selection.py`.
"""

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("RDNA3 W4A16 kernel is ROCm-only", allow_module_level=True)

from vllm.model_executor.kernels.linear import (  # noqa: E402
    choose_mp_linear_kernel,
)
from vllm.model_executor.kernels.linear.mixed_precision.MPLinearKernel import (  # noqa: E402
    MPLinearLayerConfig,
)
from vllm.model_executor.kernels.linear.mixed_precision.rdna3_w4a16 import (  # noqa: E402
    RDNA3W4A16LinearKernel,
)
from vllm.platforms.rocm import on_gfx1100  # noqa: E402
from vllm.scalar_type import scalar_types  # noqa: E402

WEIGHT_TYPE = scalar_types.uint4b8  # symmetric int4, bias = 8

# The kernel is only selectable when running on gfx1100 with the custom op
# compiled in; otherwise can_implement rejects and selection falls through.
gfx1100_only = pytest.mark.skipif(
    not (
        on_gfx1100()
        and hasattr(torch.ops, "_rocm_C")
        and hasattr(torch.ops._rocm_C, "gptq_gemm_rdna3")
    ),
    reason="requires gfx1100 with the _rocm_C.gptq_gemm_rdna3 op built in",
)


@gfx1100_only
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_selection_prefers_rdna3(dtype):
    """A supported W4A16 GPTQ config resolves to the RDNA3 kernel on gfx1100."""
    config = MPLinearLayerConfig(
        full_weight_shape=(1024, 256),
        partition_weight_shape=(1024, 256),
        weight_type=WEIGHT_TYPE,
        act_type=dtype,
        group_size=128,
        zero_points=False,
        has_g_idx=False,
    )
    assert choose_mp_linear_kernel(config).__name__ == "RDNA3W4A16LinearKernel"


@gfx1100_only
@pytest.mark.parametrize(
    "weight_type,group_size,N,full_k,expected_ok",
    [
        (scalar_types.uint4b8, 128, 256, 1024, True),  # nominal: supported
        (scalar_types.uint4b8, -1, 256, 1024, False),  # channelwise unsupported
        (scalar_types.uint4b8, 128, 252, 1024, False),  # N not a multiple of 8
        (scalar_types.uint4b8, 96, 256, 1024, False),  # group does not divide K
        (scalar_types.uint8b128, 128, 256, 1024, False),  # wrong quant type
    ],
    ids=["ok", "channelwise", "bad_n", "group_ndiv_k", "wrong_qtype"],
)
def test_can_implement(weight_type, group_size, N, full_k, expected_ok):
    """can_implement gates on quant type, group size, and N divisibility."""
    config = MPLinearLayerConfig(
        full_weight_shape=(full_k, N),
        partition_weight_shape=(full_k, N),
        weight_type=weight_type,
        act_type=torch.float16,
        group_size=group_size,
        zero_points=False,
        has_g_idx=False,
    )
    ok, reason = RDNA3W4A16LinearKernel.can_implement(config)
    assert ok is expected_ok, reason
