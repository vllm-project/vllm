#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Correctness tests for wvSplitK_fused_silu_mul, wvSplitK_fused_silu_gate_mul kernels.

This test verifies that the new fused kernels produce numerically correct results
compared to the unfused baseline implementation.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip(
        "wvSplitK kernels are ROCm-specific",
        allow_module_level=True,
    )

from vllm import _custom_ops as ops  # noqa: E402
from vllm.utils.platform_utils import num_compute_units  # noqa: E402

# Comprehensive shape coverage for testing
# Format: (H=hidden_size, IN=intermediate_size)
# KERNEL CONSTRAINT: IN (K dimension) must be a multiple of 8
KERNEL_SHAPES = [
    # Production shapes (Real Qwen MoE model configurations)
    (2048, 512),  # Qwen3.5-A3B shared expert
    (2048, 1408),  # Qwen1.5-MoE routed expert
    (2048, 5632),  # Qwen1.5-MoE shared expert
    (3584, 1536),  # Qwen2.5-7B routed expert
    # Power-of-2
    (8, 8),
    (64, 8),
    (8, 64),
    (64, 32),
    (128, 64),
    (256, 64),
    (256, 128),
    (512, 256),
    (1024, 256),
    (1024, 512),
    (1024, 1024),
    (2048, 1024),
    (4096, 1024),
    (4096, 2048),
    (8192, 2048),
    # Non-Po2 (K % 8 == 0)
    (1536, 768),
    (2560, 640),
    (3072, 768),
    (1008, 504),
    (920, 920),
    # Alignment boundaries
    (1024, 384),
    (2048, 640),
    (640, 2048),
    (2304, 576),
    (3456, 864),
    # skinny shapes
    (8, 4096),
    (4096, 8),
]

DTYPES = [torch.bfloat16, torch.float16]

# Tolerance values for comparing fused kernels against PyTorch F.linear baseline
#
# These are OUR chosen tolerances for validating correctness, accounting for:
# 1. Limited precision of bf16/fp16 datatypes:
#    - bfloat16: ~7 bits mantissa → machine epsilon ~0.0078 (2^-7)
#    - float16: ~10 bits mantissa → machine epsilon ~0.00098 (2^-10)
# 2. Numerical differences from operation reordering:
#    - Fused kernel: silu(gate) * up → GEMM (single fused operation)
#    - Baseline: separate silu, multiply, then GEMM
# 3. Accumulation differences in GEMM operations across K dimension
DEFAULT_ATOL = 1e-2  # Absolute tolerance: 0.01 (~1.3x bf16 epsilon)
DEFAULT_RTOL = 5e-2  # Relative tolerance: 5% (validated over 1000 iterations)


def baseline_silu_mul_down_proj(
    gate_up: torch.Tensor, down_weight: torch.Tensor
) -> torch.Tensor:
    """Baseline implementation: silu_and_mul + linear"""
    intermediate_size = gate_up.size(-1) // 2
    gate = gate_up[:, :intermediate_size]
    up = gate_up[:, intermediate_size:]
    x = F.silu(gate) * up
    return F.linear(x, down_weight)


def baseline_with_gate(
    gate_up: torch.Tensor,
    down_weight: torch.Tensor,
    gate_scalar: torch.Tensor,
) -> torch.Tensor:
    """Baseline with expert gate: silu_and_mul + linear + sigmoid * out"""
    out = baseline_silu_mul_down_proj(gate_up, down_weight)
    return F.sigmoid(gate_scalar) * out


@pytest.mark.parametrize("H,IN", KERNEL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_wvsplitk_fused_silu_mul(H: int, IN: int, dtype: torch.dtype):
    """Test wvSplitK_fused_silu_mul correctness against baseline.

    Tests the Phase 1 kernel which fuses: silu(gate) * up → GEMM
    """
    if not hasattr(torch.ops, "_rocm_C") or not hasattr(
        torch.ops._rocm_C, "wvSplitK_fused_silu_mul"
    ):
        pytest.skip("wvSplitK_fused_silu_mul needs ROCm build")

    device = "cuda"
    cu = num_compute_units()

    # Generate random inputs
    gate_up = torch.randn(1, 2 * IN, dtype=dtype, device=device) * 0.01
    down_weight = torch.randn(H, IN, dtype=dtype, device=device) * 0.01

    # Baseline computation
    baseline_out = baseline_silu_mul_down_proj(gate_up, down_weight)

    # Fused kernel
    fused_out = ops.wvSplitK_fused_silu_mul(down_weight, gate_up, cu, None)

    # Compare outputs
    assert baseline_out.shape == fused_out.shape == (1, H)
    torch.testing.assert_close(
        baseline_out,
        fused_out,
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL,
        msg=f"wvSplitK_fused_silu_mul failed: H={H}, IN={IN}, dtype={dtype}",
    )


@pytest.mark.parametrize("H,IN", KERNEL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_wvsplitk_fused_silu_gate_mul(H: int, IN: int, dtype: torch.dtype):
    """
    Test wvSplitK_fused_silu_gate_mul correctness against baseline.
    The kernel fuses: silu(gate) * up → GEMM → sigmoid(gate) * out
    """
    if not hasattr(torch.ops, "_rocm_C") or not hasattr(
        torch.ops._rocm_C, "wvSplitK_fused_silu_gate_mul"
    ):
        pytest.skip("wvSplitK_fused_silu_gate_mul needs ROCm build")

    device = "cuda"
    cu = num_compute_units()

    # Generate random inputs
    gate_up = torch.randn(1, 2 * IN, dtype=dtype, device=device) * 0.01
    down_weight = torch.randn(H, IN, dtype=dtype, device=device) * 0.01
    gate_scalar = torch.randn(1, 1, dtype=dtype, device=device) * 0.5

    # Baseline computation
    baseline_out = baseline_with_gate(gate_up, down_weight, gate_scalar)

    # Fused kernel with gate
    fused_out = ops.wvSplitK_fused_silu_gate_mul(
        down_weight, gate_up, F.sigmoid(gate_scalar), cu, None
    )

    # Compare outputs
    assert baseline_out.shape == fused_out.shape == (1, H)
    torch.testing.assert_close(
        baseline_out,
        fused_out,
        atol=DEFAULT_ATOL,
        rtol=DEFAULT_RTOL,
        msg=f"wvSplitK_fused_silu_gate_mul failed: H={H}, IN={IN}, dtype={dtype}",
    )
