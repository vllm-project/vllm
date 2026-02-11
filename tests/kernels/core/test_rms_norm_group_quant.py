# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Tests for group quantization support in fused rms_norm + static fp8 quant kernels.
Tests both backward compatibility (group_size=0) and new group quant paths.
"""

import pytest
import torch

from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()
DTYPES = [torch.bfloat16, torch.float]
NUM_TOKENS = [1, 7, 256]
HIDDEN_SIZES = [128, 512, 1024, 5120]
GROUP_SIZES = [64, 128]
SEEDS = [0]


def set_random_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, epsilon: float) -> torch.Tensor:
    """Reference RMS norm in float32."""
    orig_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + epsilon)
    return (x * weight.float()).to(orig_dtype)


def ref_rms_norm_group_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    epsilon: float,
    group_size: int,
) -> torch.Tensor:
    """Reference: RMS norm -> static fp8 quant with per-group scale."""
    normed = ref_rms_norm(x, weight, epsilon)
    # Reshape for group quantization
    hidden_size = normed.shape[-1]
    num_groups = hidden_size // group_size
    flat = normed.reshape(-1, num_groups, group_size).float()
    # Apply per-group scale: scale shape is [num_groups]
    scale_inv = (1.0 / scale).reshape(1, num_groups, 1)
    scaled = flat * scale_inv
    # Clamp to fp8 range and cast
    finfo = torch.finfo(FP8_DTYPE)
    scaled = scaled.clamp(finfo.min, finfo.max)
    result = scaled.reshape(normed.shape).to(FP8_DTYPE)
    return result


# =============================================================
# Test 1: Backward compatibility - group_size=0 matches original
# =============================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_rms_norm_group_quant_backward_compat(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
):
    """group_size=0 should produce identical results to the original kernel."""
    set_random_seed(seed)
    torch.set_default_device("cuda")

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    scale = torch.tensor([0.1], dtype=torch.float32)

    # Original call (without group_size, uses default=0)
    out_original = torch.empty(num_tokens, hidden_size, dtype=FP8_DTYPE)
    torch.ops._C.rms_norm_static_fp8_quant(out_original, x, weight, scale, 1e-6)

    # Explicit group_size=0
    out_group0 = torch.empty(num_tokens, hidden_size, dtype=FP8_DTYPE)
    torch.ops._C.rms_norm_static_fp8_quant(out_group0, x, weight, scale, 1e-6, 0)

    torch.testing.assert_close(
        out_original.to(torch.float32),
        out_group0.to(torch.float32),
        atol=0.0,
        rtol=0.0,  # Must be bitwise identical
    )


# =============================================================
# Test 2: Group quant correctness vs reference
# =============================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_rms_norm_group_quant_correctness(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    dtype: torch.dtype,
    seed: int,
):
    """Group quant kernel output should match reference implementation."""
    if hidden_size % group_size != 0:
        pytest.skip(
            f"hidden_size {hidden_size} not divisible by group_size {group_size}"
        )

    set_random_seed(seed)
    torch.set_default_device("cuda")

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)

    num_groups = hidden_size // group_size
    scale = torch.rand(num_groups, dtype=torch.float32) * 0.2 + 0.01  # positive scales

    # Kernel output
    out_kernel = torch.empty(num_tokens, hidden_size, dtype=FP8_DTYPE)
    torch.ops._C.rms_norm_static_fp8_quant(
        out_kernel, x, weight, scale, 1e-6, group_size
    )

    # Reference output
    out_ref = ref_rms_norm_group_quant(x, weight, scale, 1e-6, group_size)

    torch.testing.assert_close(
        out_kernel.to(torch.float32),
        out_ref.to(torch.float32),
        atol=48.0,
        rtol=0.2,  # fp8 rounding at high magnitudes
    )


# =============================================================
# Test 3: Fused add + rms_norm + group quant backward compat
# =============================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fused_add_rms_norm_group_quant_backward_compat(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    seed: int,
):
    """Fused add variant: group_size=0 should match original."""
    set_random_seed(seed)
    torch.set_default_device("cuda")

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype)
    scale = torch.tensor([0.1], dtype=torch.float32)

    # Original
    x1 = x.clone()
    res1 = residual.clone()
    out1 = torch.empty(num_tokens, hidden_size, dtype=FP8_DTYPE)
    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
        out1, x1, res1, weight, scale, 1e-6
    )

    # group_size=0
    x2 = x.clone()
    res2 = residual.clone()
    out2 = torch.empty(num_tokens, hidden_size, dtype=FP8_DTYPE)
    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
        out2, x2, res2, weight, scale, 1e-6, 0
    )

    torch.testing.assert_close(
        out1.to(torch.float32),
        out2.to(torch.float32),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(res1, res2, atol=0.0, rtol=0.0)


# =============================================================
# Test 4: Fused add + rms_norm + group quant correctness
# =============================================================
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("hidden_size", HIDDEN_SIZES)
@pytest.mark.parametrize("group_size", GROUP_SIZES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
def test_fused_add_rms_norm_group_quant_correctness(
    num_tokens: int,
    hidden_size: int,
    group_size: int,
    dtype: torch.dtype,
    seed: int,
):
    """Fused add variant: group quant should match reference."""
    if hidden_size % group_size != 0:
        pytest.skip(
            f"hidden_size {hidden_size} not divisible by group_size {group_size}"
        )

    set_random_seed(seed)
    torch.set_default_device("cuda")

    weight = torch.empty(hidden_size, dtype=dtype).normal_(mean=1.0, std=0.1)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype)
    residual = torch.randn(num_tokens, hidden_size, dtype=dtype)

    num_groups = hidden_size // group_size
    scale = torch.rand(num_groups, dtype=torch.float32) * 0.2 + 0.01

    # Kernel
    x_k = x.clone()
    res_k = residual.clone()
    out_kernel = torch.empty(num_tokens, hidden_size, dtype=FP8_DTYPE)
    torch.ops._C.fused_add_rms_norm_static_fp8_quant(
        out_kernel, x_k, res_k, weight, scale, 1e-6, group_size
    )

    # Reference: add residual -> rms_norm -> group quant
    res_ref = residual + x
    out_ref = ref_rms_norm_group_quant(res_ref, weight, scale, 1e-6, group_size)

    torch.testing.assert_close(
        out_kernel.to(torch.float32),
        out_ref.to(torch.float32),
        atol=48.0,
        rtol=0.2,
    )
    # Residual should match x + original residual
    torch.testing.assert_close(res_k, res_ref, atol=1e-2, rtol=1e-2)
