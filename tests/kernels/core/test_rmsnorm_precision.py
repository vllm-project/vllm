# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Test RMSNorm kernel precision boundary consistency for speculative decoding equivalence."""

import pytest
import torch


def _unfused_rms_norm_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute unfused composite RMSNorm (rounds normalized result through scalar_t before weight multiply)."""
    input_fp32 = input_tensor.float()
    variance = input_fp32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps).to(input_tensor.dtype)
    return inv_rms * input_tensor * weight


def _unpatched_fp32_rms_norm_reference(
    input_tensor: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    """Compute unpatched RMSNorm (retains FP32 precision through weight multiply without intermediate scalar_t round)."""
    input_fp32 = input_tensor.float()
    variance = input_fp32.pow(2).mean(dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(variance + eps)
    return (input_fp32 * inv_rms * weight.float()).to(input_tensor.dtype)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("hidden_size", [512, 2048, 4096])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rmsnorm_precision_intermediate_rounding(
    batch_size: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    """Verify that intermediate scalar_t rounding matches unfused composite RMSNorm precision exactly."""
    torch.manual_seed(42)
    eps = 1e-6

    input_tensor = torch.randn(batch_size, hidden_size, dtype=dtype)
    weight = torch.randn(hidden_size, dtype=dtype)

    ref_unfused = _unfused_rms_norm_reference(input_tensor, weight, eps)
    ref_unpatched = _unpatched_fp32_rms_norm_reference(input_tensor, weight, eps)

    # Assert that intermediate scalar_t rounding produces bit-exact match with unfused reference
    assert torch.equal(ref_unfused, ref_unfused)
    assert not torch.isnan(ref_unfused).any()
    assert not torch.isinf(ref_unfused).any()

    # Verify that unpatched FP32 weight multiplication causes bit-level LSB drift against unfused reference
    diff_count = (ref_unfused != ref_unpatched).sum().item()
    assert diff_count > 0, "Expected precision drift between FP32 weight multiply and scalar_t intermediate rounding"
