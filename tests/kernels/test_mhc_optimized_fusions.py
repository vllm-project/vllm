# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for optimized MHC kernel fusions.

These tests verify correctness by comparing optimized fused kernels
against the original separate kernel calls.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.utils.import_utils import has_tilelang

if not has_tilelang() or not current_platform.is_cuda_alike():
    pytest.skip("TileLang required for MHC tests", allow_module_level=True)

from vllm.model_executor.kernels.mhc.tilelang import (
    hc_head_fused_kernel_tilelang,
    mhc_post_tilelang,
)
from vllm.model_executor.kernels.mhc.optimized_wrappers import (
    mhc_post_hc_head_fused,
    mhc_post_hc_head_norm_fused,
    mhc_post_mean_fused,
)


def create_test_inputs(num_tokens, hc_mult, hidden_size, device="cuda", dtype=torch.bfloat16):
    """Create test inputs for MHC operations."""
    x = torch.randn(num_tokens, hidden_size, device=device, dtype=dtype)
    residual = torch.randn(num_tokens, hc_mult, hidden_size, device=device, dtype=dtype)
    post_mix = torch.randn(num_tokens, hc_mult, device=device, dtype=torch.float32)
    comb_mix = torch.randn(num_tokens, hc_mult, hc_mult, device=device, dtype=torch.float32)

    # Normalize mixing weights to avoid numerical overflow
    post_mix = torch.sigmoid(post_mix) * 2.0
    comb_mix = torch.softmax(comb_mix, dim=-1)

    return x, residual, post_mix, comb_mix


def create_hc_head_params(hc_mult, hidden_size, device="cuda"):
    """Create HC head parameters."""
    hc_dim = hc_mult * hidden_size
    fn = torch.randn(hc_mult, hc_dim, device=device, dtype=torch.float32) * 0.01
    scale = torch.ones(1, device=device, dtype=torch.float32)
    base = torch.randn(hc_mult, device=device, dtype=torch.float32) * 0.1
    return fn, scale, base


@pytest.mark.parametrize("num_tokens", [1, 8, 64, 256])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_size", [512, 1024])
def test_mhc_post_hc_head_norm_fused_mtp_stash(num_tokens, hc_mult, hidden_size):
    """The MTP variant must stash the pre-hc_head residual unchanged.

    The 3-way fused kernel with MTP runs when ``mtp_buffer`` is provided. The
    buffer contract from the model is: ``mtp_buffer[:num_tokens]`` holds
    ``mhc_post(...).flatten(1)`` in the global token order. This test feeds a
    buffer that is pre-filled with a sentinel and checks the kernel overwrites
    the leading rows with the exact pre-hc_head residual, leaving the tail
    untouched.
    """
    device = "cuda"
    torch.manual_seed(42)

    x, residual, post_mix, comb_mix = create_test_inputs(
        num_tokens, hc_mult, hidden_size, device
    )
    fn, scale, base = create_hc_head_params(hc_mult, hidden_size, device)
    norm_weight = torch.ones(hidden_size, device=device, dtype=torch.bfloat16)
    norm_eps = 1e-5

    # Reference: the unfused chain produces the stashed residual.
    residual_out = mhc_post_tilelang(
        x, residual, post_mix.unsqueeze(-1), comb_mix
    )
    expected_stash = residual_out.flatten(1)

    # Pre-fill the buffer so a missed write is detectable.
    buffer = torch.full(
        (2 * num_tokens, hc_mult * hidden_size),
        7.0,
        device=device,
        dtype=torch.bfloat16,
    )
    output_fused = mhc_post_hc_head_norm_fused(
        x,
        residual,
        post_mix.unsqueeze(-1),
        comb_mix,
        fn,
        scale,
        base,
        norm_weight,
        1e-6,
        1e-3,
        norm_eps,
        mtp_buffer=buffer,
    )

    # Stash must match the unfused pre-hc_head, and only the first rows.
    torch.testing.assert_close(
        buffer[:num_tokens],
        expected_stash,
        rtol=1e-2,
        atol=1e-2,
        msg="MTP stash mismatch for {num_tokens=}, {hidden_size=}",
    )
    assert torch.all(buffer[num_tokens:] == 7.0), "MTP stash overflowed the buffer"

    # The merged output must also match the unfused hc_head + RMSNorm chain.
    hc_out = hc_head_fused_kernel_tilelang(
        residual_out, fn, scale, base, 1e-6, 1e-3
    )
    variance = hc_out.float().pow(2).mean(-1, keepdim=True)
    expected_out = (
        hc_out * torch.rsqrt(variance + norm_eps) * norm_weight
    ).bfloat16()
    torch.testing.assert_close(
        output_fused,
        expected_out,
        rtol=1e-2,
        atol=1e-2,
        msg="Fused norm output mismatch with MTP stash for {num_tokens=}",
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 64, 256])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_size", [512, 1024])
def test_mhc_post_hc_head_norm_fused_cmp_mtp_vanilla(num_tokens, hc_mult, hidden_size):
    """The MTP and non-MTP variants must produce identical outputs.

    Besides exercising both kernel paths, this protects the model's fused
    branch against accidental divergence between the two variants.
    """
    device = "cuda"
    torch.manual_seed(42)

    x, residual, post_mix, comb_mix = create_test_inputs(
        num_tokens, hc_mult, hidden_size, device
    )
    fn, scale, base = create_hc_head_params(hc_mult, hidden_size, device)
    norm_weight = torch.ones(hidden_size, device=device, dtype=torch.bfloat16)

    output_vanilla = mhc_post_hc_head_norm_fused(
        x,
        residual,
        post_mix.unsqueeze(-1),
        comb_mix,
        fn,
        scale,
        base,
        norm_weight,
        1e-6,
        1e-3,
        1e-5,
    )
    buffer = torch.empty(
        num_tokens, hc_mult * hidden_size, device=device, dtype=torch.bfloat16
    )
    output_mtp = mhc_post_hc_head_norm_fused(
        x,
        residual,
        post_mix.unsqueeze(-1),
        comb_mix,
        fn,
        scale,
        base,
        norm_weight,
        1e-6,
        1e-3,
        1e-5,
        mtp_buffer=buffer,
    )
    torch.testing.assert_close(
        output_mtp,
        output_vanilla,
        rtol=1e-2,
        atol=1e-2,
        msg="MTP and vanilla fused variants diverge for {num_tokens=}",
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 64, 256])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_size", [512, 1024, 2048])
def test_mhc_post_hc_head_fused_correctness(num_tokens, hc_mult, hidden_size):
    """Test that fused mhc_post + hc_head matches separate calls."""
    device = "cuda"
    torch.manual_seed(42)

    # Create inputs
    x, residual, post_mix, comb_mix = create_test_inputs(num_tokens, hc_mult, hidden_size, device)
    fn, scale, base = create_hc_head_params(hc_mult, hidden_size, device)

    rms_eps = 1e-6
    hc_eps = 1e-3

    # Original separate operations
    residual_out_orig = mhc_post_tilelang(
        x, residual, post_mix.unsqueeze(-1), comb_mix
    )
    output_orig = hc_head_fused_kernel_tilelang(
        residual_out_orig, fn, scale, base, rms_eps, hc_eps
    )

    # Optimized fused operation
    output_fused = mhc_post_hc_head_fused(
        x, residual, post_mix.unsqueeze(-1), comb_mix,
        fn, scale, base, rms_eps, hc_eps
    )

    # Check results match
    torch.testing.assert_close(
        output_fused, output_orig,
        rtol=1e-2, atol=1e-2,
        msg=f"Fused kernel output mismatch for {num_tokens=}, {hc_mult=}, {hidden_size=}"
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 64])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_size", [512, 1024])
def test_mhc_post_hc_head_norm_fused_correctness(num_tokens, hc_mult, hidden_size):
    """Test that fused mhc_post + hc_head + norm matches separate calls."""
    device = "cuda"
    torch.manual_seed(42)

    # Create inputs
    x, residual, post_mix, comb_mix = create_test_inputs(num_tokens, hc_mult, hidden_size, device)
    fn, scale, base = create_hc_head_params(hc_mult, hidden_size, device)
    norm_weight = torch.ones(hidden_size, device=device, dtype=torch.bfloat16)

    rms_eps = 1e-6
    hc_eps = 1e-3
    norm_eps = 1e-5

    # Original separate operations
    residual_out_orig = mhc_post_tilelang(
        x, residual, post_mix.unsqueeze(-1), comb_mix
    )
    hc_out_orig = hc_head_fused_kernel_tilelang(
        residual_out_orig, fn, scale, base, rms_eps, hc_eps
    )
    # RMSNorm
    variance = hc_out_orig.float().pow(2).mean(-1, keepdim=True)
    output_orig = (hc_out_orig * torch.rsqrt(variance + norm_eps) * norm_weight).bfloat16()

    # Optimized fused operation
    output_fused = mhc_post_hc_head_norm_fused(
        x, residual, post_mix.unsqueeze(-1), comb_mix,
        fn, scale, base, norm_weight, rms_eps, hc_eps, norm_eps
    )

    # Check results match
    torch.testing.assert_close(
        output_fused, output_orig,
        rtol=1e-2, atol=1e-2,
        msg=f"Fused kernel with norm output mismatch for {num_tokens=}, {hc_mult=}, {hidden_size=}"
    )


@pytest.mark.parametrize("num_tokens", [1, 8, 64, 256])
@pytest.mark.parametrize("hc_mult", [4])
@pytest.mark.parametrize("hidden_size", [512, 1024])
def test_mhc_post_mean_fused_correctness(num_tokens, hc_mult, hidden_size):
    """Test that fused mhc_post + mean matches separate calls."""
    device = "cuda"
    torch.manual_seed(42)

    # Create inputs
    x, residual, post_mix, comb_mix = create_test_inputs(num_tokens, hc_mult, hidden_size, device)

    # Original separate operations
    residual_out_orig = mhc_post_tilelang(
        x, residual, post_mix.unsqueeze(-1), comb_mix
    )
    mean_out_orig = residual_out_orig.mean(dim=1)

    # Optimized fused operation
    residual_out_fused, mean_out_fused = mhc_post_mean_fused(
        x, residual, post_mix.unsqueeze(-1), comb_mix
    )

    # Check both outputs match
    torch.testing.assert_close(
        residual_out_fused, residual_out_orig,
        rtol=1e-3, atol=1e-3,
        msg=f"Fused kernel residual output mismatch for {num_tokens=}, {hc_mult=}, {hidden_size=}"
    )

    torch.testing.assert_close(
        mean_out_fused, mean_out_orig,
        rtol=2e-2, atol=2e-2,
        msg=f"Fused kernel mean output mismatch for {num_tokens=}, {hc_mult=}, {hidden_size=}"
    )


if __name__ == "__main__":
    # Quick smoke test
    print("Running MHC optimized fusion tests...")
    test_mhc_post_hc_head_fused_correctness(64, 4, 1024)
    print("✓ mhc_post_hc_head_fused correctness test passed")

    test_mhc_post_hc_head_norm_fused_correctness(64, 4, 1024)
    print("✓ mhc_post_hc_head_norm_fused correctness test passed")

    test_mhc_post_mean_fused_correctness(64, 4, 1024)
    print("✓ mhc_post_mean_fused correctness test passed")

    print("\nAll tests passed!")
