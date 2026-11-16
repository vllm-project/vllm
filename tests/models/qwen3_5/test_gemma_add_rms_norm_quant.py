# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.5 Gemma add-RMSNorm + input quant vs the unfused production chain
(flashinfer gemma norm, then scaled_fp8_quant / scaled_fp4_quant), plus the
unquantized `gemma_add_rms_norm` (pre-SM80 fallback) vs a naive fp32
reference."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA required")

# Partial and multiple 128-row scale tiles, both sides of the num_warps
# switches (128 for NVFP4, 384 for FP8); 1 and 7 are decode sizes.
NUM_TOKENS = [1, 7, 128, 129, 384, 385]


def _fp8_codes(q: torch.Tensor) -> torch.Tensor:
    """Sign-magnitude e4m3 bytes -> ordered ints (+0 == -0)."""
    b = q.view(torch.uint8).int()
    return torch.where(b >= 128, -(b & 0x7F), b)


def _fp4_codes(packed: torch.Tensor) -> torch.Tensor:
    """Packed e2m1 nibbles (low first) -> ordered ints (+0 == -0)."""
    c = torch.stack([packed & 0xF, packed >> 4], dim=-1).flatten(-2).int()
    return torch.where(c >= 8, -(c & 7), c)


def _swizzled_sf_rows(sf: torch.Tensor, m: int, h: int) -> torch.Tensor:
    """(m, h // 16) view of a 128x4-swizzled NVFP4 scale tensor."""
    r = torch.arange(m, device=sf.device)[:, None]
    k = torch.arange(h // 16, device=sf.device)[None, :]
    num_k_tiles = (h // 16 + 3) // 4
    off = (
        ((r >> 7) * num_k_tiles + (k >> 2)) * 512
        + (r & 31) * 16
        + ((r >> 5) & 3) * 4
        + (k & 3)
    )
    return sf.view(torch.uint8).flatten()[off]


# 5120: a production hidden size; 2112: not a power of 2 (33 64-wide scale tiles).
@pytest.mark.parametrize("H", [5120, 2112])
@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.skipif(
    not current_platform.has_device_capability(100),
    reason="fused Gemma norm + FP8/NVFP4 quant targets SM100+",
)
@pytest.mark.parametrize("nvfp4", [False, True])
@pytest.mark.parametrize("has_residual", [True, False])
@torch.inference_mode()
def test_gemma_add_rms_norm_quant_matches_chain(
    H: int, num_tokens: int, nvfp4: bool, has_residual: bool
):
    from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm

    from vllm.models.qwen3_5.nvidia.ops.fused_norm_quant import (
        gemma_add_rms_norm_quant,
    )

    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    # Checkpoint-like Gemma weights: (1 + w) from ~0 (w near -1) to 3.5.
    w = (torch.rand(H, generator=g, device="cuda") * 3.5 - 1.0).bfloat16()
    w[:64] = -0.99609375
    x = torch.randn(num_tokens, H, generator=g, device="cuda").bfloat16()
    residual = (torch.randn(num_tokens, H, generator=g, device="cuda") * 4).bfloat16()
    residual[:, 7] *= 50  # outlier channel
    # FP8: static input scale. NVFP4: global scale (input_global_scale_inv).
    scale = torch.tensor([1 / 0.0013 if nvfp4 else 0.11], device="cuda")

    ref_res = residual.clone()
    if has_residual:
        ref_y = x.clone()
        gemma_fused_add_rmsnorm(ref_y, ref_res, w, 1e-6)
    else:
        ref_y = gemma_rmsnorm(x, w, 1e-6)
    if nvfp4:
        ref_q, ref_sf = ops.scaled_fp4_quant(
            ref_y, scale, True, backend="flashinfer-cutedsl"
        )
    else:
        ref_q, _ = ops.scaled_fp8_quant(ref_y, scale, group_shape=(-1, -1))

    x_in = x.clone()
    res = residual.clone()
    q, sf, y = gemma_add_rms_norm_quant(
        x_in, res if has_residual else None, w, 1e-6, scale, nvfp4, write_y=True
    )

    torch.testing.assert_close(x_in, x, rtol=0, atol=0)
    torch.testing.assert_close(res, ref_res, rtol=0, atol=0)
    # The fp32 sum of squares is reduced in a different order; rare 1-ulp
    # differences of the bf16 output are expected.
    assert (y != ref_y).float().mean().item() <= 1e-4
    torch.testing.assert_close(y, ref_y, rtol=2**-7, atol=0)

    if nvfp4:
        dist = (_fp4_codes(q) - _fp4_codes(ref_q.view(torch.uint8))).abs()
        sf_mismatch = _swizzled_sf_rows(sf, num_tokens, H) != _swizzled_sf_rows(
            ref_sf, num_tokens, H
        )
        assert sf_mismatch.float().mean().item() <= 1e-4
    else:
        assert sf is scale
        dist = (_fp8_codes(q) - _fp8_codes(ref_q)).abs()
    assert dist.max().item() <= 1
    assert (dist > 0).float().mean().item() <= 1e-4


# 2048: Qwen3.5-2B/35B-A3B hidden size; 2056: not a multiple of 16, to
# exercise the column mask (every production hidden size is).
@pytest.mark.parametrize("H", [2048, 2056])
# Decode sizes, and both sides of the num_warps switch (384).
@pytest.mark.parametrize("num_tokens", [1, 7, 384, 385])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("has_residual", [True, False])
@torch.inference_mode()
def test_gemma_add_rms_norm_correctness(
    H: int, num_tokens: int, dtype: torch.dtype, has_residual: bool
):
    """The unquantized kernel (pre-SM80 fallback) vs a naive fp32 reference."""
    from vllm.models.qwen3_5.nvidia.ops.fused_norm_quant import gemma_add_rms_norm

    g = torch.Generator(device="cuda").manual_seed(num_tokens)
    w = (torch.rand(H, generator=g, device="cuda") * 3.5 - 1.0).to(dtype)
    w[:64] = -0.99609375
    x = torch.randn(num_tokens, H, generator=g, device="cuda").to(dtype)
    residual = (torch.randn(num_tokens, H, generator=g, device="cuda") * 4).to(dtype)
    residual[:, 7] *= 50  # outlier channel

    summed = x.float() + residual.float() if has_residual else x.float()
    var = summed.pow(2).mean(dim=-1, keepdim=True)
    ref_y = (summed * torch.rsqrt(var + 1e-6) * (1.0 + w.float())).to(dtype)

    x_in = x.clone()
    res = residual.clone()
    y = gemma_add_rms_norm(x_in, res if has_residual else None, w, 1e-6)

    torch.testing.assert_close(x_in, x, rtol=0, atol=0)
    if has_residual:
        # The residual update is an fp32 add rounded once, bitwise-exact.
        torch.testing.assert_close(res, summed.to(dtype), rtol=0, atol=0)
    # House kernel-test tolerances; the kernel differs from the fp32
    # reference only by reduction-order rounding (1-ulp output flips).
    tol = {"atol": 1e-3, "rtol": 1e-3 if dtype == torch.float16 else 1.6e-2}
    torch.testing.assert_close(y, ref_y, **tol)
