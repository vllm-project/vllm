# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared (platform-neutral) MiniMax-M3 Gemma RMSNorm kernels.

Covers three things:
  1. ``common/ops/gemma_rmsnorm`` produces numerically correct output vs a
     pure-PyTorch fp32 reference (plain norm + fused-add-residual).
  2. ``nvidia/model.MiniMAXGemmaRMSNorm`` dispatches to the common Triton path
     on non-CUDA tensors (regression for issue #51200: was a hard crash).
  3. The AMD ``amd/ops/gemma_rmsnorm`` shim still exports the same symbols
     (import-level regression guard).

These tests run on CPU via the Triton CPU backend — no GPU required.
"""

import pytest
import torch

from vllm.models.minimax_m3.common.ops.gemma_rmsnorm import (
    gemma_fused_add_rmsnorm,
    gemma_rmsnorm,
)

EPS = 1e-6


# ---------------------------------------------------------------------------
# PyTorch fp32 reference
# ---------------------------------------------------------------------------
def _ref_gemma_rmsnorm(
    x: torch.Tensor,
    w: torch.Tensor,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    orig_dtype = x.dtype
    xf = x.float()
    res_out = None
    if residual is not None:
        xf = xf + residual.float()
        res_out = xf.to(orig_dtype)
    xf = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    xf = xf * (1.0 + w.float())
    out = xf.to(orig_dtype)
    return out if residual is None else (out, res_out)


def _relerr(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / (b.float().norm() + 1e-8)).item()


# ---------------------------------------------------------------------------
# common/ops/gemma_rmsnorm — correctness
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("shape", [(1, 128), (37, 512), (8, 1024)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("seed", [0, 42])
@torch.inference_mode()
def test_common_gemma_rmsnorm(shape, dtype, seed):
    torch.manual_seed(seed)
    x = torch.randn(*shape, dtype=dtype)
    w = torch.randn(shape[-1], dtype=dtype) * 0.1
    got = gemma_rmsnorm(x, w, EPS)
    ref = _ref_gemma_rmsnorm(x, w, EPS)
    assert got.shape == x.shape
    assert got.dtype == dtype
    assert _relerr(got, ref) < 5e-3


@pytest.mark.parametrize("shape", [(1, 128), (16, 512)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@torch.inference_mode()
def test_common_gemma_fused_add_rmsnorm(shape, dtype):
    torch.manual_seed(0)
    x = torch.randn(*shape, dtype=dtype)
    res = torch.randn(*shape, dtype=dtype)
    w = torch.randn(shape[-1], dtype=dtype) * 0.1
    got_out, got_res = gemma_fused_add_rmsnorm(x, res, w, EPS)
    ref_out, ref_res = _ref_gemma_rmsnorm(x, w, EPS, residual=res)
    assert got_out.shape == x.shape
    assert _relerr(got_out, ref_out) < 5e-3
    # residual_out is the pre-norm sum — must be bit-exact
    assert torch.equal(got_res, ref_res)


@torch.inference_mode()
def test_common_gemma_rmsnorm_strided():
    """Non-contiguous input (qkv split slice) is handled correctly."""
    torch.manual_seed(0)
    T, H, D = 5, 8, 128
    qkv = torch.randn(T, H * D * 2, dtype=torch.bfloat16)
    q = qkv[..., : H * D].view(T, H, D)
    assert not q.is_contiguous()
    w = torch.randn(D, dtype=torch.bfloat16) * 0.1
    got = gemma_rmsnorm(q, w, EPS)
    ref = _ref_gemma_rmsnorm(q, w, EPS)
    assert got.shape == q.shape
    assert _relerr(got, ref) < 5e-3


# ---------------------------------------------------------------------------
# nvidia/model.MiniMAXGemmaRMSNorm — non-CUDA fallback (issue #51200)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def test_nvidia_rmsnorm_cpu_fallback_plain():
    """MiniMAXGemmaRMSNorm must not crash on a CPU tensor (was CUDA-only)."""
    from vllm.models.minimax_m3.nvidia.model import MiniMAXGemmaRMSNorm

    norm = MiniMAXGemmaRMSNorm(hidden_size=128, eps=EPS)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    out = norm(x)
    ref = _ref_gemma_rmsnorm(x, norm.weight.data, EPS)
    assert out.shape == x.shape
    assert _relerr(out, ref) < 5e-3


@torch.inference_mode()
def test_nvidia_rmsnorm_cpu_fallback_fused():
    """MiniMAXGemmaRMSNorm fused-residual path must not crash on CPU."""
    from vllm.models.minimax_m3.nvidia.model import MiniMAXGemmaRMSNorm

    norm = MiniMAXGemmaRMSNorm(hidden_size=256, eps=EPS)
    x = torch.randn(8, 256, dtype=torch.bfloat16)
    res = torch.randn(8, 256, dtype=torch.bfloat16)
    out, res_out = norm(x, residual=res)
    ref_out, ref_res = _ref_gemma_rmsnorm(x, norm.weight.data, EPS, residual=res)
    assert out.shape == x.shape
    assert _relerr(out, ref_out) < 5e-3
    assert torch.equal(res_out, ref_res)


# ---------------------------------------------------------------------------
# amd shim — import-level regression guard
# ---------------------------------------------------------------------------
def test_amd_shim_exports_correct_symbols():
    """amd/ops/gemma_rmsnorm must still export gemma_rmsnorm and
    gemma_fused_add_rmsnorm after being refactored into a re-export shim."""
    from vllm.models.minimax_m3.amd.ops import gemma_fused_add_rmsnorm as amd_fused
    from vllm.models.minimax_m3.amd.ops import gemma_rmsnorm as amd_norm

    assert callable(amd_norm)
    assert callable(amd_fused)
    # Must be the same objects as the common implementations.
    assert amd_norm is gemma_rmsnorm
    assert amd_fused is gemma_fused_add_rmsnorm
