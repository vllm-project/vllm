# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the fused Inkling vision/audio tower kernels.

Each kernel is checked against a pure-PyTorch reference implementing the
towers' original op sequence (native ``rms_norm`` semantics: fp32 variance,
bf16-rounded weight multiply; exact-erf GELU; fp32-accumulated embedding sum).
The fused kernels keep the same accumulation dtype and rounding points, so
outputs differ from the reference only by reduction order — a few bf16 ulps.
"""

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

if not current_platform.is_cuda():
    pytest.skip("requires CUDA", allow_module_level=True)

from vllm.models.inkling.common.towers import fold_timespace_to_depth
from vllm.models.inkling.nvidia.ops.mm_towers import dmel_embed_sum_norm, rmsnorm_gelu

DTYPE = torch.bfloat16


def _bf16_spacing(x: torch.Tensor) -> torch.Tensor:
    return torch.exp2(torch.floor(torch.log2(x.float().abs().clamp(min=1e-30)))) * (
        2**-7
    )


def _assert_close_ulps(
    got: torch.Tensor,
    ref: torch.Tensor,
    max_ulps: float = 4.0,
    atol: float = 1e-3,
    pre_act: torch.Tensor | None = None,
) -> None:
    """Fused vs reference differ only by fp32 reduction order, which shows up
    as a few bf16 ulps at any magnitude — a fixed rtol misrepresents that, so
    compare against the reference's local ulp spacing. ``pre_act`` (the
    reference pre-activation) adds derivative-propagated slack: near
    ``gelu(x) ~ 0`` an ulp-level input flip legitimately moves the output by
    many of ITS (tiny) ulps, bounded by |gelu'| <= 1.13 times the input ulps."""
    g, r = got.float(), ref.float()
    tol = torch.clamp(max_ulps * _bf16_spacing(ref), min=atol)
    if pre_act is not None:
        tol = tol + 2.5 * _bf16_spacing(pre_act)
    bad = (g - r).abs() > tol
    assert not bad.any(), (
        f"{int(bad.sum())}/{ref.numel()} elements beyond tolerance; "
        f"max abs diff {(g - r).abs().max().item():.3e}"
    )


def _ref_rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    # Matches ir.ops.rms_norm: fp32 normalize, bf16 weight multiply.
    x32 = x.float()
    var = x32.pow(2).mean(dim=-1, keepdim=True)
    xn = x32 * torch.rsqrt(var + eps)
    return (xn.to(weight.dtype) * weight).to(x.dtype)


@pytest.mark.parametrize("rows", [1, 7, 1200, 38400])
@pytest.mark.parametrize("dim", [128, 320, 4800, 6144])
@pytest.mark.parametrize("gelu", [True, False])
def test_rmsnorm_gelu(rows: int, dim: int, gelu: bool) -> None:
    torch.manual_seed(rows * dim)
    x = torch.randn(rows, dim, device="cuda", dtype=DTYPE)
    w = torch.randn(dim, device="cuda", dtype=DTYPE)

    h = _ref_rms_norm(x, w, 1e-5)
    ref = F.gelu(h) if gelu else h
    got = rmsnorm_gelu(x, w, 1e-5, gelu=gelu)

    _assert_close_ulps(got, ref, pre_act=h if gelu else None)


@pytest.mark.parametrize("n", [1, 5, 64])
@pytest.mark.parametrize(
    "shape,fold",
    [
        ((2, 8, 8, 128), (1, 2)),  # the real L0 -> L1 vision transition
        ((2, 4, 4, 320), (1, 2)),
        ((2, 8, 8, 128), (2, 2)),  # temporal + spatial fold
    ],
)
def test_rmsnorm_gelu_folded_store(
    n: int, shape: tuple[int, ...], fold: tuple[int, int]
) -> None:
    torch.manual_seed(n)
    x = torch.randn(n, *shape, device="cuda", dtype=DTYPE)
    w = torch.randn(shape[-1], device="cuda", dtype=DTYPE)

    plain = rmsnorm_gelu(x, w, 1e-5, gelu=True)
    ref = fold_timespace_to_depth(plain, *fold)
    got = rmsnorm_gelu(x, w, 1e-5, gelu=True, fold=fold)

    # The folded store is a pure permutation of the unfolded output.
    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, rtol=0, atol=0)


@pytest.mark.parametrize("num_frames", [1, 3, 100, 4097])
@pytest.mark.parametrize("with_norm", [True, False])
def test_dmel_embed_sum_norm(num_frames: int, with_norm: bool) -> None:
    torch.manual_seed(num_frames)
    n_bins, vocab, dim = 80, 16, 6144
    idx = torch.randint(
        0, vocab, (num_frames, n_bins), device="cuda", dtype=torch.int32
    )
    table = torch.randn(n_bins * vocab, dim, device="cuda", dtype=DTYPE)
    norm_w = torch.randn(dim, device="cuda", dtype=DTYPE)

    flat = (torch.arange(n_bins, device="cuda", dtype=torch.int32) * vocab).unsqueeze(
        0
    ) + idx
    ref = (
        F.embedding(flat.reshape(-1).long(), table)
        .reshape(num_frames, n_bins, dim)
        .sum(dim=1)
    )
    if with_norm:
        ref = _ref_rms_norm(ref, norm_w, 1e-6)

    got = dmel_embed_sum_norm(idx, table, norm_w if with_norm else None, 1e-6)
    _assert_close_ulps(got, ref)


def test_empty_inputs() -> None:
    x = torch.empty(0, 320, device="cuda", dtype=DTYPE)
    w = torch.randn(320, device="cuda", dtype=DTYPE)
    assert rmsnorm_gelu(x, w, 1e-5).shape == (0, 320)

    idx = torch.empty(0, 80, device="cuda", dtype=torch.int32)
    table = torch.randn(1280, 6144, device="cuda", dtype=DTYPE)
    assert dmel_embed_sum_norm(idx, table, None, 0.0).shape == (0, 6144)
