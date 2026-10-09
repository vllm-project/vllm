# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CUDA iHC boundary must match the unfused post, pre and RMSNorm path."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.models.hy_v4.nvidia.triton_ihc import triton_ihc_post, triton_ihc_pre

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not ops.hy_v4_ihc_boundary_supported(4, 6144),
    reason="needs the CUDA hy_v4_ihc_boundary op",
)

HC, MAGNITUDE, HC_EPS, NORM_EPS, VARIANCE_EPS = 4, 2.0, 1e-6, 1e-5, 1e-5


def _rms_norm(x, weight, eps):
    """Match the CUDA rms_norm: one rounding of x * rsqrt(mean(x^2) + eps) * w."""
    xf = x.float()
    scale = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * scale * weight.float()).to(x.dtype)


def _unfused(xa, residual, post, weight, scale, base, norm_weight):
    if xa is not None:
        residual = triton_ihc_post(xa, residual, post)
    hidden, gates = triton_ihc_pre(
        residual, weight, scale, base, MAGNITUDE, HC_EPS, NORM_EPS
    )
    if norm_weight is not None:
        hidden = _rms_norm(hidden, norm_weight, VARIANCE_EPS)
    return residual, hidden, gates


def _fp32_reduction_reference(residual, weight, scale, base, norm_weight):
    """Gated reduction kept in high precision into the norm (no BF16 rounding
    before it), as round_before_norm=False and the HPC library compute."""
    r = residual.double()
    flat = r.flatten(1)
    mixes = flat @ weight.double().t()
    rstd = torch.rsqrt(flat.square().mean(-1, keepdim=True) + NORM_EPS)
    pre = torch.sigmoid(mixes[:, :HC] * rstd * scale[0] + base[:HC]) + HC_EPS
    hidden = (pre[:, :, None] * r).sum(1)
    if norm_weight is not None:
        hidden = hidden * torch.rsqrt(
            hidden.square().mean(-1, keepdim=True) + VARIANCE_EPS
        )
        hidden = hidden * norm_weight.double()
    return hidden.to(torch.bfloat16)


def _inputs(num_tokens, hidden_size, offset_views):
    torch.manual_seed(num_tokens)
    device = "cuda"
    # Offset views start one token in: still 16-byte aligned, but not at the
    # allocation base.
    pad = 1 if offset_views else 0
    residual = (torch.randn(num_tokens + pad, HC, hidden_size, device=device) * 2).to(
        torch.bfloat16
    )[pad:]
    xa = torch.randn(num_tokens + pad, hidden_size, device=device).to(torch.bfloat16)[
        pad:
    ]
    post = torch.rand(num_tokens, HC, device=device) * 2
    weight = torch.randn(2 * HC, HC * hidden_size, device=device) * 6e-3
    scale = torch.tensor([0.5, 0.5], device=device)
    base = torch.randn(2 * HC, device=device) * 0.5
    norm_weight = (1 + 0.1 * torch.randn(hidden_size, device=device)).to(torch.bfloat16)
    return xa, residual, post, weight, scale, base, norm_weight


# Token counts cover each split-count bucket and its boundary (16/17, 128/129,
# 512/513, 1024/1025, 8192/8193) plus partial 16-token tiles.
@pytest.mark.parametrize("num_tokens", [1, 7, 16, 17, 128, 129, 513, 1025, 2051, 8193])
@pytest.mark.parametrize("hidden_size", [1088, 2048, 4096, 5120, 6144, 7168])
@pytest.mark.parametrize("with_post", [True, False])
@pytest.mark.parametrize("with_norm", [True, False])
@pytest.mark.parametrize("offset_views", [False, True])
@pytest.mark.parametrize("round_before_norm", [False, True])
def test_boundary_matches_reference(
    num_tokens, hidden_size, with_post, with_norm, offset_views, round_before_norm
):
    xa, residual, post, weight, scale, base, norm_weight = _inputs(
        num_tokens, hidden_size, offset_views
    )
    if not with_post:
        xa = post = None
    if not with_norm:
        norm_weight = None

    new_residual, hidden, gates = ops.hy_v4_ihc_boundary(
        residual,
        xa,
        post,
        weight,
        scale,
        base,
        norm_weight,
        MAGNITUDE,
        HC_EPS,
        NORM_EPS,
        VARIANCE_EPS,
        round_before_norm,
    )

    if with_post:
        # The post step is one FMA: the exact result rounded to BF16, up to
        # double rounding through FP32 (one BF16 step).
        exact = post.double()[:, :, None] * xa.double()[:, None, :]
        exact = exact + residual.double()
        one_step = exact.to(torch.bfloat16).double().abs() * 2.0**-7
        assert ((new_residual.double() - exact).abs() <= one_step).all()
    else:
        assert new_residual.data_ptr() == residual.data_ptr()
    # From the same residual, gates and the reduced hidden state differ from
    # the reference only by FP32 summation order: the unfused pre and RMSNorm
    # when rounding before the norm, else a high-precision reduction.
    _, ref_hidden, ref_gates = _unfused(
        None, new_residual, None, weight, scale, base, norm_weight
    )
    if not round_before_norm:
        ref_hidden = _fp32_reduction_reference(
            new_residual, weight, scale, base, norm_weight
        )
    torch.testing.assert_close(gates, ref_gates, rtol=1e-4, atol=1e-4)
    rel_l2 = (hidden.float() - ref_hidden.float()).norm() / ref_hidden.float().norm()
    assert rel_l2 < 2e-4, rel_l2


@pytest.mark.parametrize("num_tokens", [1, 300])
def test_boundary_is_deterministic(num_tokens):
    xa, residual, post, weight, scale, base, norm_weight = _inputs(
        num_tokens, 6144, offset_views=False
    )
    args = (residual, xa, post, weight, scale, base, norm_weight)
    args += (MAGNITUDE, HC_EPS, NORM_EPS, VARIANCE_EPS)
    first = ops.hy_v4_ihc_boundary(*args)
    for _ in range(3):
        for a, b in zip(first, ops.hy_v4_ihc_boundary(*args)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
