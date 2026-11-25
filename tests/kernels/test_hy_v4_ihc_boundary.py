# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The CUDA iHC boundary must match the unfused post, pre and RMSNorm path."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8_packed_for_deepgemm,
)
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


@pytest.mark.parametrize("num_tokens", [1, 8, 1024])
@pytest.mark.parametrize("hidden_size", [3072, 6144])
@pytest.mark.parametrize("with_post", [True, False])
@pytest.mark.parametrize("with_norm", [True, False])
@pytest.mark.parametrize("offset_views", [False, True])
@pytest.mark.parametrize("round_before_norm", [False, True])
def test_boundary_matches_reference(
    num_tokens, hidden_size, with_post, with_norm, offset_views, round_before_norm
):
    _check_boundary(
        num_tokens, hidden_size, with_post, with_norm, offset_views, round_before_norm
    )


def _check_boundary(
    num_tokens,
    hidden_size,
    with_post,
    with_norm,
    offset_views,
    round_before_norm,
):
    xa, residual, post, weight, scale, base, norm_weight = _inputs(
        num_tokens, hidden_size, offset_views
    )
    if not with_post:
        xa = post = None
    if not with_norm:
        norm_weight = None

    new_residual, hidden, gates, _ = ops.hy_v4_ihc_boundary(
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
    diff = (hidden.float() - ref_hidden.float()).abs()
    rel_l2 = diff.norm() / ref_hidden.float().norm()
    # One flipped BF16 rounding can dominate a one-token row on its own.
    one_flip = (diff > 0).sum() <= 1 and (
        diff <= ref_hidden.float().abs() * 2.0**-7
    ).all()
    assert rel_l2 < 2e-4 or one_flip, rel_l2


def test_tma_path_writes_the_two_kernel_residual():
    """The post step is token-local, so 1025 tokens (TMA stats kernel on
    B200 / B300) write the same residual for the first 1024 tokens as a
    1024-token call (two-kernel path)."""
    xa, residual, post, weight, scale, base, norm_weight = _inputs(
        1025, 6144, offset_views=False
    )
    rest = (weight, scale, base, norm_weight, MAGNITUDE, HC_EPS, NORM_EPS)
    long = ops.hy_v4_ihc_boundary(residual, xa, post, *rest, VARIANCE_EPS)
    short = ops.hy_v4_ihc_boundary(
        residual[:1024], xa[:1024], post[:1024], *rest, VARIANCE_EPS
    )
    assert torch.equal(long[0][:1024], short[0])
    torch.testing.assert_close(long[2][:1024], short[2], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("num_tokens", [1, 8, 1024])
def test_boundary_is_deterministic(num_tokens):
    xa, residual, post, weight, scale, base, norm_weight = _inputs(
        num_tokens, 6144, offset_views=False
    )
    args = (residual, xa, post, weight, scale, base, norm_weight) + (
        MAGNITUDE,
        HC_EPS,
        NORM_EPS,
        VARIANCE_EPS,
    )
    first = ops.hy_v4_ihc_boundary(*args)
    for _ in range(3):
        for a, b in zip(first, ops.hy_v4_ihc_boundary(*args)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("num_tokens", [1, 8, 1024])
@pytest.mark.parametrize("hidden_size", [3072, 6144])
@pytest.mark.parametrize("with_post", [True, False])
def test_fp8_out_matches_separate_quant(num_tokens, hidden_size, with_post):
    xa, residual, post, weight, scale, base, norm_weight = _inputs(
        num_tokens, hidden_size, offset_views=False
    )
    # Per-128-channel norm weights over many binades, so group scales differ.
    norm_weight = (
        (
            norm_weight.float().view(-1, 128)
            * torch.exp2(torch.randint(-8, 8, (hidden_size // 128, 1), device="cuda"))
        )
        .flatten()
        .to(torch.bfloat16)
    )
    if not with_post:
        xa = post = None
    args = (residual, xa, post, weight, scale, base, norm_weight) + (
        MAGNITUDE,
        HC_EPS,
        NORM_EPS,
        VARIANCE_EPS,
    )
    plain = ops.hy_v4_ihc_boundary(*args)
    fused = ops.hy_v4_ihc_boundary(*args, fp8_out=True)
    assert plain[3] is None
    for a, b in zip(plain[:3], fused[:3]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert fused[3] is not None
    q, s = fused[3]
    ref_q, ref_s = per_token_group_quant_fp8_packed_for_deepgemm(
        fused[1], group_size=128, use_ue8m0=True
    )
    assert s.shape == ref_s.shape and s.stride() == ref_s.stride()
    assert torch.equal(q.view(torch.uint8), ref_q.view(torch.uint8))
    assert torch.equal(s, ref_s)


@pytest.mark.parametrize("num_tokens", [1, 8, 1024])
@pytest.mark.parametrize("hidden_size", [2816, 3072])
def test_layers_run_post_and_head_on_the_boundary(
    default_vllm_config, dist_init, num_tokens, hidden_size
):
    """A lone post step and the last post with the head and final norm run on
    the boundary op, matching the unfused layers. The head gate is loaded in
    place after construction, as the weight loader does."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from vllm.models.hy_v4.nvidia.hc import HYV4HCHeadLayer, HYV4HCLayer
    from vllm.transformers_utils.configs.hy_v4 import HYV4Config

    config = HYV4Config(hidden_size=hidden_size)
    xa, residual, post, *_ = _inputs(num_tokens, hidden_size, offset_views=False)
    with torch.device("cuda"):
        layer = HYV4HCLayer(config, 0)
        head = HYV4HCHeadLayer(
            config, hidden_size, hc_mult=HC, hc_eps=config.hc_eps
        ).cuda()
        norm = RMSNorm(hidden_size, config.rms_norm_eps).to(torch.bfloat16)
    assert layer.fused_post and head.packed_gate is not None
    with torch.no_grad():
        head.hc_head_fn.weight.copy_(torch.randn_like(head.hc_head_fn.weight) * 6e-3)
        head.hc_head_scale.fill_(0.5)
        head.hc_head_base.copy_(torch.randn_like(head.hc_head_base) * 0.5)
        norm.weight.copy_(1 + 0.1 * torch.randn_like(norm.weight))

    new_residual = layer.post(xa, residual, post)
    # One FMA, as in `test_boundary_matches_reference`: within one BF16 step.
    exact = post.double()[:, :, None] * xa.double()[:, None, :] + residual.double()
    one_step = exact.to(torch.bfloat16).double().abs() * 2.0**-7
    assert ((new_residual.double() - exact).abs() <= one_step).all()

    hidden = head.fused_post(xa, residual, post, norm)
    assert hidden is not None
    ref = _rms_norm(head(new_residual), norm.weight, norm.variance_epsilon)
    diff = (hidden.float() - ref.float()).abs()
    one_flip = (diff > 0).sum() <= 1 and (diff <= ref.float().abs() * 2.0**-7).all()
    assert diff.norm() / ref.float().norm() < 2e-4 or one_flip
