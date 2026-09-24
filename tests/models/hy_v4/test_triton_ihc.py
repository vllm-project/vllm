# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.models.hy_v4.amd.triton_ihc import (
    triton_ihc_post,
    triton_ihc_post_pre,
    triton_ihc_post_pre_rms_norm,
    triton_ihc_pre,
    triton_ihc_supported,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON

pytestmark = pytest.mark.skipif(
    not current_platform.is_rocm() or not HAS_TRITON or not torch.cuda.is_available(),
    reason="Requires ROCm, Triton, and a GPU",
)

_HC_MULT = 4
_HIDDEN_SIZE = 6144
_DTYPE = torch.bfloat16


def _pre_reference(x, weight, scale, base, magnitude, hc_eps, norm_eps):
    flat = x.float().flatten(1)
    reciprocal_rms = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = flat @ weight.float().transpose(0, 1) * reciprocal_rms
    pre = torch.sigmoid(mixes[:, :_HC_MULT] * scale[0] + base[:_HC_MULT]) + hc_eps
    post = (
        magnitude
        * torch.sigmoid(mixes[:, _HC_MULT:] * scale[1] + base[_HC_MULT : 2 * _HC_MULT])
        + hc_eps
    )
    return (pre.unsqueeze(-1) * x.float()).sum(1).to(x.dtype), post


def _post_pre_reference(
    x, residual, attn_post, weight, scale, base, magnitude, hc_eps, norm_eps
):
    y = attn_post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()
    y = y.to(x.dtype)
    reduced, mlp_post = _pre_reference(
        y, weight, scale, base, magnitude, hc_eps, norm_eps
    )
    return reduced, mlp_post, y


def _rms_norm_reference(x, weight, eps):
    x_float = x.float()
    return (
        x_float * torch.rsqrt(x_float.square().mean(-1, keepdim=True) + eps) * weight
    ).to(x.dtype)


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
def test_rocm_triton_ihc_pre_matches_eager(num_tokens):
    device = torch.device("cuda")
    x = torch.randn((num_tokens, _HC_MULT, _HIDDEN_SIZE), device=device, dtype=_DTYPE)
    weight = torch.randn(
        (2 * _HC_MULT, _HC_MULT * _HIDDEN_SIZE),
        device=device,
        dtype=torch.float32,
    )
    scale = torch.randn(2, device=device, dtype=torch.float32)
    base = torch.randn(2 * _HC_MULT, device=device, dtype=torch.float32)
    args = (2.0, 1e-6, 1e-5)

    actual_output, actual_post = triton_ihc_pre(x, weight, scale, base, *args)
    expected_output, expected_post = _pre_reference(x, weight, scale, base, *args)
    torch.testing.assert_close(actual_output, expected_output, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_post, expected_post, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
def test_rocm_triton_ihc_post_matches_eager(num_tokens):
    device = torch.device("cuda")
    x = torch.randn((num_tokens, _HIDDEN_SIZE), device=device, dtype=_DTYPE)
    residual = torch.randn(
        (num_tokens, _HC_MULT, _HIDDEN_SIZE), device=device, dtype=_DTYPE
    )
    post = torch.randn((num_tokens, _HC_MULT), device=device, dtype=torch.float32)

    actual = triton_ihc_post(x, residual, post)
    expected = (
        post.float().unsqueeze(-1) * x.float().unsqueeze(-2) + residual.float()
    ).to(_DTYPE)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
def test_rocm_triton_ihc_post_pre_matches_eager(num_tokens):
    device = torch.device("cuda")
    x = torch.randn((num_tokens, _HIDDEN_SIZE), device=device, dtype=_DTYPE)
    residual = torch.randn(
        (num_tokens, _HC_MULT, _HIDDEN_SIZE), device=device, dtype=_DTYPE
    )
    attn_post = torch.randn((num_tokens, _HC_MULT), device=device, dtype=torch.float32)
    weight = torch.randn(
        (2 * _HC_MULT, _HC_MULT * _HIDDEN_SIZE),
        device=device,
        dtype=torch.float32,
    )
    scale = torch.randn(2, device=device, dtype=torch.float32)
    base = torch.randn(2 * _HC_MULT, device=device, dtype=torch.float32)
    args = (2.0, 1e-6, 1e-5)

    actual_reduced, actual_post, actual_residual = triton_ihc_post_pre(
        x, residual, attn_post, weight, scale, base, *args
    )
    expected_reduced, expected_post, expected_residual = _post_pre_reference(
        x, residual, attn_post, weight, scale, base, *args
    )
    torch.testing.assert_close(actual_reduced, expected_reduced, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_post, expected_post, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("num_tokens", [1, 8, 32, 128])
def test_rocm_triton_ihc_post_pre_rms_norm_matches_eager(num_tokens):
    device = torch.device("cuda")
    x = torch.randn((num_tokens, _HIDDEN_SIZE), device=device, dtype=_DTYPE)
    residual = torch.randn(
        (num_tokens, _HC_MULT, _HIDDEN_SIZE), device=device, dtype=_DTYPE
    )
    attn_post = torch.randn((num_tokens, _HC_MULT), device=device, dtype=torch.float32)
    weight = torch.randn(
        (2 * _HC_MULT, _HC_MULT * _HIDDEN_SIZE),
        device=device,
        dtype=torch.float32,
    )
    scale = torch.randn(2, device=device, dtype=torch.float32)
    base = torch.randn(2 * _HC_MULT, device=device, dtype=torch.float32)
    norm_weight = torch.randn(_HIDDEN_SIZE, device=device, dtype=torch.float32)
    args = (2.0, 1e-6, 1e-5)

    actual_reduced, actual_post, actual_residual = triton_ihc_post_pre_rms_norm(
        x,
        residual,
        attn_post,
        weight,
        scale,
        base,
        norm_weight,
        *args,
    )
    expected_reduced, expected_post, expected_residual = _post_pre_reference(
        x, residual, attn_post, weight, scale, base, *args
    )
    expected_reduced = _rms_norm_reference(expected_reduced, norm_weight, args[-1])
    torch.testing.assert_close(actual_reduced, expected_reduced, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual_post, expected_post, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(actual_residual, expected_residual, rtol=2e-2, atol=2e-2)


def test_rocm_triton_ihc_dispatch():
    x = torch.empty((1, _HC_MULT, _HIDDEN_SIZE), device="cuda", dtype=_DTYPE)
    assert triton_ihc_supported(x)
