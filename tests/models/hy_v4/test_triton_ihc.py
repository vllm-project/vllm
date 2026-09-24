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


def _make_fused_inputs(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
):
    device = torch.device("cuda")
    x = torch.randn((num_tokens, hidden_size), device=device, dtype=dtype)
    residual = torch.randn(
        (num_tokens, _HC_MULT, hidden_size), device=device, dtype=dtype
    )
    attn_post = torch.randn((num_tokens, _HC_MULT), device=device, dtype=torch.float32)
    weight = torch.randn(
        (2 * _HC_MULT, _HC_MULT * hidden_size),
        device=device,
        dtype=torch.float32,
    )
    scale = torch.randn(2, device=device, dtype=torch.float32)
    base = torch.randn(2 * _HC_MULT, device=device, dtype=torch.float32)
    norm_weight = torch.randn(hidden_size, device=device, dtype=torch.float32)
    return x, residual, attn_post, weight, scale, base, norm_weight


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden_size", [4096, 6144])
def test_rocm_triton_ihc_fused_production_shapes(dtype, hidden_size):
    inputs = _make_fused_inputs(2, hidden_size, dtype)
    x, residual, attn_post, weight, scale, base, norm_weight = inputs
    args = (2.0, 1e-6, 1e-5)

    actual = triton_ihc_post_pre_rms_norm(*inputs, *args)
    expected = _post_pre_reference(x, residual, attn_post, weight, scale, base, *args)
    expected = (
        _rms_norm_reference(expected[0], norm_weight, args[-1]),
        expected[1],
        expected[2],
    )
    for actual_tensor, expected_tensor in zip(actual, expected):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=2e-2, atol=2e-2)


def test_rocm_triton_ihc_empty_and_noncontiguous_inputs():
    device = torch.device("cuda")
    weight = torch.randn(
        (2 * _HC_MULT, _HC_MULT * 4096), device=device, dtype=torch.float32
    )
    scale = torch.randn(2, device=device)
    base = torch.randn(2 * _HC_MULT, device=device)
    empty = torch.empty((0, _HC_MULT, 4096), device=device, dtype=_DTYPE)
    output, post = triton_ihc_pre(empty, weight, scale, base, 2.0, 1e-6, 1e-5)
    assert output.shape == (0, 4096)
    assert post.shape == (0, _HC_MULT)

    x = torch.randn((2, _HC_MULT, 8192), device=device, dtype=_DTYPE)[..., ::2]
    noncontiguous_weight = torch.randn((_HC_MULT * 4096, 2 * _HC_MULT), device=device).T
    assert not x.is_contiguous()
    assert not noncontiguous_weight.is_contiguous()
    actual = triton_ihc_pre(x, noncontiguous_weight, scale, base, 2.0, 1e-6, 1e-5)
    expected = _pre_reference(x, noncontiguous_weight, scale, base, 2.0, 1e-6, 1e-5)
    torch.testing.assert_close(actual[0], expected[0], rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(actual[1], expected[1], rtol=2e-3, atol=2e-3)


def test_rocm_triton_ihc_post_rejects_invalid_gate_shape():
    x = torch.empty((2, 4096), device="cuda", dtype=_DTYPE)
    residual = torch.empty((2, _HC_MULT, 4096), device="cuda", dtype=_DTYPE)
    post = torch.empty((1, _HC_MULT), device="cuda", dtype=torch.float32)
    with pytest.raises(AssertionError):
        triton_ihc_post(x, residual, post)


def test_rocm_triton_ihc_fused_compiles_and_captures():
    inputs = _make_fused_inputs(2, 4096, _DTYPE)
    args = (*inputs, 2.0, 1e-6, 1e-5)
    compiled = torch.compile(triton_ihc_post_pre_rms_norm, fullgraph=True)
    eager = triton_ihc_post_pre_rms_norm(*args)
    compiled_output = compiled(*args)
    for actual, expected in zip(compiled_output, eager):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)

    for _ in range(2):
        triton_ihc_post_pre_rms_norm(*args)
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = triton_ihc_post_pre_rms_norm(*args)
    graph.replay()
    torch.accelerator.synchronize()
    for actual, expected in zip(captured, eager):
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
