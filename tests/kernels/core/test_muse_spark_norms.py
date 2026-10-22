# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.models.muse_spark.amd.model as model_ops
from vllm.models.muse_spark.common.layers import (
    MuseSpark1xNormPreservingResidualConnection,
    MuseSpark1xResidualStream,
    MuseSpark1xRMSNorm,
    MuseSpark1xRMSNormWithInputScale,
)
from vllm.platforms import current_platform

HIDDEN, EPS = 8192, 1e-5

requires_cuda = pytest.mark.skipif(
    not current_platform.is_cuda_alike(),
    reason="the fused RMSNorm is a Triton kernel",
)


@pytest.fixture(autouse=True)
def bf16_default_dtype():
    """Build the norms the way the model does: under vLLM's dtype context."""
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    yield
    torch.set_default_dtype(previous)


def make_norm(
    cls,
    *,
    has_weight=True,
    zero_centered_gamma=True,
    is_post_norm=True,
    eps=EPS,
):
    norm = cls(
        HIDDEN,
        eps=eps,
        has_weight=has_weight,
        is_post_norm=is_post_norm,
        zero_centered_gamma=zero_centered_gamma,
        post_norm_gain_center_type="one",
        residual_id=3,
    ).cuda()
    if has_weight:
        with torch.no_grad():
            norm.weight.copy_(
                torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1
            )
    return norm


def make_residual_boundary(rows, *, post_input_scale=False):
    post_norm = make_norm(
        MuseSpark1xRMSNormWithInputScale if post_input_scale else MuseSpark1xRMSNorm,
        has_weight=post_input_scale,
        is_post_norm=True,
        eps=1e-8,
    )
    pre_norm = make_norm(MuseSpark1xRMSNorm, is_post_norm=False)
    connection = MuseSpark1xNormPreservingResidualConnection(HIDDEN).cuda()
    with torch.no_grad():
        connection.gate.copy_(torch.linspace(-0.9, 0.9, HIDDEN, device="cuda"))
    connection.materialize_coefficients()
    branch = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    stream = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.float32)
    boundary = MuseSpark1xResidualStream(
        stream=stream,
        branch=branch,
        post_norm=post_norm,
        connection=connection,
    )
    return boundary, pre_norm


def reference(norm, x):
    """The same norm in fp64, weight and all."""
    x64 = x.double()
    weight = norm.weight.double()
    if norm.zero_centered_gamma:
        weight = weight + norm.gain_center
    if norm.applies_weight_before_norm:
        x64 = x64 * weight
        return x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS)
    normed = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS)
    return normed * weight


def rel_err(got, ref64):
    return (got.double() - ref64).abs().max().item() / ref64.abs().max().item()


@requires_cuda
@pytest.mark.parametrize("cls", [MuseSpark1xRMSNorm, MuseSpark1xRMSNormWithInputScale])
@pytest.mark.parametrize("num_tokens", [1, 4096])
def test_nothing_is_rounded_before_the_store(cls, num_tokens):
    """An fp32 result must carry fp32 error, not bf16 error (~4e-3 relative)."""
    torch.manual_seed(0)
    norm = make_norm(cls)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", dtype=torch.bfloat16)

    got = norm(x, out_dtype=torch.float32)
    assert got.dtype == torch.float32
    assert rel_err(got, reference(norm, x)) < 1e-5


@requires_cuda
@pytest.mark.parametrize("cls", [MuseSpark1xRMSNorm, MuseSpark1xRMSNormWithInputScale])
def test_folding_the_cast_in_beats_casting_around_it(cls):
    """Reading the fp32 stream directly must not be worse than pre-casting."""
    torch.manual_seed(0)
    norm = make_norm(cls)
    x = torch.randn(4096, HIDDEN, device="cuda", dtype=torch.float32) * 3
    ref = reference(norm, x)

    folded = norm(x, out_dtype=torch.bfloat16)
    cast_first = norm(x.to(torch.bfloat16))

    assert folded.dtype == torch.bfloat16
    assert rel_err(folded, ref) <= rel_err(cast_first, ref)


@requires_cuda
@pytest.mark.parametrize("cls", [MuseSpark1xRMSNorm, MuseSpark1xRMSNormWithInputScale])
@pytest.mark.parametrize("has_weight", [True, False])
def test_triton_matches_torch(cls, has_weight):
    torch.manual_seed(0)
    norm = make_norm(cls, has_weight=has_weight)
    x = torch.randn(4096, HIDDEN, device="cuda", dtype=torch.float32) * 3

    fused = norm(x, out_dtype=torch.bfloat16)
    native = norm.forward_native(x, None, torch.bfloat16)

    # Different reduction orders can flip values on a bf16 rounding boundary:
    # a few elements in tens of millions, a few ulp each.
    diff = fused != native
    assert diff.float().mean() < 1e-5, f"{diff.sum().item()} elements differ"
    torch.testing.assert_close(fused, native, rtol=2e-2, atol=0.0)


@requires_cuda
def test_out_dtype_defaults_to_the_input_dtype():
    norm = make_norm(MuseSpark1xRMSNorm)
    for dtype in (torch.float32, torch.bfloat16):
        x = torch.randn(8, HIDDEN, device="cuda", dtype=dtype)
        assert norm(x).dtype == dtype
        assert norm.forward_native(x).dtype == dtype


@requires_cuda
@pytest.mark.parametrize("rows", [255, 256])
@pytest.mark.parametrize("post_input_scale", [False, True])
def test_triton_residual_boundary_matches_native(monkeypatch, rows, post_input_scale):
    torch.manual_seed(0)
    boundary, pre_norm = make_residual_boundary(rows, post_input_scale=post_input_scale)

    actual_out, actual_stream = model_ops.norm_residual_then_norm(
        boundary, pre_norm, out_dtype=torch.bfloat16
    )

    monkeypatch.setattr(model_ops, "supports_norm_residual_norm", lambda *_: False)
    expected_out, expected_stream = model_ops.norm_residual_then_norm(
        boundary, pre_norm, out_dtype=torch.bfloat16
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=0.015625, atol=0.03125)
    torch.testing.assert_close(actual_stream, expected_stream, rtol=1e-5, atol=1e-5)
