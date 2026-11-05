# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

import vllm.models.muse_spark.common.ops.fused_rmsnorm as fused_rmsnorm_ops
import vllm.models.muse_spark.nvidia.model as model_ops
from vllm.models.muse_spark.nvidia.model import (
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
requires_b300_dual_norm = pytest.mark.skipif(
    not current_platform.is_device_capability(103)
    or model_ops.gated_residual_next_rmsnorm is None,
    reason="the FlashInfer dual norm requires CuTe DSL on B300",
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
# 256 rows is where the Triton boundary kernel switches configs on SM100.
@pytest.mark.parametrize("rows", [255, 256])
@pytest.mark.parametrize("post_input_scale", [False, True])
def test_triton_residual_boundary_matches_native(monkeypatch, rows, post_input_scale):
    torch.manual_seed(0)
    boundary, pre_norm = make_residual_boundary(rows, post_input_scale=post_input_scale)

    monkeypatch.setattr(model_ops, "gated_residual_next_rmsnorm", None)
    actual_out, actual_stream = model_ops.norm_residual_then_norm(
        boundary, pre_norm, out_dtype=torch.bfloat16
    )

    monkeypatch.setattr(model_ops, "supports_norm_residual_norm", lambda *_: False)
    expected_out, expected_stream = model_ops.norm_residual_then_norm(
        boundary, pre_norm, out_dtype=torch.bfloat16
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=0.015625, atol=0.03125)
    torch.testing.assert_close(actual_stream, expected_stream, rtol=1e-5, atol=1e-5)


@requires_b300_dual_norm
@pytest.mark.parametrize("rows", [0, 128])
@pytest.mark.parametrize("post_input_scale", [False, True])
def test_flashinfer_residual_boundary_matches_triton(
    monkeypatch, rows, post_input_scale
):
    """Both attention and FFN boundaries dispatch to FlashInfer with the same math."""
    if post_input_scale and not model_ops._FLASHINFER_DUAL_NORM_HAS_POST_WEIGHT:
        pytest.skip("FlashInfer does not support the post-norm input scale")
    torch.manual_seed(1000 + rows)
    boundary, pre_norm = make_residual_boundary(rows, post_input_scale=post_input_scale)
    assert pre_norm.weight is not None
    assert model_ops.supports_gated_residual_next_rmsnorm(
        boundary.branch,
        boundary.stream,
        pre_norm.weight,
        boundary.connection.alpha,
        boundary.connection.beta,
    )

    candidate = model_ops.gated_residual_next_rmsnorm
    assert candidate is not None
    called = False

    def tracked_candidate(*args, **kwargs):
        nonlocal called
        called = True
        return candidate(*args, **kwargs)

    monkeypatch.setattr(model_ops, "gated_residual_next_rmsnorm", tracked_candidate)
    actual_out, actual_stream = model_ops.norm_residual_then_norm(
        boundary, pre_norm, out_dtype=torch.bfloat16
    )
    assert called

    monkeypatch.setattr(model_ops, "gated_residual_next_rmsnorm", None)
    expected_out, expected_stream = model_ops.norm_residual_then_norm(
        boundary, pre_norm, out_dtype=torch.bfloat16
    )

    torch.testing.assert_close(actual_out, expected_out, rtol=0.015625, atol=0.03125)
    torch.testing.assert_close(actual_stream, expected_stream, rtol=1e-5, atol=1e-5)


@requires_cuda
def test_unsupported_flashinfer_post_norm_uses_triton(monkeypatch):
    """Keep weighted post-norms on Triton when FlashInfer lacks the post weight."""
    boundary, pre_norm = make_residual_boundary(17)
    boundary.post_norm = make_norm(MuseSpark1xRMSNormWithInputScale)
    monkeypatch.setattr(model_ops, "_FLASHINFER_DUAL_NORM_HAS_POST_WEIGHT", False)

    def unexpected_flashinfer_call(*args, **kwargs):
        pytest.fail("unsupported post-norm dispatched to FlashInfer")

    monkeypatch.setattr(
        model_ops, "gated_residual_next_rmsnorm", unexpected_flashinfer_call
    )
    actual_out, actual_stream = model_ops.norm_residual_then_norm(
        boundary, pre_norm, out_dtype=torch.bfloat16
    )
    monkeypatch.setattr(model_ops, "supports_norm_residual_norm", lambda *_: False)
    expected_out, expected_stream = model_ops.norm_residual_then_norm(
        boundary, pre_norm, out_dtype=torch.bfloat16
    )
    torch.testing.assert_close(actual_out, expected_out, rtol=0.015625, atol=0.03125)
    torch.testing.assert_close(actual_stream, expected_stream, rtol=1e-5, atol=1e-5)


def postnorm_inputs(rows: int, mode: int):
    branch = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.bfloat16)
    stream = torch.randn(rows, HIDDEN, device="cuda", dtype=torch.float32)
    weight = (
        torch.randn(HIDDEN, device="cuda", dtype=torch.bfloat16) * 0.1 if mode else None
    )
    gate = torch.empty(HIDDEN, device="cuda", dtype=torch.float32).uniform_(-3, 3)
    beta = torch.sigmoid(gate)
    alpha = torch.sqrt(torch.clamp(1.0 - beta.square(), min=1e-3))
    return branch, stream, weight, alpha, beta


def postnorm(inputs, mode: int):
    branch, stream, weight, alpha, beta = inputs
    return fused_rmsnorm_ops.fused_rmsnorm_fwd(
        branch,
        weight,
        EPS,
        gain_center=1.0 if mode else 0.0,
        input_scale=bool(mode),
        residual_stream=stream,
        alpha=alpha,
        beta=beta,
    )


@pytest.fixture
def requires_postnorm_gated_residual():
    if (
        not current_platform.is_cuda()
        or fused_rmsnorm_ops.postnorm_gated_residual is None
    ):
        pytest.skip("FlashInfer CuTe DSL on CUDA is required")
    if not fused_rmsnorm_ops.supports_gated_residual_rmsnorm(*postnorm_inputs(1, 1)):
        pytest.skip("installed FlashInfer does not support this device")


@pytest.mark.usefixtures("requires_postnorm_gated_residual")
@pytest.mark.parametrize("mode", [0, 1])
def test_postnorm_gated_residual_matches_triton(monkeypatch, mode):
    candidate = fused_rmsnorm_ops.postnorm_gated_residual
    calls = []

    def tracked_candidate(*args):
        calls.append(True)
        return candidate(*args)

    monkeypatch.setattr(fused_rmsnorm_ops, "postnorm_gated_residual", tracked_candidate)
    torch.manual_seed(1017 + 10 * mode)
    inputs = postnorm_inputs(17, mode)
    got = postnorm(inputs, mode)
    assert len(calls) == 1

    monkeypatch.setattr(fused_rmsnorm_ops, "postnorm_gated_residual", None)
    expected = postnorm(inputs, mode)
    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.usefixtures("requires_postnorm_gated_residual")
def test_postnorm_gated_residual_empty_input():
    got = postnorm(postnorm_inputs(0, 0), 0)
    assert got.shape == (0, HIDDEN)
    assert got.dtype == torch.float32


@pytest.mark.usefixtures("requires_postnorm_gated_residual")
def test_postnorm_gated_residual_cuda_graph_replays_changing_inputs(monkeypatch):
    mode = 1
    torch.manual_seed(2001)
    inputs = list(postnorm_inputs(17, mode))

    # Compile the shape-specialized CuTe kernel before graph capture.
    postnorm(inputs, mode)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        got = postnorm(inputs, mode)

    candidate = fused_rmsnorm_ops.postnorm_gated_residual
    for seed in (3001, 4001):
        torch.manual_seed(seed)
        for static, fresh in zip(inputs, postnorm_inputs(17, mode)):
            static.copy_(fresh)
        graph.replay()
        torch.accelerator.synchronize()
        replayed = got.clone()

        monkeypatch.setattr(fused_rmsnorm_ops, "postnorm_gated_residual", None)
        expected = postnorm(inputs, mode)
        monkeypatch.setattr(fused_rmsnorm_ops, "postnorm_gated_residual", candidate)
        torch.testing.assert_close(replayed, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tokens", [0, 1, 257])
def test_o_norm_gate_reads_the_strided_gate_slice(tokens):
    """The gate is a non-contiguous view into the QKV-gate projection."""
    head_dim, num_heads, num_kv_heads, eps = 64, 32, 2, 1e-5
    if model_ops.o_norm_sigmoid_gate is None or not (
        model_ops.supports_o_norm_sigmoid_gate(torch.bfloat16, head_dim)
    ):
        pytest.skip("installed FlashInfer has no O-norm gate for this device")

    generator = torch.Generator(device="cuda").manual_seed(919 + tokens)
    q_size, kv_size = num_heads * head_dim, num_kv_heads * head_dim
    gate_offset = q_size + 2 * kv_size
    projection = torch.empty(
        tokens, gate_offset + q_size, dtype=torch.bfloat16, device="cuda"
    ).uniform_(-8.0, 8.0, generator=generator)
    gate = projection[:, gate_offset:].view(tokens, num_heads, head_dim)
    x = torch.randn(
        tokens,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device="cuda",
        generator=generator,
    )
    values = x.float()
    inv_rms = torch.rsqrt(values.square().mean(dim=-1, keepdim=True) + eps)
    expected = (values * inv_rms * torch.sigmoid(gate.float())).bfloat16()
    output_ptr = x.data_ptr()

    model_ops.o_norm_sigmoid_gate(x, gate, eps)

    assert x.data_ptr() == output_ptr
    torch.testing.assert_close(x, expected, atol=0.015625, rtol=0.015625)
