# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regressions for Beam normalization and activation rounding."""

import pytest
import torch


def _torch_rms_norm(x, weight, eps):
    y = x.float()
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        y = y * weight.float()
    return y.to(x.dtype)


def _torch_boundary(x, residual, post_weight, post_eps, next_weight, next_eps, scales):
    if post_eps is not None:
        x = _torch_rms_norm(x, post_weight, post_eps)
    residual = residual.to(x.dtype)
    if scales is not None:
        x = x * scales[0]
        residual = residual * scales[1]
    hidden = x + residual
    normalized = (
        _torch_rms_norm(hidden, next_weight, next_eps) if next_eps is not None else None
    )
    return hidden, normalized


def _assert_boundary_close(actual, expected, *, relative_tolerance=1e-4):
    for value, reference in zip(actual, expected):
        if reference is None:
            assert value is None
            continue
        bf16 = reference.dtype == torch.bfloat16
        torch.testing.assert_close(
            value,
            reference,
            rtol=0.016 if bf16 else 2e-5,
            atol=0.016 if bf16 else 2e-6,
        )
        # Allow isolated BF16 rounding differences, but catch omitted casts.
        if reference.numel():
            error = (value.float() - reference.float()).norm()
            assert error <= reference.float().norm() * (
                relative_tolerance if bf16 else 2e-6
            )


def _torch_megamoe_finalize(experts, weights, routed_scale, post_weight, post_eps):
    # BF16 x FP32 fits in FP64; round once per fused multiply-add to FP32.
    value = (experts[:, 0].double() * weights[:, 0, None].double()).float()
    for k in range(1, experts.shape[1]):
        value = (
            experts[:, k].double() * weights[:, k, None].double() + value.double()
        ).float()
    value = (value.bfloat16().float() * routed_scale).bfloat16()
    if post_eps is not None:
        value = _torch_rms_norm(value, post_weight, post_eps)
    return value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,hidden,top_k",
    [(0, 5120, 8), (33, 5120, 8), (7, 256, 1), (3, 8192, 32)],
)
@pytest.mark.parametrize("norm", ["none", "unweighted", "weighted"])
@torch.inference_mode()
def test_megamoe_finalize_preserves_rounding_and_ignores_capacity(
    rows, hidden, top_k, norm
):
    """Padded experts stay unread and the live result keeps both BF16 casts."""
    pytest.importorskip("flashinfer")
    from vllm.models.beam.nvidia.moe_finalize import beam_moe_finalize

    torch.manual_seed(42)
    experts = torch.randn(rows + 3, top_k, hidden, device="cuda", dtype=torch.bfloat16)
    experts[:rows:2].mul_(1e-3)
    experts[rows:] = float("nan")
    weights = torch.rand(rows + 3, top_k, device="cuda", dtype=torch.float32)
    weights[rows:] = float("nan")
    post_weight = (
        torch.randn(hidden, device="cuda", dtype=torch.bfloat16)
        if norm == "weighted"
        else None
    )
    post_eps = None if norm == "none" else 1e-6
    storage = torch.full((rows + 3, hidden), 37, device="cuda", dtype=experts.dtype)
    output = storage[:rows]
    actual = beam_moe_finalize(experts, weights, output, 0.137, post_weight, post_eps)
    expected = _torch_megamoe_finalize(
        experts[:rows], weights[:rows], 0.137, post_weight, post_eps
    )
    assert actual is output
    if post_eps is None:
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    else:
        _assert_boundary_close((actual,), (expected,))
    torch.testing.assert_close(storage[rows:], torch.full_like(storage[rows:], 37))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(
    "rows,hidden,top_k",
    [(0, 5120, 8), (33, 5120, 8), (3, 8192, 32)],
)
@pytest.mark.parametrize(
    "post_norm,next_norm,scales,routed_scale",
    [
        ("none", "none", None, 1.0),
        ("unweighted", "weighted", (0.173, 0.937), 0.137),
        ("weighted", "unweighted", None, 0.137),
    ],
)
@torch.inference_mode()
def test_megamoe_local_boundary_preserves_rounding_and_ignores_capacity(
    rows, hidden, top_k, post_norm, next_norm, scales, routed_scale
):
    """DP finalization keeps every BF16 boundary and ignores workspace padding."""
    pytest.importorskip("flashinfer")
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.models.beam.nvidia.moe_finalize import (
        beam_moe_finalize_norm_residual_norm,
    )

    torch.manual_seed(42)
    capacity = rows + 3
    experts = torch.randn(capacity, top_k, hidden, device="cuda", dtype=torch.bfloat16)
    experts[:rows:2].mul_(1e-3)
    experts[rows:] = float("nan")
    weights = torch.rand(capacity, top_k, device="cuda")
    weights[rows:] = float("nan")
    mapping = torch.arange(capacity * top_k, device="cuda", dtype=torch.int32).view(
        capacity, top_k
    )
    output = UnfinalizedMoEOutput(experts.view(-1, hidden), weights, mapping)
    residual = torch.randn(rows, hidden, device="cuda", dtype=experts.dtype)
    original_residual = residual.clone()

    def norm_args(kind):
        weight = (
            torch.randn(hidden, device="cuda", dtype=experts.dtype)
            if kind == "weighted"
            else None
        )
        return weight, None if kind == "none" else 1e-6

    post_weight, post_eps = norm_args(post_norm)
    next_weight, next_eps = norm_args(next_norm)
    actual = beam_moe_finalize_norm_residual_norm(
        output,
        residual,
        post_weight,
        post_eps,
        next_weight,
        next_eps,
        scales,
        routed_scale,
    )
    finalized = _torch_megamoe_finalize(
        experts[:rows], weights[:rows], routed_scale, None, None
    )
    expected = _torch_boundary(
        finalized, residual, post_weight, post_eps, next_weight, next_eps, scales
    )
    _assert_boundary_close(actual, expected)
    torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("strategy", ["fused", "unfused"])
@torch.inference_mode()
def test_megamoe_local_boundary_graph_replays_with_live_inputs(strategy):
    """Captured DP finalization reads current routing, residual, and norm weights."""
    pytest.importorskip("flashinfer")
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.models.beam.nvidia.moe_finalize import (
        beam_moe_finalize_norm_residual_norm,
    )

    torch.manual_seed(42)
    experts = torch.randn(19, 8, 5120, device="cuda", dtype=torch.bfloat16)
    weights = torch.rand(19, 8, device="cuda")
    mapping = torch.arange(19 * 8, device="cuda", dtype=torch.int32).view(19, 8)
    output = UnfinalizedMoEOutput(experts.view(-1, 5120), weights, mapping)
    residual = torch.randn(17, 5120, device="cuda", dtype=experts.dtype)
    post_weight = torch.randn(5120, device="cuda", dtype=experts.dtype)
    next_weight = torch.randn_like(post_weight)
    scales = (0.173, 0.937)

    def run():
        return beam_moe_finalize_norm_residual_norm(
            output,
            residual,
            post_weight,
            1e-6,
            next_weight,
            1e-6,
            scales,
            0.137,
            strategy=strategy,
        )

    run()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = run()
    for _ in range(2):
        experts.add_(0.125)
        weights.copy_(weights.roll(1, dims=1))
        residual.sub_(0.25)
        post_weight.neg_()
        next_weight.mul_(0.5)
        graph.replay()
        finalized = _torch_megamoe_finalize(
            experts[:17], weights[:17], 0.137, None, None
        )
        expected = _torch_boundary(
            finalized, residual, post_weight, 1e-6, next_weight, 1e-6, scales
        )
        _assert_boundary_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 8, 128])
@pytest.mark.parametrize("cap", [7.3, 30.0])
@torch.inference_mode()
def test_softcap_rounds_into_original_logits_without_touching_padding(dtype, rows, cap):
    from vllm.models.beam.nvidia.activation import beam_softcap_logits

    torch.manual_seed(42)
    storage = torch.randn(rows, 264, device="cuda", dtype=dtype) * (cap * 3)
    logits = storage[:, :257]
    if rows:
        logits[0, :6] = torch.tensor(
            [0.0, -0.0, float("inf"), -float("inf"), float("nan"), 1e-38],
            device=logits.device,
            dtype=dtype,
        )
    original = storage.clone()
    expected = (torch.tanh(logits.float() / cap) * cap).to(dtype)
    actual = beam_softcap_logits(logits, cap)

    assert actual is logits
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(
        storage[:, 257:], original[:, 257:], rtol=0, atol=0, equal_nan=True
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "rows,width", [(0, 2560), (1, 5120), (8, 2560), (33, 513), (2048, 2560)]
)
@torch.inference_mode()
def test_dense_activation_rounds_after_product(default_vllm_config, dtype, rows, width):
    """BF16 I/O must not round the SiLU result before multiplication."""
    import vllm._custom_ops  # noqa: F401
    from vllm.models.beam.nvidia.activation import beam_silu_mul

    torch.manual_seed(42)
    x = torch.randn(rows, 2 * width, device="cuda", dtype=dtype) * 4
    expected = torch.empty(rows, width, device="cuda", dtype=torch.float32)
    if rows:
        torch.ops._C.silu_and_mul(expected, x.float())
    actual = beam_silu_mul(x)
    torch.testing.assert_close(
        actual,
        expected.to(dtype),
        rtol=0.008 if dtype == torch.bfloat16 else 1e-6,
        atol=1e-7,
    )
    if rows:
        error = (actual.float() - expected.to(dtype).float()).norm()
        assert error <= expected.norm() * 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("has_weight", [False, True])
@pytest.mark.parametrize("shape", [(33, 513), (17, 6, 128)])
@torch.inference_mode()
def test_fused_norm_preserves_fp32_affine_for_strided_rows(dtype, has_weight, shape):
    """QKV slices and near-epsilon values retain the reference's FP32 affine."""
    from vllm.models.beam.nvidia.layernorm import beam_rms_norm

    generator = torch.Generator(device="cuda").manual_seed(42)
    storage = torch.randn(
        (*shape[:-1], shape[-1] + 64),
        device="cuda",
        dtype=dtype,
        generator=generator,
    )
    x = storage[..., 1 : shape[-1] + 1]
    x[::2].mul_(1e-3)
    weight = (
        torch.randn(shape[-1], device="cuda", dtype=dtype, generator=generator)
        if has_weight
        else None
    )
    expected = _torch_rms_norm(x, weight, 1e-6)
    actual = beam_rms_norm(x, weight, 1e-6)
    # FP32 reduction trees may differ; allow one BF16 ULP after final rounding.
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.008 if dtype == torch.bfloat16 else 1e-6,
        atol=1e-7,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("shape", [(0, 5120), (33, 513)])
@pytest.mark.parametrize(
    "post,next_,scaled",
    [(True, True, True), (True, True, False), (False, True, True), (True, False, True)],
)
@torch.inference_mode()
def test_norm_residual_norm_preserves_intermediate_rounding(
    dtype, shape, post, next_, scaled
):
    """Both residual and next input must retain BF16 casts between fused stages."""
    from vllm.models.beam.nvidia.layernorm import beam_norm_residual_norm

    torch.manual_seed(42)
    x = torch.randn(shape[0], shape[1] + 32, device="cuda", dtype=dtype)[
        :, 1 : shape[1] + 1
    ]
    residual = torch.randn_like(x)
    x[::2].mul_(1e-3)
    post_weight = torch.randn(shape[1], device="cuda", dtype=dtype) if post else None
    next_weight = torch.randn(shape[1], device="cuda", dtype=dtype) if next_ else None
    scales = (0.125, 0.9682458365518543) if scaled else None
    original_x, original_residual = x.clone(), residual.clone()
    args = (
        x,
        residual,
        post_weight,
        1e-6 if post else None,
        next_weight,
        1e-6 if next_ else None,
        scales,
    )
    expected = _torch_boundary(*args)
    actual = beam_norm_residual_norm(*args)
    _assert_boundary_close(actual, expected)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(residual, original_residual, rtol=0, atol=0)


def _separate_qk_norm_rope(
    q,
    k,
    qw,
    kw,
    eps,
    positions,
    cache,
    head_dim,
    rotary_dim=0,
    neox=True,
    mode="none",
    offset=1,
    divisor=1.0,
):
    from vllm import _custom_ops as ops

    q = _torch_rms_norm(
        q.reshape(-1, q.shape[-1] // head_dim, head_dim), qw, eps
    ).reshape_as(q)
    k = _torch_rms_norm(
        k.reshape(-1, k.shape[-1] // head_dim, head_dim), kw, eps
    ).reshape_as(k)
    if rotary_dim and q.shape[0]:
        qf, kf = q.float(), k.float()
        ops.rotary_embedding(positions, qf, kf, head_dim, cache, neox)
        q, k = qf.to(q.dtype), kf.to(k.dtype)
    elif mode != "none":
        p = positions.float() + offset
        if mode == "default":
            scale = p.log() / divisor
        elif mode == "root10":
            scale = (0.1 * torch.log1p(p / 2048)).exp()
        else:
            scale = (130 * torch.expm1(0.001 * p.log())).clamp(min=1.0)
        q = (q * scale[:, None]).to(q.dtype)
    return q, k


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("rows", [0, 1, 17, 512, 2049])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("floor,temperature", [(0.13, 0.7), (0.0, 1.0)])
@torch.inference_mode()
def test_output_gate_preserves_activation_dtype_rounding(
    dtype, rows, bias, packed, floor, temperature
):
    from vllm.models.beam.nvidia.attention_ops import beam_output_gate

    torch.manual_seed(42)
    x = torch.randn(rows, 768, device="cuda", dtype=dtype)
    gate = torch.randn(rows, 1792 if packed else 768, device="cuda", dtype=dtype) * 8
    gate = gate[:, -768:]
    weight = torch.randn(6, device="cuda", dtype=dtype) if bias else None
    kwargs = dict(num_heads=6, head_dim=128, floor=floor, temperature=temperature)
    # Preserve each activation-dtype operation in the parent implementation.
    g = gate.view(rows, 6, 128)
    if weight is not None:
        g = g + weight[None, :, None]
    expected = x * (
        floor + (1 - floor) * torch.sigmoid(g.reshape_as(gate) / temperature)
    )
    actual = beam_output_gate(x, gate, weight, **kwargs)
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-7)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "rotary_dim,neox,weighted", [(0, True, False), (128, True, True), (64, False, True)]
)
@pytest.mark.parametrize("head_dim,q_heads,k_heads", [(128, 6, 1), (80, 4, 2)])
@torch.inference_mode()
def test_qk_norm_rope_keeps_norm_rounding_and_fp32_rotation(
    dtype, rotary_dim, neox, weighted, head_dim, q_heads, k_heads
):
    from vllm.models.beam.nvidia.attention_ops import beam_qk_norm_rope

    torch.manual_seed(42)
    rows = 17
    rotary_dim = min(rotary_dim, head_dim)
    sizes = (q_heads * head_dim, k_heads * head_dim, k_heads * head_dim)
    qkv = torch.randn(rows, sum(sizes) + 1, device="cuda", dtype=dtype)[:, 1:]
    q, k, _ = qkv.split(sizes, -1)
    q[::2].mul_(1e-3)
    qw = torch.randn(head_dim, device="cuda", dtype=dtype) if weighted else None
    kw = torch.randn(head_dim, device="cuda", dtype=dtype) if weighted else None
    positions = torch.arange(rows, device="cuda") * 3
    angle = torch.randn(max(1, rows * 3), rotary_dim // 2, device="cuda")
    cache = torch.cat((angle.cos(), angle.sin()), -1)
    args = (q, k, qw, kw, 1e-6, positions, cache, head_dim, rotary_dim, neox)
    expected = _separate_qk_norm_rope(*args)
    actual = beam_qk_norm_rope(*args)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(
            value, reference, rtol=0.016 if dtype == torch.bfloat16 else 2e-5, atol=2e-6
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 8, 513, 2049])
@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("nhd", [False, True])
@pytest.mark.parametrize("rotary_dim,neox", [(128, True), (64, False), (0, True)])
@torch.inference_mode()
def test_qk_norm_rope_inserts_only_mapped_kv(rows, kv_heads, nhd, rotary_dim, neox):
    """Cache insertion preserves Q/K and follows live slot mappings on replay."""
    from vllm.models.beam.nvidia.attention_ops import beam_qk_norm_rope

    torch.manual_seed(42)
    sizes = (6 * kv_heads * 128, kv_heads * 128, kv_heads * 128)
    qkv = torch.randn(rows, sum(sizes) + 1, device="cuda", dtype=torch.bfloat16)[:, 1:]
    q, k, v = qkv.split(sizes, -1)
    weight = torch.randn(128, device="cuda")
    positions = torch.arange(rows, device="cuda")
    rope = torch.randn(rows, rotary_dim, device="cuda") if rotary_dim else None
    args = (q, k, weight, weight, 1e-6, positions, rope, 128, rotary_dim, neox)
    pages = (rows + 127) // 128 + 1
    shape = (pages, 128, kv_heads, 256) if nhd else (pages, kv_heads, 128, 256)
    cache = torch.zeros(shape, device="cuda", dtype=q.dtype)
    if nhd:
        cache = cache.transpose(1, 2)
    live = max(1, rows - 1)
    slots = torch.randperm(pages * 128, device="cuda")[:live].clone()
    if live > 1:
        slots[-1] = -1

    def fused():
        return beam_qk_norm_rope(
            *args, ssmax_mode="root10", value=v, kv_cache=cache, slot_mapping=slots
        )

    fused()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = fused()
    # Mutate both data and metadata after capture, including a previously skipped slot.
    qkv.add_(0.125)
    slots.copy_(torch.randperm(pages * 128, device="cuda")[:live])
    if live > 1:
        slots[0] = -1
    cache.zero_()
    graph.replay()
    expected = beam_qk_norm_rope(*args, ssmax_mode="root10")
    for result, reference in zip(actual, expected):
        torch.testing.assert_close(result, reference, rtol=0, atol=0)
    valid = slots >= 0
    mapped = slots[valid]
    reference_cache = torch.zeros_like(cache)
    reference_cache[mapped // 128, :, mapped % 128, :128] = expected[1][:live][
        valid
    ].reshape(-1, kv_heads, 128)
    reference_cache[mapped // 128, :, mapped % 128, 128:] = v[:live][valid].reshape(
        -1, kv_heads, 128
    )
    torch.testing.assert_close(cache, reference_cache, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "mode,offset,divisor",
    [
        ("default", 1, 1.0),
        ("default", 16, 2.772588722239781),
        ("root10", 1, 1.0),
        ("root1000", 1, 1.0),
    ],
)
@torch.inference_mode()
def test_ssmax_fuses_after_norm_rounding_and_scales_only_queries(
    dtype, mode, offset, divisor
):
    from vllm.models.beam.nvidia.attention_ops import beam_qk_norm_rope

    torch.manual_seed(42)
    positions = torch.tensor(
        [0, 1, 15, 2047, 2048, 16384, 100000, 1048575], device="cuda"
    )
    qkv = torch.randn(len(positions), 1025, device="cuda", dtype=dtype)[:, 1:]
    q, k, _ = qkv.split((768, 128, 128), -1)
    qw, kw = [torch.randn(128, device="cuda", dtype=dtype) for _ in range(2)]
    args = (q, k, qw, kw, 1e-6, positions, None, 128, 0, True, mode, offset, divisor)
    actual, expected = beam_qk_norm_rope(*args), _separate_qk_norm_rope(*args)
    for value, reference in zip(actual, expected):
        # Norm and position scaling each round to BF16; allow two rounding steps.
        torch.testing.assert_close(
            value,
            reference,
            rtol=0.016 if dtype == torch.bfloat16 else 2e-6,
            atol=2e-6,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rotary_dim", [0, 128])
@torch.inference_mode()
def test_attention_fusions_replay_with_changed_positions_and_gates(rotary_dim):
    from vllm.models.beam.nvidia.attention_ops import (
        beam_output_gate,
        beam_qk_norm_rope,
    )

    torch.manual_seed(42)
    qkv = torch.randn(17, 1025, device="cuda", dtype=torch.bfloat16)[:, 1:]
    q, k, _ = qkv.split((768, 128, 128), -1)
    positions = torch.arange(17, device="cuda")
    angle = torch.randn(128, 64, device="cuda")
    cache = torch.cat((angle.cos(), angle.sin()), -1)
    gate = torch.randn_like(q)

    def run():
        nq, nk = beam_qk_norm_rope(
            q,
            k,
            None,
            None,
            1e-6,
            positions,
            cache,
            128,
            rotary_dim,
            ssmax_mode="root10",
        )
        return beam_output_gate(
            nq, gate, None, num_heads=6, head_dim=128, floor=0.13, temperature=0.7
        ), nk

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = run()
    graph.replay()
    torch.accelerator.synchronize()
    positions.add_(31)
    gate.neg_()
    qkv.mul_(0.5)
    expected = run()
    graph.replay()
    torch.accelerator.synchronize()
    for actual, reference in zip(captured, expected):
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.skipif(not torch.accelerator.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rotary_dim", [0, 128])
@pytest.mark.parametrize("insert_kv", [False, True])
@torch.inference_mode()
def test_fused_kernels_reuse_specializations_across_token_counts(
    monkeypatch, rotary_dim, insert_kv
):
    """Warm tile buckets once, then vary batch sizes, positions offsets and scales."""
    from triton import knobs

    from vllm.models.beam.nvidia.activation import beam_softcap_logits
    from vllm.models.beam.nvidia.attention_ops import (
        beam_output_gate,
        beam_qk_norm_rope,
    )
    from vllm.models.beam.nvidia.layernorm import beam_norm_residual_norm, beam_rms_norm

    device = "cuda"
    dtype = torch.bfloat16
    rows = 8193
    qkv = torch.randn(rows, 1024, device=device, dtype=dtype)
    kv_cache = torch.zeros((65, 1, 128, 256), device=device, dtype=dtype)
    slots = torch.arange(rows, device=device)
    positions = torch.arange(rows + 16, device=device)
    cache = torch.randn(rows + 16, 128, device=device)
    weight = torch.ones(128, device=device, dtype=dtype)
    bias = torch.zeros(6, device=device, dtype=dtype)
    x = torch.randn(rows, 5120, device=device, dtype=dtype)
    residual = torch.randn_like(x)
    norm_weight = torch.ones(5120, device=device, dtype=dtype)

    def run(tokens, offset, scales):
        q, k = beam_qk_norm_rope(
            qkv[:tokens, :768],
            qkv[:tokens, 768:896],
            weight,
            weight,
            1e-6,
            positions[offset : offset + tokens],
            cache if rotary_dim else None,
            128,
            rotary_dim,
            ssmax_mode="none" if rotary_dim else "root10",
            value=qkv[:tokens, 896:] if insert_kv else None,
            kv_cache=kv_cache if insert_kv else None,
            slot_mapping=slots[: tokens - 1] if insert_kv else None,
        )
        beam_output_gate(
            q, q, bias, num_heads=6, head_dim=128, floor=0.1, temperature=1.2
        )
        beam_rms_norm(x[:tokens], norm_weight, 1e-6)
        beam_softcap_logits(x[:tokens], 30.0)
        beam_norm_residual_norm(
            x[:tokens],
            residual[:tokens],
            norm_weight,
            1e-6,
            norm_weight,
            1e-6,
            scales,
        )

    for tokens in (1, 512, 2049):
        run(tokens, 0, (0.125, 0.9682458365518543))

    def unexpected_compile(**kwargs):
        if kwargs["fn"].module.startswith("vllm.models.beam.nvidia"):
            pytest.fail(f"Unexpected shape specialization: {kwargs['fn'].name}")

    with monkeypatch.context() as patch:
        patch.setattr(knobs.runtime, "jit_cache_hook", unexpected_compile)
        token_counts = (2, 7, 17, 511, 513, 2048, 2050, 8193)
        for i, tokens in enumerate(token_counts):
            run(tokens, i % 16, (0.125 + i / 1000, 0.968 - i / 1000))
    torch.accelerator.synchronize()


def _beam_allreduce_worker(rank, world_size, port, hidden_size, host="127.0.0.1"):
    from datetime import timedelta

    import torch.distributed as dist

    from vllm.models.beam.nvidia.allreduce import (
        BeamAllReduceWorkspace,
        beam_allreduce_norm_residual_norm,
    )

    torch.accelerator.set_device_index(rank % torch.accelerator.device_count())
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://{host}:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    group = dist.new_group(backend="gloo")
    separate_cutoff = (10 * 1024**2 * world_size + 2 * hidden_size - 1) // (
        2 * hidden_size
    )
    max_tokens = max(513, separate_cutoff + 1)
    workspace = BeamAllReduceWorkspace(group, hidden_size, max_tokens)
    torch.manual_seed(42)
    residual = torch.randn(max_tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
    torch.manual_seed(43 + rank)
    x = torch.randn_like(residual)
    # Exercise shared Lamport generations across batch and implementation changes.
    cutoff = workspace.small_batch_max_tokens
    batches = [0, 1, cutoff, cutoff + 1, 513, 1]
    if workspace.separate_norm_min_tokens is not None:
        cutoff = workspace.separate_norm_min_tokens
        batches += [cutoff - 1, cutoff, cutoff + 1, 1]
    for iteration, tokens in enumerate(batches):
        torch.manual_seed(42 + iteration)
        residual.normal_()
        weight.normal_()
        torch.manual_seed(43 + iteration + rank)
        x.normal_()
        x[:, ::7] = -0.0
        post, next_norm = iteration % 3 != 0, iteration % 3 != 1
        scales = (0.125, 0.9682458365518543) if iteration % 2 else None
        args = (
            x[:tokens],
            residual[:tokens],
            weight if post else None,
            1e-6 if post else None,
            weight if next_norm else None,
            1e-6 if next_norm else None,
            scales,
        )

        def check(result, args):
            partial, r, pw, peps, nw, neps, scales = args
            # The collective rounds the FP32 sum before the post-norm.
            reduced = partial.float()
            dist.all_reduce(reduced)
            reduced = reduced.bfloat16()
            expected = _torch_boundary(reduced, r, pw, peps, nw, neps, scales)
            _assert_boundary_close(result, expected, relative_tolerance=1e-3)

        check(beam_allreduce_norm_residual_norm(*args, workspace), args)
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = beam_allreduce_norm_residual_norm(*args, workspace)
        for _ in range(2):
            x.mul_(0.5)
            graph.replay()
            check(captured, args)
        graph.reset()
    # A small configured capacity still needs enough space for forced one-shot.
    cutoff = workspace.small_batch_max_tokens
    small_workspace = BeamAllReduceWorkspace(group, hidden_size, cutoff)
    args = (x[:cutoff], residual[:cutoff], weight, 1e-6, weight, 1e-6, None)
    check(beam_allreduce_norm_residual_norm(*args, small_workspace), args)
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,hidden_size", [(2, 1024), (2, 8192), (4, 5120), (8, 5120)]
)
def test_allreduce_boundary_handles_batch_changes_and_graph_replay(
    world_size, hidden_size
):
    """Changing token counts must not consume stale Lamport slots or deadlock."""
    from torch.multiprocessing import spawn

    from vllm.utils.network_utils import get_open_port

    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs")
    mnnvl = pytest.importorskip("flashinfer.comm.mnnvl")
    if not mnnvl.is_multicast_supported(0):
        pytest.skip("NVLink multicast required")
    spawn(
        _beam_allreduce_worker,
        args=(world_size, get_open_port(), hidden_size),
        nprocs=world_size,
    )


def _beam_finalize_worker(rank, world_size, port, expert_parallel, hidden_size):
    import torch.distributed as dist

    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.models.beam.nvidia.allreduce import (
        BeamAllReduceWorkspace,
        beam_allreduce_norm_residual_norm,
        beam_finalize_allreduce_norm_residual_norm,
    )

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    group = dist.new_group(backend="gloo")
    n, k = hidden_size, 8
    workspace = BeamAllReduceWorkspace(group, n, 4095)
    if expert_parallel:
        cutoff = workspace.ep_finalize_oneshot_max_tokens
        batches = [0, 1, cutoff, cutoff + 1, 513]
        if hidden_size == 5120:
            batches.append(4095)
        batches.append(2)
    else:
        batches = [0, 1, 127, 513, 2]
    for iteration, m in enumerate(batches):
        torch.manual_seed(100 + iteration)
        residual = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, device="cuda", dtype=torch.bfloat16)
        owners = torch.randint(world_size, (m, k), device="cuda")
        owners[::3] = torch.arange(m, device="cuda")[::3, None] % world_size
        shared_weights = torch.rand(m, k, device="cuda") / k
        torch.manual_seed(200 + rank + iteration)
        experts = torch.randn(m * k, n, device="cuda", dtype=torch.bfloat16)
        weights = (
            shared_weights if expert_parallel else torch.rand(m, k, device="cuda") / k
        )
        if iteration % 2:
            weights = weights.bfloat16()
        permutation = torch.randperm(m * k, device="cuda", dtype=torch.int32).view(m, k)
        if expert_parallel:
            permutation.masked_fill_(owners != rank, -1)
        else:
            permutation[::3, -1] = -1
        x = UnfinalizedMoEOutput(experts, weights, permutation)
        routed_scale = 2.5 if iteration % 2 else 1.0
        post = iteration % 3 != 0
        next_norm = iteration % 3 != 1
        scales = (0.125, 0.9682458365518543)

        args = (
            x,
            residual,
            weight if post else None,
            1e-6 if post else None,
            weight if next_norm else None,
            1e-6 if next_norm else None,
            scales,
            workspace,
            routed_scale,
        )

        def run(args=args):
            return beam_finalize_allreduce_norm_residual_norm(
                *args,
                oneshot_max_tokens=(
                    workspace.ep_finalize_oneshot_max_tokens if expert_parallel else 0
                ),
            )

        def check(actual, args=args):
            x, residual, pw, peps, nw, neps, scales, _, routed_scale = args
            m, n = residual.shape
            local = torch.zeros(m, n, device="cuda")
            for expert in range(x.expert_weights.shape[1]):
                rows = x.expanded_idx_to_permuted_idx[:, expert].long()
                values = x.gemm2_permuted[rows.clamp_min(0)].float()
                values = torch.where(rows[:, None] >= 0, values, 0)
                local = local + values * x.expert_weights[:, expert, None].float()
            local = (local.bfloat16().float() * routed_scale).bfloat16().float()
            dist.all_reduce(local)
            expected = _torch_boundary(
                local.bfloat16(),
                residual,
                pw,
                peps,
                nw,
                neps,
                scales,
            )
            _assert_boundary_close(actual, expected)

        check(run())
        if expert_parallel and m == workspace.ep_finalize_oneshot_max_tokens:
            tight_workspace = BeamAllReduceWorkspace(group, n, m)
            args = (*args[:-2], tight_workspace, routed_scale)
            check(run(args), args)
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run(args)
        for _ in range(2):
            experts.mul_(0.5)
            if expert_parallel:
                weights.mul_(0.75)
                permutation.copy_(permutation.roll(1, dims=0))
            graph.replay()
            check(captured, args)
            # Alternate two-stage finalization with the one-stage decode path
            # on the same Lamport buffers, including during graph replay.
            small = residual[:8]
            actual = beam_allreduce_norm_residual_norm(
                small, small, weight, 1e-6, weight, 1e-6, scales, args[-2]
            )
            reduced = small.float()
            dist.all_reduce(reduced)
            expected = _torch_boundary(
                reduced.bfloat16(), small, weight, 1e-6, weight, 1e-6, scales
            )
            _assert_boundary_close(actual, expected)
        graph.reset()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize(
    "world_size,expert_parallel,hidden_size",
    [
        (2, False, 5120),
        (4, False, 5120),
        (4, True, 5120),
        (2, True, 1024),
        (4, True, 2048),
        (4, True, 8192),
    ],
)
def test_moe_finalize_allreduce_preserves_local_rounding_and_replay(
    world_size, expert_parallel, hidden_size
):
    """Empty local routes and live routing metadata retain BF16 casts on replay."""
    from torch.multiprocessing import spawn

    from vllm.utils.network_utils import get_open_port

    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs")
    mnnvl = pytest.importorskip("flashinfer.comm.mnnvl")
    if not mnnvl.is_multicast_supported(0):
        pytest.skip("NVLink multicast required")
    spawn(
        _beam_finalize_worker,
        args=(world_size, get_open_port(), expert_parallel, hidden_size),
        nprocs=world_size,
    )


def _beam_moe_gather_worker(rank, world_size, port, hidden_size):
    from datetime import timedelta

    import torch.distributed as dist

    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput
    from vllm.models.beam.nvidia.allreduce import (
        BeamAllReduceWorkspace,
        beam_allreduce_norm_residual_norm,
    )
    from vllm.models.beam.nvidia.moe_finalize import (
        beam_moe_finalize_gather_norm_residual_norm,
    )

    torch.accelerator.set_device_index(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=120),
    )
    group = dist.new_group(backend="gloo")
    workspace = BeamAllReduceWorkspace(group, hidden_size, 513)

    def make_case(tokens, seed, *, post, next_norm, scales):
        local_rows = (tokens + world_size - 1) // world_size
        valid_rows = max(0, min(local_rows, tokens - rank * local_rows))
        torch.manual_seed(seed)
        residual = torch.randn(tokens, hidden_size, device="cuda", dtype=torch.bfloat16)
        affine = torch.randn(hidden_size, device="cuda", dtype=torch.bfloat16)
        torch.manual_seed(seed + rank + 1)
        experts = torch.randn(
            local_rows + 3, 8, hidden_size, device="cuda", dtype=torch.bfloat16
        )
        weights = torch.rand(local_rows + 3, 8, device="cuda")
        permutation = torch.arange(
            (local_rows + 3) * 8, device="cuda", dtype=torch.int32
        ).view(local_rows + 3, 8)
        experts[:valid_rows:2].mul_(1e-3)
        experts[:valid_rows, :, ::97] = -0.0
        experts[:valid_rows, :, 1::97] = -1e-40
        # Both capacity padding and SP padding must stay outside live outputs.
        experts[valid_rows:] = float("nan")
        weights[valid_rows:] = float("nan")
        return (
            UnfinalizedMoEOutput(experts.flatten(0, 1), weights, permutation),
            residual,
            affine if post == "weighted" else None,
            None if post == "none" else 1e-6,
            affine if next_norm == "weighted" else None,
            None if next_norm == "none" else 1e-6,
            scales,
            workspace,
            0.137,
        )

    def check(actual, args):
        output, residual, pw, peps, nw, neps, scales, _, scale = args
        tokens = residual.shape[0]
        local_rows = (tokens + world_size - 1) // world_size
        valid_rows = max(0, min(local_rows, tokens - rank * local_rows))
        weights = output.expert_weights[:valid_rows]
        experts = output.gemm2_permuted.view(-1, 8, hidden_size)[:valid_rows]
        local = torch.zeros(
            local_rows, hidden_size, dtype=residual.dtype, device=residual.device
        )
        local[:valid_rows] = _torch_megamoe_finalize(experts, weights, scale, pw, peps)
        gathered = torch.empty(
            world_size * local_rows,
            hidden_size,
            dtype=local.dtype,
            device=local.device,
        )
        if tokens:
            dist.all_gather_into_tensor(gathered, local)
        expected = _torch_boundary(
            gathered[:tokens], residual, None, None, nw, neps, scales
        )
        _assert_boundary_close(actual, expected, relative_tolerance=1e-3)

    batches = (0, 1, world_size + 1, 513)
    norms = ("none", "unweighted", "weighted")
    for iteration, tokens in enumerate(batches):
        args = make_case(
            tokens,
            200 + iteration,
            post=norms[iteration % 3],
            next_norm=norms[(iteration + 1) % 3],
            scales=(0.125, 0.9682458365518543) if iteration % 2 else None,
        )
        for strategy in ("unfused", "oneshot", "twoshot", None):
            check(
                beam_moe_finalize_gather_norm_residual_norm(*args, strategy=strategy),
                args,
            )

    # Independent captured graphs share the Lamport generation state. Replaying
    # one then two stages must work after eager collectives and changed inputs.
    graphs = []
    for strategy, tokens in (
        ("oneshot", world_size - 1),
        ("oneshot", 17),
        ("twoshot", world_size - 1),
        ("twoshot", 129),
    ):
        args = make_case(
            tokens,
            300 + tokens,
            post="weighted",
            next_norm="weighted",
            scales=(0.125, 0.9682458365518543),
        )
        beam_moe_finalize_gather_norm_residual_norm(*args, strategy=strategy)
        torch.accelerator.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = beam_moe_finalize_gather_norm_residual_norm(
                *args, strategy=strategy
            )
        graphs.append((graph, actual, args))
    # Without a reference collective between replays, empty ranks may lag more
    # than a Lamport generation. Padded rows must still acknowledge each rank.
    for graph, actual, args in graphs:
        if rank == world_size - 1:
            torch.cuda._sleep(2_000_000)
        for _ in range(16):
            graph.replay()
        check(actual, args)
    for iteration in range(2):
        for graph, actual, args in graphs[:: 1 if iteration % 2 else -1]:
            output = args[0]
            output.gemm2_permuted.mul_(0.75)
            output.expert_weights.copy_(output.expert_weights.roll(1, dims=1))
            args[1].mul_(0.5)
            args[2].neg_()
            graph.replay()
            check(actual, args)
            # Attention AR and MoE gather reuse this workspace in production.
            residual, weight = args[1], args[2]
            x = residual * (rank + 1)
            ar_args = (x, residual, weight, 1e-6, weight, 1e-6, None, workspace)
            actual_ar = beam_allreduce_norm_residual_norm(*ar_args)
            reduced = x.float()
            dist.all_reduce(reduced)
            _assert_boundary_close(
                actual_ar,
                _torch_boundary(
                    reduced.bfloat16(), residual, weight, 1e-6, weight, 1e-6, None
                ),
                relative_tolerance=1e-3,
            )
    torch.accelerator.synchronize()
    for graph, _, _ in graphs:
        graph.reset()
    dist.barrier()
    dist.destroy_process_group()


@pytest.mark.parametrize("world_size,hidden_size", [(2, 8192), (4, 5120)])
def test_megamoe_gather_restores_tokens_and_alternates_graph_strategies(
    world_size, hidden_size
):
    """Keep SP rank order, BF16 casts, and generation state across all strategies."""
    from torch.multiprocessing import spawn

    from vllm.utils.network_utils import get_open_port

    if torch.accelerator.device_count() < world_size:
        pytest.skip(f"Requires {world_size} GPUs")
    mnnvl = pytest.importorskip("flashinfer.comm.mnnvl")
    if not mnnvl.is_multicast_supported(0):
        pytest.skip("NVLink multicast required")
    spawn(
        _beam_moe_gather_worker,
        args=(world_size, get_open_port(), hidden_size),
        nprocs=world_size,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("rows", [1, 2, 4, 8, 12, 16, 24, 64])
@pytest.mark.parametrize("qkv_rows,gate_rows", [(2048, 1536), (1024, 768)])
@torch.inference_mode()
def test_packed_attention_projection(rows, qkv_rows, gate_rows):
    """The shared GEMMs preserve QKV/gate ordering and BF16 output accuracy."""
    from vllm.models.beam.nvidia.projection import (
        packed_projection,
        use_low_latency_gemm,
    )

    torch.manual_seed(42)
    x = torch.randn(rows, 5120, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(qkv_rows + gate_rows, 5120, device="cuda", dtype=x.dtype) * 0.01
    packed = packed_projection(x, w, use_low_latency_gemm(w))
    for actual, weight in zip(
        packed.split([qkv_rows, gate_rows], -1),
        w.split([qkv_rows, gate_rows]),
    ):
        expected = (x.double() @ weight.double().T).to(x.dtype)
        torch.testing.assert_close(actual, expected, rtol=0.008, atol=1e-5)
