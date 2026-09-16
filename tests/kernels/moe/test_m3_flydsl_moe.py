# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL MoE (gfx950, ``vllm/models/minimax_m3/amd/ops/moe_mxfp8``): the
decode / mid / prefill chains against aiter's a8w8 call and a float reference on
dequantized MXFP8 experts, the package dispatch, and the experts class the MXFP8
oracle selects. Routing either mimics aiter's fused shared expert (last expert,
every token, weight 1) or is the routed-only top-k of ModelOpt MXFP8.
"""

from dataclasses import dataclass

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950
from vllm.utils.torch_utils import set_random_seed

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only"),
    pytest.mark.skipif(not on_gfx950(), reason="gfx950 only"),
]

HIDDEN, INTER, NUM_ROUTED, TOPK = 6144, 768, 128, 4
NUM_EXPERTS = NUM_ROUTED + 1
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0
CHECK_TOKENS = 16
# Translate the original angular budgets via ||a-b||/||b|| = sqrt(2*(1-c))
# for equal norms. Actual relative L2 also penalizes magnitude errors. These
# are empirical layer regression limits; the reducer test below uses a bound.
MATCHED_ERROR = (2 * (1 - 0.9999)) ** 0.5
QUANTIZED_ERROR = (2 * (1 - 0.998)) ** 0.5
FP8_PARTIAL_ERROR = (2 * (1 - 0.997)) ** 0.5


def _quantize_mxfp8(w: torch.Tensor):
    """bf16 [E, N, K] -> (fp8 [E, N, K], e8m0 uint8 [E, N, K/32])."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    e, n, k = w.shape
    q, s = per_1x32_mx_quant_hip(
        w.reshape(e * n, k), quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    return q.view(e, n, k), s.view(torch.uint8).view(e, n, k // 32)


def _routing(m: int, device, shared: bool = True):
    """Top-k over the routed experts (weights renormalized x2) plus, with
    ``shared``, aiter's fused shared expert: the last expert with weight 1."""
    routed = torch.rand((m, NUM_ROUTED), device=device).topk(TOPK, dim=1).indices
    w = torch.rand((m, TOPK), device=device)
    w = w / w.sum(dim=1, keepdim=True) * 2.0
    if shared:
        routed = torch.cat([routed, torch.full((m, 1), NUM_ROUTED, device=device)], 1)
        w = torch.cat([w, torch.ones((m, 1), device=device)], dim=1)
    return routed.to(torch.int32).contiguous(), w.to(torch.float32).contiguous()


def _gamma(n: int, dtype=torch.bfloat16) -> float:
    u = torch.finfo(dtype).eps / 2
    return n * u / (1 - n * u)


def _sample_rows(m: int, count: int = CHECK_TOKENS) -> list[int]:
    """Keep endpoints and evenly spaced deterministic rows, without duplicates."""
    return torch.linspace(0, m - 1, min(m, count), dtype=torch.float64).long().tolist()


def _dequantize(q: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    scale = torch.ldexp(
        torch.ones_like(scales, dtype=torch.float32), scales.int() - 127
    )
    return (q.float().unflatten(-1, (-1, 32)) * scale.unsqueeze(-1)).flatten(-2)


def _mxfp8_roundtrip(x: torch.Tensor) -> torch.Tensor:
    """RNE E4M3, using the kernel's FP32 ceil-power-of-two scale algorithm."""
    groups = x.float().unflatten(-1, (-1, 32))
    bits = (groups.abs().amax(-1, keepdim=True) * (1.0 / 448.0)).view(torch.int32)
    exponent = ((bits >> 23) & 255) + (((bits & 0x7FFFFF) != 0) & ((bits >> 23) < 255))
    exponent = torch.where(bits == 0, 127, exponent)
    scale = torch.ldexp(torch.ones_like(groups[..., :1]), exponent - 127)
    quantized = (groups / scale).to(torch.float8_e4m3fn).float()
    return (quantized * scale).flatten(-2)


def _activation(h: torch.Tensor) -> torch.Tensor:
    g, up = h.chunk(2, dim=-1)
    g, up = g.clamp(max=7.0), up.clamp(-7.0, 7.0)
    # Preserve the two multiplications used by the new epilogues.
    sigmoid = torch.reciprocal(1.0 + torch.exp2((g * 1.702) * -1.4426950408889634))
    return (g * sigmoid) * (up + 1.0)


@dataclass
class Reference:
    value: torch.Tensor
    float_value: torch.Tensor


def _reference(x, raw, ids, weights, tokens, *, mode="decode") -> Reference:
    """Compute an independent reference, grouping selected rows by expert.

    Modes: decode, mid, bf16 (prefill), fp8 (prefill). Centers are FP32 sums
    before final output rounding. Decode models three down-projection partials
    per expert through M=64, otherwise one. All inputs must be finite, with no
    FP32 scaling overflow/underflow; zero blocks are supported.
    """
    w13q, s13, w2q, s2 = raw
    x_ref = x[tokens].float()
    selected_ids, selected_weights = ids[tokens], weights[tokens].float()
    x_quant = x_ref if mode == "decode" else _mxfp8_roundtrip(x_ref)
    split = 3 if mode == "decode" and x.shape[0] <= 64 else 1
    value = torch.zeros_like(x_ref)
    float_value = torch.zeros_like(x_ref)
    # Reference accumulation in FP64 avoids order-dependent reference output.
    accum = value.double()
    float_accum = float_value.double()
    for expert in selected_ids.unique().tolist():
        rows, slots = torch.where(selected_ids == expert)
        route = selected_weights[rows, slots, None]
        w13 = _dequantize(w13q[expert], s13[expert])
        w2 = _dequantize(w2q[expert], s2[expert])
        a_float = _activation(x_ref[rows] @ w13.T)
        if mode == "decode":
            a_float = a_float.bfloat16().float()
            a_quant = a_float
        else:
            a_quant = _mxfp8_roundtrip(_activation(x_quant[rows] @ w13.T))
        float_accum.index_add_(0, rows, (route * (a_float @ w2.T)).double())
        for a_slice, w_slice in zip(a_quant.chunk(split, -1), w2.chunk(split, -1)):
            partial = a_slice @ w_slice.T
            if mode == "fp8":
                partial = _mxfp8_roundtrip(partial)
            partial = route * partial
            accum.index_add_(0, rows, partial.double())
    value = accum.float()
    return Reference(value, float_accum.float())


@pytest.fixture(scope="module")
def m3_weights():
    from vllm._aiter_ops import rocm_aiter_ops

    set_random_seed(0)
    device = torch.device("cuda")
    w13 = torch.randn(
        (NUM_EXPERTS, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device=device
    )
    w2 = torch.randn((NUM_EXPERTS, HIDDEN, INTER), dtype=torch.bfloat16, device=device)
    w13_q, w13_s = _quantize_mxfp8(w13 * 0.02)
    w2_q, w2_s = _quantize_mxfp8(w2 * 0.02)
    raw = (w13_q, w13_s, w2_q, w2_s)
    return raw, rocm_aiter_ops.shuffle_mxfp8_moe_weights(w13_q, w2_q, w13_s, w2_s)


def _run_aiter(shuffled, x, topk_ids, topk_weights):
    """The production call (``AiterMxfp8Experts.apply``)."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    w13, w2, w13_s, w2_s = shuffled
    return fused_moe(
        x,
        w13,
        w2,
        topk_weights,
        topk_ids,
        activation=ActivationType.Swiglu,
        quant_type=QuantType.per_1x32,
        w1_scale=w13_s,
        w2_scale=w2_s,
        gate_mode=GateMode.INTERLEAVE.value,
        swiglu_limit=SWIGLU_LIMIT,
    )


def _kw(shuffled):
    return dict(
        hidden_size=HIDDEN, intermediate_size=INTER, num_experts=shuffled[0].shape[0]
    )


def _decode(shuffled, x, topk_ids, topk_weights, fused_shared_expert=True):
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import decode

    w13, w2, w13_s, w2_s = shuffled
    return decode.a16w8_decode_moe(
        x,
        w13,
        w13_s,
        w2,
        w2_s,
        topk_weights,
        topk_ids,
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_limit=SWIGLU_LIMIT,
        fused_shared_expert=fused_shared_expert,
        **_kw(shuffled),
    )


def _mid(shuffled, x, topk_ids, topk_weights):
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import mid

    w13, w2, w13_s, w2_s = shuffled
    return mid.a8w8_mid_moe(
        x, w13, w13_s, w2, w2_s, topk_weights, topk_ids, **_kw(shuffled)
    )


def _prefill(shuffled, x, topk_ids, topk_weights, out_mode="bf16"):
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import prefill

    w13, w2, w13_s, w2_s = shuffled
    return prefill.a8w8_prefill_moe(
        x,
        w13,
        w13_s,
        w2,
        w2_s,
        topk_weights,
        topk_ids,
        out_mode=out_mode,
        **_kw(shuffled),
    )


def _weights_for(m3_weights, shared):
    raw, shuffled = m3_weights
    count = NUM_ROUTED + int(shared)
    return tuple(w[:count] for w in raw), tuple(w[:count] for w in shuffled)


def _run_chain(shuffled, x, ids, weights, out_mode="bf16"):
    if x.shape[0] <= 256:
        return _decode(shuffled, x, ids, weights, shuffled[0].shape[0] == NUM_EXPERTS)
    if x.shape[0] <= 3071:
        return _mid(shuffled, x, ids, weights)
    return _prefill(shuffled, x, ids, weights, out_mode)


def _assert_relative_l2(actual, expected, *, limit, per_row=True):
    assert actual.shape == expected.shape
    assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
    dim = -1 if per_row else None
    scale = expected.float().norm(dim=dim)
    error = (actual.float() - expected.float()).norm(dim=dim)
    failed = error > limit * scale
    assert not failed.any(), (
        f"{int(failed.sum())} comparisons exceed the error budget; "
        f"max relative L2={float((error / scale.clamp_min(1e-30)).max()):.6g}"
    )


# Each case exercises a kernel variant or a dispatch boundary. In particular,
# M=2/4 have dedicated GEMM1 tiles and M=64/65 changes GEMM2 split-K.
@pytest.mark.parametrize(
    "m,shared,out_mode",
    [
        (1, True, "bf16"),
        (2, False, "bf16"),
        (4, True, "bf16"),
        (16, True, "bf16"),
        (17, False, "bf16"),
        (40, True, "bf16"),
        (40, False, "bf16"),
        (64, True, "bf16"),
        (65, True, "bf16"),
        (256, False, "bf16"),
        (257, False, "bf16"),
        (768, False, "bf16"),
        (1536, True, "bf16"),
        (3071, False, "bf16"),
        (3072, True, "bf16"),
        (16384, False, "bf16"),
        (32768, False, "bf16"),
        (4096, False, "fp8"),
    ],
)
def test_mxfp8_moe_accuracy(m3_weights, m, shared, out_mode):
    raw, shuffled = _weights_for(m3_weights, shared)
    set_random_seed(m)
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=shuffled[0].device)
    ids, weights = _routing(m, x.device, shared)
    # Always exercise the last routed expert; retain distinct IDs per token.
    ids[-1, :TOPK] = torch.tensor([0, 1, 2, NUM_ROUTED - 1], device=x.device)
    if m in (768, 16384):
        ids[: m // 2, :TOPK] = ids[-1, :TOPK].clone()
        x[0].zero_()
        weights[-1].zero_()
    out = _run_chain(shuffled, x, ids, weights, out_mode)
    assert out.shape == x.shape and out.dtype == torch.bfloat16
    if shared:
        baseline = _run_aiter(shuffled, x, ids, weights)
        _assert_relative_l2(
            out, baseline, limit=QUANTIZED_ERROR if m <= 256 else MATCHED_ERROR
        )
    # AITER's tuned shared-expert topology supplies an all-row differential
    # check above. For routed-only experts, check every row independently.
    tokens = _sample_rows(m) if shared else list(range(m))
    mode = "decode" if m <= 256 else "mid" if m <= 3071 else out_mode
    ref = _reference(x, raw, ids, weights, tokens, mode=mode)
    # Small GEMM differences can cross a quantizer midpoint. Check matched
    # precision in aggregate and float-reference quality on every tested row.
    _assert_relative_l2(out[tokens], ref.value, limit=MATCHED_ERROR, per_row=False)
    _assert_relative_l2(
        out[tokens],
        ref.float_value,
        limit=MATCHED_ERROR
        if m <= 256
        else FP8_PARTIAL_ERROR
        if out_mode == "fp8"
        else QUANTIZED_ERROR,
    )
    if m >= 3072:
        repeated = _run_chain(shuffled, x, ids, weights, out_mode)
        assert torch.equal(out, repeated)
    if m in (768, 16384):
        assert torch.count_nonzero(out[[0, m - 1]]) == 0


@pytest.mark.parametrize(
    "m,shared", [(16, True), (256, False), (768, False), (3072, False)]
)
def test_mxfp8_moe_graph_replay(m3_weights, m, shared):
    """Multiple calls reuse scratch; each replay reads new input/routing values."""
    _, shuffled = _weights_for(m3_weights, shared)
    set_random_seed(m)
    inputs = [
        (
            torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=shuffled[0].device),
            *_routing(m, shuffled[0].device, shared),
        )
        for _ in range(2)
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for inp in inputs:
            _run_chain(shuffled, *inp)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = [_run_chain(shuffled, *inp) for inp in inputs]
    for replay in range(2):
        for x, ids, weights in inputs:
            x.normal_()
            new_ids, new_weights = _routing(m, x.device, shared)
            if replay == 0:
                new_ids[:, :TOPK] = torch.tensor(
                    [0, 1, 2, NUM_ROUTED - 1], device=x.device
                )
            else:
                new_weights[0].zero_()
            ids.copy_(new_ids)
            weights.copy_(new_weights)
        eager = [_run_chain(shuffled, *inp).clone() for inp in inputs]
        graph.replay()
        torch.accelerator.synchronize()
        for got, want in zip(outputs, eager):
            if m >= 3072:
                assert torch.equal(got, want)
            else:
                _assert_relative_l2(got, want, limit=MATCHED_ERROR)


def test_mxfp8_moe_gates():
    from vllm.models.minimax_m3.amd.ops import moe_mxfp8 as k

    assert k.supports_shapes(HIDDEN, INTER)
    for hidden, inter in [(1536, INTER), (4096, INTER), (HIDDEN, 1536)]:
        assert not k.supports_shapes(hidden, inter)
    x = torch.empty((k.MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device="meta")
    assert k.supports_batch(x[:1]) and k.supports_batch(x)
    for bad in [
        x[:0],
        x.float(),
        x.t(),
        x[:, :-1],
        x[::2],
        x.new_empty((k.MAX_TOKENS + 1, HIDDEN)),
    ]:
        assert not k.supports_batch(bad)
    assert not k.supports_batch(x, topk=8)
    assert k.supports_routing(128, 4, fused_shared_expert=False)
    assert k.supports_routing(129, 5, fused_shared_expert=True)
    assert not k.supports_routing(129, 4, fused_shared_expert=False)
    assert not k.supports_routing(257, 5, fused_shared_expert=True)


def test_decode_cache_distinguishes_routing_width():
    """The inline routing stride is a compiled constant, including for one token."""
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import decode

    kw = dict(D_INTER=INTER, NE=NUM_EXPERTS, n_tokens=1, inline_sort=True)
    for get, shape in [
        (decode.get_gemm1, dict(D_HIDDEN=HIDDEN)),
        (decode.get_gemm2, dict(N_OUT=HIDDEN)),
    ]:
        routed = get(TOPK=4, **shape, **kw)
        shared = get(TOPK=5, **shape, **kw)
        assert routed is not shared
        assert get(TOPK=4, **shape, **kw) is routed


@pytest.mark.parametrize("out_mode", ["bf16", "fp8"])
def test_prefill_reduction_rounding_bound(out_mode):
    """Bound the actual reduction inputs, including cancellation and tiny scales.

    With no upstream GEMM/activation error, FP32 accumulation contributes
    gamma_(k-1) for BF16 inputs or gamma_k for weighted FP8 FMAs, followed by
    one BF16 rounding. The bound scales with sum(abs(partials)), not abs(sum).
    See Higham, The Accuracy of Floating Point Summation, equations 2.2-2.6.
    """
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.launch import (
        _get_reduce_bf16,
        _run_compiled,
    )
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8.prefill import _get_reduce_fp8

    set_random_seed(0)
    m, topk = 3, TOPK + 1
    device = torch.device("cuda")
    values = torch.randn((m, topk, HIDDEN), device=device)
    values[0, :, :4] = values.new_tensor([256, 1, -256, 0, 0])[:, None]
    output = torch.full((m, HIDDEN), float("nan"), dtype=torch.bfloat16, device=device)
    if out_mode == "bf16":
        values = values.bfloat16()
        partials = values.double()
        _run_compiled(
            _get_reduce_bf16(HIDDEN, topk),
            values.flatten(),
            output.flatten(),
            m,
            torch.cuda.current_stream(),
        )
        additions = topk - 1
    else:
        values = values.to(torch.float8_e4m3fn)
        scales = torch.randint(
            118, 136, (m, topk, HIDDEN // 32), dtype=torch.uint8, device=device
        )
        weights = torch.rand((m, topk), device=device)
        # E8M0 byte zero represents 2**-127, not an IEEE zero scale.
        scales[1, :, :1] = 0
        values[1, :, :32] = 128
        weights[1] = 1
        scale = torch.ldexp(
            torch.ones_like(scales, dtype=torch.float64), scales.int() - 127
        ).repeat_interleave(32, dim=-1)
        partials = values.double() * scale * weights.double().unsqueeze(-1)
        _run_compiled(
            _get_reduce_fp8(HIDDEN, topk),
            values.view(torch.uint8).flatten(),
            scales.flatten(),
            weights.flatten(),
            output.flatten(),
            m,
            torch.cuda.current_stream(),
        )
        additions = topk
    reference = partials.sum(dim=1)
    unit_roundoff = torch.finfo(torch.bfloat16).eps / 2
    # Also include the much smaller rounding allowance for the FP64 oracle sum.
    accumulation_error = (
        _gamma(additions, torch.float32) + _gamma(topk - 1, torch.float64)
    ) * partials.abs().sum(dim=1)
    bound = (1 + unit_roundoff) * accumulation_error + unit_roundoff * reference.abs()
    assert torch.isfinite(output).all()
    assert ((output.double() - reference).abs() <= bound).all()


def test_cached_reduction_on_multiple_devices():
    """A cached HIP function handle must belong to the launch's device."""
    if torch.accelerator.device_count() < 2:
        pytest.skip("Requires two GPUs")
    from vllm.models.minimax_m3.amd.ops.moe_flydsl_common.launch import (
        _get_reduce_bf16,
        _run_compiled,
    )

    topk = TOPK + 1
    launch = _get_reduce_bf16(HIDDEN, topk)
    for value, device in enumerate([0, 1, 0], start=1):
        with torch.accelerator.device_index(device):
            partials = torch.full(
                (3, topk, HIDDEN), value, dtype=torch.bfloat16, device=f"cuda:{device}"
            )
            output = torch.empty(
                (3, HIDDEN), dtype=partials.dtype, device=partials.device
            )
            _run_compiled(
                launch,
                partials.flatten(),
                output.flatten(),
                3,
                torch.cuda.current_stream(),
            )
            assert torch.equal(output, torch.full_like(output, value * topk))


def test_mxfp8_moe_dispatch_and_warm_up(monkeypatch):
    """Warm every routing/output variant once, retry failures, and avoid capture."""
    from vllm.models.minimax_m3.amd.ops import moe_mxfp8 as k

    monkeypatch.setattr(k, "_warmed", {})
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    calls = []
    fail = False

    def chain(name):
        def run(x, *args, out=None, **kwargs):
            calls.append((name, x.shape[0]))
            if fail and x.shape[0] == 4096:
                raise RuntimeError("compile failed")
            return x.new_empty(x.shape) if out is None else out

        return run

    monkeypatch.setattr(k.decode, "a16w8_decode_moe", chain("decode"))
    monkeypatch.setattr(k.mid, "a8w8_mid_moe", chain("mid"))
    monkeypatch.setattr(k.prefill, "a8w8_prefill_moe", chain("prefill"))

    def run(m, shared=False, out=None):
        e, topk = NUM_ROUTED + int(shared), TOPK + int(shared)
        x = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device="meta")
        w = torch.empty(e, device="meta")
        ids = torch.empty((m, topk), dtype=torch.int32, device="meta")
        return k.mxfp8_moe(
            x,
            w,
            w,
            w,
            w,
            ids.float(),
            ids,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            num_experts=e,
            fused_shared_expert=shared,
            out=out,
        )

    for m, name in [(256, "decode"), (257, "mid"), (3071, "mid")]:
        calls.clear()
        out = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device="meta")
        assert run(m, out=out) is out
        assert calls == [(name, m)]
    fail = True
    with pytest.raises(RuntimeError, match="compile failed"):
        run(4096)
    fail = False
    calls.clear()
    run(4096)
    assert calls[-1] == ("prefill", 4096)
    assert {k.mid.block_m_for(m) for name, m in calls if name == "mid"} == {32, 64, 128}
    calls.clear()
    run(4096)
    assert calls == [("prefill", 4096)]
    calls.clear()
    run(32768)
    assert calls[-1] == ("prefill", 32768)
    assert any(name == "prefill" and 4096 < m < 32768 for name, m in calls)
    for shared, mode in [(True, "bf16"), (False, "fp8")]:
        monkeypatch.setenv("AITER_FLYDSL_STAGE2_FP8", str(int(mode == "fp8")))
        calls.clear()
        run(4096, shared)
        assert any(name == "mid" for name, m in calls)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    calls.clear()
    run(4096, shared=True)
    assert calls == [("prefill", 4096)]
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    calls.clear()
    run(4096, shared=True)
    assert any(name == "mid" for name, m in calls)


def _fake_moe_config(**overrides):
    from dataclasses import replace

    from tests.kernels.moe.utils import make_dummy_moe_config
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    cfg = make_dummy_moe_config(
        num_experts=NUM_ROUTED,
        num_local_experts=NUM_ROUTED,
        experts_per_token=TOPK,
        hidden_dim=HIDDEN,
        intermediate_size=INTER * 4,
        max_num_tokens=65536,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
    )
    cfg = replace(
        cfg,
        moe_parallel_config=replace(cfg.moe_parallel_config, tp_size=4),
        intermediate_size_per_partition_unpadded=INTER,
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_beta=1.0,
        swiglu_limit=SWIGLU_LIMIT,
    )
    return replace(cfg, **overrides)


def test_experts_class_supported_config(monkeypatch):
    from dataclasses import replace

    import vllm.envs as envs
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.experts.minimax_m3_flydsl_mxfp8_moe import (  # noqa: E501
        MiniMaxM3FlyDSLMxfp8Experts as Cls,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
        kMxfp8Static,
    )

    monkeypatch.setattr(envs, "VLLM_ROCM_USE_M3_FLYDSL_MOE", True)
    monkeypatch.setattr(envs, "VLLM_BATCH_INVARIANT", False)

    def check(cfg):
        return Cls.is_supported_config(
            Cls, cfg, kMxfp8Static, kMxfp8Dynamic, mk.FusedMoEActivationFormat.Standard
        )

    cfg = _fake_moe_config()
    assert check(cfg) == (True, None)
    assert check(replace(cfg, num_local_experts=129)) == (True, None)
    rejected = [
        dict(hidden_dim=1536, hidden_dim_unpadded=1536),
        dict(hidden_dim=4096, hidden_dim_unpadded=4096),
        dict(intermediate_size=1536 * 4),
        dict(num_experts=257, num_local_experts=257),
        dict(num_local_experts=130),
        dict(experts_per_token=8),
        dict(
            moe_parallel_config=replace(cfg.moe_parallel_config, use_ep=True, ep_size=2)
        ),
        dict(has_bias=True),
        dict(hidden_dim_unpadded=HIDDEN - 256),
        dict(swiglu_limit=None),
        dict(swiglu_limit=SWIGLU_LIMIT + 1e-10),
        dict(swiglu_alpha=SWIGLU_ALPHA + 1e-10),
        dict(swiglu_beta=1.0 + 1e-10),
    ]
    for changes in rejected:
        ok, reason = check(replace(cfg, **changes))
        assert not ok and reason, changes
    monkeypatch.setattr(envs, "VLLM_ROCM_USE_M3_FLYDSL_MOE", False)
    assert not check(cfg)[0]


def _apply(experts, output, x, shuffled, ids, weights, **overrides):
    args = dict(
        output=output,
        hidden_states=x,
        w1=shuffled[0],
        w2=shuffled[1],
        topk_weights=weights,
        topk_ids=ids,
        activation=experts.moe_config.activation,
        global_num_experts=NUM_ROUTED,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    args.update(overrides)
    experts.apply(**args)


@pytest.mark.parametrize(
    "m,shared,out_dtype", [(16, True, torch.bfloat16), (257, False, torch.float32)]
)
def test_experts_class_apply(m3_weights, m, shared, out_dtype):
    from vllm.model_executor.layers.fused_moe.config import FusedMoEQuantConfig
    from vllm.model_executor.layers.fused_moe.experts.minimax_m3_flydsl_mxfp8_moe import (  # noqa: E501
        MiniMaxM3FlyDSLMxfp8Experts as Cls,
    )

    raw, shuffled = _weights_for(m3_weights, shared)
    _, _, scale13, scale2 = shuffled
    cfg = _fake_moe_config(num_local_experts=NUM_ROUTED + int(shared))
    quant = FusedMoEQuantConfig.make(
        quant_dtype="mxfp8",
        block_shape=[1, 32],
        w1_scale=scale13,
        w2_scale=scale2,
        gemm1_alpha=SWIGLU_ALPHA,
        gemm1_beta=1.0,
        gemm1_clamp_limit=SWIGLU_LIMIT,
    )
    experts = Cls(cfg, quant)
    assert experts.w1_scale_val is scale13 and experts.w2_scale_val is scale2
    set_random_seed(m)
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=shuffled[0].device)
    ids, weights = _routing(m, x.device, shared)
    output = torch.full(x.shape, float("nan"), dtype=out_dtype, device=x.device)
    _apply(experts, output, x, shuffled, ids, weights)
    reference = _reference(
        x, raw, ids, weights, list(range(m)), mode="decode" if m <= 256 else "mid"
    )
    _assert_relative_l2(output, reference.value, limit=MATCHED_ERROR)


@pytest.mark.parametrize(
    "case", ["dtype", "strided", "router_weight", "expert_mask", "routing_width"]
)
def test_experts_class_falls_back(monkeypatch, case):
    from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (
        AiterMxfp8Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.minimax_m3_flydsl_mxfp8_moe import (  # noqa: E501
        MiniMaxM3FlyDSLMxfp8Experts as Cls,
    )

    experts = object.__new__(Cls)
    experts.moe_config = _fake_moe_config()
    x = torch.zeros((2, HIDDEN), dtype=torch.bfloat16)
    ids = torch.zeros((2, TOPK), dtype=torch.int32)
    weights = ids.float()
    overrides = {}
    if case == "dtype":
        x = x.float()
    elif case == "strided":
        x = torch.zeros((2, HIDDEN * 2), dtype=torch.bfloat16)[:, ::2]
    elif case == "router_weight":
        overrides["apply_router_weight_on_input"] = True
    elif case == "expert_mask":
        overrides["expert_map"] = torch.ones(NUM_ROUTED + 1, dtype=torch.int32)
    else:
        ids, weights = ids[:, :3], weights[:, :3]
    out = torch.empty_like(x)
    calls = []

    def fallback(self, output, hidden_states, *args):
        calls.append((hidden_states, args))
        output.copy_(hidden_states + 1)

    monkeypatch.setattr(AiterMxfp8Experts, "apply", fallback)
    w = torch.empty((NUM_ROUTED, 0, 0))
    _apply(experts, out, x, (w, w), ids, weights, **overrides)
    assert len(calls) == 1 and calls[0][0] is x
    torch.testing.assert_close(out, x + 1, rtol=0, atol=0)
