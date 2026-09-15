# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL MoE (gfx950, ``vllm/models/minimax_m3/amd/ops/moe_mxfp8``): the
decode / mid / prefill chains against aiter's a8w8 call and a float reference on
dequantized MXFP8 experts, the package dispatch, and the experts class the MXFP8
oracle selects. Routing either mimics aiter's fused shared expert (last expert,
every token, weight 1) or is the routed-only top-k of ModelOpt MXFP8.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only"),
    pytest.mark.skipif(not on_gfx950(), reason="gfx950 only"),
]

HIDDEN, INTER, NUM_ROUTED, TOPK = 6144, 768, 128, 4
NUM_EXPERTS = NUM_ROUTED + 1
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0
CHECK_TOKENS = 64


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
    routed = torch.stack(
        [torch.randperm(NUM_ROUTED, device=device)[:TOPK] for _ in range(m)]
    )
    w = torch.rand((m, TOPK), device=device)
    w = w / w.sum(dim=1, keepdim=True) * 2.0
    if shared:
        routed = torch.cat([routed, torch.full((m, 1), NUM_ROUTED, device=device)], 1)
        w = torch.cat([w, torch.ones((m, 1), device=device)], dim=1)
    return routed.to(torch.int32).contiguous(), w.to(torch.float32).contiguous()


def _dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    sc = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.to(torch.int32) - 127)
    return (q.float().view(n, k // 32, 32) * sc.unsqueeze(-1)).view(n, k)


def _float_reference(x, raw, topk_ids, topk_weights, tokens, stage1_bf16: bool):
    """Dequantized experts in fp32; ``stage1_bf16`` rounds the intermediate to
    bf16 as the decode chain does (the fp8 chains quantize it instead)."""
    w13_q, w13_s, w2_q, w2_s = raw
    out = torch.zeros((len(tokens), HIDDEN), dtype=torch.float32, device=x.device)
    xf = x.float()
    for i, t in enumerate(tokens):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            h = xf[t] @ _dequant(w13_q[e], w13_s[e]).T
            g = h[:INTER].clamp(max=SWIGLU_LIMIT)
            u = h[INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            a = g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)
            if stage1_bf16:
                a = a.to(torch.bfloat16).float()
            out[i] += float(topk_weights[t, j]) * (a @ _dequant(w2_q[e], w2_s[e]).T)
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@pytest.fixture(scope="module")
def m3_weights():
    from vllm._aiter_ops import rocm_aiter_ops

    torch.manual_seed(0)
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


def _kw():
    return dict(hidden_size=HIDDEN, intermediate_size=INTER, num_experts=NUM_EXPERTS)


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
        **_kw(),
    )


def _mid(shuffled, x, topk_ids, topk_weights):
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import mid

    w13, w2, w13_s, w2_s = shuffled
    return mid.a8w8_mid_moe(x, w13, w13_s, w2, w2_s, topk_weights, topk_ids, **_kw())


def _prefill(shuffled, x, topk_ids, topk_weights, out_mode="bf16"):
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import prefill

    w13, w2, w13_s, w2_s = shuffled
    return prefill.a8w8_prefill_moe(
        x, w13, w13_s, w2, w2_s, topk_weights, topk_ids, out_mode=out_mode, **_kw()
    )


# ---------------------------------------------------------------- decode (a16w8)


@pytest.mark.parametrize("m", [1, 2, 4, 8, 12, 16, 17, 32, 64, 128, 256])
def test_a16w8_decode_matches_reference(m3_weights, m):
    """M <= 16: inline-sort path; 17..256: sort_decode + sorted GEMMs (the
    wide-first layout of the fused shared expert from 40 tokens)."""
    from vllm.models.minimax_m3.amd.ops import moe_mxfp8 as k

    assert k.supports_shapes(HIDDEN, INTER) and m <= k.MAX_DECODE_TOKENS
    raw, shuffled = m3_weights
    torch.manual_seed(m)
    device = shuffled[0].device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device)
    out = _decode(shuffled, x, topk_ids, topk_weights)
    ref_aiter = _run_aiter(shuffled, x, topk_ids, topk_weights)
    torch.accelerator.synchronize()
    assert out.shape == (m, HIDDEN) and out.dtype == torch.bfloat16
    ref = _float_reference(x, raw, topk_ids, topk_weights, range(m), stage1_bf16=True)
    scale = ref.abs().max().item()
    assert _cos(out, ref) > 0.9999
    assert (out.float() - ref).abs().max().item() < 0.02 * scale
    # aiter quantizes the activations to MXFP8 as well: ~0.999 against the float
    # reference, and the same distance from us
    assert _cos(out, ref_aiter) > 0.998


@pytest.mark.parametrize("m", [16, 256])
def test_a16w8_decode_graph_replay(m3_weights, m):
    """HIP-graph capture with different routing per call (how vLLM runs it)."""
    _, shuffled = m3_weights
    torch.manual_seed(1)
    device = shuffled[0].device
    inputs = [
        (
            torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device),
            *_routing(m, device),
        )
        for _ in range(4)
    ]
    eager = [_decode(shuffled, *inp).clone() for inp in inputs]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _decode(shuffled, *inputs[0])
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [_decode(shuffled, *inp) for inp in inputs]
    g.replay()
    g.replay()
    torch.accelerator.synchronize()
    for got, want in zip(outs, eager):
        assert _cos(got, want) > 0.9999


@pytest.mark.parametrize("m", [40, 192, 256])
def test_a16w8_decode_separate_shared_expert(m3_weights, m):
    """The routing vLLM produces for ModelOpt MXFP8, where the shared expert is
    a separate module: top-k over the routed experts only, no expert routed by
    every token. The wide-first sort layout (from 40 tokens) budgets the last
    expert's blocks from M and must stay off; these M broke without the flag."""
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import decode

    assert decode.wide_for(m)
    raw, shuffled = m3_weights
    torch.manual_seed(m)
    device = shuffled[0].device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device, shared=False)
    out = _decode(shuffled, x, topk_ids, topk_weights, fused_shared_expert=False)
    ref_aiter = _run_aiter(shuffled, x, topk_ids, topk_weights)
    torch.accelerator.synchronize()
    ref = _float_reference(x, raw, topk_ids, topk_weights, range(m), stage1_bf16=True)
    assert _cos(out, ref) > 0.9999
    assert _cos(out, ref_aiter) > 0.998


# ------------------------------------------------------------- mid batch (a8w8)


@pytest.mark.parametrize("m", [300, 512, 1024, 2048, 3071])
def test_a8w8_mid_matches_aiter(m3_weights, m):
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import (
        MAX_MID_TOKENS,
        MIN_MID_TOKENS,
        mid,
    )

    assert MIN_MID_TOKENS <= m <= MAX_MID_TOKENS
    assert mid.block_m_for(m) in (32, 64, 128)
    raw, shuffled = m3_weights
    torch.manual_seed(m)
    device = shuffled[0].device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device)
    out = _mid(shuffled, x, topk_ids, topk_weights)
    out2 = _mid(shuffled, x, topk_ids, topk_weights)
    ref_aiter = _run_aiter(shuffled, x, topk_ids, topk_weights)
    torch.accelerator.synchronize()
    assert out.shape == (m, HIDDEN) and out.dtype == torch.bfloat16
    assert not torch.isnan(out).any()
    # the bf16 atomics of gemm2 add the topk terms in arrival order: the same
    # inputs differ in the last bit of a few elements, never more
    assert _cos(out, out2) > 0.99999
    assert (out.float() - out2.float()).abs().max() < 0.1
    # aiter's a8w8 chain quantizes x and the intermediate the same way and its
    # stage 2 accumulates with bf16 atomics as well: cos 0.99999
    assert _cos(out, ref_aiter) > 0.9999
    # both are ~0.999 to the float reference (the fp8 activation quant
    # dominates); ours must be as close as aiter's own kernels
    tokens = torch.randperm(m, device=device)[:CHECK_TOKENS].tolist()
    ref = _float_reference(x, raw, topk_ids, topk_weights, tokens, stage1_bf16=False)
    c_ours = _cos(out[tokens], ref)
    c_aiter = _cos(ref_aiter[tokens], ref)
    assert c_ours > 0.998 and c_ours > c_aiter - 1e-4, (c_ours, c_aiter)


# --------------------------------------------------------------- prefill (a8w8)


@pytest.mark.parametrize("m", [3072, 4096, 16384])
def test_a8w8_prefill_matches_aiter(m3_weights, m):
    """3072/4096: 128-row sort blocks; 16384: 256-row blocks (gemm1 BM256,
    gemm2 skipping the all-padding 128-row tiles)."""
    _check_prefill(m3_weights, m, "bf16")


def test_a8w8_prefill_fp8_route_out(m3_weights):
    """gemm2's fp8 output mode (AITER_FLYDSL_STAGE2_FP8=1): fp8 partials +
    reduce_fp8, deterministic, one more quantization."""
    _check_prefill(m3_weights, 4096, "fp8")


def _check_prefill(m3_weights, m, out_mode):
    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import prefill

    assert prefill.supports_shapes(HIDDEN, INTER)
    assert prefill.block_m_for(m) == (256 if m >= 16384 else 128)
    raw, shuffled = m3_weights
    torch.manual_seed(m)
    device = shuffled[0].device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device)
    out = _prefill(shuffled, x, topk_ids, topk_weights, out_mode)
    out2 = _prefill(shuffled, x, topk_ids, topk_weights, out_mode)
    ref_aiter = _run_aiter(shuffled, x, topk_ids, topk_weights)
    torch.accelerator.synchronize()
    assert out.shape == (m, HIDDEN) and out.dtype == torch.bfloat16
    # deterministic: the same inputs give the same bits (a race shows up here)
    assert torch.equal(out, out2)
    # aiter's a8w8 chain quantizes x and the intermediate the same way but its
    # stage 2 accumulates with bf16 atomics: cos 0.99999, not bit-identical; the
    # fp8 partials add one more quantization (cos ~0.9985)
    assert _cos(out, ref_aiter) > (0.998 if out_mode == "fp8" else 0.9999)
    # both are ~0.999 to the float reference (the fp8 activation quant
    # dominates); ours must be as close as aiter's own kernels (fp8 mode: a
    # little further, by its extra quantization)
    tokens = torch.randperm(m, device=device)[:CHECK_TOKENS].tolist()
    ref = _float_reference(x, raw, topk_ids, topk_weights, tokens, stage1_bf16=False)
    ours = out[tokens].float()
    theirs = ref_aiter[tokens].float()
    slack = 2e-3 if out_mode == "fp8" else 1e-3
    assert _cos(ours, ref) > 0.997
    assert _cos(ours, ref) >= _cos(theirs, ref) - slack
    err_ours = (ours - ref).abs().max().item()
    err_theirs = (theirs - ref).abs().max().item()
    assert err_ours <= err_theirs * (2.0 if out_mode == "fp8" else 1.1) + 1e-3, (
        err_ours,
        err_theirs,
    )


# ------------------------------------------------- the package and the experts class


def test_mxfp8_moe_gates():
    from vllm.models.minimax_m3.amd.ops import moe_mxfp8 as k

    assert k.MAX_DECODE_TOKENS + 1 == k.MIN_MID_TOKENS
    assert k.MAX_MID_TOKENS + 1 == k.MIN_PREFILL_TOKENS == 3072
    assert k.MAX_TOKENS == k.MAX_PREFILL_TOKENS == 65536
    assert k.supports_shapes(6144, 768)
    assert not k.supports_shapes(6144, 1536)  # TP2: the fp8 gemm2 pipeline is K = 768
    assert not k.supports_shapes(6144 + 128, 768)  # gemm1 unroll: (K/128 - 4) % 4
    device = torch.device("cuda")
    ok = torch.empty((k.MAX_TOKENS, HIDDEN), dtype=torch.bfloat16, device="meta")
    assert k.supports_batch(ok[:1]) and k.supports_batch(ok)
    assert not k.supports_batch(ok[:0])
    assert not k.supports_batch(ok.float())
    assert not k.supports_batch(ok.t())
    assert not k.supports_batch(
        torch.empty((k.MAX_TOKENS + 1, HIDDEN), dtype=torch.bfloat16, device="meta")
    )
    assert k.supports_batch(
        torch.empty((4, HIDDEN), dtype=torch.bfloat16, device=device)
    )


def test_mxfp8_moe_dispatch_and_warm_up(m3_weights, monkeypatch):
    """``mxfp8_moe`` picks the chain by batch size, runs the one-time warm-up on
    the first prefill-range call (the profile run) and writes into ``out``."""
    from vllm.models.minimax_m3.amd.ops import moe_mxfp8 as k

    raw, shuffled = m3_weights
    w13, w2, w13_s, w2_s = shuffled
    device = w13.device
    calls: list[tuple[str, int]] = []

    def record(name, module, attr):
        real = getattr(module, attr)

        def wrapped(x, *args, **kwargs):
            calls.append((name, x.shape[0]))
            return real(x, *args, **kwargs)

        monkeypatch.setattr(module, attr, wrapped)

    record("decode", k.decode, "a16w8_decode_moe")
    record("mid", k.mid, "a8w8_mid_moe")
    record("prefill", k.prefill, "a8w8_prefill_moe")
    monkeypatch.setattr(k, "_warmed", False)

    def run(x, ids, w, **kwargs):
        return k.mxfp8_moe(x, w13, w13_s, w2, w2_s, w, ids, **_kw(), **kwargs)

    torch.manual_seed(0)
    x = torch.randn((4096, HIDDEN), dtype=torch.bfloat16, device=device)
    ids, w = _routing(4096, device)
    run(x[:256], ids[:256], w[:256], fused_shared_expert=True)
    assert calls == [("decode", 256)]
    calls.clear()
    out = torch.empty((512, HIDDEN), dtype=torch.bfloat16, device=device)
    res = run(x[:512], ids[:512], w[:512], out=out)
    assert res is out and calls == [("mid", 512)]
    calls.clear()
    out = run(x, ids, w)
    assert calls == [
        ("mid", 512),
        ("mid", 768),
        ("mid", 1536),
        ("prefill", 3072),
        ("prefill", 4096),
    ]
    assert out.shape == (4096, HIDDEN) and out.dtype == torch.bfloat16
    assert _cos(out, _run_aiter(shuffled, x, ids, w)) > 0.9999
    calls.clear()
    run(x, ids, w)
    assert calls == [("prefill", 4096)]


def test_oracle_prefers_flydsl_experts_for_aiter_backend():
    from vllm.model_executor.layers.fused_moe.experts.aiter_mxfp8_moe import (
        AiterMxfp8Experts,
    )
    from vllm.model_executor.layers.fused_moe.experts.minimax_m3_flydsl_mxfp8_moe import (  # noqa: E501
        MiniMaxM3FlyDSLMxfp8Experts,
    )
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
    from vllm.model_executor.layers.fused_moe.oracle.mxfp8 import (
        _mxfp8_backend_to_kernel_cls,
    )

    assert _mxfp8_backend_to_kernel_cls(Fp8MoeBackend.AITER_MXFP8) == [
        MiniMaxM3FlyDSLMxfp8Experts,
        AiterMxfp8Experts,
    ]
    assert issubclass(MiniMaxM3FlyDSLMxfp8Experts, AiterMxfp8Experts)


def _fake_moe_config(**overrides):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation

    cfg = dict(
        num_experts=NUM_ROUTED,
        experts_per_token=TOPK,
        is_act_and_mul=True,
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_beta=1.0,
        swiglu_limit=SWIGLU_LIMIT,
        hidden_dim=HIDDEN,
        hidden_dim_unpadded=HIDDEN,
        intermediate_size_per_partition=INTER,
        intermediate_size_per_partition_unpadded=INTER,
        has_bias=False,
        is_lora_enabled=False,
        routing_method=None,
        router_logits_dtype=torch.float32,
        moe_parallel_config=SimpleNamespace(
            use_ep=False, dp_size=1, tp_size=4, use_batched_activation_format=False
        ),
    )
    cfg.update(overrides)
    return SimpleNamespace(**cfg)


def test_experts_class_supported_config(monkeypatch):
    import vllm.model_executor.layers.fused_moe.modular_kernel as mk
    from vllm.model_executor.layers.fused_moe.experts.minimax_m3_flydsl_mxfp8_moe import (  # noqa: E501
        MiniMaxM3FlyDSLMxfp8Experts as Cls,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        kMxfp8Dynamic,
        kMxfp8Static,
    )

    def check(cfg):
        return Cls.is_supported_config(
            Cls, cfg, kMxfp8Static, kMxfp8Dynamic, mk.FusedMoEActivationFormat.Standard
        )

    assert check(_fake_moe_config()) == (True, None)
    ok, reason = check(_fake_moe_config(intermediate_size_per_partition=1536))
    assert not ok and "1536" in reason  # TP2
    ok, reason = check(
        _fake_moe_config(
            moe_parallel_config=SimpleNamespace(use_ep=True, dp_size=1, tp_size=4)
        )
    )
    assert not ok and "expert parallelism" in reason
    ok, reason = check(_fake_moe_config(swiglu_limit=None))
    assert not ok and "swiglu_limit" in reason
    ok, reason = check(_fake_moe_config(hidden_dim_unpadded=HIDDEN - 256))
    assert not ok and "padded" in reason
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_ROCM_USE_M3_FLYDSL_MOE", False)
    ok, reason = check(_fake_moe_config())
    assert not ok and "VLLM_ROCM_USE_M3_FLYDSL_MOE" in reason


@pytest.mark.parametrize(
    "m, fused", [(16, True), (256, False), (512, False), (4096, True)]
)
def test_experts_class_apply(m3_weights, monkeypatch, m, fused):
    """``apply`` as the modular kernel calls it: the routed output lands in
    ``output``; a fused shared expert is recognised from the weights having one
    expert more than ``global_num_experts``."""
    from vllm.model_executor.layers.fused_moe.experts.minimax_m3_flydsl_mxfp8_moe import (  # noqa: E501
        MiniMaxM3FlyDSLMxfp8Experts as Cls,
    )
    from vllm.models.minimax_m3.amd.ops import moe_mxfp8 as k

    raw, shuffled = m3_weights
    w13, w2, w13_s, w2_s = shuffled
    device = w13.device
    monkeypatch.setattr(k, "_warmed", True)  # the dispatch test covers the warm-up
    experts = object.__new__(Cls)
    experts.moe_config = _fake_moe_config()
    experts.w1_scale_val, experts.w2_scale_val = w13_s, w2_s
    torch.manual_seed(m)
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    ids, w = _routing(m, device, shared=fused)
    output = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
    experts.apply(
        output,
        x,
        w13,
        w2,
        w,
        ids,
        activation=experts.moe_config.activation,
        global_num_experts=NUM_ROUTED if fused else NUM_EXPERTS,
        expert_map=None,
        a1q_scale=None,
        a2_scale=None,
        workspace13=None,
        workspace2=None,
        expert_tokens_meta=None,
        apply_router_weight_on_input=False,
    )
    torch.accelerator.synchronize()
    assert _cos(output, _run_aiter(shuffled, x, ids, w)) > 0.998
