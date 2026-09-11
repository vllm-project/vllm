# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL a8w8 prefill MoE (gfx950, MXFP8 weights and activations,
3072 <= M <= 65536).

Weights are quantized to MXFP8 (fp8 e4m3 + per-32 e8m0, the checkpoint format)
and shuffled exactly like ``ModelOptMxFp8FusedMoE`` does for the AITER_MXFP8
backend (``shuffle_mxfp8_moe_weights``); routing mimics aiter's fused shared
expert. The output is compared with the production aiter a8w8 call
(``fused_moe`` per_1x32, gate/up INTERLEAVE; the chain replaces its two GEMMs and
keeps its fp8 quant and reduction) and, on a sample of tokens, with a float
reference on the dequantized experts.
"""

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


def _routing(m: int, device):
    routed = torch.stack(
        [torch.randperm(NUM_ROUTED, device=device)[:TOPK] for _ in range(m)]
    )
    shared = torch.full((m, 1), NUM_ROUTED, device=device)
    topk_ids = torch.cat([routed, shared], dim=1).to(torch.int32)
    w = torch.rand((m, TOPK), device=device)
    w = w / w.sum(dim=1, keepdim=True) * 2.0
    topk_weights = torch.cat([w, torch.ones((m, 1), device=device)], dim=1)
    return topk_ids, topk_weights.to(torch.float32)


def _dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    sc = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.to(torch.int32) - 127)
    return (q.float().view(n, k // 32, 32) * sc.unsqueeze(-1)).view(n, k)


def _float_reference(x, raw, topk_ids, topk_weights, tokens):
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


def _run(shuffled, x, topk_ids, topk_weights, out_mode="bf16"):
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_prefill import a8w8_prefill_moe

    w13, w2, w13_s, w2_s = shuffled
    return a8w8_prefill_moe(
        x,
        w13,
        w13_s,
        w2,
        w2_s,
        topk_weights,
        topk_ids,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_experts=NUM_EXPERTS,
        out_mode=out_mode,
    )


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


@pytest.mark.parametrize("m", [3072, 4096, 16384])
def test_a8w8_prefill_moe_matches_aiter(m3_weights, m):
    """3072/4096: 128-row sort blocks; 16384: 256-row blocks (gemm1 BM256,
    gemm2 skipping the all-padding 128-row tiles)."""
    _check(m3_weights, m, "bf16")


def test_a8w8_prefill_moe_fp8_route_out(m3_weights):
    """gemm2's fp8 output mode (AITER_FLYDSL_STAGE2_FP8=1): fp8 partials +
    reduce_fp8, deterministic, one more quantization."""
    _check(m3_weights, 4096, "fp8")


def _check(m3_weights, m, out_mode):
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_prefill import (
        block_m_for,
        supports_shapes,
    )

    assert supports_shapes(HIDDEN, INTER)
    assert block_m_for(m) == (256 if m >= 16384 else 128)
    raw, shuffled = m3_weights
    torch.manual_seed(m)
    device = shuffled[0].device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device)

    out = _run(shuffled, x, topk_ids, topk_weights, out_mode)
    out2 = _run(shuffled, x, topk_ids, topk_weights, out_mode)
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
    ref = _float_reference(x, raw, topk_ids, topk_weights, tokens)
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


def test_a8w8_fast_path_gate():
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_mid import MIN_MID_TOKENS
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_prefill import (
        MAX_PREFILL_TOKENS,
        supports_batch,
        supports_shapes,
    )

    device = torch.device("cuda")
    assert supports_shapes(6144, 768)
    assert not supports_shapes(6144, 1536)  # gemm2 pipeline is written for K = 768
    assert not supports_shapes(6144 + 128, 768)  # gemm1 unroll: (K/128 - 4) % 4
    ok = torch.empty((MIN_MID_TOKENS, HIDDEN), dtype=torch.bfloat16, device=device)
    assert supports_batch(ok)
    assert not supports_batch(ok[: MIN_MID_TOKENS - 1])  # the decode package's
    assert not supports_batch(ok.float())
    assert not supports_batch(ok.t())
    assert not supports_batch(
        torch.empty(
            (MAX_PREFILL_TOKENS + 1, HIDDEN), dtype=torch.bfloat16, device="meta"
        )
    )


def test_a8w8_fast_path_routes_by_batch(m3_weights, monkeypatch):
    """``install_prefill_fast_path`` wraps ``quant_method.apply``: 257..3071 go
    to the mid chain, 3072 and up to the prefill chain, the first prefill-range
    call also warms up the other kernel configurations, and everything else
    (the decode range, unfused shared experts) stays with the wrapped apply."""
    from types import SimpleNamespace

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.models.minimax_m3.amd.ops import moe_a8w8_prefill

    raw, shuffled = m3_weights
    w13, w2, w13_s, w2_s = shuffled
    device = w13.device
    calls: list[tuple[str, int]] = []

    def record(name, fn):
        def wrapped(x, *args, **kwargs):
            calls.append((name, x.shape[0]))
            return fn(x, *args, **kwargs)

        return wrapped

    monkeypatch.setattr(
        moe_a8w8_prefill, "a8w8_mid_moe", record("mid", moe_a8w8_prefill.a8w8_mid_moe)
    )
    monkeypatch.setattr(
        moe_a8w8_prefill,
        "a8w8_prefill_moe",
        record("prefill", moe_a8w8_prefill.a8w8_prefill_moe),
    )
    monkeypatch.setattr(moe_a8w8_prefill, "_warmed", False)

    class ModelOptMxFp8FusedMoE:  # the gate keys on the class name + backend
        mxfp8_backend = SimpleNamespace(value="AITER_MXFP8")

        def apply(self, layer, x, topk_weights, topk_ids, *args, **kwargs):
            calls.append(("aiter", x.shape[0]))
            return torch.zeros_like(x)

    layer = SimpleNamespace(
        quant_method=ModelOptMxFp8FusedMoE(),
        activation=MoEActivation.SWIGLUOAI_UNINTERLEAVE,
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_limit=SWIGLU_LIMIT,
        swiglu_beta=None,
        apply_router_weight_on_input=False,
        expert_map=None,
        moe_config=SimpleNamespace(
            use_ep=False,
            has_bias=False,
            hidden_dim=HIDDEN,
            intermediate_size_per_partition=INTER,
            hidden_dim_unpadded=None,
            intermediate_size_per_partition_unpadded=None,
        ),
        w13_weight=w13,
        w13_weight_scale=w13_s,
        w2_weight=w2,
        w2_weight_scale=w2_s,
    )
    assert moe_a8w8_prefill.install_prefill_fast_path(layer, prefix="t")
    assert moe_a8w8_prefill.install_prefill_fast_path(layer)  # idempotent
    apply = layer.quant_method.apply

    torch.manual_seed(0)
    x = torch.randn((4096, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(4096, device)
    apply(layer, x[:256], topk_weights[:256], topk_ids[:256])
    assert calls == [("aiter", 256)]
    calls.clear()
    out = apply(layer, x[:512], topk_weights[:512], topk_ids[:512])
    assert calls == [("mid", 512)] and out.shape == (512, HIDDEN)
    calls.clear()
    out = apply(layer, x, topk_weights, topk_ids)
    assert calls == [
        ("mid", 512),
        ("mid", 768),
        ("mid", 1536),
        ("prefill", 3072),
        ("prefill", 4096),
    ]
    assert out.shape == (4096, HIDDEN) and out.dtype == torch.bfloat16
    ref = _run_aiter(shuffled, x, topk_ids, topk_weights)
    assert _cos(out, ref) > 0.9999
    calls.clear()
    apply(layer, x, topk_weights, topk_ids)
    assert calls == [("prefill", 4096)]
    calls.clear()
    apply(layer, x[:512], topk_weights[:512], topk_ids[:512], shared_experts=object())
    assert calls == [("aiter", 512)]
