# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL a16w8 decode MoE (gfx950, MXFP8 weights, bf16 x, M <= 256).

Weights are quantized to MXFP8 (fp8 e4m3 + per-32 e8m0, the checkpoint format)
and shuffled exactly like ``ModelOptMxFp8FusedMoE`` does for the AITER_MXFP8
backend at load time (``shuffle_mxfp8_moe_weights``); routing mimics aiter's
fused shared expert; the output is compared with a float reference on the
dequantized experts and with the production aiter a8w8 call (``fused_moe``
per_1x32, gate/up INTERLEAVE), which quantizes the activations to MXFP8 as
well and therefore sits a little further from the float reference.
"""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only"),
    pytest.mark.skipif(not on_gfx950(), reason="gfx950 only"),
]

# MiniMax-M3 at TP4: hidden 6144, intermediate 3072 / 4, 128 routed experts
# + 1 shared expert appended as id 128, top-4 routed + shared = 5 pairs.
HIDDEN, INTER, NUM_ROUTED, TOPK = 6144, 768, 128, 4
NUM_EXPERTS = NUM_ROUTED + 1
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0


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


def _float_reference(x, raw, topk_ids, topk_weights):
    w13_q, w13_s, w2_q, w2_s = raw
    out = torch.zeros((x.shape[0], HIDDEN), dtype=torch.float32, device=x.device)
    xf = x.float()
    for t in range(x.shape[0]):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            h = xf[t] @ _dequant(w13_q[e], w13_s[e]).T
            g = h[:INTER].clamp(max=SWIGLU_LIMIT)
            u = h[INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            a = g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)
            a = a.to(torch.bfloat16).float()  # the kernels round stage 1 to bf16
            out[t] += float(topk_weights[t, j]) * (a @ _dequant(w2_q[e], w2_s[e]).T)
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


def _run(shuffled, x, topk_ids, topk_weights, fused_shared_expert=True):
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_decode import a16w8_decode_moe

    w13, w2, w13_s, w2_s = shuffled
    return a16w8_decode_moe(
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
        swiglu_alpha=SWIGLU_ALPHA,
        swiglu_limit=SWIGLU_LIMIT,
        fused_shared_expert=fused_shared_expert,
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


@pytest.mark.parametrize("m", [1, 2, 4, 8, 12, 16, 17, 32, 64, 128, 256])
def test_a16w8_decode_moe_matches_reference(m3_weights, m):
    """M <= 16: inline-sort path; 17..256: sort_decode + sorted GEMMs."""
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_decode import (
        MAX_DECODE_TOKENS,
        supports_batch,
        supports_shapes,
    )

    assert supports_shapes(HIDDEN, INTER)
    raw, shuffled = m3_weights
    torch.manual_seed(m)
    device = shuffled[0].device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device)
    assert supports_batch(x) and m <= MAX_DECODE_TOKENS
    out = _run(shuffled, x, topk_ids, topk_weights)
    ref_aiter = _run_aiter(shuffled, x, topk_ids, topk_weights)
    torch.accelerator.synchronize()
    assert out.shape == (m, HIDDEN) and out.dtype == torch.bfloat16
    ref = _float_reference(x, raw, topk_ids, topk_weights)
    scale = ref.abs().max().item()
    assert _cos(out, ref) > 0.9999
    assert (out.float() - ref).abs().max().item() < 0.02 * scale
    # aiter quantizes the activations to MXFP8 as well: ~0.999 against the float
    # reference, and the same distance from us
    assert _cos(out, ref_aiter) > 0.998


@pytest.mark.parametrize("m", [16, 256])
def test_a16w8_decode_moe_graph_replay(m3_weights, m):
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
    eager = [_run(shuffled, *inp).clone() for inp in inputs]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        _run(shuffled, *inputs[0])
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = [_run(shuffled, *inp) for inp in inputs]
    g.replay()
    g.replay()
    torch.accelerator.synchronize()
    for got, want in zip(outs, eager):
        assert _cos(got, want) > 0.9999


@pytest.mark.parametrize("m", [40, 192, 256])
def test_a16w8_decode_moe_separate_shared_expert(m3_weights, m):
    """The routing vLLM produces for ModelOpt MXFP8, where the shared expert is
    a separate module: top-k over the routed experts only, no expert routed by
    every token. The wide-first sort layout (from 40 tokens) budgets the last
    expert's blocks from M and must stay off; these M broke without the flag."""
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_decode import wide_for

    assert wide_for(m)
    raw, shuffled = m3_weights
    torch.manual_seed(m)
    device = shuffled[0].device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device)
    topk_ids, topk_weights = topk_ids[:, :TOPK].contiguous(), topk_weights[:, :TOPK]
    out = _run(shuffled, x, topk_ids, topk_weights, fused_shared_expert=False)
    ref_aiter = _run_aiter(shuffled, x, topk_ids, topk_weights.contiguous())
    torch.accelerator.synchronize()
    ref = _float_reference(x, raw, topk_ids, topk_weights)
    assert _cos(out, ref) > 0.9999
    assert _cos(out, ref_aiter) > 0.998
