# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL a8w8 mid-batch MoE (gfx950, MXFP8 weights and activations,
256 < M < 3072): sort in 32/64/128-row blocks, gemm1 with the A tile through
LDS, gemm2 accumulating bf16 atomically (no partials, no reduction).

Weights are quantized to MXFP8 and shuffled like ``ModelOptMxFp8FusedMoE`` does
for the AITER_MXFP8 backend; routing mimics aiter's fused shared expert. The
output is compared with the production aiter a8w8 call (whose stage 2 is
atomic as well at these batch sizes) and, on a sample of tokens, with a float
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


def _run(shuffled, x, topk_ids, topk_weights):
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_mid import a8w8_mid_moe

    w13, w2, w13_s, w2_s = shuffled
    return a8w8_mid_moe(
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
    )


@pytest.mark.parametrize("m", [300, 512, 1024, 2048, 3071])
def test_a8w8_mid_moe_matches_aiter(m3_weights, m):
    from vllm.models.minimax_m3.amd.ops.moe_a8w8_mid import (
        MAX_MID_TOKENS,
        MIN_MID_TOKENS,
        block_m_for,
    )

    assert MIN_MID_TOKENS <= m <= MAX_MID_TOKENS
    assert block_m_for(m) in (32, 64, 128)
    raw, shuffled = m3_weights
    torch.manual_seed(m)
    device = shuffled[0].device
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device)

    out = _run(shuffled, x, topk_ids, topk_weights)
    out2 = _run(shuffled, x, topk_ids, topk_weights)
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
    ref = _float_reference(x, raw, topk_ids, topk_weights, tokens)
    c_ours = _cos(out[tokens], ref)
    c_aiter = _cos(ref_aiter[tokens], ref)
    assert c_ours > 0.998 and c_ours > c_aiter - 1e-4, (c_ours, c_aiter)
