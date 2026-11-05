# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL W4A4 MoE prefill chain at TP4 vs aiter's ``fused_moe`` (the
``AITER_MXFP4_MXFP4`` backend's call) and a float reference, routed only and with
aiter's fused shared expert (expert 129, every token, weight 1)."""

import pytest
import torch

from vllm.platforms import current_platform
from vllm.platforms.rocm import on_gfx950

pytestmark = [
    pytest.mark.skipif(not current_platform.is_rocm(), reason="ROCm only"),
    pytest.mark.skipif(not on_gfx950(), reason="gfx950 only"),
]

HIDDEN, INTER, NUM_ROUTED, TOPK = 6144, 768, 128, 4
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0
CHECK_TOKENS = 64


def _quantize_like_backend(w13: torch.Tensor, w2: torch.Tensor):
    """Per-1x32 MXFP4 + the shuffles oracle/mxfp4.py applies for
    AITER_MXFP4_MXFP4."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_f4_quant
    from aiter.utility.fp4_utils import e8m0_shuffle

    from vllm._aiter_ops import rocm_aiter_ops

    e = w13.shape[0]
    fp4 = torch.float4_e2m1fn_x2
    w13_q, w13_s = per_1x32_f4_quant(w13, quant_dtype=dtypes.fp4x2)
    w2_q, w2_s = per_1x32_f4_quant(w2, quant_dtype=dtypes.fp4x2)
    w13_q = w13_q.view(e, w13.shape[1], w13.shape[2] // 2)
    w2_q = w2_q.view(e, w2.shape[1], w2.shape[2] // 2)
    w13_s = w13_s.view(torch.uint8).view(e, w13.shape[1], w13.shape[2] // 32)
    w2_s = w2_s.view(torch.uint8).view(e, w2.shape[1], w2.shape[2] // 32)
    raw = (w13_q, w13_s, w2_q, w2_s)
    w13_k, w2_k = rocm_aiter_ops.shuffle_weights(w13_q.view(fp4), w2_q.view(fp4))
    w13_sk = e8m0_shuffle(w13_s.view(e * w13.shape[1], -1)).view(e, w13.shape[1], -1)
    w2_sk = e8m0_shuffle(w2_s.view(e * w2.shape[1], -1)).view(e, w2.shape[1], -1)
    shuffled = (w13_k.contiguous(), w13_sk.contiguous())
    return raw, shuffled + (w2_k.contiguous(), w2_sk.contiguous())


def _routing(m: int, shared: bool, device):
    routed = torch.stack(
        [torch.randperm(NUM_ROUTED, device=device)[:TOPK] for _ in range(m)]
    )
    w = torch.rand((m, TOPK), device=device)
    w = w / w.sum(dim=1, keepdim=True)
    if shared:
        routed = torch.cat([routed, torch.full((m, 1), NUM_ROUTED, device=device)], 1)
        w = torch.cat([w, torch.ones((m, 1), device=device)], dim=1)
    return routed.to(torch.int32), w.to(torch.float32)


def _dequant(q: torch.Tensor, s: torch.Tensor, n_cols: int) -> torch.Tensor:
    from aiter.utility import fp4_utils

    v = fp4_utils.mxfp4_to_f32(q.view(torch.uint8)).view(q.shape[0], -1)
    sc = fp4_utils.e8m0_to_f32(s.view(torch.uint8)).view(q.shape[0], -1)
    return (v.view(q.shape[0], -1, 32) * sc.unsqueeze(-1)).view(q.shape[0], n_cols)


_FP4_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _fp4_quant_dequant(h: torch.Tensor) -> torch.Tensor:
    """Per-32 MXFP4 of fp32 rows: e8m0 scale ceil_pow2(amax / 6) (aiter's RoundUp,
    computed on the f32 bits like the kernels), round-to-nearest-even onto e2m1."""
    g = h.view(*h.shape[:-1], -1, 32)
    amax = g.abs().amax(-1, keepdim=True)
    bits = (amax * torch.tensor(1.0 / 6.0, dtype=torch.float32)).view(torch.int32)
    e = (bits >> 23) & 0xFF
    e = e + (((bits & 0x7FFFFF) != 0) & (e < 255)).to(torch.int32)
    scale = (e << 23).view(torch.float32)
    v = (g / torch.where(scale > 0, scale, 1.0)).abs()
    mids = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=h.device)
    idx = torch.bucketize(v, mids)  # a tie goes to the lower code ...
    idx = idx + ((v == 0.75) | (v == 1.75) | (v == 3.5)).to(idx.dtype)  # ... or even
    q = torch.tensor(_FP4_GRID, device=h.device)[idx] * g.sign()
    return (q * scale).view_as(h)


def _float_reference(x, raw, topk_ids, topk_weights, tokens, same_operands):
    """fp32 MoE on the dequantized experts. ``same_operands``: the kernel's x (the
    per-token MXFP4 codes) and its intermediate rounding (bf16, then MXFP4);
    otherwise bf16 x and no intermediate quant (the W4A4 error, cos ~0.98)."""
    from vllm.models.minimax_m3.amd.ops.moe_mxfp4.prefill import mxfp4_quant

    w13_q, w13_s, w2_q, w2_s = raw
    xs = x[tokens]
    if same_operands:
        q, s = mxfp4_quant(xs, None, None)
        xs = _dequant(q, s, HIDDEN)
    xs = xs.float()
    deq = {}
    out = torch.zeros((len(tokens), HIDDEN), dtype=torch.float32, device=x.device)
    for i, t in enumerate(tokens):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            if e not in deq:
                deq[e] = (
                    _dequant(w13_q[e], w13_s[e], HIDDEN),
                    _dequant(w2_q[e], w2_s[e], INTER),
                )
            w13e, w2e = deq[e]
            h = xs[i] @ w13e.T
            g = h[:INTER].clamp(max=SWIGLU_LIMIT)
            u = h[INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            a = g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)
            if same_operands:
                a = _fp4_quant_dequant(a.bfloat16().float())
            out[i] += float(topk_weights[t, j]) * (a @ w2e.T)
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@pytest.fixture(scope="module", params=[False, True], ids=["routed", "fused_shared"])
def weights(request):
    torch.manual_seed(0)
    num_experts = NUM_ROUTED + int(request.param)
    w13 = torch.randn(
        (num_experts, 2 * INTER, HIDDEN), dtype=torch.bfloat16, device="cuda"
    )
    w2 = torch.randn((num_experts, HIDDEN, INTER), dtype=torch.bfloat16, device="cuda")
    raw, shuffled = _quantize_like_backend(w13 * 0.02, w2 * 0.02)
    return request.param, raw, shuffled


def _run(shuffled, x, topk_ids, topk_weights):
    from vllm.models.minimax_m3.amd.ops.moe_mxfp4.prefill import (
        a4w4_prefill_moe,
        mxfp4_quant,
    )

    w13, w13_s, w2, w2_s = shuffled
    return a4w4_prefill_moe(
        mxfp4_quant(x, None, None),
        w13,
        w13_s,
        w2,
        w2_s,
        topk_weights,
        topk_ids,
        hidden_size=HIDDEN,
        intermediate_size=INTER,
        num_experts=w13.shape[0],
    )


def _run_aiter(shuffled, x, topk_ids, topk_weights):
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    w13, w13_s, w2, w2_s = shuffled
    return fused_moe(
        x,
        w13,
        w2,
        topk_weights,
        topk_ids,
        quant_type=QuantType.per_1x32,
        activation=ActivationType.Swiglu,
        w1_scale=w13_s,
        w2_scale=w2_s,
        swiglu_limit=SWIGLU_LIMIT,
        gate_mode=GateMode.SEPARATED.value,
    )


@pytest.mark.parametrize("m", [4096, 16384, 32768])
def test_moe_mxfp4_prefill(weights, m):
    """4096: 128-row sort blocks; 16384 and 32768: 256-row blocks."""
    from vllm.models.minimax_m3.amd.ops.moe_mxfp4.prefill import supports_shapes

    assert supports_shapes(HIDDEN, INTER)
    shared, raw, shuffled = weights
    torch.manual_seed(m)
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device="cuda")
    topk_ids, topk_weights = _routing(m, shared, x.device)
    out = _run(shuffled, x, topk_ids, topk_weights)
    out2 = _run(shuffled, x, topk_ids, topk_weights)
    ref_aiter = _run_aiter(shuffled, x, topk_ids, topk_weights)
    torch.accelerator.synchronize()
    assert out.shape == (m, HIDDEN) and out.dtype == torch.bfloat16
    assert torch.equal(out, out2)  # no atomics: the same bits every call
    tokens = torch.randperm(m, device=x.device)[:CHECK_TOKENS].tolist()
    ref = _float_reference(x, raw, topk_ids, topk_weights, tokens, True)
    ref_hp = _float_reference(x, raw, topk_ids, topk_weights, tokens, False)
    cos, cos_hp = _cos(out[tokens], ref), _cos(out[tokens], ref_hp)
    aiter_hp = _cos(ref_aiter[tokens], ref_hp)
    print(
        f"\nRESULT m={m} shared={shared}: cos vs same-operand ref {cos:.6f}, vs "
        f"high-precision ref {cos_hp:.6f} (aiter {aiter_hp:.6f})"
    )
    # the kernel computes what it should; W4A4 loses no more than aiter's path
    assert cos > 0.9999
    assert cos_hp >= aiter_hp - 1e-3
