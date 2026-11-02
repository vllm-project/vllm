# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MiniMax-M3 FlyDSL MXFP8 MoE prefill chain vs aiter and a float reference, routed
only and with aiter's fused shared expert (expert 129, every token, weight 1)."""

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
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0
CHECK_TOKENS = 64


def _quantize(x: torch.Tensor):
    """Aiter's per-1x32 MX quant."""
    from aiter import dtypes
    from aiter.ops.quant import per_1x32_mx_quant_hip

    *lead, k = x.shape
    q, s = per_1x32_mx_quant_hip(
        x.reshape(-1, k), quant_dtype=dtypes.fp8, scale_type=dtypes.fp8_e8m0
    )
    return q.view(*lead, k), s.view(torch.uint8).view(*lead, k // 32)


def _dequant(q: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    n, k = q.shape
    sc = torch.ldexp(torch.ones_like(s, dtype=torch.float32), s.to(torch.int32) - 127)
    return (q.float().view(n, k // 32, 32) * sc.unsqueeze(-1)).view(n, k)


@pytest.fixture(
    scope="module", params=[NUM_ROUTED, NUM_ROUTED + 1], ids=["routed", "fse"]
)
def m3_weights(request):
    from vllm.model_executor.layers.fused_moe.oracle.fp8 import (
        Fp8MoeBackend,
        convert_to_fp8_moe_kernel_format,
    )

    torch.manual_seed(0)
    device = torch.device("cuda")
    e = request.param
    w13 = torch.randn((e, 2 * INTER, HIDDEN), device=device) * 0.02
    w2 = torch.randn((e, HIDDEN, INTER), device=device) * 0.02
    raw = (*_quantize(w13.bfloat16()), *_quantize(w2.bfloat16()))
    shuffled = convert_to_fp8_moe_kernel_format(
        fp8_backend=Fp8MoeBackend.AITER_MXFP8,
        layer=SimpleNamespace(weight_block_size=[1, 32]),
        w13=raw[0],
        w2=raw[2],
        w13_scale=raw[1],
        w2_scale=raw[3],
        w13_input_scale=None,
        w2_input_scale=None,
    )
    return raw, shuffled


def _routing(m: int, device, shared: bool):
    ids = torch.stack(
        [torch.randperm(NUM_ROUTED, device=device)[:TOPK] for _ in range(m)]
    )
    w = torch.rand((m, TOPK), device=device)
    w = w / w.sum(dim=1, keepdim=True) * 2.0
    if shared:
        ids = torch.cat([ids, torch.full((m, 1), NUM_ROUTED, device=device)], 1)
        w = torch.cat([w, torch.ones((m, 1), device=device)], dim=1)
    return ids.to(torch.int32).contiguous(), w.to(torch.float32).contiguous()


def _float_reference(x, raw, topk_ids, topk_weights, tokens):
    w13_q, w13_s, w2_q, w2_s = raw
    out = torch.zeros((len(tokens), HIDDEN), dtype=torch.float32, device=x.device)
    for i, t in enumerate(tokens):
        for j in range(topk_ids.shape[1]):
            e = int(topk_ids[t, j])
            h = x[t].float() @ _dequant(w13_q[e], w13_s[e]).T
            g = h[:INTER].clamp(max=SWIGLU_LIMIT)
            u = h[INTER:].clamp(-SWIGLU_LIMIT, SWIGLU_LIMIT)
            a = g * torch.sigmoid(SWIGLU_ALPHA * g) * (u + 1.0)
            out[i] += float(topk_weights[t, j]) * (a @ _dequant(w2_q[e], w2_s[e]).T)
    return out


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().flatten(), b.float().flatten()
    return float((a @ b) / (a.norm() * b.norm() + 1e-12))


@pytest.mark.parametrize("m", [4096, 16384, 32768])
def test_prefill_matches_aiter(m3_weights, m):
    """4096: 128-row gemm2 tiles; 16384: 256-row tiles; 32768: the large n-split."""
    from aiter import ActivationType, QuantType
    from aiter.fused_moe import fused_moe
    from aiter.ops.flydsl.moe_common import GateMode

    from vllm.models.minimax_m3.amd.ops.moe_mxfp8 import prefill

    raw, (w13, w2, w13_s, w2_s) = m3_weights
    torch.manual_seed(m)
    device = w13.device
    num_experts = w13.shape[0]
    x = torch.randn((m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids, topk_weights = _routing(m, device, num_experts > NUM_ROUTED)

    def ours():
        return prefill.a8w8_prefill_moe(
            _quantize(x),
            w13,
            w13_s,
            w2,
            w2_s,
            topk_weights,
            topk_ids,
            hidden_size=HIDDEN,
            intermediate_size=INTER,
            num_experts=num_experts,
        )

    out, out2 = ours(), ours()
    ref_aiter = fused_moe(
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
    torch.accelerator.synchronize()
    assert out.shape == (m, HIDDEN) and out.dtype == torch.bfloat16
    assert torch.equal(out, out2)  # no atomics: the same bits every call
    tokens = torch.randperm(m, device=device)[:CHECK_TOKENS].tolist()
    ref = _float_reference(x, raw, topk_ids, topk_weights, tokens)
    ours_t, theirs = out[tokens].float(), ref_aiter[tokens].float()
    assert _cos(ours_t, ref) > 0.997
    assert _cos(ours_t, ref) >= _cos(theirs, ref) - 1e-3
    err_ours = (ours_t - ref).abs().max().item()
    err_theirs = (theirs - ref).abs().max().item()
    assert err_ours <= err_theirs * 1.1 + 1e-3, (err_ours, err_theirs)
