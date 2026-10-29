# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused projections around Qwen3.5 gated full attention vs the stock ops."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only fused projections", allow_module_level=True)

from vllm.model_executor.layers.rocm_qwen_full_attn_decode import (  # noqa: E402
    fused_back,
    fused_front,
)
from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: E402

# Qwen3.8-27B full-attention layer shapes.
HID, NQH, NKV, HD, ROT = 5120, 24, 4, 256, 64
MROPE_SECTION = [11, 11, 10]
QG = NQH * 2 * HD
QKV = QG + 2 * NKV * HD
EPS = 1e-6
DEV = "cuda"
BF = torch.bfloat16


def _gemma_norm(x, w):
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    return (xf * (1.0 + w.float())).to(x.dtype)


def _add_norm(h, r, w):
    s = h.float() + r.float()
    xf = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + EPS)
    return (xf * (1.0 + w.float())).to(h.dtype), s.to(h.dtype)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@pytest.mark.parametrize("n,mrope", [(1, False), (5, True), (16, False), (32, True)])
@torch.inference_mode()
def test_fused_projections_match_stock(n, mrope):
    torch.manual_seed(0)
    with set_current_vllm_config(VllmConfig()):
        rope = get_rope(
            head_size=HD,
            max_position=262144,
            rope_parameters={
                "mrope_interleaved": True,
                "mrope_section": MROPE_SECTION,
                "partial_rotary_factor": 0.25,
                "rope_theta": 10000000,
                "rope_type": "default",
            },
            dtype=BF,
        ).to(DEV)
    w = {
        "ln": 0.1 * torch.randn(HID, device=DEV, dtype=BF),
        "post": 0.1 * torch.randn(HID, device=DEV, dtype=BF),
        "qn": 0.1 * torch.randn(HD, device=DEV, dtype=BF),
        "kn": 0.1 * torch.randn(HD, device=DEV, dtype=BF),
        "qkv": 0.02 * torch.randn(QKV, HID, device=DEV, dtype=BF),
        "o": 0.02 * torch.randn(HID, NQH * HD, device=DEV, dtype=BF),
    }
    pos = torch.randint(0, 20000, (n,), device=DEV)
    pos3 = pos.expand(3, n).contiguous()
    if mrope:
        pos3[1] = (pos3[1] - 7).clamp_min(0)
        pos3[2] = pos3[2] // 3
    h = torch.randn(n, HID, device=DEV, dtype=BF)
    r = 4 * torch.randn(n, HID, device=DEV, dtype=BF)
    attn = torch.randn(n, NQH * HD, device=DEV, dtype=BF)

    x, r_mid_ref = _add_norm(h, r, w["ln"])
    qkv = x @ w["qkv"].t()
    q_gate, k_ref, v_ref = qkv.split([QG, NKV * HD, NKV * HD], dim=-1)
    q_ref, gate_ref = torch.chunk(q_gate.view(-1, NQH, 2 * HD), 2, dim=-1)
    q_ref = _gemma_norm(q_ref, w["qn"]).reshape(-1, NQH * HD)
    gate_ref = gate_ref.reshape(-1, NQH * HD)
    k_ref = _gemma_norm(k_ref.reshape(-1, NKV, HD), w["kn"]).reshape(-1, NKV * HD)
    q_ref, k_ref = rope.forward_native(pos3, q_ref, k_ref)
    gated = attn * torch.sigmoid(gate_ref)
    x2_ref, r2_ref = _add_norm(gated @ w["o"].t(), r_mid_ref, w["post"])

    cs = rope.cos_sin_cache.to(DEV)

    def run(gate_in):
        q, gate, k, v, r_mid = fused_front(
            h, r, pos3, w["ln"], w["qkv"], w["qn"], w["kn"], cs,
            NQH, NKV, HD, ROT, MROPE_SECTION, EPS,
        )  # fmt: skip
        x2, r2 = fused_back(attn, gate_in, r_mid, w["o"], w["post"], EPS)
        return q, gate, k, v, r_mid, x2, r2

    q, gate, k, v, r_mid, x2, r2 = run(gate_ref)
    # The fused path rounds intermediates to bf16 at different points.
    for got, ref in [
        (r_mid, r_mid_ref),
        (q, q_ref),
        (gate, gate_ref),
        (k, k_ref),
        (v, v_ref),
        (x2, x2_ref),
        (r2, r2_ref),
    ]:
        assert _rel(got, ref) < 5e-3

    # Arrival counters must self-reset across HIP-graph replays.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        outs = run(gate_ref)
    for _ in range(50):
        g.replay()
    torch.accelerator.synchronize()
    for got, ref in zip(outs, (q, gate, k, v, r_mid, x2, r2)):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
