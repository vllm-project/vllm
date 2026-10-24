# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Qwen3.5 gated full-attention decode layer vs the stock op chain."""

import pytest
import torch

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm-only fused decode layer", allow_module_level=True)

from vllm.model_executor.layers.rocm_qwen_full_attn_decode import (  # noqa: E402
    fused_attn,
    fused_back,
    fused_front,
)
from vllm.model_executor.layers.rotary_embedding import get_rope  # noqa: E402
from vllm.v1.attention.ops.chunked_prefill_paged_decode import (  # noqa: E402
    chunked_prefill_paged_decode,
)
from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (  # noqa: E402
    triton_reshape_and_cache_flash,
)

# Qwen3.8-27B full-attention layer shapes.
HID, NQH, NKV, HD, ROT = 5120, 24, 4, 256, 64
MROPE_SECTION = [11, 11, 10]
QG = NQH * 2 * HD
QKV = QG + 2 * NKV * HD
EPS = 1e-6
X = 8
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


def _make_caches(nblk, block, pad):
    """ROCM_ATTN K/V views of one buffer with a padded (hybrid) page stride."""
    half = NKV * HD * block
    page = 2 * half + pad
    raw = torch.randn(nblk * page, device=DEV, dtype=BF) * 0.5
    kc = raw.as_strided(
        (nblk, NKV, HD // X, block, X), (page, HD * block, block * X, X, 1)
    )
    vc = raw.as_strided(
        (nblk, NKV, HD, block), (page, HD * block, block, 1), storage_offset=half
    )
    return raw, kc, vc


def _stock(h, r, pos3, slot, kc, vc, bt, sl, nseq, w, rope):
    x, r2 = _add_norm(h, r, w["ln"])
    qkv = x @ w["qkv"].t()
    q_gate, k, v = qkv.split([QG, NKV * HD, NKV * HD], dim=-1)
    q, gate = torch.chunk(q_gate.view(-1, NQH, 2 * HD), 2, dim=-1)
    q = _gemma_norm(q, w["qn"]).reshape(-1, NQH * HD)
    k = _gemma_norm(k.reshape(-1, NKV, HD), w["kn"]).reshape(-1, NKV * HD)
    q, k = rope.forward_native(pos3, q, k)
    k3, v3 = k.view(-1, NKV, HD), v.reshape(-1, NKV, HD)
    kvs = torch.ones((), device=DEV, dtype=torch.float32)
    triton_reshape_and_cache_flash(k3, v3, kc, vc, slot, "auto", kvs, kvs)
    out = torch.zeros(q.shape[0], NQH, HD, device=DEV, dtype=BF)
    chunked_prefill_paged_decode(
        query=q.view(-1, NQH, HD)[:nseq],
        key=k3[:nseq],
        value=v3[:nseq],
        output=out[:nseq],
        kv_cache_dtype="auto",
        key_cache=kc,
        value_cache=vc,
        block_table=bt,
        query_start_loc=torch.arange(nseq + 1, device=DEV, dtype=torch.int32),
        seq_lens=sl,
        max_seq_len=int(sl.max()),
        max_query_len=1,
        k_scale=kvs,
        v_scale=kvs,
        sm_scale=HD**-0.5,
    )
    a = out.view(-1, NQH * HD) * torch.sigmoid(gate.reshape(-1, NQH * HD))
    return _add_norm(a @ w["o"].t(), r2, w["post"])


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


def _fused(h, r, pos3, slot, kc, vc, bt, sl, w, cs):
    q, gate, k, v, res_mid = fused_front(
        h, r, pos3, w["ln"], w["qkv"], w["qn"], w["kn"], cs, kc, vc, slot,
        NQH, NKV, HD, ROT, MROPE_SECTION, EPS,
    )  # fmt: skip
    attn = torch.empty_like(q)
    fused_attn(q, gate, attn, kc, vc, bt, sl, NQH, NKV, HD, HD**-0.5)
    return fused_back(attn, res_mid, w["o"], w["post"], EPS)


# (graph rows, real sequences, block size, page padding, context length,
# distinct MRoPE t/h/w positions); rows beyond the real sequences are
# CUDA-graph pad rows (slot -1).
@pytest.mark.parametrize(
    "n,nseq,block,pad,ctx,mrope",
    [
        (1, 1, 32, 0, 300, False),
        (4, 3, 48, 96, 2000, True),
        (8, 6, 544, 128, 8000, False),
        (32, 30, 48, 96, 1200, True),
    ],
)
@torch.inference_mode()
def test_fused_full_attn_decode_matches_stock(n, nseq, block, pad, ctx, mrope):
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
    max_blocks = (ctx + block - 1) // block
    nblk = nseq * max_blocks + 8
    raw_ref, kc_ref, vc_ref = _make_caches(nblk, block, pad)
    raw = raw_ref.clone()
    kc = raw.as_strided(kc_ref.shape, kc_ref.stride())
    vc = raw.as_strided(vc_ref.shape, vc_ref.stride(), vc_ref.storage_offset())
    bt = torch.randperm(nblk, device=DEV)[: nseq * max_blocks]
    bt = bt.view(nseq, max_blocks).to(torch.int32)
    sl = torch.randint(ctx // 2, ctx, (nseq,), device=DEV, dtype=torch.int32)
    sl[0] = 1
    pos = (sl - 1).long()
    slot = bt[torch.arange(nseq, device=DEV), pos // block].long() * block
    slot = slot + pos % block
    slot = torch.cat([slot, slot.new_full((n - nseq,), -1)])
    pos = torch.cat([pos, pos.new_zeros(n - nseq)])
    pos3 = pos.expand(3, n).contiguous()
    if mrope:
        pos3[1] = (pos3[1] - 7).clamp_min(0)
        pos3[2] = pos3[2] // 3
    h = torch.randn(n, HID, device=DEV, dtype=BF)
    r = 4 * torch.randn(n, HID, device=DEV, dtype=BF)

    x_ref, r_ref = _stock(h, r, pos3, slot, kc_ref, vc_ref, bt, sl, nseq, w, rope)

    cs = rope.cos_sin_cache.to(DEV)
    x, res = _fused(h, r, pos3, slot, kc, vc, bt, sl, w, cs)
    # bf16 rounding of intermediates differs from the stock chain by <= 1 ulp.
    assert _rel(x[:nseq], x_ref[:nseq]) < 5e-3
    assert _rel(res[:nseq], r_ref[:nseq]) < 5e-3
    assert _rel(raw, raw_ref) < 1e-3

    # Arrival counters must self-reset across HIP-graph replays.
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        xg, rg = _fused(h, r, pos3, slot, kc, vc, bt, sl, w, cs)
    for _ in range(50):
        g.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(xg, x, rtol=0, atol=0)
    torch.testing.assert_close(rg, res, rtol=0, atol=0)
