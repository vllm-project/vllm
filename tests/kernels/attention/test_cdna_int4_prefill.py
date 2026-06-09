# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CDNA INT4 per-token-head paged-prefill kernel."""

from __future__ import annotations

import pytest
import torch

from vllm.platforms import current_platform

if not current_platform.is_rocm():
    pytest.skip("ROCm only", allow_module_level=True)

from vllm.platforms.rocm import on_mi3xx  # noqa: E402

if not on_mi3xx():
    pytest.skip("CDNA (gfx942 / gfx950 / gfx90a) only",
                allow_module_level=True)

from vllm.v1.attention.ops.cdna_int4_prefill import (  # noqa: E402
    cdna_int4_paged_prefill,
    is_available,
    pack_scale_zp,
)

if not is_available():
    pytest.skip("paged_prefill_attn_cdna_int4 op not registered",
                allow_module_level=True)


def _quantize_int4_per_token_head(t_fp: torch.Tensor):
    """Asymmetric uint4 per-(token, head) quantization.

    Returns (packed_uint8 [..., heads, dim/2], scale_fp32, zp_int32).
    Element at byte index i contains: low nibble = element 2i,
                                       high nibble = element 2i+1.
    Reconstruction: x ≈ (q[i] - zp) * scale.
    """
    fp32 = t_fp.to(torch.float32)
    lo = fp32.amin(dim=-1)
    hi = fp32.amax(dim=-1)
    rng = (hi - lo).clamp_min(1e-8)
    scale = (rng / 15.0).clamp_min(1e-8)
    zp_real = -lo / scale
    zp = zp_real.round().clamp(0, 15).to(torch.int32)
    q_fp = (fp32 / scale.unsqueeze(-1) + zp.unsqueeze(-1)).round()
    q = q_fp.clamp(0, 15).to(torch.uint8)
    # Pack pairs of nibbles.
    even = q[..., 0::2]
    odd = q[..., 1::2]
    packed = (even | (odd << 4)).contiguous()
    return packed, scale, zp


def _dequantize_int4(packed: torch.Tensor, scale: torch.Tensor,
                     zp: torch.Tensor, head_size: int, dtype):
    """Round-trip packed nibbles back to fp16/bf16 for the reference."""
    even = packed & 0xF
    odd = (packed >> 4) & 0xF
    # Interleave back: [..., dim] = stack(even, odd) along last axis.
    out_shape = packed.shape[:-1] + (head_size,)
    out = torch.empty(out_shape, device=packed.device, dtype=torch.float32)
    out[..., 0::2] = (even.to(torch.float32)
                      - zp.unsqueeze(-1).to(torch.float32))
    out[..., 1::2] = (odd.to(torch.float32)
                      - zp.unsqueeze(-1).to(torch.float32))
    out = out * scale.unsqueeze(-1).to(torch.float32)
    return out.to(dtype)


def _ref_attention(q, k, v, sm_scale, causal):
    M, Hq, D = q.shape
    N, Hkv, _ = k.shape
    g = Hq // Hkv
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    qf = q.to(torch.float32).transpose(0, 1)
    kf = k.to(torch.float32).transpose(0, 1)
    vf = v.to(torch.float32).transpose(0, 1)
    scores = torch.einsum("hmd,hnd->hmn", qf, kf) * sm_scale
    if causal:
        m_idx = torch.arange(M, device=q.device).unsqueeze(1)
        n_idx = torch.arange(N, device=q.device).unsqueeze(0)
        ctx = N - M
        mask = (ctx + m_idx) >= n_idx
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    out = torch.einsum("hmn,hnd->hmd", weights, vf)
    return out.transpose(0, 1)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize(
    "qlen,ctxlen,num_q_heads,num_kv_heads",
    [
        (16, 0, 4, 1),
        (32, 64, 8, 2),
        (64, 512, 16, 4),
        (128, 1024, 32, 8),
    ],
)
def test_cdna_int4_prefill_matches_reference(
    dtype, head_size, qlen, ctxlen, num_q_heads, num_kv_heads
):
    torch.manual_seed(0)
    device = "cuda"
    block_size = 16

    seq_len = qlen + ctxlen
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    num_blocks_total = max(8, num_blocks_needed * 2)

    q_fp = (0.1 * torch.randn(qlen, num_q_heads, head_size,
                              device=device, dtype=dtype))
    full_k_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size,
                                   device=device, dtype=dtype))
    full_v_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size,
                                   device=device, dtype=dtype))

    k_chunk_fp = full_k_fp[ctxlen:].contiguous()
    v_chunk_fp = full_v_fp[ctxlen:].contiguous()

    k_cache = torch.zeros(num_blocks_total, block_size, num_kv_heads,
                          head_size // 2, dtype=torch.uint8, device=device)
    v_cache = torch.zeros_like(k_cache)
    k_scale_cache = torch.zeros(num_blocks_total, block_size, num_kv_heads,
                                dtype=torch.float32, device=device)
    v_scale_cache = torch.zeros_like(k_scale_cache)

    block_table = torch.zeros(1, max(1, num_blocks_needed), dtype=torch.int32,
                              device=device)
    for i in range(num_blocks_needed):
        block_table[0, i] = i

    deq_ctx_k = None
    deq_ctx_v = None
    if ctxlen > 0:
        ctx_k_fp = full_k_fp[:ctxlen]
        ctx_v_fp = full_v_fp[:ctxlen]
        k_packed, k_scale, k_zp = _quantize_int4_per_token_head(ctx_k_fp)
        v_packed, v_scale, v_zp = _quantize_int4_per_token_head(ctx_v_fp)
        k_steg = pack_scale_zp(k_scale, k_zp)
        v_steg = pack_scale_zp(v_scale, v_zp)
        for t in range(ctxlen):
            blk = t // block_size
            slot = t % block_size
            k_cache[blk, slot] = k_packed[t]
            v_cache[blk, slot] = v_packed[t]
            k_scale_cache[blk, slot] = k_steg[t]
            v_scale_cache[blk, slot] = v_steg[t]
        # Build the reference K/V by dequantizing the packed cache.
        deq_ctx_k = _dequantize_int4(k_packed, k_scale, k_zp, head_size, dtype)
        deq_ctx_v = _dequantize_int4(v_packed, v_scale, v_zp, head_size, dtype)

    cu_seqlens_q = torch.tensor([0, qlen], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    out = torch.empty_like(q_fp)
    sm_scale = float(head_size) ** -0.5
    cdna_int4_paged_prefill(
        out=out, q=q_fp, k_chunk=k_chunk_fp, v_chunk=v_chunk_fp,
        k_cache=k_cache, v_cache=v_cache,
        k_scale_cache=k_scale_cache, v_scale_cache=v_scale_cache,
        block_table=block_table, cu_seqlens_q=cu_seqlens_q,
        seq_lens=seq_lens, max_query_len=qlen, sm_scale=sm_scale, causal=True,
    )

    if ctxlen > 0:
        ref_k = torch.cat([deq_ctx_k, k_chunk_fp], dim=0)
        ref_v = torch.cat([deq_ctx_v, v_chunk_fp], dim=0)
    else:
        ref_k = k_chunk_fp
        ref_v = v_chunk_fp
    ref = _ref_attention(q_fp, ref_k, ref_v, sm_scale, causal=True)
    # Tolerance rationale: the KV cache is INT4-quantized (4-bit, 16 levels)
    # vs INT8's 256, so the quant step — and hence the per-element RMS
    # relative quant error — is ~16x larger (~6-7% per element). Averaging
    # over the head_size-wide Q·K / P·V dot products pulls the aggregate
    # output error down to ~2%, with fp16 / MFMA accumulation and the softmax
    # as the floor. rtol = atol = 4e-2 (2x the INT8 tolerance) gives ~2x
    # margin over the observed error, confirmed across the swept shapes.
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=4e-2,
                               atol=4e-2)
