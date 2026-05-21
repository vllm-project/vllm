# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the CDNA INT8 per-token-head paged-prefill kernel.

Skips cleanly on non-ROCm and on ROCm hardware that is not gfx942 / gfx950 /
gfx90a, so this file is a no-op on CUDA / RDNA dev boxes.
"""

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

from vllm.v1.attention.ops.cdna_int8_prefill import (  # noqa: E402
    cdna_int8_paged_prefill,
    is_available,
)

if not is_available():
    pytest.skip(
        "paged_prefill_attn_cdna_int8 op not registered "
        "(vLLM must be built with PYTORCH_ROCM_ARCH=gfx942 or gfx950)",
        allow_module_level=True,
    )


def _quantize_int8_per_token_head(t_fp: torch.Tensor):
    """[..., heads, dim] fp16/bf16 -> (int8 cache, fp32 scale [..., heads])."""
    absmax = t_fp.abs().amax(dim=-1).clamp_min(1e-8)
    scale = (absmax / 127.0).to(torch.float32)
    q = (t_fp.to(torch.float32) / scale.unsqueeze(-1)).round()
    q = q.clamp(-128, 127).to(torch.int8)
    return q, scale


def _ref_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   sm_scale: float, causal: bool) -> torch.Tensor:
    """[M, Hq, D] x [N, Hkv, D] x [N, Hkv, D] -> [M, Hq, D] fp32 ref."""
    M, Hq, D = q.shape
    N, Hkv, _ = k.shape
    g = Hq // Hkv
    # Expand KV across the GQA groups.
    k = k.repeat_interleave(g, dim=1)
    v = v.repeat_interleave(g, dim=1)
    # [Hq, M, D] x [Hq, D, N] -> [Hq, M, N]
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
    return out.transpose(0, 1)  # [M, Hq, D]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize(
    "qlen,ctxlen,num_q_heads,num_kv_heads",
    [
        (16, 0, 4, 1),       # pure prefill, no context, GQA 4:1
        (32, 64, 8, 2),      # short prefill with prior context
        (64, 512, 16, 4),    # decode-shaped chunk with substantial context
        (128, 2048, 32, 8),  # large prefill, Qwen-shape
    ],
)
def test_cdna_int8_prefill_matches_reference(
    dtype, head_size, qlen, ctxlen, num_q_heads, num_kv_heads
):
    torch.manual_seed(0)
    device = "cuda"
    block_size = 16

    seq_len = qlen + ctxlen
    num_blocks_needed = (seq_len + block_size - 1) // block_size
    num_blocks_total = max(8, num_blocks_needed * 2)

    # Build a single-sequence batch (num_seqs=1) for clarity.
    q_fp = (0.1 * torch.randn(qlen, num_q_heads, head_size,
                              device=device, dtype=dtype))
    # Full K/V (context + current chunk), GQA-shaped.
    full_k_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size,
                                   device=device, dtype=dtype))
    full_v_fp = (0.1 * torch.randn(seq_len, num_kv_heads, head_size,
                                   device=device, dtype=dtype))

    # The kernel reads ctxlen tokens from the cache and qlen tokens from the
    # current chunk.
    k_chunk_fp = full_k_fp[ctxlen:].contiguous()
    v_chunk_fp = full_v_fp[ctxlen:].contiguous()

    # Per-token-head quantize the context portion into the int8 cache.
    ctx_k_fp = full_k_fp[:ctxlen]
    ctx_v_fp = full_v_fp[:ctxlen]
    k_cache_q, k_scale_ctx = (None, None)
    v_cache_q, v_scale_ctx = (None, None)
    if ctxlen > 0:
        k_cache_q, k_scale_ctx = _quantize_int8_per_token_head(ctx_k_fp)
        v_cache_q, v_scale_ctx = _quantize_int8_per_token_head(ctx_v_fp)

    # Allocate paged cache and populate the slots used by this sequence.
    k_cache = torch.zeros(num_blocks_total, block_size, num_kv_heads,
                          head_size, dtype=torch.int8, device=device)
    v_cache = torch.zeros_like(k_cache)
    k_scale_cache = torch.zeros(num_blocks_total, block_size, num_kv_heads,
                                dtype=torch.float32, device=device)
    v_scale_cache = torch.zeros_like(k_scale_cache)

    # Block table: identity 0..num_blocks_needed-1 for this single sequence.
    block_table = torch.zeros(1, max(1, num_blocks_needed), dtype=torch.int32,
                              device=device)
    for i in range(num_blocks_needed):
        block_table[0, i] = i

    if ctxlen > 0:
        for t in range(ctxlen):
            blk = t // block_size
            slot = t % block_size
            k_cache[blk, slot] = k_cache_q[t]
            v_cache[blk, slot] = v_cache_q[t]
            k_scale_cache[blk, slot] = k_scale_ctx[t]
            v_scale_cache[blk, slot] = v_scale_ctx[t]

    cu_seqlens_q = torch.tensor([0, qlen], dtype=torch.int32, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device=device)

    out = torch.empty_like(q_fp)
    sm_scale = float(head_size) ** -0.5

    cdna_int8_paged_prefill(
        out=out,
        q=q_fp,
        k_chunk=k_chunk_fp,
        v_chunk=v_chunk_fp,
        k_cache=k_cache,
        v_cache=v_cache,
        k_scale_cache=k_scale_cache,
        v_scale_cache=v_scale_cache,
        block_table=block_table,
        cu_seqlens_q=cu_seqlens_q,
        seq_lens=seq_lens,
        max_query_len=qlen,
        sm_scale=sm_scale,
        causal=True,
    )

    # Build the dequantised reference: cache portion goes through round-trip,
    # chunk portion stays in fp16/bf16.
    if ctxlen > 0:
        ctx_k_deq = (k_cache_q.to(torch.float32)
                     * k_scale_ctx.unsqueeze(-1)).to(dtype)
        ctx_v_deq = (v_cache_q.to(torch.float32)
                     * v_scale_ctx.unsqueeze(-1)).to(dtype)
        ref_k = torch.cat([ctx_k_deq, k_chunk_fp], dim=0)
        ref_v = torch.cat([ctx_v_deq, v_chunk_fp], dim=0)
    else:
        ref_k = k_chunk_fp
        ref_v = v_chunk_fp

    ref = _ref_attention(q_fp, ref_k, ref_v, sm_scale, causal=True)
    torch.testing.assert_close(out.to(torch.float32), ref, rtol=2e-2,
                               atol=2e-2)
