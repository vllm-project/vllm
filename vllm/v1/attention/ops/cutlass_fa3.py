# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Vendored CUTLASS FA3 MLA attention kernel wrapper.

This module wraps the CUTLASS FlashAttention3 Sm90 kernel from sgl-attn,
providing a Python interface compatible with vLLM's sparse MLA attention
backend. The kernel is vendored as a self-contained C++ extension
(_cutlass_fa3_C) and does NOT depend on sglang, sgl_kernel, or any sgl*
modules.

The FA3 kernel supports MLA (Multi-head Latent Attention) with:
  - Separate Q_rope and QV (Q_nope) components
  - Paged KV cache with page_size=1
  - Variable-length sequences via cu_seqlens
  - Split-KV parallelism with automatic split count
  - SM90 (Hopper) CUTLASS warpgroup MMA + TMA

Source: https://github.com/sgl-project/sgl-attn (commit bcf72ccc)
"""

import torch

from vllm.platforms import current_platform

_cutlass_fa3_available = False
if current_platform.is_cuda():
    try:
        import vllm._cutlass_fa3_C  # noqa: F401

        _cutlass_fa3_available = True
    except ImportError:
        pass


def is_cutlass_fa3_available() -> bool:
    """Check if the CUTLASS FA3 extension is available.

    Requires CUDA >= 12.4 and SM90 (Hopper) GPU.
    """
    return _cutlass_fa3_available


def flash_attn_with_kvcache(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    k: torch.Tensor | None = None,
    v: torch.Tensor | None = None,
    qv: torch.Tensor | None = None,
    rotary_cos: torch.Tensor | None = None,
    rotary_sin: torch.Tensor | None = None,
    cache_seqlens: torch.Tensor | None = None,
    cache_batch_idx: torch.Tensor | None = None,
    cache_leftpad: torch.Tensor | None = None,
    page_table: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k_new: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    rotary_seqlens: torch.Tensor | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    attention_chunk: int | None = None,
    softcap: float = 0.0,
    rotary_interleaved: bool = True,
    scheduler_metadata: torch.Tensor | None = None,
    num_splits: int = 0,
    pack_gqa: bool | None = None,
    sm_margin: int = 0,
    return_softmax_lse: bool = False,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """CUTLASS FA3 attention with paged KV cache for MLA.

    MLA mode shapes (DeepSeek-V3.2, varlen mode with cu_seqlens_q):
      q:       [T, N, 64]      -- RoPE query (total_q=T, heads=N, dim=64)
      qv:      [T, N, 512]     -- NoPE query (total_q=T, heads=N, dim_v=512)
      k_cache: [S, 1, 1, 64]   -- Paged RoPE keys (pages, pg_sz=1, kv_h=1, d)
      v_cache: [S, 1, 1, 512]  -- Paged NoPE latent (pages, pg_sz=1, kv_h=1, dv)
      page_table: [T, topk]    -- Global cache slot indices per token

    The FA3 kernel internally computes:
      score = Q_rope @ K_rope^T + QV(Q_nope) @ V_cache(C_KV)^T
      output = softmax(score * scale) @ V_cache(C_KV)

    FA3 MLA constraints (from flash_api.cpp):
      headdim_qk <= 64, headdim_v >= 256, SM90 only, BF16/FP16 only

    FA3 produces 3 sub-kernels:
      1. prepare_varlen_num_blocks_kernel (scheduler)
      2. FlashAttnFwdSm90 (main attention, TMA+WGMMA)
      3. FlashAttnFwdCombine (split-KV merge, when num_splits > 1)

    Args:
        q: Query tensor for RoPE component.
        k_cache: Paged K cache (RoPE keys).
        v_cache: Paged V cache (NoPE latent).
        qv: Query tensor for NoPE/value component (MLA specific).
        page_table: Page table mapping tokens to cache slots.
        cache_seqlens: Number of valid KV entries per batch element.
        cu_seqlens_q: Cumulative query sequence lengths.
        cu_seqlens_k_new: Cumulative KV sequence lengths.
        max_seqlen_q: Maximum query sequence length.
        softmax_scale: Softmax scale factor (default: q.shape[-1]**-0.5).
        causal: Whether to apply causal masking.
        window_size: (left, right) attention window sizes.
        softcap: Logits soft cap value (0.0 = disabled).
        num_splits: Number of split-KV splits (0 = auto).
        return_softmax_lse: Whether to return log-sum-exp values.

    Returns:
        Attention output tensor, or tuple of (output, softmax_lse).
    """
    assert _cutlass_fa3_available, (
        "CUTLASS FA3 requires CUDA >= 12.4 and SM90 (Hopper) GPU. "
        "The _cutlass_fa3_C extension was not compiled or could not be loaded."
    )
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    attention_chunk_val = 0 if attention_chunk is None else int(attention_chunk)

    out, softmax_lse, *rest = torch.ops._cutlass_fa3_C.fwd.default(
        q,  # 0: q
        k_cache,  # 1: k (paged KV cache)
        v_cache,  # 2: v (paged KV cache)
        k,  # 3: k_new
        v,  # 4: v_new
        qv,  # 5: q_v (MLA NoPE query)
        None,  # 6: out buffer
        cu_seqlens_q,  # 7: cu_seqlens_q
        None,  # 8: cu_seqlens_k
        cu_seqlens_k_new,  # 9: cu_seqlens_k_new
        None,  # 10: seqused_q
        cache_seqlens,  # 11: seqused_k
        max_seqlen_q,  # 12: max_seqlen_q
        None,  # 13: max_seqlen_k
        page_table,  # 14: page_table
        cache_batch_idx,  # 15: kv_batch_idx
        cache_leftpad,  # 16: leftpad_k
        rotary_cos,  # 17: rotary_cos
        rotary_sin,  # 18: rotary_sin
        rotary_seqlens,  # 19: seqlens_rotary
        q_descale,  # 20: q_descale
        k_descale,  # 21: k_descale
        v_descale,  # 22: v_descale
        softmax_scale,  # 23: softmax_scale
        causal,  # 24: is_causal
        window_size[0],  # 25: window_size_left
        window_size[1],  # 26: window_size_right
        attention_chunk_val,  # 27: attention_chunk
        softcap,  # 28: softcap
        rotary_interleaved,  # 29: is_rotary_interleaved
        scheduler_metadata,  # 30: scheduler_metadata
        num_splits,  # 31: num_splits
        pack_gqa,  # 32: pack_gqa
        sm_margin,  # 33: sm_margin
        sinks,  # 34: sinks
    )
    return (out, softmax_lse) if return_softmax_lse else out
