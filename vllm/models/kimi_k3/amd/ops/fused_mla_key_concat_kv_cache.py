# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: mirror of vllm/models/kimi_k3/nvidia/ops/fused_mla_key_concat_kv_cache.py
"""Fused MLA prefill and decode epilogues for Kimi-K3.

Thin wrappers over the CUDA ops in
``csrc/libtorch_stable/fused_kimi_k3_mla_key_concat_kv_cache_kernel.cu``, which
mirror ``fused_deepseek_v4_qnorm_rope_kv_rope_full_cache_{bf16,fp8}_insert``.

- ``fused_mla_key_concat_kv_cache_insert`` (bf16): optionally apply RoPE,
  concat the full per-head key ``[k_nope | k_pe]`` into ``k_out``, and insert
  the latent ``[kv_c_normed | k_pe]`` into the paged cache.
- ``fused_mla_qkv_quant_kv_cache_fp8_insert`` (fp8): additionally quantize
  ``q``/``k``/``v`` to E4M3 with ``q_scale`` / ``k_scale`` / ``v_scale`` (the
  cache shares ``k_scale``, as in ``concat_and_cache_mla``).

The optional ``positions`` / ``cos_sin_cache`` pair enables GPT-J-style RoPE
inside the epilogue. Omitting both keeps the K3 NoPE fast path. The kernels use
Programmatic Dependent Launch to overlap the tail of the producing GEMMs on
sm_90+.
"""

import torch


def fused_mla_key_concat_kv_cache_insert(
    q: torch.Tensor,  # [Tp, H, qk_head_dim], RoPE is applied in place
    k_nope: torch.Tensor,  # [Tp, H, qk_nope_head_dim]
    k_pe: torch.Tensor,  # [Tp, qk_rope_head_dim] or [Tp, 1, qk_rope_head_dim]
    kv_c_normed: torch.Tensor,  # [Tp, kv_lora_rank]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, kv_lora_rank + rope]
    slot_mapping: torch.Tensor,  # [Tp] int64
    positions: torch.Tensor | None = None,  # [Tp] int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_position, rope]
) -> torch.Tensor:
    """Apply optional RoPE, concat K, and insert the paged latent (bf16).

    Returns the full key ``[Tp, H, qk_nope_head_dim + qk_rope_head_dim]``;
    optionally rotates ``q`` and writes ``kv_cache`` in place.
    """
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    tp, num_heads, qk_nope_head_dim = k_nope.shape
    qk_head_dim = qk_nope_head_dim + k_pe.shape[1]
    k_out = torch.empty(
        (tp, num_heads, qk_head_dim), dtype=k_nope.dtype, device=k_nope.device
    )
    if tp == 0:
        return k_out
    torch.ops._C.fused_kimi_k3_mla_key_concat_kv_cache_insert(
        q,
        k_nope,
        k_pe,
        kv_c_normed,
        k_out,
        kv_cache,
        slot_mapping,
        kv_cache.shape[1],
        positions,
        cos_sin_cache,
    )
    return k_out


def fused_mla_key_concat_ds_mla_insert(
    q: torch.Tensor,  # [Tp, H, qk_head_dim], RoPE is applied in place
    k_nope: torch.Tensor,  # [Tp, H, qk_nope_head_dim]
    k_pe: torch.Tensor,  # [Tp, qk_rope_head_dim] or [Tp, 1, qk_rope_head_dim]
    kv_c_normed: torch.Tensor,  # [Tp, kv_lora_rank]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, 656] uint8 (fp8_ds_mla)
    slot_mapping: torch.Tensor,  # [Tp] int64
    positions: torch.Tensor | None = None,  # [Tp] int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_position, rope]
) -> torch.Tensor:
    """Concat full K (bf16) and insert the latent in the fp8_ds_mla layout.

    The cache uses DeepSeek's 656-byte block-scaled layout (NoPE in 4 tiles of
    128 with per-tile dynamic fp8 scales, RoPE as bf16) -- self-scaling, so no
    scale argument. Returns the bf16 full key; optionally rotates ``q`` and
    writes ``kv_cache`` in place.
    """
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    tp, num_heads, qk_nope_head_dim = k_nope.shape
    qk_head_dim = qk_nope_head_dim + k_pe.shape[1]
    k_out = torch.empty(
        (tp, num_heads, qk_head_dim), dtype=k_nope.dtype, device=k_nope.device
    )
    if tp == 0:
        return k_out
    torch.ops._C.fused_kimi_k3_mla_key_concat_ds_mla_insert(
        q,
        k_nope,
        k_pe,
        kv_c_normed,
        k_out,
        kv_cache,
        slot_mapping,
        kv_cache.shape[1],
        positions,
        cos_sin_cache,
    )
    return k_out


def fused_mla_qkv_quant_kv_cache_fp8_insert(
    q: torch.Tensor,  # [Tp, H, qk_head_dim]
    k_nope: torch.Tensor,  # [Tp, H, qk_nope_head_dim]
    k_pe: torch.Tensor,  # [Tp, qk_rope_head_dim] or [Tp, 1, qk_rope_head_dim]
    kv_c_normed: torch.Tensor,  # [Tp, kv_lora_rank]
    v: torch.Tensor,  # [Tp, H, v_head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, kv_lora_rank + rope] fp8
    slot_mapping: torch.Tensor,  # [Tp] int64
    q_scale_inv: torch.Tensor,  # scalar fp32, 1 / q scale (attention query)
    k_scale_inv: torch.Tensor,  # scalar fp32, 1 / k scale (attention key)
    v_scale_inv: torch.Tensor,  # scalar fp32, 1 / v scale (attention value)
    cache_scale_inv: torch.Tensor,  # scalar fp32, 1 / kv scale (cache latent)
    positions: torch.Tensor | None = None,  # [Tp] int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_position, rope]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize q/k/v to fp8 and insert the fp8 latent into the paged cache.

    The attention key ``k_fp8`` and the cache latent use *separate* scales
    (``k_scale_inv`` vs ``cache_scale_inv``): the cache must be quantized with
    ``_k_scale`` (read back by decode / context), while the prefill attention
    q/k/v currently stay unscaled (the prefill flash path does not dequantize).

    Returns ``(q_fp8, k_fp8, v_fp8)``; writes the fp8 ``kv_cache`` in place.
    """
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    tp, num_heads, _ = q.shape
    qk_head_dim = q.shape[2]
    v_head_dim = v.shape[2]
    fp8 = torch.float8_e4m3fn
    q_fp8 = torch.empty((tp, num_heads, qk_head_dim), dtype=fp8, device=q.device)
    k_fp8 = torch.empty((tp, num_heads, qk_head_dim), dtype=fp8, device=q.device)
    v_fp8 = torch.empty((tp, num_heads, v_head_dim), dtype=fp8, device=q.device)
    if tp == 0:
        return q_fp8, k_fp8, v_fp8
    torch.ops._C.fused_kimi_k3_mla_qkv_quant_kv_cache_fp8_insert(
        q,
        k_nope,
        k_pe,
        kv_c_normed,
        v,
        q_fp8,
        k_fp8,
        v_fp8,
        kv_cache,
        slot_mapping,
        q_scale_inv,
        k_scale_inv,
        v_scale_inv,
        cache_scale_inv,
        kv_cache.shape[1],
        positions,
        cos_sin_cache,
    )
    return q_fp8, k_fp8, v_fp8


def fused_mla_decode_q_concat_kv_cache_insert(
    ql_nope: torch.Tensor,  # [B, H, kv_lora_rank]  (BMM1 output, absorbed q)
    q_pe: torch.Tensor,  # [B, H, qk_rope_head_dim]
    kv_c_normed: torch.Tensor,  # [B, kv_lora_rank]
    k_pe: torch.Tensor,  # [B, qk_rope_head_dim] or [B, 1, qk_rope_head_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, entry]
    slot_mapping: torch.Tensor,  # [B] int64
    *,
    ds_mla: bool = False,
    q_scale_inv: torch.Tensor | None = None,  # scalar fp32, 1 / q scale
    cache_scale_inv: torch.Tensor | None = None,  # scalar fp32, 1 / kv scale
    positions: torch.Tensor | None = None,  # [B] int64
    cos_sin_cache: torch.Tensor | None = None,  # [max_position, rope]
) -> torch.Tensor:
    """Concat the latent decode query ``mqa_q = [ql_nope | q_pe]`` and insert the
    latent ``[kv_c_normed | k_pe]`` into the paged cache, in one launch (runs
    right before ``forward_mqa``).

    Dispatched by cache format:
      - bf16          -> bf16 mqa_q, bf16 cache
      - plain fp8     -> fp8 mqa_q (q_scale_inv), fp8 cache (cache_scale_inv)
      - fp8_ds_mla    -> bf16 mqa_q, 656B block-scaled cache

    Returns ``mqa_q`` of shape ``[B, H, kv_lora_rank + qk_rope_head_dim]``;
    writes ``kv_cache`` in place.
    """
    k_pe = k_pe.reshape(k_pe.shape[0], -1)
    b, num_heads, kv_lora_rank = ql_nope.shape
    entry = kv_lora_rank + q_pe.shape[-1]
    fp8_q = q_scale_inv is not None
    out_dtype = torch.float8_e4m3fn if fp8_q else ql_nope.dtype
    mqa_q = torch.empty((b, num_heads, entry), dtype=out_dtype, device=ql_nope.device)
    if b == 0:
        return mqa_q

    if ds_mla:
        cache = (
            kv_cache if kv_cache.dtype == torch.uint8 else kv_cache.view(torch.uint8)
        )
        torch.ops._C.fused_kimi_k3_mla_decode_q_concat_ds_mla_insert(
            ql_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            mqa_q,
            cache,
            slot_mapping,
            cache.shape[1],
            positions,
            cos_sin_cache,
        )
    elif fp8_q:
        assert cache_scale_inv is not None, "fp8 decode requires cache_scale_inv"
        cache = (
            kv_cache
            if kv_cache.dtype == torch.float8_e4m3fn
            else kv_cache.view(torch.float8_e4m3fn)
        )
        torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_fp8_insert(
            ql_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            mqa_q,
            cache,
            slot_mapping,
            q_scale_inv,
            cache_scale_inv,
            cache.shape[1],
            positions,
            cos_sin_cache,
        )
    else:
        torch.ops._C.fused_kimi_k3_mla_decode_q_concat_kv_cache_insert(
            ql_nope,
            q_pe,
            kv_c_normed,
            k_pe,
            mqa_q,
            kv_cache,
            slot_mapping,
            kv_cache.shape[1],
            positions,
            cos_sin_cache,
        )
    return mqa_q
