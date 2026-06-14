# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Fused Triton kernel that combines RoPE (rotary positional embedding) and
KV cache reshape+store into a single kernel launch.

This eliminates the intermediate memory writes between the RoPE and
cache update steps, reducing kernel launch overhead and global memory
traffic.
"""

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import FP8_DTYPE
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import is_quantized_kv_cache

FP8_MAX = 448.0  # FP8 E4M3 max value


@triton.jit
def fused_rope_and_cache_kernel(
    # Query inputs
    query_ptr,  # [num_tokens, num_heads, head_size]
    # Key inputs
    key_ptr,  # [num_tokens, num_kv_heads, head_size]
    # Value inputs
    value_ptr,  # [num_tokens, num_kv_heads, head_size]
    # Position embedding
    positions_ptr,  # [num_tokens]
    cos_sin_cache_ptr,  # [max_position, 2 * head_size]
    # Cache outputs
    key_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache_ptr,  # [num_blocks, block_size, num_kv_heads, head_size]
    slot_mapping_ptr,  # [num_tokens]
    # Scaling factors for quantized cache
    k_scale_ptr,  # float32
    v_scale_ptr,  # float32
    # Strides for query (3D tensor)
    stride_q_tok: tl.int64,
    stride_q_head: tl.int64,
    # Strides for key (3D tensor)
    stride_k_tok: tl.int64,
    stride_k_head: tl.int64,
    # Strides for value (3D tensor)
    stride_v_tok: tl.int64,
    stride_v_head: tl.int64,
    # Strides for cos_sin_cache
    stride_cs_pos: tl.int64,
    stride_cs_dim: tl.int64,
    # Strides for key_cache (4D tensor)
    stride_kc_blk: tl.int64,
    stride_kc_slot: tl.int64,
    stride_kc_head: tl.int64,
    # Strides for value_cache (4D tensor)
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    # Strides for scale cache
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    # Configuration
    block_size: tl.int32,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    num_heads: tl.int32,
    num_kv_heads: tl.int32,
    IS_NEOX: tl.constexpr,
    FP8_KV_CACHE: tl.constexpr,
    PER_TOKEN_QUANT: tl.constexpr,
    # Autotune parameters
    TILE_SIZE: tl.constexpr,
):
    """Fused RoPE + KV cache kernel.

    Each program handles one (token, head) pair for key and value.
    For query, only RoPE is applied (query is not written to cache).
    """
    tok = tl.program_id(0)
    head = tl.program_id(1)

    # Load slot mapping
    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    block_idx = slot // block_size
    slot_in_blk = slot % block_size

    # Load position for this token
    pos = tl.load(positions_ptr + tok).to(tl.int64)

    half = head_size // 2
    half_offs = tl.arange(0, HEAD_SIZE_PADDED // 2)
    half_mask = half_offs < half

    # Load cos/sin for this position
    if IS_NEOX:
        # NeOX: cos_sin_cache[pos] has shape [2 * head_size]
        # First head_size elements are cos, second half are sin
        cos_offs = pos * stride_cs_pos + half_offs
        sin_offs = pos * stride_cs_pos + half_offs + half
    else:
        # GPT-J: cos_sin_cache[pos] has shape [head_size]
        # First half is cos, second half is sin (same offsets)
        cos_offs = pos * stride_cs_pos + half_offs
        sin_offs = pos * stride_cs_pos + half_offs

    cos = tl.load(
        cos_sin_cache_ptr + cos_offs * stride_cs_dim,
        mask=half_mask,
        other=1.0,
    )
    sin = tl.load(
        cos_sin_cache_ptr + sin_offs * stride_cs_dim,
        mask=half_mask,
        other=0.0,
    )

    # ---- Process Query: RoPE only (inplace) ----
    if head < num_heads:
        q_src = tok * stride_q_tok + head * stride_q_head

        if IS_NEOX:
            # NeOX: interleaved pairs (dim 0,1), (dim 2,3), ...
            real_offs = half_offs * 2
            imag_offs = half_offs * 2 + 1

            real = tl.load(
                query_ptr + q_src + real_offs, mask=half_mask, other=0.0
            ).to(tl.float32)
            imag = tl.load(
                query_ptr + q_src + imag_offs, mask=half_mask, other=0.0
            ).to(tl.float32)

            new_real = real * cos - imag * sin
            new_imag = real * sin + imag * cos

            tl.store(query_ptr + q_src + real_offs, new_real, mask=half_mask)
            tl.store(query_ptr + q_src + imag_offs, new_imag, mask=half_mask)
        else:
            # GPT-J: first half real, second half imag
            real = tl.load(
                query_ptr + q_src + half_offs, mask=half_mask, other=0.0
            ).to(tl.float32)
            imag = tl.load(
                query_ptr + q_src + half_offs + half, mask=half_mask, other=0.0
            ).to(tl.float32)

            new_real = real * cos - imag * sin
            new_imag = real * sin + imag * cos

            tl.store(query_ptr + q_src + half_offs, new_real, mask=half_mask)
            tl.store(query_ptr + q_src + half_offs + half, new_imag, mask=half_mask)

    # ---- Process Key: RoPE + cache write ----
    if head < num_kv_heads:
        k_src = tok * stride_k_tok + head * stride_k_head

        if IS_NEOX:
            # NeOX: interleaved pairs
            real_offs = half_offs * 2
            imag_offs = half_offs * 2 + 1

            real = tl.load(
                key_ptr + k_src + real_offs, mask=half_mask, other=0.0
            ).to(tl.float32)
            imag = tl.load(
                key_ptr + k_src + imag_offs, mask=half_mask, other=0.0
            ).to(tl.float32)

            new_real = real * cos - imag * sin
            new_imag = real * sin + imag * cos

            # Store rotated key back (inplace)
            tl.store(key_ptr + k_src + real_offs, new_real, mask=half_mask)
            tl.store(key_ptr + k_src + imag_offs, new_imag, mask=half_mask)

            # Prepare value for cache
            if FP8_KV_CACHE:
                if PER_TOKEN_QUANT:
                    # Compute per-token scale dynamically
                    k_combined = tl.cat(new_real, new_imag, can_reorder=False)
                    k_combined_mask = half_offs < head_size
                    k_flat = tl.where(k_combined_mask, k_combined[:head_size], 0.0)
                    k_scale_val = tl.maximum(tl.max(tl.abs(k_flat)) / 448.0, 1e-6)
                    k_cache_val = tl.clamp(k_flat / k_scale_val, -128.0, 127.0)

                    # Store scale to cache
                    tl.store(
                        k_scale_ptr
                        + block_idx * stride_ks_blk
                        + slot_in_blk * stride_ks_slot
                        + head * stride_ks_head,
                        k_scale_val,
                    )
                else:
                    k_combined = tl.cat(new_real, new_imag, can_reorder=False)
                    k_combined_mask = half_offs < head_size
                    scale = tl.load(k_scale_ptr)
                    k_cache_val = tl.clamp(
                        tl.where(k_combined_mask, k_combined[:head_size], 0.0) / scale,
                        -FP8_MAX,
                        FP8_MAX,
                    )
            else:
                k_combined = tl.cat(new_real, new_imag, can_reorder=False)
                k_combined_mask = half_offs < head_size
                k_cache_val = tl.where(k_combined_mask, k_combined[:head_size], 0.0)
        else:
            # GPT-J: first half real, second half imag
            real = tl.load(
                key_ptr + k_src + half_offs, mask=half_mask, other=0.0
            ).to(tl.float32)
            imag = tl.load(
                key_ptr + k_src + half_offs + half, mask=half_mask, other=0.0
            ).to(tl.float32)

            new_real = real * cos - imag * sin
            new_imag = real * sin + imag * cos

            # Store rotated key back (inplace)
            tl.store(key_ptr + k_src + half_offs, new_real, mask=half_mask)
            tl.store(key_ptr + k_src + half_offs + half, new_imag, mask=half_mask)

            # Prepare value for cache
            if FP8_KV_CACHE:
                if PER_TOKEN_QUANT:
                    k_combined = tl.cat(new_real, new_imag, can_reorder=False)
                    k_combined_mask = half_offs < head_size
                    k_flat = tl.where(k_combined_mask, k_combined[:head_size], 0.0)
                    k_scale_val = tl.maximum(tl.max(tl.abs(k_flat)) / 448.0, 1e-6)
                    k_cache_val = tl.clamp(k_flat / k_scale_val, -128.0, 127.0)

                    tl.store(
                        k_scale_ptr
                        + block_idx * stride_ks_blk
                        + slot_in_blk * stride_ks_slot
                        + head * stride_ks_head,
                        k_scale_val,
                    )
                else:
                    k_combined = tl.cat(new_real, new_imag, can_reorder=False)
                    k_combined_mask = half_offs < head_size
                    scale = tl.load(k_scale_ptr)
                    k_cache_val = tl.clamp(
                        tl.where(k_combined_mask, k_combined[:head_size], 0.0) / scale,
                        -FP8_MAX,
                        FP8_MAX,
                    )
            else:
                k_combined = tl.cat(new_real, new_imag, can_reorder=False)
                k_combined_mask = half_offs < head_size
                k_cache_val = tl.where(k_combined_mask, k_combined[:head_size], 0.0)

        # Store key to cache
        kc_offs = tl.arange(0, HEAD_SIZE_PADDED)
        kc_mask = kc_offs < head_size
        kc_idx = (
            block_idx * stride_kc_blk
            + slot_in_blk * stride_kc_slot
            + head * stride_kc_head
            + kc_offs
        )
        tl.store(key_cache_ptr + kc_idx, k_cache_val, mask=kc_mask)

    # ---- Process Value: cache write only ----
    if head < num_kv_heads:
        v_src = tok * stride_v_tok + head * stride_v_head
        v_offs = tl.arange(0, HEAD_SIZE_PADDED)
        v_mask = v_offs < head_size_v

        v = tl.load(
            value_ptr + v_src + v_offs,
            mask=v_mask,
            other=0.0,
        ).to(tl.float32)

        if FP8_KV_CACHE:
            if PER_TOKEN_QUANT:
                v_scale_val = tl.maximum(tl.max(tl.abs(v)) / 448.0, 1e-6)
                v_cache_val = tl.clamp(v / v_scale_val, -128.0, 127.0)

                tl.store(
                    v_scale_ptr
                    + block_idx * stride_vs_blk
                    + slot_in_blk * stride_vs_slot
                    + head * stride_vs_head,
                    v_scale_val,
                )
            else:
                scale = tl.load(v_scale_ptr)
                v_cache_val = tl.clamp(v / scale, -FP8_MAX, FP8_MAX)
        else:
            v_cache_val = v

        # Store value to cache
        vc_idx = (
            block_idx * stride_vc_blk
            + slot_in_blk * stride_vc_slot
            + head * stride_vc_head
            + v_offs
        )
        tl.store(value_cache_ptr + vc_idx, v_cache_val, mask=v_mask)


def triton_fused_rope_and_cache(
    query: torch.Tensor,  # [num_tokens, num_heads, head_size]
    key: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    value: torch.Tensor,  # [num_tokens, num_kv_heads, head_size_v]
    positions: torch.Tensor,  # [num_tokens]
    cos_sin_cache: torch.Tensor,  # [max_position, head_size] or [max_position, 2 * head_size]
    is_neox: bool,
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size]
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size_v]
    slot_mapping: torch.Tensor,  # [num_tokens]
    k_scale: torch.Tensor,  # float32
    v_scale: torch.Tensor,  # float32
    fp8_kv_cache: bool,
    k_scale_cache: torch.Tensor | None = None,  # for per-token quant
    v_scale_cache: torch.Tensor | None = None,
):
    """Fused RoPE + KV cache update.

    Applies RoPE to query and key inplace, then writes key and value
    to the paged KV cache in a single kernel launch.
    """
    num_tokens = query.shape[0]
    num_heads = query.shape[1]
    num_kv_heads = key.shape[1]
    head_size = key.shape[2]
    head_size_v = value.shape[2]
    head_size_padded = triton.next_power_of_2(max(head_size, head_size_v))

    block_size = key_cache.shape[1]
    FP8_KV_CACHE = fp8_kv_cache
    PER_TOKEN_QUANT = k_scale_cache is not None

    # Set kernel tuning parameters
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_warps = 4
        num_stages = 2
    else:
        num_warps = 8
        num_stages = 4

    TILE_SIZE = min(256, head_size_padded)

    # Grid: (num_tokens, max(num_heads, num_kv_heads))
    grid = (num_tokens, max(num_heads, num_kv_heads))

    fused_rope_and_cache_kernel[grid](
        # Pointers
        query_ptr=query,
        key_ptr=key,
        value_ptr=value,
        positions_ptr=positions,
        cos_sin_cache_ptr=cos_sin_cache,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        slot_mapping_ptr=slot_mapping,
        k_scale_ptr=k_scale if not PER_TOKEN_QUANT else k_scale_cache,
        v_scale_ptr=v_scale if not PER_TOKEN_QUANT else v_scale_cache,
        # Strides
        stride_q_tok=query.stride(0),
        stride_q_head=query.stride(1),
        stride_k_tok=key.stride(0),
        stride_k_head=key.stride(1),
        stride_v_tok=value.stride(0),
        stride_v_head=value.stride(1),
        stride_cs_pos=cos_sin_cache.stride(0),
        stride_cs_dim=cos_sin_cache.stride(1),
        stride_kc_blk=key_cache.stride(0),
        stride_kc_slot=key_cache.stride(1),
        stride_kc_head=key_cache.stride(2),
        stride_vc_blk=value_cache.stride(0),
        stride_vc_slot=value_cache.stride(1),
        stride_vc_head=value_cache.stride(2),
        stride_ks_blk=k_scale_cache.stride(0) if PER_TOKEN_QUANT else 0,
        stride_ks_slot=k_scale_cache.stride(1) if PER_TOKEN_QUANT else 0,
        stride_ks_head=k_scale_cache.stride(2) if PER_TOKEN_QUANT else 0,
        stride_vs_blk=v_scale_cache.stride(0) if PER_TOKEN_QUANT else 0,
        stride_vs_slot=v_scale_cache.stride(1) if PER_TOKEN_QUANT else 0,
        stride_vs_head=v_scale_cache.stride(2) if PER_TOKEN_QUANT else 0,
        # Config
        block_size=block_size,
        head_size=head_size,
        head_size_v=head_size_v,
        HEAD_SIZE_PADDED=head_size_padded,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        IS_NEOX=is_neox,
        FP8_KV_CACHE=FP8_KV_CACHE,
        PER_TOKEN_QUANT=PER_TOKEN_QUANT,
        # Autotune
        TILE_SIZE=TILE_SIZE,
        num_warps=num_warps,
        num_stages=num_stages,
    )