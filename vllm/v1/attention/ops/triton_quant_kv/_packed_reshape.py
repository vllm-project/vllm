# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reshape (write-path) kernel for the sub-byte packed INT4 KV cache mode.

The INT4 reshape kernel is invoked by the shared Python launcher
:func:`_run_reshape_kernel`.
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_quant_kv._pack_unpack import pack_int4_nibbles


@triton.jit
def _reshape_cache_int4_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    value_cache_ptr,
    k_scale_cache_ptr,
    v_scale_cache_ptr,
    slot_mapping_ptr,
    stride_key_tok: tl.int64,
    stride_key_head: tl.int64,
    stride_val_tok: tl.int64,
    stride_val_head: tl.int64,
    stride_kc_blk: tl.int64,
    stride_kc_slot: tl.int64,
    stride_kc_head: tl.int64,
    stride_vc_blk: tl.int64,
    stride_vc_slot: tl.int64,
    stride_vc_head: tl.int64,
    stride_ks_blk: tl.int64,
    stride_ks_slot: tl.int64,
    stride_ks_head: tl.int64,
    stride_vs_blk: tl.int64,
    stride_vs_slot: tl.int64,
    stride_vs_head: tl.int64,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_v: tl.constexpr,
    PACKED_HEAD_PADDED: tl.constexpr,
):
    """INT4 asymmetric quantization with zero-point steganography."""
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    half_offs = tl.arange(0, PACKED_HEAD_PADDED)
    even_offs = half_offs * 2
    odd_offs = half_offs * 2 + 1

    half_k = head_size // 2
    even_k_mask = even_offs < head_size
    odd_k_mask = odd_offs < head_size
    key_base = key_ptr + tok * stride_key_tok + head * stride_key_head

    k_even = tl.load(key_base + even_offs, mask=even_k_mask, other=0.0).to(tl.float32)
    k_odd = tl.load(key_base + odd_offs, mask=odd_k_mask, other=0.0).to(tl.float32)

    k_min = tl.minimum(
        tl.min(tl.where(even_k_mask, k_even, float("inf"))),
        tl.min(tl.where(odd_k_mask, k_odd, float("inf"))),
    )
    k_max = tl.maximum(
        tl.max(tl.where(even_k_mask, k_even, float("-inf"))),
        tl.max(tl.where(odd_k_mask, k_odd, float("-inf"))),
    )
    k_scale = tl.maximum((k_max - k_min) / 15.0, 1e-6)
    k_zp_f = tl.clamp(
        tl.where(
            -k_min / k_scale >= 0,
            (-k_min / k_scale + 0.5).to(tl.int32),
            (-k_min / k_scale - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    inv_k = 1.0 / k_scale
    k_even_s = k_even * inv_k + k_zp_f
    k_odd_s = k_odd * inv_k + k_zp_f
    k_even_q = tl.clamp(
        tl.where(
            k_even_s >= 0,
            (k_even_s + 0.5).to(tl.int32),
            (k_even_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )
    k_odd_q = tl.clamp(
        tl.where(
            k_odd_s >= 0,
            (k_odd_s + 0.5).to(tl.int32),
            (k_odd_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    k_zp_int = k_zp_f.to(tl.int32)
    k_scale_bits = k_scale.to(tl.int32, bitcast=True)
    k_scale_packed = ((k_scale_bits & -16) | (k_zp_int & 0xF)).to(
        tl.float32, bitcast=True
    )

    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale_packed,
    )

    k_packed = pack_int4_nibbles(k_even_q.to(tl.uint8), k_odd_q.to(tl.uint8))
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + half_offs,
        k_packed,
        mask=half_offs < half_k,
    )

    half_v = head_size_v // 2
    even_v_mask = even_offs < head_size_v
    odd_v_mask = odd_offs < head_size_v
    val_base = value_ptr + tok * stride_val_tok + head * stride_val_head

    v_even = tl.load(val_base + even_offs, mask=even_v_mask, other=0.0).to(tl.float32)
    v_odd = tl.load(val_base + odd_offs, mask=odd_v_mask, other=0.0).to(tl.float32)

    v_min = tl.minimum(
        tl.min(tl.where(even_v_mask, v_even, float("inf"))),
        tl.min(tl.where(odd_v_mask, v_odd, float("inf"))),
    )
    v_max = tl.maximum(
        tl.max(tl.where(even_v_mask, v_even, float("-inf"))),
        tl.max(tl.where(odd_v_mask, v_odd, float("-inf"))),
    )
    v_scale = tl.maximum((v_max - v_min) / 15.0, 1e-6)
    v_zp_f = tl.clamp(
        tl.where(
            -v_min / v_scale >= 0,
            (-v_min / v_scale + 0.5).to(tl.int32),
            (-v_min / v_scale - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    inv_v = 1.0 / v_scale
    v_even_s = v_even * inv_v + v_zp_f
    v_odd_s = v_odd * inv_v + v_zp_f
    v_even_q = tl.clamp(
        tl.where(
            v_even_s >= 0,
            (v_even_s + 0.5).to(tl.int32),
            (v_even_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )
    v_odd_q = tl.clamp(
        tl.where(
            v_odd_s >= 0,
            (v_odd_s + 0.5).to(tl.int32),
            (v_odd_s - 0.5).to(tl.int32),
        ).to(tl.float32),
        0.0,
        15.0,
    )

    v_zp_int = v_zp_f.to(tl.int32)
    v_scale_bits = v_scale.to(tl.int32, bitcast=True)
    v_scale_packed = ((v_scale_bits & -16) | (v_zp_int & 0xF)).to(
        tl.float32, bitcast=True
    )

    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale_packed,
    )

    v_packed = pack_int4_nibbles(v_even_q.to(tl.uint8), v_odd_q.to(tl.uint8))
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + half_offs,
        v_packed,
        mask=half_offs < half_v,
    )


def _run_reshape_kernel(
    kernel,
    *,
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    packing_factor: int,
) -> None:
    """Launch the packed INT4 reshape kernel."""
    num_tokens, num_kv_heads, head_size = key.shape
    head_size_v = value.shape[2]
    assert head_size % packing_factor == 0 and head_size_v % packing_factor == 0
    packed_padded = triton.next_power_of_2(
        max(head_size, head_size_v) // packing_factor
    )
    if current_platform.is_rocm() or current_platform.is_xpu():
        num_warps = 4
    else:
        num_warps = min(16, max(1, packed_padded // 32))

    kernel[(num_tokens, num_kv_heads)](
        key_ptr=key,
        value_ptr=value,
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        k_scale_cache_ptr=k_scale_cache,
        v_scale_cache_ptr=v_scale_cache,
        slot_mapping_ptr=slot_mapping,
        stride_key_tok=key.stride(0),
        stride_key_head=key.stride(1),
        stride_val_tok=value.stride(0),
        stride_val_head=value.stride(1),
        stride_kc_blk=key_cache.stride(0),
        stride_kc_slot=key_cache.stride(1),
        stride_kc_head=key_cache.stride(2),
        stride_vc_blk=value_cache.stride(0),
        stride_vc_slot=value_cache.stride(1),
        stride_vc_head=value_cache.stride(2),
        stride_ks_blk=k_scale_cache.stride(0),
        stride_ks_slot=k_scale_cache.stride(1),
        stride_ks_head=k_scale_cache.stride(2),
        stride_vs_blk=v_scale_cache.stride(0),
        stride_vs_slot=v_scale_cache.stride(1),
        stride_vs_head=v_scale_cache.stride(2),
        block_size=key_cache.shape[1],
        head_size=head_size,
        head_size_v=head_size_v,
        PACKED_HEAD_PADDED=packed_padded,
        num_warps=num_warps,
    )
