# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reshape (write-path) kernels for the sub-byte packed KV cache modes.

INT4 and INT2 use different quantization math so their reshape kernels
stay separate.  Both are invoked by a shared Python launcher
:func:`_run_reshape_kernel`.

The Lloyd-Max helpers also live here because ``_lloyd_max_quantize_4`` is
only used inside the INT2 reshape kernel, and ``_lloyd_max_dequant_4`` is
imported from here by the attention kernel (see
:mod:`._packed_attention`).
"""

from __future__ import annotations

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_quant_kv._pack_unpack import (
    pack_int2_quartet,
    pack_int4_nibbles,
)


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


@triton.jit
def _lloyd_max_quantize_4(z):
    """Quantize N(0,1) values to 4 Lloyd-Max centroids (INT2).

    Returns index in [0, 3].  Boundaries: [-0.9816, 0, 0.9816].
    """
    return tl.where(
        z < 0.0,
        tl.where(z < -0.9816, 0, 1).to(tl.uint8),
        tl.where(z < 0.9816, 2, 3).to(tl.uint8),
    )


@triton.jit
def _lloyd_max_dequant_4(idx):
    """Look up INT2 Lloyd-Max centroid for N(0,1).  idx in [0..3]."""
    return tl.where(
        idx < 2,
        tl.where(idx == 0, -1.5104, -0.4528),
        tl.where(idx == 2, 0.4528, 1.5104),
    )


@triton.jit
def _reshape_cache_int2_kernel(
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
    """INT2 Hadamard + Lloyd-Max 4-centroid quantization.

    Packs 4 × 2-bit indices per byte → head_size/4 bytes per head.
    """
    tok = tl.program_id(0)
    head = tl.program_id(1)

    slot = tl.load(slot_mapping_ptr + tok).to(tl.int64)
    if slot < 0:
        return

    blk = slot // block_size
    slot_in_blk = slot % block_size

    qtr_offs = tl.arange(0, PACKED_HEAD_PADDED)
    offs_0 = qtr_offs * 4
    offs_1 = qtr_offs * 4 + 1
    offs_2 = qtr_offs * 4 + 2
    offs_3 = qtr_offs * 4 + 3

    qtr_k = head_size // 4
    mask_0k = offs_0 < head_size
    mask_1k = offs_1 < head_size
    mask_2k = offs_2 < head_size
    mask_3k = offs_3 < head_size
    key_base = key_ptr + tok * stride_key_tok + head * stride_key_head

    k0 = tl.load(key_base + offs_0, mask=mask_0k, other=0.0).to(tl.float32)
    k1 = tl.load(key_base + offs_1, mask=mask_1k, other=0.0).to(tl.float32)
    k2 = tl.load(key_base + offs_2, mask=mask_2k, other=0.0).to(tl.float32)
    k3 = tl.load(key_base + offs_3, mask=mask_3k, other=0.0).to(tl.float32)

    k_sq = (
        tl.sum(tl.where(mask_0k, k0 * k0, 0.0))
        + tl.sum(tl.where(mask_1k, k1 * k1, 0.0))
        + tl.sum(tl.where(mask_2k, k2 * k2, 0.0))
        + tl.sum(tl.where(mask_3k, k3 * k3, 0.0))
    )
    k_norm = tl.sqrt(k_sq + 1e-12)

    k_inv_sigma = tl.sqrt(float(head_size)) / k_norm
    q0 = _lloyd_max_quantize_4(k0 * k_inv_sigma)
    q1 = _lloyd_max_quantize_4(k1 * k_inv_sigma)
    q2 = _lloyd_max_quantize_4(k2 * k_inv_sigma)
    q3 = _lloyd_max_quantize_4(k3 * k_inv_sigma)

    k_packed = pack_int2_quartet(q0, q1, q2, q3)
    tl.store(
        key_cache_ptr
        + blk * stride_kc_blk
        + slot_in_blk * stride_kc_slot
        + head * stride_kc_head
        + qtr_offs,
        k_packed,
        mask=qtr_offs < qtr_k,
    )

    # Store norm/d^1.5 as scale; see module docstring for the math.
    k_scale = k_norm / float(head_size**1.5)
    tl.store(
        k_scale_cache_ptr
        + blk * stride_ks_blk
        + slot_in_blk * stride_ks_slot
        + head * stride_ks_head,
        k_scale,
    )

    qtr_v = head_size_v // 4
    mask_0v = offs_0 < head_size_v
    mask_1v = offs_1 < head_size_v
    mask_2v = offs_2 < head_size_v
    mask_3v = offs_3 < head_size_v
    val_base = value_ptr + tok * stride_val_tok + head * stride_val_head

    v0 = tl.load(val_base + offs_0, mask=mask_0v, other=0.0).to(tl.float32)
    v1 = tl.load(val_base + offs_1, mask=mask_1v, other=0.0).to(tl.float32)
    v2 = tl.load(val_base + offs_2, mask=mask_2v, other=0.0).to(tl.float32)
    v3 = tl.load(val_base + offs_3, mask=mask_3v, other=0.0).to(tl.float32)

    v_sq = (
        tl.sum(tl.where(mask_0v, v0 * v0, 0.0))
        + tl.sum(tl.where(mask_1v, v1 * v1, 0.0))
        + tl.sum(tl.where(mask_2v, v2 * v2, 0.0))
        + tl.sum(tl.where(mask_3v, v3 * v3, 0.0))
    )
    v_norm = tl.sqrt(v_sq + 1e-12)
    v_inv_sigma = tl.sqrt(float(head_size_v)) / v_norm
    vq0 = _lloyd_max_quantize_4(v0 * v_inv_sigma)
    vq1 = _lloyd_max_quantize_4(v1 * v_inv_sigma)
    vq2 = _lloyd_max_quantize_4(v2 * v_inv_sigma)
    vq3 = _lloyd_max_quantize_4(v3 * v_inv_sigma)

    v_packed = pack_int2_quartet(vq0, vq1, vq2, vq3)
    tl.store(
        value_cache_ptr
        + blk * stride_vc_blk
        + slot_in_blk * stride_vc_slot
        + head * stride_vc_head
        + qtr_offs,
        v_packed,
        mask=qtr_offs < qtr_v,
    )

    v_scale = v_norm / float(head_size_v**1.5)
    tl.store(
        v_scale_cache_ptr
        + blk * stride_vs_blk
        + slot_in_blk * stride_vs_slot
        + head * stride_vs_head,
        v_scale,
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
    """Launch a packed reshape kernel (INT4 or INT2)."""
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
