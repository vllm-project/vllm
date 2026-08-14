# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""gfx950 Gluon partial kernel for DeepSeek-V4 sparse-MLA decode.

Adapted from AITER's MIT-licensed ``pa_decode_sparse`` gfx950 kernel. The vLLM
variant is intentionally limited to the packed two-segment DSv4 cache contract
and writes the existing Triton reducer's partial-buffer format.
"""

import torch

from vllm.triton_utils import gl, gluon

_GFX950_BLOCK_H = 16
_GFX950_BLOCK_K = 64
_GFX950_HEAD_SIZE = 512
_MAX_BUFFER_OFFSET = 2**31 - 1


def _max_addressable_bytes(tensor: torch.Tensor) -> int:
    """Return the byte span reachable from a tensor's data pointer."""
    span = 1
    for size, stride in zip(tensor.shape, tensor.stride()):
        if size > 1:
            span += (size - 1) * abs(stride)
    return span * tensor.element_size()


@gluon.jit
def _cache_load(ptr, offsets, USE_BUFFER_LOAD: gl.constexpr, mask=None, other=None):
    if USE_BUFFER_LOAD:
        return gl.amd.cdna4.buffer_load(
            ptr=ptr,
            offsets=offsets.to(gl.int32),
            mask=mask,
            other=other,
            cache=".cg",
        )
    return gl.load(
        ptr + offsets.to(gl.int64),
        mask=mask,
        other=other,
        cache_modifier=".cg",
    )


@gluon.jit
def _decode_e8m0_scales(encoded_scales):
    scale_bits = encoded_scales.to(gl.int32) << 23
    scale_bits = gl.where(encoded_scales == 0, 1 << 22, scale_bits)
    return scale_bits.to(gl.float32, bitcast=True)


@gluon.jit
def _slots(
    indices_ptr,
    segment_start,
    k_pos,
    segment_hi,
    num_rows,
    BLOCK_SIZE: gl.constexpr,
    MASKED: gl.constexpr,
):
    if MASKED:
        in_range = k_pos < segment_hi
        slot = gl.load(indices_ptr + segment_start + k_pos, mask=in_range, other=-1)
        valid = in_range & (slot >= 0) & (slot < num_rows)
    else:
        slot = gl.load(indices_ptr + segment_start + k_pos)
        valid = slot >= 0
    safe_slot = gl.where(valid, slot, 0)
    return (
        (safe_slot // BLOCK_SIZE).to(gl.int32),
        (safe_slot % BLOCK_SIZE).to(gl.int32),
        valid,
    )


@gluon.jit
def _gather_fp8_tile(
    cache_ptr,
    cache_bf16_ptr,
    indices_ptr,
    segment_start,
    k_start,
    segment_hi,
    cache_stride0,
    num_rows,
    full_offsets,
    rope_offsets,
    slot_offsets,
    rope_slot_offsets,
    gather_layout: gl.constexpr,
    rope_gather_layout: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    BLOCK_K: gl.constexpr,
    MASKED: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
):
    if not USE_BUFFER_LOAD:
        cache_stride0 = cache_stride0.to(gl.int64)

    block, pos, valid = _slots(
        indices_ptr,
        segment_start,
        k_start + slot_offsets,
        segment_hi,
        num_rows,
        BLOCK_SIZE,
        MASKED,
    )
    block_g = gl.convert_layout(block, gl.SliceLayout(1, gather_layout))
    pos_g = gl.convert_layout(pos, gl.SliceLayout(1, gather_layout))
    valid_g = gl.convert_layout(valid, gl.SliceLayout(1, gather_layout))

    data_offsets = (block_g * cache_stride0 + pos_g * 576)[:, None] + full_offsets[
        None, :
    ]
    scale_offsets = (block_g * cache_stride0 + BLOCK_SIZE * 576 + pos_g * 8)[
        :, None
    ] + (full_offsets[None, :] // 64)
    if MASKED:
        data = _cache_load(
            cache_ptr,
            data_offsets,
            USE_BUFFER_LOAD,
            mask=valid_g[:, None],
            other=0,
        )
        encoded_scales = _cache_load(
            cache_ptr,
            scale_offsets,
            USE_BUFFER_LOAD,
            mask=valid_g[:, None],
            other=127,
        )
    else:
        data = _cache_load(cache_ptr, data_offsets, USE_BUFFER_LOAD)
        encoded_scales = _cache_load(cache_ptr, scale_offsets, USE_BUFFER_LOAD)
    nope = (
        data.to(gl.float8e4nv, bitcast=True).to(gl.float32)
        * _decode_e8m0_scales(encoded_scales)
    ).to(gl.bfloat16)

    rope_block, rope_pos, rope_valid = _slots(
        indices_ptr,
        segment_start,
        k_start + rope_slot_offsets,
        segment_hi,
        num_rows,
        BLOCK_SIZE,
        MASKED,
    )
    rope_offsets_global = (rope_block * (cache_stride0 // 2) + rope_pos * 288 + 224)[
        :, None
    ] + rope_offsets[None, :]
    if MASKED:
        rope = _cache_load(
            cache_bf16_ptr,
            rope_offsets_global,
            USE_BUFFER_LOAD,
            mask=rope_valid[:, None],
            other=0.0,
        )
    else:
        rope = _cache_load(
            cache_bf16_ptr,
            rope_offsets_global,
            USE_BUFFER_LOAD,
        )
    return nope, rope, valid


@gluon.jit
def _consume_fp8_tile(
    nope,
    rope,
    valid,
    q_dot,
    m_i,
    l_i,
    acc,
    head_mask,
    qk_scale,
    kv_smem,
    qk_layout: gl.constexpr,
    pv_layout: gl.constexpr,
    k_layout: gl.constexpr,
    v_layout: gl.constexpr,
    p_layout: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
):
    kv_smem.store(nope)
    kv_smem.slice(448, 64, dim=1).store(rope)

    k = kv_smem.permute([1, 0]).load(k_layout)
    scores = gl.amd.cdna4.mfma(
        q_dot,
        k,
        gl.zeros([BLOCK_H, BLOCK_K], gl.float32, layout=qk_layout),
    )
    column_mask = gl.convert_layout(valid, gl.SliceLayout(0, qk_layout))[None, :]
    if not HEAD_ALIGNED:
        column_mask = (
            gl.convert_layout(head_mask, gl.SliceLayout(1, qk_layout))[:, None]
            & column_mask
        )
    scores = gl.where(column_mask, scores, float("-inf"))

    block_max = gl.max(scores, axis=1)
    m_new = gl.maximum(m_i, block_max)
    m_new = gl.where(m_new > float("-inf"), m_new, 0.0)
    m_new_scaled = m_new * qk_scale
    p = gl.exp2(scores * qk_scale - m_new_scaled[:, None])
    alpha = gl.exp2(m_i * qk_scale - m_new_scaled)
    l_new = l_i * alpha + gl.sum(p, axis=1)

    v = kv_smem.load(v_layout)
    p_dot = gl.convert_layout(p.to(gl.bfloat16), p_layout)
    alpha_pv = gl.convert_layout(alpha, gl.SliceLayout(1, pv_layout))
    acc = acc * alpha_pv[:, None]
    acc = gl.amd.cdna4.mfma(p_dot, v, acc)
    return m_new, l_new, acc


@gluon.jit
def _process_segment(
    q_dot,
    cache_ptr,
    cache_bf16_ptr,
    indices_ptr,
    segment_start,
    lo,
    hi,
    cache_stride0,
    num_rows,
    m_i,
    l_i,
    acc,
    head_mask,
    qk_scale,
    kv_smem,
    qk_layout: gl.constexpr,
    pv_layout: gl.constexpr,
    k_layout: gl.constexpr,
    v_layout: gl.constexpr,
    p_layout: gl.constexpr,
    gather_layout: gl.constexpr,
    rope_gather_layout: gl.constexpr,
    slot_layout: gl.constexpr,
    BLOCK_SIZE: gl.constexpr,
    BLOCK_H: gl.constexpr,
    BLOCK_K: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
):
    full_offsets = gl.arange(
        0,
        512,
        layout=gl.SliceLayout(0, gather_layout),
    )
    rope_offsets = gl.arange(
        0,
        64,
        layout=gl.SliceLayout(0, rope_gather_layout),
    )
    slot_offsets = gl.arange(0, BLOCK_K, layout=slot_layout)
    rope_slot_offsets = gl.arange(
        0,
        BLOCK_K,
        layout=gl.SliceLayout(1, rope_gather_layout),
    )

    full_hi = lo + ((hi - lo) // BLOCK_K) * BLOCK_K
    num_full = (full_hi - lo) // BLOCK_K
    if num_full > 0:
        nope, rope, valid = _gather_fp8_tile(
            cache_ptr,
            cache_bf16_ptr,
            indices_ptr,
            segment_start,
            lo,
            hi,
            cache_stride0,
            num_rows,
            full_offsets,
            rope_offsets,
            slot_offsets,
            rope_slot_offsets,
            gather_layout,
            rope_gather_layout,
            BLOCK_SIZE,
            BLOCK_K,
            False,
            USE_BUFFER_LOAD,
        )
        for tile_idx in range(1, num_full):
            next_nope, next_rope, next_valid = _gather_fp8_tile(
                cache_ptr,
                cache_bf16_ptr,
                indices_ptr,
                segment_start,
                lo + tile_idx * BLOCK_K,
                hi,
                cache_stride0,
                num_rows,
                full_offsets,
                rope_offsets,
                slot_offsets,
                rope_slot_offsets,
                gather_layout,
                rope_gather_layout,
                BLOCK_SIZE,
                BLOCK_K,
                False,
                USE_BUFFER_LOAD,
            )
            m_i, l_i, acc = _consume_fp8_tile(
                nope,
                rope,
                valid,
                q_dot,
                m_i,
                l_i,
                acc,
                head_mask,
                qk_scale,
                kv_smem,
                qk_layout,
                pv_layout,
                k_layout,
                v_layout,
                p_layout,
                BLOCK_H,
                BLOCK_K,
                HEAD_ALIGNED,
            )
            nope, rope, valid = next_nope, next_rope, next_valid
        m_i, l_i, acc = _consume_fp8_tile(
            nope,
            rope,
            valid,
            q_dot,
            m_i,
            l_i,
            acc,
            head_mask,
            qk_scale,
            kv_smem,
            qk_layout,
            pv_layout,
            k_layout,
            v_layout,
            p_layout,
            BLOCK_H,
            BLOCK_K,
            HEAD_ALIGNED,
        )

    if full_hi < hi:
        nope, rope, valid = _gather_fp8_tile(
            cache_ptr,
            cache_bf16_ptr,
            indices_ptr,
            segment_start,
            full_hi,
            hi,
            cache_stride0,
            num_rows,
            full_offsets,
            rope_offsets,
            slot_offsets,
            rope_slot_offsets,
            gather_layout,
            rope_gather_layout,
            BLOCK_SIZE,
            BLOCK_K,
            True,
            USE_BUFFER_LOAD,
        )
        m_i, l_i, acc = _consume_fp8_tile(
            nope,
            rope,
            valid,
            q_dot,
            m_i,
            l_i,
            acc,
            head_mask,
            qk_scale,
            kv_smem,
            qk_layout,
            pv_layout,
            k_layout,
            v_layout,
            p_layout,
            BLOCK_H,
            BLOCK_K,
            HEAD_ALIGNED,
        )
    return m_i, l_i, acc


@gluon.jit
def _sparse_attn_decode_partial_gfx950_kernel(
    q_ptr,
    main_cache_ptr,
    main_cache_bf16_ptr,
    main_indices_ptr,
    main_indptr_ptr,
    extra_cache_ptr,
    extra_cache_bf16_ptr,
    extra_indices_ptr,
    extra_indptr_ptr,
    part_m_ptr,
    part_l_ptr,
    part_acc_ptr,
    scale: gl.constexpr,
    q_stride0: gl.constexpr,
    q_stride1: gl.constexpr,
    main_cache_stride0,
    extra_cache_stride0,
    pm_stride0: gl.constexpr,
    pm_stride_s: gl.constexpr,
    pa_stride0: gl.constexpr,
    pa_stride_s: gl.constexpr,
    pa_stride_h: gl.constexpr,
    main_num_rows,
    extra_num_rows,
    num_heads: gl.constexpr,
    HAS_EXTRA: gl.constexpr,
    HAS_ATTN_SINK: gl.constexpr,
    MAIN_BLOCK_SIZE: gl.constexpr,
    EXTRA_BLOCK_SIZE: gl.constexpr,
    NUM_SPLITS: gl.constexpr,
    HEAD_ALIGNED: gl.constexpr,
    USE_BUFFER_LOAD: gl.constexpr,
):
    num_warps: gl.constexpr = gl.num_warps()
    query_idx = gl.program_id(0)
    split_id = gl.program_id(1)
    head_block = gl.program_id(2)

    qk_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )
    pv_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 16],
        transposed=True,
        warps_per_cta=[1, num_warps],
    )
    q_layout: gl.constexpr = gl.DotOperandLayout(0, qk_layout, 8)
    k_layout: gl.constexpr = gl.DotOperandLayout(1, qk_layout, 8)
    p_layout: gl.constexpr = gl.DotOperandLayout(0, pv_layout, 8)
    v_layout: gl.constexpr = gl.DotOperandLayout(1, pv_layout, 8)

    gather_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    rope_gather_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    slot_layout: gl.constexpr = gl.SliceLayout(1, gather_layout)
    q_blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )
    kv_shared_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[512, 8]],
        [64, 512],
        [1, 0],
    )

    head_base = head_block * 16
    head_offsets_q = gl.arange(
        0,
        16,
        layout=gl.SliceLayout(1, q_blocked_layout),
    )
    dim_offsets_q = gl.arange(
        0,
        512,
        layout=gl.SliceLayout(0, q_blocked_layout),
    )
    heads_q = head_base + head_offsets_q
    head_mask_q = heads_q < num_heads
    q_offsets = (
        query_idx * q_stride0 + heads_q[:, None] * q_stride1 + dim_offsets_q[None, :]
    ).to(gl.int32)
    q = gl.amd.cdna4.buffer_load(
        ptr=q_ptr,
        offsets=q_offsets,
        mask=head_mask_q[:, None],
        other=0.0,
    )
    q_dot = gl.convert_layout(q, q_layout)

    head_offsets_pv = gl.arange(
        0,
        16,
        layout=gl.SliceLayout(1, pv_layout),
    )
    heads_pv = head_base + head_offsets_pv
    head_mask_pv = heads_pv < num_heads

    m_i = gl.full(
        [16],
        float("-inf"),
        gl.float32,
        layout=gl.SliceLayout(1, qk_layout),
    )
    l_i = gl.zeros(
        [16],
        gl.float32,
        layout=gl.SliceLayout(1, qk_layout),
    )
    acc = gl.zeros(
        [16, 512],
        gl.float32,
        layout=pv_layout,
    )
    kv_smem = gl.allocate_shared_memory(
        gl.bfloat16,
        [64, 512],
        kv_shared_layout,
    )

    main_start = gl.load(main_indptr_ptr + query_idx)
    main_end = gl.load(main_indptr_ptr + query_idx + 1)
    main_len = main_end - main_start
    main_chunk = (main_len + NUM_SPLITS - 1) // NUM_SPLITS
    main_lo = split_id * main_chunk
    main_hi = gl.minimum(main_lo + main_chunk, main_len)

    rcp_ln2: gl.constexpr = 1.4426950408889634
    qk_scale: gl.constexpr = scale * rcp_ln2
    m_i, l_i, acc = _process_segment(
        q_dot,
        main_cache_ptr,
        main_cache_bf16_ptr,
        main_indices_ptr,
        main_start,
        main_lo,
        main_hi,
        main_cache_stride0,
        main_num_rows,
        m_i,
        l_i,
        acc,
        head_mask_pv,
        qk_scale,
        kv_smem,
        qk_layout,
        pv_layout,
        k_layout,
        v_layout,
        p_layout,
        gather_layout,
        rope_gather_layout,
        slot_layout,
        MAIN_BLOCK_SIZE,
        16,
        64,
        HEAD_ALIGNED,
        USE_BUFFER_LOAD,
    )
    if HAS_EXTRA:
        extra_start = gl.load(extra_indptr_ptr + query_idx)
        extra_end = gl.load(extra_indptr_ptr + query_idx + 1)
        extra_len = extra_end - extra_start
        extra_chunk = (extra_len + NUM_SPLITS - 1) // NUM_SPLITS
        extra_lo = split_id * extra_chunk
        extra_hi = gl.minimum(extra_lo + extra_chunk, extra_len)
        m_i, l_i, acc = _process_segment(
            q_dot,
            extra_cache_ptr,
            extra_cache_bf16_ptr,
            extra_indices_ptr,
            extra_start,
            extra_lo,
            extra_hi,
            extra_cache_stride0,
            extra_num_rows,
            m_i,
            l_i,
            acc,
            head_mask_pv,
            qk_scale,
            kv_smem,
            qk_layout,
            pv_layout,
            k_layout,
            v_layout,
            p_layout,
            gather_layout,
            rope_gather_layout,
            slot_layout,
            EXTRA_BLOCK_SIZE,
            16,
            64,
            HEAD_ALIGNED,
            USE_BUFFER_LOAD,
        )

    m_pv = gl.convert_layout(m_i, gl.SliceLayout(1, pv_layout))
    l_pv = gl.convert_layout(l_i, gl.SliceLayout(1, pv_layout))
    if NUM_SPLITS == 1:
        if HAS_ATTN_SINK:
            m_scaled = m_pv * scale
            sink = gl.amd.cdna4.buffer_load(
                ptr=part_m_ptr,
                offsets=heads_pv.to(gl.int32),
                mask=head_mask_pv,
                other=float("-inf"),
            ).to(gl.float32)
            m_final = gl.maximum(m_scaled, sink)
            alpha = gl.exp2((m_scaled - m_final) * rcp_ln2)
            l_final = l_pv * alpha + gl.exp2((sink - m_final) * rcp_ln2)
            acc = acc * alpha[:, None]
        else:
            l_final = l_pv
        denom = gl.maximum(l_final, 1.0e-30)
        out = gl.where(l_final[:, None] > 0.0, acc / denom[:, None], 0.0)
        dim_offsets_out = gl.arange(
            0,
            512,
            layout=gl.SliceLayout(0, pv_layout),
        )
        out_offsets = (
            query_idx * pa_stride0
            + heads_pv[:, None] * pa_stride_h
            + dim_offsets_out[None, :]
        ).to(gl.int32)
        gl.amd.cdna4.buffer_store(
            out.to(part_acc_ptr.dtype.element_ty),
            ptr=part_acc_ptr,
            offsets=out_offsets,
            mask=head_mask_pv[:, None],
        )
    else:
        m_natural = gl.where(
            m_pv > float("-inf"),
            m_pv * scale,
            -3.4028234663852886e38,
        )
        partial_row_base = query_idx * pm_stride0 + split_id * pm_stride_s
        gl.amd.cdna4.buffer_store(
            m_natural,
            ptr=part_m_ptr + partial_row_base,
            offsets=heads_pv.to(gl.int32),
            mask=head_mask_pv,
        )
        gl.amd.cdna4.buffer_store(
            l_pv,
            ptr=part_l_ptr + partial_row_base,
            offsets=heads_pv.to(gl.int32),
            mask=head_mask_pv,
        )
        dim_offsets_acc = gl.arange(
            0,
            512,
            layout=gl.SliceLayout(0, pv_layout),
        )
        partial_acc_base = query_idx * pa_stride0 + split_id * pa_stride_s
        partial_acc_offsets = (
            partial_acc_base
            + heads_pv[:, None] * pa_stride_h
            + dim_offsets_acc[None, :]
        ).to(gl.int32)
        gl.amd.cdna4.buffer_store(
            acc,
            ptr=part_acc_ptr,
            offsets=partial_acc_offsets,
            mask=head_mask_pv[:, None],
        )


def launch_sparse_attn_decode_partial_gfx950(
    q: torch.Tensor,
    main_cache: torch.Tensor,
    main_indices: torch.Tensor,
    main_indptr: torch.Tensor,
    extra_cache: torch.Tensor,
    extra_indices: torch.Tensor,
    extra_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    part_m: torch.Tensor,
    part_l: torch.Tensor,
    part_acc: torch.Tensor,
    scale: float,
    num_heads: int,
    has_extra: bool,
    has_attn_sink: bool,
    num_splits: int,
) -> None:
    """Launch the fixed gfx950 packed-cache partial kernel."""
    assert q.shape[-1] == _GFX950_HEAD_SIZE
    assert main_cache.dtype == torch.uint8
    assert extra_cache.dtype == torch.uint8

    use_buffer_load = (
        max(
            _max_addressable_bytes(main_cache),
            _max_addressable_bytes(extra_cache),
        )
        < _MAX_BUFFER_OFFSET
    )
    heads_blocks = (num_heads + _GFX950_BLOCK_H - 1) // _GFX950_BLOCK_H
    grid = (q.shape[0], num_splits, heads_blocks)
    if num_splits == 1:
        part_m = attn_sink
        part_l = out
        part_acc = out
        pm_stride0 = pm_stride_s = 0
        pa_stride0 = out.stride(0)
        pa_stride_s = 0
        pa_stride_h = out.stride(1)
    else:
        pm_stride_s = part_m.stride(0)
        pm_stride0 = num_splits * pm_stride_s
        pa_stride_s = part_acc.stride(0)
        pa_stride0 = num_splits * pa_stride_s
        pa_stride_h = part_acc.stride(1)
    _sparse_attn_decode_partial_gfx950_kernel[grid](
        q,
        main_cache,
        main_cache.view(torch.bfloat16),
        main_indices,
        main_indptr,
        extra_cache,
        extra_cache.view(torch.bfloat16),
        extra_indices,
        extra_indptr,
        part_m,
        part_l,
        part_acc,
        scale,
        q.stride(0),
        q.stride(1),
        main_cache.stride(0),
        extra_cache.stride(0),
        pm_stride0,
        pm_stride_s,
        pa_stride0,
        pa_stride_s,
        pa_stride_h,
        main_cache.shape[0] * main_cache.shape[1],
        extra_cache.shape[0] * extra_cache.shape[1],
        num_heads,
        HAS_EXTRA=has_extra,
        HAS_ATTN_SINK=has_attn_sink,
        MAIN_BLOCK_SIZE=main_cache.shape[1],
        EXTRA_BLOCK_SIZE=extra_cache.shape[1],
        NUM_SPLITS=num_splits,
        HEAD_ALIGNED=num_heads % _GFX950_BLOCK_H == 0,
        USE_BUFFER_LOAD=use_buffer_load,
        num_warps=4,
        waves_per_eu=0,
    )
