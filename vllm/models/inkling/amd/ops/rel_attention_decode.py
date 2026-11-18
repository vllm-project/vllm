# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 LightSeek Foundation
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
"""Split-KV decode for Inkling relative attention on ROCm.

Adapted from LightSeek TokenSpeed's portable Triton relative-MHA decode.
"""

from __future__ import annotations

import os

import torch

from vllm.triton_utils import tl, triton

_MIN_BLOCK_KV = tl.constexpr(32)


def decode_split_count(max_kv_len: int, window_left: int) -> int:
    """Return the number of parallel KV partitions for decode."""
    if window_left >= 0:
        effective_len = max(1, min(max_kv_len, window_left + 1))
        return min(4, max(1, triton.cdiv(effective_len, 128)))
    return min(32, max(1, triton.cdiv(max(1, max_kv_len), 2048)))


def use_split_kv_decode(
    *,
    max_query_len: int,
    max_kv_len: int,
    page_size: int,
    window_left: int,
) -> bool:
    """Select split-KV only where it outperforms the single-pass kernel."""
    if os.getenv("INKLING_SPLIT_KV", "1") != "1":
        return False
    if max_query_len != 1:
        return False
    if page_size >= 64:
        return True
    if window_left >= 0:
        return page_size >= 32
    return max_kv_len >= 8192


@triton.jit
def _split_kv_stage1(
    q_ptr,
    rel_ptr,
    k_ptr,
    v_ptr,
    block_table_ptr,
    cache_seqlens,
    mid_out_ptr,
    mid_lse_ptr,
    softmax_scale,
    stride_q_t,
    stride_q_h,
    stride_k_b,
    stride_k_p,
    stride_k_h,
    stride_k_d,
    stride_v_b,
    stride_v_p,
    stride_v_h,
    stride_v_d,
    stride_r_t,
    stride_r_h,
    stride_r_e,
    stride_mo_t,
    stride_mo_h,
    stride_mo_s,
    stride_ml_t,
    stride_ml_h,
    stride_ml_s,
    stride_bt_b: tl.constexpr,
    page_size: tl.constexpr,
    window_left: tl.constexpr,
    gqa_group_size: tl.constexpr,
    num_q_heads: tl.constexpr,
    head_dim: tl.constexpr,
    rel_extent: tl.constexpr,
    max_kv_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_s = tl.program_id(2)

    head_blocks_per_kv = tl.cdiv(gqa_group_size, BLOCK_H)
    kv_head = pid_h // head_blocks_per_kv
    valid_block_h: tl.constexpr = min(BLOCK_H, gqa_group_size)
    off_h = pid_h * valid_block_h + tl.arange(0, BLOCK_H)
    h_valid = (off_h < (pid_h + 1) * valid_block_h) & (off_h < num_q_heads)
    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim

    cache_len = tl.load(cache_seqlens + pid_t)
    effective_len = (
        tl.minimum(cache_len, window_left + 1) if window_left >= 0 else cache_len
    )
    kv_offset = cache_len - effective_len
    split_len = (
        tl.cdiv(tl.cdiv(effective_len, max_kv_splits), _MIN_BLOCK_KV) * _MIN_BLOCK_KV
    )
    split_start = split_len * pid_s
    split_end = tl.minimum(split_start + split_len, effective_len)

    q_offsets = pid_t * stride_q_t + off_h[:, None] * stride_q_h + off_d[None, :]
    row_max = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    if split_end > split_start:
        q = tl.load(
            q_ptr + q_offsets,
            mask=h_valid[:, None] & d_valid[None, :],
            other=0.0,
        )
        for start_n in range(split_start, split_end, BLOCK_N):
            off_n = start_n + tl.arange(0, BLOCK_N)
            n_valid = off_n < split_end
            token_idx = kv_offset + off_n
            logical_page = token_idx // page_size
            page_offset = token_idx % page_size
            physical_page = tl.load(
                block_table_ptr + pid_t * stride_bt_b + logical_page,
                mask=n_valid,
                other=0,
            )
            k_offsets = (
                physical_page[None, :].to(tl.int64) * stride_k_b
                + page_offset[None, :] * stride_k_p
                + kv_head * stride_k_h
                + off_d[:, None] * stride_k_d
            )
            k = tl.load(
                k_ptr + k_offsets,
                mask=d_valid[:, None] & n_valid[None, :],
                other=0.0,
            )
            scores = tl.dot(q, k.to(q.dtype)) * softmax_scale

            rel_dist = cache_len - 1 - token_idx
            rel_valid = (rel_dist >= 0) & (rel_dist < rel_extent)
            rel_idx = tl.maximum(0, tl.minimum(rel_dist, rel_extent - 1))
            rel_offsets = (
                pid_t * stride_r_t
                + off_h[:, None] * stride_r_h
                + rel_idx[None, :] * stride_r_e
            )
            rel_bias = tl.load(
                rel_ptr + rel_offsets,
                mask=h_valid[:, None] & rel_valid[None, :] & n_valid[None, :],
                other=0.0,
            )
            scores += rel_bias.to(tl.float32)
            scores = tl.where(
                h_valid[:, None] & n_valid[None, :],
                scores,
                float("-inf"),
            )

            v_offsets = (
                physical_page[:, None].to(tl.int64) * stride_v_b
                + page_offset[:, None] * stride_v_p
                + kv_head * stride_v_h
                + off_d[None, :] * stride_v_d
            )
            v = tl.load(
                v_ptr + v_offsets,
                mask=n_valid[:, None] & d_valid[None, :],
                other=0.0,
            )

            next_max = tl.maximum(tl.max(scores, axis=1), row_max)
            old_scale = tl.exp(row_max - next_max)
            probs = tl.exp(scores - next_max[:, None])
            acc *= old_scale[:, None]
            acc += tl.dot(probs.to(v.dtype), v)
            row_sum = row_sum * old_scale + tl.sum(probs, axis=1)
            row_max = next_max

        mid_out_offsets = (
            pid_t * stride_mo_t
            + off_h[:, None] * stride_mo_h
            + pid_s * stride_mo_s
            + off_d[None, :]
        )
        tl.store(
            mid_out_ptr + mid_out_offsets,
            acc / row_sum[:, None],
            mask=h_valid[:, None] & d_valid[None, :],
        )
        mid_lse_offsets = (
            pid_t * stride_ml_t + off_h * stride_ml_h + pid_s * stride_ml_s
        )
        tl.store(
            mid_lse_ptr + mid_lse_offsets,
            row_max + tl.log(row_sum),
            mask=h_valid,
        )


@triton.jit
def _split_kv_stage2(
    mid_out_ptr,
    mid_lse_ptr,
    out_ptr,
    cache_seqlens,
    stride_mo_t,
    stride_mo_h,
    stride_mo_s,
    stride_ml_t,
    stride_ml_h,
    stride_ml_s,
    stride_o_t,
    stride_o_h,
    window_left: tl.constexpr,
    head_dim: tl.constexpr,
    max_kv_splits: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    cache_len = tl.load(cache_seqlens + pid_t)
    effective_len = (
        tl.minimum(cache_len, window_left + 1) if window_left >= 0 else cache_len
    )
    split_len = (
        tl.cdiv(tl.cdiv(effective_len, max_kv_splits), _MIN_BLOCK_KV) * _MIN_BLOCK_KV
    )

    off_d = tl.arange(0, BLOCK_D)
    d_valid = off_d < head_dim
    row_max = -float("inf")
    row_sum = 0.0
    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for split_id in range(max_kv_splits):
        split_start = split_len * split_id
        split_end = tl.minimum(split_start + split_len, effective_len)
        if split_end > split_start:
            value = tl.load(
                mid_out_ptr
                + pid_t * stride_mo_t
                + pid_h * stride_mo_h
                + split_id * stride_mo_s
                + off_d,
                mask=d_valid,
                other=0.0,
            )
            split_lse = tl.load(
                mid_lse_ptr
                + pid_t * stride_ml_t
                + pid_h * stride_ml_h
                + split_id * stride_ml_s
            )
            next_max = tl.maximum(split_lse, row_max)
            old_scale = tl.exp(row_max - next_max)
            split_scale = tl.exp(split_lse - next_max)
            acc = acc * old_scale + value * split_scale
            row_sum = row_sum * old_scale + split_scale
            row_max = next_max

    tl.store(
        out_ptr + pid_t * stride_o_t + pid_h * stride_o_h + off_d,
        acc / row_sum,
        mask=d_valid,
    )


@torch.no_grad()
def inkling_rel_attention_split_kv_decode(
    q: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    *,
    block_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    softmax_scale: float,
    window_left: int,
    rel_extent: int,
    rel_logits: torch.Tensor,
    max_kv_len: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """Run split-KV relative attention for single-token decode."""
    num_kv_heads = key_cache.shape[2]
    gqa_group_size = q.shape[1] // num_kv_heads
    max_kv_splits = decode_split_count(max_kv_len, window_left)
    block_d = triton.next_power_of_2(q.shape[2])
    block_h = min(8, gqa_group_size)

    mid_out = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        q.shape[2],
        dtype=torch.float32,
        device=q.device,
    )
    mid_lse = torch.empty(
        q.shape[0],
        q.shape[1],
        max_kv_splits,
        dtype=torch.float32,
        device=q.device,
    )
    stage1_grid = (
        q.shape[0],
        triton.cdiv(q.shape[1], block_h),
        max_kv_splits,
    )
    _split_kv_stage1[stage1_grid](
        q,
        rel_logits,
        key_cache,
        value_cache,
        block_table,
        cache_seqlens,
        mid_out,
        mid_lse,
        softmax_scale,
        q.stride(0),
        q.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_cache.stride(3),
        rel_logits.stride(0),
        rel_logits.stride(1),
        rel_logits.stride(2),
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        block_table.stride(0),
        page_size=key_cache.shape[1],
        window_left=window_left,
        gqa_group_size=gqa_group_size,
        num_q_heads=q.shape[1],
        head_dim=q.shape[2],
        rel_extent=rel_extent,
        max_kv_splits=max_kv_splits,
        BLOCK_D=block_d,
        BLOCK_H=block_h,
        BLOCK_N=key_cache.shape[1],
        num_warps=4,
        num_stages=1,
    )
    stage2_grid = (q.shape[0], q.shape[1])
    _split_kv_stage2[stage2_grid](
        mid_out,
        mid_lse,
        out,
        cache_seqlens,
        mid_out.stride(0),
        mid_out.stride(1),
        mid_out.stride(2),
        mid_lse.stride(0),
        mid_lse.stride(1),
        mid_lse.stride(2),
        out.stride(0),
        out.stride(1),
        window_left=window_left,
        head_dim=q.shape[2],
        max_kv_splits=max_kv_splits,
        BLOCK_D=block_d,
        num_warps=4,
        num_stages=2,
    )
    return out
