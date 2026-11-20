# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Kitty read kernels: split-KV decode and context dequant for continuations.

Both run before this step's seal and residual write, so the cache holds
positions [0, old_len) and this step's tokens come from the step's K/V.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.kitty.tiles import (
    KITTY_PAGE_SIZE,
    kitty_load_k_tile,
    kitty_load_v_tile,
)
from vllm.v1.attention.ops.triton_decode_attention import _fwd_kernel_stage2

NUM_SPLITS = 16
# tl.dot needs at least 16 rows, so a KV head's query group is padded to 16.
G_PAD = 16


@triton.jit
def _kitty_decode_stage1(
    q_ptr,
    stride_qt,
    stride_qh,
    k_new_ptr,
    stride_kt,
    stride_kh,
    v_new_ptr,
    stride_vt,
    stride_vh,
    cache_ptr,
    stride_cb,
    stride_ch,
    res_ptr,
    stride_rb,
    stride_rh,
    res_ids_ptr,
    block_table_ptr,
    stride_bt,
    old_lens_ptr,
    mid_o_ptr,
    stride_mb,
    stride_mh,
    stride_ms,
    sm_scale,
    G: tl.constexpr,
    D: tl.constexpr,
    DB: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    G_PAD: tl.constexpr,
):
    """Stage 1 of split-KV decode: partial attention over one span of chunks.

    Grid is (request r, KV head h, split). The request's num_chunks 128-token
    chunks are divided evenly across NUM_SPLITS programs; programs with no
    chunks return without writing. The G query heads sharing KV head h are
    processed together as a [G_PAD, D] tile so the scores use `tl.dot`.

    Each chunk's K/V is read from the packed page when the block is sealed
    and from the residual otherwise (see `kitty_load_k_tile`). Positions at
    or beyond old_len are masked out. The split that owns chunk 0 also seeds
    its online softmax with the step's own K/V, which is not yet in the cache.

    For each query head, writes the normalized partial output to
    mid_o[r, head, split, :D] and its log-sum-exp to mid_o[r, head, split, D],
    the layout `_fwd_kernel_stage2` reduces over.
    """
    r = tl.program_id(0)
    h = tl.program_id(1)
    split = tl.program_id(2)

    old_len = tl.load(old_lens_ptr + r)
    num_chunks = tl.maximum(tl.cdiv(old_len, 128), 1)
    per_split = tl.cdiv(num_chunks, NUM_SPLITS)
    start = split * per_split
    end = tl.minimum(start + per_split, num_chunks)
    if start >= end:
        return

    s_k = tl.maximum(old_len // 128, 1)
    s_v = tl.maximum(old_len // 128 - 1, 1)
    offs_g = tl.arange(0, G_PAD)
    offs_d = tl.arange(0, D)
    offs_t = tl.arange(0, 128)
    g_mask = offs_g < G
    heads = h * G + offs_g

    q = tl.load(
        q_ptr + r * stride_qt + heads[:, None] * stride_qh + offs_d[None, :],
        mask=g_mask[:, None],
        other=0.0,
    )
    res_base = res_ptr + tl.load(res_ids_ptr + r).to(tl.int64) * stride_rb
    res_base += h * stride_rh
    bt_row = block_table_ptr + r.to(tl.int64) * stride_bt

    m_i = tl.full([G_PAD], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([G_PAD], dtype=tl.float32)
    acc = tl.zeros([G_PAD, D], dtype=tl.float32)

    if start == 0:
        # The new token joins chunk 0, so every split has a valid position.
        k_new = tl.load(k_new_ptr + r * stride_kt + h * stride_kh + offs_d)
        v_new = tl.load(v_new_ptr + r * stride_vt + h * stride_vh + offs_d)
        k_new = k_new.to(tl.float32)
        v_new = v_new.to(tl.float32)
        m_i = tl.sum(q.to(tl.float32) * k_new[None, :], axis=1) * sm_scale
        l_i = tl.full([G_PAD], 1.0, dtype=tl.float32)
        acc = tl.broadcast_to(v_new[None, :], (G_PAD, D))

    for b in range(start, end):
        page_id = tl.load(bt_row + b).to(tl.int64)
        page_ptr = cache_ptr + page_id * stride_cb + h * stride_ch
        k = kitty_load_k_tile(
            page_ptr, res_base, b, old_len, s_k, offs_t, offs_d, D, DB
        )
        scores = tl.dot(q, k) * sm_scale
        scores = tl.where((b * 128 + offs_t < old_len)[None, :], scores, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        v = kitty_load_v_tile(
            page_ptr, res_base, b, old_len, s_v, offs_t, offs_d, D, DB
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

    mid = mid_o_ptr + r.to(tl.int64) * stride_mb + heads * stride_mh + split * stride_ms
    tl.store(mid[:, None] + offs_d[None, :], acc / l_i[:, None], mask=g_mask[:, None])
    tl.store(mid + D, m_i + tl.log(l_i), mask=g_mask)


def kitty_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    packed_cache: torch.Tensor,
    residual: torch.Tensor,
    residual_block_ids: torch.Tensor,
    block_table: torch.Tensor,
    old_lens: torch.Tensor,
    num_chunks: torch.Tensor,
    sm_scale: float,
    d_boost: int,
    mid_o: torch.Tensor,
    lse: torch.Tensor,
) -> None:
    """Decode attention for B one-token requests over the Kitty cache.

    Each request attends to cache positions [0, old_len) plus its own new
    token, whose K/V are passed in directly because they are written to the
    cache only after attention. Runs `_kitty_decode_stage1` to compute
    NUM_SPLITS partial results per (request, query head), then the shared
    Triton `_fwd_kernel_stage2` to merge them into `output`.

    Args:
        query: [B, H_Q, D] queries of the decode tokens.
        key: [B, H_KV, D] keys of the decode tokens.
        value: [B, H_KV, D] values of the decode tokens.
        output: [B, H_Q, D] attention output, written in place.
        packed_cache: [num_blocks, H_KV, 128, C] uint8 packed pages, with
            C = `kitty_state_content_bytes(D, DB)`. Each (block, KV head)
            slice is one page, laid out as described in `tiles`.
        residual: [num_res_blocks, H_KV, 640, D] unquantized sink and tail
            rows in the model dtype.
        residual_block_ids: [B] residual block index of each request.
        block_table: [B, max_blocks] packed page ids of each request.
        old_lens: [B] cache length of each request before this step.
        num_chunks: [B] max(1, cdiv(old_len, 128)), the number of chunks
            stage 2 merges over.
        sm_scale: softmax scale applied to q·k.
        d_boost: DB, number of K channels per page stored at 4 bits.
        mid_o: [>= B, H_Q, NUM_SPLITS, D + 1] fp32 scratch for the per-split
            outputs (first D entries) and log-sum-exps (last entry).
        lse: [>= B, H_Q] fp32 scratch for the merged log-sum-exp.

    """
    batch, num_q_heads, head_size = query.shape
    num_kv_heads = key.shape[1]
    group = num_q_heads // num_kv_heads
    assert group <= G_PAD
    assert query.stride(-1) == key.stride(-1) == value.stride(-1) == 1
    _kitty_decode_stage1[(batch, num_kv_heads, NUM_SPLITS)](
        query,
        query.stride(0),
        query.stride(1),
        key,
        key.stride(0),
        key.stride(1),
        value,
        value.stride(0),
        value.stride(1),
        packed_cache,
        packed_cache.stride(0),
        packed_cache.stride(1),
        residual,
        residual.stride(0),
        residual.stride(1),
        residual_block_ids,
        block_table,
        block_table.stride(0),
        old_lens,
        mid_o,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        sm_scale,
        G=group,
        D=head_size,
        DB=d_boost,
        NUM_SPLITS=NUM_SPLITS,
        G_PAD=G_PAD,
        num_warps=4,
    )
    _fwd_kernel_stage2[(batch, num_q_heads)](
        mid_o,
        output,
        lse,
        num_chunks,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_SPLITS,
        BLOCK_DV=head_size,
        Lv=head_size,
        OUTPUT_FP16=int(output.dtype == torch.float16),
        num_warps=4,
        num_stages=2,
    )


@triton.jit
def _kitty_dequant_context_kernel(
    k_out_ptr,
    v_out_ptr,
    stride_ot,
    stride_oh,
    cache_ptr,
    stride_cb,
    stride_ch,
    res_ptr,
    stride_rb,
    stride_rh,
    res_ids_ptr,
    block_table_ptr,
    stride_bt,
    req,
    old_len,
    D: tl.constexpr,
    DB: tl.constexpr,
):
    """Dequantize one 128-token block of one KV head of request `req`.

    Grid is (block b, KV head h). Loads block b's K and V from the packed page
    if sealed, else from the residual, and writes rows [128 * b, 128 * b + 128)
    clipped to old_len into k_out/v_out at head h.
    """
    b = tl.program_id(0)
    h = tl.program_id(1)
    offs_t = tl.arange(0, 128)
    offs_d = tl.arange(0, D)
    s_k = tl.maximum(old_len // 128, 1)
    s_v = tl.maximum(old_len // 128 - 1, 1)
    page_id = tl.load(block_table_ptr + req * stride_bt + b).to(tl.int64)
    page_ptr = cache_ptr + page_id * stride_cb + h * stride_ch
    res_base = res_ptr + tl.load(res_ids_ptr + req).to(tl.int64) * stride_rb
    res_base += h * stride_rh
    valid = b * 128 + offs_t < old_len
    k = kitty_load_k_tile(page_ptr, res_base, b, old_len, s_k, offs_t, offs_d, D, DB)
    v = kitty_load_v_tile(page_ptr, res_base, b, old_len, s_v, offs_t, offs_d, D, DB)
    rows = (b * 128 + offs_t).to(tl.int64) * stride_ot + h * stride_oh
    tl.store(k_out_ptr + rows[None, :] + offs_d[:, None], k, mask=valid[None, :])
    tl.store(v_out_ptr + rows[:, None] + offs_d[None, :], v, mask=valid[:, None])


# TODO: `kitty_dequant_context` is only used by continuation prefills, and
# materializes the entire cache in HBM. A fused prefill kernel that dequantizes
# on the fly is better. Also, batched continuation prefills are needed.
def kitty_dequant_context(
    packed_cache: torch.Tensor,
    residual: torch.Tensor,
    residual_block_ids: torch.Tensor,
    block_table: torch.Tensor,
    req: int,
    old_len: int,
    k_out: torch.Tensor,
    v_out: torch.Tensor,
    d_boost: int,
) -> None:
    """Materialize a request's cached K/V as dense tensors.

    Used only for continuation prefills, which run flash attention over the
    dequantized context followed by the step's own tokens. Sealed blocks are
    dequantized from their packed pages; the sink and tails are copied from
    the residual. Does nothing if old_len is 0.

    Args:
        packed_cache: [num_blocks, H_KV, 128, C] uint8 packed pages, with
            C = `kitty_state_content_bytes(D, DB)`. Each (block, KV head)
            slice is one page, laid out as described in `tiles`.
        residual: [num_res_blocks, H_KV, 640, D] unquantized sink and tail
            rows in the model dtype.
        residual_block_ids: [num_reqs] residual block index of each request.
        block_table: [num_reqs, max_blocks] packed page ids of each request.
        req: batch index of the request to dequantize.
        old_len: cache length of the request before this step.
        k_out: [>= old_len, H_KV, D] output keys in the model dtype; rows
            [0, old_len) are written.
        v_out: same shape and strides as k_out, for values.
        d_boost: DB, number of K channels per page stored at 4 bits.

    """
    if old_len == 0:
        return
    num_kv_heads, head_size = residual.shape[1], residual.shape[3]
    assert k_out.stride() == v_out.stride() and k_out.stride(-1) == 1
    _kitty_dequant_context_kernel[
        (triton.cdiv(old_len, KITTY_PAGE_SIZE), num_kv_heads)
    ](
        k_out,
        v_out,
        k_out.stride(0),
        k_out.stride(1),
        packed_cache,
        packed_cache.stride(0),
        packed_cache.stride(1),
        residual,
        residual.stride(0),
        residual.stride(1),
        residual_block_ids,
        block_table,
        block_table.stride(0),
        req,
        old_len,
        D=head_size,
        DB=d_boost,
        num_warps=4,
    )
