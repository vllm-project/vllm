# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Query-tiled block-sparse verify-decode attention for MiniMax M3.

``_gqa_sparse_decode_kernel`` launches one program per (query token, top-k
chunk).  Under speculative decoding a verify step carries ``decode_query_len``
(``dnum``) query rows per request, so the selected KV blocks are re-streamed
``dnum`` times -- once per row -- even though adjacent draft rows of a request
select heavily overlapping blocks.  See vllm-project/vllm#47763.

This kernel keeps the request on the grid axis instead of the query token, loads
each selected KV block once, and sweeps every draft row of that request against
it with a single ``tl.dot``.  Two properties make that safe:

* **Exactness.**  Top-k is chosen per query token, so rows of one request do not
  in general agree.  The kernel therefore iterates the *union* of the rows'
  block lists and carries a per-block row bitmask, applying membership alongside
  the causal bound.  Row ``r`` attends to exactly the blocks it selected -- the
  result is numerically the same computation the shipped kernel performs.
* **Occupancy.**  Tiling the query axis collapses the grid from ``total_q`` to
  ``num_reqs``, which starves a large GPU at serving batch sizes.  Split-K over
  the union restores it, and the split is *rank-balanced* -- chunks are cut over
  the runtime union size, not a compile-time bound, so a request whose rows
  overlap heavily does not leave most of its chunks empty.  This matters more
  than it sounds: without it the kernel is slowest exactly when the underlying
  idea works best.

Softmax is base-2 (``exp2``/``log2``) with ``log2e`` folded into the scale, to
match ``sparse_attn.py``.
"""

import torch

from vllm.triton_utils import tl, triton

from vllm.models.minimax_m3.common.ops.sparse_attn import (
    SPARSE_BLOCK_SIZE,
    _FP8_DTYPES,
)

__all__ = ["build_verify_block_lists", "minimax_m3_sparse_attn_verify_decode"]


@triton.jit
def _gqa_sparse_verify_decode_kernel(
    q_ptr,  # [num_reqs, dnum, num_heads, head_dim]
    kv_ptr,  # [num_blocks, 2, 128, num_kv_heads, head_dim]
    blk_ptr,  # [num_kv_heads, num_reqs, max_union] int32 logical block ids
    msk_ptr,  # [num_kv_heads, num_reqs, max_union] int32 row bitmask
    nblk_ptr,  # [num_kv_heads, num_reqs] int32 union size
    block_table_ptr,  # [num_reqs, max_blocks]
    prefix_lens,  # [num_reqs] int32 context length before the draft rows
    o_partial_ptr,  # [chunks, num_reqs, dnum, num_heads, head_dim] fp32
    lse_partial_ptr,  # [chunks, num_reqs, dnum, num_heads] fp32
    sm_scale_log2e,
    gqa_group_size,
    head_dim,
    stride_q_b, stride_q_r, stride_q_h, stride_q_d,
    stride_kv_blk, stride_kv_pos, stride_kv_h, stride_kv_d,
    v_elem_offset,  # K->V displacement; absorbs the cache layout (see launcher)
    stride_blk_h, stride_blk_b,
    stride_nblk_h,
    stride_bt_b,
    stride_op_c, stride_op_b, stride_op_r, stride_op_h, stride_op_d,
    stride_lp_c, stride_lp_b, stride_lp_r, stride_lp_h,
    BLOCK_SIZE_Q: tl.constexpr,  # == dnum
    BLOCK_SIZE_K: tl.constexpr,  # == SPARSE_BLOCK_SIZE
    BLOCK_SIZE_D: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_QH: tl.constexpr,
    NUM_CHUNKS: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    pid_b = tl.program_id(0) // NUM_CHUNKS
    pid_c = tl.program_id(0) % NUM_CHUNKS
    pid_kh = tl.program_id(1)

    prefix = tl.load(prefix_lens + pid_b)
    n_union = tl.load(nblk_ptr + pid_kh * stride_nblk_h + pid_b)

    # Rank-balanced split: cut over the runtime union size so every chunk gets
    # work even when the draft rows agree and the union collapses toward topk.
    chunk = (n_union + NUM_CHUNKS - 1) // NUM_CHUNKS
    lo = pid_c * chunk
    hi = tl.minimum(lo + chunk, n_union)

    off_d = tl.arange(0, BLOCK_SIZE_D)
    off_n = tl.arange(0, BLOCK_SIZE_K)
    qh = tl.arange(0, BLOCK_SIZE_QH)
    qh_r = qh // BLOCK_SIZE_H  # draft row served by this lane
    qh_h = qh % BLOCK_SIZE_H  # head within the GQA group
    d_mask = off_d < head_dim
    lane = (qh_h < gqa_group_size) & (qh_r < BLOCK_SIZE_Q)

    q = tl.load(
        q_ptr + pid_b * stride_q_b + qh_r[:, None] * stride_q_r
        + (pid_kh * gqa_group_size + qh_h)[:, None] * stride_q_h
        + off_d[None, :] * stride_q_d,
        mask=lane[:, None] & d_mask[None, :], other=0.0)

    qpos = prefix + qh_r  # row r of the verify group sits at prefix + r
    m_i = tl.full((BLOCK_SIZE_QH,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_SIZE_QH,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_SIZE_QH, BLOCK_SIZE_D), dtype=tl.float32)

    for i in range(lo, hi):
        blk = tl.load(blk_ptr + pid_kh * stride_blk_h + pid_b * stride_blk_b + i)
        bits = tl.load(msk_ptr + pid_kh * stride_blk_h + pid_b * stride_blk_b + i)
        page = tl.load(block_table_ptr + pid_b * stride_bt_b + blk).to(tl.int64)
        pos = blk * BLOCK_SIZE_K + off_n

        kv_base = kv_ptr + page * stride_kv_blk + pid_kh * stride_kv_h
        k = tl.load(
            kv_base + off_n[None, :] * stride_kv_pos
            + off_d[:, None] * stride_kv_d,
            mask=d_mask[:, None], other=0.0)
        if USE_FP8:
            k = k.to(q.dtype)
        qk = tl.dot(q, k) * sm_scale_log2e

        # causal bound and per-row membership, folded into one mask
        row_in = ((bits >> qh_r) & 1) == 1
        ok = (pos[None, :] <= qpos[:, None]) & row_in[:, None] & lane[:, None]
        qk = tl.where(ok, qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        p = tl.where(ok, tl.exp2(qk - m_safe[:, None]), 0.0)
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp2(m_i - m_safe))
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            kv_base + v_elem_offset + off_n[:, None] * stride_kv_pos
            + off_d[None, :] * stride_kv_d,
            mask=d_mask[None, :], other=0.0)
        if USE_FP8:
            v = v.to(q.dtype)
        acc += tl.dot(p.to(v.dtype), v)
        m_i = m_new

    lse = tl.where(l_i > 0, m_i + tl.log2(tl.where(l_i > 0, l_i, 1.0)),
                   float("-inf"))
    out = acc / tl.where(l_i > 0, l_i, 1.0)[:, None]

    tl.store(
        o_partial_ptr + pid_c * stride_op_c + pid_b * stride_op_b
        + qh_r[:, None] * stride_op_r
        + (pid_kh * gqa_group_size + qh_h)[:, None] * stride_op_h
        + off_d[None, :] * stride_op_d,
        out, mask=lane[:, None] & d_mask[None, :])
    tl.store(
        lse_partial_ptr + pid_c * stride_lp_c + pid_b * stride_lp_b
        + qh_r * stride_lp_r
        + (pid_kh * gqa_group_size + qh_h) * stride_lp_h,
        lse, mask=lane)


@triton.jit
def _merge_verify_out_kernel(
    o_partial_ptr, lse_partial_ptr, o_ptr,
    stride_op_c, stride_op_b, stride_op_r, stride_op_h, stride_op_d,
    stride_lp_c, stride_lp_b, stride_lp_r, stride_lp_h,
    stride_o_n, stride_o_h, stride_o_d,
    dnum, head_dim,
    NUM_CHUNKS: tl.constexpr, BLOCK_SIZE_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)
    b = pid_t // dnum
    r = pid_t % dnum
    off_d = tl.arange(0, BLOCK_SIZE_D)
    d_mask = off_d < head_dim

    m = float("-inf")
    for c in range(NUM_CHUNKS):
        m = tl.maximum(
            m, tl.load(lse_partial_ptr + c * stride_lp_c + b * stride_lp_b
                       + r * stride_lp_r + pid_h * stride_lp_h))
    acc = tl.zeros((BLOCK_SIZE_D,), dtype=tl.float32)
    den = 0.0
    for c in range(NUM_CHUNKS):
        lse = tl.load(lse_partial_ptr + c * stride_lp_c + b * stride_lp_b
                      + r * stride_lp_r + pid_h * stride_lp_h)
        w = tl.where(lse == float("-inf"), 0.0, tl.exp2(lse - m))
        o = tl.load(o_partial_ptr + c * stride_op_c + b * stride_op_b
                    + r * stride_op_r + pid_h * stride_op_h + off_d * stride_op_d,
                    mask=d_mask, other=0.0)
        acc += w * o
        den += w
    tl.store(o_ptr + (b * dnum + r) * stride_o_n + pid_h * stride_o_h
             + off_d * stride_o_d,
             acc / tl.where(den > 0, den, 1.0), mask=d_mask)



@triton.jit
def _build_union_kernel(
    t_ptr,  # topk_idx [num_kv_heads, num_reqs * dnum, topk]
    blk_ptr,  # [num_kv_heads, num_reqs, MAXB] int32
    msk_ptr,  # [num_kv_heads, num_reqs, MAXB] int32
    nblk_ptr,  # [num_kv_heads, num_reqs] int32
    stride_t_h, stride_t_n, stride_t_k,
    stride_o_h, stride_o_b,
    stride_nb_h,
    topk,
    n_slots,  # dnum*topk; MAXB is its power-of-two round-up
    DNUM: tl.constexpr,
    MAXB: tl.constexpr,
):
    """Union of a request's per-row top-k lists, in one program per (head, req).

    A torch sort + per-row scatter_reduce costs ~190 us at serving shapes -- an
    order of magnitude more than the attend it feeds, and it runs once per
    sparse layer. MAXB is at most dnum*topk (128 for dnum=8, topk=16), so an
    O(MAXB^2) first-occurrence test fits comfortably in registers and collapses
    the whole build into a single cheap launch.
    """
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)

    i = tl.arange(0, MAXB)
    row = i // topk
    in_range = i < n_slots
    ids = tl.load(t_ptr + pid_h * stride_t_h
                  + (pid_b * DNUM + row) * stride_t_n
                  + (i % topk) * stride_t_k,
                  mask=in_range, other=-1)
    valid = (ids >= 0) & in_range

    same = (ids[:, None] == ids[None, :]) & valid[:, None] & valid[None, :]
    earlier = same & (i[None, :] < i[:, None])
    is_first = valid & (tl.sum(earlier.to(tl.int32), axis=1) == 0)
    rank = tl.cumsum(is_first.to(tl.int32), axis=0) - 1
    n_union = tl.sum(is_first.to(tl.int32), axis=0)

    # bit r of entry i is set iff ANY occurrence of that id belongs to row r --
    # every occurrence counts, not just the first one.
    bits = tl.zeros((MAXB,), dtype=tl.int32)
    for r in tl.static_range(DNUM):
        present = tl.sum((same & (row[None, :] == r)).to(tl.int32), axis=1) > 0
        bits |= tl.where(present, 1 << r, 0).to(tl.int32)

    out = blk_ptr + pid_h * stride_o_h + pid_b * stride_o_b
    tl.store(out + rank, ids.to(tl.int32), mask=is_first)
    tl.store(msk_ptr + pid_h * stride_o_h + pid_b * stride_o_b + rank, bits,
             mask=is_first)
    tl.store(nblk_ptr + pid_h * stride_nb_h + pid_b, n_union)


@torch.no_grad()
def build_verify_block_lists(
    topk_idx: torch.Tensor,  # [num_kv_heads, num_reqs * dnum, topk]
    num_reqs: int,
    dnum: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Union of each request's per-row top-k lists, plus a row bitmask.

    Returns ``(blocks, row_mask, union_size)`` shaped
    ``([H, R, dnum*topk], [H, R, dnum*topk], [H, R])``.

    Cost is one sort plus ``dnum`` scatter-reduces -- no per-request loop, so
    this stays a handful of kernel launches independent of batch size.
    """
    num_kv_heads, _, topk = topk_idx.shape
    maxb = dnum * topk
    dev = topk_idx.device
    blocks = torch.zeros(num_kv_heads, num_reqs, maxb, dtype=torch.int32, device=dev)
    row_mask = torch.zeros(num_kv_heads, num_reqs, maxb, dtype=torch.int32, device=dev)
    union_size = torch.zeros(num_kv_heads, num_reqs, dtype=torch.int32, device=dev)
    _build_union_kernel[(num_reqs, num_kv_heads)](
        topk_idx, blocks, row_mask, union_size,
        topk_idx.stride(0), topk_idx.stride(1), topk_idx.stride(2),
        blocks.stride(0), blocks.stride(1), union_size.stride(0),
        topk, maxb, DNUM=dnum, MAXB=triton.next_power_of_2(maxb),
    )
    return blocks, row_mask, union_size


@torch.no_grad()
def minimax_m3_sparse_attn_verify_decode(
    q: torch.Tensor,  # [num_reqs * dnum, num_heads, head_dim]
    kv_cache: torch.Tensor,
    topk_idx: torch.Tensor,  # [num_kv_heads, num_reqs * dnum, topk]
    block_table: torch.Tensor,
    prefix_lens: torch.Tensor,  # [num_reqs] int32, context before the draft rows
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    dnum: int,
    num_chunks: int | None = None,
) -> None:
    """Query-tiled verify-decode. Equivalent to the per-token decode kernel."""
    total_q, num_heads, head_dim = q.shape
    num_reqs = total_q // dnum
    assert total_q == num_reqs * dnum
    gqa = num_heads // num_kv_heads

    blocks, row_mask, union_size = build_verify_block_lists(
        topk_idx, num_reqs, dnum)

    if num_chunks is None:
        # keep the machine fed: the q-tile removes the token axis from the grid,
        # so without split-K the launch is only num_reqs CTAs.
        target = 256
        num_chunks = max(1, min(topk_idx.shape[-1],
                                target // max(1, num_reqs * num_kv_heads)))
        num_chunks = 1 << (num_chunks.bit_length() - 1)

    block_q = dnum
    block_h = triton.next_power_of_2(gqa)
    while block_q * block_h < 16:  # tl.dot needs M >= 16
        block_h *= 2
    block_d = triton.next_power_of_2(head_dim)

    # The main KV cache has had two layouts:
    #   [num_blocks, num_kv_heads, 128, 2*head_dim]   -> V starts at head_dim
    #   [num_blocks, 2, 128, num_kv_heads, head_dim]  -> V is the second plane
    # Take the K/V displacement from the shape rather than hard-coding either.
    if kv_cache.ndim == 5:
        s_blk, s_kv, s_pos, s_h, s_d = kv_cache.stride()
        v_elem_offset = s_kv
    else:
        s_blk, s_h, s_pos, s_d = kv_cache.stride()
        v_elem_offset = head_dim * s_d

    o_partial = torch.empty(num_chunks, num_reqs, dnum, num_heads, head_dim,
                            dtype=torch.float32, device=q.device)
    lse_partial = torch.empty(num_chunks, num_reqs, dnum, num_heads,
                              dtype=torch.float32, device=q.device)
    q4 = q.view(num_reqs, dnum, num_heads, head_dim)

    _gqa_sparse_verify_decode_kernel[(num_reqs * num_chunks, num_kv_heads)](
        q4, kv_cache, blocks, row_mask, union_size, block_table, prefix_lens,
        o_partial, lse_partial,
        sm_scale * 1.4426950408889634, gqa, head_dim,
        q4.stride(0), q4.stride(1), q4.stride(2), q4.stride(3),
        s_blk, s_pos, s_h, s_d, v_elem_offset,
        blocks.stride(0), blocks.stride(1), union_size.stride(0),
        block_table.stride(0),
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2),
        o_partial.stride(3), o_partial.stride(4),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        lse_partial.stride(3),
        BLOCK_SIZE_Q=block_q, BLOCK_SIZE_K=SPARSE_BLOCK_SIZE,
        BLOCK_SIZE_D=block_d, BLOCK_SIZE_H=block_h,
        BLOCK_SIZE_QH=block_q * block_h, NUM_CHUNKS=num_chunks,
        USE_FP8=kv_cache.dtype in _FP8_DTYPES,
    )
    _merge_verify_out_kernel[(num_reqs * dnum, num_heads)](
        o_partial, lse_partial, output,
        o_partial.stride(0), o_partial.stride(1), o_partial.stride(2),
        o_partial.stride(3), o_partial.stride(4),
        lse_partial.stride(0), lse_partial.stride(1), lse_partial.stride(2),
        lse_partial.stride(3),
        output.stride(0), output.stride(1), output.stride(2),
        dnum, head_dim, NUM_CHUNKS=num_chunks, BLOCK_SIZE_D=block_d,
    )
