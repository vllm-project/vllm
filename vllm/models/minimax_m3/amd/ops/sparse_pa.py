# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""AITER page-16 sparse paged-attention helpers for MiniMax-M3 on ROCm."""

import torch

try:
    from vllm.triton_utils import tl, triton
except ModuleNotFoundError:
    import triton
    import triton.language as tl

from vllm.models.minimax_m3.common.ops.sparse_attn import SPARSE_BLOCK_SIZE

ASM_PAGE_SIZE = 16
PAGES_PER_SPARSE_BLOCK = SPARSE_BLOCK_SIZE // ASM_PAGE_SIZE

_FP8_DTYPES = {
    dtype
    for dtype in (
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e4m3fnuz", None),
        getattr(torch, "float8_e5m2", None),
        getattr(torch, "float8_e5m2fnuz", None),
    )
    if dtype is not None
}


def _is_fp8_kv_cache_tensor(kv_cache: torch.Tensor) -> bool:
    return kv_cache.dtype in _FP8_DTYPES


@triton.jit
def _write_sparse_block_table_row_from_values(
    blk,  # [BLOCK_SIZE_T] int32 selected logical block ids for this query
    bt_row,  # [max_blocks] int32, this request's logical page table
    sbt_row,  # [topk * PAGES_PER_BLOCK] int32, physical 16-page table
    ctx_ptr,  # int32, this query's attended context length
    abs_pos,  # absolute position of this query token, may be negative
    max_topk,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    """Compact one query's selected logical blocks into physical page-16 ids.

    ``BLOCK_PAGE_STRIDE`` is how many page ids a block spans, which is
    ``PAGES_PER_BLOCK`` when each K/V side is its own dense plane and twice
    that when both sides share a block. Padded rows carry a negative
    ``abs_pos``, which clamps the causal range to empty.
    """
    causal_len = tl.maximum(abs_pos + 1, 0)
    self_blk = abs_pos // SPARSE_BLOCK_SIZE_C

    off_t = tl.arange(0, BLOCK_SIZE_T)
    valid = (off_t < max_topk) & (causal_len > 0) & (blk >= 0) & (blk <= self_blk)
    is_tail = valid & (blk == self_blk)
    is_full = valid & (blk < self_blk)

    n_full = tl.sum(is_full.to(tl.int32), axis=0)
    n_valid = tl.sum(valid.to(tl.int32), axis=0)
    earlier_full = tl.cumsum(is_full.to(tl.int32), axis=0) - is_full.to(tl.int32)
    slot = tl.where(is_full, earlier_full, n_full)

    logical_page = tl.load(bt_row + blk, mask=valid, other=0).to(tl.int32)
    base_phys = logical_page * BLOCK_PAGE_STRIDE
    dst_base = slot * PAGES_PER_BLOCK

    for j in tl.static_range(PAGES_PER_BLOCK):
        tl.store(sbt_row + dst_base + j, base_phys + j, mask=valid)

    n_used = n_valid * PAGES_PER_BLOCK
    off_w = tl.arange(0, BLOCK_SIZE_T * PAGES_PER_BLOCK)
    row_width = max_topk * PAGES_PER_BLOCK
    tl.store(
        sbt_row + off_w,
        tl.zeros_like(off_w),
        mask=(off_w >= n_used) & (off_w < row_width),
    )

    tail_tokens = causal_len - self_blk * SPARSE_BLOCK_SIZE_C
    has_tail = tl.sum(is_tail.to(tl.int32), axis=0) > 0
    ctx = n_full * SPARSE_BLOCK_SIZE_C + tl.where(has_tail, tail_tokens, 0)
    ctx = tl.where(has_tail, ctx, tl.minimum(n_valid * SPARSE_BLOCK_SIZE_C, causal_len))
    tl.store(ctx_ptr, ctx)


@triton.jit
def _write_sparse_block_table_row(
    topk_row,  # [topk] int32, selected logical block ids for this query
    bt_row,  # [max_blocks] int32, this request's logical page table
    sbt_row,  # [topk * PAGES_PER_BLOCK] int32, physical 16-page table
    ctx_ptr,  # int32, this query's attended context length
    abs_pos,  # absolute position of this query token, may be negative
    max_topk,
    stride_topk_k,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    off_t = tl.arange(0, BLOCK_SIZE_T)
    blk = tl.load(topk_row + off_t * stride_topk_k, mask=off_t < max_topk, other=-1)
    _write_sparse_block_table_row_from_values(
        blk,
        bt_row,
        sbt_row,
        ctx_ptr,
        abs_pos,
        max_topk,
        SPARSE_BLOCK_SIZE_C,
        PAGES_PER_BLOCK,
        BLOCK_PAGE_STRIDE,
        BLOCK_SIZE_T,
    )


@triton.jit
def _build_sparse_block_table_kernel(
    topk_ptr,  # [1, batch * DECODE_QUERY_LEN, topk] int32, logical block ids
    block_table_ptr,  # [batch, max_blocks] int32, logical 128-page table
    seq_lens_ptr,  # [batch] int32
    sparse_bt_ptr,  # [batch * DECODE_QUERY_LEN, topk * 8] int32
    sparse_ctx_ptr,  # [batch * DECODE_QUERY_LEN] int32
    max_topk,
    stride_topk_n,
    stride_topk_k,
    stride_bt_b,
    stride_sbt_b,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
    DECODE_QUERY_LEN: tl.constexpr,
):
    # One program per query token. The decode indexer flattens speculative rows
    # as ``req_id * DECODE_QUERY_LEN + local_q``, and a request's query tokens
    # are the last DECODE_QUERY_LEN positions of its sequence, so both the
    # request id and the absolute position follow from the program id.
    pid_n = tl.program_id(0)
    req_id = pid_n // DECODE_QUERY_LEN
    local_q = pid_n % DECODE_QUERY_LEN
    seq_len = tl.load(seq_lens_ptr + req_id)
    abs_pos = seq_len - DECODE_QUERY_LEN + local_q

    _write_sparse_block_table_row(
        topk_ptr + pid_n * stride_topk_n,
        block_table_ptr + req_id * stride_bt_b,
        sparse_bt_ptr + pid_n * stride_sbt_b,
        sparse_ctx_ptr + pid_n,
        abs_pos,
        max_topk,
        stride_topk_k,
        SPARSE_BLOCK_SIZE_C,
        PAGES_PER_BLOCK,
        BLOCK_PAGE_STRIDE,
        BLOCK_SIZE_T,
    )


@triton.jit
def _build_sparse_block_table_prefill_kernel(
    topk_ptr,  # [1, total_q, topk]
    block_table_ptr,  # [batch, max_blocks]
    req_id_ptr,  # [total_q]
    abs_pos_ptr,  # [total_q]
    sparse_bt_ptr,  # [total_q, topk * 8]
    sparse_ctx_ptr,  # [total_q]
    max_topk,
    stride_topk_n,
    stride_topk_k,
    stride_bt_b,
    stride_sbt_n,
    SPARSE_BLOCK_SIZE_C: tl.constexpr,
    PAGES_PER_BLOCK: tl.constexpr,
    BLOCK_PAGE_STRIDE: tl.constexpr,
    BLOCK_SIZE_T: tl.constexpr,
):
    pid_n = tl.program_id(0)
    req_id = tl.load(req_id_ptr + pid_n)
    abs_pos = tl.load(abs_pos_ptr + pid_n)

    _write_sparse_block_table_row(
        topk_ptr + pid_n * stride_topk_n,
        block_table_ptr + req_id * stride_bt_b,
        sparse_bt_ptr + pid_n * stride_sbt_n,
        sparse_ctx_ptr + pid_n,
        abs_pos,
        max_topk,
        stride_topk_k,
        SPARSE_BLOCK_SIZE_C,
        PAGES_PER_BLOCK,
        BLOCK_PAGE_STRIDE,
        BLOCK_SIZE_T,
    )


def minimax_m3_alloc_sparse_block_table(
    topk_idx: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert topk_idx.shape[0] == 1, "AITER sparse PA requires num_kv_heads == 1"
    total_q = topk_idx.shape[1]
    width = topk_idx.shape[-1] * PAGES_PER_SPARSE_BLOCK
    return (
        torch.empty((total_q, width), dtype=torch.int32, device=topk_idx.device),
        torch.empty((total_q,), dtype=torch.int32, device=topk_idx.device),
    )


@torch.no_grad()
def minimax_m3_build_sparse_block_table_prefill(
    topk_idx: torch.Tensor,  # [1, total_q, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    query_req_id: torch.Tensor,  # [total_q]
    query_abs_pos: torch.Tensor,  # [total_q]
    block_page_stride: int = PAGES_PER_SPARSE_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one page-16 sparse block table row per prefill query token."""
    sparse_bt, sparse_ctx = minimax_m3_alloc_sparse_block_table(topk_idx)
    topk = topk_idx.shape[-1]
    _build_sparse_block_table_prefill_kernel[(topk_idx.shape[1],)](
        topk_idx,
        block_table,
        query_req_id,
        query_abs_pos,
        sparse_bt,
        sparse_ctx,
        topk,
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        sparse_bt.stride(0),
        SPARSE_BLOCK_SIZE_C=SPARSE_BLOCK_SIZE,
        PAGES_PER_BLOCK=PAGES_PER_SPARSE_BLOCK,
        BLOCK_PAGE_STRIDE=block_page_stride,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
    )
    return sparse_bt, sparse_ctx


@torch.no_grad()
def minimax_m3_build_sparse_block_table_decode(
    topk_idx: torch.Tensor,  # [1, batch * decode_query_len, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    seq_lens: torch.Tensor,  # [batch]
    decode_query_len: int = 1,
    block_page_stride: int = PAGES_PER_SPARSE_BLOCK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build one page-16 sparse block table row per decode query token."""
    total_q = topk_idx.shape[1]
    expected_q = block_table.shape[0] * decode_query_len
    assert total_q == expected_q, (
        "MiniMax-M3 decode top-k rows must equal batch * decode_query_len: "
        f"{total_q} != {expected_q}"
    )
    sparse_bt, sparse_ctx = minimax_m3_alloc_sparse_block_table(topk_idx)
    topk = topk_idx.shape[-1]
    _build_sparse_block_table_kernel[(total_q,)](
        topk_idx,
        block_table,
        seq_lens,
        sparse_bt,
        sparse_ctx,
        topk,
        topk_idx.stride(1),
        topk_idx.stride(2),
        block_table.stride(0),
        sparse_bt.stride(0),
        SPARSE_BLOCK_SIZE_C=SPARSE_BLOCK_SIZE,
        PAGES_PER_BLOCK=PAGES_PER_SPARSE_BLOCK,
        BLOCK_PAGE_STRIDE=block_page_stride,
        BLOCK_SIZE_T=triton.next_power_of_2(topk),
        DECODE_QUERY_LEN=decode_query_len,
    )
    return sparse_bt, sparse_ctx


# Retained for the decode_query_len == 1 callers (and tests) that predate the
# speculative-decode rows.
minimax_m3_build_sparse_block_table = minimax_m3_build_sparse_block_table_decode


@triton.jit
def _insert_index_cache_kernel(
    index_k_ptr,  # [num_tokens, head_dim]
    index_cache_ptr,  # [num_blocks, block_size, head_dim]
    slot_mapping_ptr,  # [num_tokens]
    stride_k_t,
    stride_k_d,
    stride_c_b,
    stride_c_t,
    stride_c_d,
    stride_slot_t,
    CACHE_BLOCK_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    offs_d = tl.arange(0, BLOCK_D)
    slot = tl.load(slot_mapping_ptr + pid_t * stride_slot_t)
    block_id = slot // CACHE_BLOCK_SIZE
    block_off = slot - block_id * CACHE_BLOCK_SIZE

    src = index_k_ptr + pid_t * stride_k_t + offs_d * stride_k_d
    dst = (
        index_cache_ptr
        + block_id * stride_c_b
        + block_off * stride_c_t
        + offs_d * stride_c_d
    )
    mask = (slot >= 0) & (offs_d < HEAD_DIM)
    value = tl.load(src, mask=offs_d < HEAD_DIM, other=0.0)
    tl.store(dst, value, mask=mask)


@torch.no_grad()
def minimax_m3_insert_index_cache(
    index_k: torch.Tensor,
    index_cache: torch.Tensor,
    index_slot_mapping: torch.Tensor,
) -> None:
    """Scatter index keys into MiniMax-M3's key-only side cache."""
    if index_k.numel() == 0 or index_cache.numel() == 0:
        return
    if index_k.dim() != 2 or index_cache.dim() != 3:
        raise ValueError("MiniMax-M3 index cache insert expects [N,D] and [B,T,D]")
    if index_k.shape[1] != index_cache.shape[2]:
        raise ValueError("MiniMax-M3 index key dim must match index cache head dim")
    if index_slot_mapping.dim() != 1 or index_slot_mapping.shape[0] != index_k.shape[0]:
        raise ValueError("MiniMax-M3 index slot mapping must be a length-N vector")
    if index_cache.stride(2) != 1:
        raise ValueError("MiniMax-M3 index cache requires contiguous head dimension")

    head_dim = index_k.shape[1]
    _insert_index_cache_kernel[(index_k.shape[0],)](
        index_k,
        index_cache,
        index_slot_mapping,
        index_k.stride(0),
        index_k.stride(1),
        index_cache.stride(0),
        index_cache.stride(1),
        index_cache.stride(2),
        index_slot_mapping.stride(0),
        CACHE_BLOCK_SIZE=index_cache.shape[1],
        HEAD_DIM=head_dim,
        BLOCK_D=triton.next_power_of_2(head_dim),
        num_warps=4,
    )


def _sides_are_packed(k_cache: torch.Tensor, v_cache: torch.Tensor) -> bool:
    """Whether a block holds both its K and V pages instead of one side."""
    return v_cache.shape[0] != k_cache.shape[0]


def minimax_m3_sparse_block_page_stride(
    k_cache: torch.Tensor, v_cache: torch.Tensor
) -> int:
    """How many page ids one sparse block spans.

    Each side owns ``PAGES_PER_SPARSE_BLOCK`` pages. When the sides are dense
    planes a block is exactly that wide; when they interleave, a block covers
    both sides' pages and V is reached from the same page id at a fixed offset.
    """
    return PAGES_PER_SPARSE_BLOCK * (2 if _sides_are_packed(k_cache, v_cache) else 1)


def _gluon_scale_arg(
    scale: torch.Tensor | None,
    *,
    num_phys_pages: int,
    num_kv_heads: int,
) -> torch.Tensor | None:
    if scale is None:
        return None
    if scale.numel() == 1:
        return scale
    if scale.dim() != 2 or scale.shape[0] != num_kv_heads:
        raise ValueError(
            "MiniMax-M3 AITER sparse PA supports scalar KV scales or "
            "[num_kv_heads, max_kv_tokens] scales"
        )
    max_tokens = num_phys_pages * ASM_PAGE_SIZE
    if scale.shape[1] < max_tokens:
        raise ValueError(
            "MiniMax-M3 AITER sparse PA KV scale tensor is smaller than the "
            f"cache token capacity ({scale.shape[1]} < {max_tokens})"
        )
    scale = scale[:, :max_tokens]
    return (
        scale.transpose(0, 1)
        .contiguous()
        .view(num_phys_pages, ASM_PAGE_SIZE, num_kv_heads)
        .permute(0, 2, 1)
        .contiguous()
        .unsqueeze(-1)
    )


@torch.no_grad()
def minimax_m3_sparse_attn_decode_aiter(
    q: torch.Tensor,  # [batch * decode_query_len, num_heads, head_dim]
    k_cache: torch.Tensor,  # [phys16, num_kv_heads, head_dim // x, 16, x]
    v_cache: torch.Tensor,  # [phys16, num_kv_heads, 16 // x, head_dim, x]
    topk_idx: torch.Tensor,  # [1, batch * decode_query_len, topk]
    block_table: torch.Tensor,  # [batch, max_blocks]
    seq_lens: torch.Tensor,  # [batch]
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,  # [batch * decode_query_len, num_heads, head_dim]
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    decode_query_len: int = 1,
    sparse_block_table: torch.Tensor | None = None,
    sparse_context_lens: torch.Tensor | None = None,
) -> None:
    if (sparse_block_table is None) != (sparse_context_lens is None):
        raise ValueError(
            "MiniMax-M3 prepared sparse block table and context lengths "
            "must be provided together"
        )
    if sparse_block_table is None:
        sparse_block_table, sparse_context_lens = (
            minimax_m3_build_sparse_block_table_decode(
                topk_idx,
                block_table,
                seq_lens,
                decode_query_len,
                minimax_m3_sparse_block_page_stride(k_cache, v_cache),
            )
        )
    assert sparse_context_lens is not None
    _run_gluon_decode(
        q,
        k_cache,
        v_cache,
        sparse_block_table,
        sparse_context_lens,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
    )


@torch.no_grad()
def minimax_m3_sparse_attn_prefill_aiter(
    q: torch.Tensor,  # [total_q, num_heads, head_dim]
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    topk_idx: torch.Tensor,  # [1, total_q, topk]
    block_table: torch.Tensor,
    query_req_id: torch.Tensor,  # [total_q] int32, owning request per token
    query_abs_pos: torch.Tensor,  # [total_q] int32, absolute position per token
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
) -> None:
    sparse_bt, sparse_ctx = minimax_m3_build_sparse_block_table_prefill(
        topk_idx,
        block_table,
        query_req_id,
        query_abs_pos,
        minimax_m3_sparse_block_page_stride(k_cache, v_cache),
    )
    _run_gluon_decode(
        q,
        k_cache,
        v_cache,
        sparse_bt,
        sparse_ctx,
        num_kv_heads,
        sm_scale,
        output,
        k_scale,
        v_scale,
    )


def _run_gluon_decode(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    sparse_bt: torch.Tensor,
    sparse_ctx: torch.Tensor,
    num_kv_heads: int,
    sm_scale: float,
    output: torch.Tensor,
    k_scale: torch.Tensor | None,
    v_scale: torch.Tensor | None,
) -> None:
    from aiter import dtypes as aiter_dtypes
    from aiter.ops.triton.gluon.pa_decode_gluon import (
        get_recommended_splits,
        pa_decode_gluon,
    )

    if not q.is_contiguous():
        q = q.contiguous()
    if not output.is_contiguous():
        raise ValueError("MiniMax-M3 AITER sparse PA output must be contiguous")

    total_q, num_q_heads, head_size = q.shape
    if head_size != 128:
        raise ValueError("MiniMax-M3 AITER sparse PA requires head_dim == 128")
    group_size = num_q_heads // num_kv_heads

    nphys16, hkv = k_cache.shape[0], k_cache.shape[1]
    # V is offset by half a block when the sides interleave, so it holds fewer
    # pages than K even though both are addressed by the same page id.
    v_nphys16 = v_cache.shape[0]
    k_cache_view = k_cache.view(nphys16 * hkv, 1, *k_cache.shape[2:])
    v_cache_view = v_cache.view(v_nphys16 * hkv, 1, *v_cache.shape[2:])
    q_view = q.view(total_q * num_kv_heads, group_size, head_size)
    out_view = output.view(total_q * num_kv_heads, group_size, head_size)

    num_seqs = total_q * num_kv_heads
    max_context_partition_num = get_recommended_splits(num_seqs, 1)
    context_partition_size = 256
    intermediate_shape = (num_seqs, 1, max_context_partition_num, group_size)
    exp_sums = torch.empty(intermediate_shape, dtype=torch.float32, device=q.device)
    max_logits = torch.empty_like(exp_sums)
    temporary_output = torch.empty(
        *intermediate_shape, head_size, dtype=q.dtype, device=q.device
    )

    is_fp8 = _is_fp8_kv_cache_tensor(k_cache)
    compute_type = aiter_dtypes.fp8 if is_fp8 else q.dtype
    if is_fp8:
        if _sides_are_packed(k_cache, v_cache) and not all(
            scale is None or scale.numel() == 1 for scale in (k_scale, v_scale)
        ):
            raise NotImplementedError(
                "MiniMax-M3 AITER sparse PA supports only scalar KV scales "
                "when K and V pages share a block; per-token scales would "
                "need the same page renumbering."
            )
        k_scale_arg = _gluon_scale_arg(
            k_scale, num_phys_pages=nphys16, num_kv_heads=hkv
        )
        v_scale_arg = _gluon_scale_arg(
            v_scale, num_phys_pages=v_nphys16, num_kv_heads=hkv
        )
    else:
        k_scale_arg = v_scale_arg = None

    pa_decode_gluon(
        output=out_view,
        query=q_view,
        key_cache=k_cache_view,
        value_cache=v_cache_view,
        context_lengths=sparse_ctx,
        block_tables=sparse_bt,
        softmax_scale=sm_scale,
        query_length=1,
        max_context_partition_num=max_context_partition_num,
        context_partition_size=context_partition_size,
        compute_type=compute_type,
        query_scale=None,
        key_scale=k_scale_arg,
        value_scale=v_scale_arg,
        exp_sums=exp_sums,
        max_logits=max_logits,
        temporary_output=temporary_output,
        alibi_slopes=None,
        sinks=None,
        sliding_window=-1,
        ps=True,
    )
