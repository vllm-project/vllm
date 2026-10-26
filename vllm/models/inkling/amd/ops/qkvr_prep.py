# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import aux_stream

LOW_BLOCK_M = 32
LOW_BLOCK_N = 64
LOW_NUM_WARPS = 4
THROUGHPUT_BLOCK_M = 32
THROUGHPUT_BLOCK_N = 128
THROUGHPUT_GROUP_M = 2
THROUGHPUT_NUM_WARPS = 4
SMALL_TOKEN_THRESHOLD = 128
SMALL_NUM_WARPS = 2
Q_BLOCK_ROWS = 8
Q_NUM_WARPS = 2
KV_BLOCK_ROWS = 4
KV_NUM_WARPS = 2


@triton.jit(do_not_specialize=["rows"])
def _rel_proj_low_latency_kernel(
    qkvr_ptr,
    rel_proj_ptr,
    rel_out_ptr,
    log_scaling_ptr,
    rows,
    stride_x_t,
    R_OFFSET: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    APPLY_LOG_SCALING: tl.constexpr,
):
    row = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, 16)
    token = row // NUM_Q_HEADS
    head = row % NUM_Q_HEADS
    relative = tl.load(
        qkvr_ptr
        + token[:, None] * stride_x_t
        + R_OFFSET
        + head[:, None] * 16
        + inner[None, :],
        mask=row[:, None] < rows,
        other=0.0,
    )
    projection = tl.load(
        rel_proj_ptr + inner[:, None] * REL_EXTENT + col[None, :],
        mask=col[None, :] < REL_EXTENT,
        other=0.0,
    )
    values = tl.dot(relative, projection, out_dtype=tl.float32).to(
        rel_out_ptr.dtype.element_ty
    )
    values = values.to(tl.float32)
    if APPLY_LOG_SCALING:
        values *= tl.load(log_scaling_ptr + token, mask=row < rows, other=1.0)[:, None]
    tl.store(
        rel_out_ptr + row[:, None] * REL_EXTENT + col[None, :],
        values.to(rel_out_ptr.dtype.element_ty),
        mask=(row[:, None] < rows) & (col[None, :] < REL_EXTENT),
    )


@triton.jit(do_not_specialize=["rows"])
def _rel_proj_throughput_kernel(
    qkvr_ptr,
    rel_proj_ptr,
    rel_out_ptr,
    log_scaling_ptr,
    rows,
    stride_x_t,
    R_OFFSET: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GROUP_M: tl.constexpr,
    APPLY_LOG_SCALING: tl.constexpr,
):
    row_group = tl.program_id(0)
    col = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    inner = tl.arange(0, 16)
    projection = tl.load(
        rel_proj_ptr + inner[:, None] * REL_EXTENT + col[None, :],
        mask=col[None, :] < REL_EXTENT,
        other=0.0,
    )
    row_offsets = tl.arange(0, BLOCK_M)
    for group_offset in tl.static_range(GROUP_M):
        row = (row_group * GROUP_M + group_offset) * BLOCK_M + row_offsets
        token = row // NUM_Q_HEADS
        head = row % NUM_Q_HEADS
        relative = tl.load(
            qkvr_ptr
            + token[:, None] * stride_x_t
            + R_OFFSET
            + head[:, None] * 16
            + inner[None, :],
            mask=row[:, None] < rows,
            other=0.0,
        )
        values = tl.dot(relative, projection, out_dtype=tl.float32).to(
            rel_out_ptr.dtype.element_ty
        )
        values = values.to(tl.float32)
        if APPLY_LOG_SCALING:
            values *= tl.load(log_scaling_ptr + token, mask=row < rows, other=1.0)[
                :, None
            ]
        tl.store(
            rel_out_ptr + row[:, None] * REL_EXTENT + col[None, :],
            values.to(rel_out_ptr.dtype.element_ty),
            mask=(row[:, None] < rows) & (col[None, :] < REL_EXTENT),
        )


def use_rel_proj_throughput(rows: int, rel_extent: int) -> bool:
    min_rows = 8192 if rel_extent == 512 else 2048
    return rows >= min_rows


def qkvr_rel_proj(
    qkvr: torch.Tensor,
    rel_proj: torch.Tensor,
    rel_out: torch.Tensor,
    log_scaling: torch.Tensor | None,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d_rel: int,
) -> None:
    rows = qkvr.shape[0] * num_q_heads
    rel_extent = rel_proj.shape[1]
    assert d_rel == 16 and rel_proj.shape[0] == 16
    r_offset = num_q_heads * head_dim + 2 * num_kv_heads * head_dim
    log_scaling_ptr = log_scaling if log_scaling is not None else qkvr
    common = dict(
        R_OFFSET=r_offset,
        NUM_Q_HEADS=num_q_heads,
        REL_EXTENT=rel_extent,
        APPLY_LOG_SCALING=log_scaling is not None,
    )

    if use_rel_proj_throughput(rows, rel_extent):
        grid = (
            triton.cdiv(rows, THROUGHPUT_BLOCK_M * THROUGHPUT_GROUP_M),
            triton.cdiv(rel_extent, THROUGHPUT_BLOCK_N),
        )
        _rel_proj_throughput_kernel[grid](
            qkvr,
            rel_proj,
            rel_out,
            log_scaling_ptr,
            rows,
            qkvr.stride(0),
            BLOCK_M=THROUGHPUT_BLOCK_M,
            BLOCK_N=THROUGHPUT_BLOCK_N,
            GROUP_M=THROUGHPUT_GROUP_M,
            num_warps=THROUGHPUT_NUM_WARPS,
            **common,
        )
        return

    grid = (
        triton.cdiv(rows, LOW_BLOCK_M),
        triton.cdiv(rel_extent, LOW_BLOCK_N),
    )
    _rel_proj_low_latency_kernel[grid](
        qkvr,
        rel_proj,
        rel_out,
        log_scaling_ptr,
        rows,
        qkvr.stride(0),
        BLOCK_M=LOW_BLOCK_M,
        BLOCK_N=LOW_BLOCK_N,
        num_warps=LOW_NUM_WARPS,
        **common,
    )


@triton.jit(do_not_specialize=["tokens", "stride_block_table_req", "max_blocks"])
def _qkvr_qkv_kernel(
    qkvr_ptr,
    q_norm_weight_ptr,
    q_out_ptr,
    rel_proj_ptr,
    rel_out_ptr,
    k_weight_ptr,
    v_weight_ptr,
    k_norm_weight_ptr,
    conv_cache_ptr,
    key_cache_ptr,
    value_cache_ptr,
    positions_ptr,
    seq_idx_ptr,
    conv_slot_mapping_ptr,
    conv_block_table_ptr,
    query_start_ptr,
    attention_slot_mapping_ptr,
    log_scaling_ptr,
    tokens,
    eps,
    stride_x_t,
    stride_cc_block,
    stride_cc_head,
    stride_cc_token,
    stride_cc_dim,
    stride_kc_block,
    stride_kc_token,
    stride_kc_head,
    stride_vc_block,
    stride_vc_token,
    stride_vc_head,
    stride_block_table_req,
    max_blocks,
    conv_block_size,
    attention_page_size,
    Q_WIDTH: tl.constexpr,
    KV_WIDTH: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    OFF_K: tl.constexpr,
    OFF_V: tl.constexpr,
    APPLY_LOG_SCALING: tl.constexpr,
    D_REL: tl.constexpr,
    REL_EXTENT: tl.constexpr,
    REL_EXTENT_PADDED: tl.constexpr,
):
    block = tl.program_id(0)
    num_q_rows = tokens * NUM_Q_HEADS
    dims = tl.arange(0, HEAD_DIM)

    if block < num_q_rows:
        row = block
        token = row // NUM_Q_HEADS
        head = row % NUM_Q_HEADS
        values = tl.load(
            qkvr_ptr + token * stride_x_t + head * HEAD_DIM + dims,
        ).to(tl.float32)
        weight = tl.load(q_norm_weight_ptr + dims).to(tl.float32)
        rstd = tl.rsqrt(tl.sum(values * values, axis=0) / HEAD_DIM + eps)
        normalized = values * rstd * weight
        if APPLY_LOG_SCALING:
            normalized = normalized.to(q_out_ptr.dtype.element_ty).to(tl.float32)
            normalized *= tl.load(log_scaling_ptr + token)
        tl.store(
            q_out_ptr + row * HEAD_DIM + dims,
            normalized.to(q_out_ptr.dtype.element_ty),
        )
        rel_cols = tl.arange(0, REL_EXTENT_PADDED)
        rel_mask = rel_cols < REL_EXTENT
        projected = tl.zeros([REL_EXTENT_PADDED], dtype=tl.float32)
        rel_offset = Q_WIDTH + 2 * KV_WIDTH + head * D_REL
        for rel_dim in tl.static_range(D_REL):
            rel_value = tl.load(
                qkvr_ptr + token * stride_x_t + rel_offset + rel_dim
            ).to(tl.float32)
            proj = tl.load(
                rel_proj_ptr + rel_dim * REL_EXTENT + rel_cols,
                mask=rel_mask,
                other=0.0,
            ).to(tl.float32)
            projected += rel_value * proj
        projected = projected.to(rel_out_ptr.dtype.element_ty).to(tl.float32)
        if APPLY_LOG_SCALING:
            projected *= tl.load(log_scaling_ptr + token)
        tl.store(
            rel_out_ptr + row * REL_EXTENT + rel_cols,
            projected.to(rel_out_ptr.dtype.element_ty),
            mask=rel_mask,
        )
    else:
        row = block - num_q_rows
        if row < tokens * NUM_KV_HEADS:
            token = row // NUM_KV_HEADS
            head = row % NUM_KV_HEADS
            position = tl.load(positions_ptr + token)
            request = tl.load(seq_idx_ptr + token)
            conv_slot = tl.load(conv_slot_mapping_ptr + token)
            query_start = tl.load(query_start_ptr + token)
            attention_slot = tl.load(attention_slot_mapping_ptr + token)
            valid = conv_slot >= 0

            k_col = Q_WIDTH + head * HEAD_DIM
            v_col = Q_WIDTH + KV_WIDTH + head * HEAD_DIM
            k_value = tl.load(qkvr_ptr + token * stride_x_t + k_col + dims)
            v_value = tl.load(qkvr_ptr + token * stride_x_t + v_col + dims)

            safe_slot = tl.maximum(conv_slot, 0)
            cache_base = (
                conv_cache_ptr
                + (safe_slot // conv_block_size) * stride_cc_block
                + head * stride_cc_head
                + (safe_slot % conv_block_size) * stride_cc_token
            )
            tl.store(
                cache_base + (OFF_K + dims) * stride_cc_dim,
                k_value,
                mask=valid,
            )
            tl.store(
                cache_base + (OFF_V + dims) * stride_cc_dim,
                v_value,
                mask=valid,
            )

            acc_k = tl.zeros([HEAD_DIM], dtype=tl.float32)
            acc_v = tl.zeros([HEAD_DIM], dtype=tl.float32)
            for tap in tl.static_range(WINDOW_SIZE):
                source_position = position - (WINDOW_SIZE - 1) + tap
                source_row = token - (WINDOW_SIZE - 1) + tap
                in_window = valid & (source_position >= 0)
                intra = in_window & (source_row >= query_start)
                cached = in_window & (source_row < query_start)
                safe_row = tl.maximum(source_row, 0)
                source_k = tl.load(
                    qkvr_ptr + safe_row * stride_x_t + k_col + dims,
                    mask=intra,
                    other=0.0,
                ).to(tl.float32)
                source_v = tl.load(
                    qkvr_ptr + safe_row * stride_x_t + v_col + dims,
                    mask=intra,
                    other=0.0,
                ).to(tl.float32)
                safe_position = tl.maximum(source_position, 0)
                logical_block = tl.minimum(
                    safe_position // conv_block_size, max_blocks - 1
                )
                physical_block = tl.load(
                    conv_block_table_ptr
                    + request * stride_block_table_req
                    + logical_block,
                    mask=cached,
                    other=0,
                ).to(tl.int64)
                tap_base = (
                    conv_cache_ptr
                    + physical_block * stride_cc_block
                    + head * stride_cc_head
                    + (safe_position % conv_block_size) * stride_cc_token
                )
                source_k += tl.load(
                    tap_base + (OFF_K + dims) * stride_cc_dim,
                    mask=cached,
                    other=0.0,
                ).to(tl.float32)
                source_v += tl.load(
                    tap_base + (OFF_V + dims) * stride_cc_dim,
                    mask=cached,
                    other=0.0,
                ).to(tl.float32)
                k_weight = tl.load(
                    k_weight_ptr + (head * HEAD_DIM + dims) * WINDOW_SIZE + tap
                ).to(tl.float32)
                v_weight = tl.load(
                    v_weight_ptr + (head * HEAD_DIM + dims) * WINDOW_SIZE + tap
                ).to(tl.float32)
                acc_k += source_k * k_weight
                acc_v += source_v * v_weight

            k_rounded = (acc_k + k_value.to(tl.float32)).to(qkvr_ptr.dtype.element_ty)
            v_rounded = (acc_v + v_value.to(tl.float32)).to(qkvr_ptr.dtype.element_ty)
            k_float = k_rounded.to(tl.float32)
            k_norm_weight = tl.load(k_norm_weight_ptr + dims).to(tl.float32)
            rstd = tl.rsqrt(tl.sum(k_float * k_float, axis=0) / HEAD_DIM + eps)
            k_normalized = (k_float * rstd * k_norm_weight).to(
                qkvr_ptr.dtype.element_ty
            )

            safe_attention_slot = tl.maximum(attention_slot, 0)
            attention_block = safe_attention_slot // attention_page_size
            attention_offset = safe_attention_slot % attention_page_size
            tl.store(
                key_cache_ptr
                + attention_block * stride_kc_block
                + attention_offset * stride_kc_token
                + head * stride_kc_head
                + dims,
                k_normalized,
                mask=attention_slot >= 0,
            )
            tl.store(
                value_cache_ptr
                + attention_block * stride_vc_block
                + attention_offset * stride_vc_token
                + head * stride_vc_head
                + dims,
                v_rounded,
                mask=attention_slot >= 0,
            )


@triton.jit(do_not_specialize=["num_rows"])
def _q_kernel(
    qkvr_ptr,
    q_norm_weight_ptr,
    q_out_ptr,
    log_scaling_ptr,
    num_rows,
    stride_x_t,
    eps,
    NUM_Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
    APPLY_LOG_SCALING: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dims = tl.arange(0, HEAD_DIM)
    row_mask = rows < num_rows
    tokens = rows // NUM_Q_HEADS
    heads = rows % NUM_Q_HEADS
    values = tl.load(
        qkvr_ptr
        + tokens[:, None] * stride_x_t
        + heads[:, None] * HEAD_DIM
        + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    ).to(tl.float32)
    weight = tl.load(q_norm_weight_ptr + dims).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(values * values, axis=1) / HEAD_DIM + eps)
    normalized = values * rstd[:, None] * weight[None, :]
    if APPLY_LOG_SCALING:
        normalized = normalized.to(q_out_ptr.dtype.element_ty).to(tl.float32)
        tau = tl.load(log_scaling_ptr + tokens, mask=row_mask, other=1.0)
        normalized *= tau[:, None]
    tl.store(
        q_out_ptr + rows[:, None] * HEAD_DIM + dims[None, :],
        normalized.to(q_out_ptr.dtype.element_ty),
        mask=row_mask[:, None],
    )


@triton.jit(do_not_specialize=["tokens", "stride_block_table_req", "max_blocks"])
def _kv_kernel(
    qkvr_ptr,
    k_weight_ptr,
    v_weight_ptr,
    k_norm_weight_ptr,
    conv_cache_ptr,
    key_cache_ptr,
    value_cache_ptr,
    positions_ptr,
    seq_idx_ptr,
    conv_slot_mapping_ptr,
    conv_block_table_ptr,
    query_start_ptr,
    attention_slot_mapping_ptr,
    tokens,
    eps,
    stride_x_t,
    stride_cc_block,
    stride_cc_head,
    stride_cc_token,
    stride_cc_dim,
    stride_kc_block,
    stride_kc_token,
    stride_kc_head,
    stride_vc_block,
    stride_vc_token,
    stride_vc_head,
    stride_block_table_req,
    max_blocks,
    conv_block_size,
    attention_page_size,
    Q_WIDTH: tl.constexpr,
    KV_WIDTH: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    OFF_K: tl.constexpr,
    OFF_V: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    dims = tl.arange(0, HEAD_DIM)
    row_mask = token < tokens
    head_id = tl.program_id(1)
    head = tl.full([BLOCK_ROWS], head_id, tl.int64)
    position = tl.load(positions_ptr + token, mask=row_mask, other=0)
    request = tl.load(seq_idx_ptr + token, mask=row_mask, other=0)
    conv_slot = tl.load(conv_slot_mapping_ptr + token, mask=row_mask, other=-1)
    query_start = tl.load(query_start_ptr + token, mask=row_mask, other=0)
    attention_slot = tl.load(
        attention_slot_mapping_ptr + token, mask=row_mask, other=-1
    )
    valid = row_mask & (conv_slot >= 0)

    k_col = Q_WIDTH + head * HEAD_DIM
    v_col = Q_WIDTH + KV_WIDTH + head * HEAD_DIM
    k_value = tl.load(
        qkvr_ptr + token[:, None] * stride_x_t + k_col[:, None] + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    v_value = tl.load(
        qkvr_ptr + token[:, None] * stride_x_t + v_col[:, None] + dims[None, :],
        mask=row_mask[:, None],
        other=0.0,
    )
    safe_slot = tl.maximum(conv_slot, 0)
    cache_base = (
        conv_cache_ptr
        + (safe_slot // conv_block_size) * stride_cc_block
        + head * stride_cc_head
        + (safe_slot % conv_block_size) * stride_cc_token
    )
    tl.store(
        cache_base[:, None] + (OFF_K + dims[None, :]) * stride_cc_dim,
        k_value,
        mask=valid[:, None],
    )
    tl.store(
        cache_base[:, None] + (OFF_V + dims[None, :]) * stride_cc_dim,
        v_value,
        mask=valid[:, None],
    )

    acc_k = tl.zeros([BLOCK_ROWS, HEAD_DIM], dtype=tl.float32)
    acc_v = tl.zeros([BLOCK_ROWS, HEAD_DIM], dtype=tl.float32)
    for tap in tl.static_range(WINDOW_SIZE):
        source_position = position - (WINDOW_SIZE - 1) + tap
        source_row = token - (WINDOW_SIZE - 1) + tap
        in_window = valid & (source_position >= 0)
        intra = in_window & (source_row >= query_start)
        cached = in_window & (source_row < query_start)
        safe_row = tl.maximum(source_row, 0)
        source_k = tl.load(
            qkvr_ptr + safe_row[:, None] * stride_x_t + k_col[:, None] + dims[None, :],
            mask=intra[:, None],
            other=0.0,
        ).to(tl.float32)
        source_v = tl.load(
            qkvr_ptr + safe_row[:, None] * stride_x_t + v_col[:, None] + dims[None, :],
            mask=intra[:, None],
            other=0.0,
        ).to(tl.float32)
        safe_position = tl.maximum(source_position, 0)
        logical_block = tl.minimum(safe_position // conv_block_size, max_blocks - 1)
        physical_block = tl.load(
            conv_block_table_ptr + request * stride_block_table_req + logical_block,
            mask=cached,
            other=0,
        ).to(tl.int64)
        tap_base = (
            conv_cache_ptr
            + physical_block * stride_cc_block
            + head * stride_cc_head
            + (safe_position % conv_block_size) * stride_cc_token
        )
        source_k += tl.load(
            tap_base[:, None] + (OFF_K + dims[None, :]) * stride_cc_dim,
            mask=cached[:, None],
            other=0.0,
        ).to(tl.float32)
        source_v += tl.load(
            tap_base[:, None] + (OFF_V + dims[None, :]) * stride_cc_dim,
            mask=cached[:, None],
            other=0.0,
        ).to(tl.float32)
        k_weight = tl.load(
            k_weight_ptr + (head_id * HEAD_DIM + dims) * WINDOW_SIZE + tap
        ).to(tl.float32)
        v_weight = tl.load(
            v_weight_ptr + (head_id * HEAD_DIM + dims) * WINDOW_SIZE + tap
        ).to(tl.float32)
        acc_k += source_k * k_weight[None, :]
        acc_v += source_v * v_weight[None, :]

    k_rounded = (acc_k + k_value.to(tl.float32)).to(qkvr_ptr.dtype.element_ty)
    v_rounded = (acc_v + v_value.to(tl.float32)).to(qkvr_ptr.dtype.element_ty)
    k_float = k_rounded.to(tl.float32)
    k_norm_weight = tl.load(k_norm_weight_ptr + dims).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(k_float * k_float, axis=1) / HEAD_DIM + eps)
    k_normalized = (k_float * rstd[:, None] * k_norm_weight[None, :]).to(
        qkvr_ptr.dtype.element_ty
    )

    safe_attention_slot = tl.maximum(attention_slot, 0)
    attention_block = safe_attention_slot // attention_page_size
    attention_offset = safe_attention_slot % attention_page_size
    attention_mask = row_mask & (attention_slot >= 0)
    tl.store(
        key_cache_ptr
        + attention_block[:, None] * stride_kc_block
        + attention_offset[:, None] * stride_kc_token
        + head[:, None] * stride_kc_head
        + dims[None, :],
        k_normalized,
        mask=attention_mask[:, None],
    )
    tl.store(
        value_cache_ptr
        + attention_block[:, None] * stride_vc_block
        + attention_offset[:, None] * stride_vc_token
        + head[:, None] * stride_vc_head
        + dims[None, :],
        v_rounded,
        mask=attention_mask[:, None],
    )


def _run_tiled_q(
    qkvr: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_out: torch.Tensor,
    positions: torch.Tensor,
    *,
    eps: float,
    num_q_heads: int,
    head_dim: int,
    log_scaling: torch.Tensor | None,
) -> None:
    num_rows = qkvr.shape[0] * num_q_heads
    _q_kernel[(triton.cdiv(num_rows, Q_BLOCK_ROWS),)](
        qkvr,
        q_norm_weight,
        q_out,
        log_scaling if log_scaling is not None else positions,
        num_rows,
        qkvr.stride(0),
        eps,
        NUM_Q_HEADS=num_q_heads,
        HEAD_DIM=head_dim,
        BLOCK_ROWS=Q_BLOCK_ROWS,
        APPLY_LOG_SCALING=log_scaling is not None,
        num_warps=Q_NUM_WARPS,
    )


def _run_tiled_kv(
    qkvr: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    conv_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    positions: torch.Tensor,
    seq_idx: torch.Tensor,
    conv_slot_mapping: torch.Tensor,
    conv_block_table: torch.Tensor,
    query_start: torch.Tensor,
    attention_slot_mapping: torch.Tensor,
    *,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    off_k: int,
    off_v: int,
    conv_block_size: int,
) -> None:
    tokens = qkvr.shape[0]
    _kv_kernel[(triton.cdiv(tokens, KV_BLOCK_ROWS), num_kv_heads)](
        qkvr,
        k_weight,
        v_weight,
        k_norm_weight,
        conv_cache,
        key_cache,
        value_cache,
        positions,
        seq_idx,
        conv_slot_mapping,
        conv_block_table,
        query_start,
        attention_slot_mapping,
        tokens,
        eps,
        qkvr.stride(0),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        conv_cache.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        conv_block_table.stride(0),
        conv_block_table.shape[1],
        conv_block_size,
        key_cache.shape[1],
        Q_WIDTH=num_q_heads * head_dim,
        KV_WIDTH=num_kv_heads * head_dim,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        WINDOW_SIZE=k_weight.shape[1],
        OFF_K=off_k,
        OFF_V=off_v,
        BLOCK_ROWS=KV_BLOCK_ROWS,
        num_warps=KV_NUM_WARPS,
    )


def _run_fused_small(
    qkvr: torch.Tensor,
    q_norm_weight: torch.Tensor,
    q_out: torch.Tensor,
    rel_proj: torch.Tensor,
    rel_out: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    conv_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    positions: torch.Tensor,
    seq_idx: torch.Tensor,
    conv_slot_mapping: torch.Tensor,
    conv_block_table: torch.Tensor,
    query_start: torch.Tensor,
    attention_slot_mapping: torch.Tensor,
    *,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    off_k: int,
    off_v: int,
    conv_block_size: int,
    log_scaling: torch.Tensor | None,
) -> None:
    tokens = qkvr.shape[0]

    num_q_rows = tokens * num_q_heads
    grid = (num_q_rows + tokens * num_kv_heads,)
    _qkvr_qkv_kernel[grid](
        qkvr,
        q_norm_weight,
        q_out,
        rel_proj,
        rel_out,
        k_weight,
        v_weight,
        k_norm_weight,
        conv_cache,
        key_cache,
        value_cache,
        positions,
        seq_idx,
        conv_slot_mapping,
        conv_block_table,
        query_start,
        attention_slot_mapping,
        log_scaling if log_scaling is not None else positions,
        tokens,
        eps,
        qkvr.stride(0),
        conv_cache.stride(0),
        conv_cache.stride(1),
        conv_cache.stride(2),
        conv_cache.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        conv_block_table.stride(0),
        conv_block_table.shape[1],
        conv_block_size,
        key_cache.shape[1],
        Q_WIDTH=num_q_heads * head_dim,
        KV_WIDTH=num_kv_heads * head_dim,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        WINDOW_SIZE=k_weight.shape[1],
        OFF_K=off_k,
        OFF_V=off_v,
        APPLY_LOG_SCALING=log_scaling is not None,
        D_REL=16,
        REL_EXTENT=rel_proj.shape[1],
        REL_EXTENT_PADDED=triton.next_power_of_2(rel_proj.shape[1]),
        num_warps=SMALL_NUM_WARPS,
    )


def fused_qkvr_prep(
    qkvr: torch.Tensor,
    k_weight: torch.Tensor,
    v_weight: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    rel_proj: torch.Tensor,
    eps: float,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    d_rel: int,
    conv_cache: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    positions: torch.Tensor,
    conv_block_table: torch.Tensor,
    seq_idx: torch.Tensor,
    conv_slot_mapping: torch.Tensor,
    query_start: torch.Tensor,
    attention_slot_mapping: torch.Tensor,
    off_k: int,
    off_v: int,
    conv_block_size: int,
    log_scaling: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert d_rel == 16 and rel_proj.shape[0] == 16
    assert head_dim == 128
    assert qkvr.is_contiguous()
    assert k_weight.stride() == (k_weight.shape[1], 1)
    assert v_weight.stride() == (v_weight.shape[1], 1)
    assert rel_proj.stride() == (rel_proj.shape[1], 1)
    assert conv_cache.stride(3) == 1
    assert key_cache.stride(3) == 1 and value_cache.stride(3) == 1
    tokens = qkvr.shape[0]
    q_out = torch.empty(
        (tokens, num_q_heads * head_dim), dtype=qkvr.dtype, device=qkvr.device
    )
    rel_out = torch.empty(
        (tokens, num_q_heads, rel_proj.shape[1]),
        dtype=qkvr.dtype,
        device=qkvr.device,
    )
    if tokens == 0:
        return q_out, rel_out

    if tokens < SMALL_TOKEN_THRESHOLD:
        _run_fused_small(
            qkvr,
            q_norm_weight,
            q_out,
            rel_proj,
            rel_out,
            k_weight,
            v_weight,
            k_norm_weight,
            conv_cache,
            key_cache,
            value_cache,
            positions,
            seq_idx,
            conv_slot_mapping,
            conv_block_table,
            query_start,
            attention_slot_mapping,
            eps=eps,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            off_k=off_k,
            off_v=off_v,
            conv_block_size=conv_block_size,
            log_scaling=log_scaling,
        )
        return q_out, rel_out

    kv_stream = aux_stream()
    assert kv_stream is not None
    current_stream = torch.cuda.current_stream()
    kv_stream.wait_stream(current_stream)
    with torch.cuda.stream(kv_stream):
        _run_tiled_kv(
            qkvr,
            k_weight,
            v_weight,
            k_norm_weight,
            conv_cache,
            key_cache,
            value_cache,
            positions,
            seq_idx,
            conv_slot_mapping,
            conv_block_table,
            query_start,
            attention_slot_mapping,
            eps=eps,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            off_k=off_k,
            off_v=off_v,
            conv_block_size=conv_block_size,
        )
    _run_tiled_q(
        qkvr,
        q_norm_weight,
        q_out,
        positions,
        eps=eps,
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        log_scaling=log_scaling,
    )
    qkvr_rel_proj(
        qkvr,
        rel_proj,
        rel_out,
        log_scaling,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        d_rel=d_rel,
    )
    current_stream.wait_stream(kv_stream)
    return q_out, rel_out
