# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU operators for the Qwen4Exp weight-free QSA path."""

from __future__ import annotations

import math

import torch

from vllm.triton_utils import tl, triton

from ..runtime import has_active_triton_cpu_backend

_MAX_GRID_AXIS = 65_535


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
        + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE,
        other=0.0,
    )

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + columns,
            mask=columns < TOPK,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < PAGE_TABLE_WIDTH)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_req
            + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr
            + safe_page[None, :] * stride_k_block
            + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head
            + dim_offsets[:, None],
            mask=valid[None, :],
            other=0.0,
        )
        values = tl.load(
            v_cache_ptr
            + safe_page[:, None] * stride_v_block
            + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head
            + dim_offsets[None, :],
            mask=valid[:, None],
            other=0.0,
        )
        scores = tl.dot(query, keys) * softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(
            valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0
        )
        accumulator = tl.dot(
            probabilities.to(values.dtype),
            values,
            acc=accumulator * alpha[:, None],
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        partial_offset = (
            (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets[:, None]
        ).to(tl.int64)
        tl.store(
            partial_output_ptr + partial_offset * HEAD_DIM + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        lse_offset = (
            (split_id * num_rows + row) * NUM_QUERY_HEADS + first_head + head_offsets
        ).to(tl.int64)
        tl.store(
            partial_lse_ptr + lse_offset,
            partial_lse,
            mask=head_offsets < GROUP_SIZE,
        )


@triton.jit
def _qsa_merge_splitk_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_output_row,
    stride_output_head,
    num_rows,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    head = tl.program_id(1)
    split_offsets = tl.arange(0, BLOCK_SPLITS)
    dim_offsets = tl.arange(0, HEAD_DIM)
    split_mask = split_offsets < NUM_SPLITS
    partial_rows = (split_offsets.to(tl.int64) * num_rows + row) * NUM_QUERY_HEADS
    lse = tl.load(
        partial_lse_ptr + partial_rows + head,
        mask=split_mask,
        other=-float("inf"),
    )
    lse_max = tl.max(lse, axis=0)
    has_values = lse_max > -float("inf")
    shifted = tl.where(split_mask & has_values, lse - lse_max, -float("inf"))
    weights = tl.math.exp2(shifted)
    denominator = tl.sum(weights, axis=0)
    partial_output = tl.load(
        partial_output_ptr
        + (partial_rows[:, None] + head) * HEAD_DIM
        + dim_offsets[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    merged = tl.sum(partial_output * weights[:, None], axis=0)
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged,
    )


def _validate_scoring_inputs(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
) -> None:
    if q.ndim != 3 or q.shape[1] <= 0 or q.shape[2] <= 0:
        raise ValueError("QSA query must be [rows, heads, head_dim]")
    if k_cache.ndim != 4 or k_cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, head_dim]")
    if k_cache.shape[3] != q.shape[2]:
        raise ValueError("QSA query and cache dimensions must match")
    if page_table.ndim != 2:
        raise ValueError("QSA page table must be two-dimensional")
    if q.shape[0] and (not all(k_cache.shape[:2]) or not all(page_table.shape)):
        raise ValueError("QSA paged scoring cache and page table must be nonempty")
    if token_to_req.shape != (q.shape[0],):
        raise ValueError("QSA request mapping must match query rows")
    if query_positions.shape != (q.shape[0],):
        raise ValueError("QSA query positions must match query rows")
    if sequence_lengths.shape != (page_table.shape[0],):
        raise ValueError("QSA sequence lengths must match page-table requests")
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")


def qsa_mqa_paged(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    num_columns: int | None = None,
    score_scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute QSA scores from a paged compressed-key cache with Torch."""

    _validate_scoring_inputs(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )
    if q.device.type != "cpu":
        raise RuntimeError("CPU QSA scoring requires CPU tensors")
    capacity = page_table.shape[1] * k_cache.shape[1]
    columns = capacity if num_columns is None else num_columns
    if columns < 0 or columns > capacity:
        raise ValueError("QSA score width must be within cache capacity")
    divisor = math.sqrt(q.shape[2]) if score_scale is None else score_scale
    if divisor <= 0:
        raise ValueError("QSA score scale must be positive")

    logits = torch.full(
        (q.shape[0], columns),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    visible_blocks = torch.zeros(q.shape[0], dtype=torch.int32, device=q.device)
    for row in range(q.shape[0]):
        request = int(token_to_req[row])
        if request < 0 or request >= page_table.shape[0]:
            continue
        visible = min(
            int((query_positions[row] + 1) // compress_ratio),
            int(sequence_lengths[request] // compress_ratio),
            columns,
        )
        visible = max(visible, 0)
        visible_blocks[row] = visible
        if visible == 0:
            continue
        logical = torch.arange(visible, device=q.device)
        pages = page_table[request, logical // k_cache.shape[1]].long()
        if torch.any((pages < 0) | (pages >= k_cache.shape[0])):
            raise ValueError("QSA page table contains an invalid physical page")
        offsets = logical % k_cache.shape[1]
        keys = k_cache[pages, offsets, 0]
        scores = torch.einsum("hd,kd->hk", q[row].float(), keys.float())
        logits[row, :visible] = scores.clamp_min_(0).sum(dim=0) / divisor
    return logits, visible_blocks


def expand_qsa_block_indices(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    compress_ratio: int,
    token_topk: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Expand compressed blocks and append the causal open-group tail."""

    if block_indices.device.type != "cpu":
        raise RuntimeError("CPU QSA index expansion requires CPU tensors")
    if compress_ratio <= 0 or token_topk % compress_ratio:
        raise ValueError(
            "QSA token top-k must be divisible by a positive compression ratio"
        )
    block_topk = token_topk // compress_ratio
    if block_indices.shape != (query_positions.numel(), block_topk):
        raise ValueError("QSA compressed top-k has an invalid shape")
    if token_to_req.shape != query_positions.shape:
        raise ValueError("QSA request mapping must match query positions")
    if sequence_lengths.ndim != 1 or not sequence_lengths.shape[0]:
        raise ValueError("QSA request sequence lengths must be nonempty")
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty(
            (block_indices.shape[0], output_width),
            dtype=torch.int32,
            device=block_indices.device,
        )
    elif out.shape != (block_indices.shape[0], output_width):
        raise ValueError("QSA expansion output has an invalid shape")
    out.fill_(-1)

    offsets = torch.arange(compress_ratio, device=block_indices.device)
    for row in range(block_indices.shape[0]):
        request = int(token_to_req[row])
        if request < 0 or request >= sequence_lengths.shape[0]:
            continue
        position = int(query_positions[row])
        sequence_length = int(sequence_lengths[request])
        selected = block_indices[row][block_indices[row] >= 0].long()
        expanded = (selected[:, None] * compress_ratio + offsets).flatten()
        tail_start = ((position + 1) // compress_ratio) * compress_ratio
        tail = torch.arange(
            tail_start,
            min(position + 1, sequence_length),
            device=block_indices.device,
        )
        tokens = torch.cat((expanded, tail))
        tokens = tokens[(tokens >= 0) & (tokens < sequence_length)]
        count = min(tokens.numel(), output_width)
        out[row, :count] = tokens[:count].to(torch.int32)
    return out


def qsa_select_paged_tokens(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_topk: int,
    compress_ratio: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Score, top-k select, and expand QSA indices with Torch."""

    if token_topk <= 0 or token_topk % compress_ratio:
        raise ValueError("QSA token top-k must be positive and divisible by ratio")
    rows = q.shape[0]
    output_width = token_topk + compress_ratio - 1
    if out is None:
        out = torch.empty((rows, output_width), dtype=torch.int32, device=q.device)
    if out.shape != (rows, output_width):
        raise ValueError("QSA selection output has an invalid shape")
    if not rows:
        return out

    logits, visible_blocks = qsa_mqa_paged(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        compress_ratio,
    )
    block_topk = token_topk // compress_ratio
    selected = torch.full((rows, block_topk), -1, dtype=torch.int32, device=q.device)
    for row in range(rows):
        count = min(int(visible_blocks[row]), block_topk)
        if count:
            selected[row, :count] = torch.topk(
                logits[row], count, sorted=False
            ).indices.to(torch.int32)
    return expand_qsa_block_indices(
        selected,
        query_positions,
        sequence_lengths,
        token_to_req,
        compress_ratio,
        token_topk,
        out,
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run sparse GQA over paged BF16 CPU K/V caches with Triton."""

    if q.device.type != "cpu":
        raise RuntimeError("paged CPU QSA requires Triton and CPU tensors")
    if not has_active_triton_cpu_backend():
        raise RuntimeError("paged CPU QSA requires an active Triton CPU backend")
    if q.ndim != 3 or k_cache.ndim != 4 or v_cache.shape != k_cache.shape:
        raise ValueError("QSA sparse attention received invalid Q/K/V shapes")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != q.shape[0]:
        raise ValueError("QSA indices must have one row per query")
    if token_to_req.shape != (q.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA sparse attention metadata has invalid shapes")
    if not all(k_cache.shape[:3]) or not all(block_table.shape):
        raise ValueError("QSA sparse attention cache and block table must be nonempty")
    if logical_indices.shape[1] <= 0:
        raise ValueError("QSA sparse attention requires a positive selection width")
    if q.shape[2] != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    if head_dim < 16 or head_dim & (head_dim - 1):
        raise ValueError("QSA sparse attention head size must be a power of two")
    if (
        q.dtype != torch.bfloat16
        or k_cache.dtype != q.dtype
        or v_cache.dtype != q.dtype
    ):
        raise ValueError("QSA sparse attention requires BF16 Q/K/V")
    if logical_indices.dtype != torch.int32 or block_table.dtype != torch.int32:
        raise ValueError("QSA indices and block table must be int32")
    if token_to_req.dtype != torch.int32:
        raise ValueError("QSA request mapping must be int32")
    tensors = (k_cache, v_cache, logical_indices, block_table, token_to_req)
    if any(tensor.device != q.device for tensor in tensors):
        raise ValueError("QSA sparse attention tensors must share one device")
    if q.stride(2) != 1 or k_cache.stride(3) != 1 or v_cache.stride(3) != 1:
        raise ValueError("QSA Q/K/V head dimensions must be contiguous")
    if logical_indices.stride(1) != 1 or block_table.stride(1) != 1:
        raise ValueError("QSA index and block-table rows must be contiguous")
    if token_to_req.stride(0) != 1:
        raise ValueError("QSA request mapping must be contiguous")
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape or out.dtype != q.dtype or out.device != q.device:
        raise ValueError("QSA sparse output must match its query")
    if out.stride(2) != 1:
        raise ValueError("QSA sparse output head dimension must be contiguous")
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    block_n = 16
    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, 64)
    if k_cache.shape[2] > _MAX_GRID_AXIS or num_splits > _MAX_GRID_AXIS:
        raise ValueError("QSA sparse attention exceeds Triton grid bounds")
    if q.shape[1] > _MAX_GRID_AXIS:
        raise ValueError("QSA sparse merge exceeds Triton grid bounds")

    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    _qsa_sparse_paged_gqa_splitk_kernel[(q.shape[0], k_cache.shape[2], num_splits)](
        q,
        k_cache,
        v_cache,
        logical_indices,
        block_table,
        token_to_req,
        partial_output,
        partial_lse,
        out,
        q.stride(0),
        q.stride(1),
        k_cache.stride(0),
        k_cache.stride(1),
        k_cache.stride(2),
        v_cache.stride(0),
        v_cache.stride(1),
        v_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        out.stride(0),
        out.stride(1),
        q.shape[0],
        k_cache.shape[0],
        block_table.shape[0],
        TOPK=logical_indices.shape[1],
        PAGE_SIZE=k_cache.shape[1],
        PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        num_cpu_threads=0,
    )
    if num_splits == 1:
        return out

    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output,
        partial_lse,
        out,
        out.stride(0),
        out.stride(1),
        q.shape[0],
        HEAD_DIM=q.shape[2],
        NUM_QUERY_HEADS=q.shape[1],
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_cpu_threads=0,
    )
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a CPU QSA cache with Torch."""

    if cache.device.type != "cpu":
        raise RuntimeError("CPU QSA cache stores require CPU tensors")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    valid = (slot_mapping >= 0) & (slot_mapping < cache.shape[0] * cache.shape[1])
    if not torch.any(valid):
        return
    slots = slot_mapping[valid].long()
    cache[slots // cache.shape[1], slots % cache.shape[1], 0] = rows[valid]


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,
    raw_positions: torch.Tensor,
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from CPU compressor state and current rows."""

    if raw_keys.device.type != "cpu":
        raise RuntimeError("CPU QSA compression requires CPU tensors")
    rows = token_to_req.numel()
    if compress_ratio <= 0:
        raise ValueError("QSA compression ratio must be positive")
    if raw_keys.ndim != 3 or raw_keys.shape[:2] != (rows, 1):
        raise ValueError("QSA raw keys must be [rows, 1, head_size]")
    if raw_positions.shape != (rows, 1, 3) or raw_positions.dtype != torch.int64:
        raise ValueError("QSA raw positions must be [rows, 1, 3] int64")
    if logical_positions.shape != (rows,) or compressed_slots.shape != (rows,):
        raise ValueError("QSA compression metadata must match token rows")
    if compressor_state_cache.ndim != 4 or compressor_state_cache.shape[2] != 1:
        raise ValueError("QSA compressor-state cache has an invalid shape")
    if (
        compressor_state_cache.shape[1] < compress_ratio
        or compressor_state_cache.shape[3] != raw_keys.shape[2]
        or compressor_state_cache.dtype != raw_keys.dtype
    ):
        raise ValueError(
            "QSA compressor-state cache does not match the compression layout"
        )
    if (
        compressor_state_block_table.ndim != 2
        or compressor_state_block_table.shape[1] < 1
    ):
        raise ValueError(
            "QSA compressor-state block table must contain one block per request"
        )
    if query_start_loc.ndim != 1 or query_start_loc.shape[0] < 2:
        raise ValueError("QSA query starts must contain a terminal offset")
    num_requests = query_start_loc.shape[0] - 1
    if compressor_state_block_table.shape[0] < num_requests:
        raise ValueError("QSA compressor-state block table has too few request rows")
    if rope_cache is not None:
        raise NotImplementedError("CPU QSA compression does not support MRoPE")

    pooled = torch.zeros_like(raw_keys)
    first_positions = torch.zeros((rows, 3), dtype=torch.int64, device=raw_keys.device)
    row_lookup = {
        (int(token_to_req[row]), int(logical_positions[row])): row
        for row in range(rows)
    }
    for row in range(rows):
        if int(compressed_slots[row]) < 0:
            continue
        request = int(token_to_req[row])
        if request < 0 or request >= num_requests:
            continue
        end = int(logical_positions[row])
        start = end - compress_ratio + 1
        keys: list[torch.Tensor] = []
        positions: list[torch.Tensor] = []
        for position in range(start, end + 1):
            source_row = row_lookup.get((request, position))
            if source_row is not None:
                keys.append(raw_keys[source_row, 0])
                positions.append(raw_positions[source_row, 0])
                continue
            block = int(compressor_state_block_table[request, 0])
            if block < 0 or block >= compressor_state_cache.shape[0]:
                raise ValueError("QSA compressor-state table contains an invalid block")
            offset = position % compressor_state_cache.shape[1]
            keys.append(compressor_state_cache[block, offset, 0])
            positions.append(
                torch.full((3,), position, dtype=torch.int64, device=raw_keys.device)
            )
        pooled[row, 0] = torch.stack(keys).float().mean(dim=0).to(raw_keys.dtype)
        first_positions[row] = positions[0]
    return pooled, first_positions


__all__ = [
    "expand_qsa_block_indices",
    "qsa_compress_groups_with_ratio",
    "qsa_mqa_paged",
    "qsa_select_paged_tokens",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
