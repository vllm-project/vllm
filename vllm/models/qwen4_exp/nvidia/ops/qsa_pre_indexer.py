# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused QSA pre-indexer kernel for Qwen4Exp."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _norm_rope(
    x,
    pos_t,
    pos_h,
    pos_w,
    cos_sin_ptr,
    cos_sin_stride,
    norm_weight_ptr,
    eps,
    IS_MROPE: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    """Apply Gemma RMSNorm and selected-axis NeoX RoPE to register rows."""
    TILE_T: tl.constexpr = x.shape[0]
    TILE_H: tl.constexpr = x.shape[1]
    D: tl.constexpr = x.shape[2]
    ROWS: tl.constexpr = TILE_T * TILE_H
    HALF: tl.constexpr = D // 2
    QUARTER: tl.constexpr = D // 4
    pairs = tl.arange(0, QUARTER)
    if IS_MROPE:
        # Qwen interleaves temporal, height, and width rotary pairs. Each axis
        # still indexes the same position-major cos/sin table.
        h_mask = ((pairs % 3) == 1) & (pairs <= 3 * MROPE_H)
        w_mask = ((pairs % 3) == 2) & (pairs <= 3 * MROPE_W)
        t_mask = ~(h_mask | w_mask)
        base = cos_sin_ptr + pairs[None, :]
        pos_rows = (pos_t, pos_h, pos_w)
        axis_masks = (t_mask, h_mask, w_mask)
        cos = tl.zeros((TILE_T, QUARTER), dtype=cos_sin_ptr.dtype.element_ty)
        sin = tl.zeros((TILE_T, QUARTER), dtype=cos_sin_ptr.dtype.element_ty)
        for axis in tl.static_range(3):
            cos += tl.load(
                base + pos_rows[axis][:, None] * cos_sin_stride,
                mask=axis_masks[axis][None, :],
                other=0,
            )
            sin += tl.load(
                base + pos_rows[axis][:, None] * cos_sin_stride + QUARTER,
                mask=axis_masks[axis][None, :],
                other=0,
            )
    else:
        cos = tl.load(cos_sin_ptr + pos_t[:, None] * cos_sin_stride + pairs[None, :])
        sin = tl.load(
            cos_sin_ptr + pos_t[:, None] * cos_sin_stride + QUARTER + pairs[None, :]
        )

    cos = tl.reshape(
        tl.broadcast_to(cos[:, None, :], (TILE_T, TILE_H, QUARTER)),
        (ROWS, QUARTER),
    )
    sin = tl.reshape(
        tl.broadcast_to(sin[:, None, :], (TILE_T, TILE_H, QUARTER)),
        (ROWS, QUARTER),
    )
    x = tl.reshape(x, (ROWS, D)).to(tl.float32)
    weight = tl.load(norm_weight_ptr + tl.arange(0, D)).to(tl.float32) + 1.0
    rrms = tl.rsqrt(tl.sum(x * x, axis=1) / D + eps)
    y = (x * rrms[:, None] * weight[None, :]).to(cos.dtype)
    rotated, passthrough = tl.split(
        tl.permute(tl.reshape(y, (ROWS, 2, HALF)), (0, 2, 1))
    )
    r0, r1 = tl.split(tl.permute(tl.reshape(rotated, (ROWS, 2, QUARTER)), (0, 2, 1)))
    out0 = r0 * cos - r1 * sin
    out1 = r1 * cos + r0 * sin
    rotated = tl.reshape(tl.permute(tl.join(out0, out1), (0, 2, 1)), (ROWS, HALF))
    result = tl.reshape(tl.permute(tl.join(rotated, passthrough), (0, 2, 1)), (ROWS, D))
    return tl.reshape(result, (TILE_T, TILE_H, D))


@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_state_blocks",
        "num_compressed_blocks",
        "num_k_work",
    ]
)
def _qsa_pre_indexer_kernel(
    q_ptr,
    q_stride_token,
    k_ptr,
    k_stride_token,
    pos_ptr,
    pos_stride_axis,
    pos_stride_token,
    cos_sin_ptr,
    q_norm_weight_ptr,
    k_norm_weight_ptr,
    eps,
    q_out_ptr,
    q_out_stride_token,
    q_out_stride_head,
    state_cache_ptr,
    state_cache_stride_block,
    state_cache_stride_token,
    state_slots_ptr,
    state_table_ptr,
    state_table_stride_req,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    k_work_metadata_ptr,
    compressed_cache_ptr,
    compressed_cache_stride_block,
    compressed_cache_stride_token,
    num_tokens,
    num_state_blocks,
    num_compressed_blocks,
    num_k_work,
    HQ: tl.constexpr,
    D: tl.constexpr,
    TILE_T_Q: tl.constexpr,
    TILE_H_Q: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    STATE_SIZE: tl.constexpr,
    COMP_PAGE_SIZE: tl.constexpr,
    IS_2D_POSITIONS: tl.constexpr,
    IS_K_MROPE: tl.constexpr,
    CACHE_HAS_ROPE_POS: tl.constexpr,
    MROPE_H: tl.constexpr,
    MROPE_W: tl.constexpr,
):
    pid = tl.program_id(0)
    # K work occupies the first programs; the remaining programs tile Q. This
    # keeps both paths in one launch while leaving their register shapes
    # independent.
    if pid >= num_k_work:
        q_pid = pid - num_k_work
        num_head_tiles: tl.constexpr = tl.cdiv(HQ, TILE_H_Q)
        token_tile = q_pid // num_head_tiles
        head_tile = q_pid % num_head_tiles
        tokens = token_tile * TILE_T_Q + tl.arange(0, TILE_T_Q)
        heads = head_tile * TILE_H_Q + tl.arange(0, TILE_H_Q)
        valid_tokens = tokens < num_tokens
        valid_heads = heads < HQ
        dims = tl.arange(0, D)
        mask = valid_tokens[:, None, None] & valid_heads[None, :, None]
        x = tl.load(
            q_ptr
            + tokens[:, None, None] * q_stride_token
            + heads[None, :, None] * D
            + dims[None, None, :],
            mask=mask,
            other=0.0,
        )
        pos_t = tl.load(pos_ptr + tokens * pos_stride_token, mask=valid_tokens, other=0)
        if IS_2D_POSITIONS:
            pos_h = tl.load(
                pos_ptr + pos_stride_axis + tokens * pos_stride_token,
                mask=valid_tokens,
                other=0,
            )
            pos_w = tl.load(
                pos_ptr + 2 * pos_stride_axis + tokens * pos_stride_token,
                mask=valid_tokens,
                other=0,
            )
        else:
            pos_h = pos_t
            pos_w = pos_t
        y = _norm_rope(
            x,
            pos_t,
            pos_h,
            pos_w,
            cos_sin_ptr,
            D // 2,
            q_norm_weight_ptr,
            eps,
            IS_2D_POSITIONS,
            MROPE_H,
            MROPE_W,
        )
        tl.store(
            q_out_ptr
            + tokens[:, None, None] * q_out_stride_token
            + heads[None, :, None] * q_out_stride_head
            + dims[None, None, :],
            y,
            mask=mask,
        )

    if pid < num_k_work:
        # One work item owns one completed compression group. Work item zero
        # additionally commits the request's current raw-K suffix below.
        work_metadata = tl.load(k_work_metadata_ptr + pid * 2 + tl.arange(0, 2))
        request, work_in_request = tl.split(work_metadata)
        if request < 0:
            return

        query_start = tl.load(query_start_loc_ptr + request)
        query_end = tl.load(query_start_loc_ptr + request + 1)
        query_len = query_end - query_start
        chunk_end = tl.load(logical_positions_ptr + query_end - 1)
        chunk_start = chunk_end - query_len + 1
        num_groups = (chunk_end + 1) // COMPRESS_RATIO - chunk_start // COMPRESS_RATIO
        dims = tl.arange(0, D)

        if work_in_request < num_groups:
            first_boundary = (
                (chunk_start + COMPRESS_RATIO) // COMPRESS_RATIO
            ) * COMPRESS_RATIO - 1
            end_position = first_boundary + work_in_request * COMPRESS_RATIO
            boundary_token = query_start + end_position - chunk_start
            valid_token = (
                (boundary_token >= query_start)
                & (boundary_token < query_end)
                & (boundary_token < num_tokens)
            )
            compressed_slot = tl.load(
                compressed_slots_ptr + boundary_token,
                mask=valid_token,
                other=-1,
            )
            valid = (
                valid_token
                & (compressed_slot >= 0)
                & (compressed_slot < num_compressed_blocks * COMP_PAGE_SIZE)
            )
            state_block = tl.load(state_table_ptr + request * state_table_stride_req)
            state_block_valid = (state_block >= 0) & (state_block < num_state_blocks)
            safe_state_block = tl.maximum(state_block, 0).to(tl.int64)
            group_offsets = tl.arange(0, COMPRESS_RATIO)
            source_positions = end_position - (COMPRESS_RATIO - 1) + group_offsets
            source_in_chunk = source_positions >= chunk_start
            source_tokens = query_start + source_positions - chunk_start
            source_tokens_valid = (
                (source_tokens >= query_start)
                & (source_tokens < query_end)
                & (source_tokens < num_tokens)
            )
            current_base = (
                k_ptr + tl.maximum(source_tokens, 0)[:, None] * k_stride_token
            )
            cached_base = (
                state_cache_ptr
                + safe_state_block * state_cache_stride_block
                + (source_positions % STATE_SIZE)[:, None] * state_cache_stride_token
            )
            # Only the first completed group can cross the chunk boundary. Select
            # historical rows from the ring without issuing two masked loads.
            source_base = tl.where(source_in_chunk[:, None], current_base, cached_base)
            # Pointer selection obscures alignment from Triton's analysis.
            source_base = tl.multiple_of(source_base, (8, 8))
            source_valid = tl.where(
                source_in_chunk, source_tokens_valid, state_block_valid
            )
            source = tl.load(
                source_base + dims[None, :],
                mask=valid & source_valid[:, None],
                other=0.0,
            ).to(tl.float32)
            # Match the unfused path's BF16 pooled tensor before RMSNorm.
            pooled = (
                (tl.sum(source, axis=0) / COMPRESS_RATIO).to(tl.bfloat16).to(tl.float32)
            )

            first_position = end_position - (COMPRESS_RATIO - 1)
            if CACHE_HAS_ROPE_POS:
                # RoPE uses the first token in the pooled group. Its exact MRoPE
                # coordinates may live in this chunk or the raw-state ring.
                first_in_chunk = first_position >= chunk_start
                first_token = query_start + first_position - chunk_start
                first_token_valid = (
                    (first_token >= query_start)
                    & (first_token < query_end)
                    & (first_token < num_tokens)
                )
                safe_first_token = tl.maximum(first_token, 0)
                load_current_position = first_in_chunk & first_token_valid
                first_pos_t = tl.load(
                    pos_ptr + safe_first_token * pos_stride_token,
                    mask=load_current_position,
                    other=0,
                )
                if IS_2D_POSITIONS:
                    first_pos_h = tl.load(
                        pos_ptr + pos_stride_axis + safe_first_token * pos_stride_token,
                        mask=load_current_position,
                        other=0,
                    )
                    first_pos_w = tl.load(
                        pos_ptr
                        + 2 * pos_stride_axis
                        + safe_first_token * pos_stride_token,
                        mask=load_current_position,
                        other=0,
                    )
                else:
                    first_pos_h = first_pos_t
                    first_pos_w = first_pos_t
                tail = (
                    state_cache_ptr
                    + safe_state_block * state_cache_stride_block
                    + (first_position % STATE_SIZE) * state_cache_stride_token
                    + D
                ).to(tl.pointer_type(tl.int64))
                load_cached_position = ~first_in_chunk & state_block_valid
                cached_pos_t = tl.load(tail, mask=load_cached_position, other=0)
                cached_pos_h = tl.load(tail + 1, mask=load_cached_position, other=0)
                cached_pos_w = tl.load(tail + 2, mask=load_cached_position, other=0)
                pos_t = tl.where(first_in_chunk, first_pos_t.to(tl.int64), cached_pos_t)
                pos_h = tl.where(first_in_chunk, first_pos_h.to(tl.int64), cached_pos_h)
                pos_w = tl.where(first_in_chunk, first_pos_w.to(tl.int64), cached_pos_w)
            else:
                pos_t = first_position
                pos_h = first_position
                pos_w = first_position
            y = _norm_rope(
                tl.reshape(pooled, (1, 1, D)),
                pos_t + tl.arange(0, 1),
                pos_h + tl.arange(0, 1),
                pos_w + tl.arange(0, 1),
                cos_sin_ptr,
                D // 2,
                k_norm_weight_ptr,
                eps,
                IS_K_MROPE,
                MROPE_H,
                MROPE_W,
            )
            compressed_block = (compressed_slot // COMP_PAGE_SIZE).to(tl.int64)
            compressed_row = compressed_slot % COMP_PAGE_SIZE
            tl.store(
                compressed_cache_ptr
                + compressed_block * compressed_cache_stride_block
                + compressed_row * compressed_cache_stride_token
                + dims,
                tl.reshape(y, (D,)),
                mask=valid,
            )

        if work_in_request == 0:
            # This CTA may have just read historical rows from the circular buffer.
            # Keep every lane past those loads before overwriting the same ring.
            tl.debug_barrier()
            # One CTA per request commits only the suffix retained by the ring.
            num_state_rows = tl.minimum(query_len, STATE_SIZE)
            for state_offset in tl.range(0, num_state_rows):
                token = query_end - num_state_rows + state_offset
                valid_token = (
                    (token >= query_start) & (token < query_end) & (token < num_tokens)
                )
                slot = tl.load(state_slots_ptr + token, mask=valid_token, other=-1)
                valid_slot = (
                    valid_token & (slot >= 0) & (slot < num_state_blocks * STATE_SIZE)
                )
                safe_slot = tl.maximum(slot, 0)
                state_row = (
                    state_cache_ptr
                    + (safe_slot // STATE_SIZE).to(tl.int64) * state_cache_stride_block
                    + (safe_slot % STATE_SIZE) * state_cache_stride_token
                )
                k = tl.load(
                    k_ptr + tl.maximum(token, 0) * k_stride_token + dims,
                    mask=valid_slot,
                    other=0.0,
                )
                tl.store(state_row + dims, k, mask=valid_slot)
                if CACHE_HAS_ROPE_POS:
                    pos_t = tl.load(
                        pos_ptr + tl.maximum(token, 0) * pos_stride_token,
                        mask=valid_slot,
                        other=0,
                    )
                    if IS_2D_POSITIONS:
                        pos_h = tl.load(
                            pos_ptr
                            + pos_stride_axis
                            + tl.maximum(token, 0) * pos_stride_token,
                            mask=valid_slot,
                            other=0,
                        )
                        pos_w = tl.load(
                            pos_ptr
                            + 2 * pos_stride_axis
                            + tl.maximum(token, 0) * pos_stride_token,
                            mask=valid_slot,
                            other=0,
                        )
                    else:
                        pos_h = pos_t
                        pos_w = pos_t
                    tail = (state_row + D).to(tl.pointer_type(tl.int64))
                    tl.store(tail, pos_t.to(tl.int64), mask=valid_slot)
                    tl.store(tail + 1, pos_h.to(tl.int64), mask=valid_slot)
                    tl.store(tail + 2, pos_w.to(tl.int64), mask=valid_slot)


def qsa_pre_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    state_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_slots: torch.Tensor,
    k_work_metadata: torch.Tensor,
    *,
    compress_ratio: int,
    mrope_section: tuple[int, int, int] | None,
    rope_pos_offset: int | None,
) -> None:
    """Normalize Q, compress K, then update the circular raw state."""
    num_tokens = q.shape[0]
    if num_tokens == 0:
        return
    num_q_heads, head_dim = q_out.shape[1:]
    assert cos_sin_cache.shape[-1] * 2 == head_dim
    assert q.shape == (num_tokens, num_q_heads * head_dim)
    assert k.shape == (num_tokens, head_dim)
    assert q.stride(-1) == 1
    assert k.stride(-1) == 1
    assert q_out.stride(-1) == 1
    assert cos_sin_cache.is_contiguous()
    assert state_cache.stride(-1) == 1
    assert compressed_cache.stride(-1) == 1
    assert k_work_metadata.ndim == 2 and k_work_metadata.shape[1] == 2
    is_2d_positions = positions.ndim == 2
    is_k_mrope = bool(mrope_section)
    cache_has_rope_pos = rope_pos_offset is not None
    assert rope_pos_offset is None or rope_pos_offset == head_dim
    if is_2d_positions:
        assert positions.shape == (3, num_tokens)
        assert is_k_mrope
        pos_stride_axis, pos_stride_token = positions.stride()
    else:
        assert positions.shape == (num_tokens,)
        pos_stride_axis, pos_stride_token = 0, positions.stride(0)
    section = mrope_section if mrope_section is not None else (0, 0, 0)
    assert len(section) == 3

    if num_tokens <= 4096:
        TILE_T_Q, TILE_H_Q = 2, 2
    else:
        TILE_T_Q, TILE_H_Q = 2, 4
    num_k_work = k_work_metadata.shape[0]
    num_q_work = triton.cdiv(num_tokens, TILE_T_Q) * triton.cdiv(num_q_heads, TILE_H_Q)
    _qsa_pre_indexer_kernel[(num_k_work + num_q_work,)](
        q,
        q.stride(0),
        k,
        k.stride(0),
        positions,
        pos_stride_axis,
        pos_stride_token,
        cos_sin_cache,
        q_norm_weight,
        k_norm_weight,
        eps,
        q_out,
        q_out.stride(0),
        q_out.stride(1),
        state_cache,
        state_cache.stride(0),
        state_cache.stride(1),
        state_slots,
        state_block_table,
        state_block_table.stride(0),
        query_start_loc,
        logical_positions,
        compressed_slots,
        k_work_metadata,
        compressed_cache,
        compressed_cache.stride(0),
        compressed_cache.stride(1),
        num_tokens,
        state_cache.shape[0],
        compressed_cache.shape[0],
        num_k_work,
        HQ=num_q_heads,
        D=head_dim,
        TILE_T_Q=TILE_T_Q,
        TILE_H_Q=TILE_H_Q,
        COMPRESS_RATIO=compress_ratio,
        STATE_SIZE=state_cache.shape[1],
        COMP_PAGE_SIZE=compressed_cache.shape[1],
        IS_2D_POSITIONS=is_2d_positions,
        IS_K_MROPE=is_k_mrope,
        CACHE_HAS_ROPE_POS=cache_has_rope_pos,
        MROPE_H=section[1],
        MROPE_W=section[2],
        num_warps=1,
    )


__all__ = ["qsa_pre_indexer"]
