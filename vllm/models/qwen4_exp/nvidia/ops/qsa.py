# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton kernels for Qwen4Exp QSA sparse attention and cache updates."""

from __future__ import annotations

import torch

from vllm.triton_utils import HAS_TRITON, tl, triton


@triton.jit
def _nvfp4_decode_e2m1(nibble):
    """Decode one FP4 E2M1 nibble (sign + 3 magnitude bits) to float32.

    For magnitude >= 2 the value is 2^(exp - 1) * (1 + m / 2), i.e. the
    float32 bit pattern (126 << 23) + (magnitude << 22); magnitudes 0 and 1
    are the subnormals 0.0 and 0.5.
    """
    magnitude = nibble & 0x07
    magnitude_i32 = magnitude.to(tl.int32)
    sign_bits = ((nibble & 0x08).to(tl.uint32)) << 28
    normal_bits = ((126 << 23) + (magnitude_i32 << 22)).to(tl.uint32) | sign_bits
    normal = normal_bits.to(tl.uint32).to(tl.float32, bitcast=True)
    subnormal_bits = ((magnitude & 0x01).to(tl.uint32) * 0x3F000000) | sign_bits
    subnormal = subnormal_bits.to(tl.uint32).to(tl.float32, bitcast=True)
    return tl.where(magnitude < 2, subnormal, normal)


@triton.jit
def _nvfp4_scale_to_float(bits):
    """Decode an E4M3 block scale (magnitude only) to float32.

    Block scales come from absolute maxima, so the sign bit is masked. Bias
    7 -> 127 is +120 on the exponent; subnormals are mant * 2^-9.
    """
    payload = bits.to(tl.int32) & 0x7F
    exp_bits = (payload >> 3) & 0x0F
    mant = payload & 0x07
    normal_bits = ((exp_bits + 120) << 23) | (mant << 20)
    normal = normal_bits.to(tl.uint32).to(tl.float32, bitcast=True)
    subnormal = mant.to(tl.float32) / 512.0
    value = tl.where(exp_bits == 0, subnormal, normal)
    return tl.where(payload == 0, 0.0, value)


@triton.jit
def _nvfp4_scale_coord(slot, group, SWIZZLED: tl.constexpr, SCALE_DIM: tl.constexpr):
    """Storage coordinates (row, scale index) of the block scale for a
    (row in page, 16-value group) pair.

    LINEAR is the identity. SWIZZLED is the permutation within groups of four
    rows that reshape_and_cache_flash writes for the V side:
        (t, s) -> ((t // 4) * 4 + s // G, (s % G) * 4 + t % 4),  G = SCALE_DIM // 4
    """
    SWIZZLE_GROUP: tl.constexpr = SCALE_DIM // 4
    if SWIZZLED:
        return (slot // 4) * 4 + (group // SWIZZLE_GROUP), (
            group % SWIZZLE_GROUP
        ) * 4 + (slot % 4)
    return slot + group * 0, group + slot * 0


@triton.jit
def _qsa_sparse_paged_gqa_splitk_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    # Per-layer fp32 scales (1-element device buffers). Only read on the
    # quantized branches; any valid pointer is passed for bf16.
    k_scale_ptr,
    v_scale_ptr,
    # nvfp4 block-scale pages (raw uint8 bits), only read on the nvfp4 branch.
    k_scale_cache_ptr,
    v_scale_cache_ptr,
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
    # Strides of the nvfp4 block-scale pages (from nvfp4_split_data_scale);
    # data and scale regions have different row widths.
    stride_ks_block,
    stride_ks_token,
    stride_ks_head,
    stride_vs_block,
    stride_vs_token,
    stride_vs_head,
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
    # Compile-time branch selection. With both False the dequantization code
    # is compiled out and the bf16 path is unchanged.
    KV_QUANT_FP8: tl.constexpr = False,
    KV_QUANT_NVFP4: tl.constexpr = False,
    V_SCALE_SWIZZLED: tl.constexpr = True,
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

    # Dynamic bounds avoid padded main-loop iterations for uneven splits.
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
        # physical_page * block stride can overflow int32 for large caches.
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        if KV_QUANT_NVFP4:
            # A cache row holds HEAD_DIM values in HEAD_DIM // 2 bytes (two
            # E2M1 nibbles per byte, even index in the low nibble) plus
            # HEAD_DIM // 16 E4M3 block scales in a separate page region.
            # value = nibble * block_scale * layer_scale. Orientation as in the
            # bf16 branch: keys [HEAD_DIM, BLOCK_N], values [BLOCK_N, HEAD_DIM].
            byte_offsets = dim_offsets // 2
            k_bytes = tl.load(
                k_cache_ptr
                + safe_page[None, :] * stride_k_block
                + page_offset[None, :] * stride_k_token
                + kv_head * stride_k_head
                + byte_offsets[:, None],
                mask=valid[None, :],
                other=0,
            )
            k_nib = tl.where(
                (dim_offsets[:, None] & 1) == 0, k_bytes & 0x0F, (k_bytes >> 4) & 0x0F
            )
            v_bytes = tl.load(
                v_cache_ptr
                + safe_page[:, None] * stride_v_block
                + page_offset[:, None] * stride_v_token
                + kv_head * stride_v_head
                + byte_offsets[None, :],
                mask=valid[:, None],
                other=0,
            )
            v_nib = tl.where(
                (dim_offsets[None, :] & 1) == 0, v_bytes & 0x0F, (v_bytes >> 4) & 0x0F
            )
            # Block scales are loaded at their natural width
            # [HEAD_DIM // 16, BLOCK_N] and broadcast to the data tile in
            # registers: dim_offsets is tl.arange(0, HEAD_DIM), so group =
            # dim // 16 covers 16 consecutive rows, which broadcast_to +
            # reshape restores. Decoding and the layer-scale multiply run on
            # HEAD_DIM // 16 rows instead of HEAD_DIM. K scales are stored
            # linearly, V scales follow V_SCALE_SWIZZLED. HEAD_DIM // 16 is
            # passed as an expression; a constexpr binding inside a constexpr
            # branch is not reliable in Triton.
            sf_groups = tl.arange(0, HEAD_DIM // 16)
            ks_slot, ks_group = _nvfp4_scale_coord(
                page_offset[None, :], sf_groups[:, None], False, HEAD_DIM // 16
            )
            k_sf = _nvfp4_scale_to_float(
                tl.load(
                    k_scale_cache_ptr
                    + safe_page[None, :] * stride_ks_block
                    + ks_slot * stride_ks_token
                    + kv_head * stride_ks_head
                    + ks_group,
                    mask=valid[None, :],
                    other=0,
                )
            ) * tl.load(k_scale_ptr)
            keys = (
                _nvfp4_decode_e2m1(k_nib)
                * tl.reshape(
                    tl.broadcast_to(k_sf[:, None, :], (HEAD_DIM // 16, 16, BLOCK_N)),
                    (HEAD_DIM, BLOCK_N),
                )
            ).to(query.dtype)
            vs_slot, vs_group = _nvfp4_scale_coord(
                page_offset[:, None],
                sf_groups[None, :],
                V_SCALE_SWIZZLED,
                HEAD_DIM // 16,
            )
            v_sf = _nvfp4_scale_to_float(
                tl.load(
                    v_scale_cache_ptr
                    + safe_page[:, None] * stride_vs_block
                    + vs_slot * stride_vs_token
                    + kv_head * stride_vs_head
                    + vs_group,
                    mask=valid[:, None],
                    other=0,
                )
            ) * tl.load(v_scale_ptr)
            values = (
                _nvfp4_decode_e2m1(v_nib)
                * tl.reshape(
                    tl.broadcast_to(v_sf[:, :, None], (BLOCK_N, HEAD_DIM // 16, 16)),
                    (BLOCK_N, HEAD_DIM),
                )
            ).to(query.dtype)
        else:
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
            if KV_QUANT_FP8:
                # Never feed fp8 into tl.dot directly (Triton cannot multiply
                # fp8e4nv on SM12x). Upcast to the query dtype, apply the
                # per-layer scale, cast back.
                keys = keys.to(query.dtype)
                keys = (keys * tl.load(k_scale_ptr)).to(query.dtype)
                values = values.to(query.dtype)
                values = (values * tl.load(v_scale_ptr)).to(query.dtype)
        scores = tl.dot(query, keys)
        # Scaling scores avoids re-quantizing a scaled query to BF16.
        scores *= softmax_scale_log2
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
        tl.store(
            partial_output_ptr
            + (
                (split_id * num_rows + row) * NUM_QUERY_HEADS
                + first_head
                + head_offsets[:, None]
            )
            * HEAD_DIM
            + dim_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_id * num_rows + row) * NUM_QUERY_HEADS
            + first_head
            + head_offsets,
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
    lse = tl.load(
        partial_lse_ptr + (split_offsets * num_rows + row) * NUM_QUERY_HEADS + head,
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
        + ((split_offsets[:, None] * num_rows + row) * NUM_QUERY_HEADS + head)
        * HEAD_DIM
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


@triton.jit
def _store_qsa_rows_kernel(
    cache_ptr,
    slots_ptr,
    rows_ptr,
    stride_cache_block,
    stride_cache_token,
    stride_cache_dim,
    stride_rows_row,
    stride_rows_dim,
    num_rows,
    num_blocks,
    PAGE_SIZE: tl.constexpr,
    WIDTH: tl.constexpr,
    BLOCK_D: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    slot = tl.load(slots_ptr + row)
    valid = (row < num_rows) & (slot >= 0) & (slot < num_blocks * PAGE_SIZE)
    block = tl.maximum(slot, 0) // PAGE_SIZE
    token = tl.maximum(slot, 0) % PAGE_SIZE
    values = tl.load(
        rows_ptr + row * stride_rows_row + dims * stride_rows_dim,
        mask=valid & (dims < WIDTH),
        other=0,
    )
    tl.store(
        cache_ptr
        + block * stride_cache_block
        + token * stride_cache_token
        + dims * stride_cache_dim,
        values,
        mask=valid & (dims < WIDTH),
    )


@triton.jit
def _compress_qsa_groups_kernel(
    raw_keys_ptr,  # this step's raw key rows, straight from activations
    raw_positions_ptr,  # this step's per-token positions
    compressor_state_cache_ptr,  # per-request ring of previous raw keys
    rope_cache_ptr,  # packed RoPE position tail of the ring
    compressor_state_table_ptr,
    token_to_req_ptr,
    query_start_loc_ptr,
    logical_positions_ptr,
    compressed_slots_ptr,
    pooled_ptr,
    first_positions_ptr,
    stride_raw_row,
    stride_raw_dim,
    stride_raw_positions_row,
    stride_raw_positions_dim,
    stride_compressor_state_block,
    stride_compressor_state_token,
    stride_compressor_state_dim,
    stride_rope_block,
    stride_rope_token,
    stride_rope_dim,
    stride_compressor_state_table_req,
    stride_pooled_row,
    stride_pooled_dim,
    stride_positions_row,
    stride_positions_dim,
    num_rows,
    num_compressor_state_blocks,
    num_requests,
    COMPRESSOR_STATE_SIZE: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LOAD_ROPE_POSITIONS: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    dims = tl.arange(0, BLOCK_D)
    request = tl.load(token_to_req_ptr + row)
    end_position = tl.load(logical_positions_ptr + row)
    compressed_slot = tl.load(compressed_slots_ptr + row)
    valid_request = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    query_row_start = tl.load(
        query_start_loc_ptr + safe_request, mask=valid_request, other=0
    )
    query_row_end = tl.load(
        query_start_loc_ptr + safe_request + 1, mask=valid_request, other=0
    )
    chunk_start_position = end_position - (row - query_row_start)
    compressor_state_block = tl.load(
        compressor_state_table_ptr + safe_request * stride_compressor_state_table_req,
        mask=valid_request,
        other=-1,
    )
    valid_compressor_state_block = (compressor_state_block >= 0) & (
        compressor_state_block < num_compressor_state_blocks
    )
    valid_row = (
        (row < num_rows)
        & valid_request
        & (row >= query_row_start)
        & (row < query_row_end)
        & (end_position >= COMPRESS_RATIO - 1)
        & (compressed_slot >= 0)
    )
    accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

    # A group can span the compressor-state ring (older members) and this
    # step's raw rows (members at positions >= chunk_start_position).
    for group_offset in tl.range(0, COMPRESS_RATIO):
        position = end_position - (COMPRESS_RATIO - 1 - group_offset)
        use_raw = position >= chunk_start_position
        raw_row = query_row_start + position - chunk_start_position
        raw_values = tl.load(
            raw_keys_ptr + raw_row * stride_raw_row + dims * stride_raw_dim,
            mask=valid_row
            & use_raw
            & (raw_row >= query_row_start)
            & (raw_row < query_row_end)
            & (raw_row < num_rows)
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        compressor_state_values = tl.load(
            compressor_state_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64)
            * stride_compressor_state_block
            + (position % COMPRESSOR_STATE_SIZE) * stride_compressor_state_token
            + dims * stride_compressor_state_dim,
            mask=valid_row
            & ~use_raw
            & valid_compressor_state_block
            & (dims < HEAD_DIM),
            other=0.0,
        ).to(tl.float32)
        accumulator += tl.where(use_raw, raw_values, compressor_state_values)

    tl.store(
        pooled_ptr + row * stride_pooled_row + dims * stride_pooled_dim,
        accumulator / COMPRESS_RATIO,
        mask=(row < num_rows) & (dims < HEAD_DIM),
    )

    position_dims = tl.arange(0, 4)
    first_position = end_position - COMPRESS_RATIO + 1
    if LOAD_ROPE_POSITIONS:
        first_from_raw = first_position >= chunk_start_position
        raw_first_row = query_row_start + first_position - chunk_start_position
        raw_position_values = tl.load(
            raw_positions_ptr
            + raw_first_row * stride_raw_positions_row
            + position_dims * stride_raw_positions_dim,
            mask=valid_row
            & first_from_raw
            & (raw_first_row >= query_row_start)
            & (raw_first_row < query_row_end)
            & (raw_first_row < num_rows)
            & (position_dims < 3),
            other=0,
        )
        compressor_state_position_values = tl.load(
            rope_cache_ptr
            + tl.maximum(compressor_state_block, 0).to(tl.int64) * stride_rope_block
            + (first_position % COMPRESSOR_STATE_SIZE) * stride_rope_token
            + position_dims * stride_rope_dim,
            mask=valid_row
            & ~first_from_raw
            & valid_compressor_state_block
            & (position_dims < 3),
            other=0,
        )
        position_values = tl.where(
            first_from_raw,
            raw_position_values,
            compressor_state_position_values,
        )
    else:
        position_values = tl.where(valid_row, first_position, 0)
    tl.store(
        first_positions_ptr
        + row * stride_positions_row
        + position_dims * stride_positions_dim,
        position_values,
        mask=(row < num_rows) & (position_dims < 3),
    )


def qsa_sparse_paged_attention(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor | None = None,
    k_scale: torch.Tensor | None = None,
    v_scale: torch.Tensor | None = None,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
    v_scale_swizzled: bool = True,
) -> torch.Tensor:
    """Run sparse GQA directly over paged BF16, FP8-E4M3 or NVFP4 K/V caches.

    ``k_scale`` / ``v_scale`` are the per-layer scales (1-element device
    tensors), required for quantized caches. For nvfp4, ``k_cache`` /
    ``v_cache`` are the packed data pages (uint8, last dim ``head_dim // 2``)
    and ``k_scale_cache`` / ``v_scale_cache`` the block-scale pages (uint8,
    last dim ``head_dim // 16``).
    """

    if not q.is_cuda or not HAS_TRITON:
        raise RuntimeError("paged QSA sparse attention requires CUDA and Triton")
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
    use_nvfp4 = k_scale_cache is not None
    row_width = q.shape[2] // 2 if use_nvfp4 else q.shape[2]
    if row_width != k_cache.shape[3] or q.shape[1] % k_cache.shape[2]:
        raise ValueError("QSA sparse attention requires valid grouped-query heads")
    head_dim = q.shape[2]
    assert head_dim >= 16 and (head_dim & (head_dim - 1)) == 0
    assert q.dtype == torch.bfloat16
    assert k_cache.dtype == v_cache.dtype
    assert k_cache.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8)
    use_fp8 = k_cache.dtype == torch.float8_e4m3fn
    if use_nvfp4:
        if k_cache.dtype != torch.uint8:
            raise ValueError("QSA nvfp4 KV cache must be stored as uint8")
        if k_scale_cache is None or v_scale_cache is None:
            raise ValueError("QSA nvfp4 KV cache requires both scale pages")
        if k_scale_cache.dtype != torch.uint8 or v_scale_cache.dtype != torch.uint8:
            raise ValueError("QSA nvfp4 block scales must be raw uint8 bits")
        if k_scale_cache.shape[3] != head_dim // 16:
            raise ValueError("QSA nvfp4 block-scale page has the wrong width")
        if k_scale_cache.shape[:3] != k_cache.shape[:3]:
            raise ValueError("QSA nvfp4 data and scale pages disagree in shape")
        if k_scale_cache.stride(3) != 1 or v_scale_cache.stride(3) != 1:
            raise ValueError("QSA nvfp4 block scales must be contiguous per row")
    elif k_cache.dtype == torch.uint8:
        raise ValueError("QSA uint8 K/V cache without block scales is not nvfp4")
    quantized = use_fp8 or use_nvfp4
    if quantized:
        if k_scale is None or v_scale is None:
            raise ValueError("QSA quantized KV cache requires k_scale and v_scale")
        if k_scale.numel() != 1 or v_scale.numel() != 1:
            raise ValueError("QSA quantized KV cache expects scalar per-layer scales")
        if k_scale.device != q.device or v_scale.device != q.device:
            raise ValueError("QSA KV scales must live on the query device")
    assert logical_indices.dtype == block_table.dtype == torch.int32
    assert token_to_req.dtype == torch.int32
    assert q.device == k_cache.device == v_cache.device
    assert q.device == logical_indices.device == block_table.device
    assert q.device == token_to_req.device
    assert q.stride(2) == k_cache.stride(3) == v_cache.stride(3) == 1
    assert logical_indices.stride(1) == block_table.stride(1) == 1
    assert token_to_req.stride(0) == 1
    if out is None:
        out = torch.empty_like(q)
    if out.shape != q.shape:
        raise ValueError("QSA sparse output must match its query")
    assert out.dtype == q.dtype and out.device == q.device
    assert out.stride(2) == 1
    if not q.shape[0]:
        return out

    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    small_profile_limit = 8 if block_m <= 8 else 4

    # Tuned on GB300 for the Qwen-Air TP1, TP2, and TP4 attention shapes.
    # Narrow tiles favor decode; wide tiles improve throughput for prefill.
    if base_programs <= small_profile_limit:
        block_n, target_splits, partial_warps = 16, 64, 4
    elif base_programs < 32:
        block_n, target_splits, partial_warps = 16, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, partial_warps = 64, 8, 2
    elif base_programs <= 512:
        block_n, target_splits, partial_warps = 64, 4, 2
    else:
        block_n, target_splits, partial_warps = 64, 1, 2

    # Quantized caches need shared memory for the dequantized bf16 tiles next
    # to the raw tiles. On SM120 the fp8 branch with BLOCK_N=64 and two
    # pipeline stages requested 106496 B against a 101376 B limit, so the
    # wide (prefill) profiles run with one stage; the decode profile
    # (BLOCK_N=16) fits with two stages and keeps them. For nvfp4 the wide
    # tile is halved to 32: with the narrow scale tile this measured 1.7x
    # faster than 64 on SM120 (two stages at 64 were slower still,
    # occupancy-bound).
    num_stages = 2
    if quantized and block_n > 16:
        if use_nvfp4:
            block_n = 32
        num_stages = 1

    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)
    # Avoid empty splits when the selection width is smaller than the profile.
    max_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(max_useful_splits, target_splits)

    # Split=1 writes output directly and compiles out all workspace accesses.
    if num_splits == 1:
        partial_output = out
        partial_lse = out
    else:
        # FP32 partials preserve accuracy when merging independently normalized
        # splits.
        partial_output = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device
        )
        partial_lse = torch.empty(
            (num_splits, q.shape[0], q.shape[1]),
            dtype=torch.float32,
            device=q.device,
        )

    partial_grid = (q.shape[0], k_cache.shape[2], num_splits)
    # Triton needs an argument even for compiled-out branches; any valid
    # tensor is passed on the bf16 path.
    k_scale_arg = k_scale if quantized else q
    v_scale_arg = v_scale if quantized else q
    k_sf_arg = q
    v_sf_arg = q
    ks_strides = (0, 0, 0)
    vs_strides = (0, 0, 0)
    if use_nvfp4:
        assert k_scale_cache is not None and v_scale_cache is not None
        k_sf_arg = k_scale_cache
        v_sf_arg = v_scale_cache
        ks_strides = (
            k_scale_cache.stride(0),
            k_scale_cache.stride(1),
            k_scale_cache.stride(2),
        )
        vs_strides = (
            v_scale_cache.stride(0),
            v_scale_cache.stride(1),
            v_scale_cache.stride(2),
        )
    _qsa_sparse_paged_gqa_splitk_kernel[partial_grid](
        q,
        k_cache,
        v_cache,
        k_scale_arg,
        v_scale_arg,
        k_sf_arg,
        v_sf_arg,
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
        ks_strides[0],
        ks_strides[1],
        ks_strides[2],
        vs_strides[0],
        vs_strides[1],
        vs_strides[2],
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
        KV_QUANT_FP8=use_fp8,
        KV_QUANT_NVFP4=use_nvfp4,
        V_SCALE_SWIZZLED=bool(v_scale_swizzled),
        num_warps=partial_warps,
        num_stages=num_stages,
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
        num_warps=2,
        num_stages=1,
    )
    return out


def qsa_store_cache_rows(
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rows: torch.Tensor,
) -> None:
    """Store fixed-width rows in a QSA cache without boolean indexing."""

    if not cache.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA cache stores require Triton")
    if cache.ndim != 4 or cache.shape[2] != 1:
        raise ValueError("QSA cache must be [pages, page_size, 1, width]")
    if not all(cache.shape):
        raise ValueError("QSA cache dimensions must be nonzero")
    if rows.ndim == 3:
        if rows.shape[1] != 1:
            raise ValueError("QSA cache rows must have one head")
        rows = rows[:, 0]
    if rows.shape != (slot_mapping.numel(), cache.shape[3]):
        raise ValueError("QSA cache rows and slots have incompatible shapes")
    if not rows.shape[0]:
        return
    _store_qsa_rows_kernel[(rows.shape[0],)](
        cache,
        slot_mapping,
        rows,
        cache.stride(0),
        cache.stride(1),
        cache.stride(3),
        rows.stride(0),
        rows.stride(1),
        rows.shape[0],
        cache.shape[0],
        PAGE_SIZE=cache.shape[1],
        WIDTH=cache.shape[3],
        BLOCK_D=triton.next_power_of_2(cache.shape[3]),
        num_warps=4,
    )


def qsa_compress_groups_with_ratio(
    raw_keys: torch.Tensor,  # this step's raw key rows [rows, 1, head_size]
    raw_positions: torch.Tensor,  # this step's positions [rows, 1, 3] int64
    compressor_state_cache: torch.Tensor,
    compressor_state_block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_slots: torch.Tensor,
    compress_ratio: int,
    rope_cache: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pool completed groups from the compressor-state ring and raw token rows."""

    if not raw_keys.is_cuda or not HAS_TRITON:
        raise RuntimeError("QSA CUDA compression requires Triton")
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
        # The ring is wider than one group so speculative rows cannot alias
        # onto the committed keys of the group still being collected.
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
    if rope_cache is not None and (
        rope_cache.ndim != 4
        or rope_cache.shape[:3] != compressor_state_cache.shape[:3]
        or rope_cache.shape[3] != 3
        or rope_cache.dtype != torch.int64
    ):
        raise ValueError("QSA packed position view has an invalid shape or dtype")
    if rows and (
        not all(compressor_state_cache.shape)
        or not all(compressor_state_block_table.shape)
    ):
        raise ValueError("QSA compressor-state cache and block table must be nonempty")
    pooled = torch.empty(
        (rows, 1, raw_keys.shape[2]),
        dtype=raw_keys.dtype,
        device=raw_keys.device,
    )
    first_positions = torch.empty((rows, 3), dtype=torch.int64, device=raw_keys.device)
    if not rows:
        return pooled, first_positions
    if rope_cache is None:
        rope_cache = compressor_state_cache
        load_rope_positions = False
    else:
        load_rope_positions = True
    _compress_qsa_groups_kernel[(rows,)](
        raw_keys,
        raw_positions,
        compressor_state_cache,
        rope_cache,
        compressor_state_block_table,
        token_to_req,
        query_start_loc,
        logical_positions,
        compressed_slots,
        pooled,
        first_positions,
        raw_keys.stride(0),
        raw_keys.stride(2),
        raw_positions.stride(0),
        raw_positions.stride(2),
        compressor_state_cache.stride(0),
        compressor_state_cache.stride(1),
        compressor_state_cache.stride(3),
        rope_cache.stride(0),
        rope_cache.stride(1),
        rope_cache.stride(3),
        compressor_state_block_table.stride(0),
        pooled.stride(0),
        pooled.stride(2),
        first_positions.stride(0),
        first_positions.stride(1),
        rows,
        compressor_state_cache.shape[0],
        num_requests,
        COMPRESSOR_STATE_SIZE=compressor_state_cache.shape[1],
        COMPRESS_RATIO=compress_ratio,
        HEAD_DIM=raw_keys.shape[2],
        LOAD_ROPE_POSITIONS=load_rope_positions,
        BLOCK_D=triton.next_power_of_2(raw_keys.shape[2]),
        num_warps=4,
    )
    return pooled, first_positions


__all__ = [
    "qsa_compress_groups_with_ratio",
    "qsa_sparse_paged_attention",
    "qsa_store_cache_rows",
]
