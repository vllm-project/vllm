# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Triton kernels for DeepseekV4 paged K-cache management and sparse-attention index
preparation.

- quantize_and_insert_k_cache: quantize bf16 K to UE8M0 FP8 and insert into
  the paged cache.
- dequantize_and_gather_k_cache: gather and dequantize FP8 K from the paged
  cache for sparse/SWA prefill.
- compute_global_topk_indices_and_lens: map local topk indices to global KV
  cache slots and count valid entries.
- combine_topk_swa_indices: concatenate topk compressed indices with SWA
  window indices for sparse prefill.
"""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def quantize_and_insert_k_kernel(
    # Input tensors
    k_ptr,  # [num_tokens, 512] bf16
    slot_mapping_ptr,  # [num_tokens] int64
    # Output tensor
    k_cache_ptr,  # [num_blocks, block_bytes] as uint8 (flattened view)
    # Dimensions
    num_tokens,
    input_dim: tl.constexpr,  # 512
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 (paged cache block size)
    token_data_size: tl.constexpr,  # 576 bytes per token data
    block_stride: tl.constexpr,  # total bytes per block (padded)
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 8 (7 real + 1 padding)
):
    """
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - [0, 64*576): Token data, each token has 448 fp8 + 128 bf16
    - [64*576, 64*576 + 64*8): Scales, each token has 8 uint8 scales
    - [64*576 + 64*8, block_stride): Padding

    One program per token.
    """
    pid = tl.program_id(0)

    if pid >= num_tokens:
        return

    # Get slot mapping
    slot_idx = tl.load(slot_mapping_ptr + pid)
    if slot_idx == -1:
        return

    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    # Input pointer for this token
    input_row_ptr = k_ptr + pid * input_dim

    # int64: block_idx * block_stride can exceed 2^31 with many KV-cache blocks
    # (e.g. >= 57K at block_stride ~37K). Matches gather path below.
    cache_block_ptr = k_cache_ptr + block_idx.to(tl.int64) * block_stride

    # Token data pointer: token data is stored contiguously at start of block
    # Each token's data is at offset pos_in_block * token_data_size
    token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

    # Scale pointer: scales are stored after ALL token data in the block
    # Scale for this token is at offset (64 * 576) + pos_in_block * 8
    token_scale_ptr = (
        cache_block_ptr + cache_block_size * token_data_size + pos_in_block * scale_dim
    )

    # Token data layout: [0:448] fp8, [448:576] bf16
    token_fp8_ptr = token_data_ptr
    token_bf16_ptr = token_data_ptr + fp8_dim

    # ========== Quantize and store FP8 portion (first 448 elements) ==========
    # Using UE8M0 quantization strategy (scale is power of 2, stored as uint8 exponent)
    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block

        if qblock_start < fp8_dim:
            offsets = qblock_start + tl.arange(0, quant_block)
            mask = offsets < fp8_dim

            # Load bf16 input
            x = tl.load(input_row_ptr + offsets, mask=mask, other=0.0)

            # Compute absmax scale (same as CUDA kernel)
            abs_x = tl.abs(x)
            block_max = tl.max(abs_x, axis=0)
            block_max = tl.maximum(block_max, 1e-4)  # Match CUDA: fmaxf(amax, 1e-4)

            # UE8M0: Round scale UP to next power of 2
            # scale = 2^ceil(log2(block_max / fp8_max))
            raw_scale = block_max / fp8_max
            log_scale = tl.log2(raw_scale)
            exponent = tl.ceil(log_scale)  # Round UP to next integer exponent
            scale = tl.exp2(exponent)  # scale = 2^exponent (power of 2)

            # Quantize to fp8: fp8_value = bf16_value / scale
            x_scaled = x / scale
            x_clamped = tl.clamp(x_scaled, -fp8_max, fp8_max)

            # Convert to fp8, then bitcast to uint8 for storage
            x_fp8 = x_clamped.to(tl.float8e4nv)
            x_uint8 = x_fp8.to(tl.uint8, bitcast=True)

            # Store as uint8 (1 byte each)
            tl.store(token_fp8_ptr + offsets, x_uint8, mask=mask)

            # UE8M0 scale encoding: stored_value = exponent + 127 (bias)
            # During dequant: scale = 2^(stored_value - 127)
            encoded_scale = exponent + 127.0
            encoded_scale = tl.maximum(tl.minimum(encoded_scale, 255.0), 0.0)
            tl.store(token_scale_ptr + qblock_idx, encoded_scale.to(tl.uint8))

    # Padding scale at index 7
    tl.store(token_scale_ptr + 7, tl.zeros((), dtype=tl.uint8))

    # ========== Store BF16 portion (last 64 elements, no quantization) ==========
    bf16_input_offset = fp8_dim

    # Process bf16 in chunks of 16
    bf16_out_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))
    for i in tl.static_range(bf16_dim // 16):
        chunk_offsets = i * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(input_row_ptr + bf16_input_offset + chunk_offsets)
        tl.store(bf16_out_ptr + chunk_offsets, bf16_vals)


def quantize_and_insert_k_cache(
    k: torch.Tensor,  # [num_tokens, 512] bf16
    k_cache: torch.Tensor,  # [num_blocks, block_bytes] uint8
    slot_mapping: torch.Tensor,  # [num_tokens] int64
    block_size: int = 64,
    is_ue8m0: bool = True,
):
    """
    Quantize K tensor and insert into paged K cache.

    K Cache block layout (block_size=64 tokens):
    - First 64 * 576 = 36864 bytes: Token data
      - Each token: 448 bytes (fp8) + 128 bytes (bf16)
    - Next 64 * 8 = 512 bytes: Scales
      - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)
    - Padded to multiple of 576
    """
    assert k.dim() == 2 and k.shape[1] == 512, (
        f"K must be [num_tokens, 512], got {k.shape}"
    )
    assert k.dtype == torch.bfloat16, f"K must be bf16, got {k.dtype}"
    assert is_ue8m0, "Only support ue8m0 quantization."

    # NOTE: When using DP, slot_mapping.shape[0] can be less than k.shape[0] due to
    # padding. Always use slot_mapping.shape[0] as the token count.
    num_tokens = slot_mapping.shape[0]
    block_stride = k_cache.stride(0)  # bytes per block

    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    FP8_MAX = 448.0
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    grid = (num_tokens,)

    quantize_and_insert_k_kernel[grid](
        k,
        slot_mapping,
        k_cache,
        num_tokens,
        input_dim=512,
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=block_stride,
        fp8_max=FP8_MAX,
        n_quant_blocks=8,
    )


@triton.jit
def _dequantize_and_gather_k_kernel(
    out_ptr,
    out_stride0,
    out_stride1,
    k_cache_ptr,
    seq_lens_ptr,
    block_table_ptr,
    offset,
    gather_lens_ptr,
    # Constants
    max_blocks_per_seq: tl.constexpr,
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64 (quantization block size)
    cache_block_size: tl.constexpr,  # 64 or 128 (paged cache block size)
    token_data_size: tl.constexpr,  # 576 bytes per token data
    block_stride: tl.constexpr,  # total bytes per block (padded) int32
    output_dim: tl.constexpr,  # 512
    fp8_max: tl.constexpr,
    n_quant_blocks: tl.constexpr,  # 7 real blocks
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    seq_len = tl.load(seq_lens_ptr + batch_idx)
    if gather_lens_ptr is not None:  # noqa: SIM108
        gather_len = tl.load(gather_lens_ptr + batch_idx)
    else:
        # Gather all tokens
        gather_len = seq_len
    start_pos = seq_len - gather_len

    for i in range(worker_id, gather_len, num_workers):
        # Calculate the actual token index in the sequence
        pos = start_pos + i

        # Calculate which block and position within block
        block_in_seq = pos // cache_block_size
        pos_in_block = pos % cache_block_size

        # Get physical block index from block table
        block_table_row_ptr = block_table_ptr + batch_idx * max_blocks_per_seq
        physical_block_idx = tl.load(block_table_row_ptr + block_in_seq)  # int32

        # int64: physical_block_idx * block_stride can exceed 2^31 with many
        # KV-cache blocks (e.g. >= 57K at block_stride ~37K).
        cache_block_ptr = k_cache_ptr + physical_block_idx.to(tl.int64) * block_stride

        # Token data pointer
        token_data_ptr = cache_block_ptr + pos_in_block * token_data_size

        # Scale pointer: after all token data
        token_scale_ptr = (
            cache_block_ptr
            + cache_block_size * token_data_size
            + pos_in_block * scale_dim
        )

        # Token data layout: [0:448] fp8, [448:576] bf16
        token_fp8_ptr = token_data_ptr
        token_bf16_ptr = token_data_ptr + fp8_dim

        # Output pointer for this token (flattened)
        output_row_ptr = out_ptr + batch_idx * out_stride0 + (offset + i) * out_stride1

        # ========== Dequantize FP8 portion using UE8M0 ==========
        for qblock_idx in tl.static_range(n_quant_blocks):
            qblock_start = qblock_idx * quant_block

            if qblock_start < fp8_dim:
                offsets = qblock_start + tl.arange(0, quant_block)
                mask = offsets < fp8_dim

                # Load quantized fp8 values (stored as uint8)
                x_uint8 = tl.load(token_fp8_ptr + offsets, mask=mask, other=0)

                # Bitcast uint8 back to fp8
                x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)

                # Convert fp8 to float32 for computation
                x_float = x_fp8.to(tl.float32)

                # Load and decode UE8M0 scale
                # UE8M0: scale = 2^(stored_value - 127)
                encoded_scale = tl.load(token_scale_ptr + qblock_idx)
                exponent = encoded_scale.to(tl.float32) - 127.0
                scale = tl.exp2(exponent)

                # Dequantize: bf16_value = fp8_value * scale
                x_dequant = x_float * scale

                # Store as bf16
                tl.store(output_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

        # ========== Copy BF16 portion directly ==========
        bf16_output_offset = fp8_dim  # After 448 elements in output

        # Read bf16 from cache
        bf16_cache_ptr = token_bf16_ptr.to(tl.pointer_type(tl.bfloat16))

        # Process in chunks of 16
        for j in tl.static_range(bf16_dim // 16):
            chunk_offsets = j * 16 + tl.arange(0, 16)
            bf16_vals = tl.load(bf16_cache_ptr + chunk_offsets)
            tl.store(output_row_ptr + bf16_output_offset + chunk_offsets, bf16_vals)


def dequantize_and_gather_k_cache(
    # [num_reqs, max_num_tokens, head_size]
    out: torch.Tensor,
    # [num_blocks, block_size, head_bytes]
    k_cache: torch.Tensor,
    # [num_reqs]
    seq_lens: torch.Tensor,
    # [num_reqs]
    gather_lens: torch.Tensor | None,
    # [num_reqs, max_blocks_per_seq]
    block_table: torch.Tensor,
    block_size: int,
    offset: int,
) -> None:
    TOKEN_FP8_DIM = 448
    TOKEN_BF16_DIM = 64
    TOKEN_SCALE_DIM = 8
    QUANT_BLOCK_SIZE = 64
    FP8_MAX = 448.0
    TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2

    num_reqs = seq_lens.shape[0]
    NUM_WORKERS = 128
    _dequantize_and_gather_k_kernel[(num_reqs, NUM_WORKERS)](
        out,
        out.stride(0),
        out.stride(1),
        k_cache,
        seq_lens,
        block_table,
        offset,
        gather_lens,
        max_blocks_per_seq=block_table.shape[-1],
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        cache_block_size=block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=k_cache.stride(0),
        output_dim=512,
        fp8_max=FP8_MAX,
        n_quant_blocks=7,
    )


def compute_global_topk_indices_and_lens(
    topk_indices: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    is_valid_token: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map local topk indices to global KV cache slots and count valid entries.

    Fuses three operations into a single kernel:
    1. Block-table lookup (local index → global slot id)
    2. Valid-entry counting (topk_lens per token)
    3. Masking padding tokens to length 0
    """
    num_tokens = topk_indices.shape[0]
    global_topk_indices = torch.empty_like(topk_indices)
    topk_lens = torch.empty(num_tokens, dtype=torch.int32, device=topk_indices.device)
    _compute_global_topk_indices_and_lens_kernel[(num_tokens,)](
        global_topk_indices,
        global_topk_indices.stride(0),
        topk_lens,
        topk_indices,
        topk_indices.stride(0),
        topk_indices.shape[-1],
        token_to_req_indices,
        block_table,
        block_table.stride(0),
        block_size,
        is_valid_token,
        TRITON_BLOCK_SIZE=1024,
    )
    return global_topk_indices, topk_lens


@triton.jit
def _compute_global_topk_indices_and_lens_kernel(
    global_topk_indices_ptr,
    global_topk_indices_stride,
    topk_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    topk,
    token_to_req_indices_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    is_valid_token_ptr,
    TRITON_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    is_valid_token = tl.load(is_valid_token_ptr + token_idx)
    req_idx = tl.load(token_to_req_indices_ptr + token_idx)

    count = tl.zeros((), dtype=tl.int32)
    for i in range(0, topk, TRITON_BLOCK_SIZE):
        offset = i + tl.arange(0, TRITON_BLOCK_SIZE)
        mask = offset < topk

        local_idx = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
            other=-1,
        )
        is_valid = local_idx >= 0

        block_indices = local_idx // block_size
        block_numbers = tl.load(
            block_table_ptr + req_idx * block_table_stride + block_indices,
            mask=mask & is_valid,
        )
        block_offsets = local_idx % block_size

        slot_ids = block_numbers * block_size + block_offsets
        slot_ids = tl.where(is_valid, slot_ids, -1)
        tl.store(
            global_topk_indices_ptr + token_idx * global_topk_indices_stride + offset,
            slot_ids,
            mask=mask,
        )
        count += tl.sum(is_valid.to(tl.int32), axis=0)

    # Zero out length for padding tokens.
    tl.store(topk_lens_ptr + token_idx, tl.where(is_valid_token, count, 0))


# FlashMLA sparse prefill asserts `params.topk % B_TOPK == 0` (see
# flashmla/csrc/sm100/prefill/sparse/fwd/head{64,128}/phase1.cuh). B_TOPK is
# 64 for the h_q=64 kernel and 128 for h_q=128; pad to 128 to satisfy both.
# The extra slots stay as -1 sentinels and `combined_lens` caps the valid
# range via `topk_length`, so padding is a no-op at kernel level.
_SPARSE_PREFILL_TOPK_ALIGNMENT = 128


def combine_topk_swa_indices(
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    gather_lens: torch.Tensor,
    window_size: int,
    compress_ratio: int,
    topk: int,
    M: int,
    N: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens = topk_indices.shape[0]
    num_reqs = seq_lens.shape[0]
    combined_topk = (
        (topk + window_size + _SPARSE_PREFILL_TOPK_ALIGNMENT - 1)
        // _SPARSE_PREFILL_TOPK_ALIGNMENT
        * _SPARSE_PREFILL_TOPK_ALIGNMENT
    )
    combined_indices = torch.full(
        (num_tokens, combined_topk),
        fill_value=-1,
        dtype=torch.int32,
        device=topk_indices.device,
    )
    combined_lens = torch.empty(
        num_tokens, dtype=torch.int32, device=topk_indices.device
    )

    NUM_WORKERS = 128
    _combine_topk_swa_indices_kernel[(num_reqs, NUM_WORKERS)](
        combined_indices,
        combined_indices.stride(0),
        combined_lens,
        topk_indices,
        topk_indices.stride(0),
        query_start_loc,
        seq_lens,
        gather_lens,
        M,
        N,
        TOP_K=topk,
        COMPRESS_RATIO=compress_ratio,
        WINDOW_SIZE=window_size,
        PADDED_TOP_K=triton.next_power_of_2(topk_indices.shape[-1]),
    )
    return combined_indices, combined_lens


@triton.jit
def _combine_topk_swa_indices_kernel(
    combined_indices_ptr,
    combined_indices_stride,
    combined_lens_ptr,
    topk_indices_ptr,
    topk_indices_stride,
    query_start_loc_ptr,
    seq_lens_ptr,
    gather_lens_ptr,
    M,
    N,
    TOP_K: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    WINDOW_SIZE: tl.constexpr,
    PADDED_TOP_K: tl.constexpr,
):
    batch_idx = tl.program_id(0)
    worker_id = tl.program_id(1)
    num_workers = tl.num_programs(1)

    # query_start_loc is a global tensor; rebase to chunk-local offsets
    # by subtracting the chunk's starting value.
    base = tl.load(query_start_loc_ptr)
    query_start = tl.load(query_start_loc_ptr + batch_idx) - base
    query_end = tl.load(query_start_loc_ptr + batch_idx + 1) - base
    query_len = query_end - query_start
    seq_len = tl.load(seq_lens_ptr + batch_idx)
    gather_len = tl.load(gather_lens_ptr + batch_idx)
    start_pos = seq_len - query_len
    # The SWA portion of the gathered buffer starts from position
    # (seq_len - gather_len), not position 0. We need this offset
    # to correctly index into the gathered buffer.
    gather_start = seq_len - gather_len

    for token_idx in range(query_start + worker_id, query_end, num_workers):
        # topk_len is fully determined by the query token's absolute position:
        # both the C4A indexer and the C128A metadata builder emit
        # min((pos + 1) // compress_ratio, topk_tokens) valid entries.
        # Caller passes TOP_K=0 for SWA-only layers to zero this out.
        token_idx_in_query = token_idx - query_start
        pos = start_pos + token_idx_in_query
        topk_len = tl.minimum((pos + 1) // COMPRESS_RATIO, TOP_K)
        swa_len = tl.minimum(pos + 1, WINDOW_SIZE)

        offset = tl.arange(0, PADDED_TOP_K)
        mask = offset < topk_len
        topk_indices = tl.load(
            topk_indices_ptr + token_idx * topk_indices_stride + offset,
            mask=mask,
        )
        tl.store(
            combined_indices_ptr + token_idx * combined_indices_stride + offset,
            topk_indices + M * batch_idx,
            mask=mask,
        )
        offset = tl.arange(0, WINDOW_SIZE)
        # Index into gathered buffer: N + (position - gather_start)
        # For positions [pos - swa_len + 1, pos], the buffer indices are:
        # [N + pos - swa_len + 1 - gather_start, N + pos - gather_start]
        tl.store(
            combined_indices_ptr
            + token_idx * combined_indices_stride
            + topk_len
            + offset,
            M * batch_idx + N + offset + pos - swa_len + 1 - gather_start,
            mask=offset < swa_len,
        )

        combined_len = topk_len + swa_len
        tl.store(combined_lens_ptr + token_idx, combined_len)
