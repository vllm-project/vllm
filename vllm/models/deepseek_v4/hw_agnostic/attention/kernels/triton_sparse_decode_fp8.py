# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA decode: dequantize FP8 cache to BF16, then BF16 sparse MLA."""

import torch

from vllm.models.deepseek_v4.hw_agnostic.attention.kernels.triton_mla_sparse import (
    triton_bf16_mla_sparse_interface,
)
from vllm.triton_utils import tl, triton

# FP8 DS MLA cache layout constants
TOKEN_FP8_DIM = 448  # NoPE portion in FP8
TOKEN_BF16_DIM = 64  # RoPE portion in BF16
TOKEN_SCALE_DIM = 8  # UE8M0 scales per token
QUANT_BLOCK_SIZE = 64  # Elements per quant block
OUTPUT_DIM = 512  # = TOKEN_FP8_DIM + TOKEN_BF16_DIM after dequant
TOKEN_DATA_SIZE = TOKEN_FP8_DIM + TOKEN_BF16_DIM * 2  # 576 bytes per token


@triton.jit
def _dequant_gather_slots_kernel(
    # Output workspace: [total_slots, OUTPUT_DIM] bf16
    out_ptr,
    # FP8 paged cache base pointer (uint8 flat)
    cache_ptr,
    # Global slot indices: [total_slots] int32
    indices_ptr,
    # Cache geometry
    cache_block_size: tl.constexpr,
    token_data_size: tl.constexpr,  # 576
    block_stride: tl.int64,  # k_cache.stride(0) — total uint8 per block
    fp8_dim: tl.constexpr,  # 448
    bf16_dim: tl.constexpr,  # 64
    scale_dim: tl.constexpr,  # 8
    quant_block: tl.constexpr,  # 64
    output_dim: tl.constexpr,  # 512
    n_quant_blocks: tl.constexpr,  # 7
):
    """Dequantize scattered FP8 slots into a flat BF16 workspace.

    Grid: ``[total_slots]`` — one program per slot to gather.

    Cache block layout (block_size tokens):
      [0, block_size*576): Token data, each token 448 FP8 + 128 BF16
      [block_size*576, block_size*576 + block_size*8): Scales
    """
    pid = tl.program_id(0)

    # Load global slot index
    slot_idx = tl.load(indices_ptr + pid).to(tl.int64)

    # Output pointer for this slot
    out_row_ptr = out_ptr + pid * output_dim

    # Handle invalid slots (index < 0): write zeros
    if slot_idx < 0:
        zero = tl.zeros([quant_block], dtype=tl.bfloat16)
        for i in tl.static_range(0, 512, 64):
            offsets = i + tl.arange(0, quant_block)
            mask = offsets < output_dim
            tl.store(out_row_ptr + offsets, zero, mask=mask)
        return

    # Compute block and position within block
    block_idx = slot_idx // cache_block_size
    pos_in_block = slot_idx % cache_block_size

    # Block base pointer
    block_base = cache_ptr + block_idx * block_stride

    # Token data: at offset pos_in_block * token_data_size within block
    token_data_ptr = block_base + pos_in_block * token_data_size

    # Scale: after all token data, at offset
    # cache_block_size * token_data_size + pos_in_block * scale_dim
    scale_region_offset = tl.cast(cache_block_size, tl.int64) * token_data_size
    token_scale_ptr = block_base + scale_region_offset + pos_in_block * scale_dim

    # ========== Dequantize FP8 portion (448 elements) ==========
    for qblock_idx in tl.static_range(n_quant_blocks):
        qblock_start = qblock_idx * quant_block
        offsets = qblock_start + tl.arange(0, quant_block)
        mask = offsets < fp8_dim

        # Load FP8 as uint8 and bitcast
        x_uint8 = tl.load(token_data_ptr + offsets, mask=mask, other=0)
        x_fp8 = x_uint8.to(tl.float8e4nv, bitcast=True)
        x_float = x_fp8.to(tl.float32)

        # Load UE8M0 scale: scale = 2^(stored_value - 127)
        encoded_scale = tl.load(token_scale_ptr + qblock_idx)
        exponent = encoded_scale.to(tl.float32) - 127.0
        scale = tl.exp2(exponent)

        # Dequantize and store as bf16
        x_dequant = x_float * scale
        tl.store(out_row_ptr + offsets, x_dequant.to(tl.bfloat16), mask=mask)

    # ========== Copy BF16 portion (64 elements) directly ==========
    bf16_src_ptr = (token_data_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))
    bf16_out_ptr = (out_row_ptr + fp8_dim).to(tl.pointer_type(tl.bfloat16))

    for j in tl.static_range(bf16_dim // 16):
        chunk_offsets = j * 16 + tl.arange(0, 16)
        bf16_vals = tl.load(bf16_src_ptr + chunk_offsets)
        tl.store(bf16_out_ptr + chunk_offsets, bf16_vals)


def dequant_gather_slots(
    out: torch.Tensor,  # [total_slots, 512] bf16, pre-allocated
    cache: torch.Tensor,  # [num_blocks, block_size, head_bytes] uint8
    indices: torch.Tensor,  # [total_slots] int32, global slot IDs
    cache_block_size: int,  # block_size for this cache
) -> None:
    """Dequantize FP8 UE8M0 pages at scattered slot indices into bf16."""
    total_slots = indices.shape[0]
    if total_slots == 0:
        return

    block_stride = cache.stride(0)

    _dequant_gather_slots_kernel[(total_slots,)](
        out,
        cache,
        indices,
        cache_block_size=cache_block_size,
        token_data_size=TOKEN_DATA_SIZE,
        block_stride=block_stride,
        fp8_dim=TOKEN_FP8_DIM,
        bf16_dim=TOKEN_BF16_DIM,
        scale_dim=TOKEN_SCALE_DIM,
        quant_block=QUANT_BLOCK_SIZE,
        output_dim=OUTPUT_DIM,
        n_quant_blocks=7,
    )


def triton_sparse_decode_fp8(
    q: torch.Tensor,  # [num_tokens, num_heads, head_dim]
    kv_cache: torch.Tensor | None,  # [num_blocks, block_size, head_bytes] uint8
    swa_kv_cache: torch.Tensor,  # [num_blocks, swa_block_size, head_bytes] uint8
    swa_only: bool,
    topk_indices: torch.Tensor | None,  # [num_tokens, 1, topk] global slot IDs
    topk_lens: torch.Tensor | None,
    swa_indices: torch.Tensor,  # [num_tokens, 1, swa_k] global slot IDs
    swa_lens: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    head_dim: int,
    nope_head_dim: int,
    rope_head_dim: int,
    out: torch.Tensor,  # [num_tokens, num_heads, head_dim]
) -> None:
    num_tokens = q.shape[0]
    device = q.device

    # Determine max topk and swa widths
    if not swa_only and topk_indices is not None:
        topk_idx_2d = (
            topk_indices.squeeze(1) if topk_indices.dim() == 3 else topk_indices
        )
        max_topk = topk_idx_2d.shape[1]
    else:
        topk_idx_2d = None
        max_topk = 0

    swa_idx_2d = swa_indices.squeeze(1) if swa_indices.dim() == 3 else swa_indices
    max_swa = swa_idx_2d.shape[1]

    K_total = max_topk + max_swa

    # Allocate flat workspace: [num_tokens * K_total, 512] bf16
    workspace = torch.empty(
        (num_tokens * K_total, OUTPUT_DIM), dtype=torch.bfloat16, device=device
    )
    ws_3d = workspace.view(num_tokens, K_total, OUTPUT_DIM)

    # Dequant+gather topk slots from compressed cache
    if not swa_only and topk_idx_2d is not None and kv_cache is not None:
        topk_flat = topk_idx_2d.reshape(-1).to(torch.int32)
        topk_buf = torch.empty(
            (num_tokens * max_topk, OUTPUT_DIM), dtype=torch.bfloat16, device=device
        )
        compressed_block_size = kv_cache.shape[1]
        dequant_gather_slots(topk_buf, kv_cache, topk_flat, compressed_block_size)
        ws_3d[:, :max_topk, :] = topk_buf.view(num_tokens, max_topk, OUTPUT_DIM)

    # Dequant+gather SWA slots
    swa_flat = swa_idx_2d.reshape(-1).to(torch.int32)
    swa_buf = torch.empty(
        (num_tokens * max_swa, OUTPUT_DIM), dtype=torch.bfloat16, device=device
    )
    swa_block_size = swa_kv_cache.shape[1]
    dequant_gather_slots(swa_buf, swa_kv_cache, swa_flat, swa_block_size)

    ws_3d[:, max_topk:, :] = swa_buf.view(num_tokens, max_swa, OUTPUT_DIM)

    # Pack combined indices contiguously: [valid_topk..., valid_swa..., -1 padding].
    # Use a static worst-case width (K_total padded to BLOCK_N) so this path is
    # CUDA-graph safe — no host syncs (.item()) and no per-token Python loops.
    # Padded slots remain -1; the sparse-MLA kernel treats -1 as a no-op.
    _BLOCK_N = 16
    max_combined_padded = ((K_total + _BLOCK_N - 1) // _BLOCK_N) * _BLOCK_N

    combined_indices = torch.full(
        (num_tokens, max_combined_padded),
        fill_value=-1,
        dtype=torch.int32,
        device=device,
    )

    token_offsets = (
        torch.arange(num_tokens, device=device, dtype=torch.int32) * K_total
    )  # [B]
    col_range = torch.arange(
        max_combined_padded, device=device, dtype=torch.int32
    ).unsqueeze(0)  # [1, max_combined_padded]

    if not swa_only and topk_lens is not None:
        # Topk lives at columns [0, topk_lens[t]); value = t*K_total + col.
        topk_lens_col = topk_lens.to(torch.int32).unsqueeze(1)  # [B, 1]
        swa_lens_col = swa_lens.to(torch.int32).unsqueeze(1)  # [B, 1]
        topk_mask = col_range < topk_lens_col
        # SWA lives at columns [topk_lens[t], topk_lens[t] + swa_lens[t]);
        # within that range, the corresponding swa-position is col - topk_lens[t],
        # so the workspace index is t*K_total + max_topk + (col - topk_lens[t]).
        swa_pos = col_range - topk_lens_col  # [B, max_combined_padded]
        swa_mask = (col_range >= topk_lens_col) & (swa_pos < swa_lens_col)
        topk_vals = token_offsets.unsqueeze(1) + col_range
        swa_vals = token_offsets.unsqueeze(1) + max_topk + swa_pos
        combined_indices = torch.where(topk_mask, topk_vals, combined_indices)
        combined_indices = torch.where(swa_mask, swa_vals, combined_indices)
    else:
        # SWA-only: swa lives at columns [0, swa_lens[t]).
        swa_lens_col = swa_lens.to(torch.int32).unsqueeze(1)
        swa_mask = col_range < swa_lens_col
        swa_vals = token_offsets.unsqueeze(1) + col_range  # max_topk == 0
        combined_indices = torch.where(swa_mask, swa_vals, combined_indices)

    # Call BF16 sparse MLA kernel.
    out_attn, _, _ = triton_bf16_mla_sparse_interface(
        q=q,
        kv=workspace.unsqueeze(1),
        indices=combined_indices.unsqueeze(1),
        sm_scale=softmax_scale,
        d_v=q.shape[-1],
        block_dpe=0,
    )
    out.copy_(out_attn)
