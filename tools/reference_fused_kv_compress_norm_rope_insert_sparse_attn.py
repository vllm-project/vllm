# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch reference for DeepSeek V4 sparse compressor KV-cache insert.

This file mirrors the logic of the Triton kernel:

    _fused_kv_compress_norm_rope_insert_sparse_attn

from:

    vllm/models/deepseek_v4/common/ops/fused_compress_quant_cache.py

It is intentionally slow and explicit. The goal is to show the dataflow:

    state cache -> softmax compression -> RMSNorm -> FP8 NoPE cache
    -> RoPE bf16 cache -> UE8M0 scale bytes

The function below updates ``k_cache`` in place using the same byte-level block
layout as the Triton kernel:

    per KV cache block:
      [0, kv_cache_block_size * 576):       token data
      [kv_cache_block_size * 576, ...):     8 scale bytes per token

Each token's 576 data bytes are:
      448 bytes FP8 NoPE + 128 bytes bf16 RoPE

This is a reference/inspection tool, not production code.
"""

from __future__ import annotations

import torch


def _fp8_e4m3fn_to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Quantize to torch float8_e4m3fn and return the raw bytes."""
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("This reference requires torch.float8_e4m3fn support.")
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


@torch.no_grad()
def fused_kv_compress_norm_rope_insert_sparse_attn_torch(
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    kv_cache_block_size: int,
    *,
    head_size: int = 512,
    state_width: int | None = None,
    compress_ratio: int,
    overlap: bool,
    rope_head_dim: int = 64,
    fp8_max: float = 448.0,
    quant_block: int = 64,
    token_stride: int = 576,
    scale_dim: int = 8,
) -> torch.Tensor:
    """Reference implementation of the sparse DeepSeek V4 compressor kernel.

    Args:
        state_cache:
            Compressor state cache. Logical shape is typically
            ``[num_state_blocks, state_block_size, 2 * state_width]``.
            The first ``state_width`` values are KV state; the second
            ``state_width`` values are score state.
        token_to_req_indices:
            ``[num_tokens]`` request index for each scheduled token.
        positions:
            ``[num_tokens]`` absolute token positions.
        slot_mapping:
            ``[num_tokens]`` slot mapping for the compressor state cache.
            Negative slots are padding and are skipped.
        block_table:
            ``[num_reqs, max_blocks_per_req]`` physical state-cache blocks.
        block_size:
            Compressor state-cache block size, e.g. 4 for C4 or 8 for C128.
        rms_norm_weight:
            ``[head_size]`` RMSNorm weight.
        cos_sin_cache:
            ``[max_position, rope_head_dim]``. First half is cos, second half
            is sin.
        k_cache:
            uint8 KV cache tensor updated in place. It can be the actual vLLM
            logical tensor; this reference writes through a byte-level strided
            view to match the Triton pointer arithmetic.
        kv_slot_mapping:
            ``[num_tokens]`` slot mapping for the destination compressed KV
            cache. Negative slots are skipped.
        kv_cache_block_size:
            Tokens per compressed KV cache block.
        head_size:
            DeepSeek V4 main MLA head size, normally 512.
        state_width:
            Width of one half of ``state_cache``. If omitted, inferred from
            ``state_cache.shape[-1] // 2``.
        compress_ratio:
            Usually 4 for C4 layers or 128 for C128 layers.
        overlap:
            True for C4 layers, False for C128 layers.
        rope_head_dim:
            RoPE portion of the head, normally 64.
        fp8_max:
            Maximum finite E4M3 value used by the Triton kernel.
        quant_block:
            FP8 UE8M0 quantization block size, normally 64.
        token_stride:
            Bytes in the data area per token: 448 FP8 + 64 bf16 * 2 = 576.
        scale_dim:
            Scale bytes per token: 7 UE8M0 scale bytes + 1 pad byte = 8.

    Returns:
        The same ``k_cache`` tensor, after in-place update.
    """
    if state_width is None:
        state_width = state_cache.shape[-1] // 2

    if k_cache.dtype != torch.uint8:
        raise TypeError(f"k_cache must be uint8, got {k_cache.dtype}")
    if head_size % 2 != 0:
        raise ValueError("head_size must be even for GPT-J style RoPE pairs")
    if rope_head_dim % 2 != 0:
        raise ValueError("rope_head_dim must be even")
    if head_size % quant_block != 0:
        raise ValueError("head_size must be divisible by quant_block")

    device = state_cache.device
    num_tokens = positions.numel()
    nope_head_dim = head_size - rope_head_dim
    half_rope = rope_head_dim // 2
    num_pairs = head_size // 2
    nope_pairs = nope_head_dim // 2
    n_quant_blocks = head_size // quant_block
    n_nope_blocks = nope_head_dim // quant_block
    window = (1 + int(overlap)) * compress_ratio

    # Make a byte-level block view that follows the same base + stride(0)
    # arithmetic used by the Triton kernel.
    if k_cache.dim() < 2:
        raise ValueError("k_cache must expose a block dimension")
    kv_block_stride = k_cache.stride(0)
    k_cache_bytes_by_block = k_cache.as_strided(
        size=(k_cache.shape[0], kv_block_stride),
        stride=(kv_block_stride, 1),
    )

    rms_w = rms_norm_weight[:head_size].to(device=device, dtype=torch.float32)
    pair_idx = torch.arange(num_pairs, device=device)
    rope_pair_local = pair_idx - nope_pairs
    is_rope_pair = rope_pair_local >= 0
    cs_idx = rope_pair_local.clamp_min(0)

    for token_idx in range(num_tokens):
        slot_id = int(slot_mapping[token_idx].item())
        if slot_id < 0:
            continue

        position = int(positions[token_idx].item())
        if (position + 1) % compress_ratio != 0:
            continue

        req_idx = int(token_to_req_indices[token_idx].item())

        kv_rows = []
        score_rows = []
        start = position - window + 1

        for token_offset in range(window):
            pos = start + token_offset
            if pos < 0:
                kv_rows.append(torch.zeros(head_size, device=device))
                score_rows.append(
                    torch.full((head_size,), -float("inf"), device=device)
                )
                continue

            logical_block_idx = pos // block_size
            pos_in_block = pos % block_size
            physical_block = int(block_table[req_idx, logical_block_idx].item())

            # For C4, the second half of the state row stores the overlapped
            # 512-dim segment. For C128, overlap is false and this stays zero.
            head_offset = head_size if token_offset >= compress_ratio else 0
            row = state_cache[physical_block, pos_in_block]

            kv_rows.append(row[head_offset : head_offset + head_size].to(torch.float32))
            score_rows.append(
                row[
                    state_width + head_offset : state_width + head_offset + head_size
                ].to(torch.float32)
            )

        kv = torch.stack(kv_rows, dim=0)  # [window, head_size]
        score = torch.stack(score_rows, dim=0)  # [window, head_size]

        # Softmax across the compressor window, independently for each head dim.
        score = torch.softmax(score, dim=0)
        compressed_kv = (kv * score).sum(dim=0)  # [head_size]

        # RMSNorm in fp32.
        variance = (compressed_kv * compressed_kv).sum() / head_size
        normed = compressed_kv * torch.rsqrt(variance + rms_norm_eps) * rms_w

        kv_slot_idx = int(kv_slot_mapping[token_idx].item())
        if kv_slot_idx < 0:
            continue
        kv_block_idx = kv_slot_idx // kv_cache_block_size
        kv_pos_in_block = kv_slot_idx % kv_cache_block_size
        block_bytes = k_cache_bytes_by_block[kv_block_idx]

        data_offset = kv_pos_in_block * token_stride
        scale_offset = kv_cache_block_size * token_stride + kv_pos_in_block * scale_dim

        # Match Triton: fp32 -> bf16 -> fp32 before per-block UE8M0 quant.
        quant_input = normed.to(torch.bfloat16).to(torch.float32)
        quant_2d = quant_input.reshape(n_quant_blocks, quant_block)
        block_absmax = quant_2d.abs().amax(dim=1).clamp_min(1e-4)

        raw_scales = block_absmax / fp8_max
        exponents = torch.ceil(torch.log2(raw_scales))
        inv_scales = torch.exp2(-exponents)
        x_scaled = quant_2d * inv_scales[:, None]
        x_clamped = x_scaled.clamp(-fp8_max, fp8_max)
        x_uint8_flat = _fp8_e4m3fn_to_uint8(x_clamped).reshape(head_size)

        # Store the 448 NoPE FP8 bytes.
        block_bytes[data_offset : data_offset + nope_head_dim].copy_(
            x_uint8_flat[:nope_head_dim]
        )

        # Store 7 UE8M0 scale bytes plus one zero pad byte.
        encoded = (exponents + 127.0).clamp(0.0, 255.0).to(torch.uint8)
        block_bytes[scale_offset : scale_offset + n_nope_blocks].copy_(
            encoded[:n_nope_blocks]
        )
        block_bytes[scale_offset + n_nope_blocks] = 0

        # GPT-J/interleaved RoPE on the last 64 dimensions.
        pairs = normed.reshape(num_pairs, 2)
        even = pairs[:, 0]
        odd = pairs[:, 1]

        compressed_pos = (position // compress_ratio) * compress_ratio
        cos_sin = cos_sin_cache[compressed_pos].to(torch.float32)

        cos_v = torch.ones(num_pairs, device=device, dtype=torch.float32)
        sin_v = torch.zeros(num_pairs, device=device, dtype=torch.float32)
        cos_v[is_rope_pair] = cos_sin[cs_idx[is_rope_pair]]
        sin_v[is_rope_pair] = cos_sin[half_rope + cs_idx[is_rope_pair]]

        new_even = even * cos_v - odd * sin_v
        new_odd = odd * cos_v + even * sin_v
        result = torch.stack((new_even, new_odd), dim=1).reshape(head_size)

        rope_bytes = result[nope_head_dim:].to(torch.bfloat16).view(torch.uint8)
        rope_offset = data_offset + nope_head_dim
        block_bytes[rope_offset : rope_offset + rope_head_dim * 2].copy_(rope_bytes)

    return k_cache
