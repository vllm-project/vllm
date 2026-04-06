# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementation of concat_and_cache_mla_kernel.

Concatenates compressed KV (kv_c) and positional-encoding keys (k_pe)
into a packed MLA KV cache, with optional FP8 quantization.

Cache layout per slot: [kv_lora_rank | pe_dim] elements, contiguous.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _concat_and_cache_mla_kernel(
    # Pointers
    kv_c_ptr,  # [num_tokens, kv_lora_rank]
    k_pe_ptr,  # [num_tokens, pe_dim]
    kv_cache_ptr,  # [num_blocks, block_size, entry_stride]
    slot_mapping_ptr,  # [num_tokens], int64
    scale_ptr,  # [1], float32
    # Strides (in elements, not bytes)
    kv_c_stride,
    k_pe_stride,
    block_stride,
    entry_stride,
    # Dimensions
    kv_lora_rank: tl.constexpr,
    pe_dim: tl.constexpr,
    block_size,
    # Mode
    QUANTIZE_FP8: tl.constexpr,
    KV_C_BLOCK: tl.constexpr,  # next_power_of_2(kv_lora_rank)
    PE_BLOCK: tl.constexpr,  # next_power_of_2(pe_dim)
):
    """Copy one token's kv_c and k_pe into the KV cache.

    Grid: (num_tokens,)
    """
    token_idx = tl.program_id(0)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    # slot_idx == -1 for padded tokens
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    dst_base = block_idx * block_stride + block_offset * entry_stride

    # -- Copy kv_c (kv_lora_rank elements) --
    kv_c_offsets = tl.arange(0, KV_C_BLOCK)
    kv_c_mask = kv_c_offsets < kv_lora_rank
    kv_c_vals = tl.load(
        kv_c_ptr + token_idx * kv_c_stride + kv_c_offsets,
        mask=kv_c_mask,
        other=0.0,
    )
    if QUANTIZE_FP8:
        scale = tl.load(scale_ptr)
        kv_c_fp8 = (kv_c_vals.to(tl.float32) / scale).to(tl.float8e4nv)
        tl.store(kv_cache_ptr + dst_base + kv_c_offsets, kv_c_fp8, mask=kv_c_mask)
    else:
        tl.store(kv_cache_ptr + dst_base + kv_c_offsets, kv_c_vals, mask=kv_c_mask)

    # -- Copy k_pe (pe_dim elements) at offset kv_lora_rank --
    pe_offsets = tl.arange(0, PE_BLOCK)
    pe_mask = pe_offsets < pe_dim
    k_pe_vals = tl.load(
        k_pe_ptr + token_idx * k_pe_stride + pe_offsets,
        mask=pe_mask,
        other=0.0,
    )
    if QUANTIZE_FP8:
        k_pe_fp8 = (k_pe_vals.to(tl.float32) / scale).to(tl.float8e4nv)
        tl.store(
            kv_cache_ptr + dst_base + kv_lora_rank + pe_offsets,
            k_pe_fp8,
            mask=pe_mask,
        )
    else:
        tl.store(
            kv_cache_ptr + dst_base + kv_lora_rank + pe_offsets,
            k_pe_vals,
            mask=pe_mask,
        )


def concat_and_cache_mla(
    kv_c: torch.Tensor,  # [num_tokens, kv_lora_rank]
    k_pe: torch.Tensor,  # [num_tokens, pe_dim]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, entry_stride]
    slot_mapping: torch.Tensor,  # [num_tokens]
    kv_cache_dtype: str,
    scale: torch.Tensor,  # [1]
) -> None:
    """Concatenate kv_c and k_pe into an MLA KV cache.

    Drop-in replacement for the CUDA ``concat_and_cache_mla`` op.

    Args:
        kv_c: Compressed KV of shape ``[num_tokens, kv_lora_rank]``.
        k_pe: Positional-encoding keys of shape ``[num_tokens, pe_dim]``.
        kv_cache: Cache of shape ``[num_blocks, block_size, entry_stride]``.
        slot_mapping: Slot index per token (``-1`` = padding).
        kv_cache_dtype: ``"auto"`` for direct copy, ``"fp8_e4m3"`` for
            FP8 quantization with the provided ``scale``.
        scale: Scalar FP8 scale factor (used only when
            ``kv_cache_dtype != "auto"``).
    """
    num_tokens = slot_mapping.shape[0]
    kv_lora_rank = kv_c.shape[1]
    pe_dim = k_pe.shape[1]
    block_size = kv_cache.shape[1]

    kv_c_stride = kv_c.stride(0)
    k_pe_stride = k_pe.stride(0)
    block_stride = kv_cache.stride(0)
    entry_stride = kv_cache.stride(1)

    quantize_fp8 = kv_cache_dtype != "auto"

    # When quantizing to FP8, view the (uint8) cache as float8_e4m3fn so
    # that Triton's store sees an fp8-typed pointer.
    if quantize_fp8 and kv_cache.dtype == torch.uint8:
        kv_cache_view = kv_cache.view(torch.float8_e4m3fn)
    else:
        kv_cache_view = kv_cache

    grid = (num_tokens,)
    _concat_and_cache_mla_kernel[grid](
        kv_c,
        k_pe,
        kv_cache_view,
        slot_mapping,
        scale,
        kv_c_stride,
        k_pe_stride,
        block_stride,
        entry_stride,
        kv_lora_rank=kv_lora_rank,
        pe_dim=pe_dim,
        block_size=block_size,
        QUANTIZE_FP8=quantize_fp8,
        KV_C_BLOCK=triton.next_power_of_2(kv_lora_rank),
        PE_BLOCK=triton.next_power_of_2(pe_dim),
    )
