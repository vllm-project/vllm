# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton implementation of indexer_k_quant_and_cache_kernel.

Quantizes key vectors to FP8 with one scale per token and writes them
(along with the float32 scale) into a packed KV cache tensor.

Specialized for head_dim == quant_block_size == 128.

Cache layout (per cache block, flattened):
  [block_size × 128] bytes of FP8 data
  [block_size × 4]   bytes of float32 scales (one per token)
"""

import torch
import triton
import triton.language as tl

# head_dim is always 128 for the indexer kernel.
HEAD_DIM = tl.constexpr(128)


@triton.jit
def _indexer_k_quant_and_cache_kernel(
    k_ptr,  # [num_tokens, 128], bf16/fp16
    kv_cache_ptr,  # [num_blocks, block_size, 132], fp8
    kv_cache_scale_ptr,  # same memory as float32
    slot_mapping_ptr,  # [num_tokens], int64
    cache_block_size,
    cache_stride,
    USE_UE8M0: tl.constexpr,
):
    """Quantize one token's 128-element key vector to FP8.

    Grid: (num_tokens,)
    """
    token_idx = tl.program_id(0)

    slot_idx = tl.load(slot_mapping_ptr + token_idx)
    if slot_idx < 0:
        return

    block_idx = slot_idx // cache_block_size
    block_offset = slot_idx % cache_block_size
    block_start = block_idx * cache_block_size * cache_stride

    # -- Load 128 elements of k --
    offsets = tl.arange(0, HEAD_DIM)
    k_vals = tl.load(k_ptr + token_idx * HEAD_DIM + offsets).to(tl.float32)

    # -- Compute scale --
    amax = tl.max(tl.abs(k_vals))
    scale = tl.div_rn(tl.maximum(amax, 1e-4), 448.0)

    if USE_UE8M0:
        scale = tl.math.exp2(tl.math.ceil(tl.math.log2(scale)))

    # -- Quantize and store FP8 data --
    k_fp8 = tl.div_rn(k_vals, scale).to(tl.float8e4nv)
    tl.store(
        kv_cache_ptr + block_start + block_offset * HEAD_DIM + offsets,
        k_fp8,
    )

    # -- Store float32 scale (one per token, right after the FP8 data) --
    scale_byte_offset = block_start + cache_block_size * HEAD_DIM + block_offset * 4
    tl.store(kv_cache_scale_ptr + scale_byte_offset // 4, scale)


def indexer_k_quant_and_cache(
    k: torch.Tensor,  # [num_tokens, 128]
    kv_cache: torch.Tensor,  # [num_blocks, block_size, 132]
    slot_mapping: torch.Tensor,  # [num_tokens]
    quant_block_size: int,
    scale_fmt: str,
) -> None:
    """Quantize key vectors to FP8 and write into a packed KV cache.

    Drop-in replacement for the CUDA ``indexer_k_quant_and_cache`` op.
    Specialized for head_dim == quant_block_size == 128.
    """
    num_tokens = k.shape[0]
    assert k.shape[1] == 128
    assert quant_block_size == 128

    kv_cache_scale_view = kv_cache.view(torch.uint8).view(torch.float32)

    _indexer_k_quant_and_cache_kernel[(num_tokens,)](
        k,
        kv_cache,
        kv_cache_scale_view,
        slot_mapping,
        cache_block_size=kv_cache.shape[1],
        cache_stride=kv_cache.shape[2],
        USE_UE8M0=(scale_fmt == "ue8m0"),
    )
