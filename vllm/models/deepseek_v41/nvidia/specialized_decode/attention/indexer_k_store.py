# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP4 indexer key store for the FP32 output of the low-batch ``wk`` GEMM.

``indexer_k_norm_rope_store`` (``common/ops/indexer_k_store.py``) takes the
BF16 output of the ``wk`` GEMM. The low-batch ``ll_bf16`` kernel returns its
FP32 accumulation instead. This copy of the store kernel's MXFP4 path rounds
each row to BF16 as it loads it, so it writes the keys the shared kernel
writes for that row's BF16 GEMM output, without a separate cast launch.
"""

import torch

from vllm.models.deepseek_v41.common.ops import MXFP4_BLOCK_SIZE, _fp32x2_to_fp4x2
from vllm.triton_utils import tl, triton


def indexer_k_norm_rope_store_fp32(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
) -> None:
    """``indexer_k_norm_rope_store`` into an MXFP4 cache for an FP32 ``k_pre``
    ([num_tokens, 128], the FP32 ``wk(latent)``)."""
    num_tokens = kv_slot_mapping.numel()
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.float32 and k_pre.stride(1) == 1
    assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
    assert compress_ratio in (1, 2)
    if num_tokens == 0:
        return
    head_dim = k_pre.shape[1]
    _k_norm_rope_mxfp4_store_kernel[(num_tokens,)](
        k_pre,
        k_pre.stride(0),
        positions,
        rms_norm_weight,
        rms_norm_eps,
        cos_sin_cache,
        cos_sin_cache.stride(0),
        k_cache,
        kv_slot_mapping,
        k_cache.shape[1],
        HEAD_SIZE=head_dim,
        ROPE_HEAD_DIM=64,
        COMPRESS_RATIO=compress_ratio,
        TOKEN_STRIDE=head_dim // 2,
        SCALE_DIM=head_dim // MXFP4_BLOCK_SIZE,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        num_warps=1,
        launch_pdl=False,
    )


@triton.jit
def _k_norm_rope_mxfp4_store_kernel(
    k_pre_ptr,
    k_pre_stride,
    positions_ptr,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    k_cache_ptr,
    kv_slot_mapping_ptr,
    kv_cache_block_size,
    HEAD_SIZE: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    TOKEN_STRIDE: tl.constexpr,
    SCALE_DIM: tl.constexpr,
    KV_BLOCK_STRIDE: tl.constexpr,
):
    token_idx = tl.program_id(0)
    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    block = tl.arange(0, HEAD_SIZE)
    # The BF16 GEMM output, then k_norm as in the shared kernel.
    k = tl.load(k_pre_ptr + token_idx * k_pre_stride + block)
    k = k.to(tl.bfloat16).to(tl.float32)
    rms_w = tl.load(rms_norm_weight_ptr + block).to(tl.float32)
    variance = tl.sum(k * k, axis=0) / HEAD_SIZE
    k = (k * tl.rsqrt(variance + rms_norm_eps) * rms_w).to(tl.bfloat16)
    k = k.to(tl.float32)

    # GPT-J RoPE of the trailing ROPE_HEAD_DIM values at the group position.
    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_PAIRS: tl.constexpr = (HEAD_SIZE - ROPE_HEAD_DIM) // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    even, odd = tl.split(tl.reshape(k, (NUM_PAIRS, 2)))
    rope_pair = tl.arange(0, NUM_PAIRS) - NOPE_PAIRS
    is_rope_pair = rope_pair >= 0
    cs_idx = tl.maximum(rope_pair, 0)
    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)
    new_even = (even * cos_v - odd * sin_v).to(tl.bfloat16).to(tl.float32)
    new_odd = (odd * cos_v + even * sin_v).to(tl.bfloat16).to(tl.float32)

    # MXFP4 values, then one UE8M0 scale per 32 values, in the paged block.
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )
    N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // 32
    even_2d = tl.reshape(new_even, (N_QUANT_BLOCKS, 16))
    odd_2d = tl.reshape(new_odd, (N_QUANT_BLOCKS, 16))
    amax = tl.maximum(tl.max(tl.abs(even_2d), axis=1), tl.max(tl.abs(odd_2d), axis=1))
    amax = tl.maximum(amax, 6.0 * (2**-126))
    log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
    log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
    inv_scale = tl.reshape(tl.exp2(-log2_ratio), (N_QUANT_BLOCKS, 1))
    packed = _fp32x2_to_fp4x2(even_2d * inv_scale, odd_2d * inv_scale)
    tl.store(val_ptr + tl.arange(0, TOKEN_STRIDE), tl.reshape(packed, (TOKEN_STRIDE,)))
    tl.store(scale_ptr + tl.arange(0, SCALE_DIM), (log2_ratio + 127.0).to(tl.uint8))
