# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Indexer K production for DeepSeek V4.1 kv-source layers.

In v4.1 the index key is derived from the *main* compressor's latent:
``k = k_norm(wk(latent))`` (reference model.py Indexer.forward), then RoPE'd
at the group's first-token position and MXFP4/FP8-quantized into the paged
indexer K cache. The ``wk(latent)`` GEMM runs in torch; this kernel fuses the
remaining k_norm → RoPE → quant → paged store, one program per token.

Unlike the legacy (v4.0) indexer path there is no per-token pooling from a
compressor state cache: the latent already stands for a whole group, so only
group-boundary tokens ``(position + 1) % compress_ratio == 0`` produce a key.
"""

from typing import TYPE_CHECKING

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

from . import MXFP4_BLOCK_SIZE, _fp32x2_to_fp4x2

if TYPE_CHECKING:
    from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
        RocmPagedMxfp4CacheLayout,
    )

# ROCm tiled ("SHUFFLE") indexer K value layout, mirroring
# indexer_k_quant_and_cache_triton's defaults: 16 positions x 16 bytes.
_BLOCK_TILE_SIZE = 16
_HEAD_TILE_SIZE = 16


def indexer_k_norm_rope_store(
    k_pre: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    k_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    compress_ratio: int,
    use_fp4_cache: bool,
    mxfp4_layout: "RocmPagedMxfp4CacheLayout | None" = None,
) -> None:
    """k_norm → RoPE → quant → paged store for indexer keys.

    Args:
        k_pre: [num_tokens, 128] bf16, the ``wk(latent)`` projection. Only
            group-boundary rows are read.
        positions: [num_tokens] int64 token positions.
        cos_sin_cache: [max_pos, rope_head_dim] GPT-J layout (cos half, then
            sin half), from the layer's compress-RoPE instance.
        rms_norm_weight: [128] k_norm weight.
        rms_norm_eps: Epsilon of the k_norm RMSNorm.
        k_cache: uint8 paged indexer cache [num_blocks, block_size, row_bytes].
        kv_slot_mapping: [num_tokens] slots in the indexer cache (-1 = skip).
        compress_ratio: group size; keys are emitted at group boundaries.
        use_fp4_cache: MXFP4 (2 nibbles/byte + ue8m0 per 32) when True, else
            per-token FP8 with a single fp32 scale.
        mxfp4_layout: ROCm MXFP4 only: the page order aiter's paged MXFP4
            MQA-logits kernel reads, from ``rocm_paged_mxfp4_cache_layout``.

    """
    num_tokens = kv_slot_mapping.numel()
    assert k_pre.ndim == 2 and k_pre.shape[1] == 128
    assert k_pre.dtype == torch.bfloat16 and k_pre.stride(1) == 1
    assert num_tokens <= k_pre.shape[0] and num_tokens <= positions.numel()
    assert compress_ratio in (1, 2)
    if num_tokens == 0:
        return

    head_dim = k_pre.shape[1]
    if use_fp4_cache:
        token_stride = head_dim // 2
        scale_dim = head_dim // MXFP4_BLOCK_SIZE
    else:
        token_stride = head_dim
        scale_dim = 4  # single float32 scale

    # ROCm reads this cache back with the 16x16-tiled ("SHUFFLE") value
    # layout: cp_gather_indexer_k_quant_cache_triton on prefill and aiter's
    # deepgemm_fp8_paged_mqa_logits(Preshuffle=True) on decode both select it
    # from ``block_size > 1``, matching the v4.0 writer
    # (indexer_k_quant_and_cache_triton). Writing row-major here would hand
    # both readers permuted key bytes. The MXFP4 cache is read straight into
    # the matrix core by aiter's paged MXFP4 MQA-logits kernel, so it is
    # stored in that kernel's dot-operand order instead.
    block_size = k_cache.shape[1]
    shuffle = current_platform.is_rocm() and block_size > 1
    if shuffle and use_fp4_cache:
        if mxfp4_layout is None or block_size % mxfp4_layout.n_per_tile != 0:
            raise ValueError(
                f"ROCm MXFP4 indexer K cache needs a page layout ({mxfp4_layout}) "
                f"whose MFMA N divides block_size ({block_size})."
            )
    elif shuffle and (
        block_size % _BLOCK_TILE_SIZE != 0 or head_dim % _HEAD_TILE_SIZE != 0
    ):
        raise ValueError(
            f"ROCm tiled indexer K cache needs block_size "
            f"({block_size}) % {_BLOCK_TILE_SIZE} == 0 and head_dim "
            f"({head_dim}) % {_HEAD_TILE_SIZE} == 0."
        )

    launch_kwargs = {"launch_pdl": False} if current_platform.is_cuda() else {}
    _indexer_k_norm_rope_quant_store_kernel[(num_tokens,)](
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
        TOKEN_STRIDE=token_stride,
        SCALE_DIM=scale_dim,
        KV_BLOCK_STRIDE=k_cache.stride(0),
        FP8_MAX=448.0,
        USE_FP4=use_fp4_cache,
        SHUFFLE=shuffle,
        BLOCK_TILE_SIZE=_BLOCK_TILE_SIZE,
        HEAD_TILE_SIZE=_HEAD_TILE_SIZE,
        N_PER_TILE=mxfp4_layout.n_per_tile if mxfp4_layout else 0,
        D_PER_TILE=mxfp4_layout.d_per_tile if mxfp4_layout else 0,
        SCALE_LANES=mxfp4_layout.scale_lanes if mxfp4_layout else 0,
        num_warps=1,
        **launch_kwargs,
    )


@triton.jit
def _indexer_k_norm_rope_quant_store_kernel(
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
    FP8_MAX: tl.constexpr,
    USE_FP4: tl.constexpr,
    SHUFFLE: tl.constexpr,
    BLOCK_TILE_SIZE: tl.constexpr,
    HEAD_TILE_SIZE: tl.constexpr,
    N_PER_TILE: tl.constexpr,
    D_PER_TILE: tl.constexpr,
    SCALE_LANES: tl.constexpr,
):
    token_idx = tl.program_id(0)

    kv_slot_idx = tl.load(kv_slot_mapping_ptr + token_idx)
    if kv_slot_idx < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    # Only the last token of a group publishes that group's index key.
    if (position + 1) % COMPRESS_RATIO != 0:
        return

    block = tl.arange(0, HEAD_SIZE)

    # ── k_norm (fp32 throughout, bf16 roundtrip like the reference) ────
    k = tl.load(k_pre_ptr + token_idx * k_pre_stride + block).to(tl.float32)
    rms_w = tl.load(rms_norm_weight_ptr + block).to(tl.float32)
    variance = tl.sum(k * k, axis=0) / HEAD_SIZE
    k = (k * tl.rsqrt(variance + rms_norm_eps) * rms_w).to(tl.bfloat16)
    k = k.to(tl.float32)

    # ── Register-based GPT-J forward RoPE in fp32 ─────────────────────
    # A latent stands for the first token of its group, so group j takes
    # position j * compress_ratio.
    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_HEAD_DIM: tl.constexpr = HEAD_SIZE - ROPE_HEAD_DIM
    NOPE_PAIRS: tl.constexpr = NOPE_HEAD_DIM // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2

    even, odd = tl.split(tl.reshape(k, (NUM_PAIRS, 2)))  # each [NUM_PAIRS]
    pair_idx = tl.arange(0, NUM_PAIRS)
    rope_pair_local = pair_idx - NOPE_PAIRS
    is_rope_pair = rope_pair_local >= 0
    cs_idx = tl.maximum(rope_pair_local, 0)

    compressed_pos = (position // COMPRESS_RATIO) * COMPRESS_RATIO
    cache_base = cos_sin_cache_ptr + compressed_pos * cos_sin_stride
    cos_v = tl.load(cache_base + cs_idx, mask=is_rope_pair, other=1.0)
    sin_v = tl.load(cache_base + HALF_ROPE + cs_idx, mask=is_rope_pair, other=0.0)

    new_even = even * cos_v - odd * sin_v
    new_odd = odd * cos_v + even * sin_v

    # bf16 roundtrip for parity with the reference / Q-side kernel numerics.
    new_even = new_even.to(tl.bfloat16).to(tl.float32)
    new_odd = new_odd.to(tl.bfloat16).to(tl.float32)

    # ── Paged cache pointers (segregated: values first, then scales) ──
    kv_block_idx = kv_slot_idx // kv_cache_block_size
    kv_pos_in_block = kv_slot_idx % kv_cache_block_size
    cache_block_ptr = k_cache_ptr + kv_block_idx.to(tl.int64) * KV_BLOCK_STRIDE
    if SHUFFLE:
        # [block_size // BT, HEAD_SIZE // HT, BT, HT] value tiles.
        val_ptr = (
            cache_block_ptr
            + (kv_pos_in_block // BLOCK_TILE_SIZE) * BLOCK_TILE_SIZE * TOKEN_STRIDE
            + (kv_pos_in_block % BLOCK_TILE_SIZE) * HEAD_TILE_SIZE
        )
    else:
        val_ptr = cache_block_ptr + kv_pos_in_block * TOKEN_STRIDE
    scale_ptr = (
        cache_block_ptr
        + kv_cache_block_size * TOKEN_STRIDE
        + kv_pos_in_block * SCALE_DIM
    )

    if USE_FP4:
        # MXFP4: each 32-element block = 16 consecutive even/odd pairs, so
        # tiling the halves into (N_BLOCKS, 16) lands one block per row.
        N_QUANT_BLOCKS: tl.constexpr = HEAD_SIZE // 32
        HALF_BLOCK: tl.constexpr = 16
        even_2d = tl.reshape(new_even, (N_QUANT_BLOCKS, HALF_BLOCK))
        odd_2d = tl.reshape(new_odd, (N_QUANT_BLOCKS, HALF_BLOCK))

        amax = tl.maximum(
            tl.max(tl.abs(even_2d), axis=1),
            tl.max(tl.abs(odd_2d), axis=1),
        )
        amax = tl.maximum(amax, 6.0 * (2**-126))
        # ue8m0 block scale: 2^ceil(log2(amax / 6.0)), stored (exp + 127).
        log2_ratio = tl.ceil(tl.log2(amax * (1.0 / 6.0)))
        log2_ratio = tl.minimum(tl.maximum(log2_ratio, -127.0), 127.0)
        inv_scale = tl.exp2(-log2_ratio)
        ue8m0 = (log2_ratio + 127.0).to(tl.uint8)  # [N_QUANT_BLOCKS]

        inv_scale_col = tl.reshape(inv_scale, (N_QUANT_BLOCKS, 1))
        packed = _fp32x2_to_fp4x2(
            even_2d * inv_scale_col, odd_2d * inv_scale_col
        )  # (N_BLOCKS, HALF_BLOCK) uint8
        packed_flat = tl.reshape(packed, (TOKEN_STRIDE,))

        if SHUFFLE:
            # aiter's preshuffle_cache order. Each run of N_PER_TILE tokens is
            # [D_PER_TILE-byte chunk of K, token, byte]; its scales are
            # [scale % SCALE_LANES, token, scale // SCALE_LANES].
            S_HI: tl.constexpr = SCALE_DIM // SCALE_LANES
            tl.static_assert(SCALE_LANES * S_HI == SCALE_DIM)
            tl.static_assert(TOKEN_STRIDE % D_PER_TILE == 0)
            group = kv_pos_in_block // N_PER_TILE
            lane = kv_pos_in_block % N_PER_TILE
            byte = tl.arange(0, TOKEN_STRIDE)
            value_offset = (
                group * (N_PER_TILE * TOKEN_STRIDE)
                + byte // D_PER_TILE * (N_PER_TILE * D_PER_TILE)
                + lane * D_PER_TILE
                + byte % D_PER_TILE
            )
            scale_idx = tl.arange(0, SCALE_DIM)
            scale_offset = (
                kv_cache_block_size * TOKEN_STRIDE
                + group * (N_PER_TILE * SCALE_DIM)
                + scale_idx % SCALE_LANES * (N_PER_TILE * S_HI)
                + lane * S_HI
                + scale_idx // SCALE_LANES
            )
            tl.store(cache_block_ptr + value_offset, packed_flat)
            tl.store(cache_block_ptr + scale_offset, ue8m0)
        else:
            tl.store(val_ptr + tl.arange(0, TOKEN_STRIDE), packed_flat)
            tl.store(scale_ptr + tl.arange(0, SCALE_DIM), ue8m0)
    else:
        # Per-token FP8 (single 128-wide block) with one float32 scale.
        result = tl.interleave(new_even, new_odd)  # [HEAD_SIZE] fp32
        result_bf16 = result.to(tl.bfloat16).to(tl.float32)
        INV_FP8_MAX: tl.constexpr = 1.0 / FP8_MAX
        absmax = tl.maximum(tl.max(tl.abs(result_bf16), axis=0), 1e-4)
        exponent = tl.ceil(tl.log2(absmax * INV_FP8_MAX))
        inv_scale = tl.exp2(-exponent)
        x_clamped = tl.clamp(result_bf16 * inv_scale, -FP8_MAX, FP8_MAX)
        x_uint8 = x_clamped.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        if SHUFFLE:
            tiled = (
                block // HEAD_TILE_SIZE * BLOCK_TILE_SIZE * HEAD_TILE_SIZE
                + block % HEAD_TILE_SIZE
            )
            tl.store(val_ptr + tiled, x_uint8)
        else:
            tl.store(val_ptr + block, x_uint8)
        tl.store(scale_ptr.to(tl.pointer_type(tl.float32)), tl.exp2(exponent))
