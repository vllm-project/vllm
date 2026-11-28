# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused DeepSeek-V4 K production for the CDNA4 paged MXFP4 indexer."""

from typing import TYPE_CHECKING

import torch

from vllm.model_executor.warmup.jit_warmup_triton_helper import (
    DispatchSpec,
    TritonWarmupTensor,
    triton_kernel_dispatcher_with_warmup,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.rocm_paged_mxfp4_indexer import (
    rocm_paged_mxfp4_cache_layout,
)

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@triton.jit(
    do_not_specialize=[
        "state_cache_stride0",
        "state_cache_stride1",
        "block_table_stride",
        "block_size",
        "page_size",
        "rms_norm_eps",
        "cos_sin_stride",
    ]
)
def _fused_indexer_k_kernel(
    state_cache_ptr,
    state_cache_stride0,
    state_cache_stride1,
    token_to_req_indices_ptr,
    positions_ptr,
    slot_mapping_ptr,
    block_table_ptr,
    block_table_stride,
    block_size,
    rms_norm_weight_ptr,
    rms_norm_eps,
    cos_sin_cache_ptr,
    cos_sin_stride,
    kv_cache_ptr,
    kv_slot_mapping_ptr,
    page_size,
    page_stride: tl.int64,
    HEAD_SIZE: tl.constexpr,
    STATE_WIDTH: tl.constexpr,
    COMPRESS_RATIO: tl.constexpr,
    OVERLAP: tl.constexpr,
    ROPE_HEAD_DIM: tl.constexpr,
    N_PER_TILE: tl.constexpr,
    D_PER_TILE: tl.constexpr,
    SCALE_LANES: tl.constexpr,
):
    token_idx = tl.program_id(0)
    if tl.load(slot_mapping_ptr + token_idx) < 0:
        return
    position = tl.load(positions_ptr + token_idx)
    if (position + 1) % COMPRESS_RATIO != 0:
        return
    slot = tl.load(kv_slot_mapping_ptr + token_idx)
    if slot < 0:
        return

    req_idx = tl.load(token_to_req_indices_ptr + token_idx)
    tokens = tl.arange(0, (1 + OVERLAP) * COMPRESS_RATIO)
    pos = position - (1 + OVERLAP) * COMPRESS_RATIO + 1 + tokens
    block_numbers = tl.load(
        block_table_ptr + req_idx * block_table_stride + pos // block_size,
        mask=pos >= 0,
        other=0,
    )
    head_offset = (tokens >= COMPRESS_RATIO).to(tl.int32) * HEAD_SIZE
    row_base = (
        state_cache_ptr
        + block_numbers.to(tl.int64) * state_cache_stride0
        + pos % block_size * state_cache_stride1
        + head_offset
    )
    offsets = tl.arange(0, HEAD_SIZE)
    scores = tl.load(
        row_base[:, None] + STATE_WIDTH + offsets[None, :],
        mask=pos[:, None] >= 0,
        other=float("-inf"),
    )
    values = tl.load(
        row_base[:, None] + offsets[None, :], mask=pos[:, None] >= 0, other=0.0
    )
    pooled = tl.sum(values * tl.softmax(scores, dim=0), axis=0)
    pooled = pooled.to(tl.bfloat16).to(tl.float32)
    norm_weight = tl.load(rms_norm_weight_ptr + offsets).to(tl.float32)
    variance = tl.sum(pooled * pooled, axis=0) / HEAD_SIZE
    normed = pooled * tl.rsqrt(variance + rms_norm_eps) * norm_weight
    normed = normed.to(tl.bfloat16).to(tl.float32)

    NUM_PAIRS: tl.constexpr = HEAD_SIZE // 2
    NOPE_PAIRS: tl.constexpr = (HEAD_SIZE - ROPE_HEAD_DIM) // 2
    HALF_ROPE: tl.constexpr = ROPE_HEAD_DIM // 2
    even, odd = tl.split(tl.reshape(normed, (NUM_PAIRS, 2)))
    rope_pair = tl.arange(0, NUM_PAIRS) - NOPE_PAIRS
    cs_idx = tl.maximum(rope_pair, 0)
    group_pos = position // COMPRESS_RATIO * COMPRESS_RATIO
    cs_base = cos_sin_cache_ptr + group_pos * cos_sin_stride
    cosine = tl.load(cs_base + cs_idx, mask=rope_pair >= 0, other=1.0)
    sine = tl.load(cs_base + HALF_ROPE + cs_idx, mask=rope_pair >= 0, other=0.0)
    rotated_even = (even * cosine - odd * sine).to(tl.bfloat16).to(tl.float32)
    rotated_odd = (odd * cosine + even * sine).to(tl.bfloat16).to(tl.float32)

    NUM_SCALES: tl.constexpr = HEAD_SIZE // 32
    TOKEN_BYTES: tl.constexpr = HEAD_SIZE // 2
    even_blocks = tl.reshape(rotated_even, (NUM_SCALES, 16))
    odd_blocks = tl.reshape(rotated_odd, (NUM_SCALES, 16))
    amax = tl.maximum(
        tl.max(tl.abs(even_blocks), axis=1), tl.max(tl.abs(odd_blocks), axis=1)
    )
    ratio = tl.maximum(amax, 6.0 * (2**-126)) * (1.0 / 6.0)
    exponent = (ratio.to(tl.uint32, bitcast=True) + 0x7FFFFF) >> 23
    inv_scale = ((254 - exponent) << 23).to(tl.float32, bitcast=True)
    packed = tl.inline_asm_elementwise(
        "v_cvt_scalef32_pk_fp4_f32 $0, $1, $2, $3",
        constraints="=v,v,v,v",
        args=[even_blocks * inv_scale[:, None], odd_blocks * inv_scale[:, None], 1.0],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)

    page = kv_cache_ptr + (slot // page_size).to(tl.int64) * page_stride
    page_pos = slot % page_size
    group = page_pos // N_PER_TILE
    lane = page_pos % N_PER_TILE
    byte = tl.arange(0, TOKEN_BYTES)
    scale_idx = tl.arange(0, NUM_SCALES)
    S_HI: tl.constexpr = NUM_SCALES // SCALE_LANES
    tl.static_assert(NUM_SCALES % SCALE_LANES == 0)
    tl.static_assert(TOKEN_BYTES % D_PER_TILE == 0)
    value_off = (
        group * (N_PER_TILE * TOKEN_BYTES)
        + byte // D_PER_TILE * (N_PER_TILE * D_PER_TILE)
        + lane * D_PER_TILE
        + byte % D_PER_TILE
    )
    scale_off = (
        group * (N_PER_TILE * NUM_SCALES)
        + scale_idx % SCALE_LANES * (N_PER_TILE * S_HI)
        + lane * S_HI
        + scale_idx // SCALE_LANES
    )
    tl.store(page + value_off, tl.reshape(packed, (TOKEN_BYTES,)))
    tl.store(page + page_size * TOKEN_BYTES + scale_off, exponent.to(tl.uint8))


def _warmup_inputs(
    *,
    vllm_config: "VllmConfig",
    k_cache_prefix: str,
    dtype: torch.dtype,
    head_dim: int,
    rope_head_dim: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    num_index_heads: int,
) -> list[dict[str, object]]:
    cache = vllm_config.compilation_config.static_forward_context[k_cache_prefix]
    kv_cache = cache.kv_cache
    return [
        dict(
            state_cache=TritonWarmupTensor(
                torch.float32,
                shape=(1, compress_ratio, 2 * state_width),
                ptr_range_bytes=state_bytes,
            ),
            token_to_req_indices=TritonWarmupTensor(torch.int32),
            positions=TritonWarmupTensor(torch.int64),
            slot_mapping=TritonWarmupTensor(torch.int64),
            block_table=TritonWarmupTensor(torch.int32, shape=(1, 1)),
            block_size=compress_ratio,
            rms_norm_weight=TritonWarmupTensor(dtype, shape=(head_dim,)),
            rms_norm_eps=1e-6,
            cos_sin_cache=TritonWarmupTensor(torch.float32, shape=(1, rope_head_dim)),
            kv_cache=TritonWarmupTensor(
                torch.uint8,
                shape=(1, *kv_cache.shape[1:]),
                strides=kv_cache.stride(),
                ptr_range_bytes=cache_bytes,
            ),
            kv_slot_mapping=TritonWarmupTensor(torch.int64),
            head_dim=head_dim,
            rope_head_dim=rope_head_dim,
            state_width=state_width,
            compress_ratio=compress_ratio,
            overlap=overlap,
            num_index_heads=num_index_heads,
        )
        for state_bytes in (0, 1 << 32)
        for cache_bytes in (0, 1 << 32)
    ]


@triton_kernel_dispatcher_with_warmup(
    kernel=_fused_indexer_k_kernel,
    warmup_inputs=_warmup_inputs,
)
def rocm_fused_indexer_k(
    *,
    state_cache: torch.Tensor,
    token_to_req_indices: torch.Tensor,
    positions: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_table: torch.Tensor,
    block_size: int,
    rms_norm_weight: torch.Tensor,
    rms_norm_eps: float,
    cos_sin_cache: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_slot_mapping: torch.Tensor,
    head_dim: int,
    rope_head_dim: int,
    state_width: int,
    compress_ratio: int,
    overlap: bool,
    num_index_heads: int,
) -> DispatchSpec:
    layout = rocm_paged_mxfp4_cache_layout(num_index_heads, head_dim, kv_cache.shape[1])
    return (slot_mapping.shape[0],), dict(
        state_cache_stride0=state_cache.stride(0),
        state_cache_stride1=state_cache.stride(1),
        block_table_stride=block_table.stride(0),
        cos_sin_stride=cos_sin_cache.stride(0),
        page_size=kv_cache.shape[1],
        page_stride=kv_cache.stride(0),
        HEAD_SIZE=head_dim,
        STATE_WIDTH=state_width,
        COMPRESS_RATIO=compress_ratio,
        OVERLAP=overlap,
        ROPE_HEAD_DIM=rope_head_dim,
        N_PER_TILE=layout.n_per_tile,
        D_PER_TILE=layout.d_per_tile,
        SCALE_LANES=layout.scale_lanes,
        num_warps=1,
    )
