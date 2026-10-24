# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Step5 head-dim-192 QKV preparation without a sparse indexer."""

import torch

from vllm.triton_utils import tl, triton

from .qkv_indexer_norm_rope import _prepare_main


@triton.jit(do_not_specialize=["SLOT_ROWS"])
def _prepare_qkv(
    QKV,
    QWeight,
    KWeight,
    Rope,
    Positions,
    Slots,
    Q,
    K,
    V,
    KCache,
    VCache,
    QH: tl.constexpr,
    KH: tl.constexpr,
    STRIDE: tl.constexpr,
    ROTARY: tl.constexpr,
    ROPE_STRIDE: tl.constexpr,
    SLOT_ROWS,
    BLOCK_SIZE: tl.constexpr,
    CACHE_BLOCK: tl.constexpr,
    CACHE_TOKEN: tl.constexpr,
    CACHE_HEAD: tl.constexpr,
    EPS: tl.constexpr,
    BIAS: tl.constexpr,
):
    _prepare_main(
        QKV,
        QWeight,
        KWeight,
        Rope,
        Positions,
        Slots,
        Q,
        K,
        V,
        KCache,
        VCache,
        tl.program_id(0),
        tl.program_id(1),
        QH,
        KH,
        STRIDE,
        ROTARY,
        ROPE_STRIDE,
        SLOT_ROWS,
        BLOCK_SIZE,
        CACHE_BLOCK,
        CACHE_TOKEN,
        CACHE_HEAD,
        EPS,
        BIAS,
        None,
        None,
        False,
    )


def qkv_norm_rope_cache(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    slots: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    rotary_dim: int,
    eps: float,
    zero_centered: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Normalize Q/K, apply RoPE, and write BF16 K/V for SWA preparation.

    Uses the joint indexer kernel's preparation and rounding rules. Cache
    views have shape [pages, block_size, kv_heads, 192] and matching strides.
    Negative slots and tokens beyond the slot array do not write the cache.
    """
    dim = 192
    assert num_heads > 0 and num_kv_heads > 0
    assert qkv.ndim == 2 and qkv.shape[1] == (num_heads + 2 * num_kv_heads) * dim
    assert qkv.dtype == key_cache.dtype == value_cache.dtype == torch.bfloat16
    assert key_cache.shape == value_cache.shape
    assert key_cache.ndim == 4 and key_cache.shape[2:] == (num_kv_heads, dim)
    assert key_cache.stride() == value_cache.stride()
    assert qkv.stride(1) == key_cache.stride(3) == value_cache.stride(3) == 1
    assert 0 <= rotary_dim <= dim and rotary_dim % 2 == 0
    tokens = qkv.shape[0]
    assert positions.numel() >= tokens and slots.numel() <= tokens
    q = qkv.new_empty((tokens, num_heads * dim))
    k = qkv.new_empty((tokens, num_kv_heads * dim))
    v = torch.empty_like(k)
    if tokens:
        group = 4 if rotary_dim > 128 else 8
        _prepare_qkv[(tokens, triton.cdiv(num_heads + 2 * num_kv_heads, group))](
            qkv,
            q_weight,
            k_weight,
            cos_sin_cache,
            positions,
            slots,
            q,
            k,
            v,
            key_cache,
            value_cache,
            num_heads,
            num_kv_heads,
            qkv.stride(0),
            rotary_dim,
            cos_sin_cache.stride(0),
            slots.numel(),
            key_cache.shape[1],
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            eps,
            float(zero_centered),
            num_warps=4,
            enable_fp_fusion=False,
        )
    return q, k, v
