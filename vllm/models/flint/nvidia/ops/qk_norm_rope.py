# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused per-head q/k RMSNorm + NeoX RoPE + paged KV-cache write.

Numerics follow the unfused vLLM ops it replaces: RMSNorm computes
``bf16(x * rsqrt(mean(x^2) + eps) * w)`` in fp32, RoPE computes
``x * cos - y * sin`` in fp32 from the fp32 cos/sin cache (as the reference
implementation does; vLLM's rotary op would round the cache to bf16 first) and
rounds to bf16, and K/V are stored unchanged in the bf16 cache.
"""

import torch

from vllm.triton_utils import tl, triton

_BLOCK_T = 16


# A fixed token tile and no integer specialization: one compiled variant per
# (RoPE, PDL) pair, all of which CUDA graph capture and warmup reach.
@triton.jit(do_not_specialize=["num_tokens", "num_slots"])
def _qk_norm_rope_cache_kernel(
    qkv_ptr,
    qkv_stride,
    q_out_ptr,
    positions_ptr,
    q_weight_ptr,
    k_weight_ptr,
    cos_sin_ptr,
    cos_sin_stride,
    slot_mapping_ptr,
    num_tokens,
    num_slots,
    k_cache_ptr,
    v_cache_ptr,
    stride_block,
    stride_slot,
    stride_head,
    block_size,
    eps,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    APPLY_ROPE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    LAUNCH_DEPENDENTS: tl.constexpr,
):
    if LAUNCH_DEPENDENTS:
        # The PDL successor waits on this grid before reading q / K / V.
        tl.extra.cuda.gdc_launch_dependents()
    head = tl.program_id(1)
    half: tl.constexpr = HEAD_DIM // 2
    tokens = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    offs = tl.arange(0, half)[None, :]
    t_mask = (tokens < num_tokens)[:, None]
    tok64 = tokens.to(tl.int64)[:, None]
    row = qkv_ptr + tok64 * qkv_stride + head * HEAD_DIM
    x1 = tl.load(row + offs, mask=t_mask, other=0.0).to(tl.float32)
    x2 = tl.load(row + half + offs, mask=t_mask, other=0.0).to(tl.float32)

    is_q = head < NUM_Q_HEADS
    is_v = head >= NUM_Q_HEADS + NUM_KV_HEADS
    is_k = (head >= NUM_Q_HEADS) & (head < NUM_Q_HEADS + NUM_KV_HEADS)
    w_ptr = tl.where(is_q, q_weight_ptr, k_weight_ptr)
    w1 = tl.load(w_ptr + offs).to(tl.float32)
    w2 = tl.load(w_ptr + half + offs).to(tl.float32)
    var = (tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / HEAD_DIM
    rs = tl.math.rsqrt(var + eps)[:, None]
    n1 = (x1 * rs * w1).to(tl.bfloat16).to(tl.float32)
    n2 = (x2 * rs * w2).to(tl.bfloat16).to(tl.float32)
    if APPLY_ROPE:
        pos = tl.load(positions_ptr + tokens, mask=tokens < num_tokens, other=0)
        cs = cos_sin_ptr + pos.to(tl.int64)[:, None] * cos_sin_stride
        cos = tl.load(cs + offs, mask=t_mask, other=0.0).to(tl.float32)
        sin = tl.load(cs + half + offs, mask=t_mask, other=0.0).to(tl.float32)
        r1 = n1 * cos - n2 * sin
        r2 = n2 * cos + n1 * sin
        n1 = r1
        n2 = r2
    o1 = tl.where(is_v, x1, n1).to(tl.bfloat16)
    o2 = tl.where(is_v, x2, n2).to(tl.bfloat16)

    out = q_out_ptr + (tok64 * NUM_Q_HEADS + head) * HEAD_DIM
    tl.store(out + offs, o1, mask=t_mask & is_q)
    tl.store(out + half + offs, o2, mask=t_mask & is_q)
    # Keep the normalized key in qkv as well; attention may read it.
    tl.store(row + offs, o1, mask=t_mask & is_k)
    tl.store(row + half + offs, o2, mask=t_mask & is_k)

    slot = tl.load(slot_mapping_ptr + tokens, mask=tokens < num_slots, other=-1)
    slot = slot.to(tl.int64)[:, None]
    kv_head = head - NUM_Q_HEADS - tl.where(is_v, NUM_KV_HEADS, 0)
    dst = (
        (slot // block_size) * stride_block
        + (slot % block_size) * stride_slot
        + kv_head * stride_head
    )
    k_mask = (slot >= 0) & is_k
    v_mask = (slot >= 0) & is_v
    tl.store(k_cache_ptr + dst + offs, o1, mask=k_mask)
    tl.store(k_cache_ptr + dst + half + offs, o2, mask=k_mask)
    tl.store(v_cache_ptr + dst + offs, o1, mask=v_mask)
    tl.store(v_cache_ptr + dst + half + offs, o2, mask=v_mask)


def qk_norm_rope_cache(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor | None,
    k_cache: torch.Tensor | None,
    v_cache: torch.Tensor | None,
    slot_mapping: torch.Tensor | None,
    num_q_heads: int,
    num_kv_heads: int,
    head_dim: int,
    launch_dependents: bool = False,
) -> torch.Tensor:
    """Normalize and rotate q/k from ``qkv`` and write k/v into the cache.

    Args:
        qkv: [T, (Hq + 2 * Hkv) * D] bf16; k is normalized in place.
        positions: [T] token positions.
        q_weight: [D] q RMSNorm weight.
        k_weight: [D] k RMSNorm weight.
        eps: RMSNorm epsilon.
        cos_sin_cache: [max_pos, D] fp32 (cos | sin), or None for no RoPE.
        k_cache: [num_blocks, block_size, Hkv, D] view (any strides with a
            contiguous last dim), or None to skip the cache write.
        v_cache: Same shape and strides as ``k_cache``.
        slot_mapping: [<= T] int64; negative slots are skipped.
        num_q_heads: Hq.
        num_kv_heads: Hkv.
        head_dim: D.
        launch_dependents: Let a PDL successor that waits on the grid
            dependency before reading q / K / V (Flint decode attention) start
            its prologue early. Must stay False before other kernels.

    Returns:
        q: [T, Hq * D] bf16.

    """
    num_tokens = qkv.shape[0]
    q = torch.empty(
        (num_tokens, num_q_heads * head_dim), dtype=qkv.dtype, device=qkv.device
    )
    if num_tokens == 0:
        return q
    if k_cache is not None and v_cache is not None and slot_mapping is not None:
        assert k_cache.stride() == v_cache.stride() and k_cache.stride(-1) == 1
        strides = k_cache.stride()[:3]
        block_size = k_cache.shape[1]
        num_slots = slot_mapping.shape[0]
    else:
        k_cache = v_cache = slot_mapping = qkv
        strides = (0, 0, 0)
        block_size = 1
        num_slots = 0
    grid = (triton.cdiv(num_tokens, _BLOCK_T), num_q_heads + 2 * num_kv_heads)
    _qk_norm_rope_cache_kernel[grid](
        qkv,
        qkv.stride(0),
        q,
        positions,
        q_weight,
        k_weight,
        qkv if cos_sin_cache is None else cos_sin_cache,
        0 if cos_sin_cache is None else cos_sin_cache.stride(0),
        slot_mapping,
        num_tokens,
        num_slots,
        k_cache,
        v_cache,
        *strides,
        block_size,
        eps,
        NUM_Q_HEADS=num_q_heads,
        NUM_KV_HEADS=num_kv_heads,
        HEAD_DIM=head_dim,
        APPLY_ROPE=cos_sin_cache is not None,
        BLOCK_T=_BLOCK_T,
        LAUNCH_DEPENDENTS=launch_dependents,
        num_warps=4,
    )
    return q
