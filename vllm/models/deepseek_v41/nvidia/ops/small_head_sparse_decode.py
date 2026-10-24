# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sparse MLA decode for DeepSeek V4.1 on SM90 with few query heads per rank.

FlashMLA's SM90 sparse decode kernel tiles heads by 64 (the WGMMA M
dimension), so a TP8 shard with 8 heads wastes 7/8 of its tensor-core work.
This kernel tiles heads by 16 through ``tl.dot`` (mma.sync on SM90) and splits
the key set across programs instead.

KV cache layout (``fp8_ds_mla``, 584 B/token): each page holds
``page_size`` rows of 448 fp8 NoPE bytes + 64 bf16 RoPE values, followed by
``page_size`` rows of 8 UE8M0 scale bytes (7 used, one per 64 NoPE dims).
V is the full 512-dim key, so one dequantized tile feeds both GEMMs.
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


def small_head_decode_enabled(num_local_heads: int) -> bool:
    if num_local_heads not in (8, 16):
        return False
    return current_platform.is_device_capability_family(90)


@triton.jit
def _load_kv_block(
    cache_ptr,
    page_stride,
    page_size: tl.constexpr,
    indices_ptr,
    idx_base,
    limit,
    key_offs,
    BLOCK_N: tl.constexpr,
):
    """Gather and dequantize BLOCK_N keys into a [BLOCK_N, 512] bf16 tile."""
    offs = key_offs + tl.arange(0, BLOCK_N)
    in_range = offs < limit
    token = tl.load(indices_ptr + idx_base + offs, mask=in_range, other=-1)
    valid = in_range & (token >= 0)
    safe = tl.where(valid, token, 0)
    page = (safe // page_size).to(tl.int64)
    row = (safe % page_size).to(tl.int64)
    page_base = cache_ptr + page * page_stride
    data_base = page_base + row * 576

    d = tl.arange(0, 512)
    is_nope = d < 448
    raw = tl.load(
        data_base[:, None] + d[None, :],
        mask=valid[:, None] & is_nope[None, :],
        other=0,
    )
    nope = raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)

    t = tl.arange(0, 8)
    enc = tl.load(
        (page_base + page_size * 576 + row * 8)[:, None] + t[None, :],
        mask=valid[:, None] & (t[None, :] < 7),
        other=127,
    )
    scale = tl.exp2(enc.to(tl.float32) - 127.0)
    nope = tl.reshape(
        tl.reshape(nope, (BLOCK_N, 8, 64)) * scale[:, :, None], (BLOCK_N, 512)
    )

    rope_ptr = (data_base + 448).to(tl.pointer_type(tl.bfloat16))
    rope = tl.load(
        rope_ptr[:, None] + (d - 448)[None, :],
        mask=valid[:, None] & ~is_nope[None, :],
        other=0.0,
    )
    k = tl.where(is_nope[None, :], nope, rope.to(tl.float32))
    return k.to(tl.bfloat16), valid


@triton.jit
def _small_head_sparse_decode_kernel(
    q_ptr,
    q_stride_t,
    q_stride_h,
    swa_cache_ptr,
    swa_page_stride,
    swa_page_size: tl.constexpr,
    swa_indices_ptr,
    swa_indices_stride,
    swa_lens_ptr,
    extra_cache_ptr,
    extra_page_stride,
    extra_page_size: tl.constexpr,
    extra_indices_ptr,
    extra_indices_stride,
    extra_lens_ptr,
    part_o_ptr,
    part_lse_ptr,
    num_heads,
    sm_scale,
    HAS_EXTRA: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    tok = tl.program_id(0)
    split = tl.program_id(1)

    h = tl.arange(0, BLOCK_H)
    d = tl.arange(0, 512)
    q = tl.load(
        q_ptr + tok.to(tl.int64) * q_stride_t + h[:, None] * q_stride_h + d[None, :],
        mask=(h < num_heads)[:, None],
        other=0.0,
    )

    qk_scale = sm_scale * 1.4426950408889634
    m_i = tl.full([BLOCK_H], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_H], dtype=tl.float32)
    acc = tl.zeros([BLOCK_H, 512], dtype=tl.float32)

    swa_len = tl.load(swa_lens_ptr + tok)
    swa_blocks = tl.cdiv(swa_len, BLOCK_N)
    num_blocks = swa_blocks
    extra_len = 0
    if HAS_EXTRA:
        extra_len = tl.load(extra_lens_ptr + tok)
        num_blocks += tl.cdiv(extra_len, BLOCK_N)
    blocks_per_split = tl.cdiv(num_blocks, NUM_SPLITS)
    start_block = split * blocks_per_split
    end_block = tl.minimum(start_block + blocks_per_split, num_blocks)

    for blk in range(start_block, end_block):
        if blk < swa_blocks:
            k, valid = _load_kv_block(
                swa_cache_ptr,
                swa_page_stride,
                swa_page_size,
                swa_indices_ptr,
                tok.to(tl.int64) * swa_indices_stride,
                swa_len,
                blk * BLOCK_N,
                BLOCK_N,
            )
        else:
            k, valid = _load_kv_block(
                extra_cache_ptr,
                extra_page_stride,
                extra_page_size,
                extra_indices_ptr,
                tok.to(tl.int64) * extra_indices_stride,
                extra_len,
                (blk - swa_blocks) * BLOCK_N,
                BLOCK_N,
            )
        s = tl.dot(q, tl.trans(k)) * qk_scale
        s = tl.where(valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(s - m_safe[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.bfloat16), k)
        m_i = m_new

    empty = l_i == 0.0
    # log2-domain LSE; -inf marks a split that saw no valid key.
    lse = tl.where(empty, float("-inf"), m_i + tl.log2(tl.where(empty, 1.0, l_i)))
    acc = acc * tl.where(empty, 0.0, 1.0 / l_i)[:, None]

    part = (tok.to(tl.int64) * NUM_SPLITS + split) * BLOCK_H
    tl.store(part_o_ptr + (part + h[:, None]) * 512 + d[None, :], acc)
    tl.store(part_lse_ptr + part + h, lse)


@triton.jit
def _merge_splits_kernel(
    part_o_ptr,
    part_lse_ptr,
    sink_ptr,
    out_ptr,
    out_stride_t,
    out_stride_h,
    BLOCK_H: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
):
    tok = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, 512)
    s = tl.arange(0, NUM_SPLITS)
    part = (tok.to(tl.int64) * NUM_SPLITS + s) * BLOCK_H + head
    lse = tl.load(part_lse_ptr + part)
    m = tl.max(lse, 0)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp2(lse - m_safe)
    denom = tl.sum(w, 0)
    has_keys = denom > 0.0
    safe_denom = tl.where(has_keys, denom, 1.0)
    o = tl.sum(tl.load(part_o_ptr + part[:, None] * 512 + d[None, :]) * w[:, None], 0)

    total_lse2 = m_safe + tl.log2(safe_denom)
    sink2 = tl.load(sink_ptr + head) * 1.4426950408889634
    # exp(lse) / (exp(lse) + exp(sink)), evaluated in log2 space.
    o = o / safe_denom / (1.0 + tl.exp2(sink2 - total_lse2))
    o = tl.where(has_keys, o, 0.0)
    tl.store(
        out_ptr + tok.to(tl.int64) * out_stride_t + head * out_stride_h + d,
        o.to(out_ptr.dtype.element_ty),
    )


# Tuned on H20. Beyond these token counts FlashMLA is as fast or faster.
_SHORT_KEYS = 256
_SHORT_KEYS_MAX_TOKENS = 16
_LONG_KEYS_MAX_TOKENS = 12


def small_head_decode_supported(num_tokens: int, max_keys: int) -> bool:
    if max_keys <= _SHORT_KEYS:
        return num_tokens <= _SHORT_KEYS_MAX_TOKENS
    return num_tokens <= _LONG_KEYS_MAX_TOKENS


def _pick_config(num_tokens: int, max_keys: int) -> tuple[int, int, int, int]:
    if max_keys > _SHORT_KEYS:
        if num_tokens <= 3:
            return 32, 32, 4, 2
        if num_tokens <= 7:
            return 16, 32, 8, 2
    return 8, 32, 8, 2


def small_head_sparse_decode(
    q: torch.Tensor,
    swa_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_lens: torch.Tensor,
    extra_cache: torch.Tensor | None,
    extra_indices: torch.Tensor | None,
    extra_lens: torch.Tensor | None,
    attn_sink: torch.Tensor,
    sm_scale: float,
    out: torch.Tensor,
    num_heads: int,
) -> None:
    """Sparse MLA decode over SWA keys plus optional compressed (extra) keys.

    Args:
        q: ``[T, H_pad, 512]`` bf16; only the first ``num_heads`` heads are read.
        swa_cache: ``fp8_ds_mla`` paged SWA cache, ``[pages, page, 584]``.
        swa_indices: ``[>=T, (1,) width]`` int32 global slot IDs; -1 is ignored.
        swa_lens: ``[>=T]`` int32 valid SWA lengths.
        extra_cache: Optional compressed cache with the same packed layout.
        extra_indices: Optional compressed-cache global slot IDs.
        extra_lens: Optional valid compressed-cache lengths.
        attn_sink: ``[H_pad]`` fp32 per-head sink logits.
        sm_scale: Softmax scale.
        out: ``[T, H_pad, 512]``; the first ``num_heads`` heads are written.
        num_heads: real (unpadded) heads on this rank, at most 16.

    """
    assert num_heads in (8, 16)
    assert q.dtype == out.dtype == torch.bfloat16
    assert q.shape[-1] == out.shape[-1] == 512
    assert q.stride(-1) == out.stride(-1) == 1
    assert swa_cache.shape[-1] == 584
    assert extra_cache is None or extra_cache.shape[-1] == 584
    num_tokens = q.shape[0]
    if num_tokens == 0:
        return
    block_h = 16
    max_keys = swa_indices.shape[-1]
    if extra_indices is not None:
        max_keys += extra_indices.shape[-1]
    num_splits, block_n, num_warps, num_stages = _pick_config(num_tokens, max_keys)
    part_o = torch.empty(
        (num_tokens, num_splits, block_h, 512), dtype=torch.float32, device=q.device
    )
    part_lse = torch.empty(
        (num_tokens, num_splits, block_h), dtype=torch.float32, device=q.device
    )
    has_extra = extra_cache is not None
    if not has_extra:
        extra_cache, extra_indices, extra_lens = swa_cache, swa_indices, swa_lens
    assert extra_cache is not None
    assert extra_indices is not None and extra_lens is not None

    # Metadata buffers are sized for max_num_batched_tokens; keep our rows.
    swa_indices = swa_indices[:num_tokens].reshape(num_tokens, -1)
    extra_indices = extra_indices[:num_tokens].reshape(num_tokens, -1)
    # Byte-addressed: scales must be read as raw bytes, not fp8 values.
    swa_cache = swa_cache.view(torch.uint8)
    extra_cache = extra_cache.view(torch.uint8)
    _small_head_sparse_decode_kernel[(num_tokens, num_splits)](
        q,
        q.stride(0),
        q.stride(1),
        swa_cache,
        swa_cache.stride(0),
        swa_cache.shape[1],
        swa_indices,
        swa_indices.stride(0),
        swa_lens,
        extra_cache,
        extra_cache.stride(0),
        extra_cache.shape[1],
        extra_indices,
        extra_indices.stride(0),
        extra_lens,
        part_o,
        part_lse,
        num_heads,
        sm_scale,
        HAS_EXTRA=has_extra,
        BLOCK_H=block_h,
        BLOCK_N=block_n,
        NUM_SPLITS=num_splits,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    _merge_splits_kernel[(num_tokens, num_heads)](
        part_o,
        part_lse,
        attn_sink,
        out,
        out.stride(0),
        out.stride(1),
        BLOCK_H=block_h,
        NUM_SPLITS=num_splits,
        num_warps=4,
    )
