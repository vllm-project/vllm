# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DSA indexer Q/K epilogues.

Q: SSMax, optional RoPE and per-head UE8M0 FP8 quantization in one launch,
with the quantization scale folded into the head weights (extends
``fused_indexer_q_rope_quant`` from ``sparse_attn_indexer``).
K: fp32 LayerNorm and optional RoPE in one launch, optionally quantized and
written straight to the paged FP8 indexer cache.
The Q launch can also clear the top-k buffer rows that the indexer fills.

The indexer layout is ``[rope | nope]``. Intermediate roundings match the
unfused path.
"""

import torch

from vllm.model_executor.layers.quantization.utils.quant_utils import (
    get_fp8_min_max,
)
from vllm.models.deepseek_v32.common.kernels import (
    _FP8_DTYPE,
    _FP8_MAX,
    _INDEXER_CACHE_BLOCK_TILE,
    _INDEXER_CACHE_HEAD_TILE_BYTES,
    _USE_FNUZ,
    _fp8_quant_and_cache_write,
)
from vllm.models.mistral.amd.ops.ssmax import _round_bf16, _ssmax_scale
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.deep_gemm import mqa_logits_use_mx_scales
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import indexer_k_cache_is_shuffled


@triton.jit
def _indexer_q_kernel(
    positions_ptr,
    q_ptr,
    q_s0,
    q_s1,
    cos_sin_cache_ptr,
    cache_stride,
    q_fp8_ptr,
    q_fp8_s0,
    q_fp8_s1,
    weights_ptr,
    weights_s0,
    weights_s1,
    weights_out_ptr,
    weights_out_s0,
    weights_out_s1,
    softmax_scale,
    head_scale,
    log_pos_ptr,
    s_ptr,
    b_ptr,
    topk_ptr,
    topk_stride,
    fp8_min: tl.constexpr,
    fp8_max: tl.constexpr,
    clear_topk: tl.constexpr,
    topk_tokens: tl.constexpr,
    topk_block: tl.constexpr,
    is_neox: tl.constexpr,
    apply_rope: tl.constexpr,
    num_heads: tl.constexpr,
    heads_block: tl.constexpr,
    head_dim: tl.constexpr,
    dim_block: tl.constexpr,
    half_rope_dim: tl.constexpr,
    rope_block: tl.constexpr,
    nope_block: tl.constexpr,
):
    # One program per token, covering all heads; one UE8M0 scale per head.
    token = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, heads_block)
    head_mask = heads < num_heads
    q_base = q_ptr + token * q_s0 + heads[:, None] * q_s1
    out_base = q_fp8_ptr + token * q_fp8_s0 + heads[:, None] * q_fp8_s1
    out_dtype = q_fp8_ptr.dtype.element_ty
    scale = _ssmax_scale(
        log_pos_ptr, s_ptr, b_ptr, token + heads * 0, heads, head_mask
    )[:, None]

    if apply_rope:
        offs = tl.arange(0, rope_block)
        rope_mask = head_mask[:, None] & (offs < half_rope_dim)[None, :]
        if is_neox:
            x0_offs = offs[None, :]
            x1_offs = half_rope_dim + offs[None, :]
        else:
            x0_offs = offs[None, :] * 2
            x1_offs = offs[None, :] * 2 + 1
        x0 = tl.load(q_base + x0_offs, mask=rope_mask, other=0.0).to(tl.float32)
        x1 = tl.load(q_base + x1_offs, mask=rope_mask, other=0.0).to(tl.float32)
        x0 = _round_bf16(x0 * scale)
        x1 = _round_bf16(x1 * scale)
        pos = tl.load(positions_ptr + token)
        cache_row = cos_sin_cache_ptr + pos * cache_stride
        cs_mask = offs < half_rope_dim
        cos = tl.load(cache_row + offs, mask=cs_mask, other=1.0).to(tl.float32)
        sin = tl.load(cache_row + half_rope_dim + offs, mask=cs_mask, other=0.0)
        cos = cos[None, :]
        sin = sin.to(tl.float32)[None, :]
        r0 = _round_bf16(tl.fma(x0, cos, -(x1 * sin)))
        r1 = _round_bf16(tl.fma(x1, cos, x0 * sin))

        nope_offs = 2 * half_rope_dim + tl.arange(0, nope_block)[None, :]
        nope_mask = head_mask[:, None] & (nope_offs < head_dim)
        q_nope = tl.load(q_base + nope_offs, mask=nope_mask, other=0.0)
        q_nope = _round_bf16(q_nope.to(tl.float32) * scale)
        amax = tl.maximum(
            tl.maximum(tl.max(tl.abs(r0), axis=1), tl.max(tl.abs(r1), axis=1)),
            tl.max(tl.abs(q_nope), axis=1),
        )
        # Same scale formula as fused_indexer_q_rope_quant.
        q_scale = tl.math.exp2(
            tl.ceil(tl.log2(tl.maximum(amax, 1e-10) * (1.0 / fp8_max)))
        )
        inv = q_scale[:, None]
        tl.store(
            out_base + x0_offs,
            tl.clamp(r0 / inv, fp8_min, fp8_max).to(out_dtype),
            mask=rope_mask,
        )
        tl.store(
            out_base + x1_offs,
            tl.clamp(r1 / inv, fp8_min, fp8_max).to(out_dtype),
            mask=rope_mask,
        )
        tl.store(
            out_base + nope_offs,
            tl.clamp(q_nope / inv, fp8_min, fp8_max).to(out_dtype),
            mask=nope_mask,
        )
    else:
        offs = tl.arange(0, dim_block)[None, :]
        mask = head_mask[:, None] & (offs < head_dim)
        x = tl.load(q_base + offs, mask=mask, other=0.0).to(tl.float32)
        x = _round_bf16(x * scale)
        amax = tl.max(tl.abs(x), axis=1)
        # Same scale formula as the per_token_group_fp8_quant CUDA op.
        q_scale = tl.math.exp2(
            tl.ceil(tl.log2(tl.maximum(tl.div_rn(amax, fp8_max), 1e-10)))
        )
        tl.store(
            out_base + offs,
            tl.clamp(x / q_scale[:, None], fp8_min, fp8_max).to(out_dtype),
            mask=mask,
        )

    weight = tl.load(
        weights_ptr + token * weights_s0 + heads * weights_s1, mask=head_mask
    )
    tl.store(
        weights_out_ptr + token * weights_out_s0 + heads * weights_out_s1,
        weight.to(tl.float32) * q_scale * softmax_scale * head_scale,
        mask=head_mask,
    )
    if clear_topk:
        topk_offs = tl.arange(0, topk_block)
        tl.store(
            topk_ptr + token * topk_stride + topk_offs,
            -1,
            mask=topk_offs < topk_tokens,
        )


def fused_indexer_q_ssmax_rope_quant(
    positions: torch.Tensor,
    q: torch.Tensor,
    cos_sin_cache: torch.Tensor | None,
    weights: torch.Tensor,
    log_positions: torch.Tensor,
    s: torch.Tensor,
    b: torch.Tensor,
    softmax_scale: float,
    head_scale: float,
    is_neox: bool,
    topk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return FP8 ``q`` ``[T, H, D]`` and fp32 ``weights * q_scale * scales``.

    SSMax is applied first, then RoPE on the leading ``cos_sin_cache.shape[1]``
    dims (skipped if ``cos_sin_cache`` is None), then one UE8M0 scale per head.
    Rows ``[:T]`` of ``topk_indices``, if given, are filled with -1.
    """
    assert current_platform.is_rocm()
    assert q.dtype == torch.bfloat16 and q.stride(-1) == 1
    num_tokens, num_heads, head_dim = q.shape
    assert weights.shape == (num_tokens, num_heads)
    assert log_positions.shape == (num_tokens,)
    assert s.shape == b.shape == (num_heads,)
    rope_dim = 0 if cos_sin_cache is None else cos_sin_cache.shape[1]
    assert 0 <= rope_dim <= head_dim and rope_dim % 2 == 0
    if topk_indices is not None:
        assert topk_indices.dtype == torch.int32 and topk_indices.stride(-1) == 1
        assert topk_indices.shape[0] >= num_tokens
    topk_tokens = 0 if topk_indices is None else topk_indices.shape[1]

    q_fp8 = torch.empty_like(q, dtype=current_platform.fp8_dtype())
    weights_out = torch.empty_like(weights, dtype=torch.float32)
    if num_tokens == 0:
        return q_fp8, weights_out
    fp8_min, fp8_max = get_fp8_min_max()
    _indexer_q_kernel[(num_tokens,)](
        positions,
        q,
        q.stride(0),
        q.stride(1),
        q if cos_sin_cache is None else cos_sin_cache,
        0 if cos_sin_cache is None else cos_sin_cache.stride(0),
        q_fp8,
        q_fp8.stride(0),
        q_fp8.stride(1),
        weights,
        weights.stride(0),
        weights.stride(1),
        weights_out,
        weights_out.stride(0),
        weights_out.stride(1),
        softmax_scale,
        head_scale,
        log_positions,
        s,
        b,
        weights if topk_indices is None else topk_indices,
        0 if topk_indices is None else topk_indices.stride(0),
        fp8_min=fp8_min,
        fp8_max=fp8_max,
        clear_topk=topk_indices is not None,
        topk_tokens=topk_tokens,
        topk_block=triton.next_power_of_2(max(topk_tokens, 1)),
        is_neox=is_neox,
        apply_rope=rope_dim > 0,
        num_heads=num_heads,
        heads_block=triton.next_power_of_2(num_heads),
        head_dim=head_dim,
        dim_block=triton.next_power_of_2(head_dim),
        half_rope_dim=rope_dim // 2,
        rope_block=triton.next_power_of_2(max(rope_dim // 2, 1)),
        nope_block=triton.next_power_of_2(max(head_dim - rope_dim, 1)),
        num_warps=4,
    )
    return q_fp8, weights_out


@triton.jit
def _indexer_k_kernel(
    positions_ptr,
    k_ptr,
    k_stride,
    w_ptr,
    b_ptr,
    eps,
    cos_sin_cache_ptr,
    cache_stride,
    out_ptr,
    out_stride,
    slot_mapping_ptr,
    num_slots,
    cache_ptr,
    cache_scale_ptr,
    cache_block_size,
    cache_block_stride,
    head_dim: tl.constexpr,
    block: tl.constexpr,
    half_rope_dim: tl.constexpr,
    is_neox: tl.constexpr,
    apply_rope: tl.constexpr,
    write_cache: tl.constexpr,
    mx_scale: tl.constexpr,
    fp8_max: tl.constexpr,
    use_fnuz: tl.constexpr,
    shuffle: tl.constexpr,
    cache_block_tile: tl.constexpr,
    cache_head_tile: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, block)
    mask = offs < head_dim
    x = tl.load(k_ptr + token * k_stride + offs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + offs, mask=mask).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask).to(tl.float32)
    mean = tl.sum(x, axis=0) / head_dim
    diff = tl.where(mask, x - mean, 0.0)
    rstd = tl.rsqrt(tl.sum(diff * diff, axis=0) / head_dim + eps)
    normed = _round_bf16((x - mean) * rstd * w + b)

    if apply_rope:
        in_rope = offs < 2 * half_rope_dim
        if is_neox:
            cos_idx = offs % half_rope_dim
            partner = tl.where(
                offs < half_rope_dim, offs + half_rope_dim, offs - half_rope_dim
            )
            sign = tl.where(offs < half_rope_dim, -1.0, 1.0)
        else:
            cos_idx = offs // 2
            partner = offs ^ 1
            sign = tl.where(offs % 2 == 0, -1.0, 1.0)
        partner = tl.where(in_rope, partner, offs)
        pos = tl.load(positions_ptr + token)
        cache_row = cos_sin_cache_ptr + pos * cache_stride
        cos = tl.load(cache_row + cos_idx, mask=in_rope, other=1.0).to(tl.float32)
        sin = tl.load(cache_row + half_rope_dim + cos_idx, mask=in_rope, other=0.0).to(
            tl.float32
        )
        xp = tl.load(k_ptr + token * k_stride + partner, mask=mask, other=0.0).to(
            tl.float32
        )
        wp = tl.load(w_ptr + partner, mask=mask).to(tl.float32)
        bp = tl.load(b_ptr + partner, mask=mask).to(tl.float32)
        normed_p = _round_bf16((xp - mean) * rstd * wp + bp)
        roped = tl.fma(normed, cos, sign * normed_p * sin)
        normed = tl.where(in_rope, roped, normed)

    normed = _round_bf16(normed)
    if write_cache:
        # Same bytes as the ROCm insert, indexer_k_quant_and_cache_triton, on the
        # bf16 K: rounding, UE8M0 scale and (``shuffle``) the 16x16 tile layout.
        slot = tl.load(slot_mapping_ptr + token, mask=token < num_slots, other=-1)
        if slot >= 0:
            _fp8_quant_and_cache_write(
                tl.where(mask, normed, 0.0),
                mask,
                slot,
                cache_ptr,
                cache_scale_ptr,
                cache_block_size,
                cache_block_stride,
                offs,
                head_dim,
                shuffle,
                cache_block_tile,
                cache_head_tile,
                mx_scale,
                fp8_max,
                use_fnuz,
            )
    else:
        tl.store(out_ptr + token * out_stride + offs, normed.to(tl.bfloat16), mask=mask)


def _launch_indexer_k(
    positions: torch.Tensor,
    k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor | None,
    is_neox: bool,
    out: torch.Tensor | None,
    slot_mapping: torch.Tensor | None,
    kv_cache: torch.Tensor | None,
    shuffled: bool = False,
) -> None:
    assert k.ndim == 2 and k.stride(-1) == 1 and k.dtype == torch.bfloat16
    num_tokens, head_dim = k.shape
    assert positions.shape == (num_tokens,)
    assert weight.shape == bias.shape == (head_dim,)
    rope_dim = 0 if cos_sin_cache is None else cos_sin_cache.shape[1]
    assert 0 <= rope_dim <= head_dim and rope_dim % 2 == 0
    if num_tokens == 0:
        return
    write_cache = kv_cache is not None
    mx_scale = False
    head_tile = 1
    cache = cache_scale = k
    cache_block_size, cache_block_stride = 1, 0
    if kv_cache is not None:
        assert slot_mapping is not None and slot_mapping.ndim == 1
        # [num_blocks, block_size, head_dim + 4] bytes; a block holds the FP8
        # values of its tokens, then their scales (fp32, or 4 packed UE8M0
        # bytes where the MQA logits kernels take MX scales).
        assert kv_cache.ndim == 3 and kv_cache.shape[2] == head_dim + 4
        assert kv_cache.stride(2) == 1 and kv_cache.stride(1) == kv_cache.shape[2]
        mx_scale = mqa_logits_use_mx_scales()
        cache_u8 = kv_cache.view(torch.uint8)
        scale_dtype = torch.int32 if mx_scale else torch.float32
        cache, cache_scale = cache_u8.view(_FP8_DTYPE), cache_u8.view(scale_dtype)
        cache_block_size, cache_block_stride = kv_cache.shape[1], kv_cache.stride(0)
        if shuffled:
            # The block size is validated when the cache spec is built
            # (MistralIndexerCache.get_kv_cache_spec); this guards direct callers.
            assert cache_block_size % _INDEXER_CACHE_BLOCK_TILE == 0, (
                f"indexer K cache block size {cache_block_size} must be a "
                f"multiple of {_INDEXER_CACHE_BLOCK_TILE} for the shuffled layout"
            )
            # The head tile is a byte count; the kernel indexes FP8 elements.
            head_tile = _INDEXER_CACHE_HEAD_TILE_BYTES // cache.element_size()
            assert head_dim % head_tile == 0
    _indexer_k_kernel[(num_tokens,)](
        positions,
        k,
        k.stride(0),
        weight,
        bias,
        eps,
        k if cos_sin_cache is None else cos_sin_cache,
        0 if cos_sin_cache is None else cos_sin_cache.stride(0),
        k if out is None else out,
        0 if out is None else out.stride(0),
        positions if slot_mapping is None else slot_mapping,
        0 if slot_mapping is None else slot_mapping.shape[0],
        cache,
        cache_scale,
        cache_block_size,
        cache_block_stride,
        head_dim=head_dim,
        block=triton.next_power_of_2(head_dim),
        half_rope_dim=max(rope_dim // 2, 1),
        is_neox=is_neox,
        apply_rope=rope_dim > 0,
        write_cache=write_cache,
        mx_scale=mx_scale,
        fp8_max=_FP8_MAX,
        use_fnuz=_USE_FNUZ,
        shuffle=shuffled,
        cache_block_tile=_INDEXER_CACHE_BLOCK_TILE,
        cache_head_tile=head_tile,
        num_warps=1,
    )


def fused_indexer_k_norm_rope(
    positions: torch.Tensor,
    k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor | None,
    is_neox: bool,
) -> torch.Tensor:
    """Return ``rope(LayerNorm_fp32(k))`` in ``k.dtype``; ``k`` is ``[T, D]``.

    RoPE covers the leading ``cos_sin_cache.shape[1]`` dims (skipped if
    ``cos_sin_cache`` is None).
    """
    out = torch.empty_like(k, memory_format=torch.contiguous_format)
    _launch_indexer_k(
        positions, k, weight, bias, eps, cos_sin_cache, is_neox, out, None, None
    )
    return out


def fused_indexer_k_norm_rope_cache_write(
    positions: torch.Tensor,
    k: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    cos_sin_cache: torch.Tensor | None,
    is_neox: bool,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    shuffled: bool | None = None,
) -> None:
    """Write ``rope(LayerNorm_fp32(k))`` to the paged FP8 indexer cache, as the
    ROCm insert ``indexer_k_quant_and_cache_triton`` does with the bf16 result
    (UE8M0 scale per 128-dim row; tokens past ``slot_mapping`` or with a negative
    slot are skipped).

    ``shuffled`` is the cache's value layout, as reported by the cache object
    (``MistralIndexerCache.uses_shuffled_layout``); None derives it from the
    block size with the same rule (``indexer_k_cache_is_shuffled``)."""
    if shuffled is None:
        shuffled = indexer_k_cache_is_shuffled(kv_cache.shape[1])
    _launch_indexer_k(
        positions,
        k,
        weight,
        bias,
        eps,
        cos_sin_cache,
        is_neox,
        None,
        slot_mapping,
        kv_cache,
        shuffled=shuffled,
    )
