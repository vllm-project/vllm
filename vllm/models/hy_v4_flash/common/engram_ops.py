# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for the HY V4 Flash Engram layer."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit(do_not_specialize=["hashes_stride", "vocab_start", "vocab_end"])
def _lookup_norm_kernel(
    table_ptr,
    scales_ptr,
    hashes_ptr,
    gamma_ptr,
    out_ptr,
    hashes_stride,
    vocab_start,
    vocab_end,
    eps,
    HEADS: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    HAS_SCALES: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, BLOCK_H)[:, None]
    lanes = tl.arange(0, BLOCK_D)[None, :]
    head_mask = heads < HEADS
    mask = head_mask & (lanes < DIM)

    bucket = tl.load(
        hashes_ptr + token * hashes_stride + heads, mask=head_mask, other=-1
    ).to(tl.int64)
    found = head_mask & (bucket >= vocab_start) & (bucket < vocab_end)
    row = tl.where(found, bucket - vocab_start, 0)
    x = tl.load(table_ptr + row * DIM + lanes, mask=mask & found, other=0.0)
    x = x.to(tl.float32)
    if HAS_SCALES:
        x *= tl.load(scales_ptr + row, mask=found, other=0.0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, axis=1) / DIM + eps)[:, None]
    gamma = tl.load(gamma_ptr + heads * DIM + lanes, mask=mask, other=0.0)
    tl.store(
        out_ptr + (token * HEADS + heads) * DIM + lanes,
        (x * rstd * gamma.to(tl.float32)).to(out_ptr.dtype.element_ty),
        mask=mask,
    )


def lookup_norm(
    table: torch.Tensor,
    scales: torch.Tensor | None,
    hashes: torch.Tensor,
    vocab_start: int,
    vocab_end: int,
    gamma: torch.Tensor,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Gather, dequantize and RMS-normalize Engram table rows per head.

    Args:
        table: `[vocab_end - vocab_start, dim]` local table rows (FP8 or
            BF16), possibly a CUDA view of pinned host memory.
        scales: `[vocab_end - vocab_start]` FP8 row scales, or None for BF16
            tables.
        hashes: `[T, heads]` global buckets; a bucket outside
            `[vocab_start, vocab_end)`, including a negative one, zeroes its
            head.
        vocab_start: First global bucket of `table`.
        vocab_end: End of the global buckets of `table`.
        gamma: `[heads, dim]` per-head norm weights.
        dtype: output dtype.

    Returns:
        `[T, heads * dim]` normalized rows.

    """
    num_tokens, heads = hashes.shape
    dim = table.shape[1]
    assert table.is_contiguous() and gamma.shape == (heads, dim)
    assert gamma.is_contiguous() and hashes.stride(1) == 1
    out = torch.empty(num_tokens, heads * dim, dtype=dtype, device=hashes.device)
    if num_tokens == 0 or heads == 0:
        return out.zero_()
    _lookup_norm_kernel[(num_tokens,)](
        table,
        table if scales is None else scales,
        hashes,
        gamma,
        out,
        hashes.stride(0),
        vocab_start,
        vocab_end,
        torch.finfo(torch.float32).eps,
        HEADS=heads,
        DIM=dim,
        BLOCK_H=triton.next_power_of_2(heads),
        BLOCK_D=triton.next_power_of_2(dim),
        HAS_SCALES=scales is not None,
        num_warps=4,
    )
    return out


@triton.jit
def _gated_update_kernel(
    hidden_ptr,
    proj_ptr,
    hashes_ptr,
    key_norm_ptr,
    query_norm_ptr,
    pre_gamma_ptr,
    hc_scales_ptr,
    out_ptr,
    hidden_stride_t,
    hidden_stride_h,
    proj_stride,
    hashes_stride,
    out_stride_t,
    out_stride_h,
    num_hash_cols,
    eps,
    DIM: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    program = tl.program_id(0)
    token = (program // HC_MULT).to(tl.int64)
    channel = program % HC_MULT
    lanes = tl.arange(0, BLOCK_D)
    mask = lanes < DIM
    dtype: tl.constexpr = hidden_ptr.dtype.element_ty

    query = tl.load(
        hidden_ptr + token * hidden_stride_t + channel * hidden_stride_h + lanes,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    proj = proj_ptr + token * proj_stride
    key = tl.load(proj + channel * DIM + lanes, mask=mask, other=0.0).to(tl.float32)
    value = tl.load(proj + HC_MULT * DIM + lanes, mask=mask, other=0.0).to(tl.float32)
    key_norm = tl.load(key_norm_ptr + channel * DIM + lanes, mask=mask, other=0.0)
    query_norm = tl.load(query_norm_ptr + channel * DIM + lanes, mask=mask, other=0.0)

    # Round at the same tensor boundaries as the eager BF16 chain.
    key = (key * tl.rsqrt(tl.sum(key * key) / DIM + eps) * key_norm.to(tl.float32)).to(
        dtype
    )
    normed_query = (
        query * tl.rsqrt(tl.sum(query * query) / DIM + eps) * query_norm.to(tl.float32)
    ).to(dtype)
    products = (key.to(tl.float32) * normed_query.to(tl.float32)).to(dtype)
    dot = tl.sum(products.to(tl.float32)).to(dtype).to(tl.float32)
    dot = (dot * (1.0 / tl.sqrt(float(DIM)))).to(dtype).to(tl.float32)
    magnitude = tl.sqrt(tl.maximum(tl.abs(dot), 1e-6)).to(dtype).to(tl.float32)
    sign = tl.where(dot > 0, 1.0, tl.where(dot < 0, -1.0, 0.0))
    gate = tl.sigmoid(sign * magnitude).to(dtype).to(tl.float32)

    pre_gamma = tl.load(pre_gamma_ptr + lanes, mask=mask, other=0.0).to(tl.float32)
    signal = pre_gamma * value * tl.rsqrt(tl.sum(value * value) / DIM + eps)
    hc_scale = tl.load(hc_scales_ptr + channel * DIM + lanes, mask=mask, other=0.0)
    update = ((gate + hc_scale.to(tl.float32)) * signal).to(dtype).to(tl.float32)

    columns = tl.arange(0, BLOCK_C)
    hashes = tl.load(
        hashes_ptr + token * hashes_stride + columns,
        mask=columns < num_hash_cols,
        other=-1,
    )
    live = tl.max((hashes >= 0).to(tl.int32)) > 0
    tl.store(
        out_ptr + token * out_stride_t + channel * out_stride_h + lanes,
        tl.where(live, query + update, query).to(dtype),
        mask=mask,
    )


def gated_update(
    hidden: torch.Tensor,
    proj: torch.Tensor,
    hashes: torch.Tensor,
    key_norm: torch.Tensor,
    query_norm: torch.Tensor,
    pre_gamma: torch.Tensor,
    hc_scales: torch.Tensor,
) -> torch.Tensor:
    """Apply the gated Engram update to the hyper-connection streams.

    Args:
        hidden: `[T, hc, dim]` streams.
        proj: `[T, (hc + 1) * dim]` reduced key projections followed by the
            value projection.
        hashes: `[T, n_hash_cols]`; tokens whose hashes are all negative pass
            through unchanged.
        key_norm: `[hc, dim]` key RMSNorm weights.
        query_norm: `[hc, dim]` query RMSNorm weights.
        pre_gamma: `[dim]` value RMSNorm weight.
        hc_scales: `[hc, dim]` per-stream update scales.

    Returns:
        `[T, hc, dim]` updated streams.

    """
    num_tokens, hc_mult, dim = hidden.shape
    assert hidden.stride(2) == 1 and proj.stride(1) == 1 and hashes.stride(1) == 1
    assert proj.shape == (num_tokens, (hc_mult + 1) * dim)
    out = torch.empty_like(hidden)
    if num_tokens == 0:
        return out
    block = triton.next_power_of_2(dim)
    _gated_update_kernel[(num_tokens * hc_mult,)](
        hidden,
        proj,
        hashes,
        key_norm,
        query_norm,
        pre_gamma,
        hc_scales,
        out,
        hidden.stride(0),
        hidden.stride(1),
        proj.stride(0),
        hashes.stride(0),
        out.stride(0),
        out.stride(1),
        hashes.shape[1],
        torch.finfo(torch.float32).eps,
        DIM=dim,
        HC_MULT=hc_mult,
        BLOCK_D=block,
        BLOCK_C=triton.next_power_of_2(hashes.shape[1]),
        num_warps=8 if block >= 2048 else 4,
    )
    return out


# Batch and token counts change every step; keep them out of the cache key.
@triton.jit(
    do_not_specialize=[
        "num_tokens",
        "num_live",
        "num_requests",
        "skip0",
        "skip1",
        "skip2",
    ]
)
def _ngram_hash_kernel(
    input_ids_ptr,
    token_map_ptr,
    lookback_ptr,
    starts_ptr,
    seq_lens_ptr,
    multipliers_ptr,
    primes_ptr,
    offsets_ptr,
    out_ptr,
    num_tokens,
    num_live,
    num_requests,
    vocab,
    pad_id,
    skip0,
    skip1,
    skip2,
    lookback_depth,
    lookback_row,
    lookback_col,
    MAX_NGRAM: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
    layer = tl.program_id(1)
    valid = token < num_tokens

    # Request of each token: the first request whose end exceeds it.
    lo = tl.full((BLOCK_T,), 0, tl.int32)
    hi = tl.full((BLOCK_T,), num_requests, tl.int32)
    while tl.sum((lo < hi).to(tl.int32), 0) > 0:
        active = lo < hi
        mid = (lo + hi) // 2
        right = token >= tl.load(starts_ptr + mid + 1, active, other=0)
        lo = tl.where(active & right, mid + 1, lo)
        hi = tl.where(active & ~right, mid, hi)
    request = tl.minimum(lo, num_requests - 1).to(tl.int64)
    start = tl.load(starts_ptr + request)
    end = tl.load(starts_ptr + request + 1)
    chunk_start = (tl.load(seq_lens_ptr + request) - (end - start)).to(tl.int64)
    position = chunk_start + token - start

    own = tl.load(input_ids_ptr + token, valid, other=-1).to(tl.int64)
    skip = (
        (own < 0)
        | (own >= vocab)
        | (token >= num_live)
        | (token >= tl.load(starts_ptr + num_requests))
        | (own == skip0)
        | (own == skip1)
        | (own == skip2)
    )

    head = tl.arange(0, BLOCK_H)
    head_mask = head < HEADS
    blocked = tl.full((BLOCK_T,), False, tl.int1)
    rolling = tl.full((BLOCK_T,), 0, tl.int64)
    for shift in tl.static_range(MAX_NGRAM):
        history = position - shift
        in_batch = history >= chunk_start
        batch_id = tl.load(
            input_ids_ptr + tl.maximum(token - shift, 0), valid & in_batch, other=0
        ).to(tl.int64)
        col = chunk_start - 1 - history
        in_window = valid & ~in_batch & (col >= 0) & (col < lookback_depth)
        window_id = tl.load(
            lookback_ptr
            + request * lookback_row
            + tl.minimum(tl.maximum(col, 0), lookback_depth - 1) * lookback_col,
            in_window,
            other=-1,
        ).to(tl.int64)
        known = window_id >= 0
        source_id = tl.where(in_batch, batch_id, window_id)
        dead = tl.where(
            in_batch, (batch_id < 0) | (batch_id >= vocab), window_id >= vocab
        )
        mapped = tl.load(
            token_map_ptr + tl.minimum(tl.maximum(source_id, 0), vocab - 1),
            valid & (in_batch | known),
            other=0,
        ).to(tl.int64)
        source = tl.where(in_batch | known, mapped, pad_id)
        blocked |= (history < 0) | ((in_batch | known) & dead)
        rolling ^= tl.where(blocked, pad_id, source) * tl.load(
            multipliers_ptr + layer * MAX_NGRAM + shift
        )
        if shift > 0:
            col = (shift - 1) * HEADS + head
            param = layer * (MAX_NGRAM - 1) * HEADS + col
            prime = tl.load(primes_ptr + param, head_mask, other=1)
            bucket = rolling[:, None] % prime[None, :]
            bucket = tl.where(bucket < 0, bucket + prime[None, :], bucket)
            bucket += tl.load(offsets_ptr + param, head_mask, other=0)[None, :]
            row = (layer * num_tokens + token).to(tl.int64)
            tl.store(
                out_ptr + row[:, None] * ((MAX_NGRAM - 1) * HEADS) + col[None, :],
                tl.where(skip[:, None], -1, bucket).to(tl.int32),
                valid[:, None] & head_mask[None, :],
            )


def ngram_hashes(
    input_ids: torch.Tensor,
    lookback: torch.Tensor,
    starts: torch.Tensor,
    seq_lens: torch.Tensor,
    num_live: int,
    token_map: torch.Tensor,
    multipliers: torch.Tensor,
    primes: torch.Tensor,
    offsets: torch.Tensor,
    pad_id: int,
    skip_ids: tuple[int, ...],
    out: torch.Tensor,
) -> None:
    """Write the global Engram buckets of the n-grams ending at each token.

    Args:
        input_ids: `[T]` token ids of the batch.
        lookback: `[requests, depth]` ids preceding each request's chunk,
            nearest first; negative ids are unknown.
        starts: `[requests + 1]` query start offsets.
        seq_lens: `[requests]` sequence lengths including the chunk.
        num_live: Rows at or past this count are padding.
        token_map: `[vocab]` compressed id of each token.
        multipliers: `[layers, max_ngram]` per-layer hash multipliers.
        primes: `[layers, columns]` bucket count of each hash column.
        offsets: `[layers, columns]` first global bucket of each column.
        pad_id: Compressed id standing in for missing history.
        skip_ids: Up to three token ids (pad and visual tokens) whose rows
            get no buckets.
        out: `[layers, T, columns]` INT32 buckets; skipped rows are -1.

    """
    num_tokens = input_ids.numel()
    layers, max_ngram = multipliers.shape
    heads = primes.shape[1] // (max_ngram - 1)
    assert out.shape == (layers, num_tokens, primes.shape[1]) and out.is_contiguous()
    assert input_ids.is_contiguous() and len(skip_ids) <= 3
    if num_tokens == 0:
        return
    skip0, skip1, skip2 = (*skip_ids, -1, -1, -1)[:3]
    block = 32
    _ngram_hash_kernel[(triton.cdiv(num_tokens, block), layers)](
        input_ids,
        token_map,
        lookback,
        starts,
        seq_lens,
        multipliers,
        primes,
        offsets,
        out,
        num_tokens,
        min(num_live, num_tokens),
        starts.numel() - 1,
        token_map.numel(),
        pad_id,
        skip0,
        skip1,
        skip2,
        lookback.shape[1],
        lookback.stride(0),
        lookback.stride(1),
        MAX_NGRAM=max_ngram,
        HEADS=heads,
        BLOCK_T=block,
        BLOCK_H=triton.next_power_of_2(heads),
        num_warps=4,
    )
