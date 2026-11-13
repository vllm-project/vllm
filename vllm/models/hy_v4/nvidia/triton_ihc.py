# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026, The FlagOS Contributors.
"""Triton iHC pre/post kernels for HY V4.

Adapted from the SGLang HY V4 implementation:
https://github.com/sgl-project/sglang/pull/36805
"""

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton

_BLOCK_D = 1024

# the post kernel tiles [hc, BLOCK_D] at once, so it wants a narrower slice
_POST_BLOCK_D = 512
_POST_NUM_WARPS = 4

# CUDA caps grid dims y and z at 65535
_MAX_GRID_DIM = 65535

_BLOCK_K = 64
_BLOCK_M = 32
_PRE_STAGE1_WARPS = 1
_PRE_NUM_STAGES = 3
_MIN_PROGRAMS = 512

# below this the M-blocked path starves, so dispatch on token count alone: a
# decode batch, a short prefill and a chunked-prefill tail all land here
_MBLOCK_MIN_TOKENS = 512
# few tokens need K cut finely to expose programs, more prefer wider chunks
_SPLIT_K_BLOCK_K = ((2, 128), (16, 256), (64, 512), (None, 1024))


def _split_k_block_k(num_tokens: int) -> int:
    for limit, block_k in _SPLIT_K_BLOCK_K:
        if limit is None or num_tokens < limit:
            return block_k
    return _SPLIT_K_BLOCK_K[-1][1]


def triton_ihc_supported(x: torch.Tensor) -> bool:
    """Return whether the in-tree Triton path can run for this input."""
    return (
        HAS_TRITON
        and current_platform.is_cuda()
        and x.is_cuda
        and x.dtype in (torch.float16, torch.bfloat16)
        and not envs.VLLM_BATCH_INVARIANT
    )


@triton.jit
def _ihc_pre_stage1(
    x_ptr,
    weight_ptr,
    sumsq_slot_ptr,
    mix_slot_ptr,
    N,
    X_STRIDE: tl.constexpr,
    K_CHUNK: tl.constexpr,
    TWO_HC: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NSPLIT: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    # X_STRIDE is the whole row, K_CHUNK only this program's share of it
    token_block_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)

    m_off = token_block_idx * BLOCK_M + tl.arange(0, BLOCK_M)
    m_mask = m_off < N

    k_start = split_idx * K_CHUNK

    # fp32 throughout; tl.dot in bf16 blows the gate tolerance at K = hc*hidden
    sum_squares = tl.zeros((BLOCK_M,), dtype=tl.float32)
    mixes = ()
    for _ in tl.static_range(TWO_HC):
        mixes += (tl.zeros((BLOCK_M,), dtype=tl.float32),)  # type: ignore[assignment]

    x_base = x_ptr + m_off[:, None] * X_STRIDE
    for k_offset in tl.range(0, K_CHUNK, BLOCK_K, num_stages=NUM_STAGES):
        k_off = k_start + k_offset + tl.arange(0, BLOCK_K)
        k_mask = k_off < X_STRIDE

        x = tl.load(
            x_base + k_off[None, :],
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)

        sum_squares += tl.sum(x * x, axis=1)

        # reuse the x tile for every projection row, instead of re-reading it
        next_mixes = ()
        for j in tl.static_range(TWO_HC):
            weight_row = tl.load(
                weight_ptr + j * X_STRIDE + k_off, mask=k_mask, other=0.0
            )
            contribution = tl.sum(x * weight_row[None, :], axis=1)
            next_mixes += (mixes[j] + contribution,)  # type: ignore[assignment]
        mixes = next_mixes

    # private slot per (split, token); stage2 sums them in index order, so the
    # result does not depend on which block lands first
    tl.store(sumsq_slot_ptr + split_idx * N + m_off, sum_squares, mask=m_mask)
    mix_slot = mix_slot_ptr + split_idx * N * TWO_HC + m_off * TWO_HC
    for j in tl.static_range(TWO_HC):
        tl.store(mix_slot + j, mixes[j], mask=m_mask)


@triton.jit
def _ihc_pre_stage1_split_k(
    x_ptr,
    weight_ptr,
    partial_ptr,
    K_TOTAL: tl.constexpr,
    TWO_HC: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    # one program per (token, K-chunk); too few tokens for weight reuse to pay
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    k_off = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_off < K_TOTAL

    x = tl.load(
        x_ptr + token_idx * K_TOTAL + k_off,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)

    sum_squares = tl.sum(x * x, axis=0)

    # all 2*hc weight rows as one tile: one reduction, not one pass per row
    mix_idx = tl.arange(0, TWO_HC)
    weight = tl.load(
        weight_ptr + mix_idx[:, None] * K_TOTAL + k_off[None, :],
        mask=k_mask[None, :],
        other=0.0,
    )
    mixes = tl.sum(weight * x[None, :], axis=1)

    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, sum_squares)
    tl.store(partial + 1 + mix_idx, mixes)


@triton.jit
def _ihc_pre_stage2(
    x_ptr,
    mix_slot_ptr,
    sumsq_slot_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    N,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    TWO_HC: tl.constexpr,
    NSPLIT: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    # reducing the slots here rather than in a third kernel keeps pre at two
    # launches
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    split_idx = tl.arange(0, BLOCK_S)
    split_mask = split_idx < NSPLIT

    sum_squares = tl.sum(
        tl.load(sumsq_slot_ptr + split_idx * N + token_idx, mask=split_mask, other=0.0),
        axis=0,
    )

    mix_idx = tl.arange(0, TWO_HC)
    mix_parts = tl.load(
        mix_slot_ptr
        + split_idx[:, None] * N * TWO_HC
        + token_idx * TWO_HC
        + mix_idx[None, :],
        mask=split_mask[:, None],
        other=0.0,
    )
    mixes = tl.sum(mix_parts, axis=0)

    # first hc entries are the pre logits, next hc the post ones
    is_pre = mix_idx[None, :] == hc_idx[:, None]
    is_post = mix_idx[None, :] == (hc_idx[:, None] + HC_MULT)
    pre_mix = tl.sum(tl.where(is_pre, mixes[None, :], 0.0), axis=1)
    post_mix = tl.sum(tl.where(is_post, mixes[None, :], 0.0), axis=1)

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS

    if hidden_block_idx == 0:
        post = (
            MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
            + HC_EPS
        )
        tl.store(
            post_ptr + token_idx * HC_MULT + hc_idx,
            post,
            mask=hc_mask,
        )

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel

    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


@triton.jit
def _ihc_pre_stage2_split_k(
    x_ptr,
    partial_ptr,
    scale_ptr,
    base_ptr,
    output_ptr,
    post_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    TWO_HC: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_S: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
):
    # same gating as _ihc_pre_stage2, but reduces stage1's partial slots
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    split_idx = tl.arange(0, BLOCK_S)
    split_mask = split_idx < NUM_SPLITS
    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    slot_ptrs = partial_row + split_idx * PARTIAL_STRIDE

    # slot layout is [sqrsum, mix[0 .. 2*hc-1]]
    sqrsum_parts = tl.load(slot_ptrs, mask=split_mask, other=0.0)
    sum_squares = tl.sum(sqrsum_parts, axis=0)

    mix_idx = tl.arange(0, TWO_HC)
    mix_ptrs = slot_ptrs[:, None] + 1 + mix_idx[None, :]
    mix_parts = tl.load(mix_ptrs, mask=split_mask[:, None], other=0.0)
    mixes = tl.sum(mix_parts, axis=0)

    is_pre = mix_idx[None, :] == hc_idx[:, None]
    is_post = mix_idx[None, :] == (hc_idx[:, None] + HC_MULT)
    pre_mix = tl.sum(tl.where(is_pre, mixes[None, :], 0.0), axis=1)
    post_mix = tl.sum(tl.where(is_post, mixes[None, :], 0.0), axis=1)

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    post_scale = tl.load(scale_ptr + 1)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS

    if hidden_block_idx == 0:
        post = (
            MAGNITUDE * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
            + HC_EPS
        )
        tl.store(post_ptr + token_idx * HC_MULT + hc_idx, post, mask=hc_mask)

    hidden_offsets = hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    x_row = x_ptr + token_idx * K_TOTAL
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel_idx in tl.static_range(HC_MULT):
        channel = tl.load(
            x_row + channel_idx * HIDDEN_SIZE + hidden_offsets,
            mask=hidden_mask,
            other=0.0,
        ).to(tl.float32)
        channel_gate = tl.sum(tl.where(hc_idx == channel_idx, pre, 0.0), axis=0)
        output += channel_gate * channel

    tl.store(
        output_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


@triton.jit
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    output_ptr,
    N,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    BLOCK_D: tl.constexpr,
    TOKENS_PER_PROGRAM: tl.constexpr,
    EVEN_D: tl.constexpr,
    EVEN_HC: tl.constexpr,
):
    """out[n, i, :] = post[n, i] * x[n, :] + residual[n, i, :]"""
    # pure streaming, so all that is left is the access pattern: one
    # [hc, BLOCK_D] tile makes the residual loads and stores single 2D
    # accesses, and the D-major grid keeps consecutive programs contiguous.
    # tokens go on the grid's y dim, which caps at 65535, so each program
    # walks TOKENS_PER_PROGRAM of them -- 1 unless the batch exceeds the cap
    hidden_block_idx = tl.program_id(0)
    token_base = tl.program_id(1).to(tl.int64) * TOKENS_PER_PROGRAM

    hc_idx = tl.arange(0, HC_POW2)[:, None]
    hidden_offsets = (hidden_block_idx * BLOCK_D + tl.arange(0, BLOCK_D))[None, :]

    for token_step in tl.range(0, TOKENS_PER_PROGRAM):
        token_idx = token_base + token_step
        if token_idx < N:
            gate_ptr = post_ptr + token_idx * HC_MULT + hc_idx
            if EVEN_HC:
                gate = tl.load(gate_ptr)
            else:
                gate = tl.load(gate_ptr, mask=hc_idx < HC_MULT, other=0.0)

            offsets = (
                token_idx * HC_MULT * HIDDEN_SIZE
                + hc_idx * HIDDEN_SIZE
                + hidden_offsets
            )
            if EVEN_D and EVEN_HC:
                x = tl.load(x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets).to(
                    tl.float32
                )
                residual = tl.load(residual_ptr + offsets).to(tl.float32)
                output = gate * x + residual
                tl.store(output_ptr + offsets, output.to(output_ptr.dtype.element_ty))
            else:
                hidden_mask = hidden_offsets < HIDDEN_SIZE
                mask = hidden_mask if EVEN_HC else (hc_idx < HC_MULT) & hidden_mask
                x = tl.load(
                    x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
                    mask=hidden_mask,
                    other=0.0,
                ).to(tl.float32)
                residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(
                    tl.float32
                )
                output = gate * x + residual
                tl.store(
                    output_ptr + offsets,
                    output.to(output_ptr.dtype.element_ty),
                    mask=mask,
                )


def triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce iHC channels and produce the post gates."""
    assert x.dim() == 3
    assert x.is_cuda and weight.is_cuda and scale.is_cuda and base.is_cuda
    assert weight.dtype == torch.float32
    assert scale.dtype == torch.float32 and base.dtype == torch.float32

    x = x.contiguous()
    assert weight.is_contiguous()
    scale = scale.contiguous()
    base = base.contiguous()
    num_tokens, hc_mult, hidden_size = x.shape
    k_total = hc_mult * hidden_size
    assert weight.shape == (2 * hc_mult, k_total)
    assert scale.shape == (2,)
    assert base.shape == (2 * hc_mult,)

    if num_tokens == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, hc_mult), dtype=torch.float32, device=x.device),
        )

    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    hc_pow2 = triton.next_power_of_2(hc_mult)
    two_hc = 2 * hc_mult
    x_flat = x.reshape(num_tokens, k_total)

    if num_tokens < _MBLOCK_MIN_TOKENS:
        block_k = _split_k_block_k(num_tokens)
        num_splits = triton.cdiv(k_total, block_k)
        partial_stride = 1 + two_hc
        partial = torch.empty(
            (num_tokens, num_splits, partial_stride),
            dtype=torch.float32,
            device=x.device,
        )
        _ihc_pre_stage1_split_k[(num_tokens, num_splits)](
            x_flat,
            weight,
            partial,
            K_TOTAL=k_total,
            TWO_HC=two_hc,
            NUM_SPLITS=num_splits,
            BLOCK_K=block_k,
            PARTIAL_STRIDE=partial_stride,
            num_warps=1,
            enable_fp_fusion=False,
        )
        _ihc_pre_stage2_split_k[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
            x,
            partial,
            scale,
            base,
            output,
            post,
            HIDDEN_SIZE=hidden_size,
            K_TOTAL=k_total,
            HC_MULT=hc_mult,
            HC_POW2=hc_pow2,
            TWO_HC=two_hc,
            NUM_SPLITS=num_splits,
            PARTIAL_STRIDE=partial_stride,
            BLOCK_S=triton.next_power_of_2(num_splits),
            BLOCK_D=_BLOCK_D,
            MAGNITUDE=magnitude,
            NORM_EPS=norm_eps,
            HC_EPS=hc_eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return output, post

    # split K only when the token-blocks alone cannot fill the GPU; it also
    # shortens each program's serial K chain, which lowering BLOCK_M would not
    grid_m = triton.cdiv(num_tokens, _BLOCK_M)
    nsplit = 1
    while (
        grid_m * nsplit * 2 <= _MIN_PROGRAMS and k_total % (nsplit * 2 * _BLOCK_K) == 0
    ):
        nsplit *= 2

    sumsq = torch.empty(nsplit * num_tokens, dtype=torch.float32, device=x.device)
    mix = torch.empty(
        nsplit * num_tokens * two_hc, dtype=torch.float32, device=x.device
    )
    _ihc_pre_stage1[(grid_m, nsplit)](
        x_flat,
        weight,
        sumsq,
        mix,
        num_tokens,
        X_STRIDE=k_total,
        K_CHUNK=k_total // nsplit,
        TWO_HC=two_hc,
        BLOCK_M=_BLOCK_M,
        BLOCK_K=_BLOCK_K,
        NSPLIT=nsplit,
        NUM_STAGES=_PRE_NUM_STAGES,
        num_warps=_PRE_STAGE1_WARPS,
        enable_fp_fusion=False,
    )
    _ihc_pre_stage2[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
        x,
        mix,
        sumsq,
        scale,
        base,
        output,
        post,
        num_tokens,
        HIDDEN_SIZE=hidden_size,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        TWO_HC=two_hc,
        NSPLIT=nsplit,
        BLOCK_S=triton.next_power_of_2(nsplit),
        BLOCK_D=_BLOCK_D,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, post


def triton_ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Scatter a sub-block output back over the iHC residual channels."""
    assert x.dim() == 2
    assert x.is_cuda and residual.is_cuda and post.is_cuda
    assert post.dtype == torch.float32

    x = x.contiguous()
    residual = residual.contiguous()
    post = post.contiguous()
    num_tokens, hidden_size = x.shape
    hc_mult = post.shape[-1]
    assert residual.shape == (num_tokens, hc_mult, hidden_size)
    assert post.shape == (num_tokens, hc_mult)

    if num_tokens == 0:
        return torch.empty((0, hc_mult, hidden_size), dtype=x.dtype, device=x.device)

    output = torch.empty(
        (num_tokens, hc_mult, hidden_size), dtype=x.dtype, device=x.device
    )
    hc_pow2 = triton.next_power_of_2(hc_mult)
    # tokens sit on the grid's y dim for D-major contiguity, so fold them into
    # fewer programs when the batch would overrun the 65535 cap
    tokens_per_program = triton.cdiv(num_tokens, _MAX_GRID_DIM)
    grid_t = triton.cdiv(num_tokens, tokens_per_program)
    # unfused mul/add, to stay bitwise identical to the torch reference
    _ihc_post_kernel[(triton.cdiv(hidden_size, _POST_BLOCK_D), grid_t)](
        x,
        residual,
        post,
        output,
        num_tokens,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        BLOCK_D=_POST_BLOCK_D,
        TOKENS_PER_PROGRAM=tokens_per_program,
        EVEN_D=(hidden_size % _POST_BLOCK_D == 0),
        EVEN_HC=(hc_pow2 == hc_mult),
        num_warps=_POST_NUM_WARPS,
        enable_fp_fusion=False,
    )
    return output
