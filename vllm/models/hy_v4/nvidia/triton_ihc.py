# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2026, The FlagOS Contributors.
"""Triton iHC pre/post kernels for HY V4.

Adapted from the SGLang HY V4 implementation:
https://github.com/sgl-project/sglang/pull/36805
"""

from functools import cache

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


@cache
@torch.compiler.assume_constant_result
def _is_h100(device_index: int) -> bool:
    return "H100" in current_platform.get_device_name(device_index)


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

    tune_3072 = (
        hc_mult == 4
        and hidden_size == 3072
        and num_tokens <= 4096
        and _is_h100(x.device.index)
    )
    block_d = _BLOCK_D
    stage2_warps = 4
    if tune_3072 and num_tokens < 128:
        block_d, stage2_warps = 256, 1
    elif tune_3072 and 2048 <= num_tokens <= 4096:
        block_d = 2048

    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, hc_mult), dtype=torch.float32, device=x.device)
    hc_pow2 = triton.next_power_of_2(hc_mult)
    two_hc = 2 * hc_mult
    x_flat = x.reshape(num_tokens, k_total)

    mblock_min_tokens = 256 if tune_3072 else _MBLOCK_MIN_TOKENS
    if num_tokens < mblock_min_tokens:
        block_k = 512 if tune_3072 else _split_k_block_k(num_tokens)
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
        _ihc_pre_stage2_split_k[(num_tokens, triton.cdiv(hidden_size, block_d))](
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
            BLOCK_D=block_d,
            MAGNITUDE=magnitude,
            NORM_EPS=norm_eps,
            HC_EPS=hc_eps,
            num_warps=stage2_warps,
            enable_fp_fusion=False,
        )
        return output, post

    # split K only when the token-blocks alone cannot fill the GPU; it also
    # shortens each program's serial K chain, which lowering BLOCK_M would not
    block_m = 8 if tune_3072 else _BLOCK_M
    block_k = 128 if tune_3072 else _BLOCK_K
    grid_m = triton.cdiv(num_tokens, block_m)
    nsplit = 1
    while (
        grid_m * nsplit * 2 <= _MIN_PROGRAMS and k_total % (nsplit * 2 * block_k) == 0
    ):
        nsplit *= 2

    if tune_3072 and 512 <= num_tokens < 2048:
        nsplit *= 2
    elif tune_3072 and 2048 <= num_tokens <= 4096:
        nsplit *= 3

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
        BLOCK_M=block_m,
        BLOCK_K=block_k,
        NSPLIT=nsplit,
        NUM_STAGES=_PRE_NUM_STAGES,
        num_warps=_PRE_STAGE1_WARPS,
        enable_fp_fusion=False,
    )
    _ihc_pre_stage2[(num_tokens, triton.cdiv(hidden_size, block_d))](
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
        BLOCK_D=block_d,
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


@triton.jit
def _ihc_post_pre_split_k(
    x_ptr,
    residual_ptr,
    attn_post_ptr,
    weight_ptr,
    y_ptr,
    partial_ptr,
    HIDDEN_SIZE: tl.constexpr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    TWO_HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    k = split * BLOCK_K + tl.arange(0, BLOCK_K)
    mask = k < K_TOTAL
    channel = k // HIDDEN_SIZE
    gate = tl.load(attn_post_ptr + token * HC_MULT + channel, mask=mask, other=0.0)
    x = tl.load(x_ptr + token * HIDDEN_SIZE + k % HIDDEN_SIZE, mask=mask, other=0.0).to(
        tl.float32
    )
    residual = tl.load(residual_ptr + token * K_TOTAL + k, mask=mask, other=0.0).to(
        tl.float32
    )
    y = (gate * x + residual).to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + token * K_TOTAL + k, y, mask=mask)
    y = y.to(tl.float32)
    rows = tl.arange(0, TWO_HC_POW2)
    weight = tl.load(
        weight_ptr + rows[:, None] * K_TOTAL + k[None, :],
        mask=(rows[:, None] < 2 * HC_MULT) & mask[None, :],
        other=0.0,
    )
    mixes = tl.sum(weight * y[None, :], axis=1)
    partial = partial_ptr + (token * NUM_SPLITS + split) * PARTIAL_STRIDE
    tl.store(partial, tl.sum(y * y, axis=0))
    tl.store(partial + 1 + rows, mixes, mask=rows < 2 * HC_MULT)


@triton.jit
def _ihc_post_pre_projection(
    x_ptr,
    residual_ptr,
    gates_ptr,
    weight_ptr,
    y_ptr,
    partial_ptr,
    N,
    D: tl.constexpr,
    BM: tl.constexpr,
    BK: tl.constexpr,
    SPLITS: tl.constexpr,
    STAGES: tl.constexpr,
):
    tokens = tl.program_id(0).to(tl.int64) * BM + tl.arange(0, BM)
    split = tl.program_id(1)
    rows = tl.arange(0, 16)
    squares = tl.zeros((BM,), tl.float32)
    mixes = tl.zeros((BM, 16), tl.float64)
    for off in tl.range(0, D // SPLITS, BK, num_stages=STAGES):
        d = split * (D // SPLITS) + off + tl.arange(0, BK)
        x = tl.load(
            x_ptr + tokens[:, None] * D + d[None, :],
            mask=(tokens[:, None] < N) & (d[None, :] < D),
            other=0.0,
        ).to(tl.float32)
        for c in tl.range(0, 4, loop_unroll_factor=1):
            gate = tl.load(gates_ptr + tokens * 4 + c, mask=tokens < N, other=0.0)
            residual = tl.load(
                residual_ptr + (tokens[:, None] * 4 + c) * D + d[None, :],
                mask=(tokens[:, None] < N) & (d[None, :] < D),
                other=0.0,
            ).to(tl.float32)
            y = (gate[:, None] * x + residual).to(y_ptr.dtype.element_ty)
            tl.store(
                y_ptr + (tokens[:, None] * 4 + c) * D + d[None, :],
                y,
                mask=(tokens[:, None] < N) & (d[None, :] < D),
            )
            y = y.to(tl.float32)
            squares += tl.sum(y * y, axis=1)
            offset = rows[None, :] * 4 * D + c * D + d[:, None]
            valid = (rows[None, :] < 8) & (d[:, None] < D)
            w_hi = tl.load(weight_ptr + offset, mask=valid, other=0.0)
            w_lo = tl.load(weight_ptr + 8 * 4 * D + offset, mask=valid, other=0.0)
            w_tail = tl.load(weight_ptr + 2 * 8 * 4 * D + offset, mask=valid, other=0.0)
            mixes += tl.dot(y.to(tl.bfloat16), w_hi).to(tl.float64)
            mixes += tl.dot(y.to(tl.bfloat16), w_lo).to(tl.float64)
            mixes += tl.dot(y.to(tl.bfloat16), w_tail).to(tl.float64)
    partial = partial_ptr + (tokens * SPLITS + split) * 9
    tl.store(partial, squares, mask=tokens < N)
    tl.store(
        partial[:, None] + 1 + rows[None, :],
        mixes,
        mask=(tokens[:, None] < N) & (rows[None, :] < 8),
    )


@triton.jit
def _ihc_pack_weight(w_ptr, out_ptr, K: tl.constexpr, BLOCK: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w = tl.load(w_ptr + off, mask=off < K, other=0.0)
    hi = w.to(tl.bfloat16)
    lo = (w - hi.to(tl.float32)).to(tl.bfloat16)
    tail = (w - hi.to(tl.float32) - lo.to(tl.float32)).to(tl.bfloat16)
    tl.store(out_ptr + off, hi, mask=off < K)
    tl.store(out_ptr + K + off, lo, mask=off < K)
    tl.store(out_ptr + 2 * K + off, tail, mask=off < K)


@triton.jit
def _ihc_post_pre_stage2(
    y_ptr,
    sumsq_ptr,
    scale_ptr,
    base_ptr,
    norm_weight_ptr,
    output_ptr,
    post_ptr,
    N,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_S: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    HC_EPS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    FUSE_NORM: tl.constexpr,
    RMS_EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    hidden_block = tl.program_id(1)
    splits = tl.arange(0, BLOCK_S)
    rows = tl.arange(0, 2 * HC_POW2)
    partial = sumsq_ptr + (token * NUM_SPLITS + splits) * PARTIAL_STRIDE
    squares = tl.load(partial, mask=splits < NUM_SPLITS, other=0.0)
    mixes = tl.load(
        partial[:, None] + 1 + rows[None, :],
        mask=(splits[:, None] < NUM_SPLITS) & (rows[None, :] < 2 * HC_MULT),
        other=0.0,
    )
    mixes = tl.sum(mixes, axis=0).to(tl.float32)
    inv_rms = tl.rsqrt(
        tl.sum(squares, axis=0).to(tl.float32) / (HC_MULT * HIDDEN_SIZE) + NORM_EPS
    )
    hc = tl.arange(0, HC_POW2)
    pre_mix = tl.sum(tl.where(rows[None, :] == hc[:, None], mixes[None, :], 0), 1)
    post_mix = tl.sum(
        tl.where(rows[None, :] == hc[:, None] + HC_MULT, mixes[None, :], 0), 1
    )
    pre = (
        tl.sigmoid(
            pre_mix * inv_rms * tl.load(scale_ptr)
            + tl.load(base_ptr + hc, mask=hc < HC_MULT, other=0.0)
        )
        + HC_EPS
    )
    if hidden_block == 0:
        post = (
            MAGNITUDE
            * tl.sigmoid(
                post_mix * inv_rms * tl.load(scale_ptr + 1)
                + tl.load(base_ptr + HC_MULT + hc, mask=hc < HC_MULT, other=0.0)
            )
            + HC_EPS
        )
        tl.store(post_ptr + token * HC_MULT + hc, post, mask=hc < HC_MULT)
    d = hidden_block * BLOCK_D + tl.arange(0, BLOCK_D)
    output = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for channel in tl.static_range(HC_MULT):
        y = tl.load(
            y_ptr + (token * HC_MULT + channel) * HIDDEN_SIZE + d,
            mask=d < HIDDEN_SIZE,
            other=0.0,
        ).to(tl.float32)
        gate = tl.sum(tl.where(hc == channel, pre, 0.0), axis=0)
        output += gate * y
    output = output.to(output_ptr.dtype.element_ty).to(tl.float32)
    if FUSE_NORM:
        squares = tl.where(d < HIDDEN_SIZE, output * output, 0.0)
        inv_rms = tl.rsqrt(tl.sum(squares, axis=0) / HIDDEN_SIZE + RMS_EPS)
        norm_weight = tl.load(norm_weight_ptr + d, mask=d < HIDDEN_SIZE, other=0.0).to(
            tl.float32
        )
        output = output * inv_rms * norm_weight
    tl.store(output_ptr + token * HIDDEN_SIZE + d, output, mask=d < HIDDEN_SIZE)


@triton.jit
def _ihc_rms_norm(
    x_ptr,
    weight_ptr,
    output_ptr,
    D: tl.constexpr,
    BLOCK_D: tl.constexpr,
    EPS: tl.constexpr,
):
    token = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, BLOCK_D)
    x = tl.load(x_ptr + token * D + d, mask=d < D, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + d, mask=d < D, other=0.0).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / D + EPS)
    tl.store(output_ptr + token * D + d, x * inv_rms * weight, mask=d < D)


def triton_ihc_post_pre_supported(x: torch.Tensor) -> bool:
    """Use the measured fused schedule for the input shape and device."""
    return (
        triton_ihc_supported(x)
        and x.ndim == 2
        and 0 < x.shape[-1] <= 8192
        and (
            x.shape[0] <= 512
            or (
                x.dtype == torch.bfloat16
                and x.shape[-1] == 3072
                and _is_h100(x.device.index)
            )
        )
    )


def triton_ihc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    attn_post: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    norm_weight: torch.Tensor | None = None,
    rms_eps: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse post, the next pre, and optional RMSNorm.

    Returns the updated residual, reduced (optionally normalized) activation,
    and the next post gates, in the same order as `hpc.fuse_ihc_post_pre`.
    Post and pre outputs round to the activation dtype before the next step.
    Large BF16 H100 batches reuse projection weights across token tiles.
    """
    assert x.ndim == 2 and residual.ndim == 3 and attn_post.ndim == 2
    assert x.dtype in (torch.float16, torch.bfloat16)
    assert residual.dtype == x.dtype and attn_post.dtype == torch.float32
    n, d = x.shape
    hc = residual.shape[1]
    assert hc > 0 and d > 0
    assert residual.shape == (n, hc, d) and attn_post.shape == (n, hc)
    assert weight.shape == (2 * hc, hc * d)
    assert scale.shape == (2,) and base.shape == (2 * hc,)
    assert all(t.dtype == torch.float32 for t in (weight, scale, base))
    tensors = (x, residual, attn_post, weight, scale, base)
    assert all(t.is_cuda and t.device == x.device for t in tensors)
    if norm_weight is not None:
        assert norm_weight.shape == (d,)
        assert norm_weight.dtype in (x.dtype, torch.float32)
        assert norm_weight.device == x.device
        norm_weight = norm_weight.contiguous()
    x, residual, attn_post, weight, scale, base = (t.contiguous() for t in tensors)
    use_projection = (
        n >= 1024
        and hc == 4
        and d == 3072
        and x.dtype == torch.bfloat16
        and _is_h100(x.device.index)
    )
    if n > 512 and not use_projection and hc & (hc - 1) == 0:
        y = triton_ihc_post(x, residual, attn_post)
        reduced, post = triton_ihc_pre(
            y, weight, scale, base, magnitude, hc_eps, norm_eps
        )
        if norm_weight is None:
            return y, reduced, post
        output = torch.empty_like(reduced)
        _ihc_rms_norm[(n,)](
            reduced,
            norm_weight,
            output,
            d,
            triton.next_power_of_2(d),
            rms_eps,
            num_warps=4,
            enable_fp_fusion=False,
        )
        return y, output, post
    y = torch.empty((n, hc, d), dtype=x.dtype, device=x.device)
    output = torch.empty((n, d), dtype=x.dtype, device=x.device)
    post = torch.empty((n, hc), dtype=torch.float32, device=x.device)
    if n == 0:
        return y, output, post
    k = hc * d
    hc_pow2 = triton.next_power_of_2(hc)
    block_d = triton.next_power_of_2(d) if norm_weight is not None else 1024
    partial_stride = 1 + 2 * hc
    if use_projection:
        block_m = 32
        block_k = 64
        if n < 4096:
            splits = 12 if n < 2048 else 6
        elif n < 8192:
            splits = 12
        else:
            splits = 6 if n < 16384 else 3
    else:
        block_k = 512
        splits = triton.cdiv(k, block_k)
    sumsq = torch.empty(
        (n, splits, partial_stride),
        dtype=torch.float64 if use_projection else torch.float32,
        device=x.device,
    )
    if use_projection:
        # Short BF16 dot products feed FP64 partial sums to preserve gate
        # precision, including when the learned scales amplify dot errors.
        packed = torch.empty((3, 8, k), dtype=torch.bfloat16, device=x.device)
        _ihc_pack_weight[(triton.cdiv(8 * k, 1024),)](weight, packed, 8 * k, 1024)
        _ihc_post_pre_projection[(triton.cdiv(n, block_m), splits)](
            x,
            residual,
            attn_post,
            packed,
            y,
            sumsq,
            n,
            d,
            block_m,
            block_k,
            splits,
            1,
            num_warps=2,
            enable_fp_fusion=False,
        )
    else:
        _ihc_post_pre_split_k[(n, splits)](
            x,
            residual,
            attn_post,
            weight,
            y,
            sumsq,
            d,
            k,
            hc,
            2 * hc_pow2,
            splits,
            block_k,
            partial_stride,
            num_warps=1,
            enable_fp_fusion=False,
        )
    _ihc_post_pre_stage2[(n, triton.cdiv(d, block_d))](
        y,
        sumsq,
        scale,
        base,
        norm_weight if norm_weight is not None else output,
        output,
        post,
        n,
        d,
        hc,
        hc_pow2,
        splits,
        triton.next_power_of_2(splits),
        partial_stride,
        block_d,
        magnitude,
        hc_eps,
        norm_eps,
        norm_weight is not None,
        rms_eps,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return y, output, post
