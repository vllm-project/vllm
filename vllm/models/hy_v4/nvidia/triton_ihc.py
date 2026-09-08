# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton iHC pre / post / head kernels for HY V4.

Adapted from the SGLang HY V4 implementation:
https://github.com/sgl-project/sglang/pull/36805

The ops are registered as custom ops with fake impls
(torch.ops.vllm.hy_v4_ihc_pre / post / head) so they stay opaque to
torch.compile and are safe inside CUDA graphs. head is pre without the post
gates (HAS_POST=False) and shares its two kernels.
"""

import functools

import torch

from vllm import envs
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_BLOCK_K = 1024
_BLOCK_D = 1024


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
    partial_ptr,
    K_TOTAL: tl.constexpr,
    HC_MULT: tl.constexpr,
    HC_POW2: tl.constexpr,
    NUM_SPLITS: tl.constexpr,
    BLOCK_K: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    HAS_POST: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    split_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    k_offsets = split_idx * BLOCK_K + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < K_TOTAL
    x = tl.load(
        x_ptr + token_idx * K_TOTAL + k_offsets,
        mask=k_mask,
        other=0.0,
    ).to(tl.float32)
    sum_squares = tl.sum(x * x, axis=0)

    weight_offsets = hc_idx[:, None] * K_TOTAL + k_offsets[None, :]
    weight_mask = hc_mask[:, None] & k_mask[None, :]
    pre_mix = tl.sum(
        tl.load(weight_ptr + weight_offsets, mask=weight_mask, other=0.0) * x[None, :],
        axis=1,
    )
    partial = partial_ptr + (token_idx * NUM_SPLITS + split_idx) * PARTIAL_STRIDE
    tl.store(partial, sum_squares)
    tl.store(partial + 1 + hc_idx, pre_mix, mask=hc_mask)
    if HAS_POST:
        post_mix = tl.sum(
            tl.load(
                weight_ptr + HC_MULT * K_TOTAL + weight_offsets,
                mask=weight_mask,
                other=0.0,
            )
            * x[None, :],
            axis=1,
        )
        tl.store(partial + 1 + HC_POW2 + hc_idx, post_mix, mask=hc_mask)


@triton.jit
def _ihc_pre_stage2(
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
    NUM_SPLITS: tl.constexpr,
    PARTIAL_STRIDE: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MAGNITUDE: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    HAS_POST: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    hidden_block_idx = tl.program_id(1)
    hc_idx = tl.arange(0, HC_POW2)
    hc_mask = hc_idx < HC_MULT

    partial_row = partial_ptr + token_idx * NUM_SPLITS * PARTIAL_STRIDE
    sum_squares = tl.zeros((), dtype=tl.float32)
    pre_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    post_mix = tl.zeros((HC_POW2,), dtype=tl.float32)
    for split_idx in tl.static_range(NUM_SPLITS):
        partial = partial_row + split_idx * PARTIAL_STRIDE
        sum_squares += tl.load(partial)
        pre_mix += tl.load(partial + 1 + hc_idx, mask=hc_mask, other=0.0)
        if HAS_POST:
            post_mix += tl.load(
                partial + 1 + HC_POW2 + hc_idx,
                mask=hc_mask,
                other=0.0,
            )

    reciprocal_rms = tl.rsqrt(sum_squares / K_TOTAL + NORM_EPS)
    pre_scale = tl.load(scale_ptr)
    pre_base = tl.load(base_ptr + hc_idx, mask=hc_mask, other=0.0)
    pre = tl.sigmoid(pre_mix * reciprocal_rms * pre_scale + pre_base) + HC_EPS

    if HAS_POST:  # noqa: SIM102 - constexpr branch, keep apart from the runtime one
        if hidden_block_idx == 0:
            post_scale = tl.load(scale_ptr + 1)
            post_base = tl.load(base_ptr + HC_MULT + hc_idx, mask=hc_mask, other=0.0)
            post = (
                MAGNITUDE
                * tl.sigmoid(post_mix * reciprocal_rms * post_scale + post_base)
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
def _ihc_post_kernel(
    x_ptr,
    residual_ptr,
    post_ptr,
    output_ptr,
    HIDDEN_SIZE: tl.constexpr,
    HC_MULT: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """output[t, c, :] = post[t, c] * x[t, :] + residual[t, c, :].

    One program per (token, channel, hidden-dim tile), tile fastest, so
    consecutive programs stream consecutive memory of the ``[T, HC, D]``
    residual / output (the op is bandwidth-bound; x is re-read from L2).
    """
    pid = tl.program_id(0).to(tl.int64)
    n_tiles: tl.constexpr = (HIDDEN_SIZE + BLOCK_D - 1) // BLOCK_D
    tile = pid % n_tiles
    channel_idx = (pid // n_tiles) % HC_MULT
    token_idx = pid // (n_tiles * HC_MULT)

    hidden_offsets = tile * BLOCK_D + tl.arange(0, BLOCK_D)
    hidden_mask = hidden_offsets < HIDDEN_SIZE
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    x = tl.load(
        x_ptr + token_idx * HIDDEN_SIZE + hidden_offsets,
        mask=hidden_mask,
        other=0.0,
    ).to(tl.float32)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    channel_gate = tl.load(post_ptr + token_idx * HC_MULT + channel_idx)
    row = (token_idx * HC_MULT + channel_idx) * HIDDEN_SIZE
    residual = tl.load(
        residual_ptr + row + hidden_offsets, mask=hidden_mask, other=0.0
    ).to(tl.float32)
    output = channel_gate * x + residual
    tl.store(
        output_ptr + row + hidden_offsets,
        output.to(output_ptr.dtype.element_ty),
        mask=hidden_mask,
    )


@functools.cache
def _sm_count(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


# Tensor-core path for large batches (prefill, high-concurrency decode): a
# *stats* launch (row sum-of-squares + the hc_fn projection with tl.dot, the
# fp32 weight split into a bf16 hi+lo pair, hidden dim split across programs)
# and an *apply* launch (gates + channel reduction). Below _LARGE_T_MIN tokens
# the two-stage kernels above are faster (crossover between 96 and 128 tokens
# under a cold L2 on RTX 5090 and RTX PRO 6000).
_LARGE_T_MIN = 128
_LARGE_CFG = {
    # tl.dot needs M >= 16; rows beyond T are masked. Larger row blocks
    # amortize the weight slice each program streams from L2.
    "block_t_small": 16,  # T < 128
    "block_t_mid": 32,  # 128 <= T < 1024
    "block_t_large": 64,  # T >= 1024
    "block_d": 64,
    "programs_per_sm": 4,
    "max_split": 32,  # bounds the partials each apply program reduces
    "warps": 4,
    "stages": 1,
}


@triton.jit
def _ihc_stats_kernel(
    x_ptr,
    w_ptr,
    ws_ptr,
    T,
    stride_xt,
    stride_xc,
    stride_ws_t,
    stride_ws_s,
    D_PER_SPLIT,
    D: tl.constexpr,
    HC: tl.constexpr,
    N_OUT: tl.constexpr,
    N_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Partial sum(x^2) and partial x_flat @ w.T for a block of rows and a
    slice of the hidden dim (same slice in every hc channel).

    The dot runs on tensor cores with the fp32 weight split into a bf16
    hi + lo pair (x itself is bf16, so this keeps ~fp32 accuracy while the
    kernel stays purely memory bound).
    """
    pid_t = tl.program_id(0)
    split = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < T
    outs = tl.arange(0, N_PAD)
    out_mask = outs < N_OUT

    d_start = split * D_PER_SPLIT
    d_end = tl.minimum(d_start + D_PER_SPLIT, D)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    sq = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
    acc = tl.zeros([BLOCK_T, N_PAD], dtype=tl.float32)
    for d0 in range(d_start, d_end, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < d_end
        for c in tl.static_range(HC):
            x = tl.load(
                x_ptr + rows[:, None] * stride_xt + c * stride_xc + offs_d[None, :],
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            xf = x.to(tl.float32)
            sq += xf * xf
            w = tl.load(
                w_ptr + outs[:, None] * (HC * D) + c * D + offs_d[None, :],
                mask=out_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            w_hi = w.to(x.dtype)
            w_lo = (w - w_hi.to(tl.float32)).to(x.dtype)
            acc += tl.dot(x, tl.trans(w_hi)) + tl.dot(x, tl.trans(w_lo))

    sumsq = tl.sum(sq, axis=1)
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    ws_base = ws_ptr + rows * stride_ws_t + split * stride_ws_s
    tl.store(
        ws_base[:, None] + outs[None, :],
        acc,
        mask=row_mask[:, None] & out_mask[None, :],
    )
    tl.store(ws_base + N_OUT, sumsq, mask=row_mask)


@triton.jit
def _ihc_apply_kernel(
    x_ptr,
    ws_ptr,
    scale_ptr,
    base_ptr,
    y_ptr,
    post_ptr,
    T,
    stride_xt,
    stride_xc,
    stride_ws_t,
    stride_ws_s,
    stride_yt,
    stride_pt,
    D_PER_SPLIT,
    SPLIT,
    norm_eps,
    hc_eps,
    magnitude,
    D: tl.constexpr,
    HC: tl.constexpr,
    N_OUT: tl.constexpr,
    HAS_POST: tl.constexpr,
    SPLIT_PAD: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_D: tl.constexpr,
    launch_pdl: tl.constexpr,
):
    """Finalize gates from the workspace partials and reduce the channels for
    this program's slice of the hidden dim."""
    pid_t = tl.program_id(0)
    split = tl.program_id(1)
    rows = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    row_mask = rows < T
    outs = tl.arange(0, N_OUT)

    if launch_pdl:
        tl.extra.cuda.gdc_wait()

    # Reduce the split partials with one vectorized load per row block.
    s_offs = tl.arange(0, SPLIT_PAD)
    s_mask = s_offs < SPLIT
    ws_rows = ws_ptr + rows[:, None] * stride_ws_t + s_offs[None, :] * stride_ws_s
    rs_mask = row_mask[:, None] & s_mask[None, :]
    acc = tl.sum(
        tl.load(
            ws_rows[:, :, None] + outs[None, None, :],
            mask=rs_mask[:, :, None],
            other=0.0,
        ),
        axis=1,
    )
    sumsq = tl.sum(tl.load(ws_rows + N_OUT, mask=rs_mask, other=0.0), axis=1)

    rstd = tl.rsqrt(sumsq / (HC * D) + norm_eps)
    mixes = acc * rstd[:, None]
    base = tl.load(base_ptr + outs)
    if HAS_POST:
        s0 = tl.load(scale_ptr)
        s1 = tl.load(scale_ptr + 1)
        is_pre = outs < HC
        scale = tl.where(is_pre, s0, s1)
        mag = tl.where(is_pre, 1.0, magnitude)
    else:
        scale = tl.load(scale_ptr) + tl.zeros([N_OUT], dtype=tl.float32)
        mag = 1.0 + tl.zeros([N_OUT], dtype=tl.float32)
    gates = mag[None, :] * tl.sigmoid(mixes * scale[None, :] + base[None, :]) + hc_eps
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()

    if HAS_POST:  # noqa: SIM102 - constexpr branch, keep it separate from the runtime one
        if split == 0:
            post_mask = row_mask[:, None] & (outs[None, :] >= HC)
            tl.store(
                post_ptr + rows[:, None] * stride_pt + (outs[None, :] - HC),
                gates,
                mask=post_mask,
            )

    d_start = split * D_PER_SPLIT
    d_end = tl.minimum(d_start + D_PER_SPLIT, D)
    for d0 in range(d_start, d_end, BLOCK_D):
        offs_d = d0 + tl.arange(0, BLOCK_D)
        d_mask = offs_d < d_end
        y = tl.zeros([BLOCK_T, BLOCK_D], dtype=tl.float32)
        for c in tl.static_range(HC):
            g = tl.sum(tl.where(outs[None, :] == c, gates, 0.0), axis=1)
            x = tl.load(
                x_ptr + rows[:, None] * stride_xt + c * stride_xc + offs_d[None, :],
                mask=row_mask[:, None] & d_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            y += g[:, None] * x
        tl.store(
            y_ptr + rows[:, None] * stride_yt + offs_d[None, :],
            y.to(y_ptr.dtype.element_ty),
            mask=row_mask[:, None] & d_mask[None, :],
        )


def _pick_large_launch(T: int, D: int, device_index: int) -> tuple[int, int, int, int]:
    """Return (BLOCK_T, BLOCK_D, SPLIT, D_PER_SPLIT) for the tensor-core path."""
    cfg = _LARGE_CFG
    if T < 128:
        BLOCK_T = cfg["block_t_small"]
    elif T < 1024:
        BLOCK_T = cfg["block_t_mid"]
    else:
        BLOCK_T = cfg["block_t_large"]
    BLOCK_D = cfg["block_d"]
    n_row_blocks = triton.cdiv(T, BLOCK_T)
    want = max(1, (cfg["programs_per_sm"] * _sm_count(device_index)) // n_row_blocks)
    max_split = min(triton.cdiv(D, BLOCK_D), cfg["max_split"])
    SPLIT = max(1, min(want, max_split))
    D_PER_SPLIT = triton.cdiv(triton.cdiv(D, SPLIT), BLOCK_D) * BLOCK_D
    SPLIT = triton.cdiv(D, D_PER_SPLIT)
    return BLOCK_T, BLOCK_D, SPLIT, D_PER_SPLIT


def _ihc_reduce_large(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    has_post: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    T, HC, D = x.shape
    n_out = weight.shape[0]
    device_index = x.device.index or 0
    BLOCK_T, BLOCK_D, SPLIT, D_PER_SPLIT = _pick_large_launch(T, D, device_index)
    ws = torch.empty((T, SPLIT, n_out + 1), dtype=torch.float32, device=x.device)
    y = torch.empty((T, D), dtype=x.dtype, device=x.device)
    post = torch.empty((T, HC if has_post else 0), dtype=torch.float32, device=x.device)

    grid = (triton.cdiv(T, BLOCK_T), SPLIT)
    pdl = current_platform.is_arch_support_pdl()
    _ihc_stats_kernel[grid](
        x,
        weight,
        ws,
        T,
        x.stride(0),
        x.stride(1),
        ws.stride(0),
        ws.stride(1),
        D_PER_SPLIT,
        D=D,
        HC=HC,
        N_OUT=n_out,
        N_PAD=max(16, n_out),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        launch_pdl=pdl,
        num_warps=_LARGE_CFG["warps"],
        num_stages=_LARGE_CFG["stages"],
    )
    _ihc_apply_kernel[grid](
        x,
        ws,
        scale,
        base,
        y,
        post,
        T,
        x.stride(0),
        x.stride(1),
        ws.stride(0),
        ws.stride(1),
        y.stride(0),
        post.stride(0),
        D_PER_SPLIT,
        SPLIT,
        norm_eps,
        hc_eps,
        magnitude,
        D=D,
        HC=HC,
        N_OUT=n_out,
        HAS_POST=has_post,
        SPLIT_PAD=triton.next_power_of_2(SPLIT),
        BLOCK_T=BLOCK_T,
        BLOCK_D=BLOCK_D,
        launch_pdl=pdl,
        num_warps=_LARGE_CFG["warps"],
    )
    return y, post


def _ihc_reduce(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
    has_post: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shared launch for pre (with post gates) and head (without)."""
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
    n_out = 2 * hc_mult if has_post else hc_mult
    assert weight.shape == (n_out, k_total)
    assert scale.shape == (2 if has_post else 1,)
    assert base.shape == (n_out,)

    post_rows = hc_mult if has_post else 0
    if num_tokens == 0:
        return (
            torch.empty((0, hidden_size), dtype=x.dtype, device=x.device),
            torch.empty((0, post_rows), dtype=torch.float32, device=x.device),
        )
    if num_tokens >= _LARGE_T_MIN and (hc_mult & (hc_mult - 1)) == 0:
        return _ihc_reduce_large(
            x, weight, scale, base, magnitude, hc_eps, norm_eps, has_post
        )

    output = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    post = torch.empty((num_tokens, post_rows), dtype=torch.float32, device=x.device)
    hc_pow2 = triton.next_power_of_2(hc_mult)
    num_splits = triton.cdiv(k_total, _BLOCK_K)
    partial_stride = 1 + (2 if has_post else 1) * hc_pow2
    partial = torch.empty(
        (num_tokens, num_splits, partial_stride),
        dtype=torch.float32,
        device=x.device,
    )

    _ihc_pre_stage1[(num_tokens, num_splits)](
        x,
        weight,
        partial,
        K_TOTAL=k_total,
        HC_MULT=hc_mult,
        HC_POW2=hc_pow2,
        NUM_SPLITS=num_splits,
        BLOCK_K=_BLOCK_K,
        PARTIAL_STRIDE=partial_stride,
        HAS_POST=has_post,
        num_warps=8,
        enable_fp_fusion=False,
    )
    _ihc_pre_stage2[(num_tokens, triton.cdiv(hidden_size, _BLOCK_D))](
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
        NUM_SPLITS=num_splits,
        PARTIAL_STRIDE=partial_stride,
        BLOCK_D=_BLOCK_D,
        MAGNITUDE=magnitude,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        HAS_POST=has_post,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output, post


def _triton_ihc_pre(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    return _ihc_reduce(x, weight, scale, base, magnitude, hc_eps, norm_eps, True)


def _triton_ihc_pre_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    magnitude: float,
    hc_eps: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_tokens, hc_mult, hidden_size = x.shape
    return (
        x.new_empty((num_tokens, hidden_size)),
        x.new_empty((num_tokens, hc_mult), dtype=torch.float32),
    )


def _triton_ihc_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    return _ihc_reduce(x, weight, scale, base, 1.0, hc_eps, norm_eps, False)[0]


def _triton_ihc_head_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    num_tokens, _, hidden_size = x.shape
    return x.new_empty((num_tokens, hidden_size))


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
    return torch.ops.vllm.hy_v4_ihc_pre(
        x, weight, scale, base, magnitude, hc_eps, norm_eps
    )


def triton_ihc_head(
    x: torch.Tensor,
    weight: torch.Tensor,
    scale: torch.Tensor,
    base: torch.Tensor,
    hc_eps: float,
    norm_eps: float,
) -> torch.Tensor:
    """Reduce iHC channels for the head layer (no post gates)."""
    return torch.ops.vllm.hy_v4_ihc_head(x, weight, scale, base, hc_eps, norm_eps)


def _triton_ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
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
    n_tiles = triton.cdiv(hidden_size, _BLOCK_D)
    _ihc_post_kernel[(num_tokens * hc_mult * n_tiles,)](
        x,
        residual,
        post,
        output,
        HIDDEN_SIZE=hidden_size,
        HC_MULT=hc_mult,
        BLOCK_D=_BLOCK_D,
        launch_pdl=current_platform.is_arch_support_pdl(),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output


def _triton_ihc_post_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    return residual.new_empty(residual.shape)


def triton_ihc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
) -> torch.Tensor:
    """Scatter a sub-block output back over the iHC residual channels."""
    return torch.ops.vllm.hy_v4_ihc_post(x, residual, post)


def _spec_class(v: int) -> int:
    """Triton's default integer specialization class: 1, multiple of 16, other."""
    return 1 if v == 1 else (16 if v % 16 == 0 else 0)


def _launch_key(T: int, D: int, HC: int, device_index: int) -> tuple:
    if T < _LARGE_T_MIN:
        return ("small",)
    BLOCK_T, _, SPLIT, D_PER_SPLIT = _pick_large_launch(T, D, device_index)
    return (
        "large",
        BLOCK_T,
        _spec_class(T),
        _spec_class(SPLIT),
        triton.next_power_of_2(SPLIT),
        _spec_class(D_PER_SPLIT),
    )


def warmup_token_sizes(
    D: int, HC: int, max_tokens: int, device_index: int
) -> list[int]:
    """One token count per distinct Triton compile key of the three ops.

    Calling pre / head / post for each returned size JIT-compiles every kernel
    variant reachable for ``T <= max_tokens`` (see hy_v4_ihc_warmup).
    """
    seen: set[tuple] = set()
    sizes: list[int] = []
    for T in range(1, max_tokens + 1):
        key = _launch_key(T, D, HC, device_index)
        if key not in seen:
            seen.add(key)
            sizes.append(T)
    return sizes


direct_register_custom_op(
    op_name="hy_v4_ihc_pre",
    op_func=_triton_ihc_pre,
    fake_impl=_triton_ihc_pre_fake,
)
direct_register_custom_op(
    op_name="hy_v4_ihc_head",
    op_func=_triton_ihc_head,
    fake_impl=_triton_ihc_head_fake,
)
direct_register_custom_op(
    op_name="hy_v4_ihc_post",
    op_func=_triton_ihc_post,
    fake_impl=_triton_ihc_post_fake,
)
