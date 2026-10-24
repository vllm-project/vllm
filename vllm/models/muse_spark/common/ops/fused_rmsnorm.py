# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton RMSNorm for MuseSpark1x, optionally fused with the gated residual add."""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _fast2sum(a, b):
    """Fast2Sum: ``a + b == s + e`` exactly; requires ``|a| >= |b|``."""
    s = a + b
    z = s - a
    return s, b - z


@triton.jit
def _effective_weight(w_ptr, d, dm, gain_center):
    """``weight + gain_center`` in fp32; the norm rounds only at its store."""
    return tl.load(w_ptr + d, mask=dm, other=0.0).to(tl.float32) + gain_center


@triton.jit
def _blend_and_store(
    y_ptr,
    s_ptr,
    alpha_ptr,
    beta_ptr,
    idx,
    d,
    mask,
    dm,
    y,
    alpha_scalar,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
):
    """Store ``alpha * stream + beta * y``; ``y`` stays fp32 (one rounding fewer)."""
    stream = tl.load(s_ptr + idx, mask=mask, other=0.0).to(tl.float32)

    if PER_CHANNEL:
        beta = tl.load(beta_ptr + d, mask=dm, other=0.0)[None, :]
    else:
        beta = tl.load(beta_ptr)

    if GATE_STREAM:
        if PER_CHANNEL:
            alpha = tl.load(alpha_ptr + d, mask=dm, other=0.0)[None, :]
        else:
            alpha = tl.load(alpha_ptr)
        out = alpha * stream + beta * y
    else:
        out = alpha_scalar * stream + beta * y

    tl.store(y_ptr + idx, out.to(y_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _rmsnorm_fwd_kernel(
    x_ptr,
    y_ptr,
    w_ptr,
    s_ptr,
    alpha_ptr,
    beta_ptr,
    T,
    D: tl.constexpr,
    eps,
    gain_center,
    alpha_scalar,
    HAS_WEIGHT: tl.constexpr,
    INPUT_SCALE: tl.constexpr,
    KAHAN: tl.constexpr,
    SINGLE_TILE: tl.constexpr,
    RESIDUAL: tl.constexpr,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    bT: tl.constexpr,
    bD: tl.constexpr,
):
    """RMSNorm over the last dim of ``[T, D]``; fp32 math, the store rounds once.

    ``KAHAN`` uses compensated fp32 instead of fp64 for the cross-tile sum
    (Blackwell runs fp64 at 1/64 rate).
    """
    t_idx = tl.program_id(0)
    t = t_idx * bT + tl.arange(0, bT)
    tm = t < T

    if SINGLE_TILE:
        d = tl.arange(0, bD)
        dm = d < D
        idx = t[:, None] * D + d[None, :]
        mask = tm[:, None] & dm[None, :]

        x = tl.load(x_ptr + idx, mask=mask, other=0.0).to(tl.float32)
        if HAS_WEIGHT:
            w = _effective_weight(w_ptr, d, dm, gain_center)

        if INPUT_SCALE and HAS_WEIGHT:
            x = x * w[None, :]

        sum_sq = tl.sum(x * x, axis=1)
        y = x * tl.rsqrt(sum_sq * (1.0 / D) + eps)[:, None]

        if RESIDUAL:
            if HAS_WEIGHT and not INPUT_SCALE:
                y = y * w[None, :]
            _blend_and_store(
                y_ptr,
                s_ptr,
                alpha_ptr,
                beta_ptr,
                idx,
                d,
                mask,
                dm,
                y,
                alpha_scalar,
                GATE_STREAM=GATE_STREAM,
                PER_CHANNEL=PER_CHANNEL,
            )
        else:
            if HAS_WEIGHT and not INPUT_SCALE:
                y = y * w[None, :]
            tl.store(y_ptr + idx, y.to(y_ptr.dtype.element_ty), mask=mask)
    else:
        if KAHAN:
            sum_sq = tl.zeros([bT], dtype=tl.float32)
            compensation = tl.zeros([bT], dtype=tl.float32)
        else:
            sum_sq = tl.zeros([bT], dtype=tl.float64)

        for do in tl.range(0, D, bD):
            d = do + tl.arange(0, bD)
            dm = d < D
            x = tl.load(
                x_ptr + t[:, None] * D + d[None, :],
                mask=tm[:, None] & dm[None, :],
                other=0.0,
                eviction_policy="evict_last",
            ).to(tl.float32)
            if INPUT_SCALE and HAS_WEIGHT:
                w = _effective_weight(w_ptr, d, dm, gain_center)
                x = x * w[None, :]
            tile_sum = tl.sum(x * x, axis=1)
            if KAHAN:
                sum_sq, compensation = _fast2sum(sum_sq, tile_sum + compensation)
            else:
                sum_sq += tile_sum.to(tl.float64)

        # Cast here so the fp64 accumulator does not promote the row multiply
        # to fp64 (1/64 rate on Blackwell, and unlike the fp32 reference).
        rstd = tl.rsqrt(sum_sq * (1.0 / D) + eps).to(tl.float32)

        for do in tl.range(0, D, bD):
            d = do + tl.arange(0, bD)
            dm = d < D
            idx = t[:, None] * D + d[None, :]
            mask = tm[:, None] & dm[None, :]
            x = tl.load(
                x_ptr + idx, mask=mask, other=0.0, eviction_policy="evict_first"
            ).to(tl.float32)
            if HAS_WEIGHT:
                w = _effective_weight(w_ptr, d, dm, gain_center)
            if INPUT_SCALE and HAS_WEIGHT:
                x = x * w[None, :]
            if RESIDUAL:
                y = x * rstd[:, None]
                if HAS_WEIGHT and not INPUT_SCALE:
                    y = y * w[None, :]
                _blend_and_store(
                    y_ptr,
                    s_ptr,
                    alpha_ptr,
                    beta_ptr,
                    idx,
                    d,
                    mask,
                    dm,
                    y,
                    alpha_scalar,
                    GATE_STREAM=GATE_STREAM,
                    PER_CHANNEL=PER_CHANNEL,
                )
            else:
                y = x * rstd[:, None]
                if HAS_WEIGHT and not INPUT_SCALE:
                    y = y * w[None, :]
                tl.store(y_ptr + idx, y.to(y_ptr.dtype.element_ty), mask=mask)


def fused_rmsnorm_fwd(
    x: torch.Tensor,
    w: torch.Tensor | None,
    eps: float,
    *,
    gain_center: float = 0.0,
    input_scale: bool = False,
    use_kahan_summation: bool = True,
    out: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_stream: torch.Tensor | None = None,
    alpha: torch.Tensor | float | None = None,
    beta: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm with a gain-centred weight, applied before or after the norm.

    With ``residual_stream`` it returns ``alpha * residual_stream + beta * norm(x)``.
    """
    input_shape = x.shape
    x = x.reshape(-1, x.shape[-1])
    assert x.is_contiguous(), "fused RMSNorm requires a contiguous input"

    T, D = x.shape
    if w is None:
        assert gain_center == 0.0, "gain_center requires a weight"
        assert not input_scale, "input_scale requires a weight"
    else:
        assert w.shape == (D,), f"weight must be [{D}], got {tuple(w.shape)}"
        assert w.stride(-1) == 1, "fused RMSNorm requires a contiguous weight"

    residual = residual_stream is not None
    gate_stream = False
    per_channel = False
    alpha_scalar = 1.0
    if residual_stream is not None:
        assert beta is not None, "residual_stream requires beta"
        assert alpha is not None, "residual_stream requires alpha"
        assert residual_stream.shape == input_shape, (
            f"residual_stream must have x's shape {tuple(input_shape)}, got "
            f"{tuple(residual_stream.shape)}"
        )
        residual_stream = residual_stream.reshape(-1, D)
        assert residual_stream.is_contiguous(), (
            "fused RMSNorm requires a contiguous residual_stream"
        )
        assert beta.numel() in (1, D), (
            f"beta must be [1] or [{D}], got {tuple(beta.shape)}"
        )
        assert beta.is_contiguous(), "fused RMSNorm requires a contiguous beta"
        if isinstance(alpha, torch.Tensor):
            gate_stream = True
            assert alpha.shape == beta.shape, (
                f"alpha {tuple(alpha.shape)} and beta {tuple(beta.shape)} must match"
            )
            assert alpha.is_contiguous(), "fused RMSNorm requires a contiguous alpha"
        else:
            alpha_scalar = float(alpha)
        per_channel = beta.numel() > 1

    if out is None:
        y = torch.empty_like(residual_stream if residual else x, dtype=out_dtype)
    else:
        assert out_dtype is None or out.dtype == out_dtype, (
            f"out is {out.dtype}, out_dtype says {out_dtype}"
        )
        assert out.shape == input_shape, "out must have x's shape"
        y = out.reshape(-1, D)
        assert y.is_contiguous(), "out must be contiguous"

    tile_elems = 8192
    bD = min(triton.next_power_of_2(D), tile_elems)
    # Tile the residual epilogue to limit register spilling at larger batches.
    if residual:
        if T >= 2048:
            bD = min(bD, 2048)
        elif T >= 128:
            bD = min(bD, 4096)
    bT = max(1, triton.next_power_of_2(tile_elems // bD))
    # A full-width residual tile is register-bound; extra warps hide spills.
    num_warps = 16 if residual and bD >= D else 8

    _rmsnorm_fwd_kernel[(triton.cdiv(T, bT),)](
        x,
        y,
        w,
        # Unused pointers must be valid; constexprs keep them undereferenced.
        residual_stream if residual else x,
        alpha if gate_stream else (beta if residual else x),
        beta if residual else x,
        T,
        D=D,
        eps=eps,
        gain_center=gain_center,
        alpha_scalar=alpha_scalar,
        HAS_WEIGHT=w is not None,
        INPUT_SCALE=input_scale,
        KAHAN=use_kahan_summation,
        SINGLE_TILE=bD >= D,
        RESIDUAL=residual,
        GATE_STREAM=gate_stream,
        PER_CHANNEL=per_channel,
        bT=bT,
        bD=bD,
        num_warps=num_warps,
    )
    return y.view(input_shape)


# The whole row must be live for the second norm, so this path is single-tile.
_BOUNDARY_TILE_ELEMS = 8192


@triton.jit(do_not_specialize_on_alignment=["T"])
def _norm_residual_norm_fwd_kernel(
    branch_ptr,
    stream_ptr,
    new_stream_ptr,
    out_ptr,
    post_w_ptr,
    pre_w_ptr,
    alpha_ptr,
    beta_ptr,
    T,
    D: tl.constexpr,
    post_eps,
    pre_eps,
    post_gain_center,
    pre_gain_center,
    alpha_scalar,
    POST_HAS_WEIGHT: tl.constexpr,
    POST_INPUT_SCALE: tl.constexpr,
    PRE_HAS_WEIGHT: tl.constexpr,
    PRE_INPUT_SCALE: tl.constexpr,
    GATE_STREAM: tl.constexpr,
    PER_CHANNEL: tl.constexpr,
    bT: tl.constexpr,
    bD: tl.constexpr,
):
    """One residual boundary: post-norm, gated add, and the next pre-norm."""
    t = tl.program_id(0) * bT + tl.arange(0, bT)
    tm = t < T
    d = tl.arange(0, bD)
    dm = d < D
    idx = t[:, None] * D + d[None, :]
    mask = tm[:, None] & dm[None, :]

    # All loads before the first reduction: at decode this is latency-bound.
    x = tl.load(branch_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    stream = tl.load(stream_ptr + idx, mask=mask, other=0.0).to(tl.float32)
    if PER_CHANNEL:
        beta = tl.load(beta_ptr + d, mask=dm, other=0.0)[None, :]
    else:
        beta = tl.load(beta_ptr)
    if GATE_STREAM:
        if PER_CHANNEL:
            alpha = tl.load(alpha_ptr + d, mask=dm, other=0.0)[None, :]
        else:
            alpha = tl.load(alpha_ptr)
    if POST_HAS_WEIGHT:
        post_w = _effective_weight(post_w_ptr, d, dm, post_gain_center)
    if PRE_HAS_WEIGHT:
        pre_w = _effective_weight(pre_w_ptr, d, dm, pre_gain_center)

    if POST_INPUT_SCALE and POST_HAS_WEIGHT:
        x = x * post_w[None, :]
    sum_sq = tl.sum(x * x, axis=1)
    y = x * tl.rsqrt(sum_sq * (1.0 / D) + post_eps)[:, None]
    if POST_HAS_WEIGHT and not POST_INPUT_SCALE:
        y = y * post_w[None, :]

    if GATE_STREAM:
        new_stream = alpha * stream + beta * y
    else:
        new_stream = alpha_scalar * stream + beta * y
    tl.store(
        new_stream_ptr + idx,
        new_stream.to(new_stream_ptr.dtype.element_ty),
        mask=mask,
    )

    z = new_stream
    if PRE_INPUT_SCALE and PRE_HAS_WEIGHT:
        z = z * pre_w[None, :]
    pre_sum_sq = tl.sum(z * z, axis=1)
    z = z * tl.rsqrt(pre_sum_sq * (1.0 / D) + pre_eps)[:, None]
    if PRE_HAS_WEIGHT and not PRE_INPUT_SCALE:
        z = z * pre_w[None, :]

    tl.store(out_ptr + idx, z.to(out_ptr.dtype.element_ty), mask=mask)


def supports_norm_residual_norm(
    branch: torch.Tensor,
    stream: torch.Tensor,
) -> bool:
    """Whether :func:`fused_norm_residual_norm_fwd` can serve this boundary."""
    if not (branch.is_cuda and stream.is_cuda):
        return False
    if branch.shape != stream.shape or branch.ndim < 2:
        return False
    if stream.dtype is not torch.float32:
        return False
    if not (branch.is_contiguous() and stream.is_contiguous()):
        return False
    return triton.next_power_of_2(branch.shape[-1]) <= _BOUNDARY_TILE_ELEMS


def fused_norm_residual_norm_fwd(
    branch: torch.Tensor,
    stream: torch.Tensor,
    post_weight: torch.Tensor | None,
    pre_weight: torch.Tensor | None,
    post_eps: float,
    pre_eps: float,
    *,
    alpha: torch.Tensor | float,
    beta: torch.Tensor,
    post_gain_center: float = 0.0,
    pre_gain_center: float = 0.0,
    post_input_scale: bool = False,
    pre_input_scale: bool = False,
    out_dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """A whole residual boundary in one kernel.

    Computes, in fp32::

        new_stream = alpha * stream + beta * post_norm(branch)
        out = pre_norm(new_stream)

    Only the sum-of-squares order differs from two ``fused_rmsnorm_fwd`` calls;
    vs fp64 at D=8192, 9e-6 of the bf16 outputs differ by one ulp.
    """
    input_shape = branch.shape
    assert supports_norm_residual_norm(branch, stream), (
        "fused norm/residual/norm boundary: unsupported branch "
        f"{tuple(branch.shape)}:{branch.dtype} over stream "
        f"{tuple(stream.shape)}:{stream.dtype}; check "
        "supports_norm_residual_norm() first"
    )

    branch = branch.reshape(-1, input_shape[-1])
    stream = stream.reshape(-1, input_shape[-1])
    T, D = branch.shape

    for weight, gain_center, input_scale, name in (
        (post_weight, post_gain_center, post_input_scale, "post"),
        (pre_weight, pre_gain_center, pre_input_scale, "pre"),
    ):
        if weight is None:
            assert gain_center == 0.0, f"{name}_gain_center requires a weight"
            assert not input_scale, f"{name}_input_scale requires a weight"
        else:
            assert weight.shape == (D,), (
                f"{name}_weight must be [{D}], got {tuple(weight.shape)}"
            )
            assert weight.stride(-1) == 1, f"{name}_weight must be contiguous"

    assert beta.numel() in (1, D), f"beta must be [1] or [{D}], got {tuple(beta.shape)}"
    assert beta.is_contiguous(), "beta must be contiguous"
    per_channel = beta.numel() > 1
    gate_stream = isinstance(alpha, torch.Tensor)
    alpha_scalar = 1.0
    if gate_stream:
        assert isinstance(alpha, torch.Tensor)
        assert alpha.shape == beta.shape, (
            f"alpha {tuple(alpha.shape)} and beta {tuple(beta.shape)} must match"
        )
        assert alpha.is_contiguous(), "alpha must be contiguous"
    else:
        alpha_scalar = float(alpha)

    out = torch.empty_like(branch, dtype=out_dtype)
    new_stream = torch.empty_like(stream)

    bD = triton.next_power_of_2(D)
    bT = max(1, _BOUNDARY_TILE_ELEMS // bD)

    num_warps = 32 if bD >= 8192 else (16 if bD >= 4096 else 8)
    if current_platform.is_rocm():
        # 64-thread wavefronts: 32 warps would exceed the 1024-thread limit.
        num_warps = min(num_warps, 16)
    _norm_residual_norm_fwd_kernel[(triton.cdiv(T, bT),)](
        branch,
        stream,
        new_stream,
        out,
        # Unused pointers must be valid; constexprs keep them undereferenced.
        post_weight if post_weight is not None else branch,
        pre_weight if pre_weight is not None else branch,
        alpha if gate_stream else beta,
        beta,
        T,
        D=D,
        post_eps=post_eps,
        pre_eps=pre_eps,
        post_gain_center=post_gain_center,
        pre_gain_center=pre_gain_center,
        alpha_scalar=alpha_scalar,
        POST_HAS_WEIGHT=post_weight is not None,
        POST_INPUT_SCALE=post_input_scale,
        PRE_HAS_WEIGHT=pre_weight is not None,
        PRE_INPUT_SCALE=pre_input_scale,
        GATE_STREAM=gate_stream,
        PER_CHANNEL=per_channel,
        bT=bT,
        bD=bD,
        num_warps=num_warps,
    )
    return out.view(input_shape), new_stream.view(input_shape)
