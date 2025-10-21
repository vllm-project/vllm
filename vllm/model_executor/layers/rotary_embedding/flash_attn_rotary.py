# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2025, Tri Dao.
# As of 2025-04-23, we require triton >= 3.0


import torch
import triton
import triton.language as tl


@triton.jit
def _rotary_1c(
    X,
    OUT,
    stride_out_nheads,
    stride_out_seqlen,
    stride_out_headdim,
    stride_seqlen,
    stride_nheads,
    stride_headdim,
    rh,
    rm,
    rk_half,
    sin,
    cos,
    nheads,
    seqlen,
    ROTARY_DIM_HALF: tl.constexpr,
    ROTARY_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    if not INTERLEAVED:
        # Load the 1st and 2nd halves of X, do calculation, then
        # store to 1st and 2nd halves of OUT
        rk_half = tl.max_contiguous(tl.multiple_of(rk_half, 4), 4)
        X = X + (
            rh[:, None, None] * stride_nheads
            + rm[None, :, None] * stride_seqlen
            + rk_half[None, None, :] * stride_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk_half[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk_half[None, None, :] < ROTARY_DIM_HALF)
        )
        x0 = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x1 = tl.load(
            X + ROTARY_DIM_HALF * stride_headdim,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        tl.store(OUT, o0, mask=mask)
        tl.store(OUT + ROTARY_DIM_HALF * stride_out_headdim, o1, mask=mask)
    else:
        rk = tl.arange(0, BLOCK_K)
        X = X + (
            rh[:, None, None] * stride_nheads
            + rm[None, :, None] * stride_seqlen
            + rk[None, None, :] * stride_headdim
        )
        OUT = OUT + (
            rh[:, None, None] * stride_out_nheads
            + rm[None, :, None] * stride_out_seqlen
            + rk[None, None, :] * stride_out_headdim
        )
        mask = (
            (rh[:, None, None] < nheads)
            & (rm[None, :, None] < seqlen)
            & (rk[None, None, :] < ROTARY_DIM)
        )
        x = tl.load(X, mask=mask, other=0.0).to(tl.float32)
        x0, x1 = tl.split(tl.reshape(x, [BLOCK_H, BLOCK_M, BLOCK_K // 2, 2]))
        o0 = x0 * cos - x1 * sin
        o1 = x0 * sin + x1 * cos
        o = tl.reshape(tl.join(o0, o1), [BLOCK_H, BLOCK_M, BLOCK_K])
        tl.store(OUT, o, mask=mask)


@triton.jit
def rotary_kernel(
    OUT_X,  # Pointers to matrices
    OUT_Y,  # Pointers to matrices
    IN_X,
    IN_Y,
    FREQS,
    CU_SEQLENS,
    SEQLEN_OFFSETS,  # this could be int or a pointer
    # Matrix dimensions
    seqlen,
    nheads,
    seqlen_ro,
    # strides
    stride_out_x_batch,
    stride_out_x_seqlen,
    stride_out_x_nheads,
    stride_out_x_headdim,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_nheads,
    stride_x_headdim,
    stride_out_y_batch,
    stride_out_y_seqlen,
    stride_out_y_nheads,
    stride_out_y_headdim,
    stride_y_batch,
    stride_y_seqlen,
    stride_y_nheads,
    stride_y_headdim,
    # Meta-parameters
    # We want ROTARY_DIM to be constexpr, otherwise
    # the triton compiler doesn't know that the mask
    # is constant every 8 elements, and it will
    # generate LDG.16 instead of LDG.128
    ROTARY_DIM: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    INTERLEAVED: tl.constexpr,
    CONJUGATE: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    BLOCK_K: tl.constexpr = triton.next_power_of_2(ROTARY_DIM)
    ROTARY_DIM_HALF: tl.constexpr = ROTARY_DIM // 2
    pid_head = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    pid_batch = tl.program_id(axis=2)

    if not IS_VARLEN:
        IN_X = IN_X + pid_batch * stride_x_batch
        IN_Y = IN_Y + pid_batch * stride_y_batch
        OUT_X = OUT_X + pid_batch * stride_out_x_batch
        OUT_Y = OUT_Y + pid_batch * stride_out_y_batch
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        seqlen = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        IN_X = IN_X + start_idx * stride_x_seqlen
        IN_Y = IN_Y + start_idx * stride_y_seqlen
        OUT_X = OUT_X + start_idx * stride_out_x_seqlen
        OUT_Y = OUT_Y + start_idx * stride_out_y_seqlen

    if pid_m * BLOCK_M >= seqlen:
        return

    rh = pid_head * BLOCK_H + tl.arange(0, BLOCK_H)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)

    rk_half = tl.arange(0, BLOCK_K // 2)
    FREQS = FREQS + (rm_cs[:, None] * ROTARY_DIM_HALF + rk_half[None, :])
    mask_cs = (rm_cs[:, None] < seqlen_ro) & (rk_half[None, :] < ROTARY_DIM_HALF)
    freqs = tl.load(FREQS, mask=mask_cs, other=0.0).to(tl.float32)
    cos = tl.cos(freqs)
    sin = tl.sin(freqs)
    if CONJUGATE:
        sin = -sin
    _rotary_1c(
        IN_X,
        OUT_X,
        stride_out_x_nheads,
        stride_out_x_seqlen,
        stride_out_x_headdim,
        stride_x_seqlen,
        stride_x_nheads,
        stride_x_headdim,
        rh,
        rm,
        rk_half,
        sin,
        cos,
        nheads,
        seqlen,
        ROTARY_DIM_HALF,
        ROTARY_DIM,
        BLOCK_H,
        BLOCK_M,
        BLOCK_K,
        INTERLEAVED,
    )
    _rotary_1c(
        IN_Y,
        OUT_Y,
        stride_out_y_nheads,
        stride_out_y_seqlen,
        stride_out_y_headdim,
        stride_y_seqlen,
        stride_y_nheads,
        stride_y_headdim,
        rh,
        rm,
        rk_half,
        sin,
        cos,
        nheads,
        seqlen,
        ROTARY_DIM_HALF,
        ROTARY_DIM,
        BLOCK_H,
        BLOCK_M,
        BLOCK_K,
        INTERLEAVED,
    )


def apply_rotary_2c(
    x: torch.Tensor,
    y: torch.Tensor,
    freqs: torch.Tensor,
    seqlen_offsets: int | torch.Tensor = 0,
    cu_seqlens: torch.Tensor | None = None,
    max_seqlen: int | None = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> torch.Tensor:
    """
    Arguments:
        x: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        y: (batch, seqlen, nheads, headdim) if cu_seqlens is None
            else (total_seqlen, nheads, headdim).
        freqs: (seqlen_ro, rotary_dim / 2)
        seqlen_offsets: integer or integer tensor of size (batch,)
        cu_seqlens: (batch + 1,) or None
        max_seqlen: int
    Returns:
        out_x: (batch, seqlen, nheads, headdim)
        out_y: (batch, seqlen, nheads, headdim)
    """
    is_varlen = cu_seqlens is not None
    assert x.shape == y.shape
    if cu_seqlens is None:
        batch, seqlen, nheads, headdim = x.shape
    else:
        assert max_seqlen is not None, (
            "If cu_seqlens is passed in, then max_seqlen must be passed"
        )
        total_seqlen, nheads, headdim = x.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen
    seqlen_ro, rotary_dim = freqs.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    freqs = freqs.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    output_x = torch.empty_like(x) if not inplace else x
    output_y = torch.empty_like(y) if not inplace else y
    if rotary_dim < headdim and not inplace:
        output_x[..., rotary_dim:].copy_(x[..., rotary_dim:])
        output_y[..., rotary_dim:].copy_(y[..., rotary_dim:])

    grid = lambda META: (
        triton.cdiv(nheads, META["BLOCK_H"]),
        triton.cdiv(seqlen, META["BLOCK_M"]),
        batch,
    )  # noqa
    BLOCK_M = 16 if rotary_dim <= 128 else 8

    # Need this, otherwise Triton tries to launch from cuda:0 and we get
    # ValueError: Pointer argument (at 0) cannot be accessed from Triton
    # (cpu tensor?)
    with torch.cuda.device(x.device.index):
        torch.library.wrap_triton(rotary_kernel)[grid](
            output_x,  # data ptrs
            output_y,  # data ptrs
            x,
            y,
            freqs,
            cu_seqlens,
            seqlen_offsets,
            seqlen,  # shapes
            nheads,
            seqlen_ro,
            output_x.stride(0)
            if not is_varlen
            else 0,  # batch_strides if not varlen else 0
            output_x.stride(-3),  # seqlen_stride or total_seqlen_stride
            output_x.stride(-2),  # nheads_stride
            output_x.stride(-1),  # headdim_stride
            x.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            x.stride(-3),  # seqlen stride or total_seqlen_stride
            x.stride(-2),  # nheads stride
            x.stride(-1),  # headdim stride
            output_y.stride(0)
            if not is_varlen
            else 0,  # batch_strides if not varlen else 0
            output_y.stride(-3),  # seqlen_stride or total_seqlen_stride
            output_y.stride(-2),  # nheads_stride
            output_y.stride(-1),  # headdim_stride
            y.stride(0) if not is_varlen else 0,  # batch_strides if not varlen else 0
            y.stride(-3),  # seqlen stride or total_seqlen_stride
            y.stride(-2),  # nheads stride
            y.stride(-1),  # headdim stride
            rotary_dim,
            isinstance(seqlen_offsets, torch.Tensor),
            is_varlen,
            interleaved,
            conjugate,
            BLOCK_M=BLOCK_M,
            BLOCK_H=2,
        )
    return output_x, output_y
