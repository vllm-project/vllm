# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_state.py

# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .mamba_ssm import softplus


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_H": 2}),
        triton.Config({"BLOCK_SIZE_H": 4}),
        triton.Config({"BLOCK_SIZE_H": 8}),
        triton.Config({"BLOCK_SIZE_H": 16}),
        triton.Config({"BLOCK_SIZE_H": 32}),
        triton.Config({"BLOCK_SIZE_H": 64}),
    ],
    key=["chunk_size", "nheads"],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr,
    A_ptr,
    dt_bias_ptr,
    dt_out_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimension
    seqlen,
    nheads: tl.constexpr,
    chunk_size: tl.constexpr,
    dt_min: tl.constexpr,
    dt_max: tl.constexpr,
    # Strides
    stride_dt_seqlen: tl.int64,
    stride_dt_head: tl.constexpr,
    stride_A_head: tl.constexpr,
    stride_dt_bias_head: tl.constexpr,
    stride_dt_out_head: tl.int64,
    stride_dt_out_chunk: tl.int64,
    stride_dt_out_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
    BLOCK_SIZE_CHUNK: tl.constexpr,
):
    # if dt is long, may cause problems, so use 64 bit
    # https://github.com/triton-lang/triton/issues/1058
    pid_c = tl.program_id(axis=0).to(tl.int64)
    pid_h = tl.program_id(axis=1)

    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)

    dt_ptr += chunk_seqlen_start * stride_dt_seqlen
    dt_out_ptr += pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (
        offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen
    )
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (
        offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize
    )
    dA_cs_ptrs = dA_cumsum_ptr + (
        offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize
    )
    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    dt = tl.load(
        dt_ptrs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit),
        other=0.0,
    ).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(
            dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0
        ).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)

    dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.where(
        (offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0
    )
    tl.store(
        dt_out_ptrs,
        dt,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(
        dA_cs_ptrs,
        dA_cs,
        mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size),
    )


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    states_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    seqlen,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    stride_states_chunk: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_c = tl.program_id(axis=1).to(tl.int64)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    b_ptr += (
        chunk_seqlen_start * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(
        tl.float32
    )
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        scale = tl.exp(dA_cs_last - dA_cs_k) * dt_k
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)

        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64},
            num_stages=3,
            num_warps=8,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=4,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 32, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=5,
            num_warps=2,
        ),
        triton.Config(
            {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32},
            num_stages=4,
            num_warps=2,
        ),
    ],
    key=["hdim", "dstate", "chunk_size"],
)
@triton.jit
def _chunk_state_varlen_kernel(
    # Pointers to matrices
    x_ptr,
    b_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    chunk_states_ptr,
    cu_seqlens_ptr,
    states_ptr,
    initstates_ptr,
    # Matrix dimensions
    hdim: tl.constexpr,
    dstate: tl.constexpr,
    chunk_size: tl.constexpr,
    nheads_ngroups_ratio: tl.constexpr,
    # Strides
    stride_x_seqlen: tl.int64,
    stride_x_head: tl.int64,
    stride_x_hdim: tl.constexpr,
    stride_b_seqlen: tl.int64,
    stride_b_head: tl.int64,
    stride_b_dstate: tl.constexpr,
    stride_dt_head: tl.int64,
    stride_dt_chunk: tl.int64,
    stride_dt_csize: tl.constexpr,
    stride_dA_cs_head: tl.int64,
    stride_dA_cs_chunk: tl.int64,
    stride_dA_cs_csize: tl.constexpr,
    stride_chunk_states_chunk: tl.int64,
    stride_chunk_states_head: tl.int64,
    stride_chunk_states_hdim: tl.int64,
    stride_chunk_states_dstate: tl.constexpr,
    stride_states_batch: tl.int64,
    stride_states_head: tl.int64,
    stride_states_hdim: tl.int64,
    stride_states_dstate: tl.constexpr,
    stride_init_states_batch: tl.int64,
    stride_init_states_head: tl.int64,
    stride_init_states_hdim: tl.int64,
    stride_init_states_dstate: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    pid_c = (end_idx - 1) // chunk_size
    b_ptr += (
        pid_c * chunk_size * stride_b_seqlen
        + (pid_h // nheads_ngroups_ratio) * stride_b_head
    )
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += (
        pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head
    )

    if HAS_INITSTATES:
        # if there are init states provided, we differentiate between states (which
        # are boundary conditions at a chunk boundary) and initstates (which are boundary
        # conditions when a new example in a cont batch starts)
        initstates_ptr += pid_h * stride_init_states_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (
        offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen
    )
    b_ptrs = b_ptr + (
        offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen
    )
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(
        dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize
    ).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = end_idx - pid_c * chunk_size
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < hdim)
            & (offs_k[None, :] < chunk_size_limit - k)
            & (offs_k[None, :] >= start_idx_cur - k),
            other=0.0,
        )
        b = tl.load(
            b_ptrs,
            mask=(offs_k[:, None] < chunk_size_limit - k)
            & (offs_n[None, :] < dstate)
            & (offs_k[:, None] >= start_idx_cur - k),
            other=0.0,
        ).to(tl.float32)
        dA_cs_k = tl.load(
            dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0
        ).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(
            tl.float32
        )
        scale = tl.where(
            (offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
            tl.exp(dA_cs_last - dA_cs_k) * dt_k,
            0.0,
        )
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # If the sequence starts after the last chunk idx, we don't need to add the contribution from the last chunk
    # If HAS_INITSTATES==True need to consider two possibilities
    # - if start_idx < pid_c * chunk_size, then we need to take the past_states_ptrs
    # - if state_idx >= pid * chunk_size, then we need to insert initstates
    if (
        (start_idx < pid_c * chunk_size)  # first chunk
        or (HAS_INITSTATES)
    ):
        dA_cs_boundary = 0.0  # default

        if not HAS_INITSTATES:
            past_states_ptrs = chunk_states_ptr + (
                offs_m[:, None] * stride_chunk_states_hdim
                + offs_n[None, :] * stride_chunk_states_dstate
            )
        else:
            # - this seems repetitive, buts its to help the compiler
            if start_idx < pid_c * chunk_size:
                past_states_ptrs = chunk_states_ptr + (
                    offs_m[:, None] * stride_chunk_states_hdim
                    + offs_n[None, :] * stride_chunk_states_dstate
                )
            else:
                past_states_ptrs = initstates_ptr + (
                    pid_b * stride_init_states_batch
                    + offs_m[:, None] * stride_init_states_hdim
                    + offs_n[None, :] * stride_init_states_dstate
                )

                # need to adjust the boundary
                if start_idx > pid_c * chunk_size:
                    dA_cs_boundary = tl.load(
                        dA_cumsum_ptr
                        + (start_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize
                    ).to(tl.float32)

        past_states = tl.load(
            past_states_ptrs,
            mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate),
            other=0.0,
        ).to(tl.float32)

        scale = tl.exp(dA_cs_last - dA_cs_boundary)
        acc += past_states * scale

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (
        offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate
    )
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _chunk_cumsum_fwd(
    dt,
    A,
    chunk_size,
    cu_chunk_seqlens,
    dt_bias=None,
    dt_softplus=False,
    dt_limit=(0.0, float("inf")),
):
    seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = cu_chunk_seqlens.shape[0] - 1
    dt_out = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    dA_cumsum = torch.empty(
        nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32
    )
    grid_chunk_cs = lambda META: (nchunks, triton.cdiv(nheads, META["BLOCK_SIZE_H"]))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt_ptr=dt,
            A_ptr=A,
            dt_bias_ptr=dt_bias,
            dt_out_ptr=dt_out,
            dA_cumsum_ptr=dA_cumsum,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            seqlen=seqlen,
            nheads=nheads,
            chunk_size=chunk_size,
            dt_min=dt_limit[0],
            dt_max=dt_limit[1],
            stride_dt_seqlen=dt.stride(0),
            stride_dt_head=dt.stride(1),
            stride_A_head=A.stride(0),
            stride_dt_bias_head=dt_bias.stride(0) if dt_bias is not None else 0,
            stride_dt_out_head=dt_out.stride(0),
            stride_dt_out_chunk=dt_out.stride(1),
            stride_dt_out_csize=dt_out.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            DT_SOFTPLUS=dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out


def _chunk_state_fwd(
    B, x, dt, dA_cumsum, cu_chunk_seqlens, states=None, states_in_fp32=True
):
    seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape

    if states is not None:
        assert states.shape == (nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty(
            (nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype
        )

    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        nchunks,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x_ptr=x,
            b_ptr=B,
            states_ptr=states,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            cu_chunk_seqlens_ptr=cu_chunk_seqlens,
            hdim=headdim,
            dstate=dstate,
            chunk_size=chunk_size,
            seqlen=seqlen,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_b_seqlen=B.stride(0),
            stride_b_head=B.stride(1),
            stride_b_dstate=B.stride(2),
            stride_states_chunk=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_dt_head=dt.stride(0),
            stride_dt_chunk=dt.stride(1),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
        )
    return states


def chunk_state_varlen(
    B, x, dt, dA_cumsum, cu_seqlens, chunk_states, initial_states=None
):
    total_seqlen, nheads, headdim = x.shape
    _, nchunks, chunk_size = dt.shape
    _, ngroups, dstate = B.shape
    batch = cu_seqlens.shape[0] - 1
    cu_seqlens = cu_seqlens.contiguous()
    assert nheads % ngroups == 0
    assert B.shape == (total_seqlen, ngroups, dstate)
    assert dt.shape == (nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert chunk_states.shape == (nchunks, nheads, headdim, dstate)

    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)

    states = torch.empty(
        batch,
        nheads,
        headdim,
        dstate,
        dtype=chunk_states.dtype,
        device=chunk_states.device,
    )

    initial_states_strides = (
        (
            initial_states.stride(0),
            initial_states.stride(1),
            initial_states.stride(2),
            initial_states.stride(3),
        )
        if initial_states is not None
        else (0, 0, 0, 0)
    )

    grid = lambda META: (
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        batch,
        nheads,
    )
    with torch.cuda.device(x.device.index):
        _chunk_state_varlen_kernel[grid](
            x_ptr=x,
            b_ptr=B,
            dt_ptr=dt,
            dA_cumsum_ptr=dA_cumsum,
            chunk_states_ptr=chunk_states,
            cu_seqlens_ptr=cu_seqlens,
            states_ptr=states,
            initstates_ptr=initial_states,
            hdim=headdim,
            dstate=dstate,
            chunk_size=chunk_size,
            nheads_ngroups_ratio=nheads // ngroups,
            stride_x_seqlen=x.stride(0),
            stride_x_head=x.stride(1),
            stride_x_hdim=x.stride(2),
            stride_b_seqlen=B.stride(0),
            stride_b_head=B.stride(1),
            stride_b_dstate=B.stride(2),
            stride_dt_head=dt.stride(0),
            stride_dt_chunk=dt.stride(1),
            stride_dt_csize=dt.stride(2),
            stride_dA_cs_head=dA_cumsum.stride(0),
            stride_dA_cs_chunk=dA_cumsum.stride(1),
            stride_dA_cs_csize=dA_cumsum.stride(2),
            stride_chunk_states_chunk=chunk_states.stride(0),
            stride_chunk_states_head=chunk_states.stride(1),
            stride_chunk_states_hdim=chunk_states.stride(2),
            stride_chunk_states_dstate=chunk_states.stride(3),
            stride_states_batch=states.stride(0),
            stride_states_head=states.stride(1),
            stride_states_hdim=states.stride(2),
            stride_states_dstate=states.stride(3),
            stride_init_states_batch=initial_states_strides[0],
            stride_init_states_head=initial_states_strides[1],
            stride_init_states_hdim=initial_states_strides[2],
            stride_init_states_dstate=initial_states_strides[3],
            HAS_INITSTATES=initial_states is not None,
        )
    return states


# TODO: Improve autotuning configuration
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 4,
                "BLOCK_SIZE_N": 4,
                "BLOCK_SIZE_H": 4,
            },
            num_stages=4,
            num_warps=8,
        ),
    ],
    key=["nheads", "dstate", "headdim"],
)
@triton.jit
def _state_cache_fwd_kernel(
    # Pointers to matrices
    states_ptr,
    cache_state_ptr,
    state_indices_tensor_ptr,
    n_blocks_to_fill_ptr,
    block_idx_first_scheduled_token_ptr,
    last_chunk_ptr,
    num_computed_tokens_ptr,
    # Matrix dimensions
    block_size_to_align: tl.constexpr,
    chunk_size: tl.constexpr,
    nheads: tl.constexpr,
    headdim: tl.constexpr,
    dstate: tl.constexpr,
    nseq: tl.constexpr,
    # Strides
    state_indices_stride: tl.constexpr,
    state_chunk_stride: tl.constexpr,
    state_nheads_stride: tl.constexpr,
    state_headdim_stride: tl.constexpr,
    state_dstate_stride: tl.constexpr,
    cache_state_cacheline_stride: tl.constexpr,
    cache_state_nheads_stride: tl.constexpr,
    cache_state_headdim_stride: tl.constexpr,
    cache_state_dstate_stride: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # single-sequence id
    idx_seq = tl.program_id(0) % nseq
    idx_block_to_fill = tl.program_id(0) // nseq

    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(2) // num_pid_n
    pid_n = tl.program_id(2) % num_pid_n
    pid_h = tl.program_id(1)

    # elements along the number of heads
    idx_nheads = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    # elements along the head dimension
    idx_headdim = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # elements along the state dimension
    idx_dstate = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # The chunk_stride is the number of chunks per mamba block
    # e.g., if mamba_block_size = 512 and chunk_size = 256,
    # then chunk_stride = 2
    chunk_stride = block_size_to_align // chunk_size

    ssm_state_input_coord = tl.load(last_chunk_ptr + idx_seq).to(tl.int64)

    # Compute the index of the cache -> target
    # In this case, it is the
    ssm_state_output_coord = tl.load(
        state_indices_tensor_ptr + idx_seq * state_indices_stride
    ).to(tl.int64)

    if n_blocks_to_fill_ptr is not None:
        # Number of blocks that need to be written.
        # If larger than the number of blocks to fill for the current sequence, return
        n_blocks_to_fill = tl.load(n_blocks_to_fill_ptr + idx_seq)

        if idx_block_to_fill > n_blocks_to_fill:
            return
        else:
            if block_idx_first_scheduled_token_ptr is None:
                idx_block_cache = tl.full((), 0, dtype=tl.int64)
            else:
                # Get the current block index for the sequence
                # Block index for the first scheduled token
                idx_block_cache = (
                    tl.load(block_idx_first_scheduled_token_ptr + idx_seq)
                    + idx_block_to_fill
                )

            # Compute the index of the cache -> target
            # Get the index from the state_indices_tensor
            ssm_state_output_coord = tl.load(
                state_indices_tensor_ptr
                + idx_seq * state_indices_stride
                + idx_block_cache
            ).to(tl.int64)

            if idx_block_to_fill < n_blocks_to_fill:
                # First chunk index for this sequence
                if idx_seq == 0:
                    first_chunk = tl.full((), 0, dtype=tl.int64)
                else:
                    first_chunk = 1 + tl.load(last_chunk_ptr + (idx_seq - 1)).to(
                        tl.int64
                    )

                # First chunk that is aligned on the mamba block boundary
                first_aligned_chunk = first_chunk + chunk_stride - 1

                # Calculate the number of computed tokens that were not
                # already cached
                if num_computed_tokens_ptr is not None:
                    num_unaligned_computed_tokens = (
                        tl.load(num_computed_tokens_ptr + idx_seq).to(tl.int64)
                        % block_size_to_align
                    )

                    if num_unaligned_computed_tokens > 0:
                        # If the number of computed tokens is not block aligned,
                        # then we need to shift the index accordingly
                        first_aligned_chunk -= (
                            num_unaligned_computed_tokens // chunk_size
                        )

                ssm_state_input_coord = (
                    first_aligned_chunk + idx_block_to_fill * chunk_stride
                )

    ssm_state_input_ptr = (
        states_ptr
        + ssm_state_input_coord * state_chunk_stride
        + (idx_nheads * state_nheads_stride)[:, None, None]
        + (idx_headdim * state_headdim_stride)[None, :, None]
        + (idx_dstate * state_dstate_stride)[None, None, :]
    )
    mask = (
        (idx_nheads < nheads)[:, None, None]
        & (idx_headdim < headdim)[None, :, None]
        & (idx_dstate < dstate)[None, None, :]
    )
    ssm_state_input = tl.load(ssm_state_input_ptr, mask, 0.0)

    ssm_state_output_ptr = (
        cache_state_ptr
        + ssm_state_output_coord * cache_state_cacheline_stride
        + (idx_nheads * cache_state_nheads_stride)[:, None, None]
        + (idx_headdim * cache_state_headdim_stride)[None, :, None]
        + (idx_dstate * cache_state_dstate_stride)[None, None, :]
    )

    tl.store(ssm_state_output_ptr, ssm_state_input, mask)


# TODO: Improve autotuning configuration
@triton.autotune(
    configs=[
        triton.Config(
            {
                "BLOCK_SIZE_M": 4,
                "BLOCK_SIZE_N": 4,
                "BLOCK_SIZE_H": 4,
            },
            num_stages=4,
            num_warps=8,
        ),
    ],
    key=["nheads", "dstate", "headdim"],
)
@triton.jit
def _init_state_fwd_kernel(
    # Pointers to matrices
    ssm_state_ptr,
    init_states_ptr,
    state_indices_ptr,
    initial_state_idx_ptr,
    has_initial_states_ptr,
    # Matrix dimensions
    nheads: tl.constexpr,
    headdim: tl.constexpr,
    dstate: tl.constexpr,
    # Strides
    state_indices_stride: tl.constexpr,
    state_chunk_stride: tl.constexpr,
    state_nheads_stride: tl.constexpr,
    state_headdim_stride: tl.constexpr,
    state_dstate_stride: tl.constexpr,
    cache_state_cacheline_stride: tl.constexpr,
    cache_state_nheads_stride: tl.constexpr,
    cache_state_headdim_stride: tl.constexpr,
    cache_state_dstate_stride: tl.constexpr,
    # Meta-parameters
    IS_CACHE_ENABLED: tl.constexpr,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr,
):
    # single-sequence id
    idx_seq = tl.program_id(0)

    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(2) // num_pid_n
    pid_n = tl.program_id(2) % num_pid_n
    pid_h = tl.program_id(1)

    # elements along the number of heads
    idx_nheads = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)

    # elements along the head dimension
    idx_headdim = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    # elements along the state dimension
    idx_dstate = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Indicates whether the sequence with idx_seq has an initial state
    has_initial_states = tl.load(has_initial_states_ptr + idx_seq).to(tl.int32)

    if IS_CACHE_ENABLED:
        # Collect the block index which contains the initial state
        ssm_state_init_index = tl.load(initial_state_idx_ptr + idx_seq).to(tl.int64)

        ssm_state_input_coord = tl.load(
            state_indices_ptr + idx_seq * state_indices_stride + ssm_state_init_index
        ).to(tl.int64)
    else:
        ssm_state_input_coord = tl.load(state_indices_ptr + idx_seq).to(tl.int64)

    ssm_state_init_input_ptr = (
        ssm_state_ptr
        + ssm_state_input_coord * state_chunk_stride
        + (idx_nheads * state_nheads_stride)[:, None, None]
        + (idx_headdim * state_headdim_stride)[None, :, None]
        + (idx_dstate * state_dstate_stride)[None, None, :]
    )
    # The mask_load is designed such that in case there is no initial state for
    # the sequence, 0s are loaded.
    mask_load = (
        (has_initial_states > 0)
        & (idx_nheads < nheads)[:, None, None]
        & (idx_headdim < headdim)[None, :, None]
        & (idx_dstate < dstate)[None, None, :]
    )
    ssm_state_init = tl.load(ssm_state_init_input_ptr, mask_load, 0.0)

    mask_store = (
        (idx_nheads < nheads)[:, None, None]
        & (idx_headdim < headdim)[None, :, None]
        & (idx_dstate < dstate)[None, None, :]
    )
    ssm_state_init_output_ptr = (
        init_states_ptr
        + idx_seq * cache_state_cacheline_stride
        + (idx_nheads * cache_state_nheads_stride)[:, None, None]
        + (idx_headdim * cache_state_headdim_stride)[None, :, None]
        + (idx_dstate * cache_state_dstate_stride)[None, None, :]
    )

    tl.store(ssm_state_init_output_ptr, ssm_state_init, mask_store)


def _state_cache_fwd(
    ssm_states: torch.Tensor = None,
    cache_ssm_states: torch.Tensor = None,
    cu_seqlens: torch.Tensor = None,
    state_indices: torch.Tensor = None,
    block_idx_first_scheduled_token: torch.Tensor = None,
    block_idx_last_scheduled_token: torch.Tensor = None,
    num_computed_tokens: torch.Tensor = None,
    last_chunk_indices: torch.Tensor = None,
    block_size_to_align: torch.Tensor = None,
    chunk_size: torch.Tensor = None,
):
    """
    ssm_states: (nchunks, nheads, headdim, dstate)
        Current SSM states that will be
    cache_ssm_states: (cache_lines, nheads, headdim, dstate)
        Cached ssm states.
        Updated inplace with ssm_states aligned to block_size_to_align
        and at the end of the individual sequences
    cu_seqlens: (batch + 1,)
        tensor containing the start token index of each sequence
    state_indices: (batch, n_blocks + padding) int32
        indicates the corresponding state index,
        like so: state = ssm_states[state_indices[batch_id]]
    block_idx_first_scheduled_token: (batch,), dtype int32
        The pointer into state_indices, where the first cache block to be filled is located.
    block_idx_last_scheduled_token: (batch,), dtype int32
        The pointer into state_indices, where the last cache block to be filled is located.
    num_computed_tokens: (batch,), dtype int32
        The number of tokens already completed for each sequence
    last_chunk_indices: (batch,), dtype int32
        The pointer into state_indices, where the last computed cache block of the sequence is located.
    block_size_to_align: int
        The block size to align the cached states to
    chunk_size: int
        The chunk_size used for computation, e.g., for chunked prefill
    """

    _, nheads, headdim, dstate = ssm_states.shape
    nseq = cu_seqlens.shape[0] - 1  # Actually is number of sequences in the "batch"

    if block_idx_last_scheduled_token is None:
        n_blocks_to_fill = None

        grid = lambda META: (
            nseq,
            triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
            triton.cdiv(headdim, META["BLOCK_SIZE_M"])
            * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        )
    else:
        n_blocks_to_fill = (
            block_idx_last_scheduled_token - block_idx_first_scheduled_token
        )

        grid = lambda META: (
            nseq
            * (
                n_blocks_to_fill.max() + 1
            ),  # The +1 is for the last state that is always stored
            triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
            triton.cdiv(headdim, META["BLOCK_SIZE_M"])
            * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
        )

    with torch.cuda.device(ssm_states.device.index):
        _state_cache_fwd_kernel[grid](
            ssm_states,
            cache_ssm_states,
            state_indices,
            n_blocks_to_fill,
            block_idx_first_scheduled_token,
            last_chunk_indices,
            num_computed_tokens,
            block_size_to_align,
            chunk_size,
            nheads,
            headdim,
            dstate,
            nseq,
            state_indices.stride(0),
            ssm_states.stride(0),
            ssm_states.stride(1),
            ssm_states.stride(2),
            ssm_states.stride(3),
            cache_ssm_states.stride(0),
            cache_ssm_states.stride(1),
            cache_ssm_states.stride(2),
            cache_ssm_states.stride(3),
        )


def _init_state_fwd(
    cache_ssm_states: torch.Tensor,
    init_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    state_indices: torch.Tensor,
    initial_state_idx: torch.Tensor,
    has_initial_states: torch.Tensor,
):
    """
    cache_ssm_states: (cache_lines, nheads, headdim, dstate)
        Cached ssm states.
        initial states are extracted from specific cache_lines
    init_states: (batch, nheads, headdim, dstate)
        tensor which will hold the initial states after this kernel
    cu_seqlens: (batch + 1,)
        tensor containing the start token index of each sequence
    state_indices: (batch, n_blocks + padding) int32
        indicates the corresponding state index,
        like so: state = cache_ssm_states[state_indices[batch_id]]
    initial_state_idx: (batch,), int32
        The pointer into state_indices, where the cache block containing the initial state is located.
    has_initial_states: (batch,) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
        [single boolean for each sequence in the batch: True or False]
    """

    _, nheads, headdim, dstate = cache_ssm_states.shape
    nseq = cu_seqlens.shape[0] - 1  # Number of sequences in the "batch"

    grid = lambda META: (
        nseq,
        triton.cdiv(nheads, META["BLOCK_SIZE_H"]),
        triton.cdiv(headdim, META["BLOCK_SIZE_M"])
        * triton.cdiv(dstate, META["BLOCK_SIZE_N"]),
    )

    with torch.cuda.device(init_states.device.index):
        _init_state_fwd_kernel[grid](
            cache_ssm_states,
            init_states,
            state_indices,
            initial_state_idx,
            has_initial_states,
            nheads,
            headdim,
            dstate,
            state_indices.stride(0),
            cache_ssm_states.stride(0),
            cache_ssm_states.stride(1),
            cache_ssm_states.stride(2),
            cache_ssm_states.stride(3),
            init_states.stride(0),
            init_states.stride(1),
            init_states.stride(2),
            init_states.stride(3),
            IS_CACHE_ENABLED=initial_state_idx is not None,
        )
