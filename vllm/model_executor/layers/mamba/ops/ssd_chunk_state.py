# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from .softplus import softplus


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}),
        triton.Config({'BLOCK_SIZE_H': 2}),
        triton.Config({'BLOCK_SIZE_H': 4}),
        triton.Config({'BLOCK_SIZE_H': 8}),
        triton.Config({'BLOCK_SIZE_H': 16}),
        triton.Config({'BLOCK_SIZE_H': 32}),
        triton.Config({'BLOCK_SIZE_H': 64}),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_cumsum_fwd_kernel(
    # Pointers to matrices
    dt_ptr, A_ptr, dt_bias_ptr, dt_out_ptr, dA_cumsum_ptr,
    # Matrix dimension
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_dt_out_batch, stride_dt_out_chunk, stride_dt_out_head, stride_dt_out_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    dt_out_ptr += pid_b * stride_dt_out_batch + pid_c * stride_dt_out_chunk
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    dt_out_ptrs = dt_out_ptr + (offs_h[:, None] * stride_dt_out_head + offs_c[None, :] * stride_dt_out_csize)
    dA_cs_ptrs = dA_cumsum_ptr + (offs_h[:, None] * stride_dA_cs_head + offs_c[None, :] * stride_dA_cs_csize)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt = tl.where(dt <= 20.0, softplus(dt), dt)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    tl.store(dt_out_ptrs, dt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    dA = dt * A[:, None]
    dA_cs = tl.cumsum(dA, axis=1)
    tl.store(dA_cs_ptrs, dA_cs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_H': 1}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 2}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 4}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 8}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 16}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 32}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
        triton.Config({'BLOCK_SIZE_H': 64}, pre_hook=init_to_zero(["dA_ptr", "ddt_bias_ptr"])),
    ],
    key=['chunk_size', 'nheads'],
)
@triton.jit
def _chunk_cumsum_bwd_kernel(
    # Pointers to matrices
    ddA_ptr, ddt_out_ptr, dt_ptr, A_ptr, dt_bias_ptr,
    ddt_ptr, dA_ptr, ddt_bias_ptr,
    # Matrix dimensions
    batch, seqlen, nheads, chunk_size,
    dt_min, dt_max,
    # Strides
    stride_ddA_batch, stride_ddA_chunk, stride_ddA_head, stride_ddA_csize,
    stride_ddt_out_batch, stride_ddt_out_chunk, stride_ddt_out_head, stride_ddt_out_csize,
    stride_dt_batch, stride_dt_seqlen, stride_dt_head,
    stride_A_head,
    stride_dt_bias_head,
    stride_ddt_batch, stride_ddt_seqlen, stride_ddt_head,
    stride_dA_head,
    stride_ddt_bias_head,
    # Meta-parameters
    DT_SOFTPLUS: tl.constexpr,
    HAS_DT_BIAS: tl.constexpr,
    BLOCK_SIZE_H: tl.constexpr, BLOCK_SIZE_CHUNK: tl.constexpr,
):
    pid_b = tl.program_id(axis=0)
    pid_c = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    ddt_out_ptr += pid_b * stride_ddt_out_batch + pid_c * stride_ddt_out_chunk
    ddA_ptr += pid_b * stride_ddA_batch + pid_c * stride_ddA_chunk
    dt_ptr += pid_b * stride_dt_batch + pid_c * chunk_size * stride_dt_seqlen
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * chunk_size * stride_ddt_seqlen

    offs_h = pid_h * BLOCK_SIZE_H + tl.arange(0, BLOCK_SIZE_H)
    offs_c = tl.arange(0, BLOCK_SIZE_CHUNK)
    ddt_out_ptrs = ddt_out_ptr + (offs_h[:, None] * stride_ddt_out_head + offs_c[None, :] * stride_ddt_out_csize)
    ddA_ptrs = ddA_ptr + (offs_h[:, None] * stride_ddA_head + offs_c[None, :] * stride_ddA_csize)
    dt_ptrs = dt_ptr + (offs_h[:, None] * stride_dt_head + offs_c[None, :] * stride_dt_seqlen)
    ddt_ptrs = ddt_ptr + (offs_h[:, None] * stride_ddt_head + offs_c[None, :] * stride_ddt_seqlen)
    A_ptrs = A_ptr + offs_h * stride_A_head
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    ddA = tl.load(ddA_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    ddt_out = tl.load(ddt_out_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    A = tl.load(A_ptrs, mask=offs_h < nheads, other=0.0).to(tl.float32)
    ddt = ddA * A[:, None] + ddt_out
    dt = tl.load(dt_ptrs, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), other=0.0).to(tl.float32)
    if HAS_DT_BIAS:
        dt_bias = tl.load(dt_bias_ptr + offs_h * stride_dt_bias_head, mask=offs_h < nheads, other=0.0).to(tl.float32)
        dt += dt_bias[:, None]
    if DT_SOFTPLUS:
        dt_presoftplus = dt
        dt = tl.where(dt <= 20.0, softplus(dt), ddt)
    clamp_mask = (dt < dt_min) | (dt > dt_max)
    # As of Triton 2.2.0, tl.clamp is not available yet
    # dt = tl.clamp(dt, dt_min, dt_max)
    dt = tl.minimum(tl.maximum(dt, dt_min), dt_max)
    dt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), dt, 0.0)
    ddt = tl.where((offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit), ddt, 0.0)
    ddt = tl.where(clamp_mask, 0.0, ddt)
    if DT_SOFTPLUS:
        ddt = tl.where(dt_presoftplus <= 20.0, ddt * tl.sigmoid(dt_presoftplus), ddt)
    tl.store(ddt_ptrs, ddt, mask=(offs_h[:, None] < nheads) & (offs_c[None, :] < chunk_size_limit))
    dA = tl.sum(ddA * dt, axis=1)
    tl.atomic_add(dA_ptr + offs_h * stride_dA_head, dA, mask=offs_h < nheads)
    if HAS_DT_BIAS:
        ddt_bias = tl.sum(ddt, axis=1)
        tl.atomic_add(ddt_bias_ptr + offs_h * stride_ddt_bias_head, ddt_bias, mask=offs_h < nheads)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _chunk_state_fwd_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, states_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp((dA_cs_last - dA_cs_k)) * dt_k
        else:
            scale = tl.where(seq_idx_k == seq_idx_last, tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr", "ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, dstates_ptr, dt_ptr, dA_cumsum_ptr,
    dx_ptr, ddt_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_states_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_k[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_hdim + offs_k[:, None] * stride_states_dstate)
    if BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_states_dstate

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    acc *= tl.exp(dA_cs_last - dA_cs_m)[:, None]

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)
    ddA_cs = -(ddt * dt_m)
    ddA_cs_last = -tl.sum(ddA_cs)
    ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
    tl.atomic_add(ddA_cumsum_ptr + (chunk_size - 1) * stride_ddA_cs_csize, ddA_cs_last)

    dx = (acc * dt_m[:, None]).to(dx_ptr.dtype.element_ty)
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'dstate', 'hdim'],
)
@triton.jit
def _chunk_state_bwd_db_kernel(
    # Pointers to matrices
    x_ptr, dstates_ptr, b_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    db_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, dstate, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_db_batch, stride_db_seqlen, stride_db_split, stride_db_group, stride_db_dstate,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_DDA_CS: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_sg = tl.program_id(axis=2)
    pid_s = pid_sg // ngroups
    pid_g = pid_sg - pid_s * ngroups
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_x_head
    db_ptr += pid_b * stride_db_batch + pid_c * chunk_size * stride_db_seqlen + pid_g * stride_db_group + pid_s * stride_db_split
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_states_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dA_cs_head
    if HAS_DDA_CS:
        b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_g * stride_b_head
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_ddA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_k[None, :] * stride_x_hdim)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_dstate + offs_k[:, None] * stride_states_hdim)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    if HAS_DDA_CS:
        b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_n[None, :] * stride_b_dstate)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    for h in range(nheads_iter):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0)
        dstates = dstates.to(x_ptrs.dtype.element_ty)
        db = tl.dot(x, dstates)
        dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_last - dA_cs_m)
        else:
            scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
        db *= (scale * dt_m)[:, None]
        if HAS_DDA_CS:
            # This is the gradient wrt (dA_cs_last - dA_cs_m), i.e. the exclusive reverse cumsum
            ddA_cs = tl.sum(db * b, axis=1)
            tl.atomic_add(ddA_cumsum_ptrs + stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size - 1)
        acc += db
        x_ptrs += stride_x_head
        dstates_ptrs += stride_states_head
        dt_ptrs += stride_dt_head
        dA_cumsum_ptr += stride_dA_cs_head
        dA_cumsum_ptrs += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    # if HAS_SEQ_IDX:
    #     seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
    #     seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    #     acc = tl.where(seq_idx_m[:, None] == seq_idx_last, acc, 0.0)
    db_ptrs = db_ptr + (offs_m[:, None] * stride_db_seqlen + offs_n[None, :] * stride_db_dstate)
    tl.store(db_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 16, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_state_bwd_ddAcs_stable_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, dstates_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_states_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_k[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_states_hdim + offs_k[:, None] * stride_states_dstate)
    if BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates)
    else:
        acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_k[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_states_dstate

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    if not HAS_SEQ_IDX:
        scale = tl.exp(dA_cs_last - dA_cs_m)
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
    acc *= scale[:, None]

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    ddt = tl.sum(acc * x, axis=1)
    # ddA_cs = -(ddt * dt_m)
    # Triton 2.2.0 errors if we have the cumsum here, so we just write it out
    # then call torch.cumsum outside this kernel.
    # ddA_cs = tl.cumsum(ddt * dt_m)
    ddA_cs = ddt * dt_m
    ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    # tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
    tl.atomic_add(ddA_cumsum_ptrs + stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size - 1)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['hdim', 'dstate', 'chunk_size'],
)
@triton.jit
def _chunk_state_varlen_kernel(
    # Pointers to matrices
    x_ptr, b_ptr, dt_ptr, dA_cumsum_ptr, chunk_states_ptr, cu_seqlens_ptr, states_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_chunk_states_chunk, stride_chunk_states_head, stride_chunk_states_hdim, stride_chunk_states_dstate,
    stride_states_batch, stride_states_head, stride_states_hdim, stride_states_dstate,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(dstate, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    end_idx = tl.load(cu_seqlens_ptr + pid_b + 1)
    pid_c = (end_idx - 1) // chunk_size
    b_ptr += pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    x_ptr += pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    chunk_states_ptr += pid_c * stride_chunk_states_chunk + pid_h * stride_chunk_states_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_hdim + offs_k[None, :] * stride_x_seqlen)
    b_ptrs = b_ptr + (offs_n[None, :] * stride_b_dstate + offs_k[:, None] * stride_b_seqlen)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cs_last = tl.load(dA_cumsum_ptr + (end_idx - pid_c * chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = end_idx - pid_c * chunk_size
    start_idx = tl.load(cu_seqlens_ptr + pid_b)
    start_idx_cur = tl.maximum(start_idx - pid_c * chunk_size, 0)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k) & (offs_k[None, :] >= start_idx_cur - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate) & (offs_k[:, None] >= start_idx_cur - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size_limit - k, other=0.0).to(tl.float32)
        scale = tl.where((offs_k >= start_idx_cur - k) & (offs_k < chunk_size_limit - k),
                         tl.exp((dA_cs_last - dA_cs_k)) * dt_k, 0.0)
        b *= scale[:, None]
        b = b.to(x_ptr.dtype.element_ty)
        acc += tl.dot(x, b)
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        b_ptrs += BLOCK_SIZE_K * stride_b_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    # If the sequence starts after the last chunk idx, we don't need to add the contribution from the last chunk
    if start_idx < pid_c * chunk_size:
        chunk_states_ptrs = chunk_states_ptr + (offs_m[:, None] * stride_chunk_states_hdim + offs_n[None, :] * stride_chunk_states_dstate)
        chunk_states = tl.load(chunk_states_ptrs, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
        # scale = tl.where(start_idx < pid_c * chunk_size, tl.exp(dA_cs_last), 0.0)
        scale = tl.exp(dA_cs_last)
        acc += chunk_states * scale

    states = acc.to(states_ptr.dtype.element_ty)

    states_ptr += pid_b * stride_states_batch + pid_h * stride_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    states_ptrs = states_ptr + (offs_m[:, None] * stride_states_hdim + offs_n[None, :] * stride_states_dstate)
    c_mask = (offs_m[:, None] < hdim) & (offs_n[None, :] < dstate)
    tl.store(states_ptrs, states, mask=c_mask)


def _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads = dt.shape
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
    nchunks = math.ceil(seqlen / chunk_size)
    dt_out = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    dA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_fwd_kernel[grid_chunk_cs](
            dt, A, dt_bias, dt_out, dA_cumsum,
            batch, seqlen, nheads, chunk_size,
            dt_limit[0], dt_limit[1],
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            dt_out.stride(0), dt_out.stride(2), dt_out.stride(1), dt_out.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return dA_cumsum, dt_out


def _chunk_cumsum_bwd(ddA, ddt_out, dt, A, dt_bias=None, dt_softplus=False, dt_limit=(0.0, float("inf")), ddt=None):
    batch, seqlen, nheads = dt.shape
    _, _, nchunks, chunk_size = ddA.shape
    assert ddA.shape == (batch, nheads, nchunks, chunk_size)
    assert ddt_out.shape == (batch, nheads, nchunks, chunk_size)
    assert A.shape == (nheads,)
    if dt_bias is not None:
        assert dt_bias.shape == (nheads,)
        ddt_bias = torch.empty_like(dt_bias, dtype=torch.float32)
    else:
        ddt_bias = None
    if ddt is not None:
        assert ddt.shape == dt.shape
    else:
        ddt = torch.empty_like(dt)
    dA = torch.empty_like(A, dtype=torch.float32)
    grid_chunk_cs = lambda META: (batch, nchunks, triton.cdiv(nheads, META['BLOCK_SIZE_H']))
    with torch.cuda.device(dt.device.index):
        _chunk_cumsum_bwd_kernel[grid_chunk_cs](
            ddA, ddt_out, dt, A, dt_bias, ddt, dA, ddt_bias,
            batch, seqlen, nheads, chunk_size,
            dt_limit[0], dt_limit[1],
            ddA.stride(0), ddA.stride(2), ddA.stride(1), ddA.stride(3),
            ddt_out.stride(0), ddt_out.stride(2), ddt_out.stride(1), ddt_out.stride(3),
            dt.stride(0), dt.stride(1), dt.stride(2),
            A.stride(0),
            dt_bias.stride(0) if dt_bias is not None else 0,
            ddt.stride(0), ddt.stride(1), ddt.stride(2),
            dA.stride(0),
            ddt_bias.stride(0) if ddt_bias is not None else 0,
            dt_softplus,
            HAS_DT_BIAS=dt_bias is not None,
            BLOCK_SIZE_CHUNK=triton.next_power_of_2(chunk_size),
        )
    return ddt, dA, ddt_bias


def _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=None, states=None, states_in_fp32=True):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if states is not None:
        assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    else:
        states_dtype = torch.float32 if states_in_fp32 else B.dtype
        states = torch.empty((batch, nchunks, nheads, headdim, dstate), device=x.device, dtype=states_dtype)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_fwd_kernel[grid](
            x, B, states, dt, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return states


def _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if dx is not None:
        assert dx.shape == x.shape
    else:
        dx = torch.empty_like(x)
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dt.device, dtype=torch.float32)
    ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=dA_cumsum.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                       batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_dx_kernel[grid_dx](
            x, B, dstates, dt, dA_cumsum, dx, ddt, ddA_cumsum,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3),
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        )
    return dx, ddt.to(dt.dtype), ddA_cumsum.to(dA_cumsum.dtype)


def _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=None, B=None, ngroups=1):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    dstate = dstates.shape[-1]
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if B is not None:
        assert B.shape == (batch, seqlen, ngroups, dstate)
        B_strides = (B.stride(0), B.stride(1), B.stride(2), B.stride(3))
        # Use torch.empty since the Triton kernel will call init_to_zero
        ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=x.device, dtype=torch.float32)
        ddA_cumsum_strides = (ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3))
    else:
        B_strides = (0, 0, 0, 0)
        ddA_cumsum = None
        ddA_cumsum_strides = (0, 0, 0, 0)
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dB = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=x.device, dtype=torch.float32)
    grid_db = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                        batch * nchunks, nsplits * ngroups)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_db_kernel[grid_db](
            x, dstates, B, dt, dA_cumsum, seq_idx, dB, ddA_cumsum,
            chunk_size, dstate, headdim,
            batch, seqlen, nheads, nheads_per_program, ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            *B_strides,
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dB.stride(0), dB.stride(1), dB.stride(2), dB.stride(3), dB.stride(4),
            *ddA_cumsum_strides,
            HAS_DDA_CS=ddA_cumsum is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    dB = dB.sum(2)
    if ddA_cumsum is not None:
        # The first element of ddA_cumsum is always zero, since that dA_cumsum does not contribute
        # to the state of the chunk.
        # torch.cumsum(ddA_cumsum[..., 1:], dim=-1, out=ddA_cumsum[..., 1:])
        # But it's easier to just do the cumsum for all elements, the result will be the same.
        torch.cumsum(ddA_cumsum, dim=-1, out=ddA_cumsum)
    return dB if B is None else (dB, ddA_cumsum)


def _chunk_state_bwd_ddAcs_stable(B, x, dt, dA_cumsum, dstates, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # Use torch.empty since the Triton kernel will call init_to_zero
    ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                          batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_bwd_ddAcs_stable_kernel[grid_ddtcs](
            x, B, dstates, dt, dA_cumsum, seq_idx, ddA_cumsum,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            B.stride(0), B.stride(1), B.stride(2), B.stride(-1),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3),
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_M=max(triton.next_power_of_2(chunk_size), 16),
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        )
    torch.cumsum(ddA_cumsum[..., 1:], dim=-1, out=ddA_cumsum[..., 1:])
    return ddA_cumsum


def chunk_state_varlen(B, x, dt, dA_cumsum, cu_seqlens, chunk_states):
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
    states = torch.empty(batch, nheads, headdim, dstate, dtype=chunk_states.dtype, device=chunk_states.device)
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                    batch, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_state_varlen_kernel[grid](
            x, B, dt, dA_cumsum, chunk_states, cu_seqlens, states,
            headdim, dstate, chunk_size,
            total_seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2),
            B.stride(0), B.stride(1), B.stride(2),
            dt.stride(1), dt.stride(0), dt.stride(2),
            dA_cumsum.stride(1), dA_cumsum.stride(0), dA_cumsum.stride(2),
            chunk_states.stride(0), chunk_states.stride(1), chunk_states.stride(2), chunk_states.stride(3),
            states.stride(0), states.stride(1), states.stride(2), states.stride(3),
        )
    return states


class ChunkStateFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, x, dt, dA_cumsum, states_in_fp32=True):
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen <= nchunks * chunk_size
        _, _, ngroups, dstate = B.shape
        assert B.shape == (batch, seqlen, ngroups, dstate)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        if B.stride(-1) != 1:
            B = B.contiguous()
        if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
            x = x.contiguous()
        states = _chunk_state_fwd(B, x, dt, dA_cumsum, states_in_fp32=states_in_fp32)
        ctx.save_for_backward(B, x, dt, dA_cumsum)
        return states

    @staticmethod
    def backward(ctx, dstates):
        B, x, dt, dA_cumsum = ctx.saved_tensors
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        _, _, ngroups, dstate = B.shape
        assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
        if dstates.stride(-1) != 1:
            dstates = dstates.contiguous()
        dx, ddt, ddA_cumsum = _chunk_state_bwd_dx(B, x, dt, dA_cumsum, dstates)
        dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, ngroups=ngroups)
        dB = dB.to(B.dtype)
        return dB, dx, ddt, ddA_cumsum, None


def chunk_state(B, x, dt, dA_cumsum, states_in_fp32=True):
    """
    Argument:
        B: (batch, seqlen, ngroups, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    return ChunkStateFn.apply(B, x, dt, dA_cumsum, states_in_fp32)


def chunk_state_ref(B, x, dt, dA_cumsum):
    """
    Argument:
        B: (batch, seqlen, ngroups, headdim)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
    Return:
        states: (batch, nchunks, nheads, headdim, dstate)
    """
    # Check constraints.
    batch, seqlen, nheads, headdim = x.shape
    dstate = B.shape[-1]
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen <= nchunks * chunk_size
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    ngroups = B.shape[2]
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seqlen < nchunks * chunk_size:
        x = F.pad(x, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
        B = F.pad(B, (0, 0, 0, 0, 0, nchunks * chunk_size - seqlen))
    x = rearrange(x, "b (c l) h p -> b c l h p", l=chunk_size)
    B = rearrange(B, "b (c l) ... -> b c l ...", l=chunk_size)
    decay_states = torch.exp((dA_cumsum[:, :, :, -1:] - dA_cumsum))
    return torch.einsum("bclhn,bhcl,bhcl,bclhp->bchpn", B.to(x.dtype), decay_states.to(x.dtype), dt.to(x.dtype), x)
