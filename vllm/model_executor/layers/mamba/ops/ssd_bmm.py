# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_bmm.py

# ruff: noqa: E501,SIM102

import math

import torch

from vllm.triton_utils import tl, triton


@triton.autotune(
    configs=[
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 64
            },
            num_stages=3,
            num_warps=8),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 256,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 128,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 32,
                'BLOCK_SIZE_K': 32
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 32,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=5,
            num_warps=2),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 32
            },
            num_stages=4,
            num_warps=2),
    ],
    key=['chunk_size', 'K', 'IS_CAUSAL'],
)
@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    out_ptr,
    seq_idx_ptr,
    # Matrix dimensions
    seqlen,
    chunk_size,
    K,
    ngroups,
    stride_a_batch,
    stride_a_seqlen,
    stride_a_head,
    stride_ak,
    stride_b_batch,
    stride_b_seqlen,
    stride_b_head,
    stride_bk,
    stride_out_batch,
    stride_out_chunk,
    stride_out_head,
    stride_outm,
    stride_outn,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    dot_dtype: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid_b = tl.program_id(axis=1)
    pid_ch = tl.program_id(axis=2).to(tl.int64)
    pid_c = pid_ch // ngroups
    pid_h = pid_ch - pid_c * ngroups
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    if IS_CAUSAL:
        if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
            return
    a_ptr += pid_b * stride_a_batch + pid_c * chunk_size * stride_a_seqlen + pid_h * stride_a_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + pid_h * stride_b_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_a_seqlen +
                      offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk +
                      offs_n[None, :] * stride_b_seqlen)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs,
                    mask=(offs_m[:, None] < chunk_size_limit) &
                    (offs_k[None, :] < K - k * BLOCK_SIZE_K),
                    other=0.0).to(dot_dtype)
        b = tl.load(b_ptrs,
                    mask=(offs_k[:, None] < K - k * BLOCK_SIZE_K) &
                    (offs_n[None, :] < chunk_size_limit),
                    other=0.0).to(dot_dtype)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen,
                            mask=offs_m < chunk_size_limit,
                            other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen,
                            mask=offs_n < chunk_size_limit,
                            other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    out = acc.to(out_ptr.dtype.element_ty)

    out_ptr += pid_b * stride_out_batch + pid_c * stride_out_chunk + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_outm * offs_m[:, None] +
                          offs_n[None, :] * stride_outn)
    tl.store(out_ptrs,
             out,
             mask=(offs_m[:, None] < chunk_size) &
             (offs_n[None, :] < chunk_size))


def _bmm_chunk_fwd(a,
                   b,
                   chunk_size,
                   seq_idx=None,
                   causal=False,
                   output_dtype=None):
    """
    Argument:
        a: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        b: (batch, seqlen, k) or (batch, seqlen, ngroups, k)
        seq_idx: (batch, seqlen) or None. out[i, j] for seq_idx[i] != seq_idx[j] will be zeroed out.
        causal: if True, then out[i, j] for i > j will be arbitrary, only out[i, j] for i <= j are
            guaranteed to be correct.
    Return:
        out: (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, ngroups, chunk_size, chunk_size)
    """
    # Check constraints.
    has_groups = a.dim() == 4
    if not has_groups:
        batch, seqlen, k = a.shape
    else:
        batch, seqlen, ngroups, k = a.shape
    assert b.shape == a.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if a.stride(-1) != 1 and a.stride(1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1 and b.stride(1) != 1:
        b = b.contiguous()
    nchunks = math.ceil(seqlen / chunk_size)
    # Allocates output.
    out_dtype = a.dtype if output_dtype is None else output_dtype
    out = torch.empty(
        (batch, nchunks, chunk_size, chunk_size) if not has_groups else
        (batch, nchunks, ngroups, chunk_size, chunk_size),
        device=a.device,
        dtype=out_dtype)
    dot_dtype = (tl.bfloat16
                 if a.dtype == torch.bfloat16 or b.dtype == torch.bfloat16 else
                 (tl.float16 if a.dtype == torch.float16
                  or b.dtype == torch.float16 else tl.float32))
    grid = lambda META: (triton.cdiv(
        chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(
            chunk_size, META['BLOCK_SIZE_N']), batch, nchunks
                         if not has_groups else nchunks * ngroups)
    with torch.cuda.device(a.device.index):
        _bmm_chunk_fwd_kernel[grid](
            a,
            b,
            out,
            seq_idx,
            seqlen,
            chunk_size,
            k,
            ngroups if has_groups else 1,
            a.stride(0),
            a.stride(1),
            0 if not has_groups else a.stride(2),
            a.stride(-1),
            b.stride(0),
            b.stride(1),
            0 if not has_groups else b.stride(2),
            b.stride(-1),
            out.stride(0),
            out.stride(1),
            0 if not has_groups else out.stride(2),
            out.stride(-2),
            out.stride(-1),
            *((seq_idx.stride(0),
               seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            causal,
            dot_dtype,
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return out
