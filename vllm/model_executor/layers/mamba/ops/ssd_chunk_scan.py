# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2024, Tri Dao, Albert Gu.
# Adapted from https://github.com/state-spaces/mamba/blob/v2.2.4/mamba_ssm/ops/triton/ssd_chunk_scan.py

# ruff: noqa: E501,SIM102

import torch
from packaging import version

from vllm.triton_utils import tl, triton

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


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
                'BLOCK_SIZE_N': 64,
                'BLOCK_SIZE_K': 64
            },
            num_stages=4,
            num_warps=4),
        triton.Config(
            {
                'BLOCK_SIZE_M': 64,
                'BLOCK_SIZE_N': 128,
                'BLOCK_SIZE_K': 64
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
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
)
@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr,
    x_ptr,
    z_ptr,
    out_ptr,
    out_x_ptr,
    dt_ptr,
    dA_cumsum_ptr,
    seq_idx_ptr,
    C_ptr,
    states_ptr,
    D_ptr,
    initstates_ptr,
    chunk_indices_ptr,
    chunk_offsets_ptr,
    chunk_meta_num,
    cu_chunk_seqlens_ptr,
    # Matrix dimensions
    chunk_size,
    hdim,
    dstate,
    batch,
    seqlen,
    nheads_ngroups_ratio,
    # Strides
    stride_cb_batch,
    stride_cb_chunk,
    stride_cb_head,
    stride_cb_csize_m,
    stride_cb_csize_k,
    stride_x_batch,
    stride_x_seqlen,
    stride_x_head,
    stride_x_hdim,
    stride_z_batch,
    stride_z_seqlen,
    stride_z_head,
    stride_z_hdim,
    stride_out_batch,
    stride_out_seqlen,
    stride_out_head,
    stride_out_hdim,
    stride_dt_batch,
    stride_dt_chunk,
    stride_dt_head,
    stride_dt_csize,
    stride_dA_cs_batch,
    stride_dA_cs_chunk,
    stride_dA_cs_head,
    stride_dA_cs_csize,
    stride_seq_idx_batch,
    stride_seq_idx_seqlen,
    stride_C_batch,
    stride_C_seqlen,
    stride_C_head,
    stride_C_dstate,
    stride_states_batch,
    stride_states_chunk,
    stride_states_head,
    stride_states_hdim,
    stride_states_dstate,
    stride_init_states_batch,
    stride_init_states_head,
    stride_init_states_hdim,
    stride_init_states_dstate,
    stride_D_head,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
    HAS_INITSTATES: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1).to(tl.int64)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (
        pid_h // nheads_ngroups_ratio) * stride_cb_head
    chunk_seqlen_start = tl.load(cu_chunk_seqlens_ptr + pid_c)
    chunk_seqlen_end = tl.load(cu_chunk_seqlens_ptr + pid_c + 1)
    x_ptr += pid_b * stride_x_batch + chunk_seqlen_start * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + chunk_seqlen_start * stride_C_seqlen + (
        pid_h // nheads_ngroups_ratio) * stride_C_head

    # M-block offsets and prev states
    #  - logic in next block may override these if there is an active offset
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)

    seq_idx_ptr += pid_b * stride_seq_idx_batch + chunk_seqlen_start * stride_seq_idx_seqlen
    seq_idx = tl.load(seq_idx_ptr)
    seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen,
                           mask=pid_c >= 1,
                           other=-1)

    if HAS_INITSTATES and (seq_idx != seq_idx_prev):
        prev_states_ptr = initstates_ptr + seq_idx * stride_init_states_batch + pid_h * stride_init_states_head
        prev_states_hdim = stride_init_states_hdim
        prev_states_dstate = stride_init_states_dstate
    else:
        prev_states_ptr = states_ptr + (
            pid_c - 1) * stride_states_chunk + pid_h * stride_states_head
        prev_states_hdim = stride_states_hdim
        prev_states_dstate = stride_states_dstate

    chunk_size_limit = chunk_seqlen_end - chunk_seqlen_start

    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize,
                      mask=offs_m < chunk_size,
                      other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    offs_k_dstate = tl.arange(
        0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen +
                      offs_k_dstate[None, :] * stride_C_dstate)

    scale_m = tl.exp(dA_cs_m)
    if BLOCK_SIZE_DSTATE <= 128:
        C = tl.load(C_ptrs,
                    mask=(offs_m[:, None] < chunk_size_limit) &
                    (offs_k_dstate[None, :] < dstate),
                    other=0.0)

        if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
            # if no init states AND starting a new sequence, we need zeros
            prev_states = tl.zeros((BLOCK_SIZE_DSTATE, BLOCK_SIZE_N),
                                   dtype=C_ptr.dtype.element_ty)
        else:
            # otherwise read the previous state
            prev_states_ptrs = prev_states_ptr \
                    + offs_n[None, :] * prev_states_hdim \
                    + offs_k_dstate[:, None] * prev_states_dstate
            prev_states = tl.load(prev_states_ptrs,
                                  mask=(offs_k_dstate[:, None] < dstate) &
                                  (offs_n[None, :] < hdim),
                                  other=0.0)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)

        acc = tl.dot(C, prev_states) * scale_m[:, None]

    else:
        prev_states_ptrs = prev_states_ptr \
                        + offs_n[None, :] * prev_states_hdim \
                        + offs_k_dstate[:, None] * prev_states_dstate
        for k in range(0, dstate, BLOCK_SIZE_K):
            C = tl.load(C_ptrs,
                        mask=(offs_m[:, None] < chunk_size_limit) &
                        (offs_k_dstate[None, :] < dstate - k),
                        other=0.0)
            if not HAS_INITSTATES and (seq_idx != seq_idx_prev):
                prev_states = tl.zeros((BLOCK_SIZE_DSTATE, BLOCK_SIZE_K),
                                       dtype=C_ptr.dtype.element_ty)
            else:
                prev_states = tl.load(
                    prev_states_ptrs,
                    mask=(offs_k_dstate[:, None] < dstate - k) &
                    (offs_n[None, :] < hdim),
                    other=0.0)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc += tl.dot(C, prev_states)
            C_ptrs += BLOCK_SIZE_K
            prev_states_ptrs += BLOCK_SIZE_K
        acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m +
                        offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen +
                      offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min(
        (pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(cb_ptrs,
                     mask=(offs_m[:, None] < chunk_size) &
                     (offs_k[None, :] < chunk_size - k),
                     other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs,
                          mask=offs_k < chunk_size - k,
                          other=0.0).to(tl.float32)
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        cb *= tl.exp(dA_cs_m[:, None] - dA_cs_k[None, :])
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k,
                       other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(x_ptrs,
                    mask=(offs_k[:, None] < chunk_size_limit - k) &
                    (offs_n[None, :] < hdim),
                    other=0.0)
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n,
                        mask=offs_n < hdim,
                        other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(x_ptr + (offs_m[:, None] * stride_x_seqlen +
                                      offs_n[None, :] * stride_x_hdim),
                             mask=(offs_m[:, None] < chunk_size_limit) &
                             (offs_n[None, :] < hdim),
                             other=0.0).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        out_x_ptr += pid_b * stride_out_batch + chunk_seqlen_start * stride_out_seqlen + pid_h * stride_out_head
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] +
                                  offs_out_n[None, :])
        tl.store(out_x_ptrs,
                 acc,
                 mask=(offs_out_m[:, None] < chunk_size_limit) &
                 (offs_out_n[None, :] < hdim))

        z_ptr += pid_b * stride_z_batch + chunk_seqlen_start * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] +
                          stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_ptrs,
                    mask=(offs_out_m[:, None] < chunk_size_limit) &
                    (offs_out_n[None, :] < hdim),
                    other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += pid_b * stride_out_batch + chunk_seqlen_start * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] +
                          offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs,
             acc,
             mask=(offs_out_m[:, None] < chunk_size_limit) &
             (offs_out_n[None, :] < hdim))


def _chunk_scan_fwd(
    cb,
    x,
    dt,
    dA_cumsum,
    C,
    states,
    cu_chunk_seqlens,
    D=None,
    z=None,
    seq_idx=None,
    chunk_indices=None,
    chunk_offsets=None,
    initial_states=None,
    out=None,
):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads, )
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)

    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)

        if initial_states is not None:
            # with initial states, we need to take care of how
            # seq_idx crosses the boundaries
            assert batch == 1, "chunk scan only supports initial states with batch 1"
            assert chunk_indices is not None and chunk_offsets is not None, \
                "chunk_indices and chunk_offsets should have been set"
        else:
            chunk_indices, chunk_offsets = None, None
    else:
        chunk_indices, chunk_offsets = None, None

    assert out.shape == x.shape

    if z is not None:
        out_x = torch.empty_like(x)
        assert out_x.stride() == out.stride()
    else:
        out_x = None

    grid = lambda META: (triton.cdiv(
        chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(
            headdim, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2),
                  z.stride(3)) if z is not None else (0, 0, 0, 0))
    _chunk_scan_fwd_kernel[grid](
        cb,
        x,
        z,
        out,
        out_x,
        dt,
        dA_cumsum,
        seq_idx,
        C,
        states,
        D,
        initial_states,
        chunk_indices,
        chunk_offsets,
        len(chunk_indices) if chunk_indices is not None else 0,
        cu_chunk_seqlens,
        chunk_size,
        headdim,
        dstate,
        batch,
        seqlen,
        nheads // ngroups,
        cb.stride(0),
        cb.stride(1),
        cb.stride(2),
        cb.stride(3),
        cb.stride(4),
        x.stride(0),
        x.stride(1),
        x.stride(2),
        x.stride(3),
        z_strides[0],
        z_strides[1],
        z_strides[2],
        z_strides[3],
        out.stride(0),
        out.stride(1),
        out.stride(2),
        out.stride(3),
        dt.stride(0),
        dt.stride(2),
        dt.stride(1),
        dt.stride(3),
        dA_cumsum.stride(0),
        dA_cumsum.stride(2),
        dA_cumsum.stride(1),
        dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else
          (0, 0)),
        C.stride(0),
        C.stride(1),
        C.stride(2),
        C.stride(3),
        states.stride(0),
        states.stride(1),
        states.stride(2),
        states.stride(3),
        states.stride(4),
        *((initial_states.stride(0), initial_states.stride(1),
           initial_states.stride(2),
           initial_states.stride(3)) if initial_states is not None else
          (0, 0, 0, 0)),
        D.stride(0) if D is not None else 0,
        True,
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
        HAS_INITSTATES=initial_states is not None,
    )
    return out_x
