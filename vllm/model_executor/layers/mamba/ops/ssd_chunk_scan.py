# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
from packaging import version

import torch
import torch.nn.functional as F

import triton
import triton.language as tl

from einops import rearrange, repeat

from .ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')


def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['chunk_size', 'hdim', 'dstate', 'IS_CAUSAL'],
)
@triton.jit
def _chunk_scan_fwd_kernel(
    # Pointers to matrices
    cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, prev_states_ptr, D_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head,
    # Meta-parameters
    IS_CAUSAL: tl.constexpr,
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
    IS_TRITON_22: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Without the if (pid_c > -1), with Triton 2.1.0, I get
    # Assertion `!(srcMmaLayout && dstMmaLayout) && "Unexpected mma -> mm a layout conversion"' failed.
    # With Triton 2.2.0, this works
    if IS_TRITON_22 or pid_c > -1:
        # Faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
        offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
        prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        if BLOCK_SIZE_DSTATE <= 128:
            C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
            prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
            prev_states = prev_states.to(C_ptr.dtype.element_ty)
            acc = tl.dot(C, prev_states) * scale_m[:, None]
        else:
            for k in range(0, dstate, BLOCK_SIZE_K):
                C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate - k), other=0.0)
                # C = (C * scale_m[:, None]).to(C_ptr.dtype.element_ty)
                prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
                prev_states = prev_states.to(C_ptr.dtype.element_ty)
                acc += tl.dot(C, prev_states)
                C_ptrs += BLOCK_SIZE_K
                prev_states_ptrs += BLOCK_SIZE_K
            acc *= scale_m[:, None]

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_k[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_k * stride_dt_csize
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit if not IS_CAUSAL else min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < chunk_size - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        # If there's seq_idx, we already set cb[i, j] = 0 for seq_idx[i] != seq_idx[j].
        # So we don't need masking wrt seq_idx here.
        cb *= tl.exp((dA_cs_m[:, None] - dA_cs_k[None, :]))
        dt_k = tl.load(dt_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        cb *= dt_k
        if IS_CAUSAL:
            mask = offs_m[:, None] >= k + offs_k[None, :]
            cb = tl.where(mask, cb, 0.0)
        cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < hdim), other=0.0)
        acc += tl.dot(cb, x)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_K * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_K * stride_dt_csize
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_out_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_out_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    if HAS_D:
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        x_residual = tl.load(x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim),
                             mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc += x_residual * D

    if HAS_Z:
        out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
        tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        acc *= z * tl.sigmoid(z)

    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    out_ptrs = out_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :] * stride_out_hdim)
    tl.store(out_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_N': 256}, num_stages=4, num_warps=4),
        # triton.Config({'BLOCK_SIZE_N': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_N': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_N': 32}, num_stages=4, num_warps=8),
    ],
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_fwd_kernel_wip(
    # Pointers to matrices
    cb_ptr, x_ptr, z_ptr, out_ptr, out_x_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, C_ptr, B_ptr, prev_states_ptr, D_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_B_batch, stride_B_seqlen, stride_B_head, stride_B_dstate,
    stride_states_batch, stride_states_chunk, stride_states_head, stride_states_hdim, stride_states_dstate,
    stride_D_head,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_Z: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_DSTATE: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_n = tl.program_id(axis=0)
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    B_ptr += pid_b * stride_B_batch + pid_c * chunk_size * stride_B_seqlen + (pid_h // nheads_ngroups_ratio) * stride_B_head
    prev_states_ptr += pid_b * stride_states_batch + pid_c * stride_states_chunk + pid_h * stride_states_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head

    offs_m = tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k_dstate = tl.arange(0, BLOCK_SIZE_DSTATE)

    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_k_dstate[None, :] * stride_C_dstate)
    B_ptrs = B_ptr + (offs_m[None, :] * stride_B_seqlen + offs_k_dstate[:, None] * stride_B_dstate)
    prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_states_hdim + offs_k_dstate[:, None] * stride_states_dstate)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_m[None, :] * stride_cb_csize_k)
    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None, :] * stride_out_hdim)

    prev_states = tl.load(prev_states_ptrs, mask=(offs_k_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
    # if pid_c == 0:
    #     if pid_b == 0:
    #         if pid_h == 0:
    #             tl.device_print("", prev_states)
    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    # dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # scale_m = tl.exp(dA_cs_m)
    # C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k_dstate[None, :] < dstate), other=0.0)
    # acc = tl.dot(C, prev_states.to(C_ptr.dtype.element_ty)) * scale_m[:, None]
    # cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_m[None, :] < chunk_size), other=0.0).to(tl.float32)
    # cb *= tl.exp((dA_cs_m[:, None] - dA_cs_m[None, :]))
    # dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # cb *= dt_m
    # mask = offs_m[:, None] >= offs_m[None, :]
    # cb = tl.where(mask, cb, 0.0)
    # cb = cb.to(x_ptr.dtype.element_ty)
    # x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0)
    # acc += tl.dot(cb, x)
    # if HAS_D:
    #     if D_HAS_HDIM:
    #         D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
    #     else:
    #         D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
    #     acc += x.to(tl.float32) * D
    # tl.store(out_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    for start_m in range(0, chunk_size_limit, BLOCK_SIZE_M):
        start_m = tl.multiple_of(start_m, BLOCK_SIZE_M)
        dA_cs_m = tl.load(dA_cumsum_ptr + (start_m + offs_m) * stride_dA_cs_csize, mask=offs_m < chunk_size - start_m, other=0.0).to(tl.float32)
        if HAS_SEQ_IDX:
            seq_idx_prev = tl.load(seq_idx_ptr + start_m - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
            seq_idx_m = tl.load(seq_idx_ptr + (start_m + offs_m) * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit - start_m, other=-1)
        if not HAS_SEQ_IDX:
            scale_m = tl.exp(dA_cs_m)
        else:
            scale_m = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        C = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit - start_m) & (offs_k_dstate[None, :] < dstate), other=0.0)
        acc = tl.dot(C, prev_states.to(C_ptr.dtype.element_ty)) * scale_m[:, None]
        # cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size - start_m) & (offs_m[None, :] < chunk_size - start_m), other=0.0).to(tl.float32)
        # cb *= tl.exp((dA_cs_m[:, None] - dA_cs_m[None, :]))
        dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size - start_m, other=0.0).to(tl.float32)
        # cb *= dt_m
        # mask = offs_m[:, None] >= offs_m[None, :]
        # cb = tl.where(mask, cb, 0.0)
        # cb = cb.to(x_ptr.dtype.element_ty)
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit - start_m) & (offs_n[None, :] < hdim), other=0.0)
        # acc += tl.dot(cb, x)

        if HAS_D:
            if D_HAS_HDIM:
                D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
            else:
                D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
            acc += x.to(tl.float32) * D

        # if HAS_Z:
        #     out_x_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
        #     out_x_ptrs = out_x_ptr + (stride_out_seqlen * offs_out_m[:, None] + offs_out_n[None, :])
        #     tl.store(out_x_ptrs, acc, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim))

        #     z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
        #     z_ptrs = z_ptr + (stride_z_seqlen * offs_out_m[:, None] + stride_z_hdim * offs_out_n[None, :])
        #     z = tl.load(z_ptrs, mask=(offs_out_m[:, None] < chunk_size_limit) & (offs_out_n[None, :] < hdim), other=0.0).to(tl.float32)
        #     acc *= z * tl.sigmoid(z)

        tl.store(out_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit - start_m) & (offs_n[None, :] < hdim))

        # TODO: this is not correct, and quite a bit slower
        if start_m + BLOCK_SIZE_M < chunk_size_limit:
            # B = tl.load(B_ptrs, mask=(offs_m[None, :] < chunk_size_limit - start_m) & (offs_k_dstate[:, None] < dstate), other=0.0).to(tl.float32)
            B = tl.load(B_ptrs, mask=(offs_m[None, :] < chunk_size_limit - start_m) & (offs_k_dstate[:, None] < dstate), other=0.0)
            dA_cs_last = tl.load(dA_cumsum_ptr + (start_m + BLOCK_SIZE_M) * stride_dA_cs_csize).to(tl.float32)
            # TODO: seq_idx
            scale = tl.exp((dA_cs_last - dA_cs_m)) * dt_m
            # B *= scale
            B = B.to(x_ptr.dtype.element_ty)
            tmp = tl.dot(B, x)
            prev_states += tmp.to(prev_states.dtype)

        C_ptrs += BLOCK_SIZE_M * stride_C_seqlen
        B_ptrs += BLOCK_SIZE_M * stride_B_seqlen
        cb_ptrs += BLOCK_SIZE_M * stride_cb_csize_m + BLOCK_SIZE_M * stride_cb_csize_k
        x_ptrs += BLOCK_SIZE_M * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_M * stride_dt_csize
        out_ptrs += BLOCK_SIZE_M * stride_out_seqlen


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_M': 64}),
        triton.Config({'BLOCK_SIZE_M': 128}),
        triton.Config({'BLOCK_SIZE_M': 256}),
    ],
    key=["chunk_size", "hdim"],
)
@triton.jit
def _chunk_scan_bwd_dz_kernel(
    # Pointers to matrices
    dout_ptr, out_ptr, z_ptr, x_ptr, D_ptr, outz_ptr, dz_ptr, dout_x_ptr, dD_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_z_batch, stride_z_seqlen, stride_z_head, stride_z_hdim,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_D_head,
    stride_outz_batch, stride_outz_seqlen, stride_outz_head, stride_outz_hdim,
    stride_dz_batch, stride_dz_seqlen, stride_dz_head, stride_dz_hdim,
    stride_doutx_batch, stride_doutx_seqlen, stride_doutx_head, stride_doutx_hdim,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    HAS_DDACS: tl.constexpr,
    RECOMPUTE_OUTPUT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dout_x_ptr += pid_b * stride_doutx_batch + pid_c * chunk_size * stride_doutx_seqlen + pid_h * stride_doutx_head
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    z_ptr += pid_b * stride_z_batch + pid_c * chunk_size * stride_z_seqlen + pid_h * stride_z_head
    dz_ptr += pid_b * stride_dz_batch + pid_c * chunk_size * stride_dz_seqlen + pid_h * stride_dz_head
    if RECOMPUTE_OUTPUT:
        outz_ptr += pid_b * stride_outz_batch + pid_c * chunk_size * stride_outz_seqlen + pid_h * stride_outz_head
    if HAS_DDACS:
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    if HAS_D:
        x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dout_x_ptrs = dout_x_ptr + (offs_m[:, None] * stride_doutx_seqlen + offs_n[None, :] * stride_doutx_hdim)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None, :] * stride_out_hdim)
    z_ptrs = z_ptr + (offs_m[:, None] * stride_z_seqlen + offs_n[None, :] * stride_z_hdim)
    dz_ptrs = dz_ptr + (offs_m[:, None] * stride_dz_seqlen + offs_n[None, :] * stride_dz_hdim)
    if RECOMPUTE_OUTPUT:
        outz_ptrs = outz_ptr + (offs_m[:, None] * stride_outz_seqlen + offs_n[None, :] * stride_outz_hdim)
    if HAS_D:
        x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    out = tl.load(out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    z = tl.load(z_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    z_sigmoid = tl.sigmoid(z)
    if RECOMPUTE_OUTPUT:
        outz = out * z * z_sigmoid
        tl.store(outz_ptrs, outz, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    dz = dout * out * z_sigmoid * (1 + z * (1 - z_sigmoid))
    tl.store(dz_ptrs, dz, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    dout *= z * z_sigmoid
    tl.store(dout_x_ptrs, dout, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))
    if HAS_D:
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            dD = tl.sum(dout * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            dD = tl.sum(dout * x)
            tl.store(dD_ptr, dD)
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        out -= x * D
    if HAS_DDACS:
        ddA_cs = tl.sum(dout * out, axis=1)
        tl.store(ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size)


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
def _chunk_scan_bwd_dstates_kernel(
    # Pointers to matrices
    dout_ptr, c_ptr, dprev_states_ptr, dA_cumsum_ptr, seq_idx_ptr,
    # Matrix dimensions
    hdim, dstate, chunk_size,
    batch, seqlen, nchunks, nheads_ngroups_ratio,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_c_batch, stride_c_seqlen, stride_c_head, stride_c_dstate,
    stride_dprev_states_batch, stride_dprev_states_chunk, stride_dprev_states_head, stride_dprev_states_hdim, stride_dprev_states_dstate,
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
    c_ptr += pid_b * stride_c_batch + pid_c * chunk_size * stride_c_seqlen + (pid_h // nheads_ngroups_ratio) * stride_c_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_hdim + offs_k[None, :] * stride_dout_seqlen)
    c_ptrs = c_ptr + (offs_n[None, :] * stride_c_dstate + offs_k[:, None] * stride_c_seqlen)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    if HAS_SEQ_IDX:
        seq_idx_ptrs = seq_idx_ptr + offs_k * stride_seq_idx_seqlen

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
    for k in range(0, chunk_size_limit, BLOCK_SIZE_K):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < hdim) & (offs_k[None, :] < chunk_size_limit - k), other=0.0).to(tl.float32)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < chunk_size - k, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale_k = tl.exp(dA_cs_k)
        else:
            seq_idx_k = tl.load(seq_idx_ptrs, mask=offs_k < chunk_size_limit - k, other=-1)
            scale_k = tl.where(seq_idx_k == seq_idx_prev, tl.exp(dA_cs_k), 0.0)
        dout = (dout * scale_k).to(dout_ptr.dtype.element_ty)
        c = tl.load(c_ptrs, mask=(offs_k[:, None] < chunk_size_limit - k) & (offs_n[None, :] < dstate), other=0.0)
        acc += tl.dot(dout, c)
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        c_ptrs += BLOCK_SIZE_K * stride_c_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize
        if HAS_SEQ_IDX:
            seq_idx_ptrs += BLOCK_SIZE_K * stride_seq_idx_seqlen
    out = acc.to(dprev_states_ptr.dtype.element_ty)

    dprev_states_ptr += pid_b * stride_dprev_states_batch + pid_c * stride_dprev_states_chunk + pid_h * stride_dprev_states_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dprev_states_ptrs = dprev_states_ptr + (offs_m[:, None] * stride_dprev_states_hdim + offs_n[None, :] * stride_dprev_states_dstate)
    tl.store(dprev_states_ptrs, out, mask=(offs_m[:, None] < hdim) & (offs_n[None, :] < dstate))


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
def _chunk_scan_bwd_dc_kernel(
    # Pointers to matrices
    dout_ptr, prev_states_ptr, C_ptr, dA_cumsum_ptr, seq_idx_ptr,
    dc_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, dstate, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_prev_states_batch, stride_prev_states_chunk, stride_prev_states_head, stride_prev_states_hdim, stride_prev_states_dstate,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dc_batch, stride_dc_seqlen, stride_dc_split, stride_dc_group, stride_dc_dstate,
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
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dout_head
    dc_ptr += pid_b * stride_dc_batch + pid_c * chunk_size * stride_dc_seqlen + pid_g * stride_dc_group + pid_s * stride_dc_split
    prev_states_ptr += pid_b * stride_prev_states_batch + pid_c * stride_prev_states_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_prev_states_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dA_cs_head
    if HAS_DDA_CS:
        C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + pid_g * stride_C_head
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_ddA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_prev_states_dstate + offs_k[:, None] * stride_prev_states_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize
    if HAS_DDA_CS:
        C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_n[None, :] * stride_C_dstate)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        c = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    for h in range(nheads_iter):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
        prev_states = tl.load(prev_states_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0)
        prev_states = prev_states.to(dout_ptrs.dtype.element_ty)
        dc = tl.dot(dout, prev_states)
        dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        if not HAS_SEQ_IDX:
            scale = tl.exp(dA_cs_m)
        else:
            scale = tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
        dc *= scale[:, None]
        if HAS_DDA_CS:
            ddA_cs = tl.sum(dc * c, axis=1)
            tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)
        acc += dc
        dout_ptrs += stride_dout_head
        prev_states_ptrs += stride_prev_states_head
        dA_cumsum_ptrs += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptrs += stride_ddA_cs_head
    # if HAS_SEQ_IDX:
    #     seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
    #     seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
    #     acc = tl.where(seq_idx_m[:, None] == seq_idx_prev, acc, 0.0)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dc_ptrs = dc_ptr + (offs_m[:, None] * stride_dc_seqlen + offs_n[None, :] * stride_dc_dstate)
    tl.store(dc_ptrs, acc, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate))


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=8, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=5, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=4, pre_hook=init_to_zero(["ddt_ptr"])),
    ],
    key=['chunk_size', 'hdim'],
)
@triton.jit
def _chunk_scan_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, cb_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, D_ptr,
    dx_ptr, ddt_ptr, # dD_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_D_head,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    # stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_hdim, stride_dD_csize,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(hdim, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    # if HAS_D:
    #     dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # Idk why limiting K_MAX gives wrong results, is it a Triton bug?
    # K_MAX = min((pid_m + 1) * BLOCK_SIZE_M, chunk_size_limit)
    K_MAX = chunk_size_limit
    for k in range(0, K_MAX, BLOCK_SIZE_K):
        # For some reason setting mask to (offs_m[:, None] < chunk_size_limit) is much slower
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_k[None, :] < K_MAX - k), other=0.0)
        dout = tl.load(dout_ptrs, mask=(offs_k[:, None] < K_MAX - k) & (offs_n[None, :] < hdim), other=0.0)
        dA_cs_k = tl.load(dA_cumsum_ptrs, mask=offs_k < K_MAX - k, other=0.0).to(tl.float32)
        cb *= tl.exp(dA_cs_k[None, :] - dA_cs_m[:, None])
        # If we don't have the (k + offs_k[None, :] < K_MAX) mask, for indices outside this range,
        # we might have dA_cs_m = 0.0 and dA_cs_k very negative, and tl.exp will return inf.
        # Multiplying with cb, which is 0.0 outside the range, will make the result NaN.
        # This will cause NaN in acc, and hence NaN in dx and ddt.
        mask = (k + offs_k[None, :] >= offs_m[:, None]) & (k + offs_k[None, :] < K_MAX)
        cb = tl.where(mask, cb, 0.0)
        cb = cb.to(dout_ptr.dtype.element_ty)
        acc += tl.dot(cb, dout)
        cb_ptrs += BLOCK_SIZE_K * stride_cb_csize_k
        dout_ptrs += BLOCK_SIZE_K * stride_dout_seqlen
        dA_cumsum_ptrs += BLOCK_SIZE_K * stride_dA_cs_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    dx = acc * dt_m[:, None]
    dx_ptr += pid_b * stride_dx_batch + pid_c * chunk_size * stride_dx_seqlen + pid_h * stride_dx_head
    dx_ptrs = dx_ptr + (offs_m[:, None] * stride_dx_seqlen + offs_n[None, :] * stride_dx_hdim)
    if HAS_D:
        dout_res_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
        dout_res = tl.load(dout_res_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        dx += dout_res * D
    tl.store(dx_ptrs, dx, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim))

    x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)

    # if HAS_D:
    #     dout_new_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_csize + offs_n[None, :] * stride_dout_hdim)
    #     dout = tl.load(dout_new_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0).to(tl.float32)
    #     dD = tl.sum(x * dout, axis=0)
    #     tl.store(dD_ptr + offs_n * stride_dD_hdim, dD, mask=offs_n < N)


# Disabling HAS_DDA_CS for now since it's much slower
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 16}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 16}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 128}, num_stages=4, num_warps=8),
    ],
    key=['chunk_size', 'hdim'],
)
# @triton.heuristics({"BLOCK_SIZE_N": lambda args: max(triton.next_power_of_2(args["chunk_size"]), 16)})
# @triton.heuristics({"BLOCK_SIZE_N": lambda args: 32})
@triton.jit
def _chunk_scan_bwd_dcb_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, cb_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr,
    dcb_ptr, ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads, nheads_per_program, ngroups,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_dcb_batch, stride_dcb_chunk, stride_dcb_split, stride_dcb_group, stride_dcb_csize_m, stride_dcb_csize_n,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize_m, stride_ddA_cs_csize_n,
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
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_x_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_dA_cs_head
    if HAS_DDA_CS:
        cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + pid_g * stride_cb_head
        ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + (pid_g * (nheads // ngroups) + pid_s * nheads_per_program) * stride_ddA_cs_head + pid_m * stride_ddA_cs_csize_m
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    if HAS_DDA_CS:
        cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)
        ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_n * stride_ddA_cs_csize_n

    if pid_n * BLOCK_SIZE_N >= (pid_m + 1) * BLOCK_SIZE_M:
        dcb_ptr += pid_b * stride_dcb_batch + pid_c * stride_dcb_chunk + pid_g * stride_dcb_group + pid_s * stride_dcb_split
        dcb_ptrs = dcb_ptr + (offs_m[:, None] * stride_dcb_csize_m + offs_n[None, :] * stride_dcb_csize_n)
        tl.store(dcb_ptrs, tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=dcb_ptr.dtype.element_ty), mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))
        return

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    chunk_size_limit_n = min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if HAS_DDA_CS:
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size), other=0.0).to(tl.float32)
    nheads_iter = min(nheads_per_program, nheads // ngroups - pid_s * nheads_per_program)
    for h in range(nheads_iter):
        dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n), other=0.0)
        dcb = tl.dot(dout, x)
        dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size, other=0.0).to(tl.float32)
        dcb *= dt_n
        dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
        dA_cs_n = tl.load(dA_cumsum_ptr + offs_n * stride_dA_cs_csize, mask=offs_n < chunk_size_limit, other=0.0).to(tl.float32)
        dcb *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
        if HAS_DDA_CS:
            tl.static_assert(not HAS_SEQ_IDX, "HAS_SEQ_IDX not supported with HAS_DDA_CS yet")
            ddA_cs = dcb * cb
            mask = offs_m[:, None] >= offs_n[None, :] + 1
            ddA_cs = tl.where(mask, ddA_cs, 0.0)
            ddA_cs = tl.cumsum(ddA_cs, axis=1)
            ddA_cs = tl.where(mask, ddA_cs, 0.0)
            ddA_cs = tl.sum(ddA_cs, axis=0)
            tl.store(ddA_cumsum_ptrs + stride_ddA_cs_csize_n, ddA_cs, mask=offs_n < chunk_size - 1)
            tl.store(ddA_cumsum_ptr, 0.0)
        acc += dcb
        dout_ptrs += stride_dout_head
        x_ptrs += stride_x_head
        dt_ptrs += stride_dt_head
        dA_cumsum_ptr += stride_dA_cs_head
        if HAS_DDA_CS:
            ddA_cumsum_ptr += stride_ddA_cs_head
            ddA_cumsum_ptrs += stride_ddA_cs_head
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_n = tl.load(seq_idx_ptr + offs_n * stride_seq_idx_seqlen, mask=offs_n < chunk_size_limit, other=-2)
        acc = tl.where(seq_idx_m[:, None] == seq_idx_n[None, :], acc, 0.0)
    mask = offs_m[:, None] >= offs_n[None, :]
    acc = tl.where(mask, acc, 0.0)
    dcb_ptr += pid_b * stride_dcb_batch + pid_c * stride_dcb_chunk + pid_g * stride_dcb_group + pid_s * stride_dcb_split
    dcb_ptrs = dcb_ptr + (offs_m[:, None] * stride_dcb_csize_m + offs_n[None, :] * stride_dcb_csize_n)
    tl.store(dcb_ptrs, acc, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size))


# Not numerically stable and should not be used. Leaving here for reference.
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32}),
        triton.Config({'BLOCK_SIZE_M': 64}),
        triton.Config({'BLOCK_SIZE_M': 128}),
        triton.Config({'BLOCK_SIZE_M': 256}),
    ],
    key=["chunk_size", "hdim"],
)
@triton.jit
def _chunk_scan_bwd_ddAcs_unstable_kernel(
    # Pointers to matrices
    dout_ptr, out_ptr, dt_ptr, ddt_ptr, x_ptr, D_ptr,
    ddA_cumsum_ptr, dD_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_out_batch, stride_out_seqlen, stride_out_head, stride_out_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_D_head,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
    SUBTRACT_DDTDT: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    out_ptr += pid_b * stride_out_batch + pid_c * chunk_size * stride_out_seqlen + pid_h * stride_out_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    if HAS_D:
        x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    out_ptrs = out_ptr + (offs_m[:, None] * stride_out_seqlen + offs_n[None, :] * stride_out_hdim)
    if HAS_D:
        x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    out = tl.load(out_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    if HAS_D:
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
        if D_HAS_HDIM:
            dD = tl.sum(dout * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
            D = tl.load(D_ptr + pid_h * stride_D_head + offs_n, mask=offs_n < hdim, other=0.0).to(tl.float32)
        else:
            dD = tl.sum(dout * x)
            tl.store(dD_ptr, dD)
            D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        out -= x * D
    ddA_cs = tl.sum(dout * out, axis=1)
    if SUBTRACT_DDTDT:
        dt = tl.load(dt_ptr + offs_m * stride_dt_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        ddt = tl.load(ddt_ptr + offs_m * stride_ddt_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
        ddA_cs -= dt * ddt
    tl.store(ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize, ddA_cs, mask=offs_m < chunk_size)


@triton.autotune(
    configs=[
        # triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 16, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        # triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_K': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 16}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 32}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 128}, num_stages=4, num_warps=8),
    ],
    key=['chunk_size', 'hdim'],
)
@triton.jit
def _chunk_scan_bwd_ddAcs_stable_kernel_old(
    # Pointers to matrices
    x_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, cb_ptr,
    ddAcs_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    stride_ddAcs_batch, stride_ddAcs_chunk, stride_ddAcs_head, stride_ddAcs_csize_m, stride_ddAcs_csize_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    num_pid_n = tl.cdiv(chunk_size, BLOCK_SIZE_N)
    pid_m = tl.program_id(axis=0) // num_pid_n
    pid_n = tl.program_id(axis=0) % num_pid_n

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    chunk_size_limit_n = min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    # Doing a matmul loop with cumsum later on will cause Triton to crash
    # Instead we do just one big matmul
    # acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # for k in range(0, hdim, BLOCK_SIZE_K):
    #     dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim - k), other=0.0)
    #     x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim - k) & (offs_n[None, :] < chunk_size_limit), other=0.0)
    #     acc += tl.dot(dout, x)
    #     dout_ptrs += BLOCK_SIZE_K * stride_dout_hdim
    #     x_ptrs += BLOCK_SIZE_K * stride_x_hdim
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
    x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n), other=0.0)
    acc = tl.dot(dout, x)
    cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size), other=0.0).to(tl.float32)
    acc *= cb
    dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size, other=0.0).to(tl.float32)
    acc *= dt_n
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    dA_cs_n = tl.load(dA_cumsum_ptr + offs_n * stride_dA_cs_csize, mask=offs_n < chunk_size, other=0.0).to(tl.float32)
    acc *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
    mask = offs_m[:, None] >= offs_n[None, :] + 1
    acc = tl.where(mask, acc, 0.0)
    acc = tl.cumsum(acc, axis=1)
    acc = tl.where(mask, acc, 0.0)
    ddA_cs = tl.sum(acc, axis=0)
    ddAcs_ptr += pid_b * stride_ddAcs_batch + pid_c * stride_ddAcs_chunk + pid_h * stride_ddAcs_head + pid_m * stride_ddAcs_csize_m
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ddAcs_ptrs = ddAcs_ptr + offs_n * stride_ddAcs_csize_n
    tl.store(ddAcs_ptrs + stride_ddAcs_csize_n, ddA_cs, mask=offs_n < chunk_size - 1)
    tl.store(ddAcs_ptr, 0.0)

    # offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, 64)
    # offs_k = tl.arange(0, BLOCK_SIZE_K)
    # dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    # x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    # dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    # cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)

    # chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    # chunk_size_limit_n = min(chunk_size_limit, (pid_m + 1) * BLOCK_SIZE_M)
    # rowsum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    # dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
    # dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # ddAcs_ptr += pid_b * stride_ddAcs_batch + pid_c * stride_ddAcs_chunk + pid_h * stride_ddAcs_head + pid_m * stride_ddAcs_csize_m
    # ddAcs_ptrs = ddAcs_ptr + offs_n * stride_ddAcs_csize_n
    # for n in range(0, chunk_size_limit_n, 64):
    #     x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n - n), other=0.0)
    #     acc = tl.dot(dout, x)
    #     cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size - n), other=0.0).to(tl.float32)
    #     acc *= cb
    #     dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size - n, other=0.0).to(tl.float32)
    #     acc *= dt_n
    #     dA_cs_n = tl.load(dA_cumsum_ptr + offs_n * stride_dA_cs_csize, mask=offs_n < chunk_size - n, other=0.0).to(tl.float32)
    #     acc *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
    #     mask = offs_m[:, None] >= offs_n[None, :] + 1 + n
    #     acc = tl.where(mask, acc, 0.0)
    #     acc = tl.cumsum(acc, axis=1)
    #     acc = tl.where(mask, acc, 0.0)
    #     ddA_cs = tl.sum(acc, axis=0)
    #     tl.store(ddAcs_ptrs, ddA_cs, mask=offs_n < chunk_size - 1 - n)
    # # tl.store(ddAcs_ptr, 0.0)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        # triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4),
    ],
    key=['chunk_size', 'hdim'],
)
@triton.jit
def _chunk_scan_bwd_ddAcs_stable_kernel(
    # Pointers to matrices
    x_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, cb_ptr,
    ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, hdim,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_n,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize_m, stride_ddA_cs_csize_n,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_bc = tl.program_id(axis=1)
    pid_c = pid_bc // batch
    pid_b = pid_bc - pid_c * batch
    pid_h = tl.program_id(axis=2)
    pid_m = tl.program_id(axis=0)

    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head + pid_m * stride_ddA_cs_csize_m

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    x_ptrs = x_ptr + (offs_n[None, :] * stride_x_seqlen + offs_k[:, None] * stride_x_hdim)
    dt_ptrs = dt_ptr + offs_n * stride_dt_csize
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_n[None, :] * stride_cb_csize_n)
    ddAcs_ptrs = ddA_cumsum_ptr + offs_n * stride_ddA_cs_csize_n
    tl.store(ddA_cumsum_ptr, 0.0)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    rowsum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size, other=0.0).to(tl.float32)
    # Actually hi is (pid_m + 1) * BLOCK_SIZE_M - 1 but subtracting 1 makes it slower
    lo, hi = 0, (pid_m + 1) * BLOCK_SIZE_M
    # lo, hi = 0, chunk_size
    for start_n in range(lo, hi, BLOCK_SIZE_N):
        start_n = tl.multiple_of(start_n, BLOCK_SIZE_N)
        # Doing a matmul loop with cumsum later on will cause Triton to crash
        # Instead we do just one big matmul
        # acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        # for k in range(0, hdim, BLOCK_SIZE_K):
        #     dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim - k), other=0.0)
        #     x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim - k) & (offs_n[None, :] < chunk_size_limit), other=0.0)
        #     acc += tl.dot(dout, x)
        #     dout_ptrs += BLOCK_SIZE_K * stride_dout_hdim
        #     x_ptrs += BLOCK_SIZE_K * stride_x_hdim
        # x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit_n), other=0.0)
        x = tl.load(x_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < chunk_size_limit - start_n), other=0.0)
        acc = tl.dot(dout, x)
        dt_n = tl.load(dt_ptrs, mask=offs_n < chunk_size - start_n, other=0.0).to(tl.float32)
        acc *= dt_n
        # If there's seq_idx, we already zero'ed out cb[i, j] for seq_idx[i] != seq_idx[j]
        cb = tl.load(cb_ptrs, mask=(offs_m[:, None] < chunk_size) & (offs_n[None, :] < chunk_size - start_n), other=0.0).to(tl.float32)
        acc *= cb
        dA_cs_n = tl.load(dA_cumsum_ptr + (start_n + offs_n) * stride_dA_cs_csize, mask=offs_n < chunk_size - start_n, other=0.0).to(tl.float32)
        acc *= tl.exp(dA_cs_m[:, None] - dA_cs_n[None, :])
        mask = offs_m[:, None] >= start_n + offs_n[None, :] + 1
        acc = tl.where(mask, acc, 0.0)
        rowsum_new = rowsum + tl.sum(acc, axis=1)
        acc = rowsum[:, None] + tl.cumsum(acc, axis=1)
        rowsum = rowsum_new
        acc = tl.where(mask, acc, 0.0)
        ddA_cs = tl.sum(acc, axis=0)
        tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, ddA_cs, mask=offs_n < chunk_size - start_n - 1)
        x_ptrs += BLOCK_SIZE_N * stride_x_seqlen
        dt_ptrs += BLOCK_SIZE_N * stride_dt_csize
        cb_ptrs += BLOCK_SIZE_N * stride_cb_csize_n
        ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n

    # Need to zero out the rest, since we'll be summing the rows together
    for start_n in range(hi, chunk_size, BLOCK_SIZE_N):
        tl.store(ddAcs_ptrs + stride_ddA_cs_csize_n, tl.zeros((BLOCK_SIZE_N,), dtype=tl.float32), mask=offs_n < chunk_size - start_n - 1)
        ddAcs_ptrs += BLOCK_SIZE_N * stride_ddA_cs_csize_n


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 128}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 32}, num_stages=3, num_warps=4, pre_hook=init_to_zero(["ddA_cumsum_ptr"])),
    ],
    key=['chunk_size', 'dstate', 'hdim'],
)
@triton.jit
def _chunk_scan_bwd_ddAcs_prev_kernel(
    # Pointers to matrices
    dout_ptr, prev_states_ptr, C_ptr, dA_cumsum_ptr, seq_idx_ptr,
    ddA_cumsum_ptr,
    # Matrix dimensions
    chunk_size, dstate, hdim,
    batch, seqlen, nchunks, nheads_ngroups_ratio,
    # Strides
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_prev_states_batch, stride_prev_states_chunk, stride_prev_states_head, stride_prev_states_hdim, stride_prev_states_dstate,
    stride_C_batch, stride_C_seqlen, stride_C_head, stride_C_dstate,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_ddA_cs_batch, stride_ddA_cs_chunk, stride_ddA_cs_head, stride_ddA_cs_csize,
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
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    prev_states_ptr += pid_b * stride_prev_states_batch + pid_c * stride_prev_states_chunk + pid_h * stride_prev_states_head
    C_ptr += pid_b * stride_C_batch + pid_c * chunk_size * stride_C_seqlen + (pid_h // nheads_ngroups_ratio) * stride_C_head
    ddA_cumsum_ptr += pid_b * stride_ddA_cs_batch + pid_c * stride_ddA_cs_chunk + pid_h * stride_ddA_cs_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    dout_ptrs = dout_ptr + (offs_m[:, None] * stride_dout_seqlen + offs_k[None, :] * stride_dout_hdim)
    prev_states_ptrs = prev_states_ptr + (offs_n[None, :] * stride_prev_states_dstate + offs_k[:, None] * stride_prev_states_hdim)
    C_ptrs = C_ptr + (offs_m[:, None] * stride_C_seqlen + offs_n[None, :] * stride_C_dstate)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_m * stride_dA_cs_csize

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)
    dout = tl.load(dout_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_k[None, :] < hdim), other=0.0)
    prev_states = tl.load(prev_states_ptrs, mask=(offs_k[:, None] < hdim) & (offs_n[None, :] < dstate), other=0.0)
    prev_states = prev_states.to(dout_ptrs.dtype.element_ty)
    acc = tl.dot(dout, prev_states)
    c = tl.load(C_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < dstate), other=0.0).to(tl.float32)
    ddA_cs = tl.sum(acc * c, axis=1)
    dA_cs_m = tl.load(dA_cumsum_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    if not HAS_SEQ_IDX:
        scale = tl.exp(dA_cs_m)
    if HAS_SEQ_IDX:
        seq_idx_prev = tl.load(seq_idx_ptr - stride_seq_idx_seqlen, mask=pid_c >= 1, other=0)
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        scale =  tl.where(seq_idx_m == seq_idx_prev, tl.exp(dA_cs_m), 0.0)
    ddA_cs *= scale
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    ddA_cumsum_ptrs = ddA_cumsum_ptr + offs_m * stride_ddA_cs_csize
    tl.atomic_add(ddA_cumsum_ptrs, ddA_cs, mask=offs_m < chunk_size)


def _chunk_scan_fwd(cb, x, dt, dA_cumsum, C, states, D=None, z=None, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # Allocates output.
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
    if z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    grid = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                    batch * nchunks, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3))
                  if z is not None else (0, 0, 0, 0))
    _chunk_scan_fwd_kernel[grid](
        cb, x, z, out, out_x, dt, dA_cumsum, seq_idx, C, states, D,
        chunk_size, headdim, dstate,
        batch, seqlen, nheads // ngroups,
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
        D.stride(0) if D is not None else 0,
        True,
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
        IS_TRITON_22=TRITON_22,
    )
    return out, out_x


def _chunk_scan_fwd_wip(cb, x, dt, dA_cumsum, C, B, states, D=None, z=None, seq_idx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert B.shape == C.shape
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert states.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    # Allocates output.
    out = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
    if z is not None:
        out_x = torch.empty(batch, seqlen, nheads, headdim, device=x.device, dtype=x.dtype)
        assert out_x.stride() == out.stride()
    else:
        out_x = None
    grid = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_N']), batch * nchunks, nheads)
    z_strides = ((z.stride(0), z.stride(1), z.stride(2), z.stride(3))
                  if z is not None else (0, 0, 0, 0))
    _chunk_scan_fwd_kernel_wip[grid](
        cb, x, z, out, out_x, dt, dA_cumsum, seq_idx, C, B, states, D,
        chunk_size, headdim, dstate,
        batch, seqlen, nheads // ngroups,
        cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        z_strides[0], z_strides[1], z_strides[2], z_strides[3],
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
        dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
        *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
        C.stride(0), C.stride(1), C.stride(2), C.stride(3),
        B.stride(0), B.stride(1), B.stride(2), B.stride(3),
        states.stride(0), states.stride(1), states.stride(2), states.stride(3), states.stride(4),
        D.stride(0) if D is not None else 0,
        D is not None,
        D.dim() == 2 if D is not None else True,
        BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
        BLOCK_SIZE_M=128,
        HAS_Z=z is not None,
        HAS_SEQ_IDX=seq_idx is not None,
    )
    return out, out_x


def _chunk_scan_bwd_dz(x, z, out, dout, chunk_size, has_ddAcs=True, D=None, dz=None, recompute_output=False):
    batch, seqlen, nheads, headdim = x.shape
    assert z.shape == x.shape
    assert out.shape == x.shape
    assert dout.shape == out.shape
    nchunks = math.ceil(seqlen / chunk_size)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
    if has_ddAcs:
        ddA_cumsum = torch.empty(batch, nheads, nchunks, chunk_size, device=x.device, dtype=torch.float32)
    if D is not None:
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    if dz is not None:
        assert dz.shape == z.shape
    else:
        dz = torch.empty_like(z)
    if recompute_output:
        outz = torch.empty_like(x)
    dout_x = torch.empty_like(dout)
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    grid_dz = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dz_kernel[grid_dz](
            dout, out, z, x, D, outz if recompute_output else None,
            dz, dout_x, dD, ddA_cumsum if has_ddAcs else None,
            chunk_size, headdim,
            batch, seqlen,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            z.stride(0), z.stride(1), z.stride(2), z.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            D.stride(0) if D is not None else 0,
            *((outz.stride(0), outz.stride(1), outz.stride(2), outz.stride(3)) if recompute_output else (0, 0, 0, 0)),
            dz.stride(0), dz.stride(1), dz.stride(2), dz.stride(3),
            dout_x.stride(0), dout_x.stride(1), dout_x.stride(2), dout_x.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            *((ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3))
              if has_ddAcs else (0, 0, 0, 0)),
            D is not None,
            D.dim() == 2 if D is not None else True,
            has_ddAcs,
            BLOCK_SIZE_N=max(triton.next_power_of_2(headdim), 16),
            RECOMPUTE_OUTPUT=recompute_output,
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_bwd_dz_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return_vals = (dz, dout_x, dD, ddA_cumsum) if has_ddAcs else (dz, dout_x, dD)
    return return_vals if not recompute_output else (*return_vals, outz)


def _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=None, dtype=None):
    batch, seqlen, nheads, headdim = dout.shape
    _, _, nchunks, chunk_size = dA_cumsum.shape
    _, _, ngroups, dstate = C.shape
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    dtype = C.dtype if dtype is None else dtype
    dprev_states = torch.empty(batch, nchunks, nheads, headdim, dstate, device=C.device, dtype=dtype)
    grid_dstates = lambda META: (triton.cdiv(headdim, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                            batch * nchunks, nheads)
    with torch.cuda.device(C.device.index):
        _chunk_scan_bwd_dstates_kernel[grid_dstates](
            dout, C, dprev_states, dA_cumsum, seq_idx,
            headdim, dstate, chunk_size,
            batch, seqlen, nchunks, nheads // ngroups,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            dprev_states.stride(0), dprev_states.stride(1), dprev_states.stride(2), dprev_states.stride(3), dprev_states.stride(4),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            HAS_SEQ_IDX=seq_idx is not None,
        )
    return dprev_states


def _chunk_scan_bwd_dC(prev_states, dA_cumsum, dout, seq_idx=None, C=None, ngroups=1):
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == (batch, seqlen, nheads, headdim)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if C is not None:
        assert C.shape == (batch, seqlen, ngroups, dstate)
        C_strides = (C.stride(0), C.stride(1), C.stride(2), C.stride(3))
        ddA_cumsum_prev = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
        ddA_cumsum_prev_strides = (ddA_cumsum_prev.stride(0), ddA_cumsum_prev.stride(2), ddA_cumsum_prev.stride(1), ddA_cumsum_prev.stride(3))
    else:
        C_strides = (0, 0, 0, 0)
        ddA_cumsum_prev = None
        ddA_cumsum_prev_strides = (0, 0, 0, 0)
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(dout.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dC = torch.empty(batch, seqlen, nsplits, ngroups, dstate, device=dout.device, dtype=torch.float32)
    grid_dc = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                        batch * nchunks, nsplits * ngroups)
    with torch.cuda.device(dout.device.index):
        _chunk_scan_bwd_dc_kernel[grid_dc](
            dout, prev_states, C, dA_cumsum, seq_idx, dC, ddA_cumsum_prev,
            chunk_size, dstate, headdim,
            batch, seqlen, nheads, nheads_per_program, ngroups,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3), prev_states.stride(4),
            *C_strides,
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dC.stride(0), dC.stride(1), dC.stride(2), dC.stride(3), dC.stride(4),
            *ddA_cumsum_prev_strides,
            HAS_DDA_CS=ddA_cumsum_prev is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    dC = dC.sum(2)
    return dC if C is None else (dC, ddA_cumsum_prev)


def _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=None, CB=None, ngroups=1):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if CB is not None:
        assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
        CB_strides = (CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(3), CB.stride(4))
        BLOCK_SIZE_M_min = 16
        ddA_cumsum = torch.empty(batch, nheads, nchunks, triton.cdiv(chunk_size, BLOCK_SIZE_M_min),
                                chunk_size, device=x.device, dtype=torch.float32)
        ddA_cumsum_strides = (ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4))
    else:
        CB_strides = (0, 0, 0, 0, 0)
        ddA_cumsum = None
        ddA_cumsum_strides = (0, 0, 0, 0, 0)
    nheads_ngroups_ratio = nheads // ngroups
    sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
    nheads_per_program = max(min(math.ceil(batch * nchunks * nheads / sm_count), nheads_ngroups_ratio), 1)
    nsplits = triton.cdiv(nheads_ngroups_ratio, nheads_per_program)
    dcb = torch.empty(batch, nchunks, nsplits, ngroups, chunk_size, chunk_size, device=x.device, dtype=torch.float32)
    grid_dcb = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(chunk_size, META['BLOCK_SIZE_N']),
                        batch * nchunks, nsplits * ngroups)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dcb_kernel[grid_dcb](
            x, dout, CB, dt, dA_cumsum, seq_idx, dcb, ddA_cumsum,
            chunk_size, headdim,
            batch, seqlen, nheads, nheads_per_program, ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            *CB_strides,
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            dcb.stride(0), dcb.stride(1), dcb.stride(2), dcb.stride(3), dcb.stride(4), dcb.stride(5),
            *ddA_cumsum_strides,
            HAS_DDA_CS=ddA_cumsum is not None,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    dcb = dcb.sum(2)
    if ddA_cumsum is not None:
        BLOCK_SIZE_M_actual = _chunk_scan_bwd_dcb_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1) // BLOCK_SIZE_M_actual
        ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return dcb if CB is None else (dcb, ddA_cumsum)


def _chunk_scan_bwd_dx(cb, x, dt, dA_cumsum, dout, D=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    ngroups = cb.shape[2]
    assert nheads % ngroups == 0
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    # if D is not None:
    #     BLOCK_SIZE_M_min = 32
    #     dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_M_min), batch, nchunks, nheads, headdim, device=D.device, dtype=torch.float32)
    # else:
    #     dD = None
    dx = torch.empty_like(x)
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                        batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_dx_kernel[grid_dx](
            x, cb, dout, dt, dA_cumsum, D, dx, ddt, # dD,
            chunk_size, headdim,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(-1), cb.stride(-2),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            D.stride(0) if D is not None else 0,
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            # dD.stride(1) if dD is not None else 0, dD.stride(2) if dD is not None else 0, dD.stride(3) if dD is not None else 0, dD.stride(4) if dD is not None else 0, dD.stride(0) if dD is not None else 0,
            D is not None,
            D.dim() == 2 if D is not None else True,
        )
    # if D is not None:
    #     BLOCK_SIZE_actual = _chunk_scan_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
    #     n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
    #     dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
    return dx, ddt.to(dtype=dt.dtype)


def _chunk_scan_bwd_ddAcs_unstable(x, dt, out, dout, ddt, D=None, subtract_ddtdt=True):
    """Not numerically stable and should not be used. Leaving here for reference.
    """

    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert ddt.shape == dt.shape
    assert out.shape == x.shape
    assert dout.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    ddA_cumsum = torch.empty_like(dt)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    if D is not None:  # Triton gives wrong results if we write to the same location
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_ddAcs_unstable_kernel[grid_ddtcs](
            dout, out, dt, ddt, x, D, ddA_cumsum, dD,
            chunk_size, headdim,
            batch, seqlen,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            D.stride(0) if D is not None else 0,
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            subtract_ddtdt,
            BLOCK_SIZE_N=max(triton.next_power_of_2(headdim), 16),
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_bwd_ddAcs_unstable_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return ddA_cumsum, dD


def _chunk_scan_bwd_ddAcs_stable_old(x, dt, dA_cumsum, dout, cb):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == x.shape
    assert dA_cumsum.shape == dt.shape
    ngroups = cb.shape[2]
    assert nheads % ngroups == 0
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    BLOCK_SIZE_M_min = 16
    ddA_cumsum = torch.empty(batch, nheads, nchunks, triton.cdiv(chunk_size, BLOCK_SIZE_M_min),
                             chunk_size, device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_ddAcs_stable_kernel_old[grid_ddtcs](
            x, dout, dt, dA_cumsum, cb, ddA_cumsum,
            chunk_size, headdim,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4),
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
            BLOCK_SIZE_N=max(triton.next_power_of_2(chunk_size), 16),
        )
    BLOCK_SIZE_M_actual = _chunk_scan_bwd_ddAcs_stable_kernel_old.best_config.kwargs["BLOCK_SIZE_M"]
    n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1) // BLOCK_SIZE_M_actual
    ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return ddA_cumsum


def _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, cb):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == x.shape
    assert dA_cumsum.shape == dt.shape
    ngroups = cb.shape[2]
    assert nheads % ngroups == 0
    assert cb.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    BLOCK_SIZE_M_min = 32
    ddA_cumsum = torch.empty(batch, nheads, nchunks, triton.cdiv(chunk_size, BLOCK_SIZE_M_min),
                             chunk_size, device=x.device, dtype=torch.float32)
    grid_ddtcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']), batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_bwd_ddAcs_stable_kernel[grid_ddtcs](
            x, dout, dt, dA_cumsum, cb, ddA_cumsum,
            chunk_size, headdim,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            cb.stride(0), cb.stride(1), cb.stride(2), cb.stride(3), cb.stride(4),
            ddA_cumsum.stride(0), ddA_cumsum.stride(2), ddA_cumsum.stride(1), ddA_cumsum.stride(3), ddA_cumsum.stride(4),
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    BLOCK_SIZE_M_actual = _chunk_scan_bwd_ddAcs_stable_kernel.best_config.kwargs["BLOCK_SIZE_M"]
    n_valid_blocks = (chunk_size + BLOCK_SIZE_M_actual - 1) // BLOCK_SIZE_M_actual
    ddA_cumsum = ddA_cumsum[:, :, :, :n_valid_blocks].sum(dim=3)
    return ddA_cumsum


def _chunk_scan_bwd_ddAcs_prev(prev_states, C, dout, dA_cumsum, seq_idx=None):
    batch, nchunks, nheads, headdim, dstate = prev_states.shape
    _, seqlen, _, _ = dout.shape
    _, _, _, chunk_size = dA_cumsum.shape
    assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
    assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
    assert dout.shape == (batch, seqlen, nheads, headdim)
    ngroups = C.shape[2]
    assert nheads % ngroups == 0
    assert C.shape == (batch, seqlen, ngroups, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    ddA_cumsum_prev = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    grid_ddAcs = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(dstate, META['BLOCK_SIZE_N']),
                          batch * nchunks, nheads)
    with torch.cuda.device(dout.device.index):
        _chunk_scan_bwd_ddAcs_prev_kernel[grid_ddAcs](
            dout, prev_states, C, dA_cumsum, seq_idx, ddA_cumsum_prev,
            chunk_size, dstate, headdim,
            batch, seqlen, nchunks, nheads // ngroups,
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            prev_states.stride(0), prev_states.stride(1), prev_states.stride(2), prev_states.stride(3), prev_states.stride(4),
            C.stride(0), C.stride(1), C.stride(2), C.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            ddA_cumsum_prev.stride(0), ddA_cumsum_prev.stride(2), ddA_cumsum_prev.stride(1), ddA_cumsum_prev.stride(3),
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_K=max(triton.next_power_of_2(headdim), 16),
        )
    return ddA_cumsum_prev


class ChunkScanFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
        # Check constraints.
        batch, seqlen, nheads, headdim = x.shape
        _, _, ngroups, dstate = B.shape
        assert B.shape == (batch, seqlen, ngroups, dstate)
        _, _, nchunks, chunk_size = dt.shape
        assert seqlen == nchunks * chunk_size
        assert C.shape == B.shape
        if z is not None:
            assert z.shape == x.shape
        if D is not None:
            assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert dt.shape == (batch, nheads, nchunks, chunk_size)
        assert dA_cumsum.shape == (batch, nheads, nchunks, chunk_size)
        assert prev_states.shape == (batch, nchunks, nheads, headdim, dstate)
        if B.stride(-1) != 1:
            B = B.contiguous()
        if C.stride(-1) != 1:
            C = C.contiguous()
        if x.stride(-1) != 1 and x.stride(1) != 1:  # Either M or K dimension should be contiguous
            x = x.contiguous()
        if z is not None and z.stride(-1) != 1 and z.stride(1) != 1:  # Either M or K dimension should be contiguous
            z = z.contiguous()
        if D is not None and D.stride(-1) != 1:
            D = D.contiguous()
        CB = _bmm_chunk_fwd(C, B, chunk_size)
        out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, prev_states, D=D, z=z)
        ctx.save_for_backward(out if z is None else out_x, B, C, CB, x, dt, dA_cumsum, prev_states, D, z)
        return out

    @staticmethod
    def backward(ctx, dout):
        if dout.stride(-1) != 1:
            dout = dout.contiguous()
        out, B, C, CB, x, dt, dA_cumsum, prev_states, D, z = ctx.saved_tensors
        batch, seqlen, nheads, headdim = x.shape
        _, _, nchunks, chunk_size = dt.shape
        _, _, ngroups, dstate = B.shape
        assert dout.shape == (batch, seqlen, nheads, headdim)
        if z is not None:
            dz, dout, dD, ddA_cumsum = _chunk_scan_bwd_dz(x, z, out, dout, chunk_size=chunk_size, D=D)
        else:
            dz = None
        dprev_states = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, dtype=prev_states.dtype)
        dC = _chunk_scan_bwd_dC(prev_states, dA_cumsum, dout, ngroups=ngroups)
        dC = dC.to(C.dtype)
        dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, ngroups=ngroups)
        dCB = dCB.to(CB.dtype)
        dB = _bmm_chunk_bwd(C, dCB)
        dC = _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC)
        dx, ddt = _chunk_scan_bwd_dx(CB, x, dt, dA_cumsum, dout, D=D)
        # Formula for ddA_cumsum, assuming out is the output of the forward pass before adding x * D.
        # ddA_cumsum = torch.einsum("bclhp,bclhp->bhcl", out.float(), dout.float()) - ddt * dt
        if z is not None:
            ddA_cumsum -= ddt * dt
        else: # If z is not None, we already calculated ddA_cumsum and dD when computing dz
            ddA_cumsum, dD = _chunk_scan_bwd_ddAcs_unstable(x, dt, out, dout, ddt, D=D)
        ddA_cumsum = ddA_cumsum.to(dA_cumsum.dtype)
        return dB, dC, dx, ddt, ddA_cumsum, dprev_states, dD, dz


def chunk_scan(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """
    prev_states contains the initial_states at index 0, and the state for the next-to-last chunk at index -1.
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    return ChunkScanFn.apply(B, C, x, dt, dA_cumsum, prev_states, D, z)


def chunk_scan_ref(B, C, x, dt, dA_cumsum, prev_states, D=None, z=None):
    """
    Argument:
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, nheads, nchunks, chunk_size)
        dA_cumsum: (batch, nheads, nchunks, chunk_size)
        prev_states: (batch, nchunks, nheads, headdim, dstate)
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert B.shape == (batch, seqlen, ngroups, dstate)
    _, _, nchunks, chunk_size = dt.shape
    assert seqlen == nchunks * chunk_size
    assert C.shape == B.shape
    B = repeat(B, "b l g d -> b l (g h) d", h=nheads // ngroups)
    C = repeat(C, "b l g d -> b l (g h) d", h=nheads // ngroups)
    CB = torch.einsum("bclhn,bcshn->bchls", rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                      rearrange(B, "b (c s) h n -> b c s h n", c=nchunks))
    # (batch, nheads, nchunks, chunksize, chunksize)
    dt_segment_sum = dA_cumsum[:, :, :, :, None] - dA_cumsum[:, :, :, None, :]
    decay = torch.exp(dt_segment_sum)
    scores_decay = CB * rearrange(decay, "b h c l s -> b c h l s")
    causal_mask = torch.tril(torch.ones(chunk_size, chunk_size, device=x.device, dtype=bool), diagonal=0)
    scores_decay = scores_decay.masked_fill(~causal_mask, 0)
    out = torch.einsum('bchls,bhcs,bcshp->bclhp', scores_decay.to(x.dtype), dt.to(x.dtype),
                       rearrange(x, "b (c s) h p -> b c s h p", c=nchunks))
    state_decay_out = torch.exp(rearrange(dA_cumsum, "b h c l -> b c l h 1"))
    out_prev = torch.einsum('bclhn,bchpn->bclhp', rearrange(C, "b (c l) h n -> b c l h n", c=nchunks),
                            prev_states.to(C.dtype)) * state_decay_out
    out = out + out_prev
    out = rearrange(out, "b c l h p -> b (c l) h p")
    if D is not None:
        if D.dim() == 1:
            D = rearrange(D, "h -> h 1")
        out = out + x * D
    return out if z is None else out * F.silu(z)
