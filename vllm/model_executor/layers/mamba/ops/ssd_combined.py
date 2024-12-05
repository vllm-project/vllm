# Copyright (c) 2024, Tri Dao, Albert Gu.

"""We want triton==2.1.0 or 2.2.0 for this
"""

import math
from packaging import version

import torch

import triton
import triton.language as tl

from einops import rearrange

from .ssd_bmm import _bmm_chunk_fwd, _bmm_chunk_bwd
from .ssd_chunk_state import _chunk_cumsum_fwd, _chunk_cumsum_bwd
from .ssd_chunk_state import _chunk_state_fwd, _chunk_state_bwd_db
from .ssd_chunk_state import chunk_state_varlen
from .ssd_state_passing import _state_passing_fwd, _state_passing_bwd
from .ssd_chunk_scan import _chunk_scan_fwd, _chunk_scan_bwd_dz, _chunk_scan_bwd_dstates
from .ssd_chunk_scan import _chunk_scan_bwd_dC, _chunk_scan_bwd_dcb
from .ssd_chunk_scan import _chunk_scan_bwd_ddAcs_stable

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

def init_to_zero(names):
    return lambda nargs: [nargs[name].zero_() for name in names if nargs[name] is not None]

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
    key=['chunk_size', 'hdim', 'dstate'],
)
@triton.jit
def _chunk_scan_chunk_state_bwd_dx_kernel(
    # Pointers to matrices
    x_ptr, cb_ptr, dout_ptr, dt_ptr, dA_cumsum_ptr, seq_idx_ptr, D_ptr,
    b_ptr, dstates_ptr,
    dx_ptr, ddt_ptr, dD_ptr,
    # Matrix dimensions
    chunk_size, hdim, dstate,
    batch, seqlen, nheads_ngroups_ratio,
    # Strides
    stride_x_batch, stride_x_seqlen, stride_x_head, stride_x_hdim,
    stride_cb_batch, stride_cb_chunk, stride_cb_head, stride_cb_csize_m, stride_cb_csize_k,
    stride_dout_batch, stride_dout_seqlen, stride_dout_head, stride_dout_hdim,
    stride_dt_batch, stride_dt_chunk, stride_dt_head, stride_dt_csize,
    stride_dA_cs_batch, stride_dA_cs_chunk, stride_dA_cs_head, stride_dA_cs_csize,
    stride_seq_idx_batch, stride_seq_idx_seqlen,
    stride_D_head,
    stride_b_batch, stride_b_seqlen, stride_b_head, stride_b_dstate,
    stride_dstates_batch, stride_dstates_chunk, stride_dstates_head, stride_dstates_hdim, stride_dstates_dstate,
    stride_dx_batch, stride_dx_seqlen, stride_dx_head, stride_dx_hdim,
    stride_ddt_batch, stride_ddt_chunk, stride_ddt_head, stride_ddt_csize,
    stride_dD_batch, stride_dD_chunk, stride_dD_head, stride_dD_csize, stride_dD_hdim,
    # Meta-parameters
    HAS_D: tl.constexpr,
    D_HAS_HDIM: tl.constexpr,
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
    x_ptr += pid_b * stride_x_batch + pid_c * chunk_size * stride_x_seqlen + pid_h * stride_x_head
    cb_ptr += pid_b * stride_cb_batch + pid_c * stride_cb_chunk + (pid_h // nheads_ngroups_ratio) * stride_cb_head
    dout_ptr += pid_b * stride_dout_batch + pid_c * chunk_size * stride_dout_seqlen + pid_h * stride_dout_head
    dt_ptr += pid_b * stride_dt_batch + pid_c * stride_dt_chunk + pid_h * stride_dt_head
    ddt_ptr += pid_b * stride_ddt_batch + pid_c * stride_ddt_chunk + pid_h * stride_ddt_head
    dA_cumsum_ptr += pid_b * stride_dA_cs_batch + pid_c * stride_dA_cs_chunk + pid_h * stride_dA_cs_head
    b_ptr += pid_b * stride_b_batch + pid_c * chunk_size * stride_b_seqlen + (pid_h // nheads_ngroups_ratio) * stride_b_head
    dstates_ptr += pid_b * stride_dstates_batch + pid_c * stride_dstates_chunk + pid_h * stride_dstates_head
    if HAS_SEQ_IDX:
        seq_idx_ptr += pid_b * stride_seq_idx_batch + pid_c * chunk_size * stride_seq_idx_seqlen

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    chunk_size_limit = min(chunk_size, seqlen - pid_c * chunk_size)

    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    dA_cs_m = tl.load(dA_cumsum_ptr + offs_m * stride_dA_cs_csize, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)

    dA_cs_last = tl.load(dA_cumsum_ptr + (chunk_size - 1) * stride_dA_cs_csize).to(tl.float32)
    if not HAS_SEQ_IDX:
        scale = tl.exp(dA_cs_last - dA_cs_m)
    else:
        seq_idx_m = tl.load(seq_idx_ptr + offs_m * stride_seq_idx_seqlen, mask=offs_m < chunk_size_limit, other=-1)
        seq_idx_last = tl.load(seq_idx_ptr + (chunk_size_limit - 1) * stride_seq_idx_seqlen)
        scale = tl.where(seq_idx_m == seq_idx_last, tl.exp(dA_cs_last - dA_cs_m), 0.0)
    # Might be faster to just do 1 iteration with larger BLOCK_SIZE_K, up to block size 128
    # However, we're getting error with the Triton compiler 2.1.0 for that code path:
    # Unexpected mma -> mma layout conversion
    # Triton 2.2.0 fixes this
    offs_dstate = tl.arange(0, BLOCK_SIZE_DSTATE if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128 else BLOCK_SIZE_K)
    b_ptrs = b_ptr + (offs_m[:, None] * stride_b_seqlen + offs_dstate[None, :] * stride_b_dstate)
    dstates_ptrs = dstates_ptr + (offs_n[None, :] * stride_dstates_hdim + offs_dstate[:, None] * stride_dstates_dstate)
    if IS_TRITON_22 and BLOCK_SIZE_DSTATE <= 128:
        b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate), other=0.0)
        dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate) & (offs_n[None, :] < hdim), other=0.0)
        dstates = dstates.to(b_ptr.dtype.element_ty)
        acc = tl.dot(b, dstates) * scale[:, None]
    else:
        for k in range(0, dstate, BLOCK_SIZE_K):
            b = tl.load(b_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_dstate[None, :] < dstate - k), other=0.0)
            dstates = tl.load(dstates_ptrs, mask=(offs_dstate[:, None] < dstate - k) & (offs_n[None, :] < hdim), other=0.0)
            dstates = dstates.to(b_ptr.dtype.element_ty)
            acc += tl.dot(b, dstates)
            b_ptrs += BLOCK_SIZE_K * stride_b_dstate
            dstates_ptrs += BLOCK_SIZE_K * stride_dstates_dstate
        acc *= scale[:, None]

    # x_ptrs = x_ptr + (offs_m[:, None] * stride_x_seqlen + offs_n[None, :] * stride_x_hdim)
    # x = tl.load(x_ptrs, mask=(offs_m[:, None] < chunk_size_limit) & (offs_n[None, :] < hdim), other=0.0).to(tl.float32)
    # dt_ptrs = dt_ptr + offs_m * stride_dt_csize
    # dt_m = tl.load(dt_ptrs, mask=offs_m < chunk_size_limit, other=0.0).to(tl.float32)
    # ddt = tl.sum(acc * x, axis=1) * dt_m
    # ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    # tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)

    offs_k = tl.arange(0, BLOCK_SIZE_K)
    cb_ptrs = cb_ptr + (offs_m[:, None] * stride_cb_csize_m + offs_k[None, :] * stride_cb_csize_k)
    dout_ptrs = dout_ptr + (offs_k[:, None] * stride_dout_seqlen + offs_n[None, :] * stride_dout_hdim)
    dA_cumsum_ptrs = dA_cumsum_ptr + offs_k * stride_dA_cs_csize
    K_MAX = chunk_size_limit
    K_MIN = pid_m * BLOCK_SIZE_M
    cb_ptrs += K_MIN * stride_cb_csize_k
    dout_ptrs += K_MIN * stride_dout_seqlen
    dA_cumsum_ptrs += K_MIN * stride_dA_cs_csize
    for k in range(K_MIN, K_MAX, BLOCK_SIZE_K):
        k = tl.multiple_of(k, BLOCK_SIZE_K)
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
    if HAS_D:
        dD_ptr += pid_b * stride_dD_batch + pid_c * stride_dD_chunk + pid_h * stride_dD_head + pid_m * stride_dD_csize
        if D_HAS_HDIM:
            dD_ptrs = dD_ptr + offs_n * stride_dD_hdim
            dD = tl.sum(dout_res * x, axis=0)
            tl.store(dD_ptrs, dD, mask=offs_n < hdim)
        else:
            dD = tl.sum(dout_res * x)
            tl.store(dD_ptr, dD)
    ddt = tl.sum(acc * x, axis=1)
    ddt_ptrs = ddt_ptr + offs_m * stride_ddt_csize
    tl.atomic_add(ddt_ptrs, ddt, mask=offs_m < chunk_size)


def _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=None, seq_idx=None, dx=None):
    batch, seqlen, nheads, headdim = x.shape
    _, _, nchunks, chunk_size = dt.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert CB.shape == (batch, nchunks, ngroups, chunk_size, chunk_size)
    assert dt.shape == (batch, nheads, nchunks, chunk_size)
    assert dA_cumsum.shape == dt.shape
    assert dout.shape == x.shape
    assert dstates.shape == (batch, nchunks, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
        assert D.stride(-1) == 1
        BLOCK_SIZE_min = 32
        dD = torch.empty(triton.cdiv(chunk_size, BLOCK_SIZE_min), batch, nchunks, nheads,
                         headdim if D.dim() == 2 else 1, device=D.device, dtype=torch.float32)
    else:
        dD = None
    dD_strides = ((dD.stride(0), dD.stride(1), dD.stride(2), dD.stride(3), dD.stride(4))
                    if D is not None else (0, 0, 0, 0, 0))
    if dx is None:
        dx = torch.empty_like(x)
    else:
        assert dx.shape == x.shape
    ddt = torch.empty(batch, nheads, nchunks, chunk_size, device=dout.device, dtype=torch.float32)
    grid_dx = lambda META: (triton.cdiv(chunk_size, META['BLOCK_SIZE_M']) * triton.cdiv(headdim, META['BLOCK_SIZE_N']),
                        batch * nchunks, nheads)
    with torch.cuda.device(x.device.index):
        _chunk_scan_chunk_state_bwd_dx_kernel[grid_dx](
            x, CB, dout, dt, dA_cumsum, seq_idx, D, B, dstates, dx, ddt, dD,
            chunk_size, headdim, dstate,
            batch, seqlen, nheads // ngroups,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            CB.stride(0), CB.stride(1), CB.stride(2), CB.stride(-1), CB.stride(-2),
            dout.stride(0), dout.stride(1), dout.stride(2), dout.stride(3),
            dt.stride(0), dt.stride(2), dt.stride(1), dt.stride(3),
            dA_cumsum.stride(0), dA_cumsum.stride(2), dA_cumsum.stride(1), dA_cumsum.stride(3),
            *((seq_idx.stride(0), seq_idx.stride(1)) if seq_idx is not None else (0, 0)),
            D.stride(0) if D is not None else 0,
            B.stride(0), B.stride(1), B.stride(2), B.stride(3),
            dstates.stride(0), dstates.stride(1), dstates.stride(2), dstates.stride(3), dstates.stride(4),
            dx.stride(0), dx.stride(1), dx.stride(2), dx.stride(3),
            ddt.stride(0), ddt.stride(2), ddt.stride(1), ddt.stride(3),
            dD_strides[1], dD_strides[2], dD_strides[3], dD_strides[0], dD_strides[4],
            D is not None,
            D.dim() == 2 if D is not None else True,
            HAS_SEQ_IDX=seq_idx is not None,
            BLOCK_SIZE_DSTATE=max(triton.next_power_of_2(dstate), 16),
            IS_TRITON_22=TRITON_22
        )
    if D is not None:
        BLOCK_SIZE_actual = _chunk_scan_chunk_state_bwd_dx_kernel.best_config.kwargs["BLOCK_SIZE_M"]
        n_valid_blocks = (chunk_size + BLOCK_SIZE_actual - 1) // BLOCK_SIZE_actual
        dD = dD[:n_valid_blocks].sum(dim=(0, 1, 2)).to(dtype=D.dtype)
        if D.dim() == 1:
            dD = rearrange(dD, "h 1 -> h")
    return dx, ddt.to(dtype=dt.dtype), dD

def _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf"))):
    batch, seqlen, nheads, headdim = x.shape
    _, _, ngroups, dstate = B.shape
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert x.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert C.shape == B.shape
    if z is not None:
        assert z.shape == x.shape
    if D is not None:
        assert D.shape == (nheads, headdim) or D.shape == (nheads,)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
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
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    # # (batch, nchunks, chunk_size, chunk_size) or (batch, nchunks, nheads, chunk_size, chunk_size)
    # dA_cumsum_tmp0, dt_tmp0 = _chunk_cumsum_fwd(dt[:, :147], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp1, dt_tmp1 = _chunk_cumsum_fwd(dt[:, 147:], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    # dA_cumsum_tmp2, dt_tmp2 = _chunk_cumsum_fwd(dt[:, 147:256], A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus)
    dA_cumsum, dt = _chunk_cumsum_fwd(dt, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    # states_tmp0 = _chunk_state_fwd(B[:, :147], x[:, :147], dt_tmp0, dA_cumsum_tmp0, states_in_fp32=True)
    # states_tmp1 = _chunk_state_fwd(B[:, 147:], x[:, 147:], dt_tmp1, dA_cumsum_tmp1, states_in_fp32=True)
    # states_tmp2 = _chunk_state_fwd(B[:, 147:256], x[:, 147:256], dt_tmp2, dA_cumsum_tmp2, states_in_fp32=True)
    states, final_states = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                              initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                              seq_idx=seq_idx, chunk_size=chunk_size, out_dtype=C.dtype)
    states, final_states = [rearrange(t, "... (p n) -> ... p n", n=dstate) for t in [states, final_states]]
    # states_tmp0 = rearrange(_state_passing_fwd(rearrange(states_tmp0, "... p n -> ... (p n)"), dA_cumsum_tmp0[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    # states_tmp1 = rearrange(_state_passing_fwd(rearrange(states_tmp1, "... p n -> ... (p n)"), dA_cumsum_tmp1[:, :, :, -1], chunk_size=chunk_size), "... (p n) -> ... p n", n=dstate)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    out, out_x = _chunk_scan_fwd(CB, x, dt, dA_cumsum, C, states, D=D, z=z, seq_idx=seq_idx)
    if cu_seqlens is None:
        return out, out_x, dt, dA_cumsum, states, final_states
    else:
        assert batch == 1, "passing cu_seqlens to get the varlen states is only supported if batch dimension is 1"
        varlen_states = chunk_state_varlen(B.squeeze(0), x.squeeze(0), dt.squeeze(0), dA_cumsum.squeeze(0),
                                           cu_seqlens, states.squeeze(0))
        return out, out_x, dt, dA_cumsum, states, final_states, varlen_states


def _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, chunk_size, D=None, z=None,
                                   dt_bias=None, initial_states=None, dfinal_states=None, seq_idx=None, dt_softplus=False,
                                   dt_limit=(0.0, float("inf")),
                                   dx=None, ddt=None, dB=None, dC=None, dz=None, recompute_output=False):
    if dout.stride(-1) != 1:
        dout = dout.contiguous()
    batch, seqlen, nheads, headdim = x.shape
    nchunks = math.ceil(seqlen / chunk_size)
    _, _, ngroups, dstate = B.shape
    assert dout.shape == (batch, seqlen, nheads, headdim)
    assert dt.shape == (batch, seqlen, nheads)
    assert A.shape == (nheads,)
    assert nheads % ngroups == 0
    assert B.shape == (batch, seqlen, ngroups, dstate)
    assert C.shape == B.shape
    assert out.shape == x.shape
    if initial_states is not None:
        assert initial_states.shape == (batch, nheads, headdim, dstate)
    if seq_idx is not None:
        assert seq_idx.shape == (batch, seqlen)
    if dx is not None:
        assert dx.shape == x.shape
    if dB is not None:
        assert dB.shape == B.shape
        dB_given = dB
    else:
        dB_given = torch.empty_like(B)
    if dC is not None:
        assert dC.shape == C.shape
        dC_given = dC
    else:
        dC_given = torch.empty_like(C)
    if dz is not None:
        assert z is not None
        assert dz.shape == z.shape
    if ddt is not None:
        assert ddt.shape == dt.shape
        ddt_given = ddt
    else:
        ddt_given = torch.empty_like(dt)
    # TD: For some reason Triton (2.1.0 and 2.2.0) errors with
    # "[CUDA]: invalid device context" (e.g. during varlne test), and cloning makes it work. Idk why.
    dt_in = dt.clone()
    dA_cumsum, dt = _chunk_cumsum_fwd(dt_in, A, chunk_size, dt_bias=dt_bias, dt_softplus=dt_softplus,
                                      dt_limit=dt_limit)
    CB = _bmm_chunk_fwd(C, B, chunk_size, seq_idx=seq_idx, output_dtype=torch.float32)
    states = _chunk_state_fwd(B, x, dt, dA_cumsum, seq_idx=seq_idx, states_in_fp32=True)
    states, _ = _state_passing_fwd(rearrange(states, "... p n -> ... (p n)"), dA_cumsum[:, :, :, -1],
                                   initial_states=rearrange(initial_states, "... p n -> ... (p n)") if initial_states is not None else None,
                                   seq_idx=seq_idx, chunk_size=chunk_size)
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    if z is not None:
        dz, dout, dD, *rest = _chunk_scan_bwd_dz(x, z, out, dout, chunk_size=chunk_size, has_ddAcs=False, D=D, dz=dz, recompute_output=recompute_output)
        outz = rest[0] if recompute_output else out
    else:
        dz = None
        outz = out
    dstates = _chunk_scan_bwd_dstates(C, dA_cumsum, dout, seq_idx=seq_idx, dtype=states.dtype)
    # dstates has length nchunks, containing the gradient to initial states at index 0 and
    # gradient to the states of chunk (nchunks - 2) at index (nchunks - 1)
    # Do computation in fp32 but convert dstates and states to fp16/bf16 since dstates and states
    # will be used in matmul in the next kernels.
    dstates, ddA_chunk_cumsum, dinitial_states, states = _state_passing_bwd(
        rearrange(states, "... p n -> ... (p n)"),
        dA_cumsum[:, :, :, -1],
        rearrange(dstates, "... p n -> ... (p n)"),
        dfinal_states=rearrange(dfinal_states, "... p n -> ... (p n)") if dfinal_states is not None else None,
        seq_idx=seq_idx,
        has_initial_states=initial_states is not None,
        dstates_dtype=x.dtype,
        states_dtype=x.dtype,
        chunk_size=chunk_size,
    )
    # dstates has length nchunks, containing the gradient to states of chunk 0 at index 0 and
    # gradient to the final states at index (nchunks - 1)
    # states has length nchunks, containing the initial states at index 0 and the state for chunk (nchunks - 2) at index (nchunks - 1)
    # The final states is not stored.
    states = rearrange(states, "... (p n) -> ... p n", n=dstate)
    dstates = rearrange(dstates, "... (p n) -> ... p n", n=dstate)
    dinitial_states = rearrange(dinitial_states, "... (p n) -> ... p n", n=dstate) if dinitial_states is not None else None
    dx, ddt, dD_from_x = _chunk_scan_chunk_state_bwd_dx(x, dt, dA_cumsum, B, CB, dout, dstates, D=D, seq_idx=seq_idx, dx=dx)
    # dB = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, ngroups=ngroups)
    dB, ddA_next = _chunk_state_bwd_db(x, dt, dA_cumsum, dstates, seq_idx=seq_idx, B=B, ngroups=ngroups)
    # dC = _chunk_scan_bwd_dC(states[:, :-1].to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    dC, ddA_cumsum_prev = _chunk_scan_bwd_dC(states.to(x.dtype), dA_cumsum, dout, seq_idx=seq_idx, C=C, ngroups=ngroups)
    # Computing ddA with the dcb kernel is much slower, so we're not using it for now
    dCB = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, ngroups=ngroups)
    # dCB, ddA_tmp = _chunk_scan_bwd_dcb(x, dt, dA_cumsum, dout, seq_idx=seq_idx, CB=CB, ngroups=ngroups)
    dCB = dCB.to(CB.dtype)
    _bmm_chunk_bwd(C, dCB, residual=dB, out=dB_given)
    _bmm_chunk_bwd(B, rearrange(dCB, "... l s -> ... s l"), residual=dC, out=dC_given)
    # If we have z, then dout_x is recomputed in fp32 so dD = (dout_x * x).sum() is more accurate
    # than dD_from_x = (dout_x * x).sum() where dout_x is in fp16/bf16
    if z is None:
        dD = dD_from_x
    # Formula for ddA_cumsum, assuming out is the output of the forward pass before adding x * D.
    # ddA_cumsum = torch.einsum("bclhp,bclhp->bhcl", out.float(), dout.float()) - ddt * dt
    # However, this is numerically unstable: when we do the reverse cumsum on ddA_cumsum, there might
    # be a lot of underflow.

    # This is already done as part of bwd_dC kernel
    # ddA_cumsum_prev = _chunk_scan_bwd_ddAcs_prev(states[:, :-1], C, dout, dA_cumsum, seq_idx=seq_idx)
    ddA_cumsum_prev[..., -1] += ddA_chunk_cumsum
    ddA_prev = ddA_cumsum_prev.flip([-1]).cumsum(dim=-1).flip([-1])
    # This is already done as part of bwd_dB kernel
    # ddA_next = _chunk_state_bwd_ddAcs_stable(B, x, dt, dA_cumsum, dstates, seq_idx=seq_idx)
    # We don't need to pass in seq_idx because CB also zeros out entries where seq_idx[i] != seq_idx[j]
    ddA = _chunk_scan_bwd_ddAcs_stable(x, dt, dA_cumsum, dout, CB)
    ddA += ddA_next + ddA_prev

    ddt_given, dA, ddt_bias = _chunk_cumsum_bwd(ddA, ddt, dt_in, A, dt_bias=dt_bias, dt_softplus=dt_softplus, dt_limit=dt_limit, ddt=ddt_given)

    # These 2 lines are just to test ddt and dA being computed by old code
    # _, dA = selective_scan_bwd(dout, x, dt, A, B, C, D=D.float(), z=z)
    # ddt_given.copy_(ddt)

    return_vals = (dx, ddt_given, dA, dB_given, dC_given, dD, dz, ddt_bias, dinitial_states)
    return return_vals if not recompute_output else (*return_vals, outz)

class MambaChunkScanCombinedFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, return_varlen_states=False):
        ctx.dt_dtype = dt.dtype
        if not return_varlen_states:
            cu_seqlens = None
        else:
            assert cu_seqlens is not None, "cu_seqlens must be provided if return_varlen_states is True"
        out, out_x, dt_out, dA_cumsum, states, final_states, *rest = _mamba_chunk_scan_combined_fwd(x, dt, A, B, C, chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, seq_idx=seq_idx, cu_seqlens=cu_seqlens, dt_softplus=dt_softplus, dt_limit=dt_limit)
        ctx.save_for_backward(out if z is None else out_x, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx)
        ctx.dt_softplus = dt_softplus
        ctx.chunk_size = chunk_size
        ctx.dt_limit = dt_limit
        ctx.return_final_states = return_final_states
        ctx.return_varlen_states = return_varlen_states
        if not return_varlen_states:
            return out if not return_final_states else (out, final_states)
        else:
            varlen_states = rest[0]
            return (out, varlen_states) if not return_final_states else (out, final_states, varlen_states)

    @staticmethod
    def backward(ctx, dout, *args):
        out, x, dt, dA_cumsum, A, B, C, D, z, dt_bias, initial_states, seq_idx = ctx.saved_tensors
        assert not ctx.return_varlen_states, "return_varlen_states is not supported in backward"
        dfinal_states = args[0] if ctx.return_final_states else None
        dx, ddt, dA, dB, dC, dD, dz, ddt_bias, dinitial_states = _mamba_chunk_scan_combined_bwd(dout, x, dt, A, B, C, out, ctx.chunk_size, D=D, z=z, dt_bias=dt_bias, initial_states=initial_states, dfinal_states=dfinal_states, seq_idx=seq_idx, dt_softplus=ctx.dt_softplus, dt_limit=ctx.dt_limit)
        return dx, ddt, dA, dB, dC, None, dD, dz, ddt_bias, dinitial_states, None, None, None, None, None, None

def mamba_chunk_scan_combined(x, dt, A, B, C, chunk_size, D=None, z=None, dt_bias=None, initial_states=None, seq_idx=None, cu_seqlens=None, dt_softplus=False, dt_limit=(0.0, float("inf")), return_final_states=False, return_varlen_states=False):
    """
    Argument:
        x: (batch, seqlen, nheads, headdim)
        dt: (batch, seqlen, nheads)
        A: (nheads)
        B: (batch, seqlen, ngroups, dstate)
        C: (batch, seqlen, ngroups, dstate)
        chunk_size: int
        D: (nheads, headdim) or (nheads,)
        z: (batch, seqlen, nheads, headdim)
        dt_bias: (nheads,)
        initial_states: (batch, nheads, headdim, dstate)
        seq_idx: (batch, seqlen)
        cu_seqlens: (num_sequences + 1) or None, only used if return_varlen_states is True
        dt_softplus: Whether to apply softplus to dt
    Return:
        out: (batch, seqlen, nheads, headdim)
    """
    return MambaChunkScanCombinedFn.apply(x, dt, A, B, C, chunk_size, D, z, dt_bias, initial_states, seq_idx, cu_seqlens, dt_softplus, dt_limit, return_final_states, return_varlen_states)