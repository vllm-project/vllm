"""
    Author: Eric Lin (xihlin)
"""
from typing import TypeVar
from functools import lru_cache
import math
import pytest
import torch

import triton
import triton.language as tl

import os

import dataclasses

Phi3SmallConfig = TypeVar('Phi3SmallConfig')

@dataclasses.dataclass
class BlockSparseParams(object):
    block_size: int
    kernel_block_size: int
    num_local_blocks: int
    vert_stride: int
    homo_head_pattern: bool = False

    @classmethod
    def from_config(cls, config: Phi3SmallConfig) -> "BlockSparseParams":
        return BlockSparseParams(
            block_size=config.blocksparse_block_size,
            kernel_block_size=config.blocksparse_triton_kernel_block_size,
            num_local_blocks=config.blocksparse_num_local_blocks,
            vert_stride=config.blocksparse_vert_stride,
            homo_head_pattern=config.blocksparse_homo_head_pattern,
        )



# triton 2.0.0: fail at backward on A100, for the examples, if h_dim=128.

# Done
#  1. strided of qkv
#  2. seq len not power of 2
#  3. bf16 with Triton May, 2023

# TODO:
#  1. wip: support non-contiguous backward, also help reduce memory allocation in training (q, k, v split)
#  2. block sparse with different BLOCK_M, BLOCK_N?
#  3. for Lq not divided by BLOCK_M, BLOCK_N, only apply mask to K/V on last batch, still need to apply mask on Q.
#     Attempt, fail to compile
#  4. For 2nd iter of inference,  BLOCK_M=1, how to make things work?  K/V maynot divided by BLOCK_N.
#  5. The inner loop can also be paralled via bigger num_stage(better) or on different thread-block (via m/L and atomic update, but this no-comm/sync between blocks)



# helper functions for 3D sparse pattern
# these function are not optimized and very inefficient. Avoid calling them too frequent.
# currently, it is only called within `get_local_strided_sparse_attention_op`, which is cached.
def dense_to_crow_col(x):
    ''' Turning a 2D/3D torch tensor (x) to CSR rows/cols indexing.
    param:
    TODO:
        1. improve efficiency, is it faster if done in CPU, or customize a cuda kernel for it?
    NOTE: col_indices padded -1
    '''
    pad = -1
    dim = x.dim()
    assert x.dim() in (2, 3)
    if x.dim() == 2:
        x = x[None]
    x = [xi.to_sparse_csr() for xi in x]
    crows = torch.vstack([xi.crow_indices() for xi in x])
    cols = [xi.col_indices() for xi in x]
    max_cols = max(len(xi) for xi in cols)
    cols = [torch.cat([xi, pad + xi.new_zeros(max_cols - xi.shape[0])]) for xi in cols]
    cols = torch.vstack(cols)
    if dim == 2:
        crows = crows[0]
        cols = cols[0]
    return crows, cols


def crow_col_to_dense(crows, cols, dtype=torch.float16):
    dim = crows.dim()
    if dim == 1:
        crows = crows[None]
        cols = cols[None]
    device = crows.device
    crows, cols = crows.cpu(), cols.cpu()  # faster in cpu
    shape = (crows.shape[0], crows.shape[1] - 1, cols.max() + 1)
    x = torch.zeros(shape, dtype=dtype)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x[i, j, cols[i, crows[i, j]:crows[i, j+1]]] = 1
    if dim == 1:
        x = x[0]
    return x.to(device)


def dense_to_ccol_row(x):
    '''Similar, but to CSC format
    '''
    x = x.transpose(-2, -1)
    return dense_to_crow_col(x)


def ccol_row_to_dense(ccol, rows, dtype=torch.float16):
    return crow_col_to_dense(ccol, rows, dtype).permute(0, 2, 1).contiguous()


def _get_sparse_attn_mask_homo_head(q_len, N_CTX, dtype, device, BLOCK=128, local_blocks=4, vert_stride=4, return_dense=False):
    '''
    :return: a tuple of 3:
        - tuple of crow_indices, col_indices representation of CSR format.
        - block dense mask
        - all token dense mask (be aware that it can be OOM if it is too big) if `return_dense==True`, otherwise, None
    '''
    with torch.no_grad():
        N_BLOCK = triton.cdiv(N_CTX, BLOCK)
        q_pos = torch.arange(N_BLOCK)[:, None]
        k_pos = torch.arange(N_BLOCK)[None]
        mask_vert_strided = (torch.arange(N_BLOCK) + 1) % vert_stride == 0
        block_mask_dense = ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)).to(device).to(dtype)
        N_BLOCK_Q = triton.cdiv(q_len, BLOCK)
        block_mask_dense_output = block_mask_dense[-N_BLOCK_Q:].contiguous().to_sparse_csr()
    if return_dense:
        mask_dense = torch.kron(block_mask_dense, block_mask_dense.new_ones((BLOCK, BLOCK)))
        causal_mask = torch.tril(torch.ones(N_CTX, N_CTX)).type_as(mask_dense)[-q_len:]
        mask_dense = mask_dense[-q_len:, :N_CTX] * causal_mask
        return (block_mask_dense_output.crow_indices(), block_mask_dense_output.col_indices()), block_mask_dense, mask_dense
    else:
        return (block_mask_dense_output.crow_indices(), block_mask_dense_output.col_indices()), block_mask_dense, None


def _get_sparse_attn_mask(n_heads, q_len, N_CTX, dtype, device, BLOCK=128, local_blocks=4, vert_stride=4, homo_head=True, return_dense=False):
    '''
    :return: a tuple of 3:
        - tuple of crow_indices, col_indices representation of CSR format.
        - block dense mask
        - all token dense mask (be aware that it can be OOM if it is too big) if `return_dense==True`, otherwise, None
    '''
    if homo_head:
        with torch.no_grad():
            (crow, col), block_mask_dense, mask_dense = _get_sparse_attn_mask_homo_head(q_len, N_CTX, dtype, device, BLOCK, local_blocks, vert_stride, return_dense)
            crow = crow[None].expand(n_heads, crow.shape[0])
            col = col[None].expand(n_heads, col.shape[0])
            if return_dense:
                mask_dense = mask_dense[None].expand(n_heads, *mask_dense.shape)
            return (crow, col), block_mask_dense, mask_dense

    with torch.no_grad():
        N_BLOCK = triton.cdiv(N_CTX, BLOCK)
        q_pos = torch.arange(N_BLOCK)[None, :, None]
        k_pos = torch.arange(N_BLOCK)[None, None]
        head_sliding_step = max(1, int(vert_stride / n_heads))  # if vert_stride <= n_heads, rotating the heads
        mask_vert_strided = [(torch.arange(N_BLOCK) + h * head_sliding_step + 1) % vert_stride == 0 for h in range(n_heads)]
        mask_vert_strided = torch.vstack(mask_vert_strided).unsqueeze(1)
        block_mask_dense = ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)).to(device).to(dtype)
        N_BLOCK_Q = triton.cdiv(q_len, BLOCK)
        block_mask_dense_output = block_mask_dense[:, -N_BLOCK_Q:]
    if return_dense:
        mask_dense = torch.kron(block_mask_dense, block_mask_dense.new_ones((BLOCK, BLOCK)))
        causal_mask = torch.tril(torch.ones(N_CTX, N_CTX)).type_as(mask_dense)[-q_len:]
        mask_dense = mask_dense[..., -q_len:, :N_CTX] * causal_mask[None]
        return dense_to_crow_col(block_mask_dense_output), block_mask_dense, mask_dense
    else:
        return dense_to_crow_col(block_mask_dense_output), block_mask_dense, None


def get_sparse_attn_mask(q, N_CTX, *args, **kwargs):
    return _get_sparse_attn_mask(q.size(1), q.size(2), N_CTX, q.dtype, q.device, *args, **kwargs)


# TODO: only apply loading/saving mask on the last iteration for EVEN_N_BLOCK, useful for 1st iteration of inference.
#    Experiment failed inside loop.
#    Another idea: only on saving? load even out of boundary(will it causes illegal access error)?
@triton.jit
def _fwd_kernel(
    Q, K, V, sm_scale,
    layout_crow_ptr,
    layout_col_ptr,
    layout_crow_stride_h, layout_crow_stride_m,
    layout_col_stride_h, layout_col_stride_m,
    TMP, L, M,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug. TMP, L, M are assumed to have contiguous layouts
    Out,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    Z, H, N_CTX,
    PAST_LEN,
    Q_ROUNDED_LEN,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    INFERENCE: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
):
    Q_LEN = N_CTX - PAST_LEN
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_h = off_hz % H
    off_z = off_hz // H
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qd
    # off_k = offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd
    off_k = offs_n[None, :] * stride_kn + offs_d[:, None] * stride_kd
    off_v = offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v
    # initialize pointer to m and l
    t_ptrs = TMP + off_hz * Q_ROUNDED_LEN + offs_m
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    if NUM_DBLOCKS >= 2:
        acc2 = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # load q: it will stay in SRAM throughout
    if EVEN_M_BLOCK:
        q = tl.load(q_ptrs)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd)
    else:
        q = tl.load(q_ptrs, mask=offs_m[:, None] < Q_LEN)
        if NUM_DBLOCKS >= 2:
            q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd, mask=offs_m[:, None] < Q_LEN)

    layout_ptr = layout_crow_ptr + off_h * layout_crow_stride_h + start_m * layout_crow_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_crow_stride_m).to(tl.int32)

    # loop over k, v and update accumulator
    for col_idx_idx in range(start_l, end_l):
        col_idx = tl.load(layout_col_ptr +  off_h * layout_col_stride_h + col_idx_idx * layout_col_stride_m).to(tl.int32)
        start_n = col_idx * BLOCK_N
        # -- compute qk ----
        if EVEN_N_BLOCK:
            k = tl.load(k_ptrs + start_n * stride_kn)
        else:
            k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_n[None, :] + start_n < N_CTX)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, k)

        if NUM_DBLOCKS >= 2:
            if EVEN_N_BLOCK:
                k = tl.load(k_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn + BLOCK_DMODEL * stride_kd, mask=offs_n[None, :] + start_n < N_CTX)
            qk += tl.dot(q2, k)

        qk *= sm_scale
        qk += tl.where(offs_m[:, None] + PAST_LEN >= (start_n + offs_n[None, :]), 0, float('-inf'))
        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        # tl.store(t_ptrs, acc_scale)
        # acc_scale = tl.load(t_ptrs)  # BUG: have to store and immediately load
        acc = acc * acc_scale[:, None]
        if NUM_DBLOCKS >= 2:
            acc2 = acc2 * acc_scale[:, None]
        p = p.to(Q.dtype.element_ty)
        # update acc
        if EVEN_N_BLOCK:
            v = tl.load(v_ptrs + start_n * stride_vn)
        else:
            v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_n[:, None] + start_n < N_CTX)
        acc += tl.dot(p, v)

        if NUM_DBLOCKS >= 2:
            if EVEN_N_BLOCK:
                v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_vd)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn + BLOCK_DMODEL * stride_vd, mask=offs_n[:, None] + start_n < N_CTX)
            acc2 += tl.dot(p, v)

        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new

    # rematerialize offsets to save registers
    # start_m = tl.program_id(0)
    # offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    if not INFERENCE:
        l_ptrs = L + off_hz * N_CTX + offs_m
        m_ptrs = M + off_hz * N_CTX + offs_m
        if EVEN_M_BLOCK:
            tl.store(l_ptrs, l_i)
            tl.store(m_ptrs, m_i)
        else:
            tl.store(l_ptrs, l_i,  mask=offs_m < Q_LEN)
            tl.store(m_ptrs, m_i,  mask=offs_m < Q_LEN)
    # initialize pointers to output
    # offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_z * stride_oz + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :] * stride_od
    out_ptrs = Out + off_o
    tl.store(out_ptrs, acc,  mask=offs_m[:, None] < Q_LEN)
    if NUM_DBLOCKS >= 2:
        tl.store(out_ptrs + BLOCK_DMODEL * stride_od, acc2,  mask=offs_m[:, None] < Q_LEN)


## backward
@triton.heuristics(
    {
        'EVEN_M_BLOCK': lambda kwargs: kwargs['N_CTX'] % kwargs['BLOCK_M'] == 0,
    }
)
@triton.jit
def _bwd_preprocess(
    Out, DO, L, # assume contiguous for Out, DO, L, NewDO, Delta layout.
    NewDO, Delta,
    N_CTX,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
):
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_d = tl.arange(0, D_HEAD)
    # load
    if EVEN_M_BLOCK:
        o = tl.load(Out + off_m[:, None] * D_HEAD + off_d[None, :]).to(tl.float32)
        do = tl.load(DO + off_m[:, None] * D_HEAD + off_d[None, :]).to(tl.float32)
    else:
        o = tl.load(Out + off_m[:, None] * D_HEAD + off_d[None, :], mask=off_m[:, None] < N_CTX).to(tl.float32)
        do = tl.load(DO + off_m[:, None] * D_HEAD + off_d[None, :], mask=off_m[:, None] < N_CTX).to(tl.float32)
    denom = tl.load(L + off_m).to(tl.float32)
    # compute
    do = do / denom[:, None]
    delta = tl.sum(o * do, axis=1)
    # write-back
    if EVEN_M_BLOCK:
        tl.store(NewDO + off_m[:, None] * D_HEAD + off_d[None, :], do)
    else:
        tl.store(NewDO + off_m[:, None] * D_HEAD + off_d[None, :], do,  mask=off_m[:, None] < N_CTX)
    tl.store(Delta + off_m, delta)


# Does not suuport unequal seqlen(q) and seqlen(k)
@triton.heuristics(
    {
        'EVEN_M_BLOCK': lambda kwargs: kwargs['N_CTX'] % kwargs['BLOCK_M'] == 0,
        'EVEN_N_BLOCK': lambda kwargs: kwargs['N_CTX'] % kwargs['BLOCK_N'] == 0,
    }
)
@triton.jit
def _bwd_kernel(
    Q, K, V, sm_scale,
    layout_ccol_ptr,
    layout_row_ptr,
    layout_ccol_stride_h, layout_ccol_stride_m,
    layout_row_stride_h, layout_row_stride_m,
    Out, DO,  # assume contigous: Out, Do, DQ, DK, DV, L, M, D, seq(q) == seq(k), with stride_oz, stride_oh, stride_om, stride_od,
    DQ, DK, DV,
    L, M,
    D,
    stride_qz, stride_qh, stride_qm, stride_qd,
    stride_kz, stride_kh, stride_kn, stride_kd,
    stride_vz, stride_vh, stride_vn, stride_vd,
    stride_oz, stride_oh, stride_om, stride_od,
    # stride_dz, stride_dh, stride_dm, stride_dd,
    Z, H, N_CTX,
    num_block,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr,
    BLOCK_N: tl.constexpr,
    EVEN_M_BLOCK: tl.constexpr,
    EVEN_N_BLOCK: tl.constexpr,
    NUM_DBLOCKS: tl.constexpr,
):
    start_n = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H
    # offset pointers for batch/head
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_h * stride_kh
    V += off_z * stride_vz + off_h * stride_vh
    DO += off_z * stride_oz + off_h * stride_oh
    DQ += off_z * stride_oz + off_h * stride_oh
    DK += off_z * stride_oz + off_h * stride_oh
    DV += off_z * stride_oz + off_h * stride_oh
    # Look like this loop can be parallelled
    # for start_n in range(0, num_block):

    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    # initialize pointers to value-like data
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kd)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :] * stride_vd)

    # pointer to row-wise quantities in value-like data
    D_ptrs = D + off_hz * N_CTX
    m_ptrs = M + off_hz * N_CTX
    # initialize dv amd dk
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    # k and v stay in SRAM throughout
    if EVEN_N_BLOCK:
        k = tl.load(k_ptrs)
        v = tl.load(v_ptrs)
    else:
        k = tl.load(k_ptrs, mask=offs_n[:, None] < N_CTX)
        v = tl.load(v_ptrs, mask=offs_n[:, None] < N_CTX)

    if NUM_DBLOCKS >= 2:
        dv2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        dk2 = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
        if EVEN_N_BLOCK:
            k2 = tl.load(k_ptrs + BLOCK_DMODEL * stride_kd)
            v2 = tl.load(v_ptrs + BLOCK_DMODEL * stride_vd)
        else:
            k2 = tl.load(k_ptrs + BLOCK_DMODEL * stride_kd, mask=offs_n[:, None] < N_CTX)
            v2 = tl.load(v_ptrs + BLOCK_DMODEL * stride_vd, mask=offs_n[:, None] < N_CTX)

    # loop over rows

    layout_ptr = layout_ccol_ptr + off_h * layout_ccol_stride_h + start_n * layout_ccol_stride_m
    start_l = tl.load(layout_ptr).to(tl.int32)
    end_l = tl.load(layout_ptr + layout_ccol_stride_m).to(tl.int32)

    for row_idx_idx in range(start_l, end_l):
        row_idx = tl.load(layout_row_ptr + off_h * layout_row_stride_h + row_idx_idx * layout_row_stride_m).to(tl.int32)
        start_m = row_idx * BLOCK_M

        # offs_qm = start_m + tl.arange(0, BLOCK_M)
        offs_m_curr = start_m + offs_m
        q_ptrs =   Q + (offs_m_curr[:, None] * stride_qm + offs_d[None, :] * stride_qd)
        do_ptrs = DO + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)
        dq_ptrs = DQ + (offs_m_curr[:, None] * stride_om + offs_d[None, :] * stride_od)

        # load q, k, v, do on-chip
        if EVEN_M_BLOCK:
            q = tl.load(q_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < N_CTX)
        # re-compute p = softmax(qk, dim=-1).T
        # NOTE: `do` is pre-divided by `l`; no normalization here
        qk = tl.dot(q, tl.trans(k))

        if NUM_DBLOCKS >= 2:
            if EVEN_M_BLOCK:
                q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd)
            else:
                q2 = tl.load(q_ptrs + BLOCK_DMODEL * stride_qd, mask=offs_m_curr[:, None] < N_CTX)
            qk += tl.dot(q2, tl.trans(k2))

        qk += tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), 0, float('-inf'))

        if EVEN_M_BLOCK:
            m = tl.load(m_ptrs + offs_m_curr)
        else:
            m = tl.load(m_ptrs + offs_m_curr, mask=offs_m_curr < N_CTX)
        p = tl.exp(qk * sm_scale - m[:, None])

        # compute dv
        if EVEN_M_BLOCK:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < N_CTX)

        if NUM_DBLOCKS >= 2:
            if EVEN_M_BLOCK:
                do2 = tl.load(do_ptrs + BLOCK_DMODEL * stride_od)
            else:
                do2 = tl.load(do_ptrs + BLOCK_DMODEL * stride_od, mask=offs_m_curr[:, None] < N_CTX)

        dv += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do)

        if NUM_DBLOCKS >= 2:
            dv2 += tl.dot(tl.trans(p.to(Q.dtype.element_ty)), do2)

        # compute dp = dot(v, do)
        if EVEN_M_BLOCK:
            Di = tl.load(D_ptrs + offs_m_curr)
        else:
            Di = tl.load(D_ptrs + offs_m_curr, mask=offs_m_curr < N_CTX)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) - Di[:, None]
        dp += tl.dot(do, tl.trans(v))

        if NUM_DBLOCKS >= 2:
            dp += tl.dot(do2, tl.trans(v2))

        # compute ds = p * (dp - delta[:, None])
        ds = p * dp * sm_scale
        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q)
        if NUM_DBLOCKS >= 2:
            dk2 += tl.dot(tl.trans(ds.to(Q.dtype.element_ty)), q2)

        # # compute dq
        dq = tl.dot(ds.to(Q.dtype.element_ty), k)
        if EVEN_M_BLOCK:
            tl.atomic_add(dq_ptrs, dq)
        else:
            tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < N_CTX)

        if NUM_DBLOCKS >= 2:
            dq2 = tl.dot(ds.to(Q.dtype.element_ty), k2)
            dq_ptrs2 = dq_ptrs + BLOCK_DMODEL * stride_od
            if EVEN_M_BLOCK:
                tl.atomic_add(dq_ptrs2, dq2)
            else:
                tl.atomic_add(dq_ptrs2, dq2, mask=offs_m_curr[:, None] < N_CTX)

    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_om + offs_d[None, :] * stride_od)
    dk_ptrs = DK + (offs_n[:, None] * stride_om + offs_d[None, :] * stride_od)
    if EVEN_N_BLOCK:
        tl.store(dv_ptrs, dv)
        tl.store(dk_ptrs, dk)
    else:
        tl.store(dv_ptrs, dv, mask=offs_n[:, None] < N_CTX)
        tl.store(dk_ptrs, dk, mask=offs_n[:, None] < N_CTX)

    if NUM_DBLOCKS >= 2:
        dv_ptrs2 = dv_ptrs + BLOCK_DMODEL * stride_od
        dk_ptrs2 = dk_ptrs + BLOCK_DMODEL * stride_od
        if EVEN_N_BLOCK:
            tl.store(dv_ptrs2, dv2)
            tl.store(dk_ptrs2, dk2)
        else:
            tl.store(dv_ptrs2, dv2, mask=offs_n[:, None] < N_CTX)
            tl.store(dk_ptrs2, dk2, mask=offs_n[:, None] < N_CTX)



def _forward(ctx, q, k, v, layout_crow_indices, layout_col_indices, sm_scale, BLOCK_M, BLOCK_N, num_warps=None, num_stages=1, inference=None, out=None):
    '''
    :param q, k, v: [batch, n_heads, seq_len, model_dim]. len of q is allowed to be different than k/v.
    :param layout_crow_indices, layout_col_indices: same as CSR.crow_indices, and CSR.col_indices used to preresent a sparse tensor.
        Each element represent a block, i.e, all elements in a block to be attentdd, or not attended at all..
    '''
    assert q.shape[-1] == k.shape[-1] == v.shape[-1]
    assert k.shape[2] == v.shape[2]
    o = out if out is not None else torch.empty_like(q).contiguous()
    grid = (triton.cdiv(q.shape[2], BLOCK_M), q.shape[0] * q.shape[1])

    q_rounded_len = grid[0] * BLOCK_M
    tmp = torch.empty((q.shape[0] * q.shape[1], q_rounded_len), device=q.device, dtype=torch.float32)

    if inference is None:
        inference = (not q.requires_grad) and (not k.requires_grad)  and (not v.requires_grad)

    if inference:
        L, m = tmp, tmp  # no need to use create new tensor
    else:
        L = torch.empty((q.shape[0] * q.shape[1], q_rounded_len), device=q.device, dtype=torch.float32)
        m = torch.empty((q.shape[0] * q.shape[1], q_rounded_len), device=q.device, dtype=torch.float32)

    if layout_col_indices.dim() == 1:
        layout_crow_indices = layout_crow_indices[None].expand(q.shape[1] , -1)
        layout_col_indices = layout_col_indices[None].expand(q.shape[1] , -1)

    assert q.shape[-1] in [64, 128]
    BLOCK_DMODEL = 64

    if num_warps is None:
        MIN_D = min(BLOCK_M, BLOCK_N, BLOCK_DMODEL)
        num_warps = max(1, 2 ** int(math.log2(MIN_D / 16)))
        # print(f'> {BLOCK_M=}, {BLOCK_N=}, {BLOCK_DMODEL=}, {num_warps=}, {num_stages=}')
    else:
        assert math.log2(num_warps) % 1 == 0, f'''"num_warps" should be power of 2, but got {num_warps}.'''

    ## For debugging:
    # print(f'>> {q.shape=}, {k.shape=}, {BLOCK_M=}, {BLOCK_N=}, {num_warps=}, {BLOCK_DMODEL=}, {q.stride()=}, {k.stride()=}')
    # print(f'>> {layout_crow_indices=}\n{layout_col_indices=}\n {layout_crow_indices.stride()=}, {layout_crow_indices.stride()=}')
    # print(f'> {q.shape=}, {k.shape=}, {layout_crow_indices.shape}, {layout_col_indices.shape}, {layout_crow_indices.stride()}, \
    #   {layout_col_indices.stride()}, {layout_crow_indices=}, {layout_col_indices=}')

    _fwd_kernel[grid](
        q, k, v, sm_scale,
        layout_crow_indices,
        layout_col_indices,
        layout_crow_indices.stride(0), layout_crow_indices.stride(1),
        layout_col_indices.stride(0), layout_col_indices.stride(1),
        tmp, L, m,
        o,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], k.shape[2],
        k.shape[2] - q.shape[2],
        q_rounded_len,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        BLOCK_DMODEL=BLOCK_DMODEL,
        EVEN_M_BLOCK=q.shape[2] % BLOCK_M == 0,
        EVEN_N_BLOCK=k.shape[2] % BLOCK_N == 0 ,
        INFERENCE=inference,
        NUM_DBLOCKS=q.shape[-1] // BLOCK_DMODEL,
        num_warps=num_warps,
        num_stages=num_stages,
    )
    if inference:
        L, m = None, None

    ctx.save_for_backward(q, k, v, o, L, m, layout_crow_indices,  layout_col_indices)
    ctx.BLOCK_M = BLOCK_M
    ctx.BLOCK_N = BLOCK_N
    ctx.BLOCK_DMODEL = BLOCK_DMODEL
    # ctx.BLOCK = BLOCK
    ctx.grid = grid
    ctx.sm_scale = sm_scale
    ctx.num_warps = num_warps
    ctx.num_stages = num_stages
    return o


def _backward(ctx, do, layout_ccol_indices, layout_row_indices, dq=None, dk=None, dv=None):
    # q, k, v, o, l, m = ctx.saved_tensors
    q, k, v, o, l, m, layout_crow_indices, layout_col_indices = ctx.saved_tensors

    ## this following too slow to do online, so get it from inputs, which is cached.
    # layout_ccol_indices, layout_row_indices = dense_to_ccol_row(crow_col_to_dense(ctx.layout_crow_indices, ctx.layout_col_indices))
    # layout_ccol_indices, layout_row_indices = dense_to_ccol_row(crow_col_to_dense(layout_crow_indices, layout_col_indices))

    if not do.is_contiguous():
        do = do.contiguous()
        ## for debugging
        # print(f'----> do is not contiguous: {do.stride()=}')
        # raise ValueError(f'>>>> output grad is not contiguous: {do.stride()=}')

    if not o.is_contiguous():
        # TODO: currently only work with contiguous q/k/v.
        raise ValueError(f'--> output is not contiguous: {o.stride()=}. This is maybe caused by q/k/v not being contiguous.')


    if layout_ccol_indices.dim() == 1:
        layout_ccol_indices = layout_ccol_indices[None].expand(q.shape[1], -1)
        layout_row_indices = layout_row_indices[None].expand(q.shape[1], -1)

    # do = do.contiguous()
    dq = dq if dq is not None else torch.zeros_like(q, dtype=torch.float32)
    dk = dk if dk is not None else torch.empty_like(k)
    dv =dv if dv is not None else  torch.empty_like(v)
    do_scaled = torch.empty_like(do)
    delta = torch.empty_like(l)

    assert o.stride() == dq.stride() == dk.stride() == dv.stride() == do_scaled.stride()

    _bwd_preprocess[(ctx.grid[0] * ctx.grid[1], )](
        o, do, l,
        do_scaled, delta,
        k.shape[2],
        BLOCK_M=ctx.BLOCK_M, D_HEAD=q.shape[-1],
    )

    grid = (triton.cdiv(q.shape[2], ctx.BLOCK_N), ctx.grid[1])

    _bwd_kernel[grid](
        q, k, v, ctx.sm_scale,
        layout_ccol_indices,
        layout_row_indices,
        layout_ccol_indices.stride(0), layout_ccol_indices.stride(1),
        layout_row_indices.stride(0), layout_row_indices.stride(1),
        o, do_scaled,
        dq, dk, dv,
        l, m,
        delta,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0], q.shape[1], q.shape[2],
        ctx.grid[0],
        BLOCK_M=ctx.BLOCK_M,
        BLOCK_N=ctx.BLOCK_N,
        BLOCK_DMODEL=ctx.BLOCK_DMODEL,
        NUM_DBLOCKS=q.shape[-1] // ctx.BLOCK_DMODEL,
        num_warps=ctx.num_warps,
        num_stages=1,
    )
    return dq, dk, dv, None, None, None


class _sparse_attention(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, layout_crow_indices, layout_col_indices, sm_scale):
        BLOCK = 128
        # shape constraints
        return _forward(ctx, q, k, v, layout_crow_indices, layout_col_indices, sm_scale, BLOCK, BLOCK)

    @staticmethod
    def backward(ctx, do):
        # q, k, v, o, l, m = ctx.saved_tensors
        q, k, v, o, l, m, layout_crow_indices, layout_col_indices = ctx.saved_tensors
        # TODO: the following is very inefficient.
        # layout_ccol_indices, layout_row_indices = dense_to_ccol_row(crow_col_to_dense(ctx.layout_crow_indices, ctx.layout_col_indices))
        layout_ccol_indices, layout_row_indices = dense_to_ccol_row(crow_col_to_dense(layout_crow_indices, layout_col_indices))
        return _backward(ctx, do, layout_ccol_indices, layout_row_indices)


# sparse_attention = _sparse_attention.apply

# suppressed
class _sparse_attention_inference(_sparse_attention):
    # TODO: does not work now, as BLOCK_M cannot be <1, as shape for tl.dot cannot be smaller than 16.
    @staticmethod
    def forward(ctx, q, k, v, layout_crow_indices, layout_col_indices, sm_scale):
        BLOCK = 128
        return _forward(ctx, q, k, v, layout_crow_indices, layout_col_indices, sm_scale, 1, BLOCK)

# sparse_attention_inference = _sparse_attention_inference.apply


def sparse_attention_factory(BLOCK_M=128, BLOCK_N=128, **kwargs):
    class _sparse_attention_config(_sparse_attention):
        @staticmethod
        def forward(ctx, q, k, v, layout_crow_indices, layout_col_indices, sm_scale):
            # shape constraints
            return _forward(ctx, q, k, v, layout_crow_indices, layout_col_indices, sm_scale, BLOCK_M, BLOCK_N,
                            **kwargs
                        )
    return _sparse_attention_config.apply


@lru_cache(maxsize=8)
def get_local_strided_sparse_attention_op(
        n_heads: int,
        max_seq_len:int,
        sparse_block_size: int=128,
        local_blocks: int=4,
        vert_stride: int=4,
        homo_head: bool=False,
        dtype=torch.bfloat16,
        device='cuda',
        active_head_range=None,
        verbose=True,
        **kwargs):
    '''
    :param n_heads: total number of attention heads (regardless of tensor/model parallel)
    :param max_seq_len: max sequence length. Need to be bigger or equal to the length of sequences.
    :param sparse_block_size: sparse block size. Default to 128
    :param local_blocks: number of nearest block to attend to. Default to 4, i.e., attention to previous 4xblock_size tokens.
    :param vert_stride: Default to 4. Meaning
    :param homo_head: if all head shared the same pattern.
    :param active_head_range: tuple of start & end of the heads, e..g, (8, 16). Default to use all heads.
                              Mainly for tensor/model parallelization where heads are splitted to different GPUs.
    '''

    if verbose:
        print((f'> new block_sparse_attn op constructed with config: '
            f'{n_heads=}, {max_seq_len=}, {sparse_block_size=}, {local_blocks=}, '
            f'{vert_stride=}, {homo_head=}, {active_head_range=}, {kwargs=}'))
    # assert math.log2(max_seq_len) % 2 == 0, f"max_seq_len should be power of 2 to be more efficient"
    _, block_sparse_pattern, _ = _get_sparse_attn_mask(n_heads, max_seq_len, max_seq_len, dtype, device,
                                                       BLOCK=sparse_block_size, local_blocks=local_blocks,
                                                       vert_stride=vert_stride, homo_head=homo_head,
                                                       return_dense=False)
    if (not homo_head) and (active_head_range is not None):
        assert isinstance(active_head_range, tuple)
        assert len(active_head_range) == 2, '"active_head_range" should be a tuple of start/end index of the heads.'
        h_start, h_end = active_head_range
        block_sparse_pattern = block_sparse_pattern[h_start:h_end]
    # print(block_sparse_pattern)
    attn_op = get_sparse_attn_op(block_sparse_pattern, sparse_block_size, **kwargs)
    attn_op.local_tokens = local_blocks * sparse_block_size
    attn_op.sparse_block_size = sparse_block_size
    return attn_op


def get_sparse_attn_op(
        sparse_pattern: torch.tensor,
        sparse_block_size: int=128,
        kernel_block_size=128,
        qkv_format='q,k,v',
          **kwargs):
    '''
    Ccreate a block-sparse op with fixed layout. This is to avoid the need to of create CSR layout and convert it to CSC layout everytime,
        which is very inefficient (use python loops on CPU.  PyTorch 1.13 supports CSR->CSC, may help.)

    :param sparse_pattern: sparse pattern of the blocks. Should be `num_blocks(q) x num_blocks(k)` or `n_heads x num_blocks x num_blocks`.
        This tensor should have lower-triangular matrices on the last 2 dimensions for causal attention
    :param sparse_block_size: sparse block size. Default to 128
    :param kernel_block_size: the tile/block size to launch a triton instance. Default to None, i.e., same as `sparse_block_size`
    :param qkv_format: Choices=['q,k,v', 'q, kv', 'qkv'], i.e., separated q,k,v, or kv packed, or qkv packed. Currently, only 'q,k,v' is supported.

    :param kwargs: keyward arguments passed to `_forward`
    '''
    # assert qkv_format in ('q,k,v', 'q, kv', 'qkv')  # to save from running `concat` at forward/backward

    assert qkv_format == 'q,k,v'

    if kernel_block_size is None:
        kernel_block_size = sparse_block_size
    else:
        assert sparse_block_size % kernel_block_size == 0, f"The sparse block size must be a multiple of {kernel_block_size}."
        assert kernel_block_size >=16 and math.log2(kernel_block_size) % 1 == 0, f"block_size must be power of 2 and at least 16, but {kernel_block_size} is given"


        # print(f'>> {sparse_pattern.shape=}')
        # print(f'{sparse_pattern=}')
        if sparse_block_size // kernel_block_size > 1:
            _mul = sparse_block_size // kernel_block_size
            # need to consider if block_m and block_n are different
            sparse_pattern = torch.kron(sparse_pattern, sparse_pattern.new_ones(_mul, _mul))
            num_sparse_blocks = sparse_pattern.size(-1)
            block_causal_mask = torch.arange(0, num_sparse_blocks)[:, None] >= torch.arange(0, num_sparse_blocks)[None]
            sparse_pattern *= block_causal_mask.type_as(sparse_pattern)
            # print(f'>> after: {sparse_pattern.shape=}')
            # print(f'{sparse_pattern=}')

    BLOCK_N = kernel_block_size
    NUM_BLOCK =  sparse_pattern.size(-1)
    MAX_SEQ_LEN = kernel_block_size * NUM_BLOCK

    grand_layout_crow_indices, grand_layout_col_indices = dense_to_crow_col(sparse_pattern)
    # sparse csc layout for backward
    grand_layout_ccol_indices, grand_layout_row_indices = dense_to_ccol_row(sparse_pattern)


    # cache GPU backward layout. limit the size to avoid OOM as time goes.
    # For inference, one only needs to cache one block as sequence length always increases
    # Therefore, this cache needs to be reconstructed per every `block_size`-steps.
    # For training/finetune, set to 8 to increase cache hit.
    # Given an input, the block_len will be the same for all layers, so cache is very helpful.

    max_cache_size = 1 if kwargs.get('inference', False) else 8

    @lru_cache(maxsize=max_cache_size)
    def get_backward_layout_by_block_len(block_len):
        assert block_len <= NUM_BLOCK
        if block_len == NUM_BLOCK:
            return (grand_layout_ccol_indices, grand_layout_row_indices)
        return dense_to_ccol_row(sparse_pattern[..., :block_len, :block_len])

    # for debugging
    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     print(f'> {sparse_pattern.cpu().tolist()=}')
    #     print('----')
    #     print(f'> {grand_layout_crow_indices.cpu().tolist()=}\n{grand_layout_col_indices.cpu().tolist()=}')


     # q, k, v separated
    class _q_k_v_sparse_attention(torch.autograd.Function):
        @staticmethod
        def forward(ctx, q, k, v, sm_scale):
            # assert q.shape[2] == 1 or q.shape[2] == k.shape[2]
            # shape constraints
            MIN_BLOCK_SIZE = 16
            assert BLOCK_N >= MIN_BLOCK_SIZE
            BLOCK_M = 16 if q.shape[2] <= 16 else BLOCK_N  # BLOCK_M has to be power of 2

            # this following code only works for causal attention
            K_BLOCKS = triton.cdiv(k.shape[2],  kernel_block_size)
            # Q_START_BLOCKS = K_BLOCKS - 1 if q.shape[2] == 1 else 0
            Q_START_BLOCKS = K_BLOCKS - triton.cdiv(q.shape[2], BLOCK_N)
            # print(Q_START_BLOCKS, K_BLOCKS)

            layout_crow_indices = grand_layout_crow_indices[..., Q_START_BLOCKS:K_BLOCKS+1]
            layout_col_indices = grand_layout_col_indices
            # print(BLOCK_M, BLOCK_N, Q_START_BLOCKS, K_BLOCKS+1, layout_crow_indices, layout_col_indices)

            return _forward(ctx, q, k, v, layout_crow_indices, layout_col_indices, sm_scale, BLOCK_M, BLOCK_N,
                            **kwargs
                        )
        @staticmethod
        def backward(ctx, do):
            q, k = ctx.saved_tensors[:2]
            assert q.shape[2] == k.shape[2], '> currently backward can only be done if q, k have same length. Contact @EricLin if you need it.'
            # assume q, k have same length
            block_len = triton.cdiv(do.shape[2], kernel_block_size)
            backward_layout = get_backward_layout_by_block_len(block_len)
            return _backward(ctx, do, *backward_layout)[:4]


    def _q_k_v_sparse_attention_fn(*args):
        return _q_k_v_sparse_attention.apply(*args)

    _q_k_v_sparse_attention_fn.sparse_pattern = sparse_pattern
    _q_k_v_sparse_attention_fn.grand_layout_crow_indices = grand_layout_crow_indices
    _q_k_v_sparse_attention_fn.grand_layout_col_indices = grand_layout_col_indices
    _q_k_v_sparse_attention_fn.grand_layout_ccol_indices = grand_layout_ccol_indices
    _q_k_v_sparse_attention_fn.grand_layout_row_indices = grand_layout_row_indices

    return _q_k_v_sparse_attention_fn


def torch_attention(q, k, v, attn_mask=None, sm_scale=None, block_attn_mask=None, block_size=128, do=None):
    '''
    q, k, v: shape=(batch, n_heads, seq, dim)
    '''
    # for verification
    if sm_scale is None:
        sm_scale = math.sqrt(float(q.size(-1)))

    if block_attn_mask is not None:
        assert attn_mask is None
        outs = []
        for s in range(0, q.size(2), block_size):
            e = min(s + block_size, q.size(2))
            q_block = q[:, :, s:e]
            attn = torch.einsum('bhmd,bhnd->bhmn', q_block, k[:, :, :e]).float() * sm_scale
            mask = block_attn_mask[..., s // block_size, : (s // block_size + 1)]
            mask = torch.kron(mask, torch.ones(block_size, block_size, device=mask.device))
            mask[..., :, s:].masked_fill_(torch.arange(0, block_size)[:, None] <= torch.arange(0, block_size)[None, :], 0)
            attn = attn.masked_fill((1 - mask).bool(), float('-inf'))
            attn = attn.softmax(-1)
            out = torch.einsum('bhmn,bhnd->bhmd', attn.type_as(v), v[:, :, :e])
            outs.append(out)
        torch_output = torch.cat(outs, dim=2)
    else:
        attn = torch.einsum('bhmd,bhnd->bhmn', q, k).float() * sm_scale
        # import ipdb; ipdb.set_trace()
        if attn_mask is not None:
            attn = attn.masked_fill((1 - attn_mask).bool(), float('-inf'))
        # print(f'> torch attn: {attn.exp().sum(-1)=}')

        attn = attn.softmax(-1)
        if do is not None:
            dv = torch.einsum('bhqk,bhqd->bhkd', attn.type_as(do), do)
            print(f'> torch_attn computed dv: {dv=}')
        torch_output = torch.einsum('bhmn,bhnd->bhmd', attn.type_as(v), v)
    return torch_output


##############
# Unit tests #
##############

@pytest.mark.parametrize('Z, H, N_CTX, D_HEAD', [(2, 8, 2048, 128), (1, 4, 4096, 64)])
def test_op(Z, H, N_CTX, D_HEAD, Q_LEN=None, dtype=torch.bfloat16, homo_head=True, kernel_block_size=None, sparse_block_size=128, backward=True,
            sparse_attention_fn=None, local_blocks=4, vert_stride=4, sm_scale=None, max_length=None):
    Q_LEN = Q_LEN or N_CTX
    torch.manual_seed(20)
    q = torch.empty((Z, H, Q_LEN, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0, std=.5) # .requires_grad_()
    k = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0, std=.5) # .requires_grad_()
    v = torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device='cuda').normal_(mean=0, std=.5) # .requires_grad_()

    if sm_scale is None:
        sm_scale = 1. / math.sqrt(D_HEAD)

    # for debugging
    # print(f'>> {q.shape=}, {k.shape=}, {v.shape=}, {homo_head=}, {kernel_block_size=}, {sparse_block_size=}, {local_blocks=}, {vert_stride=}')
    sm_scale = 0.0078125
    if backward:
        q.requires_grad_(), k.requires_grad_(), v.requires_grad_()

    # qkv = torch.empty((Z, N_CTX, 3*H*D_HEAD), dtype=dtype, device='cuda').normal_(mean=0, std=.5)
    # q = qkv[..., :H*D_HEAD]
    # k = qkv[..., H*D_HEAD:2*H*D_HEAD]
    # v = qkv[..., 2*H*D_HEAD:]
    # q = q.view(Z, N_CTX, H, -1).permute(0, 2, 1, 3)
    # k = k.view(Z, N_CTX, H, -1).permute(0, 2, 1, 3)
    # v = v.view(Z, N_CTX, H, -1).permute(0, 2, 1, 3)

    # if Q_LEN and Q_LEN < N_CTX:
    #     q = q[:, :, -Q_LEN:] # .contiguous()

    # q = q.requires_grad_()
    # k = k.requires_grad_()
    # v = v.requires_grad_()

    dout = torch.randn_like(q).contiguous()

    # dout = torch.eye(N_CTX)[:, :D_HEAD][None, None].expand_as(q).type_as(q).contiguous()
    # print(dout)

    mask_csr, _, mask_dense = get_sparse_attn_mask(q, N_CTX, BLOCK=sparse_block_size,
                            local_blocks=local_blocks, vert_stride=vert_stride, homo_head=homo_head, return_dense=True)

    if sparse_attention_fn is None:
        sparse_attention_fn = get_local_strided_sparse_attention_op(H, N_CTX,
                                                                    sparse_block_size=sparse_block_size,
                                                                    local_blocks=local_blocks,
                                                                    vert_stride=vert_stride,
                                                                    homo_head=homo_head,
                                                                    device=q.device,
                                                                    dtype=q.dtype,
                                                                    kernel_block_size=kernel_block_size)
    # reference implementation
    ref_out = torch_attention(q, k, v, mask_dense, sm_scale)

    # lengths = torch.full((Z,), fill_value=N_CTX, device='cuda')
    # cu_seqlens = torch.zeros((Z + 1,), device='cuda', dtype=torch.int32)
    # cu_seqlens[1:] = lengths.cumsum(0)
    # # qkv = torch.randn((Z * N_CTX, 3, H, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)

    # qkv_list = list(map(lambda x: x.permute(0, 2, 1, 3).contiguous().view(Z * N_CTX, 1, H, D_HEAD), [q, k, v]))
    # qkv = torch.cat(qkv_list, dim=1)
    # ref_out0 = flash_attn_func(qkv, cu_seqlens, dropout_p=0, max_s=N_CTX, softmax_scale=sm_scale, causal=True)
    # ref_out = ref_out0.view(Z, N_CTX, H, D_HEAD).permute(0, 2, 1, 3).contiguous()


    if backward:
        ref_out.backward(dout)
        ref_dv, v.grad = v.grad.clone(), None
        ref_dk, k.grad = k.grad.clone(), None
        ref_dq, q.grad = q.grad.clone(), None

    tri_out = sparse_attention_fn(q, k, v, sm_scale)

    decimal = 1 if dtype == torch.bfloat16 else 2
    assert torch.allclose(ref_out.cpu(), tri_out.cpu(), atol=1e-2, rtol=0), f'>> {ref_out[0, 0, :, 0].tolist()=}\n\n{tri_out[0, 0, :, 0].tolist()=}'

    if backward:
        tri_out.backward(dout)
        tri_dv, v.grad = v.grad.clone(), None
        tri_dk, k.grad = k.grad.clone(), None
        tri_dq, q.grad = q.grad.clone(), None

    if backward:
        assert torch.allclose(ref_dv, tri_dv, atol=1e-2, rtol=1e-2)
        assert torch.allclose(ref_dk, tri_dk, atol=1e-2, rtol=0)
        assert torch.allclose(ref_dq, tri_dq, atol=1e-2, rtol=0)

    print(f'> test passed: {Z=}, {H=}, {N_CTX=}, {D_HEAD=}, {Q_LEN=}, {dtype=}, {homo_head=}, {sparse_block_size=}')


if __name__ == '__main__':

    GPU_TYPE = os.popen('nvidia-smi --query-gpu=name --format=csv | tail -n 1').read().strip()
    # print(GPU_TYPE)
    support_backward = True # 'A100' in GPU_TYPE. Wasn't supportted in consumer A1000.

    ###############
    # benchmarking

    HAS_DENSE_TRITON_FLASH = False
    # try:
    #     from triton.ops.flash_attention import attention as triton_attention
    #     HAS_DENSE_TRITON_FLASH = True
    # except:
    #     HAS_DENSE_TRITON_FLASH = False
    #     print('> cannot import Trition flash attn')

    try:
        from flash_attn.flash_attn_interface import flash_attn_func, flash_attn_unpadded_func
        HAS_FLASH = True
    except BaseException:
        HAS_FLASH = False
        print('> cannot import flash_attn')


    # BATCH, N_HEADS, N_CTX, D_HEAD = 4, 48, 4096, 64
    BATCH, N_HEADS, N_CTX, D_HEAD = 4, 32, 4096, 128  # 6.7B model, with 4k len
    # BATCH, N_HEADS, N_CTX, D_HEAD = 4, 16, 4096, 128  # 204m model

    BLOCK_SIZE = 64
    LOCAl_BLOCKS = 8 # 4
    VERT_STRIDE = 1 # 16 # 8
    HOMO_HEAD = False
    sparse_type = 'home' if HOMO_HEAD else 'hetero'
    dtype = torch.bfloat16


    modes = ['fwd', 'bwd'] if support_backward else ['fwd']

    configs = [triton.testing.Benchmark(
        x_names=['SEQ_LEN'],
        x_vals=[2**i for i in range(8, 16)],
        line_arg='provider',
        line_vals=(['triton'] if HAS_DENSE_TRITON_FLASH else []) + (['flash'] if HAS_FLASH else []) + ['triton_sparse'],
        line_names=(['Triton-Dense'] if HAS_DENSE_TRITON_FLASH else [])  + (['Flash-Dense'] if HAS_FLASH else []) + ['Triton-Sparse'],
        styles=[('red', '-'), ('blue', '-'), ('green', '-')],
        ylabel='ms',
        plot_name=f'fused-attention-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-sparse-local{LOCAl_BLOCKS}-vert{VERT_STRIDE}-{sparse_type}-{dtype}-{mode}',
        args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'dtype': dtype, 'mode': mode}
    ) for mode in modes]


    @triton.testing.perf_report(configs)
    def bench_flash_attention(BATCH, H, SEQ_LEN, D_HEAD, mode, provider, dtype=torch.bfloat16, device='cuda', sparse_attention_fn=None):
        assert mode in ['fwd', 'bwd']
        warmup = 25
        rep = 100
        N_CTX = SEQ_LEN
        if provider == 'triton':
            q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            sm_scale = 1.3
            fn = lambda: triton_attention(q, k, v, sm_scale)
            if mode == 'bwd':
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        if provider == 'triton_sparse':
            q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
            sm_scale = 1.3
            # q_pos = torch.arange(N_CTX // BLOCK, device='cuda')[:, None]
            # k_pos = torch.arange(N_CTX // BLOCK, device='cuda')[None]
            # local_blocks = 4 # num_block per attn, block_size is tied to BLOCK
            # vert_stride =N_CTX + 1 # 4
            # mask_vert_strided = torch.arange(N_CTX // BLOCK, device='cuda') % vert_stride == vert_stride - 1
            # mask_dense = ((q_pos >= k_pos) & ((q_pos - k_pos < local_blocks) | mask_vert_strided)).type_as(q)
            # mask = mask_dense.to_sparse_csr()
            # mask_csr, _ = get_sparse_attn_mask(q, N_CTX, BLOCK=BLOCK, local_blocks=LOCAl_BLOCKS, vert_stride=VERT_STRIDE, homo_head=HOMO_HEAD)

            if sparse_attention_fn is None:
                # sparse_attention_fn = sparse_attention
                sparse_attention_fn = get_local_strided_sparse_attention_op(H, SEQ_LEN,
                                                                            local_blocks=LOCAl_BLOCKS,
                                                                            vert_stride=VERT_STRIDE,
                                                                            homo_head=HOMO_HEAD,
                                                                            sparse_block_size=BLOCK_SIZE,
                                                                            kernel_block_size=BLOCK_SIZE,
                                                                            device=q.device)
            # sparse_attention_fn = sparse_attention_factory(128, 128, num_warps=8)

            # fn = lambda: sparse_attention_fn(q, k, v, mask_csr[0], mask_csr[1], sm_scale)
            fn = lambda: sparse_attention_fn(q, k, v, sm_scale)
            if mode == 'bwd':
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        if provider == 'flash':
            lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            qkv = torch.randn((BATCH * N_CTX, 3, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
            fn = lambda: flash_attn_func(qkv, cu_seqlens, 0., N_CTX, causal=True)
            if mode == 'bwd':
                o = fn()
                do = torch.randn_like(o)
                fn = lambda: o.backward(do, retain_graph=True)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms

        # if provider == 'torch':
        #     q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
        #     k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
        #     v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=True)
        #     sm_scale = 1.3
        #     causal_mask = torch.tril(torch.ones(N_CTX, N_CTX)).type_as(q)
        #     fn = lambda:  torch_attention(q, k, v, causal_mask, sm_scale)
        #     ms = triton.testing.do_bench(fn, percentiles=None, warmup=warmup, rep=rep)
        #     return ms


    BATCH, N_HEADS, N_CTX, D_HEAD, Q_LEN = 4, 32, 4096, 128, 1  # 6.7B model, with 4k len

    BLOCK_SIZE = 64
    LOCAl_BLOCKS = 8 # 4
    VERT_STRIDE = 16 # 8
    HOMO_HEAD = False
    sparse_type = 'home' if HOMO_HEAD else 'hetero'
    dtype = torch.bfloat16
    MAX_N_CTX = 8192

    configs = [triton.testing.Benchmark(
        x_names=['PAST_LEN'],
        x_vals=[2**i - 1 for i in range(8, 14)],
        line_arg='provider',
        line_vals=['torch'] + (['flash'] if HAS_FLASH else []) + ['triton_sparse', 'triton_dense'],
        line_names=['Torch']  + (['Flash-Dense'] if HAS_FLASH else []) + ['Triton-Sparse', 'Triton-Dense'],
        styles=[('red', '-'), ('blue', '-'), ('green', '-'), ('cyan', '-')],
        ylabel='ms',
        plot_name=f'fused-attention-inference-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-sparse-local{LOCAl_BLOCKS}-vert{VERT_STRIDE}-{sparse_type}',
        args={'H': N_HEADS, 'BATCH': BATCH, 'D_HEAD': D_HEAD, 'Q_LEN': Q_LEN, 'dtype': torch.float16, 'mode': mode}
    ) for mode in ['fwd']]
    @triton.testing.perf_report(configs)
    def bench_flash_attention_inference(BATCH, H, PAST_LEN, D_HEAD, Q_LEN, mode, provider, dtype=torch.bfloat16, device='cuda'):
        assert mode in ['fwd']
        warmup = 25
        rep = 100
        N_CTX = PAST_LEN + Q_LEN
        if provider == 'torch':
            q = torch.randn((BATCH, H, Q_LEN, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            sm_scale = 1.3
            mask_csr, _, mask_dense = get_sparse_attn_mask(q, N_CTX, BLOCK=BLOCK_SIZE,
                                    local_blocks=LOCAl_BLOCKS, vert_stride=VERT_STRIDE, homo_head=VERT_STRIDE, return_dense=True)

            fn = lambda: torch_attention(q, k, v, mask_dense, sm_scale=sm_scale, block_size=2048)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        if provider == 'triton_sparse':
            q = torch.randn((BATCH, H, Q_LEN, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            sm_scale = 1.3
            sparse_attention_fn = get_local_strided_sparse_attention_op(H, MAX_N_CTX,
                                                                        local_blocks=LOCAl_BLOCKS,
                                                                        vert_stride=VERT_STRIDE,
                                                                        homo_head=HOMO_HEAD,
                                                                        sparse_block_size=BLOCK_SIZE,
                                                                        kernel_block_size=BLOCK_SIZE,
                                                                        device=q.device,
                                                                        inference=True)

            fn = lambda: sparse_attention_fn(q, k, v, sm_scale)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        if provider == 'triton_dense':
            q = torch.randn((BATCH, H, Q_LEN, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            sm_scale = 1.3
            sparse_attention_fn = get_local_strided_sparse_attention_op(H, MAX_N_CTX,
                                                                        local_blocks=1,
                                                                        vert_stride=1,
                                                                        homo_head=True,
                                                                        sparse_block_size=BLOCK_SIZE,
                                                                        kernel_block_size=BLOCK_SIZE,
                                                                        device=q.device,
                                                                        inference=True)

            fn = lambda: sparse_attention_fn(q, k, v, sm_scale)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms
        if provider == 'flash':
            assert Q_LEN == 1
            lengths = torch.full((BATCH,), fill_value=N_CTX, device=device)
            cu_seqlens = torch.zeros((BATCH + 1,), device=device, dtype=torch.int32)
            cu_seqlens[1:] = lengths.cumsum(0)
            cu_seqlens_q = torch.arange(BATCH + 1, device=device, dtype=torch.int32)

            # (total_q, nheads, headdim),
            q = torch.randn((BATCH, H, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            k = torch.randn((BATCH*N_CTX, H, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)
            v = torch.randn((BATCH*N_CTX, H, D_HEAD), dtype=dtype, device='cuda', requires_grad=False)

            fn = lambda: flash_attn_unpadded_func(q, k, v, cu_seqlens_q, cu_seqlens, 1, N_CTX, dropout_p=0, softmax_scale=1.3, causal=False)
            ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
            return ms


    test_op(1, 4, 512, 128, dtype=torch.float16, homo_head=False, backward=support_backward)
    # bench_flash_attention.run(save_path='.', print_data=True)

    bench_flash_attention_inference.run(save_path='.', print_data=True)
    exit()
    # head_dim=64
    test_op(1, 2, 1024, 64, kernel_block_size=64, sparse_block_size=64,
            dtype=torch.bfloat16, homo_head=False, backward=support_backward)
    # uneven length, bf16
    test_op(1, 16, 224, 128, dtype=torch.bfloat16, homo_head=False, backward=False, sparse_block_size=128,
            kernel_block_size=64, local_blocks=8, vert_stride=8)
    test_op(3, 2, 2047, 128, homo_head=False, backward=False)

    # diff kernel/sparse block size
    test_op(1, 16, 224, 128, dtype=torch.bfloat16, homo_head=False, backward=False, kernel_block_size=64)
    # inference
    # test_op(1, 4, 512 + 256, 128, Q_LEN=1,  dtype=torch.bfloat16, homo_head=False, backward=support_backward)

    # dense flash attn
    test_op(1, 2, 1024, 128, kernel_block_size=128, sparse_block_size=128, dtype=torch.bfloat16, homo_head=False,
            backward=support_backward, local_blocks=1, vert_stride=1)

    # fp16
    test_op(1, 4, 512 + 256, 128, dtype=torch.float16, homo_head=False, backward=support_backward)

    # longer sequence
    test_op(2, 4, 8192, 64, homo_head=False, backward=support_backward)
    test_op(2, 4, 8192, 128, dtype=torch.bfloat16, homo_head=False, backward=support_backward)

    # homo head
    test_op(3, 2, 2048, 64, homo_head=True, dtype=torch.bfloat16, backward=False)
    test_op(3, 2, 2048, 64, homo_head=True, backward=support_backward)

    # sparse_attention_fn = sparse_attention_factory(16, 128, num_warps=1, INFERENCE=True)
    # test_op(8, 1, 2047, 128, 1, backward=False, sparse_attention_fn=None)
    # test_op_inference(3, 2, 2048, 128, 2048)
    # test_op_inference(3, 2, 2047, 64, 2047)
    # test_op_inference(3, 2, 256, 64, 128)
    # test_op_inference(3, 2, 2048, 64, 1)

    bench_flash_attention.run(save_path='.', print_data=True)
    # bench_flash_attention_inference.run(save_path='.', print_data=True)

# ========================
# Some Benchmark Results #
# ========================

# fused-attention-batch4-head48-d64-sparse-local4-vert4-hetero-fwd
#    SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.057184     0.069646       0.052567
# 1    512.0      0.131688     0.187658       0.110212
# 2   1024.0      0.391844     0.524990       0.247875
# 3   2048.0      1.305190     1.456685       0.596506
# 4   4096.0      4.623019     4.968653       1.600277
# 5   8192.0     17.513062    18.332262       4.802458
# 6  16384.0     68.453377    70.337540      16.052908
# 7  32768.0    270.655487   276.020233      57.938946
# fused-attention-batch4-head48-d64-sparse-local4-vert4-hetero-bwd (num_warp=8):
# SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.190120     0.150313       0.181451
# 1    512.0      0.406348     0.391767       0.391177
# 2   1024.0      1.029704     1.182967       0.885741
# 3   2048.0      2.985456     3.843399       2.040469
# 4   4096.0      9.808897    13.073701       5.069609
# 5   8192.0     34.995201    47.863808      13.948782
# 6  16384.0    132.740097   182.579193      42.816513
# 7  32768.0    542.223389   714.820618     147.053574
# fused-attention-inference-batch4-head32-d128-sparse-local4-vert4-hetero:
# PAST_LEN  Torch-Dense  Flash-Dense  Triton-Sparse
# 0     256.0     0.050949     0.032357       0.107513
# 1     512.0     0.073624     0.050651       0.199086
# 2    1024.0     0.107472     0.080379       0.245445
# 3    2048.0     0.178423     0.129448       0.338259
# 4    4096.0     0.327647     0.223106       0.517048
# 5    8192.0     0.588423     0.411263       0.884606
# 6   16384.0     1.098898     0.798941       1.611809
# 7   32768.0     2.094537     1.594726       3.044160


# 6.7B
# fused-attention-batch4-head32-d128-sparse-local4-vert4-hetero-fwd:
#    SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.069208     0.082156       0.065097
# 1    512.0      0.138271     0.201393       0.144467
# 2   1024.0      0.391521     0.624614       0.322382
# 3   2048.0      1.268443     2.406325       0.784367
# 4   4096.0      4.455703     9.139097       2.100856
# 5   8192.0     16.764315    35.289600       6.328320
# 6  16384.0     65.221634   138.401794      21.069057
# 7  32768.0    257.251343   548.085754      76.111870
# fused-attention-batch4-head32-d128-sparse-local4-vert4-hetero-bwd:
#    SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.297118     0.266469       0.255255
# 1    512.0      0.672826     0.613685       0.552954
# 2   1024.0      1.718434     1.705066       1.251953
# 3   2048.0      4.936755     5.403875       2.927895
# 4   4096.0     15.911594    18.959362       7.436288
# 5   8192.0     55.357441    70.808578      21.140224
# 6  16384.0    208.188416   273.617920      68.018173
# 7  32768.0    806.037476  1081.453613     218.720261
# fused-attention-inference-batch4-head32-d128-sparse-local4-vert4-hetero:
#    PAST_LEN  Torch-Dense  Flash-Dense  Triton-Sparse
# 0     256.0     0.050151     0.032337       0.107593
# 1     512.0     0.073409     0.051737       0.200200
# 2    1024.0     0.107533     0.082099       0.247067
# 3    2048.0     0.177259     0.128891       0.338510
# 4    4096.0     0.325866     0.223621       0.524842
# 5    8192.0     0.586926     0.408913       0.885490
# 6   16384.0     1.100834     0.793277       1.612271
# 7   32768.0     2.098851     1.595831       3.064544

# fused-attention-batch4-head32-d128-sparse-local4-vert8-hetero-fwd:
#    SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.066673     0.082037       0.065085
# 1    512.0      0.137379     0.201880       0.143473
# 2   1024.0      0.390675     0.624234       0.312046
# 3   2048.0      1.267739     2.406950       0.696045
# 4   4096.0      4.445138     9.136333       1.665788
# 5   8192.0     16.768614    35.265533       4.380486
# 6  16384.0     65.235970   138.393600      12.997633
# 7  32768.0    257.317902   550.442993      42.821121
# fused-attention-batch4-head32-d128-sparse-local4-vert8-hetero-bwd:
#    SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.296461     0.266581       0.254022
# 1    512.0      0.671427     0.613643       0.551283
# 2   1024.0      1.719918     1.704295       1.229982
# 3   2048.0      4.945305     5.403364       2.721906
# 4   4096.0     15.934293    18.960999       6.259371
# 5   8192.0     55.406593    70.832130      15.676929
# 6  16384.0    208.750595   275.004425      44.837891
# 7  32768.0    808.057861  1080.647705     141.856766
# fused-attention-inference-batch4-head32-d128-sparse-local4-vert8-hetero:
#    PAST_LEN  Torch-Dense  Flash-Dense  Triton-Sparse
# 0     256.0     0.050739     0.032886       0.107837
# 1     512.0     0.073507     0.051996       0.200293
# 2    1024.0     0.106394     0.080679       0.240610
# 3    2048.0     0.177659     0.127660       0.287625
# 4    4096.0     0.326326     0.226971       0.377500
# 5    8192.0     0.586339     0.407367       0.559266
# 6   16384.0     1.102279     0.786221       0.920976
# 7   32768.0     2.097370     1.545090       1.644288


################
##### fp16 #####
################

# fused-attention-batch4-head16-d64-sparse-local4-vert8-hetero-fwd:
#    SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.032518     0.035472       0.029939
# 1    512.0      0.054266     0.087841       0.054320
# 2   1024.0      0.133447     0.263090       0.102045
# 3   2048.0      0.384615     1.023293       0.201763
# 4   4096.0      1.300890     4.023936       0.449555
# 5   8192.0      4.774144    15.816704       1.150854
# 6  16384.0     18.220032    62.771198       3.356001
# 7  32768.0     71.405571   250.273788      10.976142
# fused-attention-batch4-head16-d64-sparse-local4-vert8-hetero-bwd:
#    SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.083342     0.069742       0.079496
# 1    512.0      0.159894     0.170995       0.151705
# 2   1024.0      0.386071     0.522407       0.331443
# 3   2048.0      1.067715     1.737333       0.715248
# 4   4096.0      3.382731     6.219520       1.597457
# 5   8192.0     11.857793    23.560448       3.879035
# 6  16384.0     44.422142    91.251709      10.626843
# 7  32768.0    175.011841   359.473145      32.340992


################
##### bf16 #####
################

# fused-attention-batch4-head16-d64-sparse-local4-vert8-hetero-fwd:
#    SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.037636     0.035902       0.031512
# 1    512.0      0.058591     0.087229       0.058125
# 2   1024.0      0.143337     0.263919       0.108443
# 3   2048.0      0.414458     1.025985       0.214114
# 4   4096.0      1.390841     4.020010       0.480550
# 5   8192.0      5.067938    15.808171       1.230874
# 6  16384.0     19.442280    62.765057       3.597274
# 7  32768.0     75.501572   250.443771      11.768959
# fused-attention-batch4-head16-d64-sparse-local4-vert8-hetero-bwd:
#    SEQ_LEN  Triton-Dense  Flash-Dense  Triton-Sparse
# 0    256.0      0.084404     0.070663       0.082613
# 1    512.0      0.161510     0.172882       0.157661
# 2   1024.0      0.388954     0.526047       0.339855
# 3   2048.0      1.075814     1.736057       0.732420
# 4   4096.0      3.401622     6.221376       1.636039
# 5   8192.0     11.915136    23.483391       3.968725
# 6  16384.0     44.660225    91.302910      10.857130
# 7  32768.0    175.038467   359.048187      32.778240