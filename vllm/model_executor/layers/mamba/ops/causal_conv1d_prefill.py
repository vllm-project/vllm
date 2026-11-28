# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Device-driven width-four causal convolution with SiLU activation.

One launch computes outputs, updates each sequence's convolution state, and
exports checkpoint windows without host-built chunk metadata or an input copy.
FP32 uses explicit FMAs to preserve the upstream kernel's accumulation order.

Each program handles a tile of tokens and channels. It finds sequence boundaries
by binary search. Tokens after the first three read their taps from input rows;
the first three also use the cached window. The program containing a sequence's
first token owns its state and checkpoint writes, avoiding races between tiles.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID


@triton.jit(
    do_not_specialize=["T", "N", "stride_x_tok", "stride_o_tok", "stride_state_seq"]
)
def _causal_conv1d_prefill_kernel(
    x_ptr,  # [T, DIM] rows of stride_x_tok
    w_ptr,  # [DIM, 4] contiguous
    state_ptr,  # [slots, DIM, 3] any strides
    o_ptr,  # [T, DIM] rows of stride_o_tok
    cu_ptr,  # [N + 1] token offsets of the sequences, from 0
    cache_indices_ptr,  # [N] cache line of each sequence
    has_initial_state_ptr,  # [N]
    ckpt_offsets_ptr,  # [N] checkpoint offset in tokens, 0 for none
    ckpt_indices_ptr,  # [N] cache line of each checkpoint
    T,
    N,
    stride_x_tok,
    stride_o_tok,
    stride_state_seq,
    stride_state_dim: tl.constexpr,
    stride_state_tok: tl.constexpr,
    DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NULL_BLOCK: tl.constexpr,
    HAS_CKPT: tl.constexpr,
):
    t0 = tl.program_id(0).to(tl.int64) * BLOCK_M
    # tokens present: the last offset (rows beyond it are padding)
    T = min(T, tl.load(cu_ptr + N))
    if t0 >= T:
        return
    feats = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    m_f = feats < DIM

    # sequence of the first token: the last one starting at or before it (so it is not
    # empty)
    lo = N * 0
    hi = N + 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if tl.load(cu_ptr + mid).to(tl.int64) <= t0:
            lo = mid
        else:
            hi = mid
    seq = lo
    seq_start = tl.load(cu_ptr + seq).to(tl.int64)
    seq_end = tl.load(cu_ptr + seq + 1).to(tl.int64)

    w0 = tl.load(w_ptr + feats * 4 + 0, mask=m_f, other=0.0)
    w1 = tl.load(w_ptr + feats * 4 + 1, mask=m_f, other=0.0)
    w2 = tl.load(w_ptr + feats * 4 + 2, mask=m_f, other=0.0)
    w3 = tl.load(w_ptr + feats * 4 + 3, mask=m_f, other=0.0)

    rows = t0 + tl.arange(0, BLOCK_M)  # [BLOCK_M] token index
    start = tl.zeros([BLOCK_M], dtype=tl.int64) + seq_start  # each row's sequence start

    # ---- sequences starting in this block ----
    block_end = min(t0 + BLOCK_M, T)
    t = t0
    if seq_start < t0:
        t = seq_end  # next sequence start (>= t0 + 1)
        seq += 1
    while t < block_end:
        # `seq` starts at token t; skip empty sequences
        seq_end = tl.load(cu_ptr + seq + 1).to(tl.int64)
        while seq_end <= t:
            seq += 1
            seq_end = tl.load(cu_ptr + seq + 1).to(tl.int64)
        start = tl.where(rows >= t, t, start)
        seq_len = seq_end - t
        slot = tl.load(cache_indices_ptr + seq).to(tl.int64)
        state_base = state_ptr + slot * stride_state_seq + feats * stride_state_dim
        c0 = tl.zeros([BLOCK_N], dtype=x_ptr.dtype.element_ty)
        c1 = tl.zeros([BLOCK_N], dtype=x_ptr.dtype.element_ty)
        c2 = tl.zeros([BLOCK_N], dtype=x_ptr.dtype.element_ty)
        if (slot != NULL_BLOCK) & tl.load(has_initial_state_ptr + seq).to(tl.int1):
            c0 = tl.load(state_base, mask=m_f, other=0.0)
            c1 = tl.load(state_base + stride_state_tok, mask=m_f, other=0.0)
            c2 = tl.load(state_base + 2 * stride_state_tok, mask=m_f, other=0.0)
        for j in tl.static_range(3):
            if j < seq_len:
                xj = tl.load(
                    x_ptr + (t + j) * stride_x_tok + feats, mask=m_f, other=0.0
                )
                acc = tl.zeros([BLOCK_N], dtype=tl.float32)
                # Match the FP32 contraction order of causal_conv1d_fn.
                if x_ptr.dtype.element_ty == tl.float32:
                    acc = tl.fma(c0, w0, acc)
                    acc = tl.fma(c1, w1, acc)
                    acc = tl.fma(c2, w2, acc)
                    acc = tl.fma(xj, w3, acc)
                else:
                    acc += c0 * w0
                    acc += c1 * w1
                    acc += c2 * w2
                    acc += xj * w3
                acc = acc / (1 + tl.exp(-acc))
                tl.store(o_ptr + (t + j) * stride_o_tok + feats, acc, mask=m_f)
                c0 = c1
                c1 = c2
                c2 = xj
        if slot != NULL_BLOCK:
            # new window: the last 3 of (old window ++ inputs)
            if seq_len >= 3:
                c0 = tl.load(
                    x_ptr + (seq_end - 3) * stride_x_tok + feats, mask=m_f, other=0.0
                )
                c1 = tl.load(
                    x_ptr + (seq_end - 2) * stride_x_tok + feats, mask=m_f, other=0.0
                )
                c2 = tl.load(
                    x_ptr + (seq_end - 1) * stride_x_tok + feats, mask=m_f, other=0.0
                )
            tl.store(state_base, c0, mask=m_f)
            tl.store(state_base + stride_state_tok, c1, mask=m_f)
            tl.store(state_base + 2 * stride_state_tok, c2, mask=m_f)
        if HAS_CKPT:
            ckpt_offset = tl.load(ckpt_offsets_ptr + seq).to(tl.int64)
            ckpt_slot = tl.load(ckpt_indices_ptr + seq).to(tl.int64)
            if (ckpt_slot != NULL_BLOCK) & (ckpt_offset >= 3):
                ckpt_base = (
                    state_ptr + ckpt_slot * stride_state_seq + feats * stride_state_dim
                )
                for j in tl.static_range(3):
                    xj = tl.load(
                        x_ptr + (t + ckpt_offset - 3 + j) * stride_x_tok + feats,
                        mask=m_f,
                        other=0.0,
                    )
                    tl.store(ckpt_base + j * stride_state_tok, xj, mask=m_f)
        t = seq_end
        seq += 1

    # ---- tokens at least 3 into their sequence: taps from the input rows ----
    m_r = (rows < T) & (rows - start >= 3)
    mask = m_r[:, None] & m_f[None, :]
    base = x_ptr + rows[:, None] * stride_x_tok + feats[None, :]
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    x0 = tl.load(base - 3 * stride_x_tok, mask=mask, other=0.0)
    x1 = tl.load(base - 2 * stride_x_tok, mask=mask, other=0.0)
    x2 = tl.load(base - stride_x_tok, mask=mask, other=0.0)
    x3 = tl.load(base, mask=mask, other=0.0)
    if x_ptr.dtype.element_ty == tl.float32:
        acc = tl.fma(x0, w0[None, :], acc)
        acc = tl.fma(x1, w1[None, :], acc)
        acc = tl.fma(x2, w2[None, :], acc)
        acc = tl.fma(x3, w3[None, :], acc)
    else:
        acc += x0 * w0[None, :]
        acc += x1 * w1[None, :]
        acc += x2 * w2[None, :]
        acc += x3 * w3[None, :]
    acc = acc / (1 + tl.exp(-acc))
    tl.store(o_ptr + rows[:, None] * stride_o_tok + feats[None, :], acc, mask=mask)


def causal_conv1d_prefill(
    x: torch.Tensor,
    weight: torch.Tensor,
    conv_states: torch.Tensor,
    query_start_loc: torch.Tensor,
    cache_indices: torch.Tensor,
    has_initial_state: torch.Tensor,
    checkpoint_offsets: torch.Tensor | None = None,
    checkpoint_indices: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    block_m: int = 8,
    block_n: int = 128,
    num_warps: int = 2,
) -> torch.Tensor:
    """SiLU(causal conv1d) of a varlen batch, with conv-state and checkpoint-window
    I/O in
    the cache.

    x: [T, dim] tokens of all sequences (unit column stride, any row stride; no
    copy is
    made). Rows beyond the last
    offset of `query_start_loc` are padding: not read, and their output rows not
    written.
    weight: [dim, 4] contiguous. conv_states: [slots, dim, 3] with any strides,
    updated
    in place.
    query_start_loc: [N + 1] integer token offsets from 0 (empty sequences
    allowed).
    cache_indices: [N] integers; a sequence on the null block gets its outputs
    computed
    from a zero window and
    stores nothing. has_initial_state: [N] bool; False starts from a zero window.
    checkpoint_offsets / checkpoint_indices: [N] integers; where the index is not
    the
    null block and the offset is
    >= 3, the window `x[start + offset - 3 : start + offset]` is stored in that
    cache
    line.
    Returns [T, dim] (`out` when given: unit column stride).
    """
    T, dim = x.shape
    assert weight.shape == (dim, 4) and weight.is_contiguous(), weight.shape
    assert conv_states.shape[1:] == (dim, 3), conv_states.shape
    assert x.dtype == conv_states.dtype == weight.dtype
    assert x.stride(1) == 1
    if out is None:
        out = torch.empty(T, dim, device=x.device, dtype=x.dtype)
    assert out.shape == (T, dim) and out.stride(1) == 1
    if T == 0:
        return out
    # Cache indices can be a strided column of the block table. The kernel uses
    # unit strides for all per-sequence inputs.
    assert (
        query_start_loc.ndim == 1
        and cache_indices.ndim == 1
        and has_initial_state.ndim == 1
    )
    query_start_loc = query_start_loc.contiguous()
    cache_indices = cache_indices.contiguous()
    has_initial_state = has_initial_state.contiguous()
    has_ckpt = checkpoint_offsets is not None
    if checkpoint_offsets is not None:
        assert checkpoint_indices is not None
        checkpoint_offsets = checkpoint_offsets.contiguous()
        checkpoint_indices = checkpoint_indices.contiguous()
    _causal_conv1d_prefill_kernel[(triton.cdiv(T, block_m), triton.cdiv(dim, block_n))](
        x,
        weight,
        conv_states,
        out,
        query_start_loc,
        cache_indices,
        has_initial_state,
        checkpoint_offsets if has_ckpt else cache_indices,
        checkpoint_indices if has_ckpt else cache_indices,
        T,
        query_start_loc.shape[0] - 1,
        x.stride(0),
        out.stride(0),
        stride_state_seq=conv_states.stride(0),
        stride_state_dim=conv_states.stride(1),
        stride_state_tok=conv_states.stride(2),
        DIM=dim,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        NULL_BLOCK=NULL_BLOCK_ID,
        HAS_CKPT=has_ckpt,
        num_warps=num_warps,
    )
    return out
