# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import torch

from vllm.triton_utils import tl, triton

from .index import prepare_chunk_indices, prepare_chunk_offsets
from .op import exp, exp2
from .utils import FLA_CHUNK_SIZE, use_cuda_graph

NUM_WARPS = [2, 4, 8, 16]
# Triton's AMD backend fails to lower this kernel with num_stages=4.
_CHUNK_DELTA_H_NUM_STAGES = [2, 3] if torch.version.hip else [2, 3, 4]

# Autotuning replays the launch, and with a paged state cache every replay would
# advance the rows it reads and writes again; save and restore just those rows.
_tuning_state_rows: list[torch.Tensor] = []


def _save_state_rows(nargs, reset_only=False):
    if reset_only or nargs["state_indices"] is None:
        return
    _tuning_state_rows.append(nargs["ht"][nargs["state_indices"].clamp(min=0)])


def _restore_state_rows(nargs, exception=None):
    if nargs["state_indices"] is None:
        return
    nargs["ht"][nargs["state_indices"].clamp(min=0)] = _tuning_state_rows.pop()


# The grid has one program per (sequence, head, V tile), and each walks its
# sequence's chunks serially: a few long sequences need narrow V tiles to fill
# the GPU, many short ones are better served by wide tiles. Launches on the
# paged state cache are therefore tuned per bucket of sequence count; callers
# warm up one batch per bucket. Other launches keep a single tuning.
PAGED_STATE_TUNING_BUCKETS = (1, 4, 16)


def paged_state_tuning_bucket(num_seqs: int) -> int:
    return 1 if num_seqs <= 2 else 4 if num_seqs <= 8 else 16


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_GK": lambda args: args["gk"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
        "USE_STATE_CACHE": lambda args: args["state_indices"] is not None,
        "STORE_CHECKPOINT": lambda args: args["checkpoint_offsets"] is not None,
        "USE_CHECKPOINT_INDICES": lambda args: (
            args["checkpoint_state_indices"] is not None
        ),
        "NUM_SEGMENTS": lambda args: 2 if args["checkpoint_offsets"] is not None else 1,
    }
)
@triton.autotune(
    configs=[
        triton.Config({"BV": BV}, num_warps=num_warps, num_stages=num_stages)
        for num_warps in [2, 4]
        for num_stages in _CHUNK_DELTA_H_NUM_STAGES
        for BV in [32, 64]
    ],
    key=["H", "K", "V", "BT", "TUNING_BUCKET"],
    pre_hook=_save_state_rows,
    post_hook=_restore_state_rows,
    use_cuda_graph=use_cuda_graph,
)
@triton.jit(
    do_not_specialize=["T", "TUNING_BUCKET"],
    # Metadata uses scalar loads; offset views must reuse warmed kernels.
    do_not_specialize_on_alignment=[
        "cu_seqlens",
        "chunk_offsets",
        "state_indices",
        "has_initial_state",
        "checkpoint_offsets",
        "checkpoint_state_indices",
    ],
)
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
    k,
    v,
    w,
    v_new,
    g,
    gk,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    state_indices,
    has_initial_state,
    checkpoint_state,
    checkpoint_offsets,
    checkpoint_state_indices,
    stride_state,
    stride_checkpoint,
    T,
    TUNING_BUCKET,  # Autotuning key only.
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_GK: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_STATE_CACHE: tl.constexpr,
    STORE_CHECKPOINT: tl.constexpr,
    USE_CHECKPOINT_INDICES: tl.constexpr,
    NUM_SEGMENTS: tl.constexpr,
    USE_EXP2: tl.constexpr,
):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    # [BV, BK]
    b_h1 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 64:
        b_h2 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 128:
        b_h3 = tl.zeros([BV, 64], dtype=tl.float32)
    if K > 192:
        b_h4 = tl.zeros([BV, 64], dtype=tl.float32)

    # calculate offset
    # Cast before multiplying: the products overflow int32 for long batches.
    h += (boh * H + i_h).to(tl.int64) * V * K
    v += (bos * H + i_h).to(tl.int64) * V
    k += (bos * Hg + i_h // (H // Hg)).to(tl.int64) * K
    w += (bos * H + i_h).to(tl.int64) * K
    if SAVE_NEW_VALUE:
        v_new += (bos * H + i_h).to(tl.int64) * V
    stride_v = H * V
    stride_h = H * V * K
    stride_k = Hg * K
    stride_w = H * K

    o_v = i_v * BV + tl.arange(0, BV)
    m_v = o_v < V
    o_k1 = tl.arange(0, 64)
    m_k1 = o_k1 < K
    o_k2 = 64 + o_k1
    m_k2 = o_k2 < K
    o_k3 = 128 + o_k1
    m_k3 = o_k3 < K
    o_k4 = 192 + o_k1
    m_k4 = o_k4 < K
    # Rows of V whose initial state is loaded and whose final state is stored.
    m_v0 = m_v
    m_vt = m_v
    if USE_STATE_CACHE:
        # h0 and ht are both the paged state cache: sequence i_n reads its
        # initial state from row state_indices[i_n] and writes its final state
        # back there. Rows <= 0 (NULL_BLOCK_ID, PAD_SLOT_ID) are never accessed.
        i_s = tl.load(state_indices + i_n).to(tl.int64)
        m_vt = m_v & (i_s > 0)
        m_v0 = m_vt & tl.load(has_initial_state + i_n).to(tl.int1)
        if USE_INITIAL_STATE:
            h0 = h0 + i_s * stride_state + i_h * V * K
        if STORE_FINAL_STATE:
            ht = ht + i_s * stride_state + i_h * V * K
    else:
        if USE_INITIAL_STATE:
            h0 = h0 + i_nh.to(tl.int64) * V * K
        if STORE_FINAL_STATE:
            ht = ht + i_nh.to(tl.int64) * V * K
    i_split = NT
    if STORE_CHECKPOINT:
        # The state at the start of chunk i_t is the state after i_t * BT
        # tokens, so a checkpoint offset on a chunk boundary strictly inside the
        # sequence is snapshotted after the first i_split = offset / BT chunks.
        # Offset 0 means none; any other offset off a chunk boundary or past
        # the sequence, or a destination row <= 0, is skipped.
        o_c = tl.load(checkpoint_offsets + i_n).to(tl.int32)
        m_c = (o_c > 0) & (o_c % BT == 0) & (o_c < T)
        if USE_CHECKPOINT_INDICES:
            i_c = tl.load(checkpoint_state_indices + i_n).to(tl.int64)
            m_c = m_c & (i_c > 0)
        else:
            i_c = i_n.to(tl.int64)
        i_split = tl.where(m_c, o_c // BT, NT)
        checkpoint_state = checkpoint_state + i_c * stride_checkpoint + i_h * V * K

    # load initial state
    if USE_INITIAL_STATE:
        p_h0_1 = h0 + o_v[:, None] * K + o_k1[None, :]
        m_h0_1 = m_v0[:, None] & m_k1[None, :]
        b_h1 += tl.load(p_h0_1, mask=m_h0_1, other=0.0).to(tl.float32)
        if K > 64:
            p_h0_2 = h0 + o_v[:, None] * K + o_k2[None, :]
            m_h0_2 = m_v0[:, None] & m_k2[None, :]
            b_h2 += tl.load(p_h0_2, mask=m_h0_2, other=0.0).to(tl.float32)
        if K > 128:
            p_h0_3 = h0 + o_v[:, None] * K + o_k3[None, :]
            m_h0_3 = m_v0[:, None] & m_k3[None, :]
            b_h3 += tl.load(p_h0_3, mask=m_h0_3, other=0.0).to(tl.float32)
        if K > 192:
            p_h0_4 = h0 + o_v[:, None] * K + o_k4[None, :]
            m_h0_4 = m_v0[:, None] & m_k4[None, :]
            b_h4 += tl.load(p_h0_4, mask=m_h0_4, other=0.0).to(tl.float32)

    # main recurrence, split at the checkpoint chunk so that its store sits
    # between two loops: inside the loop it would cost registers on every
    # iteration
    for i_seg in tl.static_range(NUM_SEGMENTS):
        if i_seg == 0:
            t_begin = 0
            t_end = i_split
        else:
            t_begin = i_split
            t_end = NT
        for i_t in range(t_begin, t_end):
            o_t = (i_t * BT + tl.arange(0, BT)).to(tl.int64)
            m_t = o_t < T
            p_h1 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k1[None, :]
            m_h1 = m_v[:, None] & m_k1[None, :]
            tl.store(p_h1, b_h1.to(p_h1.dtype.element_ty), mask=m_h1)
            if K > 64:
                p_h2 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k2[None, :]
                m_h2 = m_v[:, None] & m_k2[None, :]
                tl.store(p_h2, b_h2.to(p_h2.dtype.element_ty), mask=m_h2)
            if K > 128:
                p_h3 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k3[None, :]
                m_h3 = m_v[:, None] & m_k3[None, :]
                tl.store(p_h3, b_h3.to(p_h3.dtype.element_ty), mask=m_h3)
            if K > 192:
                p_h4 = h + i_t.to(tl.int64) * stride_h + o_v[:, None] * K + o_k4[None, :]
                m_h4 = m_v[:, None] & m_k4[None, :]
                tl.store(p_h4, b_h4.to(p_h4.dtype.element_ty), mask=m_h4)
            p_w = w + o_t[:, None] * stride_w + o_k1[None, :]
            b_w = tl.load(p_w, mask=m_t[:, None] & m_k1[None, :], other=0.0)
            b_v = tl.dot(b_w, tl.trans(b_h1).to(b_w.dtype))
            if K > 64:
                p_w = w + o_t[:, None] * stride_w + o_k2[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k2[None, :], other=0.0)
                b_v += tl.dot(b_w, tl.trans(b_h2).to(b_w.dtype))
            if K > 128:
                p_w = w + o_t[:, None] * stride_w + o_k3[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k3[None, :], other=0.0)
                b_v += tl.dot(b_w, tl.trans(b_h3).to(b_w.dtype))
            if K > 192:
                p_w = w + o_t[:, None] * stride_w + o_k4[None, :]
                b_w = tl.load(p_w, mask=m_t[:, None] & m_k4[None, :], other=0.0)
                b_v += tl.dot(b_w, tl.trans(b_h4).to(b_w.dtype))
            p_v = v + o_t[:, None] * stride_v + o_v[None, :]
            b_v = tl.load(p_v, mask=m_t[:, None] & m_v[None, :], other=0.0) - b_v

            if SAVE_NEW_VALUE:
                p_v = v_new + o_t[:, None] * stride_v + o_v[None, :]
                tl.store(
                    p_v, b_v.to(p_v.dtype.element_ty), mask=m_t[:, None] & m_v[None, :]
                )

            last_idx = min((i_t.to(tl.int64) + 1) * BT, T) - 1
            if USE_G:
                b_g_last = tl.load(g + bos * H + last_idx * H + i_h)
                p_g = g + bos * H + i_h + o_t * H
                b_g = tl.load(p_g, mask=m_t, other=0.0)
                if USE_EXP2:
                    b_v = b_v * tl.where(m_t, exp2(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp2(b_g_last)
                else:
                    b_v = b_v * tl.where(m_t, exp(b_g_last - b_g), 0)[:, None]
                    b_g_last = exp(b_g_last)
                b_h1 *= b_g_last
                if K > 64:
                    b_h2 *= b_g_last
                if K > 128:
                    b_h3 *= b_g_last
                if K > 192:
                    b_h4 *= b_g_last

            if USE_GK:
                o_k1 = tl.arange(0, 64)
                b_gk_last1 = tl.load(
                    gk + (bos + last_idx) * H * K + i_h * K + o_k1,
                    mask=(o_k1 < K),
                    other=0.0,
                )
                if USE_EXP2:
                    b_h1 *= exp2(b_gk_last1)[None, :]
                else:
                    b_h1 *= exp(b_gk_last1)[None, :]
                if K > 64:
                    b_gk_last2 = tl.load(
                        gk + (bos + last_idx) * H * K + i_h * K + o_k2,
                        mask=(o_k2 < K),
                        other=0.0,
                    )
                    if USE_EXP2:
                        b_h2 *= exp2(b_gk_last2)[None, :]
                    else:
                        b_h2 *= exp(b_gk_last2)[None, :]
                if K > 128:
                    b_gk_last3 = tl.load(
                        gk + (bos + last_idx) * H * K + i_h * K + o_k3,
                        mask=(o_k3 < K),
                        other=0.0,
                    )
                    if USE_EXP2:
                        b_h3 *= exp2(b_gk_last3)[None, :]
                    else:
                        b_h3 *= exp(b_gk_last3)[None, :]
                if K > 192:
                    b_gk_last4 = tl.load(
                        gk + (bos + last_idx) * H * K + i_h * K + o_k4,
                        mask=(o_k4 < K),
                        other=0.0,
                    )
                    if USE_EXP2:
                        b_h4 *= exp2(b_gk_last4)[None, :]
                    else:
                        b_h4 *= exp(b_gk_last4)[None, :]
            b_v = b_v.to(k.dtype.element_ty)

            p_k = k + o_k1[:, None] + o_t[None, :] * stride_k
            b_k = tl.load(p_k, mask=m_k1[:, None] & m_t[None, :], other=0.0)
            b_h1 += tl.trans(tl.dot(b_k, b_v))
            if K > 64:
                p_k = k + o_k2[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k2[:, None] & m_t[None, :], other=0.0)
                b_h2 += tl.trans(tl.dot(b_k, b_v))
            if K > 128:
                p_k = k + o_k3[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k3[:, None] & m_t[None, :], other=0.0)
                b_h3 += tl.trans(tl.dot(b_k, b_v))
            if K > 192:
                p_k = k + o_k4[:, None] + o_t[None, :] * stride_k
                b_k = tl.load(p_k, mask=m_k4[:, None] & m_t[None, :], other=0.0)
                b_h4 += tl.trans(tl.dot(b_k, b_v))
        if STORE_CHECKPOINT:
            if i_seg == 0:
                if m_c:
                    # Unlike h, the snapshot keeps the fp32 accumulator's
                    # precision up to the destination dtype.
                    p_c = checkpoint_state + o_v[:, None] * K + o_k1[None, :]
                    m_hc = m_v[:, None] & m_k1[None, :]
                    tl.store(p_c, b_h1.to(p_c.dtype.element_ty), mask=m_hc)
                    if K > 64:
                        p_c = checkpoint_state + o_v[:, None] * K + o_k2[None, :]
                        m_hc = m_v[:, None] & m_k2[None, :]
                        tl.store(p_c, b_h2.to(p_c.dtype.element_ty), mask=m_hc)
                    if K > 128:
                        p_c = checkpoint_state + o_v[:, None] * K + o_k3[None, :]
                        m_hc = m_v[:, None] & m_k3[None, :]
                        tl.store(p_c, b_h3.to(p_c.dtype.element_ty), mask=m_hc)
                    if K > 192:
                        p_c = checkpoint_state + o_v[:, None] * K + o_k4[None, :]
                        m_hc = m_v[:, None] & m_k4[None, :]
                        tl.store(p_c, b_h4.to(p_c.dtype.element_ty), mask=m_hc)
    # epilogue
    if STORE_FINAL_STATE:
        p_ht = ht + o_v[:, None] * K + o_k1[None, :]
        m_ht = m_vt[:, None] & m_k1[None, :]
        tl.store(p_ht, b_h1.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 64:
            p_ht = ht + o_v[:, None] * K + o_k2[None, :]
            m_ht = m_vt[:, None] & m_k2[None, :]
            tl.store(p_ht, b_h2.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 128:
            p_ht = ht + o_v[:, None] * K + o_k3[None, :]
            m_ht = m_vt[:, None] & m_k3[None, :]
            tl.store(p_ht, b_h3.to(p_ht.dtype.element_ty), mask=m_ht)
        if K > 192:
            p_ht = ht + o_v[:, None] * K + o_k4[None, :]
            m_ht = m_vt[:, None] & m_k4[None, :]
            tl.store(p_ht, b_h4.to(p_ht.dtype.element_ty), mask=m_ht)


def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    gk: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = FLA_CHUNK_SIZE,
    save_new_value: bool = True,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
    use_exp2: bool = False,
    state_cache: torch.Tensor | None = None,
    state_indices: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    checkpoint_state: torch.Tensor | None = None,
    checkpoint_offsets: torch.Tensor | None = None,
    checkpoint_state_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Run the chunk-state recurrence.

    Args:
        state_cache: Paged recurrent state, ``[slots, H, V, K]`` with each row
            contiguous. Sequence ``n`` reads its initial state from row
            ``state_indices[n]`` and writes its final state back there, which
            replaces gathering ``initial_state`` and scattering the final
            state. Rows ``<= 0`` are never read or written. Replaces
            ``initial_state``/``output_final_state``; the returned final state
            is then ``None``.
        state_indices: ``[N]`` cache row per sequence.
        has_initial_state: ``[N]`` bool; false starts that sequence from a zero
            state.
        checkpoint_state: Destination of the mid-sequence state snapshots,
            ``[rows, H, V, K]`` with each row contiguous. Indexed by sequence
            without ``checkpoint_state_indices``; with them it can be the paged
            state cache itself.
        checkpoint_offsets: ``[N]`` int32 token offset, relative to each
            sequence's start, of the state to snapshot; ``0`` for none. A
            snapshot is stored only for a multiple of ``chunk_size`` below the
            sequence length, so callers must not rely on any other offset.
        checkpoint_state_indices: ``[N]`` destination row per sequence; rows
            ``<= 0`` skip the snapshot.

    Returns:
        The per-chunk states, the new values if ``save_new_value``, and the
        final state if ``output_final_state``.

    """
    # This kernel is slightly different from fla to support Q/K with different head numbers.
    # In fla, Q/K always have the same head number, so Hg is always equal to H.
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    # N: the actual number of sequences in the batch with either equal or variable lengths
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT = len(cu_seqlens) - 1, len(chunk_indices)
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    if state_cache is not None:
        if initial_state is not None or output_final_state:
            raise ValueError("state_cache replaces initial_state/output_final_state")
        if state_indices is None or has_initial_state is None:
            raise ValueError("state_cache needs state_indices and has_initial_state")
        _check_state_rows("state_cache", state_cache, H, V, K)
        state_indices = _per_sequence("state_indices", state_indices, N)
        has_initial_state = _per_sequence("has_initial_state", has_initial_state, N)
    elif state_indices is not None or has_initial_state is not None:
        raise ValueError("state_indices and has_initial_state need state_cache")
    if (checkpoint_state is None) != (checkpoint_offsets is None):
        raise ValueError("checkpoint_state and checkpoint_offsets go together")
    if checkpoint_state is not None:
        _check_state_rows("checkpoint_state", checkpoint_state, H, V, K)
        checkpoint_offsets = _per_sequence("checkpoint_offsets", checkpoint_offsets, N)
        if checkpoint_state_indices is not None:
            checkpoint_state_indices = _per_sequence(
                "checkpoint_state_indices", checkpoint_state_indices, N
            )
        elif checkpoint_state.shape[0] < N:
            raise ValueError(
                f"checkpoint_state has {checkpoint_state.shape[0]} rows for {N} "
                "sequences"
            )
    elif checkpoint_state_indices is not None:
        raise ValueError("checkpoint_state_indices needs checkpoint_state")

    h = k.new_empty(B, NT, H, V, K)
    if state_cache is not None:
        h0 = ht = state_cache
        final_state = None
    else:
        h0 = initial_state
        final_state = (
            k.new_empty(N, H, V, K, dtype=torch.float32) if output_final_state else None
        )
        ht = final_state

    v_new = torch.empty_like(u) if save_new_value else None

    def grid(meta):
        return (triton.cdiv(V, meta["BV"]), N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        gk=gk,
        h=h,
        h0=h0,
        ht=ht,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        state_indices=state_indices,
        has_initial_state=has_initial_state,
        checkpoint_state=checkpoint_state,
        checkpoint_offsets=checkpoint_offsets,
        checkpoint_state_indices=checkpoint_state_indices,
        stride_state=state_cache.stride(0) if state_cache is not None else 0,
        stride_checkpoint=(
            checkpoint_state.stride(0) if checkpoint_state is not None else 0
        ),
        T=T,
        TUNING_BUCKET=(
            paged_state_tuning_bucket(N) if state_cache is not None else 0
        ),
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        USE_EXP2=use_exp2,
    )
    return h, v_new, final_state


def _check_state_rows(
    name: str, state: torch.Tensor, H: int, V: int, K: int
) -> None:
    # Rows may be strided apart (a paged cache view), but each must be dense.
    if state.shape[1:] != (H, V, K) or state.stride()[1:] != (V * K, K, 1):
        raise ValueError(
            f"{name} must have contiguous [{H}, {V}, {K}] rows, got shape "
            f"{tuple(state.shape)} and strides {state.stride()}"
        )


def _per_sequence(name: str, tensor: torch.Tensor, N: int) -> torch.Tensor:
    if tensor.shape != (N,):
        raise ValueError(f"{name} must have shape [{N}], got {tuple(tensor.shape)}")
    # Read densely by the kernel; a strided view (a block-table column) is copied.
    return tensor.contiguous()
