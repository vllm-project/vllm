# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM Mamba-2 decode kernels.

A decode step splits the rows by ReplaySSM's flush flag. Non-flush rows
(``sketch_decode``) read their sketch plus the window ring:

    y = decay · U^T c(R C) + Σ_t s_t (B_t · C) x_t + D x.

Flush rows (``sketch_flush``) fold the window into the rotated state as
ReplaySSM does, with the FP32 rotated keys ``R B_t``, read it, and rebuild
their sketches. The two touch disjoint rows, slots and sketches, so they may
run concurrently. B and C arrive unrotated; the ring keeps the BF16 B and
``bc_pre`` holds the rotation-invariant ``B_t · C``.
"""

import torch

from vllm.model_executor.layers.mamba.ops.mamba_ssm import softplus
from vllm.model_executor.layers.mamba.ops.selective_state_update_replayssm_output_only import (  # noqa: E501
    _replayssm_output_only_precompute_kernel,
)
from vllm.model_executor.layers.mamba.ops.sketchssm_config import get_sketchssm_config
from vllm.model_executor.layers.mamba.ops.sketchssm_mamba2_build import (
    SKETCH_DOT_PRECISION,
    SKETCH_PIVOTS,
    SketchArgs,
    _sketch_head,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

_PIVOTS = tl.constexpr(SKETCH_PIVOTS)
_DOT_PRECISION = tl.constexpr(SKETCH_DOT_PRECISION)


@triton.jit
def _window(
    dt_cache_ptr,
    x_cache_ptr,
    dt_cur,
    x_cur,
    A,
    wp,
    offs_m,
    dim,
    stride_dt_cache_pos,
    stride_x_cache_pos,
    stride_x_cache_dim,
    BLOCK_K: tl.constexpr,
):
    # Window weights s_t = dt_t exp(A (sum dt - cumsum dt)_t) and values x_t.
    offs_k = tl.arange(0, BLOCK_K)
    dt_all = tl.load(
        dt_cache_ptr + offs_k * stride_dt_cache_pos, mask=offs_k < wp, other=0.0
    ).to(tl.float32)
    dt_all = tl.where(offs_k == wp, dt_cur, dt_all)
    total = A * tl.sum(dt_all, axis=0)
    scale = dt_all * tl.exp(total - A * tl.cumsum(dt_all, axis=0))
    scale = tl.where(offs_k <= wp, scale, 0.0)
    x_all = tl.load(
        x_cache_ptr
        + offs_m[:, None] * stride_x_cache_dim
        + offs_k[None, :] * stride_x_cache_pos,
        mask=(offs_m[:, None] < dim) & (offs_k[None, :] < wp),
        other=0.0,
    )
    x_all = tl.where(offs_k[None, :] == wp, x_cur[:, None], x_all).to(tl.float32)
    return tl.exp(total), scale, x_all


@triton.jit(do_not_specialize=["U_ROWS", "W_ROWS", "AG_COLS"])
def _sketch_decode_kernel(
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    query_ptr,
    D_ptr,
    out_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    bc_pre_ptr,
    write_pos_ptr,
    is_flush_ptr,
    state_indices_ptr,
    sketch_indices_ptr,
    u_ptr,
    w_ptr,
    ag_ptr,
    ranks_ptr,
    u_offsets_ptr,
    w_offsets_ptr,
    ag_offsets_ptr,
    null_block_id,
    dim,
    dstate,
    heads_per_group,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_query_batch,
    stride_query_group,
    stride_query_dstate,
    stride_D_head,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_x_cache_slot,
    stride_x_cache_head,
    stride_x_cache_pos,
    stride_x_cache_dim,
    stride_dt_cache_slot,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_slot,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    stride_bc_pre_batch,
    stride_bc_pre_group,
    stride_bc_pre_pos,
    U_ROWS,
    W_ROWS,
    AG_COLS,
    HAS_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_J: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)
    slot = tl.load(state_indices_ptr + pid_b).to(tl.int64)
    if slot == null_block_id:
        return
    if tl.load(is_flush_ptr + pid_b) != 0:
        return
    group = pid_h // heads_per_group
    wp = tl.load(write_pos_ptr + pid_b).to(tl.int32)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    mmask = offs_m < dim
    nmask = offs_n < dstate

    x_ptr += pid_b * stride_x_batch + pid_h * stride_x_head
    B_ptr += pid_b * stride_B_batch + group * stride_B_group
    query_ptr += pid_b * stride_query_batch + group * stride_query_group
    x_cache_ptr += slot * stride_x_cache_slot + pid_h * stride_x_cache_head
    dt_cache_ptr += slot * stride_dt_cache_slot + pid_h * stride_dt_cache_head
    B_cache_ptr += slot * stride_B_cache_slot + group * stride_B_cache_group

    dt_cur = tl.load(dt_ptr + pid_b * stride_dt_batch + pid_h * stride_dt_head)
    dt_cur = dt_cur.to(tl.float32)
    dt_cur += tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
    dt_cur = tl.where(dt_cur <= 20.0, softplus(dt_cur), dt_cur)
    A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
    x_cur = tl.load(x_ptr + offs_m * stride_x_dim, mask=mmask, other=0.0)
    query = tl.load(query_ptr + offs_n * stride_query_dstate, mask=nmask, other=0.0)
    decay, scale, x_all = _window(
        dt_cache_ptr,
        x_cache_ptr,
        dt_cur,
        x_cur,
        A,
        wp,
        offs_m,
        dim,
        stride_dt_cache_pos,
        stride_x_cache_pos,
        stride_x_cache_dim,
        BLOCK_K,
    )

    offs_k = tl.arange(0, BLOCK_K)
    bc = tl.load(
        bc_pre_ptr
        + pid_b * stride_bc_pre_batch
        + group * stride_bc_pre_group
        + offs_k * stride_bc_pre_pos,
        mask=offs_k <= wp,
        other=0.0,
    )
    y = tl.sum(x_all * (scale * bc)[None, :], axis=1)

    # Sketch read U^T c with c_j = a_j q_j + Σ_p g_pj (w_p · q); dense: c = q.
    sketch_index = tl.load(sketch_indices_ptr + pid_b).to(tl.int64)
    m = tl.load(ranks_ptr + pid_h)
    u_rows = tl.where(m == 0, dstate, m)
    u_base = u_ptr + (sketch_index * U_ROWS + tl.load(u_offsets_ptr + pid_h)) * dim
    w_base = w_ptr + (sketch_index * W_ROWS + tl.load(w_offsets_ptr + pid_h)) * dstate
    ag_base = ag_ptr + sketch_index * (_PIVOTS + 1) * AG_COLS
    ag_base += tl.load(ag_offsets_ptr + pid_h)
    offs_p = tl.arange(0, _PIVOTS)
    w_rows = tl.load(
        w_base + offs_p[:, None] * dstate + offs_n[None, :],
        mask=(offs_p[:, None] < tl.minimum(m, _PIVOTS)) & nmask[None, :],
        other=0.0,
    )
    dots = tl.sum(w_rows.to(tl.float32) * query[None, :], axis=1)
    read = tl.zeros([BLOCK_M], dtype=tl.float32)
    for j0 in range(0, u_rows, BLOCK_J):
        offs_j = j0 + tl.arange(0, BLOCK_J)
        jmask = offs_j < u_rows
        q_j = tl.load(query_ptr + offs_j * stride_query_dstate, mask=jmask, other=0.0)
        if m == 0:
            coef = q_j
        elif m <= _PIVOTS:
            coef = tl.sum(
                tl.where(offs_p[None, :] == offs_j[:, None], dots[None, :], 0.0),
                axis=1,
            )
        else:
            a_j = tl.load(ag_base + offs_j, mask=jmask, other=0.0)
            g_j = tl.load(
                ag_base + (1 + offs_p[None, :]) * AG_COLS + offs_j[:, None],
                mask=jmask[:, None],
                other=0.0,
            )
            coef = a_j * q_j + tl.sum(g_j * dots[None, :], axis=1)
        u_tile = tl.load(
            u_base + offs_j[:, None] * dim + offs_m[None, :],
            mask=jmask[:, None] & mmask[None, :],
            other=0.0,
        )
        read += tl.sum(u_tile.to(tl.float32) * coef[:, None], axis=0)
    y += decay * read

    if HAS_D:
        D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
        y += x_cur.to(tl.float32) * D
    out_ptr += pid_b * stride_out_batch + pid_h * stride_out_head
    tl.store(out_ptr + offs_m * stride_out_dim, y, mask=mmask)
    # Append the current token to the ring.
    tl.store(
        x_cache_ptr + wp * stride_x_cache_pos + offs_m * stride_x_cache_dim,
        x_cur,
        mask=mmask,
    )
    if pid_m == 0:
        tl.store(dt_cache_ptr + wp * stride_dt_cache_pos, dt_cur)
        if pid_h % heads_per_group == 0:
            B_cur = tl.load(B_ptr + offs_n * stride_B_dstate, mask=nmask)
            tl.store(
                B_cache_ptr + wp * stride_B_cache_pos + offs_n * stride_B_cache_dstate,
                B_cur,
                mask=nmask,
            )


@triton.jit(do_not_specialize=["batch"])
def _window_keys_kernel(
    B_ptr,
    B_cache_ptr,
    frames_t_ptr,
    keys_ptr,
    write_pos_ptr,
    flush_rows_ptr,
    state_indices_ptr,
    null_block_id,
    batch,
    stride_B_batch,
    stride_B_group,
    stride_B_cache_slot,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_keys_batch,
    stride_keys_group,
    stride_keys_pos,
    W: tl.constexpr,
    N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # keys[row, g, t, cols] = B_t @ R_g^T[:, cols] for t <= write_pos (ring
    # rows, then the current B) of each flush row.
    pid_g = tl.program_id(1)
    offs_n = tl.arange(0, N)
    offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    r = tl.load(frames_t_ptr + pid_g * N * N + offs_n[:, None] * N + offs_c[None, :])
    it = tl.program_id(0)
    row = tl.load(flush_rows_ptr + it, mask=it < batch, other=-1)
    while row >= 0:
        slot = tl.load(state_indices_ptr + row).to(tl.int64)
        if slot != null_block_id:
            wp = tl.load(write_pos_ptr + row).to(tl.int32)
            ring = B_cache_ptr + slot * stride_B_cache_slot
            ring += pid_g * stride_B_cache_group
            keys = keys_ptr + row.to(tl.int64) * stride_keys_batch
            keys += pid_g * stride_keys_group
            B_row = B_ptr + row.to(tl.int64) * stride_B_batch
            B_cur = tl.load(B_row + pid_g * stride_B_group + offs_n)
            for t0 in tl.static_range(0, W, BLOCK_T):
                if t0 <= wp:
                    offs_t = t0 + tl.arange(0, BLOCK_T)
                    B_all = tl.load(
                        ring + offs_t[:, None] * stride_B_cache_pos + offs_n[None, :],
                        mask=offs_t[:, None] < wp,
                        other=0.0,
                    )
                    B_all = tl.where(offs_t[:, None] == wp, B_cur[None, :], B_all)
                    y = tl.dot(B_all.to(tl.float32), r, input_precision=_DOT_PRECISION)
                    tl.store(
                        keys + offs_t[:, None] * stride_keys_pos + offs_c[None, :],
                        y,
                        mask=offs_t[:, None] <= wp,
                    )
        it += tl.num_programs(0)
        row = tl.load(flush_rows_ptr + it, mask=it < batch, other=-1)


@triton.jit(do_not_specialize=["batch", "U_ROWS", "W_ROWS", "AG_COLS"])
def _sketch_flush_kernel(
    state_ptr,
    x_ptr,
    dt_ptr,
    dt_bias_ptr,
    A_ptr,
    B_ptr,
    query_ptr,
    D_ptr,
    out_ptr,
    x_cache_ptr,
    dt_cache_ptr,
    B_cache_ptr,
    keys_ptr,
    write_pos_ptr,
    flush_rows_ptr,
    state_indices_ptr,
    sketch_indices_ptr,
    u_ptr,
    w_ptr,
    ag_ptr,
    ranks_ptr,
    u_offsets_ptr,
    w_offsets_ptr,
    ag_offsets_ptr,
    null_block_id,
    batch,
    heads_per_group,
    stride_keys_batch,
    stride_keys_group,
    stride_keys_pos,
    stride_state_slot,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    stride_x_batch,
    stride_x_head,
    stride_x_dim,
    stride_dt_batch,
    stride_dt_head,
    stride_dt_bias_head,
    stride_A_head,
    stride_B_batch,
    stride_B_group,
    stride_B_dstate,
    stride_query_batch,
    stride_query_group,
    stride_query_dstate,
    stride_D_head,
    stride_out_batch,
    stride_out_head,
    stride_out_dim,
    stride_x_cache_slot,
    stride_x_cache_head,
    stride_x_cache_pos,
    stride_x_cache_dim,
    stride_dt_cache_slot,
    stride_dt_cache_head,
    stride_dt_cache_pos,
    stride_B_cache_slot,
    stride_B_cache_group,
    stride_B_cache_pos,
    stride_B_cache_dstate,
    DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    U_ROWS,
    W_ROWS,
    AG_COLS,
    HAS_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_V: tl.constexpr,
):
    # For each -1 padded flush row, one program per head folds the window into
    # the state, S = decay S_0 + Σ_t s_t x_t (R B_t)^T with R B_t from
    # keys_ptr, reads y = S (R C) + D x, and rebuilds the head's sketch from
    # the new state while it is still in L2.
    pid_h = tl.program_id(1)
    group = pid_h // heads_per_group
    offs_n = tl.arange(0, DSTATE)
    offs_k = tl.arange(0, BLOCK_K)
    it = tl.program_id(0)
    row = tl.load(flush_rows_ptr + it, mask=it < batch, other=-1)
    while row >= 0:
        slot = tl.load(state_indices_ptr + row).to(tl.int64)
        if slot != null_block_id:
            wp = tl.load(write_pos_ptr + row).to(tl.int32)
            x_ring = x_cache_ptr + slot * stride_x_cache_slot
            x_ring += pid_h * stride_x_cache_head
            dt_ring = dt_cache_ptr + slot * stride_dt_cache_slot
            dt_ring += pid_h * stride_dt_cache_head
            dt_cur = tl.load(dt_ptr + row * stride_dt_batch + pid_h * stride_dt_head)
            dt_cur = dt_cur.to(tl.float32)
            dt_cur += tl.load(dt_bias_ptr + pid_h * stride_dt_bias_head).to(tl.float32)
            dt_cur = tl.where(dt_cur <= 20.0, softplus(dt_cur), dt_cur)
            A = tl.load(A_ptr + pid_h * stride_A_head).to(tl.float32)
            query_row = query_ptr + row * stride_query_batch
            query_row += group * stride_query_group
            query = tl.load(query_row + offs_n * stride_query_dstate)
            keys = tl.load(
                keys_ptr
                + row.to(tl.int64) * stride_keys_batch
                + group * stride_keys_group
                + offs_k[:, None] * stride_keys_pos
                + offs_n[None, :],
                mask=offs_k[:, None] <= wp,
                other=0.0,
            )
            head_state = state_ptr + slot * stride_state_slot
            head_state += pid_h * stride_state_head
            x_row = x_ptr + row * stride_x_batch + pid_h * stride_x_head
            out_row = out_ptr + row * stride_out_batch + pid_h * stride_out_head
            for m0 in tl.static_range(0, DIM, BLOCK_M):
                offs_m = m0 + tl.arange(0, BLOCK_M)
                mmask = offs_m < DIM
                x_cur = tl.load(x_row + offs_m * stride_x_dim, mask=mmask, other=0.0)
                decay, scale, x_all = _window(
                    dt_ring,
                    x_ring,
                    dt_cur,
                    x_cur,
                    A,
                    wp,
                    offs_m,
                    DIM,
                    stride_dt_cache_pos,
                    stride_x_cache_pos,
                    stride_x_cache_dim,
                    BLOCK_K,
                )
                s_ptrs = (
                    head_state
                    + offs_m[:, None] * stride_state_dim
                    + offs_n[None, :] * stride_state_dstate
                )
                # Bypass L1 so the sketch build below reads the new state.
                s = tl.load(
                    s_ptrs, mask=mmask[:, None], other=0.0, cache_modifier=".cg"
                )
                s = s * decay + tl.dot(
                    x_all * scale[None, :], keys, input_precision=_DOT_PRECISION
                )
                tl.store(s_ptrs, s, mask=mmask[:, None])
                y = tl.sum(s * query[None, :], axis=1)
                if HAS_D:
                    D = tl.load(D_ptr + pid_h * stride_D_head).to(tl.float32)
                    y += x_cur.to(tl.float32) * D
                tl.store(out_row + offs_m * stride_out_dim, y, mask=mmask)
                # Append the current token to the ring.
                tl.store(
                    x_ring + wp * stride_x_cache_pos + offs_m * stride_x_cache_dim,
                    x_cur,
                    mask=mmask,
                )
            tl.store(dt_ring + wp * stride_dt_cache_pos, dt_cur)
            if pid_h % heads_per_group == 0:
                B_row = B_ptr + row * stride_B_batch + group * stride_B_group
                B_ring = B_cache_ptr + slot * stride_B_cache_slot
                B_ring += group * stride_B_cache_group
                tl.store(
                    B_ring + wp * stride_B_cache_pos + offs_n * stride_B_cache_dstate,
                    tl.load(B_row + offs_n * stride_B_dstate),
                )
            tl.debug_barrier()
            _sketch_head(
                head_state,
                tl.load(sketch_indices_ptr + row),
                pid_h,
                u_ptr,
                w_ptr,
                ag_ptr,
                ranks_ptr,
                u_offsets_ptr,
                w_offsets_ptr,
                ag_offsets_ptr,
                stride_state_dim,
                stride_state_dstate,
                DIM,
                DSTATE,
                U_ROWS,
                W_ROWS,
                AG_COLS,
                BLOCK_V,
            )
        it += tl.num_programs(0)
        row = tl.load(flush_rows_ptr + it, mask=it < batch, other=-1)


def _check_inputs(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    query: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    write_pos: torch.Tensor,
    state_indices: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _, heads, dim, dstate = state.shape
    batch, groups = x.shape[0], B.shape[1]
    window = x_cache.shape[2]
    assert x.shape == out.shape == (batch, heads, dim)
    assert heads % groups == 0
    assert B.shape == (batch, groups, dstate) and B.stride(2) == 1
    assert query.shape == B.shape and query.dtype == torch.float32
    # dt, A, D and dt_bias are scalars per head broadcast over (dim, dstate).
    assert dt.shape == x.shape and dt.stride(2) == 0
    assert A.shape == (heads, dim, dstate) and A.stride(1) == A.stride(2) == 0
    assert dt_bias.shape == (heads, dim) and dt_bias.stride(1) == 0
    assert D is None or (D.shape == (heads, dim) and D.stride(1) == 0)
    assert x_cache.shape[1:] == (heads, window, dim)
    assert dt_cache.shape[1:] == (heads, window)
    assert B_cache.shape[1:] == (groups, window, dstate) and B_cache.stride(3) == 1
    assert write_pos.shape[0] >= batch and write_pos.dtype == torch.int32
    assert state_indices.shape == (batch,) and state_indices.stride(0) == 1


def _common_strides(x, dt, dt_bias, A, B, query, D, out, x_cache, dt_cache, B_cache):
    return (
        *x.stride(),
        dt.stride(0),
        dt.stride(1),
        dt_bias.stride(0),
        A.stride(0),
        *B.stride(),
        *query.stride(),
        D.stride(0) if D is not None else 0,
        *out.stride(),
        *x_cache.stride(),
        *dt_cache.stride(),
        *B_cache.stride(),
    )


def sketch_bc_pre(
    B: torch.Tensor,
    C: torch.Tensor,
    B_cache: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    bc_pre: torch.Tensor,
    state_indices: torch.Tensor,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """``bc_pre[row, group, t] = B_t · C`` over each non-flush row's window."""
    batch, groups, dstate = B.shape
    window = B_cache.shape[2]
    _replayssm_output_only_precompute_kernel[(batch, groups)](
        B,
        C,
        B_cache,
        write_pos,
        is_flush,
        bc_pre,
        state_indices,
        null_block_id,
        batch,
        groups,
        dstate,
        *B.stride(),
        *C.stride(),
        *B_cache.stride(),
        *bc_pre.stride(),
        state_indices.stride(0),
        0,
        window,
        triton.next_power_of_2(window),
        num_warps=2,
    )


def sketch_decode(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    query: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    bc_pre: torch.Tensor,
    write_pos: torch.Tensor,
    is_flush: torch.Tensor,
    state_indices: torch.Tensor,
    sketch_indices: torch.Tensor,
    out: torch.Tensor,
    sketch: SketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Decode the non-flush rows from their sketches and the window ring.

    Arguments follow ``selective_state_update_replayssm_output_only`` with
    softplus on ``dt``; ``state`` is the ``(slots, heads, dim, dstate)`` view
    of the key-major state, ``query`` is ``sketch_query(C)`` and
    ``sketch_indices`` the sketch row of each batch row.
    """
    batch = x.shape[0]
    if batch == 0:
        return
    _check_inputs(
        state,
        x,
        dt,
        A,
        B,
        query,
        D,
        dt_bias,
        x_cache,
        dt_cache,
        B_cache,
        write_pos,
        state_indices,
        out,
    )
    _, heads, dim, dstate = state.shape
    groups, window = B.shape[1], x_cache.shape[2]
    assert bc_pre.shape[0] >= batch and bc_pre.shape[1:] == (groups, window)
    assert is_flush.shape[0] >= batch and sketch_indices.shape == (batch,)
    sketch_bc_pre(
        B, C, B_cache, write_pos, is_flush, bc_pre, state_indices, null_block_id
    )
    t = sketch.tables
    block_m, block_j, num_warps = get_sketchssm_config("mamba2_decode")
    block_m = min(block_m, triton.next_power_of_2(dim))
    _sketch_decode_kernel[(triton.cdiv(dim, block_m), batch, heads)](
        x,
        dt,
        dt_bias,
        A,
        B,
        query,
        D,
        out,
        x_cache,
        dt_cache,
        B_cache,
        bc_pre,
        write_pos,
        is_flush,
        state_indices,
        sketch_indices,
        sketch.u,
        sketch.w,
        sketch.ag,
        t.ranks,
        t.u_offsets,
        t.w_offsets,
        t.ag_offsets,
        null_block_id,
        dim,
        dstate,
        heads // groups,
        *_common_strides(
            x, dt, dt_bias, A, B, query, D, out, x_cache, dt_cache, B_cache
        ),
        *bc_pre.stride(),
        U_ROWS=sketch.u.shape[1],
        W_ROWS=sketch.w.shape[1],
        AG_COLS=sketch.ag.shape[2],
        HAS_D=D is not None,
        BLOCK_M=block_m,
        BLOCK_N=triton.next_power_of_2(dstate),
        BLOCK_K=max(16, triton.next_power_of_2(window)),
        BLOCK_J=block_j,
        num_warps=num_warps,
    )


def sketch_flush(
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    query: torch.Tensor,
    D: torch.Tensor | None,
    dt_bias: torch.Tensor,
    x_cache: torch.Tensor,
    dt_cache: torch.Tensor,
    B_cache: torch.Tensor,
    write_pos: torch.Tensor,
    flush_rows: torch.Tensor,
    state_indices: torch.Tensor,
    sketch_indices: torch.Tensor,
    out: torch.Tensor,
    sketch: SketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Flush the window of the flush rows into their states, decode them and
    rebuild their sketches.

    Arguments are those of ``sketch_decode``; ``flush_rows`` lists the flush
    rows followed by -1 padding.
    """
    batch = x.shape[0]
    if batch == 0:
        return
    _check_inputs(
        state,
        x,
        dt,
        A,
        B,
        query,
        D,
        dt_bias,
        x_cache,
        dt_cache,
        B_cache,
        write_pos,
        state_indices,
        out,
    )
    assert flush_rows.shape == (batch,) and flush_rows.dtype == torch.int32
    _, heads, dim, dstate = state.shape
    groups, window = B.shape[1], x_cache.shape[2]
    # Programs walk the -1 padded flush row list.
    block_m, num_warps, rows_per_program, block_v = get_sketchssm_config("mamba2_flush")
    programs = triton.cdiv(batch, rows_per_program)

    # FP32 rotated window keys R B_t of the flush rows.
    keys = torch.empty(
        batch, groups, window, dstate, dtype=torch.float32, device=x.device
    )
    block_c = min(dstate, 64)
    _window_keys_kernel[(programs, groups, dstate // block_c)](
        B,
        B_cache,
        sketch.frames_t,
        keys,
        write_pos,
        flush_rows,
        state_indices,
        null_block_id,
        batch,
        B.stride(0),
        B.stride(1),
        *B_cache.stride()[:3],
        *keys.stride()[:3],
        W=window,
        N=dstate,
        BLOCK_T=16,
        BLOCK_C=block_c,
        num_warps=4,
    )

    t = sketch.tables
    _sketch_flush_kernel[(programs, heads)](
        state,
        x,
        dt,
        dt_bias,
        A,
        B,
        query,
        D,
        out,
        x_cache,
        dt_cache,
        B_cache,
        keys,
        write_pos,
        flush_rows,
        state_indices,
        sketch_indices,
        sketch.u,
        sketch.w,
        sketch.ag,
        t.ranks,
        t.u_offsets,
        t.w_offsets,
        t.ag_offsets,
        null_block_id,
        batch,
        heads // groups,
        *keys.stride()[:3],
        *state.stride(),
        *_common_strides(
            x, dt, dt_bias, A, B, query, D, out, x_cache, dt_cache, B_cache
        ),
        DIM=dim,
        DSTATE=dstate,
        U_ROWS=sketch.u.shape[1],
        W_ROWS=sketch.w.shape[1],
        AG_COLS=sketch.ag.shape[2],
        HAS_D=D is not None,
        BLOCK_M=block_m,
        BLOCK_K=max(16, triton.next_power_of_2(window)),
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
