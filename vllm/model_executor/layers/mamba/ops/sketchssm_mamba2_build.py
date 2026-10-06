# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SketchSSM for Mamba-2: sketch layout, B/C rotations and the sketch build.

The state ``S`` is kept key-major, ``(slot, head, dstate, dim)``, in
coordinates rotated by a calibrated orthogonal frame ``R`` per group. Prefill
runs the unrotated chunk scan and rotates its states (``sketch_rotate_states``);
decode rotates only the query ``R C`` (``sketch_query``) and, at a flush, the
window keys.

A head of rank ``m`` (0 = dense) keeps ``U``, the leading ``m`` rows of ``S``
(all rows if dense), and reads ``S^T q`` as ``U^T c`` with the ridge solution

    c = (diag(r) + Z^T Z + λI)^-1 (diag(r) + Z^T Z)[:m] q,

where ``Z`` holds the first ``min(m, 4)`` pivot rows of the normalized state
orthonormalized over ``dim`` and ``r`` the row energy they leave. It is
stored as ``c_j = a_j q_j + Σ_p g_pj (w_p · q)``: the ``w`` rows in BF16, and
``a`` and ``g`` in FP32 ``ag`` for ``m > 4`` (for ``m <= 4``, ``a = 0`` and
``g`` folds into ``w``). Sketch buffers are indexed by the persistent request
index.
"""

from dataclasses import dataclass

import torch

from vllm.model_executor.layers.mamba.ops.sketchssm_config import get_sketchssm_config
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID

SKETCH_PIVOTS = 4
SKETCH_RIDGE = 0.003
SKETCH_DTYPES = (torch.bfloat16, torch.bfloat16, torch.float32)
# FP32 dots of the rotations and the flush (3xTF32).
SKETCH_DOT_PRECISION = "tf32x3"

_PIVOTS = tl.constexpr(SKETCH_PIVOTS)
_RIDGE = tl.constexpr(SKETCH_RIDGE)


def sketch_shapes(
    ranks: torch.Tensor, head_dim: int, state_size: int
) -> tuple[tuple[int, ...], ...]:
    """Per-request ``(u, w, ag)`` shapes of one layer."""
    ranks = ranks.long()
    u_rows = int(torch.where(ranks == 0, state_size, ranks).sum())
    w_rows = int(ranks.clamp(max=SKETCH_PIVOTS).sum())
    ag_cols = int(ranks.sum())
    return (
        (max(u_rows, 1), head_dim),
        (max(w_rows, 1), state_size),
        (SKETCH_PIVOTS + 1, max(ag_cols, 1)),
    )


class SketchTables(torch.nn.Module):
    """Per-head ranks and offsets into one layer's packed sketch."""

    def __init__(self, ranks: torch.Tensor, state_size: int):
        super().__init__()
        ranks = ranks.to(torch.int32).cpu()
        u_rows = torch.where(ranks == 0, state_size, ranks)
        pivots = ranks.clamp(max=SKETCH_PIVOTS)
        tables = {
            "ranks": ranks,
            "u_offsets": u_rows.cumsum(0) - u_rows,
            "w_offsets": pivots.cumsum(0) - pivots,
            "ag_offsets": ranks.cumsum(0) - ranks,
        }
        device = torch.get_default_device()
        for name, value in tables.items():
            self.register_buffer(
                name, value.to(torch.int32).to(device), persistent=False
            )


@dataclass
class SketchArgs:
    """One layer's sketch buffers, tables and frames."""

    u: torch.Tensor  # (num_reqs, u_rows, head_dim) bf16
    w: torch.Tensor  # (num_reqs, w_rows, state_size) bf16
    ag: torch.Tensor  # (num_reqs, 5, ag_cols) fp32
    tables: SketchTables
    frames_t: torch.Tensor  # (groups, state_size, state_size) fp32, R^T


@triton.jit(do_not_specialize=["num_tokens"])
def _rotate_groups_kernel(
    x_ptr,
    y_ptr,
    frames_t_ptr,
    num_tokens,
    stride_x_token,
    stride_x_elem,
    stride_y_token,
    stride_y_elem,
    N: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    # y[t, g, cols] = x[t, g] @ R_g^T[:, cols] for BLOCK_C output columns.
    pid_t = tl.program_id(0)
    pid_g = tl.program_id(1)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_n = tl.arange(0, N)
    offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_t[:, None] < num_tokens
    x = tl.load(
        x_ptr
        + offs_t[:, None] * stride_x_token
        + (pid_g * N + offs_n[None, :]) * stride_x_elem,
        mask=mask,
        other=0.0,
    )
    r = tl.load(frames_t_ptr + pid_g * N * N + offs_n[:, None] * N + offs_c[None, :])
    y = tl.dot(x.to(tl.float32), r, input_precision=DOT_PRECISION)
    tl.store(
        y_ptr
        + offs_t[:, None] * stride_y_token
        + (pid_g * N + offs_c[None, :]) * stride_y_elem,
        y.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


def _rotate_groups(
    x: torch.Tensor, frames_t: torch.Tensor, out: torch.Tensor, block_t: int
) -> None:
    # x and out are (tokens, groups * N) with any element stride.
    num_tokens = x.shape[0]
    groups, n, _ = frames_t.shape
    if num_tokens == 0:
        return
    block_c = min(n, 64)
    grid = (triton.cdiv(num_tokens, block_t), groups, n // block_c)
    _rotate_groups_kernel[grid](
        x,
        out,
        frames_t,
        num_tokens,
        x.stride(0),
        x.stride(-1),
        out.stride(0),
        out.stride(-1),
        N=n,
        BLOCK_T=block_t,
        BLOCK_C=block_c,
        DOT_PRECISION=SKETCH_DOT_PRECISION,
        num_warps=4,
    )


@triton.jit
def _rotate_states_kernel(
    x_ptr,
    y_ptr,
    frames_t_ptr,
    heads,
    heads_per_group,
    stride_x_item,
    stride_x_head,
    stride_x_dim,
    stride_x_dstate,
    stride_y_item,
    stride_y_head,
    stride_y_dim,
    DIM: tl.constexpr,
    N: tl.constexpr,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    INVERSE: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    # y[i, h, p, cols] = x[i, h, p] @ R_g^T[:, cols] (R_g[:, cols] if INVERSE).
    item = tl.program_id(0) // heads
    head = tl.program_id(0) % heads
    group = head // heads_per_group
    offs_p = tl.program_id(1) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_n = tl.arange(0, N)
    offs_c = tl.program_id(2) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs_p[:, None] < DIM
    x = tl.load(
        x_ptr
        + item * stride_x_item
        + head * stride_x_head
        + offs_p[:, None] * stride_x_dim
        + offs_n[None, :] * stride_x_dstate,
        mask=mask,
        other=0.0,
    )
    frame = frames_t_ptr + group * N * N
    if INVERSE:
        r = tl.load(frame + offs_c[None, :] * N + offs_n[:, None])
    else:
        r = tl.load(frame + offs_n[:, None] * N + offs_c[None, :])
    y = tl.dot(x, r, input_precision=DOT_PRECISION)
    tl.store(
        y_ptr
        + item * stride_y_item
        + head * stride_y_head
        + offs_p[:, None] * stride_y_dim
        + offs_c[None, :],
        y,
        mask=mask,
    )


def sketch_rotate_states(
    states: torch.Tensor, frames_t: torch.Tensor, inverse: bool = False
) -> torch.Tensor:
    """States ``(items, heads, dim, N)`` FP32 in the rotated frame, ``S R^T``
    (``S R`` back to the model frame if ``inverse``).

    Prefill runs the unrotated chunk scan; its states enter the cache rotated.
    """
    items, heads, dim, n = states.shape
    assert states.dtype == torch.float32
    assert frames_t.is_contiguous()
    out = torch.empty(states.shape, dtype=torch.float32, device=states.device)
    if items == 0:
        return out
    block_p, block_c = 32, min(n, 64)
    grid = (items * heads, triton.cdiv(dim, block_p), n // block_c)
    _rotate_states_kernel[grid](
        states,
        out,
        frames_t,
        heads,
        heads // frames_t.shape[0],
        *states.stride(),
        *out.stride()[:3],
        DIM=dim,
        N=n,
        BLOCK_P=block_p,
        BLOCK_C=block_c,
        INVERSE=inverse,
        DOT_PRECISION=SKETCH_DOT_PRECISION,
        num_warps=4,
    )
    return out


def sketch_query(C: torch.Tensor, frames_t: torch.Tensor) -> torch.Tensor:
    """FP32 rotated query ``R C`` ``(batch, groups, N)`` of the decode C."""
    batch, groups, n = C.shape
    assert C.stride(1) == n * C.stride(2)
    query = torch.empty(batch, groups, n, dtype=torch.float32, device=C.device)
    _rotate_groups(C.flatten(1), frames_t, query.view(batch, -1), block_t=32)
    return query


@triton.jit
def _inv_norm(w, u):
    # 1 / |w|, or 0 when w keeps almost nothing of u.
    e = tl.sum(w * w)
    keep = e > 1.0e-12 * tl.sum(u * u)
    return tl.where(keep, 1.0 / tl.sqrt(tl.where(e > 0.0, e, 1.0)), 0.0)


@triton.jit
def _pivot_scalars(u0, u1, u2, u3, m):
    # Gram-Schmidt (twice) coefficients of the pivot rows u0..u3.
    z = tl.sum(u0 * 0.0)
    e0 = tl.sum(u0 * u0)
    n0 = 1.0 / tl.sqrt(tl.where(e0 > 0.0, e0, 1.0))
    p10, p11, n1 = z, z, z
    p20, p21, p22, p23, n2 = z, z, z, z, z
    p30, p31, p32, p33, p34, p35, n3 = z, z, z, z, z, z, z
    v0 = u0 * n0
    v1 = u0 * 0.0
    v2 = u0 * 0.0
    if m > 1:
        p10 = tl.sum(v0 * u1)
        w1 = u1 - v0 * p10
        p11 = tl.sum(v0 * w1)
        w1 = w1 - v0 * p11
        n1 = _inv_norm(w1, u1)
        v1 = w1 * n1
    if m > 2:
        p20 = tl.sum(v0 * u2)
        w2 = u2 - v0 * p20
        p21 = tl.sum(v1 * w2)
        w2 = w2 - v1 * p21
        p22 = tl.sum(v0 * w2)
        w2 = w2 - v0 * p22
        p23 = tl.sum(v1 * w2)
        w2 = w2 - v1 * p23
        n2 = _inv_norm(w2, u2)
        v2 = w2 * n2
    if m > 3:
        p30 = tl.sum(v0 * u3)
        w3 = u3 - v0 * p30
        p31 = tl.sum(v1 * w3)
        w3 = w3 - v1 * p31
        p32 = tl.sum(v2 * w3)
        w3 = w3 - v2 * p32
        p33 = tl.sum(v0 * w3)
        w3 = w3 - v0 * p33
        p34 = tl.sum(v1 * w3)
        w3 = w3 - v1 * p34
        p35 = tl.sum(v2 * w3)
        w3 = w3 - v2 * p35
        n3 = _inv_norm(w3, u3)
    return n0, p10, p11, n1, p20, p21, p22, p23, n2, p30, p31, p32, p33, p34, p35, n3


@triton.jit
def _pivot_directions(
    u0,
    u1,
    u2,
    u3,
    n0,
    p10,
    p11,
    n1,
    p20,
    p21,
    p22,
    p23,
    n2,
    p30,
    p31,
    p32,
    p33,
    p34,
    p35,
    n3,
):
    # Orthonormal pivot directions v1..v3 from the _pivot_scalars coefficients.
    v0 = u0 * n0
    v1 = (u1 - v0 * p10 - v0 * p11) * n1
    v2 = (u2 - v0 * p20 - v1 * p21 - v0 * p22 - v1 * p23) * n2
    v3 = (u3 - v0 * p30 - v1 * p31 - v2 * p32 - v0 * p33 - v1 * p34 - v2 * p35) * n3
    return v1, v2, v3


@triton.jit
def _sketch_head(
    state_ptr,
    sketch_index,
    head,
    u_ptr,
    w_ptr,
    ag_ptr,
    ranks_ptr,
    u_offsets_ptr,
    w_offsets_ptr,
    ag_offsets_ptr,
    stride_state_dim,
    stride_state_dstate,
    DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    U_ROWS,
    W_ROWS,
    AG_COLS,
    BLOCK_V: tl.constexpr,
):
    # Sketch of one head from its state at state_ptr, streamed in dim chunks.
    j = tl.arange(0, DSTATE)
    m = tl.load(ranks_ptr + head).to(tl.int32)
    u_rows = u_ptr + (
        (
            sketch_index.to(tl.int64) * U_ROWS
            + tl.load(u_offsets_ptr + head)
            + j[None, :]
        )
        * DIM
    )
    if m == 0:
        for c in tl.static_range(0, DIM, BLOCK_V):
            vc = c + tl.arange(0, BLOCK_V)
            mask = vc[:, None] < DIM
            s = tl.load(
                state_ptr
                + vc[:, None] * stride_state_dim
                + j[None, :] * stride_state_dstate,
                mask=mask,
                other=0.0,
            )
            tl.store(u_rows + vc[:, None], s.to(u_ptr.dtype.element_ty), mask=mask)
    else:
        pv = tl.arange(0, triton.next_power_of_2(DIM))
        pivot = state_ptr + pv * stride_state_dim
        u0 = tl.load(pivot, mask=pv < DIM, other=0.0).to(tl.float32)
        u1 = tl.load(pivot + stride_state_dstate, mask=pv < DIM, other=0.0)
        u2 = tl.load(pivot + 2 * stride_state_dstate, mask=pv < DIM, other=0.0)
        u3 = tl.load(pivot + 3 * stride_state_dstate, mask=pv < DIM, other=0.0)
        n0, p10, p11, n1, p20, p21, p22, p23, n2, p30, p31, p32, p33, p34, p35, n3 = (
            _pivot_scalars(
                u0, u1.to(tl.float32), u2.to(tl.float32), u3.to(tl.float32), m
            )
        )
        energy = tl.full((DSTATE,), 0.0, tl.float32)
        cross = tl.full((DSTATE,), 0.0, tl.float32)
        z1 = tl.full((DSTATE,), 0.0, tl.float32)
        z2 = tl.full((DSTATE,), 0.0, tl.float32)
        z3 = tl.full((DSTATE,), 0.0, tl.float32)
        for c in tl.static_range(0, DIM, BLOCK_V):
            vc = c + tl.arange(0, BLOCK_V)
            vmask = vc < DIM
            chunk = state_ptr + vc * stride_state_dim
            s = tl.load(
                chunk[:, None] + j[None, :] * stride_state_dstate,
                mask=vmask[:, None],
                other=0.0,
            ).to(tl.float32)
            tl.store(
                u_rows + vc[:, None],
                s.to(u_ptr.dtype.element_ty),
                mask=vmask[:, None] & (j[None, :] < m),
            )
            c0 = tl.load(chunk, mask=vmask, other=0.0).to(tl.float32)
            energy += tl.sum(s * s, axis=0)
            cross += tl.sum(c0[:, None] * s, axis=0)
            if m > 1:
                c1 = tl.load(chunk + stride_state_dstate, mask=vmask, other=0.0)
                c2 = tl.load(chunk + 2 * stride_state_dstate, mask=vmask, other=0.0)
                c3 = tl.load(chunk + 3 * stride_state_dstate, mask=vmask, other=0.0)
                v1, v2, v3 = _pivot_directions(
                    c0,
                    c1.to(tl.float32),
                    c2.to(tl.float32),
                    c3.to(tl.float32),
                    n0,
                    p10,
                    p11,
                    n1,
                    p20,
                    p21,
                    p22,
                    p23,
                    n2,
                    p30,
                    p31,
                    p32,
                    p33,
                    p34,
                    p35,
                    n3,
                )
                z1 += tl.sum(v1[:, None] * s, axis=0)
                if m > 2:
                    z2 += tl.sum(v2[:, None] * s, axis=0)
                if m > 3:
                    z3 += tl.sum(v3[:, None] * s, axis=0)
        # Normalize by the mean row energy; z0..z3 are the rows of Z.
        mean = tl.sum(energy) / DSTATE
        safe_mean = tl.where(mean > 0.0, mean, 1.0)
        pivot_energy = tl.sum(tl.where(j == 0, energy, 0.0))
        e = energy / safe_mean
        z0 = (
            cross
            / tl.sqrt(tl.where(pivot_energy > 0.0, pivot_energy, 1.0))
            / tl.sqrt(safe_mean)
        )
        z0 = tl.where(pivot_energy > 0.0, z0, 0.0)
        z1 = z1 / tl.sqrt(safe_mean)
        z2 = z2 / tl.sqrt(safe_mean)
        z3 = z3 / tl.sqrt(safe_mean)
        residual = tl.maximum(e - z0 * z0 - z1 * z1 - z2 * z2 - z3 * z3, 0.0)
        residual = tl.where(j < tl.minimum(m, _PIVOTS), 0.0, residual)
        denominator = residual + _RIDGE
        a = residual / denominator
        # g = (I + Z_m D^-1 Z_m^T)^-1 Z_m D^-1 by a Cholesky solve, rows g0..g3.
        b0 = tl.where(j < m, z0 / denominator, 0.0)
        g0 = b0 / (1.0 + tl.sum(z0 * b0))
        g1 = tl.full((DSTATE,), 0.0, tl.float32)
        g2 = tl.full((DSTATE,), 0.0, tl.float32)
        g3 = tl.full((DSTATE,), 0.0, tl.float32)
        if m > 1:
            b1 = tl.where(j < m, z1 / denominator, 0.0)
            l0_0 = tl.sqrt(1.0 + tl.sum(z0 * b0))
            l1_0 = tl.sum(z1 * b0) / l0_0
            l1_1 = tl.sqrt(1.0 + tl.sum(z1 * b1) - l1_0 * l1_0)
            y0 = b0 / l0_0
            y1 = (b1 - l1_0 * y0) / l1_1
            g1 = y1 / l1_1
            g0 = (y0 - l1_0 * g1) / l0_0
            if m > 2:
                b2 = tl.where(j < m, z2 / denominator, 0.0)
                l2_0 = tl.sum(z2 * b0) / l0_0
                l2_1 = (tl.sum(z2 * b1) - l2_0 * l1_0) / l1_1
                l2_2 = tl.sqrt(1.0 + tl.sum(z2 * b2) - l2_0 * l2_0 - l2_1 * l2_1)
                y2 = (b2 - l2_0 * y0 - l2_1 * y1) / l2_2
                g2 = y2 / l2_2
                g1 = (y1 - l2_1 * g2) / l1_1
                g0 = (y0 - l1_0 * g1 - l2_0 * g2) / l0_0
                if m > 3:
                    b3 = tl.where(j < m, z3 / denominator, 0.0)
                    l3_0 = tl.sum(z3 * b0) / l0_0
                    l3_1 = (tl.sum(z3 * b1) - l3_0 * l1_0) / l1_1
                    l3_2 = (tl.sum(z3 * b2) - l3_0 * l2_0 - l3_1 * l2_1) / l2_2
                    l3_3 = tl.sqrt(
                        1.0 + tl.sum(z3 * b3) - l3_0 * l3_0 - l3_1 * l3_1 - l3_2 * l3_2
                    )
                    y3 = (b3 - l3_0 * y0 - l3_1 * y1 - l3_2 * y2) / l3_3
                    g3 = y3 / l3_3
                    g2 = (y2 - l3_2 * g3) / l2_2
                    g1 = (y1 - l2_1 * g2 - l3_1 * g3) / l1_1
                    g0 = (y0 - l1_0 * g1 - l2_0 * g2 - l3_0 * g3) / l0_0
        # w_p = Z_p scaled by λ / (r + λ) on the kept rows.
        factor = tl.where(j < m, _RIDGE / denominator, 1.0)
        w_rows = w_ptr + (
            (sketch_index.to(tl.int64) * W_ROWS + tl.load(w_offsets_ptr + head))
            * DSTATE
            + j
        )
        ag_cols = (
            ag_ptr
            + sketch_index.to(tl.int64) * ((_PIVOTS + 1) * AG_COLS)
            + tl.load(ag_offsets_ptr + head)
            + j
        )
        if m <= _PIVOTS:
            # a = 0: fold g into the w rows, c = W q.
            for n in range(m):
                f0 = tl.sum(tl.where(j == n, g0, 0.0))
                f1 = tl.sum(tl.where(j == n, g1, 0.0))
                f2 = tl.sum(tl.where(j == n, g2, 0.0))
                f3 = tl.sum(tl.where(j == n, g3, 0.0))
                merged = (f0 * z0 + f1 * z1 + f2 * z2 + f3 * z3) * factor
                tl.store(w_rows + n * DSTATE, merged.to(w_ptr.dtype.element_ty))
        else:
            tl.store(w_rows, (z0 * factor).to(w_ptr.dtype.element_ty))
            tl.store(ag_cols, a, mask=j < m)
            tl.store(ag_cols + AG_COLS, g0, mask=j < m)
            if m > 1:
                tl.store(w_rows + DSTATE, (z1 * factor).to(w_ptr.dtype.element_ty))
                tl.store(ag_cols + 2 * AG_COLS, g1, mask=j < m)
            if m > 2:
                tl.store(w_rows + 2 * DSTATE, (z2 * factor).to(w_ptr.dtype.element_ty))
                tl.store(ag_cols + 3 * AG_COLS, g2, mask=j < m)
            if m > 3:
                tl.store(w_rows + 3 * DSTATE, (z3 * factor).to(w_ptr.dtype.element_ty))
                tl.store(ag_cols + 4 * AG_COLS, g3, mask=j < m)


# The prefill row pointers are slices at any offset: no alignment
# specialization, so one compile (the warmup's) serves every batch.
@triton.jit(
    do_not_specialize=[
        "state_indices_ptr",
        "sketch_indices_ptr",
        "flags_ptr",
        "U_ROWS",
        "W_ROWS",
        "AG_COLS",
    ]
)
def _sketch_build_kernel(
    state_ptr,
    u_ptr,
    w_ptr,
    ag_ptr,
    ranks_ptr,
    u_offsets_ptr,
    w_offsets_ptr,
    ag_offsets_ptr,
    state_indices_ptr,
    sketch_indices_ptr,
    flags_ptr,
    null_block_id,
    stride_state_slot,
    stride_state_head,
    stride_state_dim,
    stride_state_dstate,
    DIM: tl.constexpr,
    DSTATE: tl.constexpr,
    U_ROWS,
    W_ROWS,
    AG_COLS,
    BLOCK_V: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    if tl.load(flags_ptr + row) != 0:
        slot = tl.load(state_indices_ptr + row).to(tl.int64)
        if slot != null_block_id:
            _sketch_head(
                state_ptr + slot * stride_state_slot + head * stride_state_head,
                tl.load(sketch_indices_ptr + row),
                head,
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


def sketch_build(
    state: torch.Tensor,
    state_indices: torch.Tensor,
    sketch_indices: torch.Tensor,
    flags: torch.Tensor,
    sketch: SketchArgs,
    null_block_id: int = NULL_BLOCK_ID,
) -> None:
    """Build the sketches of the flagged rows from their states.

    Args:
        state: ``(slots, heads, dim, dstate)`` view of the key-major state.
        state_indices: ``(batch,)`` state slot of each row.
        sketch_indices: ``(batch,)`` sketch buffer row of each row.
        flags: ``(batch,)`` nonzero for the rows to build.
        sketch: The layer's sketch.
        null_block_id: State slot of padding rows, which are skipped.

    """
    batch = state_indices.shape[0]
    if batch == 0:
        return
    t = sketch.tables
    _, heads, head_dim, state_size = state.shape
    assert t.ranks.shape == (heads,)
    assert sketch_indices.shape == flags.shape == (batch,)
    assert state_indices.stride(0) == 1
    block_v, num_warps = get_sketchssm_config("mamba2_build")
    _sketch_build_kernel[(batch, heads)](
        state,
        sketch.u,
        sketch.w,
        sketch.ag,
        t.ranks,
        t.u_offsets,
        t.w_offsets,
        t.ag_offsets,
        state_indices,
        sketch_indices,
        flags,
        null_block_id,
        *state.stride(),
        DIM=head_dim,
        DSTATE=state_size,
        U_ROWS=sketch.u.shape[1],
        W_ROWS=sketch.w.shape[1],
        AG_COLS=sketch.ag.shape[2],
        BLOCK_V=block_v,
        num_warps=num_warps,
    )
