"""Helion kernels for SGLang's Kimi Delta Attention prefill path.

The public :func:`chunk_kda` entry point in this module is intended to match
``sglang.kernels.ops.attention.fla.kda.chunk_kda``.  KDA uses 64-token chunks
and keeps the cumulative per-key decay in base-2 logarithm space.
"""

from __future__ import annotations

import helion
import helion.language as hl
import torch

from vllm.third_party.flash_linear_attention.ops.index import (
    prepare_chunk_indices,
    prepare_chunk_offsets,
)

CHUNK_SIZE = 64
# Rounded identically to flash-linear-attention/SGLang before FP32 multiply.
RCP_LN2 = 1.4426950216293335
L2_NORM_EPS = 1e-6
SOFTPLUS_THRESHOLD = 20.0

# SGLang initializes torch.distributed, but these kernels have no collectives.
_IGNORED_WARNINGS = [helion.exc.ProcessGroupNameNotFound]
# K3 exposes 12 local heads at TP=8. Lower head counts share the same
# low-occupancy packed-varlen state-propagation regime.
_PREFILL_SMALL_HEAD_THRESHOLD = 12


_L2_NORM_CONFIG = helion.Config(
    block_sizes=[8],
    num_warps=4,
    num_stages=2,
    indexing="pointer",
)


@helion.kernel(
    static_shapes=False,
    config=_L2_NORM_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
def _l2norm_qk(
    q: torch.Tensor,
    k: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize Q and K rows with Triton's FP32 accumulation contract."""
    B = q.size(0)
    T = q.size(1)
    H = hl.specialize(q.size(2))
    K = hl.specialize(q.size(3))
    hl.specialize(
        (
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(1),
            k.stride(2),
            k.stride(3),
        )
    )

    q_out = torch.empty_like(q)
    k_out = torch.empty_like(k)
    q_rows = q.view(B * T * H, K)
    k_rows = k.view(B * T * H, K)
    q_out_rows = q_out.view(B * T * H, K)
    k_out_rows = k_out.view(B * T * H, K)
    block_rows = hl.register_block_size(1, 16)

    for tile_rows in hl.tile(B * T * H, block_size=block_rows):
        q_value = q_rows[tile_rows, :].float()
        k_value = k_rows[tile_rows, :].float()
        q_norm = torch.sqrt((q_value * q_value).sum(-1) + L2_NORM_EPS)
        k_norm = torch.sqrt((k_value * k_value).sum(-1) + L2_NORM_EPS)
        q_out_rows[tile_rows, :] = (q_value / q_norm[:, None]).to(q.dtype)
        k_out_rows[tile_rows, :] = (k_value / k_norm[:, None]).to(k.dtype)

    return q_out, k_out


_GATE_FIXED_CONFIG = helion.Config(
    block_sizes=[16],
    loop_orders=[[1, 2, 0]],
    num_warps=1,
    num_stages=1,
    indexing="pointer",
)


# Fixed chunks use arithmetic indexing; varlen chunks load sequence metadata.
# Their generated kernels have different occupancy and cache-policy optima.
# Eviction policies are positional in the traced load order. Retune them if
# the shared gate body gains, loses, or reorders loads.
_GATE_VARLEN_CONFIG = helion.Config(
    block_sizes=[16],
    # The sixth generated load streams the gate tile once per program.
    load_eviction_policies=["", "", "", "", "", "first"] + [""] * 11,
    loop_orders=[[2, 1, 0]],
    num_warps=8,
    num_stages=1,
    indexing="pointer",
)


def _activate_gate(
    raw_gate: torch.Tensor,
    a_log: torch.Tensor,
    lower_bound: float,
    use_lower_bound: hl.constexpr,
) -> torch.Tensor:
    a = torch.exp2(a_log.float() * RCP_LN2)
    if use_lower_bound:
        return lower_bound * torch.sigmoid(a * raw_gate)
    softplus = torch.where(
        raw_gate < SOFTPLUS_THRESHOLD,
        torch.log(1.0 + torch.exp2(raw_gate * RCP_LN2)),
        raw_gate,
    )
    return -a * softplus


@helion.kernel(
    static_shapes=False,
    config=_GATE_FIXED_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
def _gate_cumsum_operands(
    g: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    gate_scale: float,
    q_scale: float,
    lower_bound: float,
    activate: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    has_bias: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    use_lower_bound: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Compute cumulative gates and rounded Q/K operands in one pass."""
    B = g.size(0)
    T = g.size(1)
    H = hl.specialize(g.size(2))
    K = hl.specialize(g.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    hl.specialize(
        (
            g.stride(1),
            g.stride(2),
            g.stride(3),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            beta.stride(1),
            beta.stride(2),
            a_log.stride(0),
            dt_bias.stride(0),
            cu_seqlens.stride(0),
            chunk_indices.stride(0),
            chunk_indices.stride(1),
        )
    )

    out = torch.empty_like(g, dtype=torch.float32)
    qg = torch.empty_like(q)
    wk = torch.empty_like(k)
    kg = torch.empty_like(k)
    chunk_decay = torch.empty(
        [total_chunks, H, K],
        dtype=torch.float32,
        device=g.device,
    )
    g_rows = g.view(B * T * H, K)
    q_rows = q.view(B * T * H, K)
    k_rows = k.view(B * T * H, K)
    beta_rows = beta.view(B * T * H)
    out_rows = out.view(B * T * H, K)
    qg_rows = qg.view(B * T * H, K)
    wk_rows = wk.view(B * T * H, K)
    kg_rows = kg.view(B * T * H, K)
    block_k = hl.register_block_size(16, K)

    for tile_chunk, tile_h, tile_k in hl.tile(
        [total_chunks, H, K],
        block_size=[1, 1, block_k],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T
        time = hl.arange(64)
        token = begin + local_chunk * CHUNK_SIZE + time
        valid = token < end
        row = token * H + tile_h.id
        value = hl.load(
            g_rows,
            [row[:, None], tile_k.index[None, :]],
            extra_mask=valid[:, None],
        ).float()
        if activate:
            if has_bias:
                value = value + dt_bias[tile_h.id * K + tile_k.index].float()[None, :]
            value = _activate_gate(
                value,
                a_log[tile_h.id],
                lower_bound,
                use_lower_bound,
            )
        value = torch.where(valid[:, None], value, 0.0)
        value = torch.cumsum(value, dim=0) * gate_scale
        q_value = hl.load(
            q_rows,
            [row[:, None], tile_k.index[None, :]],
            extra_mask=valid[:, None],
        ).float()
        k_value = hl.load(
            k_rows,
            [row[:, None], tile_k.index[None, :]],
            extra_mask=valid[:, None],
        ).float()
        beta_value = hl.load(beta_rows, [row], extra_mask=valid).float()
        last_gate = torch.where(time[:, None] == 63, value, 0.0).sum(0)
        gate_value = torch.exp2(value)
        hl.store(
            out_rows,
            [row[:, None], tile_k.index[None, :]],
            value,
            extra_mask=valid[:, None],
        )
        hl.store(
            qg_rows,
            [row[:, None], tile_k.index[None, :]],
            (q_value * q_scale * gate_value).to(q.dtype),
            extra_mask=valid[:, None],
        )
        hl.store(
            wk_rows,
            [row[:, None], tile_k.index[None, :]],
            (k_value * beta_value[:, None] * gate_value).to(k.dtype),
            extra_mask=valid[:, None],
        )
        hl.store(
            kg_rows,
            [row[:, None], tile_k.index[None, :]],
            (k_value * torch.exp2(last_gate[None, :] - value)).to(k.dtype),
            extra_mask=valid[:, None],
        )
        hl.store(
            chunk_decay,
            [tile_chunk.id, tile_h.id, tile_k.index],
            torch.exp2(last_gate),
        )

    return out, qg, wk, kg, chunk_decay


_gate_cumsum_operands_varlen = helion.kernel(
    static_shapes=False,
    config=_GATE_VARLEN_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)(_gate_cumsum_operands.fn)


def gate_chunk_cumsum_operands(
    g: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    beta: torch.Tensor,
    *,
    q_scale: float,
    a_log: torch.Tensor | None,
    dt_bias: torch.Tensor | None,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
    lower_bound: float | None = None,
    gate_scale: float = RCP_LN2,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Gate preprocessing plus rounded operands reused by later stages."""
    flat_a_log = (
        a_log.reshape(-1)
        if a_log is not None
        else torch.empty(1, device=g.device, dtype=torch.float32)
    )
    flat_bias = (
        dt_bias.reshape(-1)
        if dt_bias is not None
        else torch.empty(1, device=g.device, dtype=torch.float32)
    )
    activate = a_log is not None
    has_bias = dt_bias is not None
    use_lower_bound = lower_bound is not None
    lower_bound_value = 0.0 if lower_bound is None else lower_bound

    is_varlen = cu_seqlens is not None
    if is_varlen:
        if g.size(0) != 1:
            raise ValueError("varlen KDA requires batch size 1")
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        metadata = cu_seqlens
        gate_kernel = _gate_cumsum_operands_varlen
    else:
        metadata = torch.empty(0, device=g.device, dtype=torch.int32)
        chunk_indices = torch.empty(0, 2, device=g.device, dtype=torch.long)
        gate_kernel = _gate_cumsum_operands

    return gate_kernel(
        g,
        q,
        k,
        beta,
        flat_a_log,
        flat_bias,
        metadata,
        chunk_indices,
        gate_scale,
        q_scale,
        lower_bound_value,
        activate,
        has_bias,
        use_lower_bound,
        is_varlen,
    )


_INTRA_MATRIX_CONFIG = helion.Config(
    block_sizes=[32],
    loop_orders=[[1, 2, 0]],
    num_warps=1,
    num_stages=2,
    indexing="pointer",
)


@helion.kernel(
    static_shapes=False,
    config=_INTRA_MATRIX_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
def _intra_matrices_wide(
    q: torch.Tensor,
    k: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    scale: float,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a full 16x64 causal matrix row per CTA."""
    B = q.size(0)
    T = q.size(1)
    H = hl.specialize(q.size(2))
    K = hl.specialize(q.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    hl.specialize(
        (
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            g.stride(1),
            g.stride(2),
            g.stride(3),
            beta.stride(1),
            beta.stride(2),
        )
    )

    aqk = torch.empty([B, T, H, CHUNK_SIZE], dtype=q.dtype, device=q.device)
    akk = torch.empty([B, T, H, CHUNK_SIZE], dtype=torch.float32, device=q.device)
    q_rows = q.view(B * T * H, K)
    k_rows = k.view(B * T * H, K)
    g_rows = g.view(B * T * H, K)
    beta_rows = beta.view(B * T * H)
    aqk_rows = aqk.view(B * T * H, CHUNK_SIZE)
    akk_rows = akk.view(B * T * H, CHUNK_SIZE)
    block_k = hl.register_block_size(32, K)

    for tile_chunk, tile_h, tile_row_block in hl.tile(
        [total_chunks, H, 4],
        block_size=[1, 1, 1],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T

        row_lane = hl.arange(16)
        col_lane = hl.arange(64)
        chunk_begin = begin + local_chunk * CHUNK_SIZE
        row_local = tile_row_block.id * 16 + row_lane
        row_token = chunk_begin + row_local
        col_token = chunk_begin + col_lane
        row_valid = row_token < end
        col_valid = col_token < end
        block_causal = col_lane < (tile_row_block.id + 1) * 16
        row = row_token * H + tile_h.id
        col = col_token * H + tile_h.id
        anchor_token = chunk_begin + tile_row_block.id * 16
        anchor = anchor_token * H + tile_h.id
        # Varlen ``end`` is a loaded scalar tensor; fixed ``end`` is symbolic.
        if is_varlen:
            diag_anchor_token = torch.minimum(anchor_token + 8, end - 1)
        else:
            diag_anchor_token = min(anchor_token + 8, end - 1)
        diag_anchor = diag_anchor_token * H + tile_h.id
        aqk_off = hl.zeros([16, 64], dtype=torch.float32)
        akk_off = hl.zeros([16, 64], dtype=torch.float32)
        aqk_diag = hl.zeros([16, 16], dtype=torch.float32)
        akk_diag = hl.zeros([16, 16], dtype=torch.float32)

        for tile_k in hl.tile(K, block_size=block_k):
            q_row = hl.load(
                q_rows,
                [row[:, None], tile_k.index[None, :]],
                extra_mask=row_valid[:, None],
            ).float()
            k_row = hl.load(
                k_rows,
                [row[:, None], tile_k.index[None, :]],
                extra_mask=row_valid[:, None],
            ).float()
            g_row = hl.load(
                g_rows,
                [row[:, None], tile_k.index[None, :]],
                extra_mask=row_valid[:, None],
            ).float()
            g_anchor = hl.load(
                g_rows,
                [anchor, tile_k.index],
                extra_mask=anchor_token < end,
            ).float()
            g_diag_anchor = hl.load(
                g_rows,
                [diag_anchor, tile_k.index],
                extra_mask=diag_anchor_token < end,
            ).float()
            if tile_row_block.id > 0:
                k_col = hl.load(
                    k_rows,
                    [col[:, None], tile_k.index[None, :]],
                    extra_mask=col_valid[:, None] & block_causal[:, None],
                ).float()
                g_col = hl.load(
                    g_rows,
                    [col[:, None], tile_k.index[None, :]],
                    extra_mask=col_valid[:, None] & block_causal[:, None],
                ).float()
                off_col = col_lane < tile_row_block.id * 16
                off_col_delta = torch.where(
                    off_col[:, None],
                    g_anchor[None, :] - g_col,
                    0.0,
                )
                off_row_delta = g_row - g_anchor[None, :]
                # Masked rows load zero; cap only their positive overflow edge.
                # Valid KDA cumulative gates are non-increasing in each chunk.
                off_row_delta = torch.clamp(off_row_delta, max=126.0)
                off_row_factor = torch.exp2(off_row_delta)
                q_off = (q_row * off_row_factor).to(torch.bfloat16)
                k_off = (k_row * off_row_factor).to(torch.bfloat16)
                k_col_off = (k_col * torch.exp2(off_col_delta)).to(torch.bfloat16)
                aqk_off = hl.dot(
                    q_off,
                    k_col_off.T,
                    acc=aqk_off,
                    out_dtype=torch.float32,
                )
                akk_off = hl.dot(
                    k_off,
                    k_col_off.T,
                    acc=akk_off,
                    out_dtype=torch.float32,
                )

            diag_delta = torch.clamp(
                g_row - g_diag_anchor[None, :],
                -126.0,
                126.0,
            )
            diag_forward_factor = torch.exp2(diag_delta)
            diag_backward_factor = torch.exp2(-diag_delta)
            q_diag = q_row * diag_forward_factor
            k_diag_fwd = k_row * diag_forward_factor
            k_diag_bwd = k_row * diag_backward_factor
            aqk_diag = hl.dot(
                q_diag,
                k_diag_bwd.T,
                acc=aqk_diag,
                out_dtype=torch.float32,
            )
            akk_diag = hl.dot(
                k_diag_fwd,
                k_diag_bwd.T,
                acc=akk_diag,
                out_dtype=torch.float32,
            )

        causal = row_local[:, None] >= col_lane[None, :]
        strictly_causal = row_local[:, None] > col_lane[None, :]
        row_beta = hl.load(
            beta_rows,
            [row],
            extra_mask=row_valid,
        ).float()
        hl.store(
            aqk_rows,
            [row[:, None], col_lane[None, :]],
            torch.where(causal & col_valid[None, :], aqk_off * scale, 0.0),
            extra_mask=row_valid[:, None],
        )
        hl.store(
            akk_rows,
            [row[:, None], col_lane[None, :]],
            torch.where(
                strictly_causal & col_valid[None, :],
                akk_off * row_beta[:, None],
                0.0,
            ),
            extra_mask=row_valid[:, None],
        )
        diag_col = tile_row_block.id * 16 + row_lane
        diag_causal = row_lane[:, None] >= row_lane[None, :]
        diag_strict = row_lane[:, None] > row_lane[None, :]
        diagonal_matrix = torch.where(
            diag_strict & row_valid[None, :],
            akk_diag * row_beta[:, None],
            0.0,
        )
        diagonal_matrix = _invert_lower_16_forward_substitution(diagonal_matrix)
        hl.store(
            aqk_rows,
            [row[:, None], diag_col[None, :]],
            torch.where(diag_causal & row_valid[None, :], aqk_diag * scale, 0.0),
            extra_mask=row_valid[:, None],
        )
        hl.store(
            akk_rows,
            [row[:, None], diag_col[None, :]],
            diagonal_matrix,
            extra_mask=row_valid[:, None],
        )

    return aqk, akk


def _invert_lower_16_forward_substitution(matrix: torch.Tensor) -> torch.Tensor:
    lane = hl.arange(16)
    strictly_lower = lane[:, None] > lane[None, :]
    inverse = -torch.where(strictly_lower, matrix, 0.0)
    for row in range(2, 16):
        value = -torch.where((lane == row)[:, None], matrix, 0.0).sum(0)
        value = torch.where(lane < row, value, 0.0)
        value = value + (value[:, None] * inverse).sum(0)
        inverse = torch.where((lane == row)[:, None], value[None, :], inverse)
    return inverse + (lane[:, None] == lane[None, :]).float()


def _assemble_lower_64_inverse(
    matrix: tuple[torch.Tensor, ...],
    output_dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Assemble the 64x64 inverse from pre-inverted 16x16 diagonal blocks."""
    m00, m10, m11, m20, m21, m22, m30, m31, m32, m33 = matrix
    i00, i11, i22, i33 = m00, m11, m22, m33

    i10 = -hl.dot(
        hl.dot(i11, m10, out_dtype=torch.float32),
        i00,
        out_dtype=torch.float32,
    )
    i21 = -hl.dot(
        hl.dot(i22, m21, out_dtype=torch.float32),
        i11,
        out_dtype=torch.float32,
    )
    i32 = -hl.dot(
        hl.dot(i33, m32, out_dtype=torch.float32),
        i22,
        out_dtype=torch.float32,
    )
    i20 = -hl.dot(
        i22,
        hl.dot(m20, i00, out_dtype=torch.float32)
        + hl.dot(m21, i10, out_dtype=torch.float32),
        out_dtype=torch.float32,
    )
    i31 = -hl.dot(
        i33,
        hl.dot(m31, i11, out_dtype=torch.float32)
        + hl.dot(m32, i21, out_dtype=torch.float32),
        out_dtype=torch.float32,
    )
    i30 = -hl.dot(
        i33,
        hl.dot(m30, i00, out_dtype=torch.float32)
        + hl.dot(m31, i10, out_dtype=torch.float32)
        + hl.dot(m32, i20, out_dtype=torch.float32),
        out_dtype=torch.float32,
    )
    return tuple(
        block.to(output_dtype)
        for block in (i00, i10, i11, i20, i21, i22, i30, i31, i32, i33)
    )


def _apply_lower_64_blocks(
    matrix: tuple[torch.Tensor, ...],
    rhs: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    """Apply a 64x64 lower-triangular matrix to four 16-row blocks."""
    i00, i10, i11, i20, i21, i22, i30, i31, i32, i33 = matrix
    x0, x1, x2, x3 = rhs
    y0 = hl.dot(i00, x0, out_dtype=torch.float32)
    y1 = hl.dot(i10, x0, out_dtype=torch.float32) + hl.dot(
        i11, x1, out_dtype=torch.float32
    )
    y2 = (
        hl.dot(i20, x0, out_dtype=torch.float32)
        + hl.dot(i21, x1, out_dtype=torch.float32)
        + hl.dot(i22, x2, out_dtype=torch.float32)
    )
    y3 = (
        hl.dot(i30, x0, out_dtype=torch.float32)
        + hl.dot(i31, x1, out_dtype=torch.float32)
        + hl.dot(i32, x2, out_dtype=torch.float32)
        + hl.dot(i33, x3, out_dtype=torch.float32)
    )
    return y0, y1, y2, y3


_SOLVE_RECOMPUTE_CONFIG = helion.Config(
    block_sizes=[64, 64],
    loop_orders=[[0, 1]],
    num_warps=1,
    num_stages=3,
    indexing="pointer",
)


@helion.kernel(
    static_shapes=False,
    config=_SOLVE_RECOMPUTE_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
def _intra_solve_recompute(
    akk: torch.Tensor,
    wk: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Solve the 64x64 system and emit W and U from pre-scaled operands."""
    B = wk.size(0)
    T = wk.size(1)
    H = hl.specialize(wk.size(2))
    K = hl.specialize(wk.size(3))
    V = hl.specialize(v.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    # Each CTA owns one (chunk, head) and loads a complete column tile before
    # overwriting it, so these output aliases do not introduce cross-CTA races.
    w = wk
    u = v
    akk_rows = akk.view(B * T * H, CHUNK_SIZE)
    wk_rows = wk.view(B * T * H, K)
    v_rows = v.view(B * T * H, V)
    beta_rows = beta.view(B * T * H)
    w_rows = w.view(B * T * H, K)
    u_rows = u.view(B * T * H, V)
    block_v = hl.register_block_size(32, V)
    block_k = hl.register_block_size(32, K)

    for tile_chunk, tile_h in hl.tile(
        [total_chunks, H],
        block_size=[1, 1],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T

        lane = hl.arange(16)
        chunk_begin = begin + local_chunk * CHUNK_SIZE
        row0 = chunk_begin + lane
        row1 = chunk_begin + 16 + lane
        row2 = chunk_begin + 32 + lane
        row3 = chunk_begin + 48 + lane
        valid0 = row0 < end
        valid1 = row1 < end
        valid2 = row2 < end
        valid3 = row3 < end
        flat0 = row0 * H + tile_h.id
        flat1 = row1 * H + tile_h.id
        flat2 = row2 * H + tile_h.id
        flat3 = row3 * H + tile_h.id
        col0 = lane
        col1 = 16 + lane
        col2 = 32 + lane
        col3 = 48 + lane

        m00 = hl.load(
            akk_rows,
            [flat0[:, None], col0[None, :]],
            extra_mask=valid0[:, None] & valid0[None, :],
        ).float()
        m10 = hl.load(
            akk_rows,
            [flat1[:, None], col0[None, :]],
            extra_mask=valid1[:, None] & valid0[None, :],
        ).float()
        m11 = hl.load(
            akk_rows,
            [flat1[:, None], col1[None, :]],
            extra_mask=valid1[:, None] & valid1[None, :],
        ).float()
        m20 = hl.load(
            akk_rows,
            [flat2[:, None], col0[None, :]],
            extra_mask=valid2[:, None] & valid0[None, :],
        ).float()
        m21 = hl.load(
            akk_rows,
            [flat2[:, None], col1[None, :]],
            extra_mask=valid2[:, None] & valid1[None, :],
        ).float()
        m22 = hl.load(
            akk_rows,
            [flat2[:, None], col2[None, :]],
            extra_mask=valid2[:, None] & valid2[None, :],
        ).float()
        m30 = hl.load(
            akk_rows,
            [flat3[:, None], col0[None, :]],
            extra_mask=valid3[:, None] & valid0[None, :],
        ).float()
        m31 = hl.load(
            akk_rows,
            [flat3[:, None], col1[None, :]],
            extra_mask=valid3[:, None] & valid1[None, :],
        ).float()
        m32 = hl.load(
            akk_rows,
            [flat3[:, None], col2[None, :]],
            extra_mask=valid3[:, None] & valid2[None, :],
        ).float()
        m33 = hl.load(
            akk_rows,
            [flat3[:, None], col3[None, :]],
            extra_mask=valid3[:, None] & valid3[None, :],
        ).float()

        inverse = _assemble_lower_64_inverse(
            (m00, m10, m11, m20, m21, m22, m30, m31, m32, m33),
            wk.dtype,
        )

        beta0 = hl.load(beta_rows, [flat0], extra_mask=valid0).float()
        beta1 = hl.load(beta_rows, [flat1], extra_mask=valid1).float()
        beta2 = hl.load(beta_rows, [flat2], extra_mask=valid2).float()
        beta3 = hl.load(beta_rows, [flat3], extra_mask=valid3).float()
        for tile_v in hl.tile(V, block_size=block_v):
            v0 = hl.load(
                v_rows,
                [flat0[:, None], tile_v.index[None, :]],
                extra_mask=valid0[:, None],
            )
            v1 = hl.load(
                v_rows,
                [flat1[:, None], tile_v.index[None, :]],
                extra_mask=valid1[:, None],
            )
            v2 = hl.load(
                v_rows,
                [flat2[:, None], tile_v.index[None, :]],
                extra_mask=valid2[:, None],
            )
            v3 = hl.load(
                v_rows,
                [flat3[:, None], tile_v.index[None, :]],
                extra_mask=valid3[:, None],
            )
            vb0 = (v0 * beta0[:, None]).to(v.dtype)
            vb1 = (v1 * beta1[:, None]).to(v.dtype)
            vb2 = (v2 * beta2[:, None]).to(v.dtype)
            vb3 = (v3 * beta3[:, None]).to(v.dtype)
            u0, u1, u2, u3 = _apply_lower_64_blocks(
                inverse,
                (vb0, vb1, vb2, vb3),
            )
            hl.store(
                u_rows,
                [flat0[:, None], tile_v.index[None, :]],
                u0,
                extra_mask=valid0[:, None],
            )
            hl.store(
                u_rows,
                [flat1[:, None], tile_v.index[None, :]],
                u1,
                extra_mask=valid1[:, None],
            )
            hl.store(
                u_rows,
                [flat2[:, None], tile_v.index[None, :]],
                u2,
                extra_mask=valid2[:, None],
            )
            hl.store(
                u_rows,
                [flat3[:, None], tile_v.index[None, :]],
                u3,
                extra_mask=valid3[:, None],
            )

        for tile_k in hl.tile(K, block_size=block_k):
            wk0 = hl.load(
                wk_rows,
                [flat0[:, None], tile_k.index[None, :]],
                extra_mask=valid0[:, None],
            )
            wk1 = hl.load(
                wk_rows,
                [flat1[:, None], tile_k.index[None, :]],
                extra_mask=valid1[:, None],
            )
            wk2 = hl.load(
                wk_rows,
                [flat2[:, None], tile_k.index[None, :]],
                extra_mask=valid2[:, None],
            )
            wk3 = hl.load(
                wk_rows,
                [flat3[:, None], tile_k.index[None, :]],
                extra_mask=valid3[:, None],
            )
            w0, w1, w2, w3 = _apply_lower_64_blocks(
                inverse,
                (wk0, wk1, wk2, wk3),
            )
            hl.store(
                w_rows,
                [flat0[:, None], tile_k.index[None, :]],
                w0,
                extra_mask=valid0[:, None],
            )
            hl.store(
                w_rows,
                [flat1[:, None], tile_k.index[None, :]],
                w1,
                extra_mask=valid1[:, None],
            )
            hl.store(
                w_rows,
                [flat2[:, None], tile_k.index[None, :]],
                w2,
                extra_mask=valid2[:, None],
            )
            hl.store(
                w_rows,
                [flat3[:, None], tile_k.index[None, :]],
                w3,
                extra_mask=valid3[:, None],
            )

    return w, u


def chunk_kda_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    wk: torch.Tensor,
    kg: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None,
    chunk_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Helion equivalent of SGLang's intra-chunk KDA preparation."""
    is_varlen = cu_seqlens is not None
    if is_varlen:
        if chunk_indices is None:
            chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        metadata = cu_seqlens
    else:
        metadata = torch.empty(0, device=q.device, dtype=torch.int32)
        chunk_indices = torch.empty(0, 2, device=q.device, dtype=torch.long)

    aqk, akk = _intra_matrices_wide(
        q,
        k,
        g,
        beta,
        metadata,
        chunk_indices,
        scale,
        is_varlen,
    )
    w, u = _intra_solve_recompute(
        akk,
        wk,
        v,
        beta,
        metadata,
        chunk_indices,
        is_varlen,
    )
    return w, u, kg, aqk


_STATE_FIXED_CONFIG = helion.Config(
    block_sizes=[16],
    num_warps=8,
    num_stages=3,
    indexing="pointer",
)


# Eviction policies are positional in the traced load order. Retune both
# varlen configs if the shared state body gains, loses, or reorders loads.
_STATE_VARLEN_CONFIG = helion.Config(
    atomic_indexing=[],
    block_sizes=[64],
    indexing=[
        "pointer",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "tensor_descriptor",
        "pointer",
        "pointer",
        "pointer",
        "pointer",
        "tensor_descriptor",
        "pointer",
        "pointer",
    ],
    l2_groupings=[4],
    load_eviction_policies=[
        "",
        "",
        "",
        "last",
        "last",
        "first",
        "first",
        "first",
        "first",
    ],
    loop_orders=[[0, 2, 1]],
    num_stages=2,
    num_warps=4,
    pid_type="flat",
    range_flattens=[None, None],
    range_multi_buffers=[None, False],
    range_num_stages=[],
    range_unroll_factors=[0, 2],
)

# Packed small-head workloads benefit from a wider V tile during state propagation.
_STATE_VARLEN_SMALL_HEAD_CONFIG = helion.Config(
    atomic_indexing=[],
    block_sizes=[32],
    indexing=["pointer"] * 12,
    l2_groupings=[1],
    load_eviction_policies=["", "", "", "last", "first", "", "", "", "first"],
    loop_orders=[[1, 2, 0]],
    num_stages=3,
    num_warps=8,
    pid_type="flat",
    range_flattens=[None, None],
    range_multi_buffers=[None, True],
    range_num_stages=[],
    range_unroll_factors=[0, 0],
)


@helion.kernel(
    static_shapes=False,
    config=_STATE_FIXED_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
def _chunk_state(
    kg: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    chunk_decay: torch.Tensor,
    initial_state: torch.Tensor,
    initial_state_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    chunk_offsets: torch.Tensor,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Propagate KDA state between chunks and update the state pool in place."""
    B = kg.size(0)
    T = kg.size(1)
    H = hl.specialize(kg.size(2))
    K = hl.specialize(kg.size(3))
    V = hl.specialize(u.size(3))
    N = cu_seqlens.size(0) - 1 if is_varlen else B
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else chunks_per_batch
    hl.specialize(
        (
            kg.stride(1),
            kg.stride(2),
            kg.stride(3),
            w.stride(1),
            w.stride(2),
            w.stride(3),
            u.stride(1),
            u.stride(2),
            u.stride(3),
            chunk_decay.stride(0),
            chunk_decay.stride(1),
            chunk_decay.stride(2),
            initial_state.stride(0),
            initial_state.stride(1),
            initial_state.stride(2),
            initial_state.stride(3),
            initial_state_indices.stride(0),
        )
    )

    h = torch.empty(
        [B, total_chunks, H, V, K],
        dtype=kg.dtype,
        device=kg.device,
    )
    v_new = u
    kg_rows = kg.view(B * T * H, K)
    w_rows = w.view(B * T * H, K)
    u_rows = u.view(B * T * H, V)
    decay_rows = chunk_decay.view(-1, K)
    v_new_rows = v_new.view(B * T * H, V)
    h_rows = h.view(B * total_chunks * H, V, K)
    block_v = hl.register_block_size(1, V)

    for tile_sequence, tile_h, tile_v in hl.tile(
        [N, H, V],
        block_size=[1, 1, block_v],
    ):
        if is_varlen:
            begin = cu_seqlens[tile_sequence.id].long()
            end = cu_seqlens[tile_sequence.id + 1].long()
            output_offset = chunk_offsets[tile_sequence.id].long()
        else:
            begin = tile_sequence.id * T
            end = begin + T
            output_offset = tile_sequence.id * chunks_per_batch
        sequence_length = end - begin
        state_index = initial_state_indices[tile_sequence.id].long()
        state = initial_state[
            state_index,
            tile_h.id,
            tile_v.index,
            :,
        ].float()

        for token_tile in hl.tile(sequence_length, block_size=64):
            global_chunk = output_offset + token_tile.id
            h_rows[
                global_chunk * H + tile_h.id,
                tile_v,
                :,
            ] = state.to(h.dtype)
            token = begin + token_tile.index
            valid = token < end
            row = token * H + tile_h.id
            w_value = hl.load(
                w_rows,
                [row[:, None], hl.arange(K)[None, :]],
                extra_mask=valid[:, None],
            )
            residual = -hl.dot(
                w_value,
                state.T.to(w.dtype),
                out_dtype=torch.float32,
            )
            residual = residual + u_rows[row, tile_v].float()
            v_new_rows[row, tile_v] = residual.to(v_new.dtype)
            decay = decay_rows[global_chunk * H + tile_h.id, :]
            state = state * decay[None, :]
            kg_value = hl.load(
                kg_rows,
                [row[:, None], hl.arange(K)[None, :]],
                extra_mask=valid[:, None],
            )
            state = state + hl.dot(
                residual.T.to(kg.dtype),
                kg_value,
                out_dtype=torch.float32,
            )

        initial_state[
            state_index,
            tile_h.id,
            tile_v.index,
            :,
        ] = state.to(initial_state.dtype)

    return h, v_new


_chunk_state_varlen = helion.kernel(
    static_shapes=False,
    config=_STATE_VARLEN_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)(_chunk_state.fn)
_chunk_state_varlen_small_head = helion.kernel(
    static_shapes=False,
    config=_STATE_VARLEN_SMALL_HEAD_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)(_chunk_state.fn)


def _select_state_kernel(*, is_varlen: bool, num_heads: int) -> helion.Kernel:
    if not is_varlen:
        return _chunk_state
    if num_heads <= _PREFILL_SMALL_HEAD_THRESHOLD:
        return _chunk_state_varlen_small_head
    return _chunk_state_varlen


_OUTPUT_CONFIG = helion.Config(
    block_sizes=[128],
    loop_orders=[[1, 2, 0]],
    l2_groupings=[32],
    num_warps=2,
    num_stages=4,
    indexing="pointer",
)


@helion.kernel(
    static_shapes=False,
    config=_OUTPUT_CONFIG,
    ignore_warnings=_IGNORED_WARNINGS,
)
def _chunk_output(
    qg: torch.Tensor,
    v_new: torch.Tensor,
    aqk: torch.Tensor,
    h: torch.Tensor,
    out: torch.Tensor,
    cu_seqlens: torch.Tensor,
    chunk_indices: torch.Tensor,
    is_varlen: hl.constexpr,  # pyrefly: ignore[bad-function-definition]
) -> torch.Tensor:
    """Compose inter-chunk state output and causal intra-chunk output."""
    B = qg.size(0)
    T = qg.size(1)
    H = hl.specialize(qg.size(2))
    K = hl.specialize(qg.size(3))
    V = hl.specialize(v_new.size(3))
    chunks_per_batch = (T + CHUNK_SIZE - 1) // CHUNK_SIZE
    total_chunks = chunk_indices.size(0) if is_varlen else B * chunks_per_batch
    h_chunks = h.size(1)
    hl.specialize(
        (
            qg.stride(1),
            qg.stride(2),
            qg.stride(3),
            v_new.stride(1),
            v_new.stride(2),
            v_new.stride(3),
            aqk.stride(1),
            aqk.stride(2),
            aqk.stride(3),
            h.stride(1),
            h.stride(2),
            h.stride(3),
            h.stride(4),
            out.stride(1),
            out.stride(2),
            out.stride(3),
        )
    )

    qg_rows = qg.view(B * T * H, K)
    v_rows = v_new.view(B * T * H, V)
    aqk_rows = aqk.view(B * T * H, CHUNK_SIZE)
    h_rows = h.view(B * h_chunks * H, V, K)
    out_rows = out.view(B * T * H, V)
    block_v = hl.register_block_size(32, V)

    # ``out`` may alias ``v_new``. Each CTA loads its complete input tile before
    # overwriting that same tile, and no other CTA reads it.
    for tile_chunk, tile_h, tile_v in hl.tile(
        [total_chunks, H, V],
        block_size=[1, 1, block_v],
    ):
        if is_varlen:
            sequence = chunk_indices[tile_chunk.id, 0].long()
            local_chunk = chunk_indices[tile_chunk.id, 1].long()
            begin = cu_seqlens[sequence].long()
            end = cu_seqlens[sequence + 1].long()
            h_chunk = tile_chunk.id
        else:
            sequence = tile_chunk.id // chunks_per_batch
            local_chunk = tile_chunk.id % chunks_per_batch
            begin = sequence * T
            end = begin + T
            h_chunk = tile_chunk.id

        lane = hl.arange(64)
        token = begin + local_chunk * CHUNK_SIZE + lane
        valid = token < end
        row = token * H + tile_h.id
        qg_value = hl.load(
            qg_rows,
            [row[:, None], hl.arange(K)[None, :]],
            extra_mask=valid[:, None],
        )
        h_value = h_rows[
            h_chunk * H + tile_h.id,
            tile_v,
            :,
        ]
        output = hl.dot(
            qg_value,
            h_value.T,
            out_dtype=torch.float32,
        )
        a_value = hl.load(
            aqk_rows,
            [row[:, None], lane[None, :]],
            extra_mask=valid[:, None],
        )
        v_value = hl.load(
            v_rows,
            [row, tile_v],
            extra_mask=valid[:, None],
        )
        output = hl.dot(
            a_value.to(v_new.dtype),
            v_value,
            acc=output,
            out_dtype=torch.float32,
        )
        hl.store(
            out_rows,
            [row, tile_v],
            output,
            extra_mask=valid[:, None],
        )

    return out


def chunk_kda(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float | None = None,
    initial_state: torch.Tensor | None = None,
    initial_state_indices: torch.Tensor | None = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.Tensor | None = None,
    A_log: torch.Tensor | None = None,
    dt_bias: torch.Tensor | None = None,
    lower_bound: float | None = None,
    output_intermediate_states: bool = False,
    **kwargs: object,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Match the public forward contract of SGLang's Triton ``chunk_kda``."""
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if initial_state is None or initial_state_indices is None:
        raise ValueError("KDA prefill requires an indexed initial-state pool")

    q = q.contiguous()
    k = k.contiguous()
    if use_qk_l2norm_in_kernel:
        q, k = _l2norm_qk(q, k)
    v = v.contiguous()
    g = g.contiguous()
    beta = beta.contiguous()
    chunk_indices = (
        prepare_chunk_indices(cu_seqlens, CHUNK_SIZE)
        if cu_seqlens is not None
        else None
    )
    g, qg, wk, kg, chunk_decay = gate_chunk_cumsum_operands(
        g,
        q,
        k,
        beta,
        q_scale=scale,
        a_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        lower_bound=lower_bound,
    )
    w, u, kg, aqk = chunk_kda_fwd_intra(
        q,
        k,
        v,
        g,
        beta,
        wk,
        kg,
        scale,
        cu_seqlens,
        chunk_indices,
    )

    is_varlen = cu_seqlens is not None
    if is_varlen:
        chunk_offsets = prepare_chunk_offsets(cu_seqlens, CHUNK_SIZE)
        metadata = cu_seqlens
    else:
        metadata = torch.empty(0, device=q.device, dtype=torch.int32)
        chunk_offsets = torch.empty(0, device=q.device, dtype=torch.long)
    state_kernel = _select_state_kernel(is_varlen=is_varlen, num_heads=q.size(2))
    h, v_new = state_kernel(
        kg,
        w,
        u,
        chunk_decay,
        initial_state,
        initial_state_indices,
        metadata,
        (
            chunk_indices
            if chunk_indices is not None
            else torch.empty(0, 2, device=q.device, dtype=torch.long)
        ),
        chunk_offsets,
        is_varlen,
    )
    if chunk_indices is None:
        chunk_indices = torch.empty(0, 2, device=q.device, dtype=torch.long)
    output = _chunk_output(
        qg,
        v_new,
        aqk,
        h,
        v,
        metadata,
        chunk_indices,
        is_varlen,
    )
    if output_intermediate_states:
        return output, h
    return output
