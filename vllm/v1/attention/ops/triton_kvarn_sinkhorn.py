# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused log-domain iterative variance-normalization for KVarN.

Matches the PyTorch reference in
``vllm/model_executor/layers/quantization/kvarn/sinkhorn.py`` semantically —
same 16 alternating col/row std-normalization passes, same best-so-far
tracking via the imbalance metric, same clamps. One Triton program per
``[R, C]`` tile; the grid dim is the number of tiles in the batch.

For ``R = C = 128`` the full tile is 64 KB fp32 — fits in a single Triton
block's register/SMEM budget on current GPUs.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton

_CLIP_STD_MIN = 1e-3
_CLIP_STD_MAX = 1e3
_LOG_S_MIN = -0.3
_LOG_S_MAX = 10.0


@triton.jit
def _sinkhorn_log_kernel(
    Tile_ptr,  # [N, R, C] fp32 input — rotated tile
    Balanced_ptr,  # [N, R, C] fp32 output
    SCol_ptr,  # [N, C] fp32 output (s_col, per-column)
    SRow_ptr,  # [N, R] fp32 output (s_row, per-row)
    # Strides
    stride_tn,
    stride_tr,
    stride_bn,
    stride_br,
    stride_sc_n,
    stride_sr_n,
    # Dims
    R: tl.constexpr,
    C: tl.constexpr,
    ITERATIONS: tl.constexpr,
    # Algorithm params (kept as tl.constexpr for the compiler)
    CLIP_STD_MIN: tl.constexpr,
    CLIP_STD_MAX: tl.constexpr,
    LOG_S_MIN: tl.constexpr,
    LOG_S_MAX: tl.constexpr,
):
    """One program per tile. Loads a R x C tile into registers, does
    ``ITERATIONS`` alternating col/row log-domain normalizations, tracks
    the best-so-far (lowest-imbalance) scales, and writes (balanced, s_col,
    s_row).
    """
    pid = tl.program_id(0)

    r_offs = tl.arange(0, R)
    c_offs = tl.arange(0, C)

    # Load tile [R, C] into registers
    tile_base = pid * stride_tn
    tile_ptrs = Tile_ptr + tile_base + r_offs[:, None] * stride_tr + c_offs[None, :]
    tile = tl.load(tile_ptrs).to(tl.float32)

    # log_s_col [C], log_s_row [R]; initialised at zero (exp = 1)
    log_s_col = tl.zeros([C], dtype=tl.float32)
    log_s_row = tl.zeros([R], dtype=tl.float32)

    # cur = tile / s_col / s_row = tile (with mu = 1 initially)
    cur = tile

    # ── initial imbalance + best snapshot ─────────────────────────────────
    col_mean0 = tl.sum(cur, axis=0) / R
    col_var0 = tl.sum(cur * cur, axis=0) / R - col_mean0 * col_mean0
    col_std0 = tl.sqrt(tl.maximum(col_var0 * R / (R - 1), 0.0))
    row_mean0 = tl.sum(cur, axis=1) / C
    row_var0 = tl.sum(cur * cur, axis=1) / C - row_mean0 * row_mean0
    row_std0 = tl.sqrt(tl.maximum(row_var0 * C / (C - 1), 0.0))

    col_max0 = tl.max(col_std0)
    col_min0 = tl.maximum(tl.min(col_std0), 1e-8)
    row_max0 = tl.max(row_std0)
    row_min0 = tl.maximum(tl.min(row_std0), 1e-8)
    imb_best = col_max0 / col_min0 + row_max0 / row_min0

    sc_best = tl.exp(log_s_col)  # ones[C]
    sr_best = tl.exp(log_s_row)  # ones[R]

    # ── iterations ────────────────────────────────────────────────────────
    for _ in tl.static_range(ITERATIONS):
        # Update column scales from cur's per-column std
        col_mean = tl.sum(cur, axis=0) / R
        col_var = tl.sum(cur * cur, axis=0) / R - col_mean * col_mean
        col_std = tl.sqrt(tl.maximum(col_var * R / (R - 1), 0.0))
        col_std_clipped = tl.maximum(tl.minimum(col_std, CLIP_STD_MAX), CLIP_STD_MIN)
        log_s_col = log_s_col + tl.log(col_std_clipped)
        log_s_col = tl.maximum(tl.minimum(log_s_col, LOG_S_MAX), LOG_S_MIN)
        s_col_lin = tl.exp(log_s_col)
        s_row_lin = tl.exp(log_s_row)
        cur = tile / s_col_lin[None, :] / s_row_lin[:, None]

        # Update row scales from new cur's per-row std
        row_mean = tl.sum(cur, axis=1) / C
        row_var = tl.sum(cur * cur, axis=1) / C - row_mean * row_mean
        row_std = tl.sqrt(tl.maximum(row_var * C / (C - 1), 0.0))
        row_std_clipped = tl.maximum(tl.minimum(row_std, CLIP_STD_MAX), CLIP_STD_MIN)
        log_s_row = log_s_row + tl.log(row_std_clipped)
        log_s_row = tl.maximum(tl.minimum(log_s_row, LOG_S_MAX), LOG_S_MIN)
        s_col_lin = tl.exp(log_s_col)
        s_row_lin = tl.exp(log_s_row)
        cur = tile / s_col_lin[None, :] / s_row_lin[:, None]

        # Imbalance + best-so-far update
        col_mean_n = tl.sum(cur, axis=0) / R
        col_var_n = tl.sum(cur * cur, axis=0) / R - col_mean_n * col_mean_n
        col_std_n = tl.sqrt(tl.maximum(col_var_n * R / (R - 1), 0.0))
        row_mean_n = tl.sum(cur, axis=1) / C
        row_var_n = tl.sum(cur * cur, axis=1) / C - row_mean_n * row_mean_n
        row_std_n = tl.sqrt(tl.maximum(row_var_n * C / (C - 1), 0.0))
        col_max_n = tl.max(col_std_n)
        col_min_n = tl.maximum(tl.min(col_std_n), 1e-8)
        row_max_n = tl.max(row_std_n)
        row_min_n = tl.maximum(tl.min(row_std_n), 1e-8)
        imb = col_max_n / col_min_n + row_max_n / row_min_n

        better = imb <= imb_best
        sc_best = tl.where(better, s_col_lin, sc_best)
        sr_best = tl.where(better, s_row_lin, sr_best)
        imb_best = tl.where(better, imb, imb_best)

    # ── final: balanced = tile / sc_best / sr_best, write outputs ─────────
    balanced = tile / sc_best[None, :] / sr_best[:, None]
    bal_ptrs = (
        Balanced_ptr + pid * stride_bn + r_offs[:, None] * stride_br + c_offs[None, :]
    )
    tl.store(bal_ptrs, balanced)
    tl.store(SCol_ptr + pid * stride_sc_n + c_offs, sc_best)
    tl.store(SRow_ptr + pid * stride_sr_n + r_offs, sr_best)


def kvarn_sinkhorn_triton(
    tiles: torch.Tensor,
    iterations: int = 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton driver for ``_sinkhorn_log_kernel``.

    Args:
        tiles: ``[N, R, C]`` fp32 (or any real dtype, cast inside). Both R
            and C must be compile-time-constant power-of-2 values; we hard-
            code R = C = 128 for the first PR.
        iterations: number of alternating col/row passes (default 16).

    Returns:
        balanced: ``[N, R, C]`` fp32.
        s_col:    ``[N, C]`` fp32.
        s_row:    ``[N, R]`` fp32.
    """
    assert tiles.ndim == 3
    N, R, C = tiles.shape
    tiles = tiles.contiguous().to(torch.float32)
    device = tiles.device

    # The Triton kernel loads the WHOLE [R, C] tile into one program's registers
    # and unrolls the iteration loop. At large head_dim that tile is huge (e.g.
    # head_dim 512 -> [512, 128] = 256 KB) and the Triton compiler hangs/explodes
    # (128/256 compile fine). Route large tiles to the batched PyTorch Sinkhorn
    # (identical algorithm). Flush is infrequent + off the decode hot path, so the
    # cost is fine; head_dim<=256 keeps the fast kernel.
    if max(R, C) > 256:
        from vllm.model_executor.layers.quantization.kvarn.sinkhorn import (
            variance_normalize_batched,
        )

        bal, s_col_b, s_row_b = variance_normalize_batched(tiles, iterations=iterations)
        return (
            bal.contiguous(),
            s_col_b.reshape(N, C).contiguous(),
            s_row_b.reshape(N, R).contiguous(),
        )

    balanced = torch.empty(N, R, C, dtype=torch.float32, device=device)
    s_col = torch.empty(N, C, dtype=torch.float32, device=device)
    s_row = torch.empty(N, R, dtype=torch.float32, device=device)

    _sinkhorn_log_kernel[(N,)](
        tiles,
        balanced,
        s_col,
        s_row,
        tiles.stride(0),
        tiles.stride(1),
        balanced.stride(0),
        balanced.stride(1),
        s_col.stride(0),
        s_row.stride(0),
        R=R,
        C=C,
        ITERATIONS=iterations,
        CLIP_STD_MIN=_CLIP_STD_MIN,
        CLIP_STD_MAX=_CLIP_STD_MAX,
        LOG_S_MIN=_LOG_S_MIN,
        LOG_S_MAX=_LOG_S_MAX,
        # num_warps=8, not 4: the program keeps the whole [R, C] fp32 tile (plus
        # a working copy) live, so at 4 warps the per-thread footprint is several
        # KB of registers -> the compiler spills to CUDA local memory, and the
        # driver permanently reserves local_bytes x max_threads x num_SMs of
        # device memory for the context (~2 GiB on a 188-SM part for the
        # [256, 128] tile; a missing-KV-capacity component). 8 warps
        # halves the per-thread footprint: ~70% less reserved local memory AND
        # ~4x faster flush (the spills were also the kernel's bottleneck).
        # Balanced-tile output is unchanged within fp32 reduction noise (~5e-7
        # rel); 16 warps saves a bit more memory but is 2x slower than 8.
        num_warps=8,
        num_stages=2,
    )
    return balanced, s_col, s_row
