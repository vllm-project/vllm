# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MXFP8 GEMM on 32x32 block-scaled weights for gfx950 (``tl.dot_scaled``).

``y = x @ w.T`` with an MXFP8 activation (e4m3 values, one E8M0 scale per
row and 32 K, ``[M, K / 32]``) and an e4m3 weight whose E8M0 scales stay in
the checkpoint's 32x32 blocks, ``[N / 32, K / 32]``, instead of being
expanded to every row. Each weight scale byte is then read once per 32 output
rows, and small-M shapes can use the packed kernel below.

Two kernels, picked per shape from a table tuned on MI355X:

* a tiled kernel with optional split-K, reduced by a second, deterministic
  launch rather than atomics;
* a packed kernel for small M, which fills the MFMA's rows with K panels
  instead of tokens and keeps only the matching-panel (block-diagonal)
  products, so a handful of tokens still streams the weight at full width.
"""

from typing import NamedTuple

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

BLOCK_ROWS = 32


@triton.jit(do_not_specialize=["M"])
def _block32_tiled_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    N_FIRST: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # N_FIRST launches output-column tiles fastest, so consecutive programs
    # share an activation tile.
    if N_FIRST:
        pid_n, pid_m = tl.program_id(0), tl.program_id(1)
    else:
        pid_m, pid_n = tl.program_id(0), tl.program_id(1)
    pid_k = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    n_mask = offs_n < N
    k0 = pid_k * K_PER_SPLIT
    offs_k = k0 + tl.arange(0, BLOCK_K)
    offs_sk = k0 // 32 + tl.arange(0, BLOCK_K // 32)
    x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
    xs_ptrs = xs_ptr + offs_m[:, None] * (K // 32) + offs_sk[None, :]
    w_ptrs = w_ptr + offs_n[:, None] * K + offs_k[None, :]
    ws_ptrs = ws_ptr + (offs_n[:, None] // 32) * (K // 32) + offs_sk[None, :]
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(0, K_PER_SPLIT, BLOCK_K):
        if EVEN_K:
            x = tl.load(x_ptrs, mask=m_mask[:, None], other=0.0)
            xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=127)
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
            ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=127)
        else:
            k_ok = (offs_k + kk) < K
            s_ok = (offs_sk + kk // 32) < K // 32
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_ok[None, :], other=0.0)
            xs = tl.load(xs_ptrs, mask=m_mask[:, None] & s_ok[None, :], other=127)
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_ok[None, :], other=0.0)
            ws = tl.load(ws_ptrs, mask=n_mask[:, None] & s_ok[None, :], other=127)
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
        xs_ptrs += BLOCK_K // 32
        ws_ptrs += BLOCK_K // 32
    out_ptrs = out_ptr + pid_k * M * N + offs_m[:, None] * stride_om + offs_n[None, :]
    tl.store(
        out_ptrs,
        acc.to(out_ptr.dtype.element_ty),
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit(do_not_specialize=["M"])
def _block32_packed_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PACK: tl.constexpr,
    EVEN_K: tl.constexpr,
):
    # Tile row r is token r // K_PACK reading K panel r % K_PACK (and likewise
    # for weight rows), so one dot covers K_PACK consecutive BLOCK_K slices.
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M * K_PACK) // K_PACK
    x_panel = tl.arange(0, BLOCK_M * K_PACK) % K_PACK
    cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N * K_PACK) // K_PACK
    w_panel = tl.arange(0, BLOCK_N * K_PACK) % K_PACK
    ks = tl.arange(0, BLOCK_K)
    gs = tl.arange(0, BLOCK_K // 32)
    row_ok = rows < M
    col_ok = cols < N
    acc = tl.zeros((BLOCK_M * K_PACK, BLOCK_N * K_PACK), dtype=tl.float32)
    for base in range(0, K, BLOCK_K * K_PACK):
        xk = base + x_panel[:, None] * BLOCK_K + ks[None, :]
        wk = base + w_panel[:, None] * BLOCK_K + ks[None, :]
        xg = base // 32 + x_panel[:, None] * (BLOCK_K // 32) + gs[None, :]
        wg = base // 32 + w_panel[:, None] * (BLOCK_K // 32) + gs[None, :]
        x_mask = row_ok[:, None]
        w_mask = col_ok[:, None]
        xs_mask = row_ok[:, None]
        ws_mask = col_ok[:, None]
        if not EVEN_K:
            x_mask = x_mask & (xk < K)
            w_mask = w_mask & (wk < K)
            xs_mask = xs_mask & (xg < K // 32)
            ws_mask = ws_mask & (wg < K // 32)
        x = tl.load(x_ptr + rows[:, None] * K + xk, mask=x_mask, other=0.0)
        w = tl.load(w_ptr + cols[:, None] * K + wk, mask=w_mask, other=0.0)
        xs = tl.load(xs_ptr + rows[:, None] * (K // 32) + xg, mask=xs_mask, other=127)
        ws = tl.load(
            ws_ptr + (cols[:, None] // 32) * (K // 32) + wg, mask=ws_mask, other=127
        )
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
    # Keep only products of matching K panels: the block diagonal.
    blocks = acc.reshape(BLOCK_M, K_PACK, BLOCK_N, K_PACK).trans(0, 2, 1, 3)
    pair = tl.arange(0, K_PACK)
    diagonal = tl.where(
        pair[None, None, :, None] == pair[None, None, None, :], blocks, 0.0
    )
    y = tl.sum(tl.sum(diagonal, 3), 2)
    out_rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    tl.store(
        out_ptr + out_rows[:, None] * stride_om + out_cols[None, :],
        y.to(out_ptr.dtype.element_ty),
        mask=(out_rows[:, None] < M) & (out_cols[None, :] < N),
    )


@triton.jit(do_not_specialize=["M"])
def _block32_splitk_reduce_kernel(
    partial_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    stride_om,
    SPLIT_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for split in tl.static_range(SPLIT_K):
        acc += tl.load(
            partial_ptr + split * M * N + offs_m[:, None] * N + offs_n[None, :],
            mask=mask,
            other=0.0,
        )
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
        acc.to(out_ptr.dtype.element_ty),
        mask=mask,
    )


class _Config(NamedTuple):
    packed: bool
    block_m: int
    block_n: int
    block_k: int
    # K panels per MFMA tile (packed) or K partitions (tiled).
    k_split: int
    n_first: bool
    num_warps: int
    num_stages: int
    matrix_instr_nonkdim: int
    waves_per_eu: int


def _tiled(bm, bn, bk, split_k, n_first, warps, stages, nonkdim, waves) -> _Config:
    return _Config(
        False, bm, bn, bk, split_k, bool(n_first), warps, stages, nonkdim, waves
    )


def _packed(bm, bn, bk, k_pack, warps, stages, nonkdim, waves) -> _Config:
    return _Config(True, bm, bn, bk, k_pack, False, warps, stages, nonkdim, waves)


# (N, K) -> [(max M, config), ...] for the DeepSeek-V4.1-Flash MXFP8 linears at
# TP2/TP4, tuned on MI355X under HIP graphs with weights cold in the LLC. A
# tier serves every M up to its bound; larger M reuse the last tier.
_TUNED: dict[tuple[int, int], list[tuple[int, _Config]]] = {
    # shared expert gate_up, TP4
    (1152, 5120): [
        (1, _packed(8, 16, 1024, 2, 2, 3, 16, 0)),
        (8, _packed(8, 16, 1024, 2, 2, 3, 16, 0)),
        (32, _tiled(32, 32, 512, 8, True, 8, 3, 16, 1)),
        (64, _tiled(64, 32, 512, 8, False, 8, 3, 16, 0)),
        (128, _tiled(32, 32, 1024, 1, True, 4, 2, 16, 1)),
        (192, _tiled(32, 32, 1024, 1, True, 4, 2, 16, 0)),
        (256, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (320, _tiled(128, 64, 256, 4, True, 8, 3, 32, 0)),
        (384, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (512, _tiled(64, 64, 512, 1, True, 4, 2, 32, 0)),
        (768, _tiled(64, 64, 512, 1, True, 4, 2, 32, 0)),
        (1024, _tiled(128, 128, 256, 2, False, 4, 2, 16, 1)),
        (1536, _tiled(128, 128, 256, 2, False, 4, 2, 16, 0)),
        (2048, _tiled(128, 128, 256, 1, False, 4, 2, 32, 0)),
        (3072, _tiled(128, 128, 256, 1, False, 4, 2, 32, 1)),
        (4096, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (8192, _tiled(128, 128, 256, 1, False, 4, 2, 32, 1)),
    ],
    # fused wq_a|wkv (replicated)
    (1792, 5120): [
        (1, _packed(8, 16, 1024, 2, 2, 3, 16, 0)),
        (8, _packed(8, 16, 1024, 2, 2, 3, 16, 1)),
        (32, _packed(16, 16, 1024, 1, 2, 3, 16, 1)),
        (64, _tiled(32, 16, 1024, 1, True, 2, 3, 16, 0)),
        (128, _tiled(32, 32, 1024, 1, True, 4, 2, 16, 1)),
        (192, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (256, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (320, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (384, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (512, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (768, _tiled(128, 128, 256, 2, True, 4, 2, 16, 0)),
        (1024, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (1536, _tiled(128, 128, 256, 1, False, 4, 2, 32, 1)),
        (2048, _tiled(128, 128, 256, 1, False, 4, 2, 32, 1)),
        (3072, _tiled(128, 128, 256, 1, False, 8, 2, 32, 1)),
        (4096, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (8192, _tiled(128, 128, 128, 1, False, 8, 2, 16, 1)),
    ],
    # shared expert gate_up, TP2
    (2304, 5120): [
        (1, _packed(8, 16, 1024, 2, 2, 3, 16, 1)),
        (8, _packed(8, 16, 1024, 2, 2, 3, 16, 0)),
        (32, _tiled(32, 64, 256, 8, False, 4, 3, 16, 0)),
        (64, _tiled(32, 32, 1024, 1, True, 4, 2, 16, 1)),
        (128, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (192, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (256, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (320, _tiled(64, 64, 512, 1, True, 4, 2, 32, 0)),
        (384, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (512, _tiled(128, 128, 256, 2, False, 4, 2, 16, 1)),
        (768, _tiled(128, 128, 256, 2, True, 4, 2, 16, 0)),
        (1024, _tiled(128, 128, 256, 1, False, 4, 2, 32, 0)),
        (1536, _tiled(128, 128, 256, 1, False, 4, 2, 32, 0)),
        (2048, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (3072, _tiled(128, 128, 128, 1, True, 8, 2, 16, 0)),
        (4096, _tiled(128, 128, 256, 1, False, 4, 2, 32, 1)),
        (8192, _tiled(128, 128, 128, 1, False, 8, 2, 32, 0)),
    ],
    # indexer wq_b
    (4096, 1280): [
        (1, _packed(4, 16, 256, 4, 4, 2, 16, 1)),
        (8, _packed(8, 16, 256, 4, 4, 3, 16, 1)),
        (32, _packed(32, 16, 1024, 1, 2, 3, 16, 1)),
        (64, _tiled(32, 32, 512, 1, True, 4, 3, 16, 1)),
        (128, _tiled(64, 32, 512, 1, True, 8, 3, 16, 0)),
        (192, _tiled(64, 32, 256, 1, True, 4, 3, 16, 0)),
        (256, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (320, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (384, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (512, _tiled(128, 64, 256, 1, True, 8, 3, 32, 0)),
        (768, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (1024, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (1536, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (2048, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (3072, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (4096, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 16, 0)),
    ],
    # shared expert down, TP4
    (5120, 576): [
        (1, _packed(8, 32, 256, 2, 2, 2, 16, 1)),
        (8, _packed(8, 32, 256, 2, 2, 2, 16, 1)),
        (32, _packed(32, 32, 256, 1, 8, 3, 16, 1)),
        (64, _tiled(32, 32, 128, 1, True, 4, 3, 16, 0)),
        (128, _tiled(32, 32, 256, 1, True, 2, 2, 16, 0)),
        (192, _tiled(64, 64, 256, 1, True, 4, 3, 32, 0)),
        (256, _tiled(32, 64, 256, 1, True, 4, 2, 16, 0)),
        (320, _tiled(64, 64, 256, 1, True, 2, 2, 32, 0)),
        (384, _tiled(128, 128, 128, 1, False, 8, 3, 32, 1)),
        (512, _tiled(64, 64, 128, 1, True, 4, 2, 32, 0)),
        (768, _tiled(64, 256, 128, 1, True, 8, 3, 32, 0)),
        (1024, _tiled(128, 64, 128, 1, True, 4, 2, 32, 0)),
        (1536, _tiled(32, 128, 128, 1, True, 4, 2, 16, 0)),
        (2048, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (3072, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (4096, _tiled(128, 128, 128, 1, True, 8, 2, 32, 0)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
    ],
    # shared expert down, TP2
    (5120, 1152): [
        (1, _packed(4, 32, 128, 4, 2, 3, 16, 1)),
        (8, _packed(8, 32, 256, 2, 4, 3, 16, 0)),
        (32, _packed(32, 32, 512, 1, 2, 3, 16, 1)),
        (64, _tiled(32, 32, 128, 1, True, 4, 3, 16, 0)),
        (128, _tiled(32, 32, 128, 1, True, 4, 3, 16, 0)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (256, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (320, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (384, _tiled(128, 64, 256, 1, True, 8, 3, 32, 1)),
        (512, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (768, _tiled(128, 128, 256, 1, True, 4, 2, 32, 0)),
        (1024, _tiled(128, 64, 128, 1, True, 8, 2, 32, 0)),
        (1536, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (2048, _tiled(128, 128, 128, 1, True, 8, 2, 32, 0)),
        (3072, _tiled(128, 256, 128, 1, True, 2, 2, 32, 0)),
        (4096, _tiled(128, 128, 128, 1, True, 8, 2, 32, 0)),
        (8192, _tiled(128, 64, 128, 1, True, 8, 2, 32, 1)),
    ],
    # wo_b, TP4
    (5120, 2048): [
        (1, _packed(4, 32, 256, 4, 2, 2, 16, 0)),
        (8, _packed(8, 32, 256, 4, 2, 3, 16, 0)),
        (32, _tiled(32, 32, 512, 1, False, 4, 3, 16, 1)),
        (64, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (128, _tiled(64, 64, 512, 1, True, 4, 2, 32, 0)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (256, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (320, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (384, _tiled(128, 64, 256, 1, True, 8, 3, 32, 0)),
        (512, _tiled(128, 128, 256, 1, True, 4, 2, 32, 0)),
        (768, _tiled(128, 128, 256, 1, True, 4, 2, 32, 0)),
        (1024, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (1536, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (2048, _tiled(128, 128, 256, 1, True, 8, 2, 32, 1)),
        (3072, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (4096, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (8192, _tiled(128, 128, 128, 1, True, 2, 2, 32, 1)),
    ],
    # wo_b, TP2
    (5120, 4096): [
        (1, _packed(4, 32, 512, 4, 2, 2, 16, 0)),
        (8, _packed(8, 32, 1024, 2, 2, 3, 16, 0)),
        (32, _packed(32, 32, 1024, 1, 4, 2, 16, 0)),
        (64, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (128, _tiled(64, 64, 512, 1, True, 4, 2, 16, 0)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 32, 0)),
        (256, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (320, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (384, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (512, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (768, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (1024, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (1536, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (2048, _tiled(128, 128, 256, 1, True, 8, 2, 32, 1)),
        (3072, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (4096, _tiled(128, 128, 128, 1, True, 8, 2, 16, 1)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
    ],
    # wq_b, TP4
    (8192, 1280): [
        (1, _packed(8, 32, 256, 2, 4, 3, 16, 1)),
        (8, _packed(8, 32, 256, 2, 4, 3, 16, 0)),
        (32, _packed(32, 32, 256, 1, 2, 3, 16, 1)),
        (64, _packed(32, 64, 256, 1, 4, 3, 16, 1)),
        (128, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (192, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (256, _tiled(128, 64, 256, 1, True, 8, 3, 32, 1)),
        (320, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (384, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (512, _tiled(128, 128, 256, 1, True, 4, 2, 32, 0)),
        (768, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (1024, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (1536, _tiled(128, 128, 128, 1, True, 8, 2, 16, 0)),
        (2048, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (3072, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (4096, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (8192, _tiled(128, 256, 128, 1, True, 2, 2, 32, 0)),
    ],
    # wq_b, TP2
    (16384, 1280): [
        (1, _packed(4, 32, 128, 4, 2, 3, 16, 1)),
        (8, _packed(8, 32, 256, 2, 4, 2, 16, 0)),
        (32, _packed(32, 64, 256, 1, 4, 3, 16, 1)),
        (64, _tiled(64, 64, 256, 1, False, 4, 3, 32, 1)),
        (128, _tiled(128, 64, 256, 1, True, 8, 3, 32, 0)),
        (192, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (256, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (320, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (384, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (512, _tiled(128, 128, 128, 1, True, 8, 2, 32, 0)),
        (768, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (1024, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (1536, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (2048, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (3072, _tiled(128, 256, 128, 1, True, 2, 2, 32, 0)),
        (4096, _tiled(128, 256, 128, 1, True, 2, 2, 32, 0)),
        (8192, _tiled(128, 256, 128, 1, True, 2, 2, 32, 0)),
    ],
}


def _default_config(M: int, N: int, K: int) -> _Config:
    """Untuned shapes: the tiers most tuned shapes settle on."""
    if M <= 8:
        return _packed(8, 16, 256, 2, 2, 3, 16, 0)
    if M <= 32:
        return _packed(32, 32, 256, 1, 2, 3, 16, 0)
    if M <= 256:
        return _tiled(64, 64, 256, 1, True, 4, 2, 16, 0)
    return _tiled(128, 128, 128, 1, True, 4, 2, 16, 0)


def _config(M: int, N: int, K: int) -> _Config:
    tiers = _TUNED.get((N, K))
    if tiers is None:
        return _default_config(M, N, K)
    for max_m, cfg in tiers:
        if max_m >= M:
            return cfg
    return tiers[-1][1]


def _rocm_mxfp8_block32_gemm_impl(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), dtype=out_dtype, device=x.device)
    if M == 0:
        return out
    cfg = _config(M, N, K)
    opts = dict(
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        matrix_instr_nonkdim=cfg.matrix_instr_nonkdim,
        waves_per_eu=cfg.waves_per_eu,
    )
    if cfg.packed:
        span = cfg.block_k * cfg.k_split
        _block32_packed_kernel[
            (triton.cdiv(M, cfg.block_m), triton.cdiv(N, cfg.block_n))
        ](
            x,
            x_scale,
            weight,
            weight_scale,
            out,
            M,
            N,
            K,
            out.stride(0),
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            BLOCK_K=cfg.block_k,
            K_PACK=cfg.k_split,
            EVEN_K=K % span == 0,
            **opts,
        )
        return out
    k_per_split = triton.cdiv(triton.cdiv(K, cfg.k_split), cfg.block_k) * cfg.block_k
    split_k = triton.cdiv(K, k_per_split)
    grid_m, grid_n = triton.cdiv(M, cfg.block_m), triton.cdiv(N, cfg.block_n)
    grid = (grid_n, grid_m, split_k) if cfg.n_first else (grid_m, grid_n, split_k)
    target = (
        out
        if split_k == 1
        else torch.empty((split_k, M, N), dtype=torch.float32, device=x.device)
    )
    _block32_tiled_kernel[grid](
        x,
        x_scale,
        weight,
        weight_scale,
        target,
        M,
        N,
        K,
        out.stride(0) if split_k == 1 else N,
        BLOCK_M=cfg.block_m,
        BLOCK_N=cfg.block_n,
        BLOCK_K=cfg.block_k,
        K_PER_SPLIT=k_per_split,
        N_FIRST=cfg.n_first,
        EVEN_K=K % k_per_split == 0,
        **opts,
    )
    if split_k > 1:
        _block32_splitk_reduce_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 64))](
            target,
            out,
            M,
            N,
            out.stride(0),
            SPLIT_K=split_k,
            BLOCK_M=32,
            BLOCK_N=64,
        )
    return out


def _rocm_mxfp8_block32_gemm_fake(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=out_dtype)


# An opaque op: the kernel and tile choice branch on the token count, which a
# compiled caller must not freeze at its trace-time value.
direct_register_custom_op(
    op_name="rocm_mxfp8_block32_gemm",
    op_func=_rocm_mxfp8_block32_gemm_impl,
    fake_impl=_rocm_mxfp8_block32_gemm_fake,
)


def rocm_mxfp8_block32_gemm(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``x @ weight.T`` for MXFP8 ``x`` and a 32x32 block-scaled MXFP8 weight.

    Args:
        x: [M, K] e4m3 activation, contiguous.
        x_scale: [M, K / 32] E8M0 (uint8) activation scales.
        weight: [N, K] e4m3 weight, contiguous.
        weight_scale: [ceil(N / 32), K / 32] E8M0 (uint8) weight block scales.
        out_dtype: Output dtype.

    Returns:
        The [M, N] product.

    """
    return torch.ops.vllm.rocm_mxfp8_block32_gemm(
        x, x_scale, weight, weight_scale, out_dtype
    )
