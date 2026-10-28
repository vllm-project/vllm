# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Portions adapted from ROCm/aiter (https://github.com/ROCm/aiter/pull/5750):
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""MXFP8 GEMM on 32x32 block-scaled weights for gfx950 (``tl.dot_scaled``).

``y = x @ w.T`` with an MXFP8 activation (e4m3 values, one E8M0 scale per
row and 32 K, ``[M, K / 32]``) and an e4m3 weight whose E8M0 scales stay in
the checkpoint's 32x32 blocks, ``[N / 32, K / 32]``, instead of being
expanded to every row. Each weight scale byte is then read once per 32 output
rows, and small-M shapes can use the packed kernel below.

Two kernels, picked per shape from a table tuned on MI355X:

* a tiled kernel for larger M;
* a packed kernel for small M, which fills the MFMA's rows with K panels
  instead of tokens and keeps only the matching-panel (block-diagonal)
  products, so a handful of tokens still streams the weight at full width.

Either can split K. The partials are normally summed in the same launch by
the last program of each output tile to finish, in split order, so the result
does not depend on scheduling. Otherwise a second launch reduces them.

The packed kernel and the in-launch split-K reduction on one XCD
(``_split_tile``, ``_sum_splits``, ``_split_counters``, ``_k_partition``) are
adapted from the group32 GEMM in ROCm/aiter#5750.
"""

from typing import NamedTuple

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
    mxfp8_e4m3_quantize,
)
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

BLOCK_ROWS = 32

# gfx950 hands out workgroups to its XCDs round-robin by program id, and 8 is
# a multiple of the XCD count in every partition mode. _split_tile hardcodes
# the same stride.
_XCD_STRIDE = 8
# One arrival counter per output tile of an in-launch split-K GEMM.
_SPLIT_COUNTERS = 1 << 16


@triton.jit
def _split_tile(N: tl.constexpr, BLOCK_N: tl.constexpr, SPLITS: tl.constexpr):
    """(pid_m, pid_n, tile, split) of a 1D grid that keeps a tile's splits on
    one XCD, so they meet in that XCD's L2."""
    pid = tl.program_id(0)
    tile = pid // (8 * SPLITS) * 8 + pid % 8
    grid_n: tl.constexpr = (N + BLOCK_N - 1) // BLOCK_N
    return tile // grid_n, tile % grid_n, tile, pid // 8 % SPLITS


@triton.jit
def _sum_splits(
    acc, out_ptrs, out_mask, slot_ptr, count_ptr, split, SPLITS: tl.constexpr
):
    """Publish this split's partial; the tile's last arrival sums all of them
    in split order and re-arms the counter."""
    BM: tl.constexpr = acc.shape[0]
    BN: tl.constexpr = acc.shape[1]
    local = tl.arange(0, BM)[:, None] * BN + tl.arange(0, BN)[None, :]
    tl.store(slot_ptr + split * BM * BN + local, acc)
    # The partial must be in L2 before the arrival is counted. The splits
    # share an XCD, so no device-scope release (an L2 writeback) is needed.
    tl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)", "=v,v", [split], dtype=tl.int32, is_pure=False, pack=1
    )
    tl.debug_barrier()
    if tl.atomic_add(count_ptr, 1, sem="acq_rel", scope="cta") == SPLITS - 1:
        total = tl.zeros((BM, BN), dtype=tl.float32)
        for s in tl.static_range(SPLITS):
            total += tl.load(slot_ptr + s * BM * BN + local, cache_modifier=".cv")
        tl.store(out_ptrs, total.to(out_ptrs.dtype.element_ty), mask=out_mask)
        tl.store(count_ptr, 0)


@triton.jit(do_not_specialize=["M"])
def _block32_tiled_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    slot_ptr,
    count_ptr,
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
    FUSED_SPLITS: tl.constexpr,
):
    if FUSED_SPLITS > 1:
        pid_m, pid_n, tile, pid_k = _split_tile(N, BLOCK_N, FUSED_SPLITS)
        if pid_m * BLOCK_M >= M:
            return
    else:
        # N_FIRST launches output-column tiles fastest, so consecutive
        # programs share an activation tile.
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
            xs = tl.load(xs_ptrs, mask=m_mask[:, None], other=0)
            w = tl.load(w_ptrs, mask=n_mask[:, None], other=0.0)
            ws = tl.load(ws_ptrs, mask=n_mask[:, None], other=0)
        else:
            k_ok = (offs_k + kk) < K
            s_ok = (offs_sk + kk // 32) < K // 32
            x = tl.load(x_ptrs, mask=m_mask[:, None] & k_ok[None, :], other=0.0)
            xs = tl.load(xs_ptrs, mask=m_mask[:, None] & s_ok[None, :], other=0)
            w = tl.load(w_ptrs, mask=n_mask[:, None] & k_ok[None, :], other=0.0)
            ws = tl.load(ws_ptrs, mask=n_mask[:, None] & s_ok[None, :], other=0)
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
        x_ptrs += BLOCK_K
        w_ptrs += BLOCK_K
        xs_ptrs += BLOCK_K // 32
        ws_ptrs += BLOCK_K // 32
    out_mask = m_mask[:, None] & n_mask[None, :]
    if FUSED_SPLITS > 1:
        _sum_splits(
            acc,
            out_ptr + offs_m[:, None] * stride_om + offs_n[None, :],
            out_mask,
            slot_ptr + tile * (FUSED_SPLITS * BLOCK_M * BLOCK_N),
            count_ptr + tile,
            pid_k,
            FUSED_SPLITS,
        )
    else:
        out_ptrs = (
            out_ptr + pid_k * M * N + offs_m[:, None] * stride_om + offs_n[None, :]
        )
        tl.store(out_ptrs, acc.to(out_ptr.dtype.element_ty), mask=out_mask)


@triton.jit(do_not_specialize=["M"])
def _block32_packed_kernel(
    x_ptr,
    xs_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    slot_ptr,
    count_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PACK: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    EVEN_K: tl.constexpr,
    FUSED_SPLITS: tl.constexpr,
):
    if FUSED_SPLITS > 1:
        pid_m, pid_n, tile, pid_k = _split_tile(N, BLOCK_N, FUSED_SPLITS)
        if pid_m * BLOCK_M >= M:
            return
        k_start = pid_k * K_PER_SPLIT
        k_stop = tl.minimum(k_start + K_PER_SPLIT, K)
    else:
        pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        if K_PER_SPLIT >= K:
            k_start = 0
            k_stop = K
        else:
            k_start = pid_k * K_PER_SPLIT
            k_stop = tl.minimum(k_start + K_PER_SPLIT, K)
    # Tile row r is token r // K_PACK reading K panel r % K_PACK (and likewise
    # for weight rows), so one dot covers K_PACK consecutive BLOCK_K slices.
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M * K_PACK) // K_PACK
    x_panel = tl.arange(0, BLOCK_M * K_PACK) % K_PACK
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N * K_PACK) // K_PACK
    w_panel = tl.arange(0, BLOCK_N * K_PACK) % K_PACK
    ks = tl.arange(0, BLOCK_K)
    gs = tl.arange(0, BLOCK_K // 32)
    row_ok = rows < M
    col_ok = cols < N
    acc = tl.zeros((BLOCK_M * K_PACK, BLOCK_N * K_PACK), dtype=tl.float32)
    for base in range(k_start, k_stop, BLOCK_K * K_PACK):
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
        xs = tl.load(xs_ptr + rows[:, None] * (K // 32) + xg, mask=xs_mask, other=0)
        ws = tl.load(
            ws_ptr + (cols[:, None] // 32) * (K // 32) + wg, mask=ws_mask, other=0
        )
        acc = tl.dot_scaled(x, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
    # Keep only products of matching K panels: the block diagonal.
    blocks = acc.reshape(BLOCK_M, K_PACK, BLOCK_N, K_PACK).trans(0, 2, 1, 3)
    pair = tl.arange(0, K_PACK)
    diagonal = tl.where(
        pair[None, None, :, None] == pair[None, None, None, :], blocks, 0.0
    )
    y = tl.sum(tl.sum(diagonal, 3), 2)
    out_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + out_rows[:, None] * stride_om + out_cols[None, :]
    out_mask = (out_rows[:, None] < M) & (out_cols[None, :] < N)
    if FUSED_SPLITS > 1:
        _sum_splits(
            y,
            out_ptrs,
            out_mask,
            slot_ptr + tile * (FUSED_SPLITS * BLOCK_M * BLOCK_N),
            count_ptr + tile,
            pid_k,
            FUSED_SPLITS,
        )
    else:
        tl.store(
            out_ptrs + pid_k * M * N, y.to(out_ptr.dtype.element_ty), mask=out_mask
        )


# _block32_packed_kernel with the MXFP8 activation quantizer folded into
# the K loop: x_ptr is the RAW bf16/fp16 [M, K] activation (row-major
# contiguous), quantized per 32-element block in registers with the exact
# _mxfp8_quant_triton_kernel arithmetic (amax -> biased E8M0 scale ->
# reciprocal rescale -> RN cast to e4m3), so the separate quant launch and
# the x_q/x_scale global round-trip disappear from the decode critical
# path. Tiling, K-panel packing, split-K reduction and stores are
# byte-for-byte the packed kernel, so the output is bit-identical to
# quantize-then-GEMM by construction (same quant arithmetic on the same
# values, same dot order).
@triton.jit(do_not_specialize=["M"])
def _block32_packed_quant_kernel(
    x_ptr,
    w_ptr,
    ws_ptr,
    out_ptr,
    slot_ptr,
    count_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_om,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_PACK: tl.constexpr,
    K_PER_SPLIT: tl.constexpr,
    EVEN_K: tl.constexpr,
    FUSED_SPLITS: tl.constexpr,
    FP8_MAX: tl.constexpr,
    TINY: tl.constexpr,
):
    if FUSED_SPLITS > 1:
        pid_m, pid_n, tile, pid_k = _split_tile(N, BLOCK_N, FUSED_SPLITS)
        if pid_m * BLOCK_M >= M:
            return
        k_start = pid_k * K_PER_SPLIT
        k_stop = tl.minimum(k_start + K_PER_SPLIT, K)
    else:
        pid_m, pid_n, pid_k = tl.program_id(0), tl.program_id(1), tl.program_id(2)
        if K_PER_SPLIT >= K:
            k_start = 0
            k_stop = K
        else:
            k_start = pid_k * K_PER_SPLIT
            k_stop = tl.minimum(k_start + K_PER_SPLIT, K)
    # Tile row r is token r // K_PACK reading K panel r % K_PACK (and likewise
    # for weight rows), so one dot covers K_PACK consecutive BLOCK_K slices.
    rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M * K_PACK) // K_PACK
    x_panel = tl.arange(0, BLOCK_M * K_PACK) % K_PACK
    cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N * K_PACK) // K_PACK
    w_panel = tl.arange(0, BLOCK_N * K_PACK) % K_PACK
    ks = tl.arange(0, BLOCK_K)
    gs = tl.arange(0, BLOCK_K // 32)
    row_ok = rows < M
    col_ok = cols < N
    acc = tl.zeros((BLOCK_M * K_PACK, BLOCK_N * K_PACK), dtype=tl.float32)
    for base in range(k_start, k_stop, BLOCK_K * K_PACK):
        xk = base + x_panel[:, None] * BLOCK_K + ks[None, :]
        wk = base + w_panel[:, None] * BLOCK_K + ks[None, :]
        wg = base // 32 + w_panel[:, None] * (BLOCK_K // 32) + gs[None, :]
        x_mask = row_ok[:, None]
        w_mask = col_ok[:, None]
        ws_mask = col_ok[:, None]
        if not EVEN_K:
            x_mask = x_mask & (xk < K)
            w_mask = w_mask & (wk < K)
            ws_mask = ws_mask & (wg < K // 32)
        # Fused MXFP8 quant, verbatim _mxfp8_quant_triton_kernel arithmetic.
        # K is a multiple of 32 and every 32-block of the tile is 32-aligned,
        # so each block is either fully inside K or fully masked (loaded 0.0,
        # scale byte 0) -- the same values the separate quantizer produces
        # for these rows. Masked rows likewise load 0.0 and quantize to
        # scale byte 0 / value 0, matching the kernel's other=0 loads.
        x = tl.load(x_ptr + rows[:, None] * K + xk, mask=x_mask, other=0.0).to(
            tl.float32
        )
        xb = tl.reshape(x, (BLOCK_M * K_PACK, BLOCK_K // 32, 32))
        amax = tl.maximum(tl.max(tl.abs(xb), axis=2), TINY)
        sb = tl.ceil(tl.log2(amax / FP8_MAX)) + 127.0
        sb = tl.minimum(tl.maximum(sb, 0.0), 254.0)
        rescale = tl.exp2(127.0 - sb)
        xq = tl.reshape(xb * rescale[:, :, None], (BLOCK_M * K_PACK, BLOCK_K)).to(
            w_ptr.dtype.element_ty
        )
        xs = sb.to(tl.uint8)
        w = tl.load(w_ptr + cols[:, None] * K + wk, mask=w_mask, other=0.0)
        ws = tl.load(
            ws_ptr + (cols[:, None] // 32) * (K // 32) + wg, mask=ws_mask, other=0
        )
        acc = tl.dot_scaled(xq, xs, "e4m3", w.T, ws, "e4m3", acc=acc)
    # Keep only products of matching K panels: the block diagonal.
    blocks = acc.reshape(BLOCK_M, K_PACK, BLOCK_N, K_PACK).trans(0, 2, 1, 3)
    pair = tl.arange(0, K_PACK)
    diagonal = tl.where(
        pair[None, None, :, None] == pair[None, None, None, :], blocks, 0.0
    )
    y = tl.sum(tl.sum(diagonal, 3), 2)
    out_rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    out_cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_ptrs = out_ptr + out_rows[:, None] * stride_om + out_cols[None, :]
    out_mask = (out_rows[:, None] < M) & (out_cols[None, :] < N)
    if FUSED_SPLITS > 1:
        _sum_splits(
            y,
            out_ptrs,
            out_mask,
            slot_ptr + tile * (FUSED_SPLITS * BLOCK_M * BLOCK_N),
            count_ptr + tile,
            pid_k,
            FUSED_SPLITS,
        )
    else:
        tl.store(
            out_ptrs + pid_k * M * N, y.to(out_ptr.dtype.element_ty), mask=out_mask
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
    # K panels per MFMA tile (packed) or K partitions reduced by a second
    # launch (tiled).
    k_split: int
    n_first: bool
    num_warps: int
    num_stages: int
    matrix_instr_nonkdim: int
    waves_per_eu: int
    # K partitions summed within the launch.
    fused_splits: int = 1


def _tiled(
    bm, bn, bk, split_k, n_first, warps, stages, nonkdim, waves, fused=1
) -> _Config:
    return _Config(
        False, bm, bn, bk, split_k, bool(n_first), warps, stages, nonkdim, waves, fused
    )


def _packed(bm, bn, bk, k_pack, warps, stages, nonkdim, waves, fused=1) -> _Config:
    return _Config(
        True, bm, bn, bk, k_pack, False, warps, stages, nonkdim, waves, fused
    )


# (N, K) -> [(max M, config), ...] for the DeepSeek-V4.1-Flash MXFP8
# linears at TP1/TP2/TP4, tuned on MI355X under HIP graphs. A tier serves
# every M up to its bound; larger M reuse the last tier.
_TUNED: dict[tuple[int, int], list[tuple[int, _Config]]] = {
    # shared expert gate_up, TP4
    (1152, 5120): [
        (1, _packed(8, 16, 1024, 2, 2, 2, 16, 1, 3)),
        (2, _packed(4, 32, 256, 4, 2, 2, 16, 1, 5)),
        (4, _packed(4, 32, 256, 4, 2, 2, 16, 0, 5)),
        (8, _packed(8, 16, 1024, 2, 2, 2, 16, 0, 3)),
        (16, _tiled(16, 32, 512, 1, True, 2, 3, 16, 0, 5)),
        (24, _tiled(32, 32, 512, 1, True, 4, 3, 16, 1, 5)),
        (32, _packed(32, 32, 1024, 1, 4, 2, 16, 0, 5)),
        (48, _tiled(64, 32, 512, 1, True, 4, 3, 16, 1, 5)),
        (64, _tiled(64, 32, 512, 1, True, 8, 3, 16, 1, 5)),
        (96, _tiled(32, 16, 1024, 1, True, 2, 3, 16, 1)),
        (128, _tiled(64, 32, 512, 1, True, 8, 2, 16, 0, 3)),
        (160, _tiled(64, 32, 512, 1, True, 8, 3, 16, 0, 2)),
        (192, _tiled(32, 32, 1024, 1, True, 4, 2, 16, 0)),
        (256, _tiled(64, 32, 256, 1, True, 4, 3, 16, 0, 3)),
        (320, _tiled(64, 32, 256, 1, True, 8, 3, 16, 0, 2)),
        (384, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (512, _tiled(64, 64, 256, 1, True, 4, 2, 16, 0, 3)),
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
        (16, _tiled(16, 64, 1024, 1, True, 4, 2, 16, 0, 5)),
        (32, _packed(16, 16, 1024, 1, 2, 3, 16, 1)),
        (48, _tiled(64, 32, 512, 1, True, 8, 2, 16, 0, 4)),
        (64, _tiled(64, 32, 512, 1, True, 8, 3, 16, 1, 4)),
        (96, _tiled(64, 32, 512, 1, True, 8, 3, 16, 0, 2)),
        (128, _tiled(64, 32, 256, 1, True, 4, 3, 16, 0, 4)),
        (192, _tiled(64, 32, 256, 1, True, 4, 3, 16, 0, 3)),
        (256, _tiled(64, 64, 512, 1, True, 4, 2, 16, 1, 2)),
        (384, _tiled(64, 64, 256, 1, True, 4, 2, 16, 0, 3)),
        (512, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (768, _tiled(128, 128, 256, 1, True, 4, 2, 16, 1, 2)),
        (1024, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (2048, _tiled(128, 128, 256, 1, False, 4, 2, 32, 1)),
        (3072, _tiled(128, 128, 256, 1, False, 8, 2, 32, 1)),
        (4096, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (8192, _tiled(128, 128, 128, 1, False, 8, 2, 16, 1)),
    ],
    # shared expert gate_up, TP2
    (2304, 5120): [
        (1, _packed(8, 16, 1024, 2, 2, 3, 16, 1)),
        (8, _packed(8, 16, 1024, 2, 2, 3, 16, 0)),
        (16, _tiled(16, 64, 1024, 1, True, 4, 3, 16, 1, 5)),
        (24, _packed(32, 32, 512, 1, 4, 2, 16, 0, 5)),
        (32, _packed(32, 32, 1024, 1, 4, 2, 16, 1, 3)),
        (48, _tiled(64, 32, 512, 1, True, 8, 3, 16, 0, 3)),
        (64, _tiled(64, 64, 512, 1, True, 8, 2, 16, 0, 5)),
        (128, _tiled(64, 32, 256, 1, True, 4, 3, 16, 0, 3)),
        (160, _tiled(64, 64, 512, 1, True, 8, 2, 16, 1, 2)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 16, 0, 2)),
        (256, _tiled(64, 64, 256, 1, True, 4, 2, 16, 0, 3)),
        (320, _tiled(64, 64, 512, 1, True, 4, 2, 32, 0)),
        (384, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (512, _tiled(128, 128, 256, 1, True, 8, 2, 16, 0, 3)),
        (768, _tiled(128, 128, 256, 1, True, 4, 2, 16, 1, 2)),
        (1536, _tiled(128, 128, 256, 1, False, 4, 2, 32, 0)),
        (2048, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (3072, _tiled(128, 128, 128, 1, True, 8, 2, 16, 0)),
        (4096, _tiled(128, 128, 256, 1, False, 4, 2, 32, 1)),
        (8192, _tiled(128, 128, 128, 1, False, 8, 2, 32, 0)),
    ],
    # indexer wq_b
    (4096, 1280): [
        (1, _packed(4, 16, 256, 4, 4, 2, 16, 1)),
        (2, _packed(4, 16, 256, 4, 4, 2, 16, 0)),
        (4, _packed(4, 16, 256, 4, 4, 2, 16, 1)),
        (8, _packed(8, 16, 256, 4, 4, 3, 16, 1)),
        (16, _packed(16, 16, 512, 2, 2, 3, 16, 1)),
        (32, _packed(32, 16, 1024, 1, 2, 3, 16, 1)),
        (64, _tiled(32, 32, 512, 1, True, 4, 3, 16, 1)),
        (128, _tiled(64, 32, 512, 1, True, 8, 3, 16, 0)),
        (192, _tiled(64, 32, 256, 1, True, 4, 3, 16, 0)),
        (256, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (384, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (512, _tiled(128, 64, 256, 1, True, 8, 3, 32, 0)),
        (1024, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (2048, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (3072, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (4096, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 16, 0)),
    ],
    # shared expert down, TP4
    (5120, 576): [
        (8, _packed(8, 32, 256, 2, 2, 2, 16, 1)),
        (16, _packed(16, 32, 512, 1, 2, 2, 16, 0)),
        (32, _packed(32, 32, 256, 1, 8, 3, 16, 1)),
        (64, _tiled(32, 32, 128, 1, True, 4, 3, 16, 0)),
        (96, _tiled(32, 64, 256, 1, True, 4, 3, 16, 0)),
        (128, _tiled(32, 32, 256, 1, True, 2, 2, 16, 0)),
        (192, _tiled(64, 64, 256, 1, True, 4, 3, 32, 0)),
        (256, _tiled(64, 64, 256, 1, True, 2, 3, 32, 1)),
        (320, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (384, _tiled(128, 64, 256, 1, True, 8, 3, 32, 1)),
        (512, _tiled(64, 64, 128, 1, True, 4, 2, 32, 0)),
        (768, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (1024, _tiled(128, 64, 128, 1, True, 4, 2, 32, 0)),
        (3072, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (4096, _tiled(128, 128, 128, 1, True, 8, 2, 32, 0)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
    ],
    # shared expert down, TP2
    (5120, 1152): [
        (1, _packed(4, 32, 128, 4, 2, 3, 16, 1)),
        (8, _packed(8, 32, 256, 2, 4, 3, 16, 0)),
        (32, _packed(32, 32, 512, 1, 2, 3, 16, 1)),
        (128, _tiled(32, 32, 128, 1, True, 4, 3, 16, 0)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
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
        (4, _packed(4, 32, 256, 4, 2, 2, 16, 1)),
        (8, _packed(8, 32, 256, 4, 2, 3, 16, 0)),
        (16, _tiled(16, 32, 1024, 1, True, 2, 3, 16, 0)),
        (32, _tiled(32, 32, 512, 1, False, 4, 3, 16, 1)),
        (64, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (96, _tiled(32, 64, 512, 1, True, 8, 2, 16, 0)),
        (128, _tiled(64, 64, 512, 1, True, 4, 2, 32, 0)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (320, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (384, _tiled(128, 64, 256, 1, True, 8, 3, 32, 0)),
        (768, _tiled(128, 128, 256, 1, True, 4, 2, 32, 0)),
        (1024, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (1536, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (2048, _tiled(128, 128, 256, 1, True, 8, 2, 32, 1)),
        (4096, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (8192, _tiled(128, 128, 128, 1, True, 2, 2, 32, 1)),
    ],
    # wo_b, TP2
    (5120, 4096): [
        (4, _packed(4, 32, 512, 4, 2, 2, 16, 0)),
        (8, _packed(8, 16, 128, 4, 2, 2, 16, 0, 2)),
        (16, _tiled(16, 32, 256, 1, True, 2, 2, 16, 0, 4)),
        (24, _tiled(32, 32, 256, 1, True, 4, 3, 16, 0, 4)),
        (32, _tiled(32, 64, 512, 1, True, 8, 3, 16, 0, 3)),
        (48, _tiled(64, 64, 512, 1, True, 8, 2, 16, 1, 3)),
        (64, _tiled(64, 64, 512, 1, True, 8, 2, 16, 0, 3)),
        (128, _tiled(64, 64, 256, 1, True, 4, 2, 16, 0, 3)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 32, 0)),
        (256, _tiled(128, 128, 256, 1, True, 4, 2, 16, 1, 2)),
        (320, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (384, _tiled(128, 128, 256, 1, True, 4, 2, 16, 0, 2)),
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
        (16, _packed(16, 32, 1024, 1, 2, 3, 16, 1)),
        (32, _packed(32, 32, 512, 1, 2, 3, 16, 0)),
        (48, _tiled(64, 32, 512, 1, True, 8, 3, 16, 1)),
        (64, _packed(32, 64, 256, 1, 4, 3, 16, 1)),
        (128, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (192, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (256, _tiled(128, 64, 256, 1, True, 8, 3, 32, 1)),
        (384, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (512, _tiled(128, 128, 256, 1, True, 4, 2, 32, 0)),
        (1024, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (1536, _tiled(128, 128, 128, 1, True, 8, 2, 16, 0)),
        (4096, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (8192, _tiled(128, 256, 128, 1, True, 2, 2, 32, 0)),
    ],
    # wq_b, TP2
    (16384, 1280): [
        (2, _packed(4, 32, 128, 4, 2, 3, 16, 1)),
        (4, _packed(4, 32, 128, 4, 4, 2, 16, 0)),
        (8, _packed(8, 32, 256, 2, 4, 2, 16, 0)),
        (32, _packed(32, 64, 256, 1, 4, 3, 16, 1)),
        (64, _tiled(64, 64, 256, 1, False, 4, 3, 32, 1)),
        (128, _tiled(128, 64, 256, 1, True, 8, 3, 32, 0)),
        (256, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (384, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (512, _tiled(128, 128, 128, 1, True, 8, 2, 32, 0)),
        (768, _tiled(128, 128, 128, 1, True, 8, 2, 16, 1)),
        (1024, _tiled(128, 128, 128, 1, True, 8, 2, 32, 1)),
        (2048, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (8192, _tiled(128, 256, 128, 1, True, 2, 2, 32, 0)),
    ],
    # wq_b, TP1
    (32768, 1280): [
        (2, _packed(8, 16, 256, 2, 2, 3, 16, 0)),
        (4, _packed(4, 32, 128, 4, 4, 2, 16, 0)),
        (8, _packed(8, 16, 256, 2, 2, 3, 16, 0)),
        (24, _packed(32, 32, 256, 1, 2, 3, 16, 0)),
        (64, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (128, _tiled(128, 128, 256, 1, True, 8, 2, 32, 1)),
        (192, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (384, _tiled(128, 128, 128, 1, True, 8, 2, 32, 0)),
        (512, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 16, 0)),
    ],
    # wo_b, TP1
    (5120, 8192): [
        (1, _packed(4, 16, 256, 4, 4, 2, 16, 0, 4)),
        (8, _packed(8, 16, 128, 4, 2, 2, 16, 0, 5)),
        (16, _packed(16, 32, 512, 1, 2, 2, 16, 0, 4)),
        (32, _packed(32, 32, 512, 1, 4, 2, 16, 0, 3)),
        (64, _tiled(64, 64, 512, 1, True, 4, 2, 16, 1, 3)),
        (128, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0, 3)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 16, 1)),
        (256, _tiled(128, 128, 256, 1, True, 4, 2, 16, 1, 3)),
        (384, _tiled(128, 128, 256, 2, False, 4, 2, 16, 1, 2)),
        (512, _tiled(128, 128, 256, 1, True, 4, 2, 16, 1)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 16, 0)),
    ],
    # shared expert gate_up, TP1
    (4608, 5120): [
        (1, _packed(8, 32, 256, 2, 4, 3, 16, 0, 5)),
        (2, _packed(8, 32, 256, 2, 4, 3, 16, 0, 8)),
        (8, _packed(8, 32, 256, 2, 4, 3, 16, 0, 6)),
        (16, _packed(16, 32, 512, 1, 2, 2, 16, 0, 6)),
        (32, _tiled(32, 32, 256, 1, True, 4, 3, 16, 0, 5)),
        (48, _tiled(64, 64, 256, 1, False, 4, 3, 32, 1, 3)),
        (64, _tiled(64, 64, 512, 1, True, 8, 2, 16, 0, 3)),
        (96, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (128, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0, 3)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (256, _tiled(128, 128, 256, 1, True, 8, 2, 16, 0, 3)),
        (384, _tiled(128, 128, 256, 2, False, 4, 2, 16, 1, 2)),
        (512, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 16, 0)),
    ],
    # shared expert down, TP1
    (5120, 2304): [
        (1, _packed(16, 32, 1024, 1, 2, 3, 16, 1)),
        (2, _packed(8, 32, 256, 2, 4, 3, 16, 0)),
        (4, _tiled(16, 32, 1024, 1, True, 2, 3, 16, 0)),
        (8, _packed(8, 32, 256, 2, 4, 3, 16, 0)),
        (16, _tiled(16, 32, 1024, 1, True, 2, 3, 16, 0)),
        (32, _tiled(32, 32, 256, 1, True, 4, 3, 16, 0, 3)),
        (64, _tiled(64, 32, 256, 1, True, 8, 3, 16, 0, 4)),
        (96, _tiled(32, 32, 512, 1, True, 4, 2, 16, 0)),
        (128, _tiled(64, 32, 256, 1, True, 8, 3, 16, 0)),
        (192, _tiled(64, 64, 512, 1, True, 4, 2, 32, 1)),
        (256, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (384, _tiled(128, 64, 256, 1, True, 8, 3, 32, 0)),
        (512, _tiled(128, 128, 256, 1, True, 4, 2, 32, 1)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 16, 0)),
    ],
    # engram wkv
    (25600, 6144): [
        (2, _packed(4, 16, 256, 4, 4, 2, 16, 0, 5)),
        (4, _packed(4, 16, 256, 4, 4, 2, 16, 0, 4)),
        (8, _packed(8, 32, 256, 4, 2, 3, 16, 0, 3)),
        (16, _tiled(16, 64, 1024, 1, True, 4, 2, 16, 0, 8)),
        (24, _tiled(64, 64, 256, 1, True, 4, 2, 16, 0)),
        (32, _tiled(64, 64, 256, 1, True, 4, 2, 32, 0)),
        (48, _tiled(64, 64, 256, 1, True, 4, 2, 16, 0)),
        (64, _tiled(128, 128, 256, 1, True, 4, 2, 32, 0)),
        (96, _tiled(128, 128, 256, 1, False, 4, 2, 32, 0)),
        (128, _tiled(128, 128, 256, 1, False, 4, 2, 32, 1)),
        (256, _tiled(128, 128, 128, 1, True, 4, 2, 32, 0)),
        (384, _tiled(128, 128, 256, 1, True, 4, 2, 16, 1)),
        (512, _tiled(128, 128, 128, 1, True, 4, 2, 16, 0)),
        (8192, _tiled(128, 128, 128, 1, True, 4, 2, 16, 0)),
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


# Retuned tiling/scheduling for the fused-quant kernel, whose per-CTA
# quant prologue shifts the best (BLOCK_M, BLOCK_N, num_warps, num_stages,
# waves_per_eu). Entries override ONLY those five axes of the stock tier
# active at M; the K-structure axes (block_k, k_split/k_pack,
# fused_splits, matrix_instr_nonkdim) stay at that tier's values so no
# output element's K-accumulation order changes and the fused output stays
# bit-identical to the two-launch path at every M. Consulted only on the
# quant_in_kernel path, never by the stock two-launch path.
_TUNED_FUSED_QUANT: dict[
    tuple[int, int], list[tuple[int, tuple[int, int, int, int, int]]]
] = {
    # (N, K): [(max_m, (block_m, block_n, warps, stages, waves)), ...]
    # DeepSeek-V4.1-Flash TP4 decode shapes:
    (1152, 5120): [(8, (8, 32, 4, 2, 0))],  # shared_gate_up
    (1792, 5120): [(8, (8, 16, 4, 3, 1))],  # fused_wqa_wkv
    (4096, 1280): [(8, (8, 16, 4, 2, 0))],  # indexer_wq_b
    (5120, 576): [(8, (8, 16, 4, 2, 0))],  # shared_down
    (5120, 2048): [(8, (8, 32, 8, 3, 0))],  # wo_b
    (8192, 1280): [(8, (8, 32, 2, 3, 1))],  # wq_b
    # DSpark draft shapes (stock falls to _default_config):
    (1024, 2048): [(8, (8, 16, 4, 2, 1))],  # draft
    (768, 1536): [(8, (8, 16, 4, 3, 0))],  # draft
}


def _fused_quant_config(M: int, N: int, K: int) -> _Config:
    """The stock tier active at M with its tiling/scheduling retuned for the
    in-kernel quant prologue (see _TUNED_FUSED_QUANT); the stock config
    where the shape has no fused-tier entry. K-structure axes are always
    the stock tier's own, so the output stays bit-exact at every M."""
    base = _config(M, N, K)
    for max_m, (bm, bn, warps, stages, waves) in _TUNED_FUSED_QUANT.get((N, K), ()):
        if max_m >= M:
            return base._replace(
                block_m=bm,
                block_n=bn,
                num_warps=warps,
                num_stages=stages,
                waves_per_eu=waves,
            )
    return base


_split_counters_by_stream: dict[tuple[int, int], torch.Tensor] = {}


def _split_counters(device: torch.device) -> torch.Tensor | None:
    """Zeroed arrival counters, one set per stream so concurrent launches never
    share a tile's counter. Each launch leaves the counters it used at zero.

    None when a stream first needs them inside a CUDA graph capture: that
    allocation would come from the graph's pool, where it can take the address
    of an intermediate that every replay overwrites. vLLM warms up on its
    capture streams first, so this only affects captures without a warmup.
    """
    key = (device.index, torch.accelerator.current_stream(device).stream_id)
    counters = _split_counters_by_stream.get(key)
    if counters is None:
        if torch.cuda.is_current_stream_capturing():
            return None
        counters = torch.zeros(_SPLIT_COUNTERS, dtype=torch.int32, device=device)
        _split_counters_by_stream[key] = counters
    return counters


def _k_partition(K: int, splits: int, step: int) -> tuple[int, int]:
    """K per split, rounded up to whole steps, and the number of non-empty
    splits."""
    size = triton.cdiv(triton.cdiv(K, splits), step) * step
    return size, triton.cdiv(K, size)


def _launch(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out: torch.Tensor,
    cfg: _Config,
    quant_in_kernel: bool = False,
) -> None:
    """Launch cfg's kernel. quant_in_kernel=True (packed cfgs only) makes
    x the raw bf16/fp16 activation and x_scale None: the kernel quantizes
    x per 32-block in registers (see _block32_packed_quant_kernel)."""
    M, K = x.shape
    N = weight.shape[0]
    opts = dict(
        num_warps=cfg.num_warps,
        num_stages=cfg.num_stages,
        matrix_instr_nonkdim=cfg.matrix_instr_nonkdim,
        waves_per_eu=cfg.waves_per_eu,
    )
    grid_m, grid_n = triton.cdiv(M, cfg.block_m), triton.cdiv(N, cfg.block_n)
    tiles = grid_m * grid_n
    step = cfg.block_k * cfg.k_split if cfg.packed else cfg.block_k
    splits = cfg.fused_splits if cfg.fused_splits > 1 or cfg.packed else cfg.k_split
    k_per_split, splits = _k_partition(K, splits, step)
    counters = None
    if cfg.fused_splits > 1 and splits > 1 and tiles <= _SPLIT_COUNTERS:
        counters = _split_counters(out.device)
    if counters is not None:
        fused = splits
        target = out
        slots = torch.empty(
            tiles * fused * cfg.block_m * cfg.block_n,
            dtype=torch.float32,
            device=out.device,
        )
        grid: tuple[int, ...] = (triton.cdiv(tiles, _XCD_STRIDE) * _XCD_STRIDE * fused,)
    else:
        # Without counters, split K through a second launch.
        fused = 1
        target = (
            out
            if splits == 1
            else torch.empty((splits, M, N), dtype=torch.float32, device=x.device)
        )
        slots = counters = target  # Unused by the kernels.
        if cfg.packed or not cfg.n_first:
            grid = (grid_m, grid_n, splits)
        else:
            grid = (grid_n, grid_m, splits)
    args: tuple
    if quant_in_kernel:
        if not cfg.packed:
            raise ValueError("quant_in_kernel needs a packed tile config")
        args = (x, weight, weight_scale, target, slots, counters, M, N, K)
    else:
        args = (x, x_scale, weight, weight_scale, target, slots, counters, M, N, K)
    stride = out.stride(0) if splits == 1 or fused > 1 else N
    if cfg.packed:
        if quant_in_kernel:
            # Same output contract as the packed kernel below (partials at
            # pid_k * M * N when FUSED_SPLITS == 1 with multiple pid_k), so
            # the split-K reduce after this branch applies unchanged.
            _block32_packed_quant_kernel[grid](
                *args,
                stride,
                BLOCK_M=cfg.block_m,
                BLOCK_N=cfg.block_n,
                BLOCK_K=cfg.block_k,
                K_PACK=cfg.k_split,
                K_PER_SPLIT=k_per_split,
                EVEN_K=K % step == 0,
                FUSED_SPLITS=fused,
                FP8_MAX=_FP8_MAX,
                TINY=_FP32_TINY,
                **opts,
            )
        else:
            _block32_packed_kernel[grid](
                *args,
                stride,
                BLOCK_M=cfg.block_m,
                BLOCK_N=cfg.block_n,
                BLOCK_K=cfg.block_k,
                K_PACK=cfg.k_split,
                K_PER_SPLIT=k_per_split,
                EVEN_K=K % step == 0,
                FUSED_SPLITS=fused,
                **opts,
            )
    else:
        _block32_tiled_kernel[grid](
            *args,
            stride,
            BLOCK_M=cfg.block_m,
            BLOCK_N=cfg.block_n,
            BLOCK_K=cfg.block_k,
            K_PER_SPLIT=k_per_split,
            N_FIRST=cfg.n_first,
            EVEN_K=K % k_per_split == 0,
            FUSED_SPLITS=fused,
            **opts,
        )
    if fused == 1 and splits > 1:
        _block32_splitk_reduce_kernel[(triton.cdiv(M, 32), triton.cdiv(N, 64))](
            target,
            out,
            M,
            N,
            out.stride(0),
            SPLIT_K=splits,
            BLOCK_M=32,
            BLOCK_N=64,
        )


def _rocm_mxfp8_block32_gemm_impl(
    x: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    M, K = x.shape
    out = torch.empty((M, weight.shape[0]), dtype=out_dtype, device=x.device)
    if M > 0:
        _launch(x, x_scale, weight, weight_scale, out, _config(M, weight.shape[0], K))
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


# --- tiny-M fused quant+GEMM ------------------------------------------------
# At M <= 8 (decode-sized batches) the fused kernel below computes the
# activation quant inside the GEMM's K loop, saving one launch and its
# global round-trip per call.

_TINY_M_FUSED_QUANT_MAX_M = 8
# Shared with _mxfp8_quant_triton_kernel (the bit-exactness contract).
_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_FP32_TINY = float(torch.finfo(torch.float32).tiny)

_FUSED_QUANT_HIT_SHAPES: set = set()


def _rocm_mxfp8_block32_gemm_quant_impl(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``x @ weight.T`` for raw bf16/fp16 ``x`` and a 32x32 block-scaled
    MXFP8 weight, quantizing ``x`` inside the GEMM at tiny M.

    M <= 8 with a packed tile config (every tuned M <= 8 tier is packed)
    runs one launch whose K loop computes the per-32-block E8M0 scale and
    the e4m3 cast exactly as ``_mxfp8_quant_triton_kernel`` and feeds
    ``tl.dot_scaled`` from registers -- bit-identical output to
    ``mxfp8_e4m3_quantize`` + ``rocm_mxfp8_block32_gemm`` with one launch
    fewer on the decode critical path. Everything else (larger M,
    non-packed tiles, or non-contiguous or non-bf16/fp16 x) runs the stock
    two-launch path with identical results. The branch is inside this
    opaque op so a compiled caller cannot freeze the token-count decision.
    """
    M, K = x.shape
    out = torch.empty((M, weight.shape[0]), dtype=out_dtype, device=x.device)
    if M <= 0:
        return out
    cfg = _config(M, weight.shape[0], K)
    eligible = (
        cfg.packed
        and M <= _TINY_M_FUSED_QUANT_MAX_M
        and x.dtype in (torch.bfloat16, torch.float16)
        and x.is_contiguous()
    )
    if eligible:
        # Retuned tiling for the in-kernel quant prologue (same K-structure
        # as the stock tier at M, so bit-exactness holds); the stock path
        # below keeps the stock config untouched.
        fused_cfg = _fused_quant_config(M, weight.shape[0], K)
        shape = (M, weight.shape[0], K)
        if shape not in _FUSED_QUANT_HIT_SHAPES:
            _FUSED_QUANT_HIT_SHAPES.add(shape)
            # warning_once keeps the engagement log one-shot per shape;
            # WARNING, not INFO, so it reaches the server log.
            logger.warning_once(
                "tiny-M fused MXFP8 quant+GEMM HIT (M=%d, N=%d, K=%d, packed "
                "tile %dx%dx%d k_pack=%d%s)",
                M,
                weight.shape[0],
                K,
                fused_cfg.block_m,
                fused_cfg.block_n,
                fused_cfg.block_k,
                fused_cfg.k_split,
                (
                    f", fused_splits={fused_cfg.fused_splits}"
                    if fused_cfg.fused_splits > 1
                    else ""
                ),
            )
        _launch(x, None, weight, weight_scale, out, fused_cfg, quant_in_kernel=True)
    else:
        x_q, x_scale = mxfp8_e4m3_quantize(x)
        _launch(x_q, x_scale, weight, weight_scale, out, cfg)
    return out


def _rocm_mxfp8_block32_gemm_quant_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=out_dtype)


# An opaque op for the same reason as rocm_mxfp8_block32_gemm: the fused vs
# two-launch choice branches on the token count.
direct_register_custom_op(
    op_name="rocm_mxfp8_block32_gemm_quant",
    op_func=_rocm_mxfp8_block32_gemm_quant_impl,
    fake_impl=_rocm_mxfp8_block32_gemm_quant_fake,
)


def rocm_mxfp8_block32_gemm_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """``x @ weight.T`` for a raw bf16/fp16 ``x`` [M, K] and a 32x32
    block-scaled MXFP8 ``weight`` [N, K] with ``weight_scale``
    [N / 32, K / 32]. Tiny-M batches quantize inside the packed GEMM (see
    ``_rocm_mxfp8_block32_gemm_quant_impl``).
    """
    return torch.ops.vllm.rocm_mxfp8_block32_gemm_quant(
        x, weight, weight_scale, out_dtype
    )
