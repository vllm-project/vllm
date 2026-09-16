# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: B008 -- FlyDSL launch signatures require typed stream defaults
# ruff: noqa: SIM114 -- Keep compile-time and device predicates separate
"""Single-launch wave-split K reduction for underfilled decode grids."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import Vector as Vec

from .gfx12_sync import (
    lds_fence_signal,
    lds_fence_wait,
)
from .rdna4_fp8_blockscale_common import (
    _f32_to_bf16_rne,
    _load_f32,
    _load_fp8_fragment_buffer,
    _make_buffer,
    _store_bf16,
)


def _cap_split_occupancy():
    """Avoid the full-occupancy scheduling cliff without adding empty CTAs."""
    from flydsl._mlir import ir

    owner = ir.InsertionPoint.current.block.owner
    owner.attributes["llvm.passthrough"] = ir.ArrayAttr.get(
        [
            ir.ArrayAttr.get(
                [ir.StringAttr.get("amdgpu-waves-per-eu"), ir.StringAttr.get("1,12")]
            )
        ]
    )


@lru_cache(maxsize=256)
def create_decode_split(
    m,
    n,
    k,
    stride_b,
    block_n=16,
    splits=4,
    batch=8,
    rotate=0,
    balanced=False,
    lean=False,
    capped=False,
):
    assert 1 <= m <= 256 and block_n in (4, 8, 16, 32, 64)
    assert splits in (2, 4, 8) and k // 128 >= splits
    assert not capped or (m > 64 and splits == 4 and block_n == 16)
    valid_rows = min(m, 16)
    grid_m = (m + 15) // 16
    nk = k // 128
    chunk = (nk + splits - 1) // splits
    threads = splits * 32
    nfrags = (block_n + 15) // 16
    fp8, f32 = fx.Float8E4M3FN, fx.Float32
    signature = (
        m,
        n,
        k,
        stride_b,
        block_n,
        splits,
        batch,
        rotate,
        balanced,
        lean,
        capped,
    )

    @fx.struct
    class SharedStorage:
        data: fx.Array[f32, splits * valid_rows * block_n, 16]

    @flyc.kernel
    def kernel(
        a: fx.Tensor, b: fx.Tensor, asc: fx.Tensor, bsc: fx.Tensor, out: fx.Tensor
    ):
        if const_expr(capped):
            _cap_split_occupancy()
        tid = fx.thread_idx.x
        lane = tid % fx.Int32(32)
        wave = tid // fx.Int32(32)
        lane16 = lane % fx.Int32(16)
        half = lane // fx.Int32(16)
        pid_n = fx.block_idx.x
        m0 = fx.Int32(0)
        if const_expr(m > 16):
            pid_n = fx.block_idx.x // fx.Int32(grid_m)
            m0 = (fx.block_idx.x % fx.Int32(grid_m)) * fx.Int32(16)
        n0 = pid_n * fx.Int32(block_n)
        begin = wave * fx.Int32(chunk)
        end = fx.min(begin + fx.Int32(chunk), fx.Int32(nk))
        if const_expr(balanced):
            begin = wave * fx.Int32(nk // splits) + fx.min(wave, fx.Int32(nk % splits))
            end = (
                begin
                + fx.Int32(nk // splits)
                + (wave < fx.Int32(nk % splits)).select(fx.Int32(1), fx.Int32(0))
            )
        ab = _make_buffer(a, fp8, 8, m * k)
        bb = _make_buffer(b, fp8, 8, n * stride_b)
        asb = _make_buffer(asc, f32, 1, m * nk * 4)
        bsp = fx.recast_iter(f32, fx.get_iter(bsc))
        ob = _make_buffer(out, fx.BFloat16, 1, m * n * 2)
        shared = (
            fx.SharedAllocator()
            .allocate(SharedStorage)
            .peek()
            .data.view(fx.make_layout(splits * valid_rows * block_n, 1))
        )
        mma = fx.make_mma_atom(fx.rocdl.WMMA(16, 16, 16, fp8, f32))
        totals = [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(nfrags)]
        for j in range_constexpr(nfrags):
            totals[j].fill(0)
        for k_iter in range(begin, end, 1):
            kb = k_iter
            if const_expr(rotate):
                kb = (k_iter + pid_n * fx.Int32(rotate)) % fx.Int32(nk)
            partials = [fx.make_rmem_tensor(8, f32) for _ in range_constexpr(nfrags)]
            for j in range_constexpr(nfrags):
                partials[j].fill(0)
            for group in range_constexpr(8 // batch):
                af = [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(batch)]
                bf = [
                    [fx.make_rmem_tensor(8, fp8) for _ in range_constexpr(nfrags)]
                    for _ in range_constexpr(batch)
                ]
                for q in range_constexpr(batch):
                    ko = (
                        kb * fx.Int32(128)
                        + fx.Int32(0 if capped else (group * batch + q) * 16)
                        + half * fx.Int32(8)
                    )
                    af[q].fill(0)
                    if const_expr(lean or m > 16):
                        if const_expr(capped):
                            _load_fp8_fragment_buffer(
                                ab,
                                (m0 + lane16) * fx.Int32(k)
                                + ko
                                + fx.Int32((group * batch + q) * 16),
                                af[q],
                            )
                        else:
                            _load_fp8_fragment_buffer(
                                ab, (m0 + lane16) * fx.Int32(k) + ko, af[q]
                            )
                    elif lane16 < fx.Int32(m):
                        _load_fp8_fragment_buffer(
                            ab, (m0 + lane16) * fx.Int32(k) + ko, af[q]
                        )
                    for j in range_constexpr(nfrags):
                        col = n0 + lane16 + fx.Int32(j * 16)
                        if const_expr(block_n < 16):
                            bf[q][j].fill(0)
                            if lane16 < fx.Int32(block_n):
                                _load_fp8_fragment_buffer(
                                    bb, col * fx.Int32(stride_b) + ko, bf[q][j]
                                )
                        elif const_expr(capped):
                            _load_fp8_fragment_buffer(
                                bb,
                                col * fx.Int32(stride_b)
                                + ko
                                + fx.Int32((group * batch + q) * 16),
                                bf[q][j],
                            )
                        else:
                            _load_fp8_fragment_buffer(
                                bb, col * fx.Int32(stride_b) + ko, bf[q][j]
                            )
                for q in range_constexpr(batch):
                    for j in range_constexpr(nfrags):
                        fx.gemm(mma, partials[j], af[q], bf[q][j], partials[j])
            bs = f32(fx.ptr_load(bsp + n0 // fx.Int32(128) * fx.Int32(nk) + kb))
            for j in range_constexpr(nfrags):
                tv, pv = Vec(totals[j].load()), Vec(partials[j].load())
                vals = []
                for r in range_constexpr(8):
                    if const_expr(lean and r >= m):
                        vals.append(f32(0.0))
                    else:
                        row = m0 + half * fx.Int32(8) + fx.Int32(r)
                        scale = _load_f32(asb, row * fx.Int32(nk) + kb) * bs
                        vals.append(tv[r] + pv[r] * scale)
                totals[j].store(Vec.from_elements(vals, f32))
        for r in range_constexpr(8):
            row = half * fx.Int32(8) + fx.Int32(r)
            valid = row < fx.Int32(valid_rows)
            if const_expr(block_n < 16):
                valid = valid & (lane16 < fx.Int32(block_n))
            if valid:
                for j in range_constexpr(nfrags):
                    idx = (
                        wave * fx.Int32(valid_rows * block_n)
                        + row * fx.Int32(block_n)
                        + lane16
                        + fx.Int32(j * 16)
                    )
                    fx.memref_store(Vec(totals[j].load())[r], shared, idx)
        lds_fence_signal()
        lds_fence_wait()
        for q in range_constexpr((valid_rows * block_n + threads - 1) // threads):
            idx = tid + fx.Int32(q * threads)
            if idx < fx.Int32(valid_rows * block_n):
                val = f32(0.0)
                for s in range_constexpr(splits):
                    val = val + f32(
                        fx.memref_load(shared, fx.Int32(s * valid_rows * block_n) + idx)
                    )
                row = m0 + idx // fx.Int32(block_n)
                col = n0 + idx % fx.Int32(block_n)
                _store_bf16(ob, row * fx.Int32(n) + col, _f32_to_bf16_rne(val))

    @flyc.jit
    def launch(
        a: fx.Tensor,
        b: fx.Tensor,
        asc: fx.Tensor,
        bsc: fx.Tensor,
        out: fx.Tensor,
        stream: fx.Stream = fx.Stream(None),
    ):
        _ = signature
        kernel(a, b, asc, bsc, out).launch(
            grid=((n // block_n) * grid_m, 1, 1), block=(threads, 1, 1), stream=stream
        )

    return launch
