# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The mHC projection of DeepSeek-V4.1 prefill on SM80, in CuTe DSL.

Every mHC boundary (see mhc.py) projects each token's flattened residual
streams, `width = 4 x hidden` BF16 values (`hidden` for the first
layer's broadcast embedding), onto 24 FP32 mixes, and needs the row's sum of
squares for their RMS. The stock epilogue (`MHC_PRE_NORM_KERNEL`) takes
both as `n_splits` partial sums over equal slices of the width and adds
them up.

Above the 256 rows the SM80 boundary kernels take (mhc.py), the stock path
projects with one TileLang kernel on CUDA cores that runs the whole width in
one CTA per two rows, since DeepGEMM's split-k tensor-core projection does
not run on SM80. It reads the 1.5 MB projection once per two rows: about
0.5 ms per boundary at a 4096-token prefill chunk on A100, 45 ms of a 346 ms
chunk.

`MhcProjection` runs it on `mma.sync` BF16 tensor cores. The grid is
`(splits, row tiles)`; a CTA projects 128 rows over one `width / splits`
slice, in steps of 64 columns:

* the rows' 64 columns go to shared memory with `cp.async`, two stages, so
  the next step's load flies during this step's MMAs; rows past the last are
  not loaded, and their outputs are not written;
* the projection's `[24, 64]` FP32 slice is read into registers one step
  ahead and split into two BF16 terms, `hi = bf16(w)` and
  `lo = bf16(w - hi)`, which go to shared memory beside the rows;
* each of the 8 warps multiplies its 16 rows by both terms (`ldmatrix`
  operands, FP32 accumulation) and adds the squares of its row fragments to
  the rows' sums of squares.

A CTA writes its slice's 24 partial mixes and the partial sum of squares of
each row, `mixes[split, row]` and `sqrsum[split, row]`.

Numerics: the rows are exact in BF16, and `hi + lo` keeps about 16 of the
weight's 24 significand bits (TF32, the precision of DeepGEMM's projection
on SM90 and SM100, keeps 11). Products and sums are FP32, so the result
differs from the FP32 CUDA-core projection by about `2^-17` of each
product plus FP32 rounding of a different summation order.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import BFloat16, Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync, warp
from cutlass.cute.runtime import make_fake_stream

from vllm.models.common.ops.moe_tail_fusion.primitives import current_cu_stream

from .mhc import MIXES, load_f32x4, warp_sum
from .mqa_logits import _fake, _pointer

# Rows per CTA (16 per warp), threads, and columns per step.
ROWS = 128
THREADS = 256
WARPS = THREADS // 32
STEP = 64
STAGES = 2
# Width slices: the partial sums the stock epilogue adds, one of the split
# counts it is compiled for. A 4096-row chunk is 512 CTAs.
SPLITS = 16
# Threads that convert the projection's slice: 24 mixes x 8 vectors of 8.
WEIGHT_THREADS = MIXES * 8


class MhcProjection:
    """Split-k mHC projection and sums of squares; see the module docstring."""

    def __init__(self, width: int, splits: int):
        assert width % (splits * STEP) == 0
        self.width = width
        self.splits = splits
        # Steps of 64 columns per CTA.
        self.steps = width // (splits * STEP)

    @cute.jit
    def __call__(
        self,
        x: cute.Pointer,
        fn: Int64,
        mixes: cute.Pointer,
        sqrsum: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        atom = cute.make_composed_layout(
            cute.make_swizzle(3, 3, 3), 0, cute.make_layout((8, STEP), stride=(STEP, 1))
        )
        x_layout = cute.tile_to_shape(atom, (ROWS, STEP, STAGES), (0, 1, 2))
        w_layout = cute.tile_to_shape(atom, (MIXES, STEP, STAGES), (0, 1, 2))

        @cute.struct
        class Storage:
            x: cute.struct.Align[
                cute.struct.MemRange[BFloat16, ROWS * STEP * STAGES], 1024
            ]
            hi: cute.struct.Align[
                cute.struct.MemRange[BFloat16, MIXES * STEP * STAGES], 1024
            ]
            lo: cute.struct.Align[
                cute.struct.MemRange[BFloat16, MIXES * STEP * STAGES], 1024
            ]

        # Rows: 256 threads as 32 rows x 8 vectors of eight BF16, four passes.
        load_x = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                BFloat16,
                num_bits_per_copy=128,
            ),
            cute.make_layout((32, 8), stride=(8, 1)),
            cute.make_layout((1, 8)),
        )
        # Projection terms: the first 192 threads as 24 mixes x 8 vectors.
        store_w = cute.make_tiled_copy_tv(
            cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), BFloat16, num_bits_per_copy=128
            ),
            cute.make_layout((MIXES, 8), stride=(8, 1)),
            cute.make_layout((1, 8)),
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(BFloat16, Float32, (16, 8, 16)), (WARPS, 1, 1)
        )
        self.kernel(
            x,
            fn,
            mixes,
            sqrsum,
            rows,
            x_layout,
            w_layout,
            load_x,
            store_w,
            tiled_mma,
            Storage,
        ).launch(
            grid=(self.splits, (rows + ROWS - 1) // ROWS, 1),
            block=[THREADS, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Pointer,
        fn: Int64,
        mixes: cute.Pointer,
        sqrsum: cute.Pointer,
        rows: Int32,
        x_layout: cute.ComposedLayout,
        w_layout: cute.ComposedLayout,
        load_x: cute.TiledCopy,
        store_w: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
        Storage: cutlass.Constexpr,
    ):
        tid, _, _ = cute.arch.thread_idx()
        split, tile, _ = cute.arch.block_idx()
        width = self.width
        row0 = tile * ROWS
        # Rows of this tile that exist.
        valid = rows - row0
        column0 = split * (self.steps * STEP)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(Storage)
        sX = storage.x.get_tensor(x_layout)
        sHi = storage.hi.get_tensor(w_layout)
        sLo = storage.lo.get_tensor(w_layout)

        thr_load = load_x.get_slice(tid)
        tXsX = thr_load.partition_D(sX)
        tXcX = thr_load.partition_S(cute.make_identity_tensor((ROWS, STEP)))

        thr_store = store_w.get_slice(tid)
        tWsHi = thr_store.partition_D(sHi)
        tWsLo = thr_store.partition_D(sLo)
        rHi = cute.make_rmem_tensor_like(tWsHi[None, None, None, 0], BFloat16)
        rLo = cute.make_rmem_tensor_like(tWsLo[None, None, None, 0], BFloat16)
        # This thread's eight weights of the next step: mix tid // 8,
        # columns 8 (tid % 8) .. + 7 of the step.
        weights = cute.make_rmem_tensor(cute.make_layout(8), Float32)
        weight_base = fn + (Int64(tid // 8) * width + column0 + (tid % 8) * 8) * 4

        thr_mma = tiled_mma.get_slice(tid)
        tCrX = thr_mma.make_fragment_A(thr_mma.partition_A(sX[None, None, 0]))
        tCrHi = thr_mma.make_fragment_B(thr_mma.partition_B(sHi[None, None, 0]))
        tCrLo = thr_mma.make_fragment_B(thr_mma.partition_B(sLo[None, None, 0]))
        copy_a = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4), BFloat16
            ),
            tiled_mma,
        )
        copy_b = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=2), BFloat16
            ),
            tiled_mma,
        )
        thr_copy_a = copy_a.get_slice(tid)
        thr_copy_b = copy_b.get_slice(tid)
        tCsX = thr_copy_a.partition_S(sX)
        tCrX_view = thr_copy_a.retile(tCrX)
        tCsHi = thr_copy_b.partition_S(sHi)
        tCrHi_view = thr_copy_b.retile(tCrHi)
        tCsLo = thr_copy_b.partition_S(sLo)
        tCrLo_view = thr_copy_b.retile(tCrLo)

        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((ROWS, MIXES)), Float32)
        acc.fill(0.0)
        # Sums of squares of the thread's two rows, g and g + 8 of its warp's
        # 16 (g = lane / 4).
        squares = cute.make_rmem_tensor(cute.make_layout(2), Float32)
        squares.fill(0.0)

        # ---- prologue: step 0's rows and weights ----
        gX = cute.make_tensor(
            x + cute.assume(Int64(row0) * width + column0, divby=8),
            cute.make_layout((ROWS, STEP), stride=(width, 1)),
        )
        tXgX = thr_load.partition_S(gX)
        for m in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
            if tXcX[0, m, 0][0] < valid:
                cute.copy(load_x, tXgX[None, m, None], tXsX[None, m, None, 0])
        cute.arch.cp_async_commit_group()
        if tid < WEIGHT_THREADS:
            for half in cutlass.range_constexpr(2):
                values = load_f32x4(weight_base + half * 16)
                for i in cutlass.range_constexpr(4):
                    weights[4 * half + i] = values[i]

        for step in cutlass.range(self.steps, unroll=1):
            stage = step % 2
            # ---- this step's projection terms (stage last read two steps
            # ago, before the previous step's barrier) ----
            if tid < WEIGHT_THREADS:
                for i in cutlass.range_constexpr(8):
                    hi = weights[i].to(BFloat16)
                    rHi[i] = hi
                    rLo[i] = (weights[i] - hi.to(Float32)).to(BFloat16)
                cute.copy(store_w, rHi, tWsHi[None, None, None, stage])
                cute.copy(store_w, rLo, tWsLo[None, None, None, stage])
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()

            # ---- the next step's loads fly during this step's MMAs ----
            if step + 1 < self.steps:
                gNext = cute.make_tensor(
                    x
                    + cute.assume(
                        Int64(row0) * width + column0 + (step + 1) * STEP, divby=8
                    ),
                    cute.make_layout((ROWS, STEP), stride=(width, 1)),
                )
                tXgNext = thr_load.partition_S(gNext)
                for m in cutlass.range_constexpr(cute.size(tXsX.shape[1])):
                    if tXcX[0, m, 0][0] < valid:
                        cute.copy(
                            load_x,
                            tXgNext[None, m, None],
                            tXsX[None, m, None, 1 - stage],
                        )
                cute.arch.cp_async_commit_group()
                if tid < WEIGHT_THREADS:
                    for half in cutlass.range_constexpr(2):
                        next_values = load_f32x4(
                            weight_base + ((step + 1) * STEP + half * 4) * 4
                        )
                        for i in cutlass.range_constexpr(4):
                            weights[4 * half + i] = next_values[i]

            # ---- acc += X hi^T + X lo^T, and the rows' squares ----
            for k in cutlass.range_constexpr(cute.size(tCsX.shape[2])):
                cute.copy(copy_a, tCsX[None, None, k, stage], tCrX_view[None, None, k])
                cute.copy(
                    copy_b, tCsHi[None, None, k, stage], tCrHi_view[None, None, k]
                )
                cute.copy(
                    copy_b, tCsLo[None, None, k, stage], tCrLo_view[None, None, k]
                )
                for terms in (tCrLo, tCrHi):
                    cute.gemm(
                        tiled_mma, acc, tCrX[None, None, k], terms[None, None, k], acc
                    )
                # An m16n8k16 A fragment holds values (v0, v1, v2) at row
                # g + 8 v1 and column 2 t + v0 + 8 v2 (t = lane % 4).
                for i in cutlass.range_constexpr(8):
                    value = tCrX[i, 0, k].to(Float32)
                    squares[(i // 2) % 2] = squares[(i // 2) % 2] + value * value

        # ---- partial mixes and sums of squares of this slice ----
        out_row = Int64(split) * Int64(rows) + Int64(row0)
        gMixes = cute.make_tensor(
            mixes + out_row * MIXES,
            cute.make_layout((ROWS, MIXES), stride=(MIXES, 1)),
        )
        tCcC = thr_mma.partition_C(cute.make_identity_tensor((ROWS, MIXES)))
        for i in cutlass.range_constexpr(cute.size(acc)):
            row, mix = tCcC[i]
            if row < valid:
                gMixes[row, mix] = acc[i]
        lane = tid % 32
        gSqrsum = cute.make_tensor(sqrsum + out_row, cute.make_layout(ROWS))
        for half in cutlass.range_constexpr(2):
            total = warp_sum(squares[half], 4)
            row = (tid // 32) * 16 + lane // 4 + 8 * half
            if lane % 4 == 0 and row < valid:
                gSqrsum[row] = total


def supports(width: int) -> bool:
    """Whether `MhcProjection` takes rows `width` values wide."""
    return width % (SPLITS * STEP) == 0


@cache
def _kernel(device_index: int, width: int):
    """`MhcProjection` compiled for a device and row width."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            MhcProjection(width, SPLITS),
            _fake(BFloat16, 16),
            Int64(0),
            _fake(Float32, 4),
            _fake(Float32, 4),
            Int32(1),
            make_fake_stream(),
        )


def mhc_projection(
    x: torch.Tensor, fn: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """`SPLITS` partial sums of the mHC projection of the BF16 rows `x`
    `[rows, width]` by the FP32 `fn` `[24, width]` (both contiguous and
    16-byte aligned) and of the rows' sums of squares: `mixes [SPLITS,
    rows, 24]` and `sqrsum [SPLITS, rows]`, as `MHC_PRE_NORM_KERNEL`
    takes them."""
    rows, width = x.shape
    assert supports(width)
    assert x.dtype == torch.bfloat16 and x.is_contiguous()
    assert x.data_ptr() % 16 == 0
    assert fn.shape == (MIXES, width) and fn.dtype == torch.float32
    assert fn.is_contiguous() and fn.data_ptr() % 16 == 0
    mixes = torch.empty(SPLITS, rows, MIXES, dtype=torch.float32, device=x.device)
    sqrsum = torch.empty(SPLITS, rows, dtype=torch.float32, device=x.device)
    if rows > 0:
        _kernel(x.device.index, width)(
            _pointer(BFloat16, x, 16),
            Int64(fn.data_ptr()),
            _pointer(Float32, mixes, 4),
            _pointer(Float32, sqrsum, 4),
            Int32(rows),
            current_cu_stream(),
        )
    return mixes, sqrsum
