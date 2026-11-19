# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""An L2 warm of a QKV front's ring weights, for the next layer's front.

The QKV front (`qkv_front.py`) TMA-loads the first `ring_stages` k-tiles of
every CTA into its ring before `griddepcontrol.wait`, and prefetches the rest
into L2 itself. With a 10-deep ring at K split 4 those k-tiles are the first
5120 columns of every weight row, the same for every cluster rank
(`ring_columns`). Read from HBM after the boundary before the front starts,
they are the front's cold-start cost.

`QkvWarm.run(m)` prefetches exactly that prefix of the weight into L2 with
`cp.async.bulk.prefetch.L2`, `PIECES` pieces per row, from `CTAS` CTAs of
`THREADS` threads, one piece per thread per round, unpaced: issued with the
MoE tail, a pace gains nothing. The kernel reads and writes no data and takes
no PDL edge; launched on a side stream beside the MoE tail, it changes no
result.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from vllm.utils.math_utils import cdiv

from ..common.host import assert_not_capturing, on_device
from ..common.primitives import bulk_prefetch_l2
from . import qkv_front as front

CTAS = 16
THREADS = 128
PIECES = 4  # per weight row
ROW_BYTES = front.HIDDEN * 2


def ring_columns(n: int, m: int) -> int:
    """The weight columns (a prefix of every row) the QKV front for N columns
    and M rows loads into its ring before its PDL wait; 0 when they are not a
    prefix (an odd ring depth)."""
    _, stages, splits, _ = front.variant(n, m)
    if stages % front.KSTRIPE:
        return 0
    return stages // front.KSTRIPE * splits * front.KSTRIPE * front.KT


class QkvWarmKernel:
    """`rows` weight rows, the first `ring_bytes` of each."""

    def __init__(self, rows: int, ring_bytes: int) -> None:
        piece = ring_bytes // PIECES
        if piece * PIECES != ring_bytes or piece % 16 or rows <= 0:
            raise ValueError(f"no warm of {rows} rows x {ring_bytes} B")
        self.piece = piece
        self.pieces = rows * PIECES
        self.rounds = cdiv(self.pieces, CTAS * THREADS)

    @cute.jit
    def __call__(self, w: Int64, stream: cuda.CUstream) -> None:
        self.kernel(w).launch(grid=(CTAS, 1, 1), block=(THREADS, 1, 1), stream=stream)

    @cute.kernel
    def kernel(self, w: Int64) -> None:
        cta, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        for r in cutlass.range_constexpr(self.rounds):
            piece = (Int32(r * CTAS) + cta) * THREADS + tidx
            if piece < Int32(self.pieces):
                row = piece // PIECES
                address = (
                    w
                    + Int64(row) * ROW_BYTES
                    + Int64((piece - row * PIECES) * self.piece)
                )
                bulk_prefetch_l2(address, Int32(self.piece))


@cache
def compile_warm(rows: int, ring_bytes: int):
    """The compiled warm; called (TVM FFI) as fn(w, stream) with the weight's
    byte address and the stream handle as ints."""
    assert_not_capturing("the QKV warm")
    return cute.compile(
        QkvWarmKernel(rows, ring_bytes),
        Int64(0),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )


def unsupported(weight: torch.Tensor, rows: tuple[int, ...]) -> str | None:
    """Why `weight` (a QKV front weight) cannot be warmed for these row
    counts, or None."""
    if not front.hidden_rows_ok(weight):
        return "the weight is not a contiguous, aligned bf16 [n, 8192] matrix"
    n = weight.shape[0]
    if not front.supports_shape(n, 1):
        return f"no QKV front for {n} columns"
    if any(ring_columns(n, m) == 0 for m in rows):
        return "the front's ring is not a prefix of the weight rows"
    return None


class QkvWarm:
    """The warm of one QKV front weight, compiled at build (before CUDA graph
    capture) for the decode row counts `rows`; `run` is graph-capturable."""

    def __init__(self, weight: torch.Tensor, rows: tuple[int, ...]) -> None:
        reason = unsupported(weight, rows)
        if reason is not None:
            raise ValueError(reason)
        self.weight = weight
        self.device = weight.get_device()
        n = weight.shape[0]
        self.ring_bytes = {m: ring_columns(n, m) * 2 for m in rows}
        for m in rows:
            on_device(self.device, compile_warm, n, self.ring_bytes[m])

    def run(self, m: int) -> None:
        """Prefetch the ring prefix the front at M rows reads, on the current
        stream."""
        fn = compile_warm(self.weight.shape[0], self.ring_bytes[m])
        on_device(self.device, _call, fn, self.device, self.weight.data_ptr())


def _call(fn, device: int, address: int) -> None:
    fn(address, torch.cuda.current_stream(device).cuda_stream)
