# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The GDN layer's projections at small row counts in one CuTe DSL kernel.

Why. A GDN layer of a decode step with one request at TP4 has 4 rows and
three dense BF16 projections around its core: `in_proj_qkvz` (20 MiB of
weight a rank), `in_proj_ba` (120 KiB) and `out_proj` (7.5 MiB). At 4
rows these are weight streams: the time is the weight bytes, and the latency
before the stream starts. The skinny GEMM planned for them issues two of its
ten weight tiles before its grid-dependency wait and streams the rest after
it, and it releases its dependents only at its end, so the core cannot start
loading its state while the projection runs. `GdnProjection` loads the
CTA's whole share of the weight into shared memory before the wait, releases
its dependents right after issuing those copies, and after the wait only
multiplies. The executor's GDN layers (`runtime/gdn.py`) run it in
tensor-parallel steps of up to 16 rows.

What it computes. `out_s = x @ w_s.T` for one or two segments `s`: the
same activations `x [rows, k]` (1 to 8 * `groups` rows) against one or two
BF16 weights `w_s [n_s, k]`, each with its own output. The in projection
passes `in_proj_qkvz` and `in_proj_ba` as two segments, so one launch
replaces the two GEMMs (and the side stream `ba` ran on); the output
projection passes one segment.

Work split. A *unit* is 16 consecutive weight rows of one segment (the last
unit of a segment may be partial); the segments' units are numbered
consecutively. CTA `c` owns units `[units * c, units * c + units)`: their
rows over the whole `k` sit in shared memory, one row every `2 k + 64`
bytes (the 64-byte pad puts the eight rows an MMA fragment load touches on
alternating halves of the banks). Warp `w` of the eight takes the 32-wide k
chunks `[w k / 256, (w + 1) k / 256)` of every unit: per chunk it loads the
activation fragment of each eight-row token group once and runs two
`m16n8k16` MMAs per unit and group (`..hc._ptx.mma_chunk`: A and B use the
same permutation of k inside a chunk, so each operand is one 16-byte load).
The warps' partial tiles meet in shared memory and are added in warp order,
then rounded once to BF16.

Sequence of a CTA:

1. before the wait: one bulk copy (`cp.async.bulk`) per weight row into
   shared memory, issued by the lanes of warp 0 (immutable weights; L2
   evict-first), each completing its bytes on one mbarrier, then the
   dependents' trigger;
2. after the wait: the activation fragments (coherent loads: an earlier kernel
   of the step wrote `x`), then every thread waits for the mbarrier's first
   phase, which completes when warp 0's 32 lanes have arrived and every
   expected byte has landed. The activation loads are issued before that wait,
   so they overlap the weight copies still in flight;
3. the MMAs, the partials to shared memory, a barrier, the warp-ordered sums
   and the BF16 stores of the rows below `rows` and the columns below
   `n_s`.

Programmatic dependent launch. The kernel launches with PDL. Before its wait
it reads only weights and writes only its own shared memory. It triggers its
dependents once its copies are issued, before its own wait; a dependent reads
this kernel's outputs only after its own `griddepcontrol.wait`, which returns
when this grid has completed, and this grid completes only after its wait
returned, so completion stays transitive along the stream. Padding token rows
(past `rows` in a group) read a copy of the last real row; their MMA columns
are dropped (an `m16n8k16` output column depends only on its own B column).
Weight rows past `n_s` are not copied; their MMA rows read stale shared
memory, possibly NaN, and are dropped the same way (an output row depends only
on its own A row).

Numerics. FP32 accumulation of exact BF16 products, the k chunks of a warp in
order and the eight warps' partials in warp order, then one rounding to BF16:
deterministic, and within BF16 rounding of any other FP32-accumulated GEMM, but
not bitwise equal to the skinny GEMM or cuBLAS, which sum k in other orders.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream
from cutlass.cutlass_dsl import dsl_user_op

from ..hc._ptx import (
    _asm,
    evict_first_policy,
    load_shared_f32,
    load_shared_x4,
    load_x4,
    mma_chunk,
    store_bf16,
    store_shared_f32x4,
)

WARP = 32
WARPS = 8
THREADS = WARP * WARPS
UNIT_ROWS = 16  # weight rows of a unit (the MMA's M)
GROUP_ROWS = 8  # token rows of a group (the MMA's N)
CHUNK = 32  # k of one chunk: two m16n8k16 MMAs
ROW_PAD = 64  # bytes after each staged weight row
BARRIER_BYTES = 16  # the copies' mbarrier, after the partial tiles
MAX_SHARED_BYTES = 227 * 1024
UNITS = 2  # 16-row weight units of a CTA
MAX_ROWS = 16  # token rows of a launch: two eight-row groups


def shared_bytes(k: int, units: int, groups: int) -> int:
    """Shared memory of a CTA: its staged weight rows, the eight warps'
    partial tiles and the mbarrier."""
    return (
        units * UNIT_ROWS * (2 * k + ROW_PAD)
        + WARPS * units * groups * WARP * 16
        + BARRIER_BYTES
    )


@dsl_user_op
def barrier_init(barrier: Int32, count: Int32, *, loc=None, ip=None) -> None:
    """Initialize the shared-memory mbarrier at `barrier` for `count`
    arrivals and fence the initialization for the asynchronous proxy."""
    _asm(
        None,
        [
            Int32(barrier).ir_value(loc=loc, ip=ip),
            Int32(count).ir_value(loc=loc, ip=ip),
        ],
        "mbarrier.init.shared::cta.b64 [$0], $1;\n\t"
        "fence.mbarrier_init.release.cluster;",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def copy_bulk_hint(
    shared: Int32,
    address: Int64,
    size: Int32,
    barrier: Int32,
    policy: Int64,
    *,
    loc=None,
    ip=None,
) -> None:
    """Expect `size` bytes on `barrier`, then copy `size` bytes of an
    immutable weight to shared memory with one bulk copy that completes them
    on `barrier`, under the L2 cache `policy`."""
    _asm(
        None,
        [
            Int32(shared).ir_value(loc=loc, ip=ip),
            Int64(address).ir_value(loc=loc, ip=ip),
            Int32(size).ir_value(loc=loc, ip=ip),
            Int32(barrier).ir_value(loc=loc, ip=ip),
            Int64(policy).ir_value(loc=loc, ip=ip),
        ],
        "mbarrier.expect_tx.relaxed.cta.shared::cta.b64 [$3], $2;\n\t"
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".L2::cache_hint [$0], [$1], $2, [$3], $4;",
        "r,l,r,r,l",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def barrier_arrive(barrier: Int32, *, loc=None, ip=None) -> None:
    """Arrive once on the shared-memory mbarrier at `barrier`."""
    _asm(
        None,
        [Int32(barrier).ir_value(loc=loc, ip=ip)],
        "mbarrier.arrive.shared::cta.b64 _, [$0];",
        "r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def barrier_wait(barrier: Int32, parity: Int32, *, loc=None, ip=None) -> None:
    """Wait until the phase of parity `parity` of the shared-memory mbarrier
    at `barrier` completes (acquire: the data of its bulk copies is visible
    to the waiting thread)."""
    _asm(
        None,
        [
            Int32(barrier).ir_value(loc=loc, ip=ip),
            Int32(parity).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred done;\n\t"
        "WAIT:\n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 done, [$0], $1;\n\t"
        "@!done bra WAIT;\n\t}",
        "r,r",
        side_effects=True,
        loc=loc,
        ip=ip,
    )


class GdnProjection:
    """`x @ w_s.T` for one or two weight segments, weights staged before the
    grid-dependency wait.

    Args:
        k: The reduction size (a multiple of 256).
        units: 16-row weight units of a CTA.
        groups: Eight-row token groups (1 or 2; up to 8 * groups rows).
        segments: Weight segments (1 or 2).

    """

    def __init__(self, k: int, units: int, groups: int, segments: int) -> None:
        assert k % (CHUNK * WARPS) == 0, k
        assert 1 <= groups <= 2 and segments in (1, 2) and units >= 1
        # A lane of warp 0 copies one weight row; every thread sums the same
        # number of outputs.
        assert units * UNIT_ROWS <= WARP
        assert (units * groups * UNIT_ROWS * GROUP_ROWS) % THREADS == 0
        self.k = k
        self.units = units
        self.groups = groups
        self.segments = segments
        self.warp_chunks = k // (CHUNK * WARPS)
        self.row_bytes = 2 * k + ROW_PAD
        self.weight_bytes = units * UNIT_ROWS * self.row_bytes
        self.partial_stride = units * groups * WARP * 16  # bytes of one warp's tiles
        # The weight rows, the warps' partial tiles and the copies' mbarrier.
        self.shared_bytes = shared_bytes(k, units, groups)
        self.barrier_offset = self.shared_bytes - BARRIER_BYTES
        assert self.barrier_offset == self.weight_bytes + WARPS * self.partial_stride
        assert self.shared_bytes <= MAX_SHARED_BYTES, self.shared_bytes

    @cute.jit
    def __call__(
        self,
        x: Int64,
        x_stride: Int32,
        rows: Int32,
        w0: Int64,
        n0: Int32,
        out0: Int64,
        out0_stride: Int32,
        w1: Int64,
        n1: Int32,
        out1: Int64,
        out1_stride: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """Launch `ctas` CTAs on `stream` (BF16 addresses, element strides)."""
        self.kernel(
            x, x_stride, rows, w0, n0, out0, out0_stride, w1, n1, out1, out1_stride
        ).launch(
            grid=(ctas, 1, 1),
            block=(THREADS, 1, 1),
            stream=stream,
            use_pdl=True,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        x: Int64,
        x_stride: Int32,
        rows: Int32,
        w0: Int64,
        n0: Int32,
        out0: Int64,
        out0_stride: Int32,
        w1: Int64,
        n1: Int32,
        out1: Int64,
        out1_stride: Int32,
    ):
        """One CTA: `units` weight units against all token rows."""
        K = self.k
        U = self.units
        G = self.groups
        RB = self.row_bytes
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(tid // WARP)
        lane = tid % WARP
        g = lane // 4
        t = lane % 4
        units0 = (n0 + UNIT_ROWS - 1) // UNIT_ROWS

        shared = (
            cutlass.memory.SmemAllocator()
            .allocate_array(cutlass.Uint8, self.shared_bytes, byte_alignment=16)
            .toint()
        )
        partial = shared + self.weight_bytes
        barrier = shared + self.barrier_offset
        if tid == 0:
            barrier_init(barrier, WARP)
        cute.arch.sync_threads()

        # 1. The CTA's weight rows, before the wait: lane i of warp 0 copies
        # local row i with one bulk copy (expecting its bytes first), then every
        # lane arrives, so the barrier's phase completes when all 32 lanes have
        # arrived and every expected byte has landed.
        if warp == 0:
            policy = evict_first_policy()
            unit = cta * U + lane // UNIT_ROWS
            r = lane % UNIT_ROWS
            destination = shared + lane * RB
            row0 = unit * UNIT_ROWS + r
            if (lane < U * UNIT_ROWS) & (unit < units0) & (row0 < n0):
                copy_bulk_hint(
                    destination, w0 + Int64(row0 * K) * 2, 2 * K, barrier, policy
                )
            if cutlass.const_expr(self.segments == 2):
                row1 = (unit - units0) * UNIT_ROWS + r
                if (lane < U * UNIT_ROWS) & (unit >= units0) & (row1 < n1):
                    copy_bulk_hint(
                        destination, w1 + Int64(row1 * K) * 2, 2 * K, barrier, policy
                    )
            barrier_arrive(barrier)

        # Dependents may launch now: they read this kernel's outputs only after
        # their own wait. The trigger follows the copies so that they are issued
        # before the wait.
        cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        # 2. The activation fragments of this warp's chunks, one 16-byte load
        # per chunk and group (padding rows repeat the last real row).
        first = warp * self.warp_chunks
        xs = []
        for c in cutlass.range_constexpr(self.warp_chunks):
            per_group = []
            for f in cutlass.range_constexpr(G):
                token = cute.min(f * GROUP_ROWS + g, rows - 1)
                per_group.append(
                    load_x4(
                        x + Int64(token * x_stride + (first + c) * CHUNK + 8 * t) * 2
                    )
                )
            xs.append(per_group)
        barrier_wait(barrier, 0)

        # 3. The MMAs, chunk-outer so that the accumulators interleave.
        accs = []
        for _j in cutlass.range_constexpr(U):
            per_group = []
            for _f in cutlass.range_constexpr(G):
                per_group.append(
                    (Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0))
                )
            accs.append(per_group)
        for c in cutlass.range_constexpr(self.warp_chunks):
            for j in cutlass.range_constexpr(U):
                base = shared + (j * UNIT_ROWS + g) * RB + (first + c) * 64 + t * 16
                lo = load_shared_x4(base)
                hi = load_shared_x4(base + 8 * RB)
                for f in cutlass.range_constexpr(G):
                    accs[j][f] = mma_chunk(accs[j][f], lo, hi, xs[c][f])
        for j in cutlass.range_constexpr(U):
            for f in cutlass.range_constexpr(G):
                store_shared_f32x4(
                    partial
                    + warp * self.partial_stride
                    + ((j * G + f) * WARP + lane) * 16,
                    accs[j][f],
                )
        cute.arch.sync_threads()

        # Lane (g', t') of a tile holds D[g'][2t'], D[g'][2t'+1], D[g'+8][2t'],
        # D[g'+8][2t'+1] (weight row by token). Output (unit j, row r, token)
        # sums the eight warps' values in warp order.
        outputs = U * UNIT_ROWS * G * GROUP_ROWS
        for step in cutlass.range_constexpr((outputs + THREADS - 1) // THREADS):
            o = step * THREADS + tid
            token = o % (G * GROUP_ROWS)
            local = o // (G * GROUP_ROWS)
            j = local // UNIT_ROWS
            r = local % UNIT_ROWS
            f = token // GROUP_ROWS
            column = token % GROUP_ROWS
            source = (r % 8) * 4 + column // 2
            index = (r // 8) * 2 + column % 2
            element = partial + (((j * G + f) * WARP + source) * 4 + index) * 4
            sum_ = load_shared_f32(element)
            for w in cutlass.range_constexpr(1, WARPS):
                sum_ = sum_ + load_shared_f32(element + w * self.partial_stride)
            unit = cta * U + j
            row0 = unit * UNIT_ROWS + r
            if (o < outputs) & (token < rows) & (unit < units0) & (row0 < n0):
                store_bf16(out0 + Int64(token * out0_stride + row0) * 2, sum_)
            if cutlass.const_expr(self.segments == 2):
                row1 = (unit - units0) * UNIT_ROWS + r
                if (o < outputs) & (token < rows) & (unit >= units0) & (row1 < n1):
                    store_bf16(out1 + Int64(token * out1_stride + row1) * 2, sum_)


@cache
def _compiled(device_index: int, k: int, units: int, groups: int, segments: int):
    """`GdnProjection` compiled for a device and shape."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            GdnProjection(k, units, groups, segments),
            Int64(0),
            Int32(1),
            Int32(1),
            Int64(0),
            Int32(1),
            Int64(0),
            Int32(1),
            Int64(0),
            Int32(1),
            Int64(0),
            Int32(1),
            Int32(1),
            make_fake_stream(),
        )


def projection_ctas(ns: list[int], units: int) -> int:
    """CTAs of a launch over weights of `ns` rows with `units` units a CTA."""
    total = sum(-(-n // UNIT_ROWS) for n in ns)
    return -(-total // units)


def projection_unsupported(ns: list[int], k: int, sms: int) -> str:
    """What a launch over BF16 weights of `ns` rows and reduction size `k`,
    for up to `MAX_ROWS` token rows, needs and misses, or `""`.

    Args:
        ns: The rows of each weight segment (one or two).
        k: The reduction size.
        sms: The device's streaming multiprocessors.

    """
    missing = []
    if not 1 <= len(ns) <= 2:
        missing.append(f"one or two weight segments, not {len(ns)}")
    if k % (CHUNK * WARPS):
        missing.append(f"a reduction size that is a multiple of {CHUNK * WARPS}")
    elif shared_bytes(k, UNITS, 2) > MAX_SHARED_BYTES:
        missing.append(
            f"{shared_bytes(k, UNITS, 2)} bytes of shared memory a CTA, more "
            f"than {MAX_SHARED_BYTES}"
        )
    # One CTA per SM (its shared memory allows no second): more CTAs than SMs
    # would stage the second wave's weights only after the first wave ends.
    if projection_ctas(ns, UNITS) > sms:
        missing.append(
            f"at most one CTA per SM ({projection_ctas(ns, UNITS)} CTAs, {sms} SMs)"
        )
    return "; ".join(missing)


def warm_up_gdn_projection(weights: list[list[torch.Tensor]]) -> None:
    """Compile and run `gdn_projection` once for each weight list of
    `weights` at one and at two eight-row groups (zero activations), so that
    no CUDA graph capture compiles or loads it.

    Args:
        weights: The weight lists the steps pass (one or two weights each).

    """
    for segment_weights in weights:
        for rows in (GROUP_ROWS, MAX_ROWS):
            x = segment_weights[0].new_zeros(rows, segment_weights[0].shape[1])
            gdn_projection(x, segment_weights)


def gdn_projection(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    outs: list[torch.Tensor] | None = None,
    units: int = UNITS,
) -> list[torch.Tensor]:
    """`x @ w.T` for each of one or two weights, in one launch.

    Args:
        x: BF16 `[rows, k]` (1 to 16 rows), unit column stride, row stride a
            multiple of 8.
        weights: One or two contiguous BF16 `[n_s, k]` weights.
        outs: Optional BF16 `[rows, n_s]` outputs with unit column stride.
        units: 16-row weight units of a CTA.

    Returns:
        The outputs.

    """
    rows, k = x.shape
    assert 1 <= rows <= MAX_ROWS and x.dtype == torch.bfloat16
    assert x.stride(1) == 1 and x.stride(0) % 8 == 0 and x.data_ptr() % 16 == 0
    assert 1 <= len(weights) <= 2
    for w in weights:
        assert w.dtype == torch.bfloat16 and w.is_contiguous() and w.shape[1] == k
    if outs is None:
        outs = [x.new_empty(rows, w.shape[0]) for w in weights]
    for w, o in zip(weights, outs):
        assert o.shape == (rows, w.shape[0]) and o.stride(1) == 1
    groups = -(-rows // GROUP_ROWS)
    segments = len(weights)
    w1, o1 = (weights[1], outs[1]) if segments == 2 else (weights[0], outs[0])
    ctas = projection_ctas([w.shape[0] for w in weights], units)
    _compiled(x.device.index, k, units, groups, segments)(
        Int64(x.data_ptr()),
        Int32(x.stride(0)),
        Int32(rows),
        Int64(weights[0].data_ptr()),
        Int32(weights[0].shape[0]),
        Int64(outs[0].data_ptr()),
        Int32(outs[0].stride(0)),
        Int64(w1.data_ptr()),
        Int32(w1.shape[0] if segments == 2 else 0),
        Int64(o1.data_ptr()),
        Int32(o1.stride(0)),
        Int32(ctas),
        cuda.CUstream(torch.cuda.current_stream().cuda_stream),
    )
    return outs
