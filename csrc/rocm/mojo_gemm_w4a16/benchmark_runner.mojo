"""Single-config W4A16 GEMM benchmark runner."""

from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.sys import get_defined_bool
from std.time import perf_counter_ns

from layout import TileTensor

from mojo.common import (
    ALayout,
    BENCH_ITERS,
    BK,
    BM,
    BN,
    CLayout,
    COMPUTE_WARPS,
    GROUP_SIZE,
    MAX_K,
    MAX_M,
    MAX_N,
    PartialLayout,
    PRODUCTION_TOTAL_THREADS,
    Prob,
    QZerosLayout,
    QWeightKPackedLayout,
    QWeightLayout,
    ScalesLayout,
    SPLITK_BLOCK_K,
    SPLITK_ROWS_PER_CTA,
    SPLITK_THREADS,
    WARMUP_ITERS,
    a_layout,
    alloc_prob,
    c_layout,
    dtype_acc,
    dtype_in,
    dtype_out,
    dtype_q,
    partial_layout,
    qzeros_layout,
    qweight_kpacked_layout,
    qweight_layout,
    scales_layout,
    validate_output,
)
from mojo.kernels.b_staged_sync import gemm_w4a16_b_staged_sync_kernel
from mojo.kernels.kpacked_dot2_splitk import gemm_w4a16_kpacked_dot2_splitk_kernel
from mojo.kernels.kpacked_wmma16_splitk import gemm_w4a16_kpacked_wmma16_splitk_kernel
from mojo.kernels.ring_ab_staged import gemm_w4a16_ring_ab_staged_kernel
from mojo.kernels.ring_b_staged import gemm_w4a16_ring_b_staged_kernel
from mojo.splitk_reduce import (
    gemm_w4a16_kpacked_splitk_reduce_kernel,
)


comptime USE_KPACKED_DECODE_KERNEL = get_defined_bool[
    "USE_KPACKED_DECODE_KERNEL", False
]()
comptime USE_RING_BONLY_KERNEL = get_defined_bool["USE_RING_BONLY_KERNEL", False]()
comptime USE_RING_BONLY_SYNC_KERNEL = get_defined_bool[
    "USE_RING_BONLY_SYNC_KERNEL", False
]()
comptime USE_WMMA16_KERNEL = get_defined_bool["USE_WMMA16_KERNEL", False]()
comptime USE_SPLITK_DECODE_KERNEL = False
comptime USE_DECODE_KERNEL = False
comptime DECODE_THREADS = SPLITK_THREADS
comptime DECODE_BLOCK_K = SPLITK_BLOCK_K
comptime DECODE_M_ROWS = SPLITK_ROWS_PER_CTA


def run_case(
    ctx: DeviceContext, mut prob: Prob, m: Int, n: Int, k: Int, verify: Bool
) raises -> Float64:
    comptime assert BM % 16 == 0
    comptime assert BN % 16 == 0
    comptime assert BK % 16 == 0
    comptime assert BK <= GROUP_SIZE
    comptime assert GROUP_SIZE % BK == 0

    var a_tt = TileTensor[dtype_in, ALayout, ImmutAnyOrigin](prob.a, a_layout)
    var qweight_tt = TileTensor[dtype_q, QWeightLayout, ImmutAnyOrigin](
        prob.qweight, qweight_layout
    )
    var qzeros_tt = TileTensor[dtype_q, QZerosLayout, ImmutAnyOrigin](
        prob.qzeros, qzeros_layout
    )
    var qweight_kpacked_tt = TileTensor[
        dtype_q, QWeightKPackedLayout, ImmutAnyOrigin
    ](prob.qweight_kpacked, qweight_kpacked_layout)
    var scales_tt = TileTensor[dtype_in, ScalesLayout, ImmutAnyOrigin](
        prob.scales, scales_layout
    )
    var c_tt = TileTensor[mut=True, dtype_out, CLayout, MutAnyOrigin](
        prob.c, c_layout
    )
    var partial_tt = TileTensor[
        mut=True, dtype_acc, PartialLayout, MutAnyOrigin
    ](prob.partial, partial_layout)
    var partial_read_tt = TileTensor[dtype_acc, PartialLayout, ImmutAnyOrigin](
        prob.partial, partial_layout
    )

    prob.c.enqueue_fill(0)
    ctx.synchronize()
    var mean_ms: Float64
    comptime if USE_RING_BONLY_KERNEL:
        comptime kernel = gemm_w4a16_ring_b_staged_kernel[
            ALayout,
            QWeightLayout,
            QZerosLayout,
            ScalesLayout,
            CLayout,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
        ]
        for _ in range(WARMUP_ITERS):
            ctx.enqueue_function[kernel, kernel](
                c_tt,
                a_tt,
                qweight_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
                block_dim=(PRODUCTION_TOTAL_THREADS, 1),
            )
        ctx.synchronize()
        prob.c.enqueue_fill(0)
        ctx.synchronize()

        var t0 = perf_counter_ns()
        for _ in range(BENCH_ITERS):
            ctx.enqueue_function[kernel, kernel](
                c_tt,
                a_tt,
                qweight_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
                block_dim=(PRODUCTION_TOTAL_THREADS, 1),
            )
        ctx.synchronize()
        mean_ms = Float64(perf_counter_ns() - t0) / Float64(BENCH_ITERS) / 1e6
    elif USE_RING_BONLY_SYNC_KERNEL:
        comptime kernel = gemm_w4a16_b_staged_sync_kernel[
            ALayout,
            QWeightLayout,
            QZerosLayout,
            ScalesLayout,
            CLayout,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
            ImmutAnyOrigin,
        ]
        for _ in range(WARMUP_ITERS):
            ctx.enqueue_function[kernel, kernel](
                c_tt,
                a_tt,
                qweight_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
                block_dim=(COMPUTE_WARPS * 32, 1),
            )
        ctx.synchronize()
        prob.c.enqueue_fill(0)
        ctx.synchronize()

        var t0 = perf_counter_ns()
        for _ in range(BENCH_ITERS):
            ctx.enqueue_function[kernel, kernel](
                c_tt,
                a_tt,
                qweight_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
                block_dim=(COMPUTE_WARPS * 32, 1),
            )
        ctx.synchronize()
        mean_ms = Float64(perf_counter_ns() - t0) / Float64(BENCH_ITERS) / 1e6
    elif USE_WMMA16_KERNEL:
        comptime split_kernel = gemm_w4a16_kpacked_wmma16_splitk_kernel[
            ALayout, QWeightKPackedLayout, QZerosLayout, ScalesLayout, PartialLayout
        ]
        comptime reduce_kernel = gemm_w4a16_kpacked_splitk_reduce_kernel[
            PartialLayout, CLayout
        ]
        for _ in range(WARMUP_ITERS):
            ctx.enqueue_function[split_kernel, split_kernel](
                partial_tt,
                a_tt,
                qweight_kpacked_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(
                    ceildiv(n, 16),
                    ceildiv(m, 16),
                    ceildiv(k, SPLITK_BLOCK_K),
                ),
                block_dim=(32, 1),
            )
            ctx.enqueue_function[reduce_kernel, reduce_kernel](
                c_tt,
                partial_read_tt,
                m,
                n,
                k,
                grid_dim=(
                    ceildiv(ceildiv(n, 4), DECODE_THREADS),
                    ceildiv(m, DECODE_M_ROWS),
                ),
                block_dim=(DECODE_THREADS, 1),
            )
        ctx.synchronize()
        prob.c.enqueue_fill(0)
        ctx.synchronize()

        var t0 = perf_counter_ns()
        for _ in range(BENCH_ITERS):
            ctx.enqueue_function[split_kernel, split_kernel](
                partial_tt,
                a_tt,
                qweight_kpacked_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(
                    ceildiv(n, 16),
                    ceildiv(m, 16),
                    ceildiv(k, SPLITK_BLOCK_K),
                ),
                block_dim=(32, 1),
            )
            ctx.enqueue_function[reduce_kernel, reduce_kernel](
                c_tt,
                partial_read_tt,
                m,
                n,
                k,
                grid_dim=(
                    ceildiv(ceildiv(n, 4), DECODE_THREADS),
                    ceildiv(m, DECODE_M_ROWS),
                ),
                block_dim=(DECODE_THREADS, 1),
            )
        ctx.synchronize()
        mean_ms = Float64(perf_counter_ns() - t0) / Float64(BENCH_ITERS) / 1e6
    elif USE_KPACKED_DECODE_KERNEL:
        comptime split_kernel = gemm_w4a16_kpacked_dot2_splitk_kernel[
            ALayout, QWeightKPackedLayout, QZerosLayout, ScalesLayout, PartialLayout
        ]
        comptime reduce_kernel = gemm_w4a16_kpacked_splitk_reduce_kernel[
            PartialLayout, CLayout
        ]
        for _ in range(WARMUP_ITERS):
            ctx.enqueue_function[split_kernel, split_kernel](
                partial_tt,
                a_tt,
                qweight_kpacked_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(
                    ceildiv(ceildiv(n, 4), SPLITK_THREADS),
                    ceildiv(m, SPLITK_ROWS_PER_CTA),
                    ceildiv(k, SPLITK_BLOCK_K),
                ),
                block_dim=(SPLITK_THREADS, 1),
            )
            ctx.enqueue_function[reduce_kernel, reduce_kernel](
                c_tt,
                partial_read_tt,
                m,
                n,
                k,
                grid_dim=(
                    ceildiv(ceildiv(n, 4), SPLITK_THREADS),
                    ceildiv(m, SPLITK_ROWS_PER_CTA),
                ),
                block_dim=(SPLITK_THREADS, 1),
            )
        ctx.synchronize()
        prob.c.enqueue_fill(0)
        ctx.synchronize()

        var t0 = perf_counter_ns()
        for _ in range(BENCH_ITERS):
            ctx.enqueue_function[split_kernel, split_kernel](
                partial_tt,
                a_tt,
                qweight_kpacked_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(
                    ceildiv(ceildiv(n, 4), SPLITK_THREADS),
                    ceildiv(m, SPLITK_ROWS_PER_CTA),
                    ceildiv(k, SPLITK_BLOCK_K),
                ),
                block_dim=(SPLITK_THREADS, 1),
            )
            ctx.enqueue_function[reduce_kernel, reduce_kernel](
                c_tt,
                partial_read_tt,
                m,
                n,
                k,
                grid_dim=(
                    ceildiv(ceildiv(n, 4), SPLITK_THREADS),
                    ceildiv(m, SPLITK_ROWS_PER_CTA),
                ),
                block_dim=(SPLITK_THREADS, 1),
            )
        ctx.synchronize()
        mean_ms = Float64(perf_counter_ns() - t0) / Float64(BENCH_ITERS) / 1e6
    else:
        comptime ring_kernel = gemm_w4a16_ring_ab_staged_kernel[
            ALayout,
            QWeightLayout,
            QZerosLayout,
            ScalesLayout,
            CLayout,
            BM,
            BN,
            BK,
        ]
        for _ in range(WARMUP_ITERS):
            ctx.enqueue_function[ring_kernel, ring_kernel](
                c_tt,
                a_tt,
                qweight_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
                block_dim=(PRODUCTION_TOTAL_THREADS, 1),
            )
        ctx.synchronize()
        prob.c.enqueue_fill(0)
        ctx.synchronize()

        var t0 = perf_counter_ns()
        for _ in range(BENCH_ITERS):
            ctx.enqueue_function[ring_kernel, ring_kernel](
                c_tt,
                a_tt,
                qweight_tt,
                qzeros_tt,
                scales_tt,
                m,
                n,
                k,
                grid_dim=(ceildiv(n, BN), ceildiv(m, BM)),
                block_dim=(PRODUCTION_TOTAL_THREADS, 1),
            )
        ctx.synchronize()
        mean_ms = Float64(perf_counter_ns() - t0) / Float64(BENCH_ITERS) / 1e6

    if verify:
        with prob.c.map_to_host() as host_c:
            validate_output("w4a16", host_c, m, n, k)

    return mean_ms


def main() raises:
    comptime M = MAX_M
    comptime N = MAX_N
    comptime K = MAX_K
    comptime VERIFY = get_defined_bool["VERIFY", False]()

    with DeviceContext() as ctx:
        var prob = alloc_prob(ctx)
        var mean_ms = run_case(ctx, prob, M, N, K, VERIFY)
        print("TUNE_RESULT=", mean_ms)
