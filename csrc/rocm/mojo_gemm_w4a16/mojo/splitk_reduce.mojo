from std.gpu import barrier, block_idx, lane_id, thread_idx, warp_id
from std.gpu.memory import AddressSpace
from std.memory import bitcast, stack_allocation
from std.utils import IndexList

from linalg.structuring import SMemArray
from layout import TensorLayout, TileTensor

from mojo.common import (
    ASSUME_EVEN_K,
    ASSUME_EVEN_MN,
    BK,
    BM,
    BN,
    CD,
    COMPUTE_WARPS,
    SPLITK_BLOCK_K,
    SPLITK_ROWS_PER_CTA,
    SPLITK_THREADS,
    GROUP_SIZE,
    LOAD_B_BY_QPACK,
    MMA_K,
    MMA_M,
    MMA_N,
    NUM_STAGES,
    PRODUCTION_TOTAL_THREADS,
    PRODUCTION_TOTAL_WARPS,
    RING_PRODUCER_WARPS,
    RING_STARTUP_ALL_WARPS,
    SCALE_AFTER_GROUP,
    SMEM_PAD,
    USE_FP16,
    USE_QZEROS,
    WARPS_M,
    WARPS_N,
    ZERO_OFFSET,
    ZP_BIAS,
    dtype_acc,
    dtype_in,
    dtype_out,
    dtype_q,
)
from mojo.kernel_common import (
    W4_KPACK_COLS,
    W4_PACK,
    accumulate_scaled_group,
    block_swizzle,
    compute_stage,
    compute_stage_direct_a,
    fdot2_bf16,
    load_a_smem_tile,
    load_b_dequant_smem_tile,
    load_b_dequant_smem_tile_qpack,
    store_accum,
    store_accum_partial,
    wmma,
)
from mojo.ring_buffer import (
    increment_counter_if_first_lane,
    stage_ptr,
    wait_for_counter,
    wait_for_ring_stage_stores,
)


def gemm_w4a16_splitk_reduce_kernel[
    pl: TensorLayout,
    cl: TensorLayout,
](
    c: TileTensor[mut=True, dtype_out, cl, MutAnyOrigin],
    partial: TileTensor[dtype_acc, pl, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert partial.flat_rank == 3
    comptime assert c.flat_rank == 2

    var row_base = Int(block_idx.y) * SPLITK_ROWS_PER_CTA
    var packed_n = Int(block_idx.x) * SPLITK_THREADS + Int(thread_idx.x)
    var col_base = packed_n * W4_PACK
    if row_base >= m or col_base >= n:
        return

    var split_count = (k + SPLITK_BLOCK_K - 1) // SPLITK_BLOCK_K
    comptime for mr in range(SPLITK_ROWS_PER_CTA):
        var row = row_base + mr
        if row < m:
            var acc = SIMD[dtype_acc, W4_PACK](0)
            for split in range(split_count):
                comptime for v in range(W4_PACK):
                    acc[v] += partial[split, row, col_base + v]

            var out_vec = acc.cast[dtype_out]()
            comptime for v in range(W4_PACK):
                c[row, col_base + v] = out_vec[v]

def gemm_w4a16_kpacked_splitk_reduce_kernel[
    pl: TensorLayout,
    cl: TensorLayout,
](
    c: TileTensor[mut=True, dtype_out, cl, MutAnyOrigin],
    partial: TileTensor[dtype_acc, pl, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert partial.flat_rank == 3
    comptime assert c.flat_rank == 2

    var row_base = Int(block_idx.y) * SPLITK_ROWS_PER_CTA
    var col_base = (
        Int(block_idx.x) * SPLITK_THREADS + Int(thread_idx.x)
    ) * W4_KPACK_COLS
    if row_base >= m or col_base >= n:
        return

    var split_count = (k + SPLITK_BLOCK_K - 1) // SPLITK_BLOCK_K
    comptime for mr in range(SPLITK_ROWS_PER_CTA):
        var row = row_base + mr
        if row < m:
            var acc = SIMD[dtype_acc, W4_KPACK_COLS](0)
            for split in range(split_count):
                comptime for v in range(W4_KPACK_COLS):
                    acc[v] += partial[split, row, col_base + v]

            var out_vec = acc.cast[dtype_out]()
            comptime for v in range(W4_KPACK_COLS):
                c[row, col_base + v] = out_vec[v]
