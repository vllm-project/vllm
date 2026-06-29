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
    KERNEL_USES_FDOT2,
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


def gemm_w4a16_kpacked_dot2_splitk_kernel[
    al: TensorLayout,
    qkl: TensorLayout,
    zl: TensorLayout,
    sl: TensorLayout,
    pl: TensorLayout,
](
    partial: TileTensor[mut=True, dtype_acc, pl, MutAnyOrigin],
    a: TileTensor[dtype_in, al, ImmutAnyOrigin],
    qweight_kpacked: TileTensor[dtype_q, qkl, ImmutAnyOrigin],
    qzeros: TileTensor[dtype_q, zl, ImmutAnyOrigin],
    scales: TileTensor[dtype_in, sl, ImmutAnyOrigin],
    m: Int,
    n: Int,
    k: Int,
):
    comptime assert partial.flat_rank == 3
    comptime assert a.flat_rank == 2
    comptime assert qweight_kpacked.flat_rank == 2
    comptime assert qzeros.flat_rank == 2
    comptime assert scales.flat_rank == 2
    comptime assert SPLITK_BLOCK_K % 8 == 0

    var split = Int(block_idx.z)
    var row_base = Int(block_idx.y) * SPLITK_ROWS_PER_CTA
    var col_base = (
        Int(block_idx.x) * SPLITK_THREADS + Int(thread_idx.x)
    ) * W4_KPACK_COLS
    if row_base >= m or col_base >= n:
        return

    var k_begin = split * SPLITK_BLOCK_K
    if k_begin >= k:
        return
    var k_end = k_begin + SPLITK_BLOCK_K
    if k_end > k:
        k_end = k

    var acc_by_m = InlineArray[SIMD[dtype_acc, W4_KPACK_COLS], SPLITK_ROWS_PER_CTA](
        fill=SIMD[dtype_acc, W4_KPACK_COLS](0)
    )
    var first_group = k_begin // GROUP_SIZE
    var last_group = (k_end + GROUP_SIZE - 1) // GROUP_SIZE
    for group in range(first_group, last_group):
        var group_start = group * GROUP_SIZE
        if group_start < k_begin:
            group_start = k_begin
        var group_end = (group + 1) * GROUP_SIZE
        if group_end > k_end:
            group_end = k_end

        var group_acc_by_m = InlineArray[
            SIMD[dtype_acc, W4_KPACK_COLS], SPLITK_ROWS_PER_CTA
        ](fill=SIMD[dtype_acc, W4_KPACK_COLS](0))
        for kk in range(group_start, group_end, W4_PACK):
            var qvec = qweight_kpacked.load_linear[width=W4_KPACK_COLS](
                IndexList[2](elems=(kk // W4_PACK, col_base))
            )
            var zero_vec = SIMD[dtype_acc, W4_KPACK_COLS](Float32(ZP_BIAS))
            comptime if USE_QZEROS:
                var zpacked = rebind[Scalar[dtype_q]](
                    qzeros[group, col_base // W4_PACK]
                )
                comptime for col in range(W4_KPACK_COLS):
                    var global_n = col_base + col
                    var shift = Scalar[dtype_q]((global_n & 7) * 4)
                    var znibble = (zpacked >> shift) & Scalar[dtype_q](0xF)
                    zero_vec[col] = znibble.cast[dtype_acc]() + Scalar[
                        dtype_acc
                    ](Float32(ZERO_OFFSET))
            # The fdot2 nibble encoding below is bf16-specific. With fp16,
            # 0x4300 | nibble represents 128 + nibble / 8, so the subsequent
            # 128-bias correction is wrong. Use direct int4 decode for fp16.
            comptime if KERNEL_USES_FDOT2 and not USE_FP16:
                comptime ONE_BITS = 0x3C00 if USE_FP16 else 0x3F80
                var ones_pair = bitcast[dtype_in](
                    SIMD[DType.int16, 2](ONE_BITS)
                )
                comptime for mr in range(SPLITK_ROWS_PER_CTA):
                    var row = row_base + mr
                    if row < m:
                        var a_pairs = InlineArray[
                            SIMD[dtype_in, 2], W4_PACK // 2
                        ](fill=SIMD[dtype_in, 2](0))
                        var sum_a: Float32 = 0.0
                        comptime for pair_idx in range(W4_PACK // 2):
                            var a_pair = SIMD[dtype_in, 2](0)
                            a_pair[0] = rebind[Scalar[dtype_in]](
                                a[row, kk + pair_idx * 2]
                            )
                            a_pair[1] = rebind[Scalar[dtype_in]](
                                a[row, kk + pair_idx * 2 + 1]
                            )
                            a_pairs[pair_idx] = a_pair
                            sum_a = fdot2_bf16(a_pair, ones_pair, sum_a)

                        comptime for col in range(W4_KPACK_COLS):
                            var qword = qvec[col]
                            var partial_dot: Float32 = 0.0
                            comptime for pair_idx in range(W4_PACK // 2):
                                var qbits = SIMD[DType.int16, 2](0)
                                var q0 = (
                                    qword >> Scalar[dtype_q](pair_idx * 8)
                                ) & Scalar[dtype_q](0xF)
                                var q1 = (
                                    qword >> Scalar[dtype_q](pair_idx * 8 + 4)
                                ) & Scalar[dtype_q](0xF)
                                qbits[0] = q0.cast[DType.int16]() | Scalar[
                                    DType.int16
                                ](0x4300)
                                qbits[1] = q1.cast[DType.int16]() | Scalar[
                                    DType.int16
                                ](0x4300)
                                var q_pair = bitcast[dtype_in](qbits)
                                partial_dot = fdot2_bf16(
                                    a_pairs[pair_idx], q_pair, partial_dot
                                )
                            group_acc_by_m[mr][col] += partial_dot - (
                                (Float32(128) + zero_vec[col]) * sum_a
                            )
            else:
                comptime for ki in range(W4_PACK):
                    var nibbles = (qvec >> Scalar[dtype_q](ki * 4)) & SIMD[
                        dtype_q, W4_KPACK_COLS
                    ](0xF)
                    var w = nibbles.cast[dtype_acc]() - zero_vec
                    comptime for mr in range(SPLITK_ROWS_PER_CTA):
                        var row = row_base + mr
                        if row < m:
                            var av = a[row, kk + ki].cast[dtype_acc]()
                            group_acc_by_m[mr] += w * av

        var scale_vec = scales.load_linear[width=W4_KPACK_COLS](
            IndexList[2](elems=(group, col_base))
        )
        comptime for mr in range(SPLITK_ROWS_PER_CTA):
            acc_by_m[mr] += group_acc_by_m[mr] * scale_vec.cast[dtype_acc]()

    comptime for mr in range(SPLITK_ROWS_PER_CTA):
        var row = row_base + mr
        if row < m:
            comptime for v in range(W4_KPACK_COLS):
                partial[split, row, col_base + v] = acc_by_m[mr][v]
