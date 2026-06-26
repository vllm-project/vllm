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


def gemm_w4a16_kpacked_wmma16_splitk_kernel[
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
    comptime assert SPLITK_BLOCK_K % MMA_K == 0
    comptime assert GROUP_SIZE % MMA_K == 0

    var lane = Int(thread_idx.x)
    var lane_lo = lane & 15
    var lane_hi = lane >> 4
    var block_m_offset = Int(block_idx.y) * MMA_M
    var block_n_offset = Int(block_idx.x) * MMA_N
    var split = Int(block_idx.z)
    if block_m_offset >= m or block_n_offset >= n:
        return

    var k_begin = split * SPLITK_BLOCK_K
    if k_begin >= k:
        return
    var k_end = k_begin + SPLITK_BLOCK_K
    if k_end > k:
        k_end = k

    var b_lds = stack_allocation[
        MMA_K * MMA_N,
        dtype_in,
        address_space=AddressSpace.SHARED,
    ]()
    var c_acc = SIMD[dtype_acc, CD](0)

    for k_tile in range(k_begin, k_end, MMA_K):
        var actual_n = block_n_offset + lane_lo
        if actual_n < n:
            var qk_row = k_tile // W4_PACK + lane_hi
            var qa = rebind[Scalar[dtype_q]](qweight_kpacked[qk_row, actual_n])
            var group = k_tile // GROUP_SIZE
            var scale = scales[group, actual_n].cast[dtype_acc]()
            var zero = Scalar[dtype_acc](Float32(ZP_BIAS))
            comptime if USE_QZEROS:
                var zpacked = rebind[Scalar[dtype_q]](
                    qzeros[group, actual_n // W4_PACK]
                )
                var zshift = Scalar[dtype_q]((actual_n & 7) * 4)
                var znibble = (zpacked >> zshift) & Scalar[dtype_q](0xF)
                zero = znibble.cast[dtype_acc]() + Scalar[dtype_acc](
                    Float32(ZERO_OFFSET)
                )
            var k_base = lane_hi * W4_PACK
            comptime for ki in range(W4_PACK):
                var nibble = (qa >> Scalar[dtype_q](ki * 4)) & Scalar[dtype_q](
                    0xF
                )
                var b_val = (
                    nibble.cast[dtype_acc]()
                    - zero
                ) * scale
                b_lds.store(
                    (k_base + ki) * MMA_N + lane_lo, b_val.cast[dtype_in]()
                )

        wait_for_ring_stage_stores()

        var a_frag = SIMD[dtype_in, MMA_K](0)
        var global_m = block_m_offset + lane_lo
        if global_m < m:
            var a_vec = a.load_linear[width=MMA_K](
                IndexList[2](elems=(global_m, k_tile))
            )
            a_frag = bitcast[dtype_in](a_vec)

        var b_frag = SIMD[dtype_in, MMA_K](0)
        comptime for i in range(MMA_K):
            var b_scalar = b_lds.load[width=1](i * MMA_N + lane_lo)
            b_frag[i] = bitcast[dtype_in](b_scalar)[0]

        c_acc = wmma(a_frag, b_frag, c_acc)
        wait_for_ring_stage_stores()

    var global_n = block_n_offset + lane_lo
    if global_n < n:
        comptime for i in range(CD):
            var global_m = block_m_offset + i * 2 + lane_hi
            if global_m < m:
                partial[split, global_m, global_n] = c_acc[i]
