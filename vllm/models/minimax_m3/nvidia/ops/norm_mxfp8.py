# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma RMSNorm (optionally fused with the residual add) emitting MXFP8.

One pass replaces FlashInfer's CuTe-DSL ``FusedAddRMSNormKernel`` (or
``RMSNormKernel``) followed by ``mxfp8_quantize(..., is_sf_swizzled_layout=True,
backend="cute-dsl")``, with bitwise-identical results:

- the norm is FlashInfer 0.7.0's kernel body (same tiling, reduction order and
  rounding), so the residual and the BF16-rounded normed activation are the
  ones today's kernel writes;
- the BF16 activation is quantized in registers with FlashInfer's MXFP8 PTX
  helpers (amax of 32 BF16 values, ``amax * RN(1/448)``, round-up UE8M0,
  ``x * 2^(127 - e)`` saturated to E4M3), and the UE8M0 scales are written in
  the 128x4 swizzled layout, zero in the padding rows.

The normed BF16 activation itself is not written.
"""

import functools

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from flashinfer.norm.kernels.fused_add_rmsnorm import FusedAddRMSNormKernel
from flashinfer.norm.kernels.rmsnorm import RMSNormKernel
from flashinfer.norm.utils import (
    get_cutlass_dtype,
    get_ptr_as_int64,
    get_sm_version,
    predicate_k,
    row_reduce_sum_multirow,
)
from flashinfer.quantization.quantization_cute_dsl_utils import (
    INV_FLOAT8_E4M3_MAX,
    bfloat2_hmax_reduce_to_f32,
    bfloat2_max_abs_4,
    compute_sf_index_swizzled_128x4_gpu,
    float_to_ue8m0_fast,
    reduce_max_4threads,
)

MXFP8_BLOCK = 32
SF_ROW_TILE = 128


def _asm(ret, args, body, constraints, side_effects=False):
    return llvm.inline_asm(
        ret,
        args,
        body,
        constraints,
        has_side_effects=side_effects,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _ue8m0_to_inv_scale_bf16x2(e: Uint32, *, loc=None, ip=None) -> Uint32:
    """FlashInfer's ``ue8m0_to_inv_scale_fast(e)`` (2^(127 - e), or 0 for e == 0 and
    below the normal range: always exactly a BF16) in both BF16 halves."""
    return Uint32(
        _asm(
            T.i32(),
            [Uint32(e).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .s32 new_exp;
                .reg .b32 hi;
                .reg .pred p_zero;
                setp.eq.u32 p_zero, $1, 0;
                sub.s32 new_exp, 254, $1;
                max.s32 new_exp, new_exp, 0;
                shl.b32 hi, new_exp, 7;
                @p_zero mov.b32 hi, 0;
                prmt.b32 $0, hi, hi, 0x1010;
            }
            """,
            "=r,r",
        )
    )


@dsl_user_op
def _bf16x4_to_e4m3x4_scaled(
    w0: Uint32, w1: Uint32, inv2: Uint32, *, loc=None, ip=None
) -> Uint32:
    """Four BF16 (w0 = values 0-1, w1 = 2-3) -> four E4M3 bytes, value 0 lowest.

    Equals FlashInfer's ``bfloat2_to_fp8x2_scaled`` (fp32 ``x * inv``, clamp +-448,
    ``cvt.rn.satfinite``) for every BF16 x and UE8M0 scale (checked exhaustively): the
    power-of-two product is exact in BF16 wherever E4M3 can represent it, ``min`` with
    448 maps NaN and overflow to 448 like the fp32 clamp, and ``satfinite`` saturates
    the negative side.
    """
    return Uint32(
        _asm(
            T.i32(),
            [
                Uint32(w0).ir_value(loc=loc, ip=ip),
                Uint32(w1).ir_value(loc=loc, ip=ip),
                Uint32(inv2).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 p0, p1, c;
                .reg .b16 q0, q1;
                mov.b32 c, 0x43E043E0;
                mul.rn.bf16x2 p0, $1, $3;
                mul.rn.bf16x2 p1, $2, $3;
                min.bf16x2 p0, p0, c;
                min.bf16x2 p1, p1, c;
                cvt.rn.satfinite.e4m3x2.bf16x2 q0, p0;
                cvt.rn.satfinite.e4m3x2.bf16x2 q1, p1;
                mov.b32 $0, {q0, q1};
            }
            """,
            "=r,r,r,r",
        )
    )


@dsl_user_op
def _st_global_v2_u32(addr: Int64, a: Uint32, b: Uint32, *, loc=None, ip=None):
    _asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(a).ir_value(loc=loc, ip=ip),
            Uint32(b).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v2.b32 [$0], {$1, $2};",
        "l,r,r",
        side_effects=True,
    )


def _check_tiling(kernel, H: int) -> None:
    # The epilogue quantizes each thread's 8-element vectors; a 32-element
    # block must be the vectors of 4 adjacent lanes (shuffle xor 1, 2).
    assert kernel.cluster_n == 1, "cluster norms are not supported"
    assert kernel.vec_size == 8, kernel.vec_size
    assert kernel.threads_per_row % 4 == 0, kernel.threads_per_row
    assert H % MXFP8_BLOCK == 0, H
    assert kernel.cols_per_tile == H, (H, kernel.cols_per_tile)


@cute.jit
def _quantize_store_mxfp8(
    tXrO: cute.Tensor,
    tXcX: cute.Tensor,
    mQ: cute.Tensor,
    mS: cute.Tensor,
    M: Int64,
    H: cutlass.Constexpr[int],
    padded_sf_cols: cutlass.Constexpr[int],
):
    """Quantize this thread's BF16 vectors (8 per vec block) to MXFP8."""
    # Per thread: ((8, vec blocks), 1, 1) values; vec block k = 8 contiguous columns.
    assert cute.size(tXcX, mode=[1]) == 1 and cute.size(tXcX, mode=[2]) == 1
    lane = cute.arch.lane_idx()
    tXrO_u32 = cute.recast_tensor(tXrO, cutlass.Uint32)
    num_vec_blocks = cute.size(tXcX, mode=[0, 1])
    for k in cutlass.range_constexpr(num_vec_blocks):
        coord = tXcX[(0, k), 0, 0]
        row = coord[0]
        col = coord[1]
        v0 = tXrO_u32[4 * k + 0]
        v1 = tXrO_u32[4 * k + 1]
        v2 = tXrO_u32[4 * k + 2]
        v3 = tXrO_u32[4 * k + 3]
        local_max = bfloat2_hmax_reduce_to_f32(bfloat2_max_abs_4(v0, v1, v2, v3))
        global_max = reduce_max_4threads(local_max)
        scale_u32 = float_to_ue8m0_fast(global_max * Float32(INV_FLOAT8_E4M3_MAX))
        inv2 = _ue8m0_to_inv_scale_bf16x2(scale_u32)
        q0 = _bf16x4_to_e4m3x4_scaled(v0, v1, inv2)
        q1 = _bf16x4_to_e4m3x4_scaled(v2, v3, inv2)
        if row < M:
            _st_global_v2_u32(get_ptr_as_int64(mQ, Int64(row) * H + col), q0, q1)
        if lane % 4 == 0:
            sf_off = compute_sf_index_swizzled_128x4_gpu(
                Int32(row), Int32(col // MXFP8_BLOCK), Int32(padded_sf_cols)
            )
            if row < M:
                mS[sf_off] = scale_u32.to(Uint8)
            else:
                mS[sf_off] = Uint8(0)


class FusedAddRMSNormMxfp8Kernel(FusedAddRMSNormKernel):
    """``FusedAddRMSNormKernel`` writing the residual and an MXFP8 output.

    The kernel body is FlashInfer 0.7.0's; only the final store of the normed
    activation is replaced by the MXFP8 epilogue, and the grid also covers the
    padding rows of the swizzled scale tensor.
    """

    def __init__(self, dtype, H: int, weight_bias: float = 0.0, sm_version=None):
        super().__init__(dtype, H, weight_bias, sm_version=sm_version)
        _check_tiling(self, H)
        self.padded_sf_cols = (H // MXFP8_BLOCK + 3) // 4 * 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        padded_M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = RMSNormKernel._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        self.kernel(mX, mR, mW, mQ, mS, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(padded_M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
        cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sR = smem.allocate_tensor(
                mR.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )
        mbar_ptr = None

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_store = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)
        tiled_copy_store = cute.make_tiled_copy(copy_atom_store, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)
        thr_copy_O = tiled_copy_store.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        tRgR = thr_copy_X.partition_S(gR)
        tRrR = cute.make_fragment_like(tRgR)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)
            tRsR = thr_copy_X.partition_D(sR)

        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        tXgO = thr_copy_O.partition_D(gX)
        tRgO = thr_copy_O.partition_D(gR)
        tXrO = cute.make_fragment_like(tXgO)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + residual, compute h, reduce =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
                cute.copy(copy_atom_async, tRgR, tRsR, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
            cute.autovec_copy(tRsR, tRrR)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            tRrR.store(cute.zeros_like(tRrR, dtype=mR.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)
                cute.copy(copy_atom_sync, tRgR, tRrR, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x_in = tXrX.load().to(Float32)
        r_in = tRrR.load().to(Float32)
        h = x_in + r_in

        tXrO.store(h.to(mR.element_type))
        if row_in_bounds:
            cute.copy(copy_atom_store, tXrO, tRgO, pred=tXpX)

        h_sq = h * h
        sum_sq = row_reduce_sum_multirow(
            h_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        # ===== Pass 2: Normalize, round to BF16, quantize to MXFP8 =====
        w = tXrW.load().to(Float32)
        y = h * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mX.element_type))
        _quantize_store_mxfp8(tXrO, tXcX, mQ, mS, M, H, self.padded_sf_cols)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


class RMSNormMxfp8Kernel(RMSNormKernel):
    """``RMSNormKernel`` whose output is MXFP8 (see FusedAddRMSNormMxfp8Kernel)."""

    def __init__(self, dtype, H: int, weight_bias: float = 0.0, sm_version=None):
        super().__init__(dtype, H, weight_bias, sm_version=sm_version)
        _check_tiling(self, H)
        self.padded_sf_cols = (H // MXFP8_BLOCK + 3) // 4 * 4

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        padded_M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        stream,
    ):
        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        self.kernel(mX, mW, mQ, mS, M, eps, enable_pdl, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(padded_M, self.rows_per_block), 1, 1],
            block=[self.num_threads, 1, 1],
            smem=self._smem_size_in_bytes(),
            stream=stream,
            use_pdl=enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mW: cute.Tensor,
        mQ: cute.Tensor,
        mS: cute.Tensor,
        M: Int64,
        eps: Float32,
        enable_pdl: cutlass.Constexpr[bool],
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        if enable_pdl:
            cute.arch.griddepcontrol_wait()

        H = self.H
        cluster_n = self.cluster_n
        weight_bias = self.weight_bias
        copy_bits = self.copy_bits
        threads_per_row = tv_layout.shape[0][0]
        rows_per_block = tiler_mn[0]
        warps_per_row = max(threads_per_row // 32, 1)
        cluster_y = cutlass.const_expr(0)

        # ===== Allocate shared memory =====
        smem = cutlass.utils.SmemAllocator()

        if cutlass.const_expr(self.use_async_copy):
            sX = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        reduction_buffer = smem.allocate_tensor(
            Float32,
            cute.make_layout((rows_per_block, warps_per_row)),
            byte_alignment=4,
        )
        mbar_ptr = None

        # ===== Coordinate tracking and tiling =====
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        mW_expanded_layout = cute.prepend(
            mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
        )
        mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
        gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # ===== Create TiledCopy atoms =====
        copy_atom_sync = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mX.element_type,
            num_bits_per_copy=copy_bits,
        )

        if cutlass.const_expr(self.use_async_copy):
            copy_atom_async = cute.make_copy_atom(
                cute.nvgpu.cpasync.CopyG2SOp(),
                mX.element_type,
                num_bits_per_copy=copy_bits,
            )
            tiled_copy_load = cute.make_tiled_copy(copy_atom_async, tv_layout, tiler_mn)
        else:
            tiled_copy_load = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        tiled_copy_W = cute.make_tiled_copy(copy_atom_sync, tv_layout, tiler_mn)

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_W = tiled_copy_W.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXcX = thr_copy_X.partition_S(cX)
        tXrX = cute.make_fragment_like(tXgX)

        if cutlass.const_expr(self.use_async_copy):
            tXsX = thr_copy_X.partition_D(sX)

        tWgW = thr_copy_W.partition_S(gW)
        tWrW = cute.make_fragment_like(tWgW)
        tXrW = thr_copy_X.retile(tWrW)

        tXrO = cute.make_fragment_like(tXgX)

        # ===== Bounds checking =====
        tXpX = predicate_k(tXcX, limit=H)
        tWpW = predicate_k(thr_copy_W.partition_S(cX), limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # ===== Pass 1: Load input + compute sum of squares =====
        if cutlass.const_expr(self.use_async_copy):
            if row_in_bounds:
                cute.copy(copy_atom_async, tXgX, tXsX, pred=tXpX)
            cute.arch.cp_async_commit_group()

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

            cute.arch.cp_async_wait_group(0)

            cute.autovec_copy(tXsX, tXrX)
        else:
            tXrX.store(cute.zeros_like(tXrX, dtype=mX.element_type))
            if row_in_bounds:
                cute.copy(copy_atom_sync, tXgX, tXrX, pred=tXpX)

            cute.copy(copy_atom_sync, tWgW, tWrW, pred=tWpW)

        x = tXrX.load().to(Float32)
        x_sq = x * x
        sum_sq = row_reduce_sum_multirow(
            x_sq, threads_per_row, reduction_buffer, mbar_ptr, cluster_n
        )

        mean_sq = sum_sq / Float32(H)
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        cute.arch.barrier()

        # ===== Pass 2: Normalize, round to BF16, quantize to MXFP8 =====
        if cutlass.const_expr(self.use_async_copy):
            cute.autovec_copy(tXsX, tXrX)
            x = tXrX.load().to(Float32)

        w = tXrW.load().to(Float32)
        y = x * rstd * (w + Float32(weight_bias))

        tXrO.store(y.to(mX.element_type))
        _quantize_store_mxfp8(tXrO, tXcX, mQ, mS, M, H, self.padded_sf_cols)

        if enable_pdl:
            cute.arch.griddepcontrol_launch_dependents()


def _fake_2d(dtype, H: int):
    sym_m = cute.sym_int(64)
    align = 128 if (H * dtype.width // 8) % 128 == 0 else 16
    return cute.runtime.make_fake_compact_tensor(
        dtype, (sym_m, H), stride_order=(1, 0), assumed_align=align
    )


@functools.cache
def _compiled(
    fused_add: bool,
    dtype_str: str,
    H: int,
    weight_bias: float,
    enable_pdl: bool,
    sm_version: int,
):
    dtype = get_cutlass_dtype(dtype_str)
    cls = FusedAddRMSNormMxfp8Kernel if fused_add else RMSNormMxfp8Kernel
    kernel_obj = cls(dtype, H, weight_bias, sm_version=sm_version)
    # Same fake-tensor contract as FlashInfer's compact norm kernels.
    x_fake = _fake_2d(dtype, H)
    w_fake = cute.runtime.make_fake_compact_tensor(dtype, (H,), assumed_align=16)
    q_fake = _fake_2d(cutlass.Uint8, H)
    s_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (cute.sym_int(),), assumed_align=16
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    tensors = (x_fake, _fake_2d(dtype, H), w_fake) if fused_add else (x_fake, w_fake)
    return cute.compile(
        kernel_obj,
        *tensors,
        q_fake,
        s_fake,
        Int64(1),
        Int64(128),
        Float32(1e-6),
        enable_pdl,
        stream_fake,
        options="--enable-tvm-ffi",
    )


def mxfp8_swizzled_scale_numel(num_tokens: int, hidden_size: int) -> int:
    padded_m = (num_tokens + SF_ROW_TILE - 1) // SF_ROW_TILE * SF_ROW_TILE
    padded_cols = (hidden_size // MXFP8_BLOCK + 3) // 4 * 4
    return padded_m * padded_cols


def _alloc_outputs(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
    M, H = x.shape
    q = torch.empty((M, H), dtype=torch.uint8, device=x.device)
    s = torch.empty(
        mxfp8_swizzled_scale_numel(M, H), dtype=torch.uint8, device=x.device
    )
    padded_m = (M + SF_ROW_TILE - 1) // SF_ROW_TILE * SF_ROW_TILE
    return q, s, padded_m


def _check_inputs(x: torch.Tensor, weight: torch.Tensor) -> None:
    assert x.dim() == 2 and x.is_contiguous() and x.dtype == torch.bfloat16, (
        x.shape,
        x.dtype,
    )
    assert weight.dtype == x.dtype and weight.is_contiguous()
    assert x.shape[0] * x.shape[1] <= 2**31 - 1


def gemma_fused_add_rmsnorm_mxfp8(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual += x`` in place, then MXFP8 of GemmaRMSNorm(residual).

    Returns ``(e4m3 [M, H], swizzled 128x4 UE8M0 scales)``, bitwise equal to
    ``flashinfer.norm.gemma_fused_add_rmsnorm`` followed by
    ``mxfp8_quantize(x, is_sf_swizzled_layout=True, backend="cute-dsl")``.
    ``x`` is not modified.
    """
    _check_inputs(x, weight)
    assert residual.shape == x.shape and residual.dtype == x.dtype
    assert residual.is_contiguous()
    M, H = x.shape
    q, s, padded_m = _alloc_outputs(x)
    kernel = _compiled(True, "bfloat16", H, 1.0, enable_pdl, get_sm_version(x.device))
    kernel(x, residual, weight, q, s, M, padded_m, eps)
    return q.view(torch.float8_e4m3fn), s


def gemma_rmsnorm_mxfp8(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    enable_pdl: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """MXFP8 of GemmaRMSNorm(x) (see gemma_fused_add_rmsnorm_mxfp8)."""
    _check_inputs(x, weight)
    M, H = x.shape
    q, s, padded_m = _alloc_outputs(x)
    kernel = _compiled(False, "bfloat16", H, 1.0, enable_pdl, get_sm_version(x.device))
    kernel(x, weight, q, s, M, padded_m, eps)
    return q.view(torch.float8_e4m3fn), s
