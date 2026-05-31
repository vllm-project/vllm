# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# mypy: ignore-errors
"""Online softmax primitives.

Contains:
- ``Softmax``: SM80/90 base class with online softmax + finalize + rescale_O.
  The ``rescale_O`` path branches on ``arch >= 100`` to emit SM100 packed
  ``fmul.f32x2`` (2× CUDA-core throughput) when available.
- ``SoftmaxSm100``: SM100-specific subclass exposing fused ``update_row_max``,
  ``scale_apply_exp2_convert`` etc. used by the UTCMMA warp-specialized kernel.
"""

import math
import operator
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from quack import layout_utils
from quack.cute_dsl_utils import ParamsBase

from . import utils


@dataclass
class Softmax(ParamsBase):
    scale_log2: Float32
    num_rows: cutlass.Constexpr[int]
    row_max: cute.Tensor
    row_sum: cute.Tensor
    arch: cutlass.Constexpr[int] = 80
    softmax_scale: Float32 | None = None

    @staticmethod
    def create(
        scale_log2: Float32,
        num_rows: cutlass.Constexpr[int],
        arch: cutlass.Constexpr[int] = 80,
        softmax_scale: Float32 | None = None,
    ):
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return Softmax(scale_log2, num_rows, row_max, row_sum, arch, softmax_scale)

    def reset(self) -> None:
        self.row_max.fill(-Float32.inf)
        self.row_sum.fill(0.0)

    def _compute_row_max(
        self, acc_S_row: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fmax_reduce(acc_S_row, init_val, arch=self.arch)

    def _compute_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, init_val: float | Float32 | None = None
    ) -> Float32:
        return utils.fadd_reduce(acc_S_row_exp, init_val, arch=self.arch)

    @cute.jit
    def online_softmax(
        self,
        acc_S: cute.Tensor,
        is_first: cutlass.Constexpr[bool] = False,
        check_inf: cutlass.Constexpr[bool] = True,
    ) -> cute.Tensor:
        """Apply online softmax and return the row_scale to rescale O.

        On SM100+ the inner ``acc_S_row * scale_log2 - row_max_scaled`` is
        rewritten as explicit ``fma_packed_f32x2`` intrinsics — the DSL
        compiler does not fuse TensorSSA ``mul + sub`` into FFMA2 (NCU
        confirms: FFMA2 count is 0 for the TensorSSA path). The packed
        rewrite issues one FFMA.F32X2 per pair, halving the scalar FFMA
        instruction count for the softmax scale/subtract stage.
        """
        acc_S_mn = layout_utils.reshape_acc_to_mn(acc_S)
        row_scale = cute.make_rmem_tensor_like(self.row_max, Float32)

        row_max = self.row_max
        row_sum = self.row_sum
        scale_log2 = self.scale_log2
        arch = self.arch

        for r in cutlass.range(cute.size(row_max), unroll_full=True):
            acc_S_row_slice = acc_S_mn[r, None]
            acc_S_row = acc_S_row_slice.load()

            row_max_cur = utils.fmax_reduce(
                acc_S_row,
                init_val=row_max[r] if cutlass.const_expr(not is_first) else None,
                arch=arch,
            )

            row_max_cur = cute.arch.warp_reduction_max(row_max_cur, threads_in_group=4)
            row_max_prev = row_max[r]
            row_max[r] = row_max_cur

            if cutlass.const_expr(check_inf):
                row_max_cur = 0.0 if row_max_cur == -Float32.inf else row_max_cur

            row_max_cur_scaled = row_max_cur * scale_log2
            minus_row_max_scaled = -row_max_cur_scaled
            n = cute.size(acc_S_row_slice)

            if cutlass.const_expr(arch >= 100 and n % 2 == 0):
                # SM100 packed f32x2 FMA path: scale + subtract in one pass.
                for i in cutlass.range(0, n, 2, unroll_full=True):
                    acc_S_row_slice[i], acc_S_row_slice[i + 1] = (
                        cute.arch.fma_packed_f32x2(
                            (acc_S_row_slice[i], acc_S_row_slice[i + 1]),
                            (scale_log2, scale_log2),
                            (minus_row_max_scaled, minus_row_max_scaled),
                        )
                    )
                for i in cutlass.range(n, unroll_full=True):
                    acc_S_row_slice[i] = cute.math.exp2(
                        acc_S_row_slice[i], fastmath=True
                    )
                acc_S_row_exp = acc_S_row_slice.load()
            else:
                acc_S_row_exp = cute.math.exp2(
                    acc_S_row * scale_log2 - row_max_cur_scaled, fastmath=True
                )
                acc_S_row_slice.store(acc_S_row_exp)

            if cutlass.const_expr(is_first):
                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=None, arch=arch
                )
                row_scale[r] = 1.0
            else:
                row_scale[r] = cute.math.exp2(
                    (row_max_prev - row_max_cur) * scale_log2, fastmath=True
                )
                acc_S_row_sum = utils.fadd_reduce(
                    acc_S_row_exp, init_val=row_sum[r] * row_scale[r], arch=arch
                )

            row_sum[r] = acc_S_row_sum

        return row_scale

    @cute.jit
    def finalize(
        self, final_scale: Float32 = 1.0, sink_val: Float32 | cute.Tensor | None = None
    ) -> cute.Tensor:
        """Finalize the online softmax by computing the scale and logsumexp.

        On SM100+ with an even ``num_rows`` and no sink_val, the loop is
        unrolled in pairs so the key per-row arithmetic ― rcp*final_scale,
        max*scale_log2 + log2(sum), and the final *LN2 ― collapses into one
        ``mul_packed_f32x2`` + one ``fma_packed_f32x2`` + one more
        ``mul_packed_f32x2`` per row pair. Sink_val path stays scalar (rare).
        """
        if cutlass.const_expr(
            sink_val is not None and isinstance(sink_val, cute.Tensor)
        ):
            assert cute.size(sink_val) == cute.size(self.row_sum)
        row_sum = self.row_sum
        row_max = self.row_max
        scale_log2 = self.scale_log2

        row_sum.store(utils.warp_reduce(row_sum.load(), operator.add, width=4))
        row_scale = cute.make_rmem_tensor_like(row_max, Float32)

        LN2 = math.log(2.0)
        num_rows = cute.size(row_sum)
        use_packed = cutlass.const_expr(
            self.arch >= 100 and num_rows % 2 == 0 and sink_val is None
        )

        if use_packed:
            for r in cutlass.range(0, num_rows, 2, unroll_full=True):
                s0 = row_sum[r]
                s1 = row_sum[r + 1]
                m0 = row_max[r]
                m1 = row_max[r + 1]
                bad0 = s0 == 0.0 or s0 != s0
                bad1 = s1 == 0.0 or s1 != s1

                # row_scale = rcp_approx(safe_sum) * final_scale — rcp is scalar
                # (no packed rcp intrinsic); the trailing multiply packs.
                rcp0 = cute.arch.rcp_approx(1.0 if bad0 else s0)
                rcp1 = cute.arch.rcp_approx(1.0 if bad1 else s1)
                row_scale[r], row_scale[r + 1] = cute.arch.mul_packed_f32x2(
                    (rcp0, rcp1), (final_scale, final_scale)
                )

                # LSE = (row_max * scale_log2 + log2(row_sum)) * LN2
                #   packed FMA for (max*scale_log2 + log2_sum), packed MUL for *LN2.
                log0 = cute.math.log2(s0, fastmath=True)
                log1 = cute.math.log2(s1, fastmath=True)
                lse_pre_0, lse_pre_1 = cute.arch.fma_packed_f32x2(
                    (m0, m1), (scale_log2, scale_log2), (log0, log1)
                )
                lse_0, lse_1 = cute.arch.mul_packed_f32x2(
                    (lse_pre_0, lse_pre_1), (LN2, LN2)
                )
                row_sum[r] = -Float32.inf if bad0 else lse_0
                row_sum[r + 1] = -Float32.inf if bad1 else lse_1
        else:
            for r in cutlass.range(num_rows, unroll_full=True):
                if cutlass.const_expr(sink_val is not None):
                    sink_val_cur = (
                        sink_val
                        if not isinstance(sink_val, cute.Tensor)
                        else sink_val[r]
                    )
                    LOG2_E = math.log2(math.e)
                    row_sum[r] += cute.math.exp2(
                        sink_val_cur * LOG2_E - row_max[r] * scale_log2, fastmath=True
                    )

                acc_O_mn_row_is_zero_or_nan = (
                    row_sum[r] == 0.0 or row_sum[r] != row_sum[r]
                )
                row_scale[r] = (
                    cute.arch.rcp_approx(
                        row_sum[r] if not acc_O_mn_row_is_zero_or_nan else 1.0
                    )
                ) * final_scale
                row_sum_cur = row_sum[r]
                row_sum[r] = (
                    (
                        row_max[r] * scale_log2
                        + cute.math.log2(row_sum_cur, fastmath=True)
                    )
                    * LN2
                    if not acc_O_mn_row_is_zero_or_nan
                    else -Float32.inf
                )
        return row_scale

    @cute.jit
    def rescale_O(self, acc_O: cute.Tensor, row_scale: cute.Tensor) -> None:
        """Scale each row of acc_O by the given scale tensor."""
        acc_O_mn = layout_utils.reshape_acc_to_mn(acc_O)
        assert cute.size(row_scale) == cute.size(acc_O_mn, mode=[0])
        n = cute.size(acc_O_mn, mode=[1])
        if cutlass.const_expr(self.arch >= 100 and n % 2 == 0):
            # SM100: pack adjacent pairs into fmul.f32x2 (2× CUDA-core throughput).
            for r in cutlass.range(cute.size(row_scale), unroll_full=True):
                scale = row_scale[r]
                for j in cutlass.range(0, n, 2, unroll_full=True):
                    acc_O_mn[r, j], acc_O_mn[r, j + 1] = cute.arch.mul_packed_f32x2(
                        (acc_O_mn[r, j], acc_O_mn[r, j + 1]), (scale, scale)
                    )
        else:
            for r in cutlass.range(cute.size(row_scale), unroll_full=True):
                acc_O_mn[r, None].store(acc_O_mn[r, None].load() * row_scale[r])


@dataclass
class SoftmaxSm100(Softmax):
    """SM100-specific softmax: single-row, explicit f32x2 pack for FMA/exp2 paths."""

    rescale_threshold: cutlass.Constexpr[float] = 0.0

    @staticmethod
    def create(
        scale_log2: Float32,
        rescale_threshold: cutlass.Constexpr[float] = 0.0,
        softmax_scale: Float32 | None = None,
    ):
        num_rows = 1
        arch = 100
        row_max = cute.make_rmem_tensor(num_rows, Float32)
        row_sum = cute.make_rmem_tensor(num_rows, Float32)
        return SoftmaxSm100(
            scale_log2,
            num_rows,
            row_max,
            row_sum,
            arch,
            softmax_scale,
            rescale_threshold=rescale_threshold,
        )

    @cute.jit
    def update_row_max(
        self, acc_S_row: cute.TensorSSA, is_first: int
    ) -> tuple[Float32, Float32]:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale = 0.0
        else:
            row_max_old = self.row_max[0]
            row_max_new = self._compute_row_max(acc_S_row, init_val=row_max_old)
            row_max_safe = row_max_new if row_max_new != -cutlass.Float32.inf else 0.0
            acc_scale_ = (row_max_old - row_max_safe) * self.scale_log2
            acc_scale = cute.math.exp2(acc_scale_, fastmath=True)
            if cutlass.const_expr(self.rescale_threshold > 0.0):
                if acc_scale_ >= -self.rescale_threshold:
                    row_max_new = row_max_old
                    row_max_safe = row_max_old
                    acc_scale = 1.0
        self.row_max[0] = row_max_new
        return row_max_safe, acc_scale

    @cute.jit
    def update_row_max_only(self, acc_S_row: cute.TensorSSA, is_first: int) -> None:
        if cutlass.const_expr(is_first):
            row_max_new = self._compute_row_max(acc_S_row)
        else:
            row_max_new = self._compute_row_max(acc_S_row, init_val=self.row_max[0])
        self.row_max[0] = row_max_new

    def update_row_sum(
        self, acc_S_row_exp: cute.TensorSSA, row_scale: Float32, is_first: int = False
    ) -> None:
        init_val = (
            self.row_sum[0] * row_scale if cutlass.const_expr(not is_first) else None
        )
        self.row_sum[0] = self._compute_row_sum(acc_S_row_exp, init_val=init_val)

    @cute.jit
    def compute_scaled_exp2_row_sum(
        self,
        acc_S_row: cute.Tensor,
        scale: Float32,
    ) -> Float32:
        return utils.fadd_exp2_scaled_reduce(acc_S_row, scale, arch=self.arch)

    @cute.jit
    def scale_subtract_rowmax(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        row_max_scaled = row_max * self.scale_log2
        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (-row_max_scaled, -row_max_scaled),
            )

    @cute.jit
    def apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        acc_S_row_converted: cute.Tensor,
        ex2_emu_freq: cutlass.Constexpr[int] = 0,
        ex2_emu_res: cutlass.Constexpr[int] = 4,
        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                if cutlass.const_expr(ex2_emu_freq == 0):
                    acc_S_row_frg[k, j] = cute.math.exp2(
                        acc_S_row_frg[k, j], fastmath=True
                    )
                    acc_S_row_frg[k + 1, j] = cute.math.exp2(
                        acc_S_row_frg[k + 1, j], fastmath=True
                    )
                else:
                    if cutlass.const_expr(
                        k % ex2_emu_freq < ex2_emu_freq - ex2_emu_res
                        or j >= frg_cnt - 1
                        or j < ex2_emu_start_frg
                    ):
                        acc_S_row_frg[k, j] = cute.math.exp2(
                            acc_S_row_frg[k, j], fastmath=True
                        )
                        acc_S_row_frg[k + 1, j] = cute.math.exp2(
                            acc_S_row_frg[k + 1, j], fastmath=True
                        )
                    else:
                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                            utils.ex2_emulation_2(
                                acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]
                            )
                        )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

    @cute.jit
    def scale_apply_exp2_convert(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
        acc_S_row_converted: cute.Tensor,
    ):
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        minus_row_max_scaled = -row_max * self.scale_log2
        for i in cutlass.range_constexpr(0, cute.size(acc_S_row.shape), 2):
            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(
                (acc_S_row[i], acc_S_row[i + 1]),
                (self.scale_log2, self.scale_log2),
                (minus_row_max_scaled, minus_row_max_scaled),
            )

        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                acc_S_row_frg[k + 1, j] = cute.math.exp2(
                    acc_S_row_frg[k + 1, j], fastmath=True
                )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )

    @cute.jit
    def scale_apply_exp2_convert_sum(
        self,
        acc_S_row: cute.Tensor,
        row_max: Float32,
        acc_S_row_converted: cute.Tensor,
        init_sum: Float32,
    ) -> Float32:
        assert cute.size(acc_S_row.shape) % 2 == 0, (
            "acc_S_row must have an even number of elements"
        )
        minus_row_max_scaled = -row_max * self.scale_log2
        acc_sum = (init_sum, Float32(0.0))

        frg_tile = 32
        assert frg_tile % 2 == 0
        frg_cnt = cute.size(acc_S_row) // frg_tile
        assert cute.size(acc_S_row) % frg_tile == 0
        acc_S_row_frg = cute.logical_divide(acc_S_row, cute.make_layout(frg_tile))
        acc_S_row_converted_frg = cute.logical_divide(
            acc_S_row_converted, cute.make_layout(frg_tile)
        )
        for j in cutlass.range_constexpr(frg_cnt):
            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):
                acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = (
                    cute.arch.fma_packed_f32x2(
                        (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
                        (self.scale_log2, self.scale_log2),
                        (minus_row_max_scaled, minus_row_max_scaled),
                    )
                )
                acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)
                acc_S_row_frg[k + 1, j] = cute.math.exp2(
                    acc_S_row_frg[k + 1, j], fastmath=True
                )
                acc_sum = cute.arch.add_packed_f32x2(
                    acc_sum,
                    (acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]),
                )
            acc_S_row_converted_frg[None, j].store(
                acc_S_row_frg[None, j].load().to(acc_S_row_converted.element_type)
            )
        return acc_sum[0] + acc_sum[1]
