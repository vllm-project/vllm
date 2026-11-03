# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Routed-expert FC1 and FC2 of the MiniMax-M3 low-batch FFN in CuTe DSL.

Both are the persistent ``tcgen05`` block-scaled GEMM of the DeepSeek-V4.1
low-batch MoE (``RoutedExpertGemm`` in its ``routed_expert_gemm_cute.py``):
128 x 8 output tiles over the (weight tile, token tile) work items of the
routing record, A at each tile's expert, double-buffered accumulators. Here
both operands are NVFP4 (E2M1 values, E4M3 scales per 16 values), the
weights as vLLM prepares them for the TRT-LLM-gen NVFP4 experts: rows
shuffled for 128-row epilogue tiles (in each block of 32, shuffled row
``s0 + 8 * s1`` holds row ``4 * s0 + s1``), scales in the 128x4 layout, and
in w13 the up and gate rows interleaved (even rows up, odd rows gate).

The activations carry one FP32 scale per row (``pt``, per-token NVFP4,
``front_cute.py``) and the weights one per expert (``alpha``), both applied
to the FP32 accumulator in the epilogue:

* FC1 (:class:`RoutedFc1`) multiplies the whole routed input (at most eight
  rows) by each tile's expert and keeps, for every row of the tile, the column
  of the token routed to it: ``up`` and ``gate`` at their scales, then
  SwiGLU-OAI ``g * sigmoid(1.702 g) * (u + 1)`` with ``g = min(gate, 7)`` and
  ``u = clamp(up, -7, 7)``, stored as BF16 ``h`` rows of the tile (the stock
  FC1 stores BF16 too, and FC2's input is quantized from it);
* FC2 (:class:`RoutedFc2`) multiplies token tile ``c`` of the quantized
  ``h`` and stores the BF16 rows ``8c ..`` of the output in the same
  permuted order, unweighted.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op

from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm_cute import (  # noqa: E501
    TILE,
    RoutedExpertGemm,
    unshuffled_row,
)

from .front_cute import SF_VEC

# E4M3 1.0 in each byte: the scale of B rows past the input's end.
UNIT_SCALES = 0x38383838

# h = SwiGLU-OAI of up ($1) and gate ($2) at scale $3, as BF16 bits of two
# columns packed (column p in the low half): $0 for (up $1, gate $2) and
# (up $4, gate $5).
_SWIGLU_OAI_X2 = """{
.reg .f32 u, g, e, h0, h1;
.reg .b16 b0, b1;
mul.f32 u, $1, $3;
mul.f32 g, $2, $3;
min.f32 g, g, 0f40E00000;
min.f32 u, u, 0f40E00000;
max.f32 u, u, 0fC0E00000;
mul.f32 e, g, 0fC01D265F;
ex2.approx.ftz.f32 e, e;
add.f32 e, e, 0f3F800000;
rcp.approx.ftz.f32 e, e;
add.f32 u, u, 0f3F800000;
mul.f32 h0, g, e;
mul.f32 h0, h0, u;
mul.f32 u, $4, $3;
mul.f32 g, $5, $3;
min.f32 g, g, 0f40E00000;
min.f32 u, u, 0f40E00000;
max.f32 u, u, 0fC0E00000;
mul.f32 e, g, 0fC01D265F;
ex2.approx.ftz.f32 e, e;
add.f32 e, e, 0f3F800000;
rcp.approx.ftz.f32 e, e;
add.f32 u, u, 0f3F800000;
mul.f32 h1, g, e;
mul.f32 h1, h1, u;
cvt.rn.bf16x2.f32 $0, h1, h0;
}"""


@dsl_user_op
def swiglu_oai_bf16x2(up0, gate0, scale, up1, gate1, *, loc=None, ip=None) -> Uint32:
    """Two columns of ``h`` of one token as packed BF16 (see above)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v).ir_value(loc=loc, ip=ip)
                for v in (up0, gate0, scale, up1, gate1)
            ],
            _SWIGLU_OAI_X2,
            "=r,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


class RoutedFc1(RoutedExpertGemm):
    """FC1 (module docstring). B is the routed input's E2M1 values; its
    linear scales are written into the layout the MMA reads by the B load
    warp."""

    route_before_wait = False

    def __init__(
        self, hidden: int, intermediate: int, cta_k: int = 256, num_ab_stage: int = 8
    ) -> None:
        super().__init__(
            2 * intermediate, 2 * intermediate // 128, hidden, cta_k, num_ab_stage
        )
        self.sf_vec_size = SF_VEC
        self.intermediate = intermediate
        # One lane per (row, scale atom) of a stage.
        assert TILE * cta_k // (4 * SF_VEC) <= 32

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        b_pt_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        h_rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w13 (2 x intermediate, hidden, experts) E2M1, K-major, with
        E4M3 scales; B: x (rows, hidden) E2M1, ``b_scale`` its scales as
        words (rows, hidden / 64), ``b_pt`` FP32 [rows]; ``alpha`` FP32 per
        expert; H BF16 pairs (``h_rows``, intermediate / 2)."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, experts), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2))
        )
        words = self.k // (4 * SF_VEC)
        b_scale = cute.make_tensor(
            b_scale_ptr, cute.make_layout((rows, words), stride=(words, 1))
        )
        h = cute.make_tensor(h_ptr, cute.make_layout(h_rows * (self.intermediate // 2)))
        b_pt = cute.make_tensor(b_pt_ptr, cute.make_layout(rows))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        route = cute.make_tensor(
            route_ptr, cute.make_layout(4 + 2 * cta_words + h_rows)
        )
        self._launch(
            a,
            sfa,
            b,
            None,
            route,
            (h, b_pt, alpha),
            (b_scale,),
            cta_words,
            ctas,
            stream,
        )

    @cute.jit
    def b_tile_index(self, n_tile: Int32) -> Int32:
        """Every item multiplies all of ``x``."""
        return Int32(0)

    @cute.jit
    def write_b_scales(
        self, sSFB: cute.Tensor, stage: Int32, k_tile: Int32, b_side: tuple
    ):
        """Copy the stage's scales of ``x`` (four per 32-bit word, one word
        per 64 columns) into rows 0-7 of B's scale atoms; rows past ``x``'s
        end get scale 1."""
        (b_scale,) = b_side
        atoms = self.cta_k // (4 * SF_VEC)
        stage_bytes = 512 * atoms
        words = cute.make_tensor(
            cute.recast_ptr(sSFB.iterator, dtype=Uint32),
            cute.make_layout(self.num_ab_stage * stage_bytes // 4),
        )
        lane = cute.arch.lane_idx()
        if lane < TILE * atoms:
            row = lane // atoms
            atom = lane - row * atoms
            value = Uint32(UNIT_SCALES)
            if row < cute.size(b_scale, mode=[0]):
                value = Uint32(b_scale[row, k_tile * atoms + atom])
            words[(stage * stage_bytes + atom * 512 + row * 16) // 4] = value
        # The MMA warp copies them to tensor memory through the async proxy.
        cute.arch.fence_proxy("async.shared", space="cta")
        cute.arch.sync_warp()

    @cute.jit
    def store(
        self,
        tDrAcc: cute.Tensor,
        tDcC: cute.Tensor,
        epi: tuple,
        mRoute: cute.Tensor,
        cta_words: Int32,
        n_tile: Int32,
        m_tile: Int32,
        item: Int32,
        exchange: cute.Tensor,
    ):
        """BF16 ``h`` of the item. The thread holds, for tokens ``t, t + 1``
        (``t`` = 2 x lane % 4), the up and gate rows of h columns ``p``
        (registers 0-3: up t, up t + 1, gate t, gate t + 1) and ``p + 1``
        (4-7), ``p`` = 64 x m_tile + 16 x warp + 2 x lane / 4."""
        mH, mPt, mAlpha = epi
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.epilogue_warp
        lane = cute.arch.lane_idx()
        token = (lane % 4) * 2
        pair = warp * 16 + (lane // 4) * 2
        alpha = Float32(mAlpha[mRoute[4 + n_tile]])
        row0 = n_tile * TILE
        valid = mRoute[4 + cta_words + n_tile] - row0
        # The tile rows of the thread's two tokens (-1: not routed here).
        row_a = Int32(-1)
        row_b = Int32(-1)
        for r in cutlass.range_constexpr(TILE):
            if r < valid:
                routed = mRoute[4 + 2 * cta_words + row0 + r]
                if routed == token:
                    row_a = Int32(r)
                if routed == token + 1:
                    row_b = Int32(r)
        h_row = self.intermediate // 2
        column = (m_tile * 64 + pair) // 2
        # A routed token is a row of x, so its scale is in bounds.
        scale_a = Float32(0.0)
        scale_b = Float32(0.0)
        if row_a >= 0:
            scale_a = Float32(mPt[token]) * alpha
        if row_b >= 0:
            scale_b = Float32(mPt[token + 1]) * alpha
        h_a = swiglu_oai_bf16x2(tDrAcc[0], tDrAcc[2], scale_a, tDrAcc[4], tDrAcc[6])
        h_b = swiglu_oai_bf16x2(tDrAcc[1], tDrAcc[3], scale_b, tDrAcc[5], tDrAcc[7])
        if row_a >= 0:
            mH[(row0 + row_a) * h_row + column] = h_a
        if row_b >= 0:
            mH[(row0 + row_b) * h_row + column] = h_b


class RoutedFc2(RoutedExpertGemm):
    """FC2 (module docstring). B is the quantized ``h``; its scales, one
    128-row atom per 8-row tile, are loaded with TMA."""

    def __init__(
        self, hidden: int, intermediate: int, cta_k: int = 256, num_ab_stage: int = 8
    ) -> None:
        super().__init__(hidden, hidden // 128, intermediate, cta_k, num_ab_stage)
        self.sf_vec_size = SF_VEC

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        b_pt_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w2 (hidden, intermediate, experts) E2M1, K-major, with E4M3
        scales; B: h (rows, intermediate) E2M1, ``rows`` the padded rows,
        ``b_pt`` FP32 [rows]; ``alpha`` FP32 per expert; C: (rows, hidden)
        BF16."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, experts), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((rows, self.k, 1), order=(1, 0, 2))
        )
        padded = cute.ceil_div(rows, TILE) * 128
        sfb = cute.make_tensor(
            sfb_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                cute.make_ordered_layout((padded, self.k, 1), order=(1, 0, 2)).shape,
                self.sf_vec_size,
            ),
        )
        c = cute.make_tensor(
            c_ptr, cute.make_ordered_layout((self.m, rows), order=(0, 1))
        )
        b_pt = cute.make_tensor(b_pt_ptr, cute.make_layout(rows))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        route = cute.make_tensor(route_ptr, cute.make_layout(4 + 2 * cta_words))
        self._launch(
            a, sfa, b, sfb, route, (c, b_pt, alpha), (), cta_words, ctas, stream
        )

    @cute.jit
    def store(
        self,
        tDrAcc: cute.Tensor,
        tDcC: cute.Tensor,
        epi: tuple,
        mRoute: cute.Tensor,
        cta_words: Int32,
        n_tile: Int32,
        m_tile: Int32,
        item: Int32,
        exchange: cute.Tensor,
    ):
        """Scale, round to BF16 (nearest even) and store the tile's rows below
        its limit at their unshuffled hidden index."""
        mC, mPt, mAlpha = epi
        row_limit = mRoute[4 + cta_words + n_tile]
        alpha = Float32(mAlpha[mRoute[4 + n_tile]])
        tDrOut = cute.make_rmem_tensor(tDrAcc.shape, Float32)
        for i in cutlass.range_constexpr(cute.size(tDrAcc)):
            coord = tDcC[i]
            scale = Float32(0.0)
            if coord[1] < row_limit:
                scale = Float32(mPt[coord[1]]) * alpha
            tDrOut[i] = tDrAcc[i] * scale
        tDrC = cute.make_rmem_tensor(tDrAcc.shape, cutlass.BFloat16)
        tDrC.store(tDrOut.load().to(cutlass.BFloat16))
        for i in cutlass.range_constexpr(cute.size(tDrC)):
            coord = tDcC[i]
            if coord[1] < row_limit:
                mC[unshuffled_row(coord[0]), coord[1]] = tDrC[i]
