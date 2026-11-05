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

* FC1 (:class:`RoutedFc1`) gathers each token tile's rows of the routed input
  (as DeepSeek-V4.1's GEMM1 does) and multiplies them by the tile's expert:
  ``up`` and ``gate`` at their scales, then SwiGLU-OAI ``g * sigmoid(1.702 g)
  * (u + 1)`` with ``g = min(gate, 7)`` and ``u = clamp(up, -7, 7)``, stored
  as BF16 ``h`` rows of the tile (the stock FC1 stores BF16 too, and FC2's
  input is quantized from it); the last item of each token tile quantizes
  the tile's ``h`` rows to per-token NVFP4, FC2's input;
* FC2 (:class:`RoutedFc2`) multiplies token tile ``c`` of the quantized
  ``h`` and stores the BF16 rows ``8c ..`` of the output in the same
  permuted order, unweighted. It does not wait for the FC1 grid: its CTAs
  take the SMs FC1 leaves (the stage counts let one CTA of each share an SM)
  and stream their weights at once, and each item waits only for its token
  tile, through the record's ready words (``signal_ready`` /
  ``ready_target`` of ``RoutedExpertGemm``).
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Int64, Uint8, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import PipelineUserType, make_pipeline_state

from vllm.models.common.ops.moe_tail_fusion.primitives import load_global_u32x4
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_front_cute import (  # noqa: E501
    max_f32,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm_cute import (  # noqa: E501
    TILE,
    RoutedExpertGemm,
    cp_async_4,
    cp_async_16,
    publish_stage,
    unshuffled_row,
)

from .front_cute import (
    H_TILE_BYTES,
    INTERMEDIATE,
    SF_VEC,
    local_amax,
    per_token_scales,
    quantize_x8,
)

H_ROW_WORDS = INTERMEDIATE // 2  # one BF16 h row as words
# quantize_h_tile: 16 threads per row, three 16-value blocks each.
assert INTERMEDIATE == 16 * 3 * SF_VEC

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


@dsl_user_op
def count_item(addr: Int64, *, loc=None, ip=None) -> Int32:
    """Add 1 to the 32-bit word at global address ``addr`` (acquire-release,
    GPU scope); returns the previous count."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "atom.acq_rel.gpu.global.add.u32 $0, [$1], 1;",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def quantize_h_tile(
    h: cute.Tensor,
    tile: Int32,
    valid: Int32,
    thread: Int32,
    h_quant: cute.Tensor,
    h_scale: cute.Tensor,
    h_pt: cute.Tensor,
):
    """Per-token NVFP4 of token tile ``tile``'s BF16 ``h`` rows below
    ``valid``, by FC1's 128 epilogue threads: thread ``t`` takes row ``t /
    16`` of the tile, the 16-value blocks ``3 (t % 16) ..`` of it. The scales
    go in the 128x4 atom layout FC2 loads with TMA (one 128-row atom per tile,
    rows 0-7 used). Rows past ``valid`` hold no FC1 output: read as zero, not
    stored."""
    r = thread // 16
    lane = thread % 16
    row = tile * TILE + r
    keep = Uint32(0) - Uint32(r < valid)
    words = cute.make_rmem_tensor((6, 4), Uint32)
    block_amax = cute.make_rmem_tensor((3,), Float32)
    for c in cutlass.range_constexpr(6):
        loaded = load_global_u32x4(h.iterator + row * (H_ROW_WORDS) + lane * 24 + 4 * c)
        for i in cutlass.range_constexpr(4):
            words[c, i] = loaded[i] & keep
    row_amax = Float32(0.0)
    for b in cutlass.range_constexpr(3):
        first = [words[2 * b, i] for i in range(4)]
        second = [words[2 * b + 1, i] for i in range(4)]
        block_amax[b] = max_f32(local_amax(first), local_amax(second))
        row_amax = max_f32(row_amax, block_amax[b])
    for shift in cutlass.range_constexpr(4):
        row_amax = max_f32(row_amax, cute.arch.shuffle_sync_bfly(row_amax, 1 << shift))
    pt, encode = per_token_scales(row_amax)
    for b in cutlass.range_constexpr(3):
        lo, sf = quantize_x8([words[2 * b, i] for i in range(4)], block_amax[b], encode)
        hi, _ = quantize_x8(
            [words[2 * b + 1, i] for i in range(4)], block_amax[b], encode
        )
        block = lane * 3 + b
        if r < valid:
            h_quant[row * (H_ROW_WORDS // 4) + 2 * block] = lo
            h_quant[row * (H_ROW_WORDS // 4) + 2 * block + 1] = hi
            h_scale[tile * H_TILE_BYTES + (block // 4) * 512 + r * 16 + block % 4] = (
                Uint8(sf)
            )
    if r < valid and lane == 0:
        h_pt[row] = pt
    # FC2 loads them with TMA.
    cute.arch.fence_proxy("async.global")


class RoutedFc1(RoutedExpertGemm):
    """FC1 (module docstring). B is tile ``c``'s rows of the routed input:
    tile row ``r`` below the tile's limit is token ``record[4 + 2C + 8c + r]``,
    gathered with its linear scales by the B load warp, so accumulator column
    ``r`` is tile row ``r``.

    FC2's input. FC2's per-token NVFP4 scale needs a whole ``h`` row, which
    the tile's 12 items (m tiles) write. Each item's epilogue counts itself in
    the tile's ``done`` word (acquire-release); the item that completes the
    count quantizes the tile's rows (:func:`quantize_h_tile`) before the
    base's ready word counts it, so a tile's ready count reaches 12 only once
    FC2's input of the tile is stored. With ``overlap``, FC1 signals those
    counts and FC2 waits for them per item (``RoutedExpertGemm.signal_ready``);
    without, FC2 waits for the FC1 grid."""

    # Wait for the front before launching what follows: FC2 reads the routing
    # record and the fused boundary kernel the routing weights before their own
    # waits.
    route_before_wait = False
    gather_b = True

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        overlap: bool,
        cta_k: int = 256,
        num_ab_stage: int = 8,
    ) -> None:
        super().__init__(
            2 * intermediate, 2 * intermediate // 128, hidden, cta_k, num_ab_stage
        )
        self.signal_ready = overlap
        self.sf_vec_size = SF_VEC
        self.intermediate = intermediate
        # A stage of a row is one or two 128-byte swizzle atoms.
        assert cta_k in (256, 512)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        b_bytes_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        b_pt_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        h_quant_ptr: cute.Pointer,
        h_scale_ptr: cute.Pointer,
        h_pt_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        done_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        h_rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w13 (2 x intermediate, hidden, experts) E2M1, K-major, with
        E4M3 scales; B: x (rows, hidden) E2M1 (``b_bytes_ptr`` the same
        bytes), ``b_scale`` its scales as words (rows, hidden / 64), ``b_pt``
        FP32 [rows]; ``alpha`` FP32 per expert; H BF16 pairs (``h_rows``,
        intermediate / 2) and FC2's input: ``h_quant`` E2M1 as words
        (``h_rows``, intermediate / 8), ``h_scale`` one atom block per tile,
        ``h_pt`` FP32 [``h_rows``]; ``done``: one zeroed int32 per tile."""
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
        b_bytes = cute.make_tensor(b_bytes_ptr, cute.make_layout(rows * self.k // 2))
        words = self.k // (4 * SF_VEC)
        b_scale = cute.make_tensor(
            b_scale_ptr, cute.make_layout((rows, words), stride=(words, 1))
        )
        h = cute.make_tensor(h_ptr, cute.make_layout(h_rows * (self.intermediate // 2)))
        h_quant = cute.make_tensor(
            h_quant_ptr, cute.make_layout(h_rows * (self.intermediate // 8))
        )
        h_scale = cute.make_tensor(
            h_scale_ptr, cute.make_layout(h_rows // TILE * H_TILE_BYTES)
        )
        h_pt = cute.make_tensor(h_pt_ptr, cute.make_layout(h_rows))
        done = cute.make_tensor(done_ptr, cute.make_layout(cta_words))
        b_pt = cute.make_tensor(b_pt_ptr, cute.make_layout(rows))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(experts))
        ready = 4 + 2 * cta_words + h_rows
        route = cute.make_tensor(route_ptr, cute.make_layout(ready + cta_words))
        self._launch(
            a,
            sfa,
            b,
            None,
            route,
            (h, b_pt, alpha, done, h_quant, h_scale, h_pt),
            (b_bytes, b_scale),
            cta_words,
            ready,
            ctas,
            stream,
        )

    @cute.jit
    def b_tile_index(self, n_tile: Int32) -> Int32:
        """B is gathered per item; the epilogue's coordinates stay in tile 0."""
        return Int32(0)

    @cute.jit
    def gather_b_warp(
        self,
        sB: cute.Tensor,
        sSFB: cute.Tensor,
        ab_full_bar: cute.Pointer,
        ab_empty_bar: cute.Pointer,
        mRoute: cute.Tensor,
        b_side: tuple,
        cta_words: Int32,
        count: Int32,
    ):
        """Gather each item's tile rows of ``x`` and their scales into B, as
        DeepSeek-V4.1's ``RoutedExpertGemm1.gather_b_warp`` does for MXFP8.

        Per stage a row is ``cta_k / 2`` bytes in 128-byte swizzle atoms
        (``c`` 16-byte chunks): lane ``l`` copies chunk ``l % c`` of tile
        rows ``l // c + j * 32 / c`` with ``cp.async``, to its address under
        the 128-byte swizzle that TMA and the MMA apply, and 32-bit scale words
        (four 16-value blocks each) ``l, l + 32, ..`` of the stage's rows.
        Rows at or past the tile's limit are not loaded; their accumulator
        columns are never stored. Each stage is published two stages late, so
        the warp never waits for a copy it just issued."""
        x, b_scale = b_side
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        row_bytes = self.cta_k // 2
        chunks = row_bytes // 16  # per row and stage
        rows_per_pass = 32 // chunks
        passes = TILE // rows_per_pass
        atoms = self.cta_k // (4 * SF_VEC)  # scale words per row and stage
        sf_words = TILE * atoms // 32  # per lane
        b_stage_bytes = TILE * row_bytes
        sf_stage_bytes = 512 * atoms
        b_base = sB.iterator.toint()
        sf_base = sSFB.iterator.toint()
        chunk = lane % chunks
        # The chunk's offset in a row of an 8-row stage: atoms of 8 x 128 bytes.
        chunk_offset = (chunk // 8) * (TILE * 128) + (chunk % 8) * 16
        token = cute.make_rmem_tensor((passes,), Int32)
        sf_token = cute.make_rmem_tensor((sf_words,), Int32)
        state = make_pipeline_state(PipelineUserType.Producer, self.num_ab_stage)
        older = Int32(-1)  # stages issued but not yet published, oldest first
        newer = Int32(-1)
        for i in cutlass.range(count, unroll=1):
            item = bid + i * grid
            n_tile = item // self.m_tiles
            row0 = n_tile * TILE
            valid = mRoute[4 + cta_words + n_tile] - row0
            for j in cutlass.range_constexpr(passes):
                token[j] = Int32(-1)
                row = lane // chunks + j * rows_per_pass
                if row < valid:
                    token[j] = mRoute[4 + 2 * cta_words + row0 + row]
            for s in cutlass.range_constexpr(sf_words):
                sf_token[s] = Int32(-1)
                if (lane + 32 * s) // atoms < valid:
                    sf_token[s] = mRoute[
                        4 + 2 * cta_words + row0 + (lane + 32 * s) // atoms
                    ]
            for k_tile in cutlass.range(self.k_tiles, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                for s in cutlass.range_constexpr(sf_words):
                    word = lane + 32 * s
                    if sf_token[s] >= 0:
                        # Row r < 32 of a 128-row scale atom: byte 16 r, four
                        # K blocks.
                        cp_async_4(
                            sf_base
                            + state.index * sf_stage_bytes
                            + (word % atoms) * 512
                            + (word // atoms) * 16,
                            (
                                b_scale.iterator
                                + sf_token[s] * cute.size(b_scale, mode=[1])
                                + k_tile * atoms
                                + word % atoms
                            ).toint(),
                        )
                for j in cutlass.range_constexpr(passes):
                    if token[j] >= 0:
                        linear = (
                            b_base
                            + state.index * b_stage_bytes
                            + (lane // chunks + j * rows_per_pass) * 128
                            + chunk_offset
                        )
                        cp_async_16(
                            linear ^ (((linear >> 7) & 7) << 4),
                            (
                                x.iterator
                                + token[j] * (self.k // 2)
                                + k_tile * row_bytes
                                + chunk * 16
                            ).toint(),
                        )
                cute.arch.cp_async_commit_group()
                if older >= 0:
                    cute.arch.cp_async_wait_group(2)
                    publish_stage(ab_full_bar + older)
                older = newer
                newer = state.index
                state.advance()
        if older >= 0:
            cute.arch.cp_async_wait_group(1)
            publish_stage(ab_full_bar + older)
        if newer >= 0:
            cute.arch.cp_async_wait_group(0)
            publish_stage(ab_full_bar + newer)
        for _ in cutlass.range(self.num_ab_stage, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
            state.advance()

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
        """BF16 ``h`` of the item, then the tile's count (class docstring).
        The thread holds, for tile rows ``t, t + 1`` (``t`` = 2 x lane % 4),
        the up and gate rows of h columns ``p`` (registers 0-3: up t, up t +
        1, gate t, gate t + 1) and ``p + 1`` (4-7), ``p`` = 64 x m_tile + 16 x
        warp + 2 x lane / 4."""
        mH, mPt, mAlpha, mDone, mHQuant, mHScale, mHPt = epi
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.epilogue_warp
        lane = cute.arch.lane_idx()
        row_a = (lane % 4) * 2
        pair = warp * 16 + (lane // 4) * 2
        alpha = Float32(mAlpha[mRoute[4 + n_tile]])
        row0 = n_tile * TILE
        valid = mRoute[4 + cta_words + n_tile] - row0
        tokens = 4 + 2 * cta_words + row0
        h_row = self.intermediate // 2
        column = (m_tile * 64 + pair) // 2
        # Rows past the limit hold no token: scale 0, not stored.
        scale_a = Float32(0.0)
        scale_b = Float32(0.0)
        if row_a < valid:
            scale_a = Float32(mPt[mRoute[tokens + row_a]]) * alpha
        if row_a + 1 < valid:
            scale_b = Float32(mPt[mRoute[tokens + row_a + 1]]) * alpha
        h_a = swiglu_oai_bf16x2(tDrAcc[0], tDrAcc[2], scale_a, tDrAcc[4], tDrAcc[6])
        h_b = swiglu_oai_bf16x2(tDrAcc[1], tDrAcc[3], scale_b, tDrAcc[5], tDrAcc[7])
        if row_a < valid:
            mH[(row0 + row_a) * h_row + column] = h_a
        if row_a + 1 < valid:
            mH[(row0 + row_a + 1) * h_row + column] = h_b
        # Every epilogue thread's stores of the item, then one count; the
        # previous count goes through ``exchange`` to the other threads.
        thread = warp * 32 + lane
        cute.arch.barrier(barrier_id=3, number_of_threads=128)
        if thread == 0:
            exchange[0] = Float32(count_item((mDone.iterator + n_tile).toint()))
        cute.arch.barrier(barrier_id=3, number_of_threads=128)
        if Int32(exchange[0]) == self.m_tiles - 1:
            # The other items' h rows, released by their counts.
            cute.arch.fence_acq_rel_gpu()
            quantize_h_tile(mH, n_tile, valid, thread, mHQuant, mHScale, mHPt)


class RoutedFc2(RoutedExpertGemm):
    """FC2 (module docstring). B is the quantized ``h``; its scales, one
    128-row atom per 8-row tile, are loaded with TMA once FC1 has counted all
    its items of the tile (``fc1_m_tiles``, the ready target), or, with 0,
    after the FC1 grid."""

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        fc1_m_tiles: int,
        cta_k: int = 256,
        num_ab_stage: int = 8,
    ) -> None:
        super().__init__(hidden, hidden // 128, intermediate, cta_k, num_ab_stage)
        self.sf_vec_size = SF_VEC
        self.ready_target = fc1_m_tiles

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
        ready = 4 + 2 * cta_words + rows
        route = cute.make_tensor(route_ptr, cute.make_layout(ready + cta_words))
        self._launch(
            a, sfa, b, sfb, route, (c, b_pt, alpha), (), cta_words, ready, ctas, stream
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
