# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared expert of the MiniMax-M3 low-batch FFN in CuTe DSL.

The stock shared expert (an MXFP8 MLP: input quantization, gate_up GEMM,
SwiGLU-OAI, quantization, down GEMM) runs five kernels; its gate_up GEMM puts
12 CTAs on 9.4 MB of weights. Here, on the aux stream:

1. :class:`SharedInputQuant`: the input rows to MXFP8 (per 32 values a
   UE8M0 scale rounded up from amax / 448, E4M3 values; FlashInfer's
   ``mxfp8_quantize``), one CTA per row, into :data:`SPLITS` 8-row tiles of
   512 columns (tile ``s``: the rows' columns ``512 s ..``, [96, 512]) with
   their scales in the per-tile 128x4 atom layout;
2. :class:`SharedFc1`: ``gate_up = x @ w_gate_up^T`` split over K into
   :data:`SPLITS` slices of 512 columns, so the 1536 weight rows give 144
   work items: the persistent ``tcgen05`` GEMM the routed FC1 and FC2 build
   on (DeepSeek-V4.1's ``RoutedExpertGemm``), with the split as the A
   operand's batch mode and the B operand's token tile; each item stores its
   FP32 partial sums;
3. :class:`SharedAct`: the partial sums added in split order, rounded to
   BF16 as the stock GEMM stores ``gate_up``, SwiGLU-OAI to BF16 ``h`` and
   ``h`` to MXFP8;
4. :class:`SharedFc2`: ``h @ w_down^T`` (48 items), stored as BF16.

The weights are the MXFP8 linears' tensors as vLLM prepares them for
FlashInfer (K-major E4M3 values, E8M0 scales in the 128x4 layout), read in
place. All four kernels use programmatic dependent launch: the GEMMs stream
their weights while the previous kernel finishes.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Uint8, Uint32

from vllm.models.common.ops.moe_tail_fusion.primitives import load_global_u32x4
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_front_cute import (  # noqa: E501
    bf16x4_to_e4m3x4,
    block_scale,
    inverse_scale,
    max_f32,
)
from vllm.models.deepseek_v41.nvidia.specialized_decode.moe.routed_expert_gemm_cute import (  # noqa: E501
    TILE,
    RoutedExpertGemm,
    tile_scale_bytes,
)

from .front_cute import HIDDEN, INTERMEDIATE, local_amax
from .gemm_cute import swiglu_oai_bf16x2

SPLITS = 12
SPLIT_K = HIDDEN // SPLITS
SF_VEC = 32
ATOM_BYTES = 512  # one 128-row x 4-scale atom


@cute.jit
def mxfp8_x8(words):
    """Eight BF16 values (four words) of a 32-value block spread over four
    consecutive threads: the E4M3 bytes as two words and the block's UE8M0
    code."""
    amax = local_amax(words)
    amax = max_f32(amax, cute.arch.shuffle_sync_bfly(amax, 1))
    amax = max_f32(amax, cute.arch.shuffle_sync_bfly(amax, 2))
    code = block_scale(amax)
    inverse = inverse_scale(code)
    return (
        bf16x4_to_e4m3x4(words[0], words[1], inverse),
        bf16x4_to_e4m3x4(words[2], words[3], inverse),
        code,
    )


@cute.jit
def store_mxfp8(
    quant: cute.Tensor,
    scales: cute.Tensor,
    words,
    row: Int32,
    first: Int32,
    split: cutlass.Constexpr[int],
):
    """Quantize the eight values at column ``first`` of ``row`` and store them
    and their block's scale. Columns are cut into tiles of ``split``: tile
    ``s`` of the operand is rows ``8 s ..`` with columns ``s * split ..``
    (``quant`` as words), its scales one atom block."""
    lo, hi, code = mxfp8_x8(words)
    tile = first // split
    offset = (tile * TILE + row) * split + first % split
    quant[offset // 4] = lo
    quant[offset // 4 + 1] = hi
    if first % SF_VEC == 0:
        block = (first % split) // SF_VEC
        scales[
            tile * tile_scale_bytes(split)
            + (block // 4) * ATOM_BYTES
            + row * 16
            + block % 4
        ] = Uint8(code)


class SharedInputQuant:
    """The input rows to MXFP8 (module docstring), one CTA of 768 threads per
    row."""

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        quant_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """``x``: BF16 [rows, 6144] as words; ``quant``: E4M3 [96, 512] as
        words (module docstring); ``scale``: :data:`SPLITS` tiles of atoms."""
        self.kernel(
            cute.make_tensor(x_ptr, cute.make_layout(rows * HIDDEN // 2)),
            cute.make_tensor(quant_ptr, cute.make_layout(TILE * HIDDEN // 4)),
            cute.make_tensor(
                scale_ptr, cute.make_layout(SPLITS * tile_scale_bytes(SPLIT_K))
            ),
        ).launch(
            grid=(rows, 1, 1), block=[HIDDEN // 8, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(self, x: cute.Tensor, quant: cute.Tensor, scales: cute.Tensor):
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_wait()
        words = load_global_u32x4(x.iterator + row * (HIDDEN // 2) + thread * 4)
        store_mxfp8(quant, scales, words, row, thread * 8, SPLIT_K)


class SharedFc1(RoutedExpertGemm):
    """gate_up, split over K (module docstring). The A operand's batch mode
    and the routing record's tile expert are the split; B's token tile ``s``
    is the input's columns of split ``s``. Item (split, m tile) stores its
    FP32 partial sums [split, 8 rows, 1536]."""

    def __init__(self, cta_k: int = 256, num_ab_stage: int = 6) -> None:
        super().__init__(
            2 * INTERMEDIATE, 2 * INTERMEDIATE // 128, SPLIT_K, cta_k, num_ab_stage
        )

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        cta_words: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w_gate_up (1536, 6144) E4M3, K-major, E8M0 scales in the 128x4
        layout; B: the quantized input (96, 512), tile ``s`` K split ``s``;
        out FP32."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        m = self.m
        a = cute.make_tensor(
            a_ptr,
            cute.make_layout((m, SPLIT_K, SPLITS), stride=(HIDDEN, 1, SPLIT_K)),
        )
        # The 128x4 layout of (m, HIDDEN) with K split: atoms of 4 scales
        # (128 columns) are 512 bytes apart, row tiles HIDDEN / 128 atoms.
        atoms = SPLIT_K // (4 * SF_VEC)
        sfa = cute.make_tensor(
            sfa_ptr,
            cute.make_layout(
                ((32, 4, m // 128), (SF_VEC, 4, atoms), SPLITS),
                stride=(
                    (16, 4, HIDDEN // (4 * SF_VEC) * ATOM_BYTES),
                    (0, 1, ATOM_BYTES),
                    atoms * ATOM_BYTES,
                ),
            ),
        )
        b = cute.make_tensor(
            b_ptr,
            cute.make_ordered_layout((SPLITS * TILE, SPLIT_K, 1), order=(1, 0, 2)),
        )
        sfb = cute.make_tensor(
            sfb_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                cute.make_ordered_layout(
                    (SPLITS * 128, SPLIT_K, 1), order=(1, 0, 2)
                ).shape,
                SF_VEC,
            ),
        )
        out = cute.make_tensor(out_ptr, cute.make_layout(SPLITS * TILE * m))
        route = cute.make_tensor(route_ptr, cute.make_layout(4 + 2 * cta_words))
        # A static record: no ready words.
        self._launch(
            a, sfa, b, sfb, route, (out,), (), cta_words, Int32(0), ctas, stream
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
        """The FP32 partial sums, at [split * 8 + row, weight row]."""
        (mOut,) = epi
        for i in cutlass.range_constexpr(cute.size(tDrAcc)):
            coord = tDcC[i]
            mOut[coord[1] * self.m + coord[0]] = tDrAcc[i]


class SharedAct:
    """Split sums, BF16 gate_up, SwiGLU-OAI and MXFP8 of ``h`` (module
    docstring), one CTA of 96 threads per row, eight columns of ``h`` per
    thread."""

    @cute.jit
    def __call__(
        self,
        partial_ptr: cute.Pointer,
        quant_ptr: cute.Pointer,
        scale_ptr: cute.Pointer,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """``partial``: FP32 [SPLITS, 8, 1536]; ``quant``: E4M3 [8, 768] as
        words; ``scale``: one tile of atoms."""
        self.kernel(
            cute.make_tensor(
                partial_ptr, cute.make_layout(SPLITS * TILE * 2 * INTERMEDIATE)
            ),
            cute.make_tensor(quant_ptr, cute.make_layout(TILE * INTERMEDIATE // 4)),
            cute.make_tensor(
                scale_ptr, cute.make_layout(tile_scale_bytes(INTERMEDIATE))
            ),
        ).launch(
            grid=(rows, 1, 1),
            block=[INTERMEDIATE // 8, 1, 1],
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(self, partial: cute.Tensor, quant: cute.Tensor, scales: cute.Tensor):
        cute.arch.griddepcontrol_launch_dependents()
        thread, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        first = thread * 8
        gate = cute.make_rmem_tensor((8,), Float32)
        up = cute.make_rmem_tensor((8,), Float32)
        gate.fill(0.0)
        up.fill(0.0)
        cute.arch.griddepcontrol_wait()
        for split in cutlass.range_constexpr(SPLITS):
            base = (split * TILE + row) * 2 * INTERMEDIATE + first
            for i in cutlass.range_constexpr(8):
                gate[i] = gate[i] + partial[base + i]
                up[i] = up[i] + partial[base + INTERMEDIATE + i]
        # gate_up as the stock GEMM stores it, then h in BF16.
        words = cute.make_rmem_tensor((4,), Uint32)
        for i in cutlass.range_constexpr(4):
            words[i] = swiglu_oai_bf16x2(
                _bf16_round(up[2 * i]),
                _bf16_round(gate[2 * i]),
                Float32(1.0),
                _bf16_round(up[2 * i + 1]),
                _bf16_round(gate[2 * i + 1]),
            )
        store_mxfp8(quant, scales, words, row, first, INTERMEDIATE)


@cute.jit
def _bf16_round(value: Float32) -> Float32:
    """``value`` rounded to BF16 (nearest even), as FP32."""
    return Float32(cutlass.BFloat16(value))


class SharedFc2(RoutedExpertGemm):
    """down: BF16 ``h @ w_down^T`` (module docstring), rows below the
    record's limit stored."""

    def __init__(self, cta_k: int = 256, num_ab_stage: int = 6) -> None:
        super().__init__(HIDDEN, HIDDEN // 128, INTERMEDIATE, cta_k, num_ab_stage)

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        rows: Int32,
        cta_words: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w_down (6144, 768) E4M3, K-major, E8M0 scales in the 128x4
        layout; B: the quantized ``h`` (8, 768); out: BF16 [rows, 6144]."""
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.sf_dtype = sfa_ptr.value_type
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((self.m, self.k, 1), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, SF_VEC)
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((TILE, self.k, 1), order=(1, 0, 2))
        )
        sfb = cute.make_tensor(
            sfb_ptr,
            blockscaled_utils.tile_atom_to_shape_SF(
                cute.make_ordered_layout((128, self.k, 1), order=(1, 0, 2)).shape,
                SF_VEC,
            ),
        )
        out = cute.make_tensor(
            out_ptr, cute.make_ordered_layout((self.m, rows), order=(0, 1))
        )
        route = cute.make_tensor(route_ptr, cute.make_layout(4 + 2 * cta_words))
        # A static record: no ready words.
        self._launch(
            a, sfa, b, sfb, route, (out,), (), cta_words, Int32(0), ctas, stream
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
        """Round to BF16 (nearest even) and store the rows below the limit."""
        (mOut,) = epi
        row_limit = mRoute[4 + cta_words + n_tile]
        tDrC = cute.make_rmem_tensor(tDrAcc.shape, cutlass.BFloat16)
        tDrC.store(tDrAcc.load().to(cutlass.BFloat16))
        for i in cutlass.range_constexpr(cute.size(tDrC)):
            coord = tDcC[i]
            if coord[1] < row_limit:
                mOut[coord[0], coord[1]] = tDrC[i]
