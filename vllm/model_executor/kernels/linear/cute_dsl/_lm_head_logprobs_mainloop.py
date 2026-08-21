# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""CuTe DSL LM-head projection for SM80+ GPUs with at least 144 KiB
of opt-in shared memory.

The GEMM mainloop is derived from CUTLASS's Ampere TensorOp GEMM example. It
uses a multistage ``cp.async`` pipeline and BF16 tensor-core MMA with FP32
accumulation. Logit tiles exist only in registers and CTA shared memory; the
kernel never writes a global ``[M, V]`` logits tensor.
"""

# Modified from NVIDIA CUTLASS v4.4.2:
# https://github.com/NVIDIA/cutlass/blob/da5e086dab31d63815acafdac9a9c5893b1c69e2/examples/python/CuTeDSL/ampere/tensorop_gemm.py  # noqa: E501

from __future__ import annotations

import math
from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils as cutlass_utils
from cutlass import cute
from cutlass.cute.runtime import make_fake_stream, make_fake_tensor

from vllm.platforms import current_platform
from vllm.triton_utils import triton
from vllm.utils.platform_utils import cuda_get_device_properties

_TILE_M = 128
_TILE_N = 256
_TILE_K = 64
_NUM_STAGES = 3
_ATOM_LAYOUT_M = 4
_ATOM_LAYOUT_N = 2
_K0_GROUP_N = 2
_TOPK_GROUP_N = 4
_TOPK_PARTIAL_WIDTH = 32
_K0_REQUIRED_SMEM = 144 * 1024
_TOPK_REQUIRED_SMEM = 144 * 1024


def _validate_device_environment(device_index: int) -> None:
    """Validate GPU requirements when a CuTe specialization compiles."""
    capability = current_platform.get_device_capability(device_index)
    if capability is None or capability.major < 8:
        current_sm = "unknown" if capability is None else capability.to_int()
        raise RuntimeError(
            "the compact prompt-logprobs path requires SM80 or newer; "
            f"the current GPU is SM{current_sm}"
        )

    try:
        (max_smem,) = cuda_get_device_properties(
            device_index, ("shared_memory_per_block_optin",), init_cuda=True
        )
    except AttributeError:
        (max_smem,) = cuda_get_device_properties(
            device_index, ("shared_memory_per_block",), init_cuda=True
        )
    required_smem = max(_K0_REQUIRED_SMEM, _TOPK_REQUIRED_SMEM)
    if max_smem < required_smem:
        raise RuntimeError(
            f"the current GPU exposes only {max_smem // 1024} KiB shared memory; "
            f"the compact prompt-logprobs path requires {required_smem // 1024} KiB"
        )


# Adapted from Dao-AILab/quack and modified for this SM80 accumulator layout:
# https://github.com/Dao-AILab/quack/blob/60d88082272a256fa9b3b2ab631c82cfa78337c6/quack/layout_utils.py  # noqa: E501
# Copyright (c) 2025, Wentao Guo, Ted Zadouri, Tri Dao.
def _reshape_acc_to_mn(acc: cute.Tensor) -> cute.Tensor:
    layout = acc.layout
    column_major = cute.make_layout(layout.shape)
    shape = (
        (column_major.shape[0][1], column_major.shape[1]),
        (
            column_major.shape[0][0],
            *column_major.shape[0][2:],
            column_major.shape[2],
        ),
        *column_major.shape[3:],
    )
    stride = (
        (column_major.stride[0][1], column_major.stride[1]),
        (
            column_major.stride[0][0],
            *column_major.stride[0][2:],
            column_major.stride[2],
        ),
        *column_major.stride[3:],
    )
    mn_layout = cute.composition(layout, cute.make_layout(shape, stride=stride))
    return cute.make_tensor(acc.iterator, mn_layout)


class _LMHeadLogprobsCpAsync:
    """BF16 LM-head GEMM whose epilogue emits one compact state per tile."""

    def __init__(
        self,
        tile_m: int,
        tile_n: int,
        tile_k: int,
        num_stages: int,
        atom_layout_m: int,
        atom_layout_n: int,
        num_topk: int,
        group_n: int,
    ) -> None:
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.tile_k = tile_k
        self.num_stages = num_stages
        self.atom_layout_mnk = (atom_layout_m, atom_layout_n, 1)
        self.num_threads = math.prod(self.atom_layout_mnk) * 32
        self.num_topk = num_topk
        self.group_n = group_n

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mTargetIds: cute.Tensor,
        mTargetLogits: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSumExp: cute.Tensor,
        mPartialRankCount: cute.Tensor,
        mPartialTopKValues: cute.Tensor,
        mPartialTopKIds: cute.Tensor,
        valid_vocab_size: cutlass.Int32,
        global_vocab_start: cutlass.Int32,
        vocab_block_start: cutlass.Int32,
        num_vocab_blocks: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        """Build static copy/MMA layouts and launch the grouped-vocab grid."""
        a_major = cutlass_utils.LayoutEnum.from_tensor(mA)
        b_major = cutlass_utils.LayoutEnum.from_tensor(mB)
        assert a_major == cutlass_utils.LayoutEnum.ROW_MAJOR
        assert b_major == cutlass_utils.LayoutEnum.ROW_MAJOR

        copy_bits = 128
        sA_layout, sA_swizzle = self._make_smem_layout(
            mA.element_type,
            (self.tile_m, self.tile_k, self.num_stages),
        )
        sB_layout, sB_swizzle = self._make_smem_layout(
            mB.element_type,
            (self.tile_n, self.tile_k, self.num_stages),
        )
        # K=0 reduces per-N-warp statistics directly from the accumulator.
        # Top-K needs the complete logits tile for row-wise warp selection.
        if cutlass.const_expr(self.num_topk == 0):
            stats_width = self.atom_layout_mnk[1] * 3
            sC_layout = cute.make_layout(
                (self.tile_m, stats_width),
                stride=(stats_width, 1),
            )
        else:
            sC_layout = cute.make_layout(
                (self.tile_m, self.tile_n), stride=(self.tile_n, 1)
            )

        # Both operands use 128-bit cp.async copies into swizzled SMEM tiles.
        copy_atom_A = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
            mA.element_type,
            num_bits_per_copy=copy_bits,
        )
        copy_atom_B = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
            mB.element_type,
            num_bits_per_copy=copy_bits,
        )
        tiled_copy_A = self._make_gmem_tiled_copy(copy_atom_A, mA.element_type)
        tiled_copy_B = self._make_gmem_tiled_copy(copy_atom_B, mB.element_type)
        mma_op = cute.nvgpu.warp.MmaF16BF16Op(
            mA.element_type, cutlass.Float32, (16, 8, 16)
        )
        tiled_mma = cute.make_tiled_mma(
            mma_op,
            cute.make_layout(self.atom_layout_mnk),
            permutation_mnk=(
                self.atom_layout_mnk[0] * 16,
                self.atom_layout_mnk[1] * 16,
                16,
            ),
        )
        # One CTA owns a row tile and group_n consecutive vocabulary tiles.
        num_row_blocks = cute.ceil_div(mA.shape[0], self.tile_m)
        num_output_groups = cute.ceil_div(num_vocab_blocks, self.group_n)
        self.kernel(
            mA,
            mB,
            mTargetIds,
            mTargetLogits,
            mPartialMax,
            mPartialSumExp,
            mPartialRankCount,
            mPartialTopKValues,
            mPartialTopKIds,
            valid_vocab_size,
            global_vocab_start,
            vocab_block_start,
            num_vocab_blocks,
            sA_layout,
            sA_swizzle,
            sB_layout,
            sB_swizzle,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        ).launch(
            grid=(
                num_row_blocks * num_output_groups,
                1,
                1,
            ),
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mTargetIds: cute.Tensor,
        mTargetLogits: cute.Tensor,
        mPartialMax: cute.Tensor,
        mPartialSumExp: cute.Tensor,
        mPartialRankCount: cute.Tensor,
        mPartialTopKValues: cute.Tensor,
        mPartialTopKIds: cute.Tensor,
        valid_vocab_size: cutlass.Int32,
        global_vocab_start: cutlass.Int32,
        vocab_block_start: cutlass.Int32,
        num_vocab_blocks: cutlass.Int32,
        sA_layout: cute.Layout,
        sA_swizzle: cute.Swizzle,
        sB_layout: cute.Layout,
        sB_swizzle: cute.Swizzle,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        # Flatten the row/vocabulary-group grid to keep the launch one-dimensional.
        block_linear, _, _ = cute.arch.block_idx()
        num_output_groups = cute.ceil_div(num_vocab_blocks, self.group_n)
        block_m = block_linear // num_output_groups
        output_group = block_linear % num_output_groups
        first_vocab_block = vocab_block_start + output_group * self.group_n

        @cute.struct
        class SharedStorageAB:
            a: cute.struct.Align[
                cute.struct.MemRange[mA.element_type, cute.cosize(sA_layout)],
                16,
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[mB.element_type, cute.cosize(sB_layout)],
                16,
            ]

        @cute.struct
        class SharedStorageC:
            c: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(sC_layout)],
                16,
            ]

        # GEMM staging and epilogue scratch are phase-disjoint, so they alias
        # one dynamic shared-memory allocation.
        smem = cutlass_utils.SmemAllocator()
        storage = smem.allocate(
            max(
                SharedStorageAB.size_in_bytes(),  # type: ignore[attr-defined]
                SharedStorageC.size_in_bytes(),  # type: ignore[attr-defined]
            ),
            byte_alignment=16,
        )
        sA = SharedStorageAB(  # type: ignore[call-arg]
            storage
        ).a.get_tensor(sA_layout, swizzle=sA_swizzle)
        sB = SharedStorageAB(  # type: ignore[call-arg]
            storage
        ).b.get_tensor(sB_layout, swizzle=sB_swizzle)
        sC = SharedStorageC(  # type: ignore[call-arg]
            storage
        ).c.get_tensor(sC_layout)

        tidx, _, _ = cute.arch.thread_idx()
        # A warp owns rows_per_warp rows. Lane row_slot carries each row's
        # LSE/rank state, while every lane carries one element of its top-32.
        running_max = cute.make_rmem_tensor(1, cutlass.Float32)
        running_sum_exp = cute.make_rmem_tensor(1, cutlass.Float32)
        running_rank_count = cute.make_rmem_tensor(1, cutlass.Int32)
        running_max[0] = -cutlass.Float32.inf
        running_sum_exp[0] = cutlass.Float32(0.0)
        running_rank_count[0] = cutlass.Int32(0)
        topk_state_size = (
            self.tile_m // (self.num_threads // 32) if self.num_topk > 0 else 1
        )
        running_topk_values = cute.make_rmem_tensor(topk_state_size, cutlass.Float32)
        running_topk_ids = cute.make_rmem_tensor(topk_state_size, cutlass.Int32)
        running_topk_values.fill(-cutlass.Float32.inf)
        running_topk_ids.fill(cutlass.Int32(0x7FFFFFFF))

        # Keep compact statistics in registers while this CTA traverses its
        # vocabulary group; only the final per-group state reaches GMEM.
        self._compute_vocab_block(
            mA,
            mB,
            mTargetIds,
            mTargetLogits,
            valid_vocab_size,
            global_vocab_start,
            first_vocab_block,
            block_m,
            running_max,
            running_sum_exp,
            running_rank_count,
            running_topk_values,
            running_topk_ids,
            sA,
            sB,
            sC,
            sA_layout,
            sA_swizzle,
            sB_layout,
            sB_swizzle,
            sC_layout,
            tiled_copy_A,
            tiled_copy_B,
            tiled_mma,
        )
        cute.arch.sync_threads()

        for group_offset in cutlass.range(1, self.group_n, unroll=1):
            local_vocab_block = output_group * self.group_n + group_offset
            if local_vocab_block < num_vocab_blocks:
                self._compute_vocab_block(
                    mA,
                    mB,
                    mTargetIds,
                    mTargetLogits,
                    valid_vocab_size,
                    global_vocab_start,
                    vocab_block_start + local_vocab_block,
                    block_m,
                    running_max,
                    running_sum_exp,
                    running_rank_count,
                    running_topk_values,
                    running_topk_ids,
                    sA,
                    sB,
                    sC,
                    sA_layout,
                    sA_swizzle,
                    sB_layout,
                    sB_swizzle,
                    sC_layout,
                    tiled_copy_A,
                    tiled_copy_B,
                    tiled_mma,
                )
                cute.arch.sync_threads()

        # Emit one compact partial state per row and vocabulary group.
        if cutlass.const_expr(self.num_topk == 0):
            threads_per_row = self.num_threads // self.tile_m
            if tidx < self.tile_m * threads_per_row:
                row_in_tile = tidx // threads_per_row
                lane_in_row = tidx % threads_per_row
                row = block_m * self.tile_m + row_in_tile
                if row < mA.shape[0] and lane_in_row == 0:
                    mPartialMax[row, output_group] = running_max[0]
                    mPartialSumExp[row, output_group] = running_sum_exp[0]
                    mPartialRankCount[row, output_group] = running_rank_count[0]
        else:
            warp_id = tidx // 32
            lane = tidx % 32
            rows_per_warp = self.tile_m // (self.num_threads // 32)
            for row_slot in cutlass.range(rows_per_warp, unroll=1):
                row_in_tile = warp_id * rows_per_warp + row_slot
                row = block_m * self.tile_m + row_in_tile
                if row < mA.shape[0]:
                    if lane == row_slot:
                        mPartialMax[row, output_group] = running_max[0]
                        mPartialSumExp[row, output_group] = running_sum_exp[0]
                        mPartialRankCount[row, output_group] = running_rank_count[0]
                    if lane < self.num_topk:
                        mPartialTopKValues[row, output_group, lane] = (
                            running_topk_values[row_slot]
                        )
                        mPartialTopKIds[row, output_group, lane] = running_topk_ids[
                            row_slot
                        ]

    @cute.jit
    def _compute_vocab_block(
        self,
        mA: cute.Tensor,
        mB: cute.Tensor,
        mTargetIds: cute.Tensor,
        mTargetLogits: cute.Tensor,
        valid_vocab_size: cutlass.Int32,
        global_vocab_start: cutlass.Int32,
        vocab_block: cutlass.Int32,
        block_m: cutlass.Int32,
        running_max: cute.Tensor,
        running_sum_exp: cute.Tensor,
        running_rank_count: cute.Tensor,
        running_topk_values: cute.Tensor,
        running_topk_ids: cute.Tensor,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sC: cute.Tensor,
        sA_layout: cute.Layout,
        sA_swizzle: cute.Swizzle,
        sB_layout: cute.Layout,
        sB_swizzle: cute.Swizzle,
        sC_layout: cute.Layout,
        tiled_copy_A: cute.TiledCopy,
        tiled_copy_B: cute.TiledCopy,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tiler_coord = (block_m, vocab_block, None)
        cta_tiler = (self.tile_m, self.tile_n, self.tile_k)

        # Shift K so the predicate-masked residual tile is consumed first;
        # all following K tiles are full-width and use the same copy layout.
        gA = cute.local_tile(
            mA,
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        gB = cute.local_tile(
            mB,
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        residual_k = mA.shape[1] - cutlass.Int32(self.tile_k) * gA.shape[2]
        gA = cute.domain_offset((0, residual_k, 0), gA)
        gB = cute.domain_offset((0, residual_k, 0), gB)
        gA = cute.make_tensor(gA.iterator.align(16), gA.layout)
        gB = cute.make_tensor(gB.iterator.align(16), gB.layout)

        # Coordinate tensors build boundary predicates without loading data.
        cA = cute.local_tile(
            cute.make_identity_tensor(mA.layout.shape),
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(1, None, 1),
        )
        cB = cute.local_tile(
            cute.make_identity_tensor(mB.layout.shape),
            tiler=cta_tiler,
            coord=tiler_coord,
            proj=(None, 1, 1),
        )
        cA = cute.domain_offset((0, residual_k, 0), cA)
        cB = cute.domain_offset((0, residual_k, 0), cB)

        # Partition GMEM sources and SMEM destinations by copy-thread ownership.
        thr_copy_A = tiled_copy_A.get_slice(tidx)
        thr_copy_B = tiled_copy_B.get_slice(tidx)
        tAgA = thr_copy_A.partition_S(gA)
        tAsA = thr_copy_A.partition_D(sA)
        tBgB = thr_copy_B.partition_S(gB)
        tBsB = thr_copy_B.partition_D(sB)
        tAcA = thr_copy_A.partition_S(cA)
        tBcB = thr_copy_B.partition_S(cB)

        tApA = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tAgA.shape[0][1],
                    cute.size(tAgA, mode=[1]),
                    cute.size(tAgA, mode=[2]),
                ),
                stride=(cute.size(tAgA, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        tBpB = cute.make_rmem_tensor(
            cute.make_layout(
                (
                    tBgB.shape[0][1],
                    cute.size(tBgB, mode=[1]),
                    cute.size(tBgB, mode=[2]),
                ),
                stride=(cute.size(tBgB, mode=[1]), 1, 0),
            ),
            cutlass.Boolean,
        )
        for rest_v in range(tApA.shape[0]):
            for m in range(tApA.shape[1]):
                tApA[rest_v, m, 0] = cute.elem_less(
                    tAcA[(0, rest_v), m, 0, 0][0], mA.shape[0]
                )
        for rest_v in range(tBpB.shape[0]):
            for n in range(tBpB.shape[1]):
                tBpB[rest_v, n, 0] = cute.elem_less(
                    tBcB[(0, rest_v), n, 0, 0][0], valid_vocab_size
                )

        # Masked elements must remain zero so they contribute nothing to MMA.
        tAsA.fill(0)
        tBsB.fill(0)
        cute.arch.sync_threads()
        num_smem_stages = cute.size(tAsA, mode=[3])
        k_tile_count = cute.size(tAgA, mode=[3])
        k_tile_index = cutlass.Int32(0)

        # Prime num_stages - 1 cp.async groups before the steady-state loop.
        for k in range(tApA.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tAcA[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_A,
                    tAgA[None, None, k, k_tile_index],
                    tAsA[None, None, k, 0],
                    pred=tApA[None, None, k],
                )
        for k in range(tBpB.shape[2]):
            if cute.elem_less(cutlass.Int32(-1), tBcB[0, 0, k, 0][1]):
                cute.copy(
                    tiled_copy_B,
                    tBgB[None, None, k, k_tile_index],
                    tBsB[None, None, k, 0],
                    pred=tBpB[None, None, k],
                )
        k_tile_index = k_tile_index + 1
        cute.arch.cp_async_commit_group()

        for k_tile in range(1, num_smem_stages - 1):
            if k_tile == k_tile_count:
                tApA.fill(0)
                tBpB.fill(0)
            cute.copy(
                tiled_copy_A,
                tAgA[None, None, None, k_tile_index],
                tAsA[None, None, None, k_tile],
                pred=tApA,
            )
            cute.copy(
                tiled_copy_B,
                tBgB[None, None, None, k_tile_index],
                tBsB[None, None, None, k_tile],
                pred=tBpB,
            )
            k_tile_index = k_tile_index + 1
            cute.arch.cp_async_commit_group()

        thr_mma = tiled_mma.get_slice(tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        # The full CTA output tile remains in FP32 registers across all K tiles.
        tCrC = cute.make_rmem_tensor(
            thr_mma.partition_shape_C((self.tile_m, self.tile_n)),
            cutlass.Float32,
        )
        tCrC.fill(0.0)

        # ldmatrix feeds register fragments consumed by BF16 mma.sync.
        atom_s2r_A = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
            mA.element_type,
        )
        atom_s2r_B = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(False, 4),
            mB.element_type,
        )
        tiled_s2r_A = cute.make_tiled_copy_A(atom_s2r_A, tiled_mma)
        tiled_s2r_B = cute.make_tiled_copy_B(atom_s2r_B, tiled_mma)
        thr_s2r_A = tiled_s2r_A.get_slice(tidx)
        thr_s2r_B = tiled_s2r_B.get_slice(tidx)
        tCsA_copy = thr_s2r_A.partition_S(sA)
        tCsB_copy = thr_s2r_B.partition_S(sB)
        tCrA_copy = thr_s2r_A.retile(tCrA)
        tCrB_copy = thr_s2r_B.retile(tCrB)

        smem_pipe_read = 0
        smem_pipe_write = num_smem_stages - 1
        tCsA_pipe = tCsA_copy[None, None, None, smem_pipe_read]
        tCsB_pipe = tCsB_copy[None, None, None, smem_pipe_read]
        num_k_block = cute.size(tCrA, mode=[2])
        if num_k_block > 1:
            cute.arch.cp_async_wait_group(num_smem_stages - 2)
            cute.arch.sync_threads()
            cute.copy(
                tiled_s2r_A,
                tCsA_pipe[None, None, 0],
                tCrA_copy[None, None, 0],
            )
            cute.copy(
                tiled_s2r_B,
                tCsB_pipe[None, None, 0],
                tCrB_copy[None, None, 0],
            )

        # Overlap the future GMEM copy, next SMEM fragment load, and current MMA.
        for k_tile in range(k_tile_count):
            for k_block in cutlass.range(num_k_block, unroll_full=True):
                if k_block == num_k_block - 1:
                    tCsA_pipe = tCsA_copy[None, None, None, smem_pipe_read]
                    tCsB_pipe = tCsB_copy[None, None, None, smem_pipe_read]
                    cute.arch.cp_async_wait_group(num_smem_stages - 2)
                    cute.arch.sync_threads()

                k_block_next = (k_block + 1) % num_k_block
                cute.copy(
                    tiled_s2r_A,
                    tCsA_pipe[None, None, k_block_next],
                    tCrA_copy[None, None, k_block_next],
                )
                cute.copy(
                    tiled_s2r_B,
                    tCsB_pipe[None, None, k_block_next],
                    tCrB_copy[None, None, k_block_next],
                )

                if k_block == 0:
                    if k_tile + num_smem_stages - 1 < k_tile_count:
                        cute.copy(
                            tiled_copy_A,
                            tAgA[None, None, None, k_tile_index],
                            tAsA[None, None, None, smem_pipe_write],
                            pred=tApA,
                        )
                        cute.copy(
                            tiled_copy_B,
                            tBgB[None, None, None, k_tile_index],
                            tBsB[None, None, None, smem_pipe_write],
                            pred=tBpB,
                        )
                    k_tile_index = k_tile_index + 1
                    cute.arch.cp_async_commit_group()
                    smem_pipe_write = smem_pipe_read
                    smem_pipe_read = smem_pipe_read + 1
                    if smem_pipe_read == num_smem_stages:
                        smem_pipe_read = 0

                cute.gemm(
                    tiled_mma,
                    tCrC,
                    tCrA[None, None, k_block],
                    tCrB[None, None, k_block],
                    tCrC,
                )

        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        # K=0 reduces fragments directly. Top-K stages logits in sC so a warp
        # can scan one complete row while preserving deterministic ordering.
        if cutlass.const_expr(self.num_topk == 0):
            self._update_stats_from_fragment(
                tCrC,
                thr_mma,
                mA,
                mTargetIds,
                mTargetLogits,
                valid_vocab_size,
                vocab_block,
                block_m,
                running_max,
                running_sum_exp,
                running_rank_count,
                sC,
            )
        else:
            tCsC = thr_mma.partition_C(sC)
            cute.autovec_copy(tCrC, tCsC)
            cute.arch.sync_threads()
            self._update_topk_warp_select_from_smem(
                sC,
                mA,
                mTargetIds,
                mTargetLogits,
                valid_vocab_size,
                global_vocab_start,
                vocab_block,
                block_m,
                running_max,
                running_sum_exp,
                running_rank_count,
                running_topk_values,
                running_topk_ids,
            )

    @cute.jit
    def _warp_bitonic_sort32_desc(
        self,
        value: cute.Tensor,
        token_id: cute.Tensor,
    ):
        # Each lane contributes one (logit, token ID) pair. The secondary key
        # keeps lower token IDs first when logits are equal.
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        for stage in cutlass.range_constexpr(5):
            segment = 2 << stage
            for step in cutlass.range_constexpr(stage + 1):
                stride = (1 << stage) >> step
                other_value = cute.arch.shuffle_sync_bfly(value[0], offset=stride)
                other_id = cute.arch.shuffle_sync_bfly(token_id[0], offset=stride)
                other_better = other_value > value[0] or (
                    other_value == value[0] and other_id < token_id[0]
                )
                self_better = value[0] > other_value or (
                    value[0] == other_value and token_id[0] < other_id
                )
                want_better = ((lane & stride) == 0) == ((lane & segment) == 0)
                if (want_better and other_better) or (not want_better and self_better):
                    value[0] = other_value
                    token_id[0] = other_id

    @cute.jit
    def _warp_bitonic_merge32_desc(
        self,
        selected_value: cute.Tensor,
        selected_id: cute.Tensor,
        candidate_value: cute.Tensor,
        candidate_id: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        # Reversing the sorted candidate sequence forms a bitonic sequence
        # with the current selection, which needs only the merge half-network.
        reversed_value = cute.arch.shuffle_sync(candidate_value[0], offset=31 - lane)
        reversed_id = cute.arch.shuffle_sync(candidate_id[0], offset=31 - lane)
        if reversed_value > selected_value[0] or (
            reversed_value == selected_value[0] and reversed_id < selected_id[0]
        ):
            selected_value[0] = reversed_value
            selected_id[0] = reversed_id

        for step in cutlass.range_constexpr(5):
            stride = 16 >> step
            other_value = cute.arch.shuffle_sync_bfly(selected_value[0], offset=stride)
            other_id = cute.arch.shuffle_sync_bfly(selected_id[0], offset=stride)
            other_better = other_value > selected_value[0] or (
                other_value == selected_value[0] and other_id < selected_id[0]
            )
            self_better = selected_value[0] > other_value or (
                selected_value[0] == other_value and selected_id[0] < other_id
            )
            want_better = (lane & stride) == 0
            if (want_better and other_better) or (not want_better and self_better):
                selected_value[0] = other_value
                selected_id[0] = other_id

    @cute.jit
    def _warp_select_add(
        self,
        candidate_value: cutlass.Float32,
        candidate_id: cutlass.Int32,
        selected_value: cute.Tensor,
        selected_id: cute.Tensor,
        buffer_length: cute.Tensor,
        value_buffer: cute.Tensor,
        id_buffer: cute.Tensor,
        buffer_row: cutlass.Int32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        # selected lane K-1 is the admission threshold for every candidate.
        threshold = cute.arch.shuffle_sync(selected_value[0], offset=self.num_topk - 1)
        threshold_id = cute.arch.shuffle_sync(selected_id[0], offset=self.num_topk - 1)
        do_add = candidate_value > threshold or (
            candidate_value == threshold and candidate_id < threshold_id
        )
        mask = cutlass.Uint32(cute.arch.vote_ballot_sync(do_add))
        if mask != 0:
            # Compact accepted lanes into sC scratch. Merge every full batch of
            # 32 so the persistent selection stays sorted in registers.
            lower_lanes = (cutlass.Uint32(1) << lane) - cutlass.Uint32(1)
            position = buffer_length[0] + cutlass.Int32(
                cute.arch.popc(mask & lower_lanes)
            )
            if do_add and position < 32:
                value_buffer[buffer_row, position] = candidate_value
                id_buffer[buffer_row, 32 + position] = candidate_id
                do_add = False

            buffer_length[0] += cutlass.Int32(cute.arch.popc(mask))
            if buffer_length[0] >= 32:
                cute.arch.sync_warp()
                batch_value = cute.make_rmem_tensor(1, cutlass.Float32)
                batch_id = cute.make_rmem_tensor(1, cutlass.Int32)
                batch_value[0] = value_buffer[buffer_row, lane]
                batch_id[0] = id_buffer[buffer_row, 32 + lane]
                self._warp_bitonic_sort32_desc(batch_value, batch_id)
                self._warp_bitonic_merge32_desc(
                    selected_value,
                    selected_id,
                    batch_value,
                    batch_id,
                )
                buffer_length[0] -= 32
                cute.arch.sync_warp()
            if do_add:
                position -= 32
                value_buffer[buffer_row, position] = candidate_value
                id_buffer[buffer_row, 32 + position] = candidate_id
            cute.arch.sync_warp()

    @cute.jit
    def _warp_select_done(
        self,
        selected_value: cute.Tensor,
        selected_id: cute.Tensor,
        buffer_length: cute.Tensor,
        value_buffer: cute.Tensor,
        id_buffer: cute.Tensor,
        buffer_row: cutlass.Int32,
    ):
        # Pad and merge the final incomplete candidate batch.
        if buffer_length[0] > 0:
            tidx, _, _ = cute.arch.thread_idx()
            lane = tidx % 32
            batch_value = cute.make_rmem_tensor(1, cutlass.Float32)
            batch_id = cute.make_rmem_tensor(1, cutlass.Int32)
            if lane < buffer_length[0]:
                batch_value[0] = value_buffer[buffer_row, lane]
                batch_id[0] = id_buffer[buffer_row, 32 + lane]
            else:
                batch_value[0] = -cutlass.Float32.inf
                batch_id[0] = cutlass.Int32(0x7FFFFFFF)
            self._warp_bitonic_sort32_desc(batch_value, batch_id)
            self._warp_bitonic_merge32_desc(
                selected_value,
                selected_id,
                batch_value,
                batch_id,
            )

    @cute.jit
    def _update_topk_warp_select_from_smem(
        self,
        sC: cute.Tensor,
        mA: cute.Tensor,
        mTargetIds: cute.Tensor,
        mTargetLogits: cute.Tensor,
        valid_vocab_size: cutlass.Int32,
        global_vocab_start: cutlass.Int32,
        vocab_block: cutlass.Int32,
        block_m: cutlass.Int32,
        running_max: cute.Tensor,
        running_sum_exp: cute.Tensor,
        running_rank_count: cute.Tensor,
        running_topk_values: cute.Tensor,
        running_topk_ids: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_id = tidx // 32
        lane = tidx % 32
        rows_per_warp = self.tile_m // (self.num_threads // 32)
        # The first 64 sC columns are dead after loading logits and are reused
        # as value/ID scratch for warp selection.
        sC_ids = cute.recast_tensor(sC, cutlass.Int32)

        for row_slot in cutlass.range(rows_per_warp, unroll=1):
            row_in_tile = warp_id * rows_per_warp + row_slot
            row = block_m * self.tile_m + row_in_tile
            row_valid = row < mA.shape[0]
            target_id = cutlass.Int32(-1)
            target_logit = cutlass.Float32(0.0)
            if row_valid:
                target_id = mTargetIds[row]
                target_logit = mTargetLogits[row]

            # A warp covers 256 vocabulary entries with eight values per lane.
            candidates = cute.make_rmem_tensor(8, cutlass.Float32)
            row_max = -cutlass.Float32.inf
            rank_count = cutlass.Int32(0)
            vocab_start = vocab_block * self.tile_n
            for candidate in cutlass.range(8, unroll=2):
                n = lane + candidate * 32
                local_vocab_id = vocab_start + n
                logit = -cutlass.Float32.inf
                if row_valid and local_vocab_id < valid_vocab_size:
                    logit = sC[row_in_tile, n]
                    # Use the separately computed global target value so rank
                    # and returned target logprob share the same TP-reduced logit.
                    if target_id == local_vocab_id:
                        logit = target_logit
                    row_max = cute.arch.fmax(row_max, logit)
                    if logit >= target_logit:
                        rank_count += 1
                candidates[candidate] = logit

            row_max = cute.arch.warp_reduction_max(row_max)
            rank_count = cute.arch.warp_reduction_sum(rank_count)
            row_sum_exp = cutlass.Float32(0.0)
            if row_valid:
                for candidate in cutlass.range(8, unroll=2):
                    row_sum_exp += cute.math.exp(
                        candidates[candidate] - row_max,
                        fastmath=True,
                    )
            row_sum_exp = cute.arch.warp_reduction_sum(row_sum_exp)

            # Lane row_slot owns this row's running LSE/rank state.
            previous_max = cute.arch.shuffle_sync(running_max[0], offset=row_slot)
            previous_sum_exp = cute.arch.shuffle_sync(
                running_sum_exp[0], offset=row_slot
            )
            previous_rank_count = cute.arch.shuffle_sync(
                running_rank_count[0], offset=row_slot
            )
            merged_max = cute.arch.fmax(previous_max, row_max)
            merged_sum_exp = previous_sum_exp * cute.math.exp(
                previous_max - merged_max, fastmath=True
            ) + row_sum_exp * cute.math.exp(row_max - merged_max, fastmath=True)
            if lane == row_slot:
                running_max[0] = merged_max
                running_sum_exp[0] = merged_sum_exp
                running_rank_count[0] = previous_rank_count + rank_count

            # Each lane holds one element of the row's sorted top-32 state.
            selected_value = cute.make_rmem_tensor(1, cutlass.Float32)
            selected_id = cute.make_rmem_tensor(1, cutlass.Int32)
            selected_value[0] = running_topk_values[row_slot]
            selected_id[0] = running_topk_ids[row_slot]
            buffer_length = cute.make_rmem_tensor(1, cutlass.Int32)
            buffer_length[0] = cutlass.Int32(0)
            for candidate in cutlass.range(8, unroll=2):
                local_vocab_id = vocab_start + lane + candidate * 32
                self._warp_select_add(
                    candidates[candidate],
                    global_vocab_start + local_vocab_id,
                    selected_value,
                    selected_id,
                    buffer_length,
                    sC,
                    sC_ids,
                    row_in_tile,
                )
            self._warp_select_done(
                selected_value,
                selected_id,
                buffer_length,
                sC,
                sC_ids,
                row_in_tile,
            )
            running_topk_values[row_slot] = selected_value[0]
            running_topk_ids[row_slot] = selected_id[0]

    @cute.jit
    def _update_stats_from_fragment(
        self,
        accumulator: cute.Tensor,
        thr_mma: cute.ThrMma,
        mA: cute.Tensor,
        mTargetIds: cute.Tensor,
        mTargetLogits: cute.Tensor,
        valid_vocab_size: cutlass.Int32,
        vocab_block: cutlass.Int32,
        block_m: cutlass.Int32,
        running_max: cute.Tensor,
        running_sum_exp: cute.Tensor,
        running_rank_count: cute.Tensor,
        sC: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx % 32
        warp_n = (tidx // 32) // self.atom_layout_mnk[0]
        coordinates = thr_mma.partition_C(
            cute.make_identity_tensor((self.tile_m, self.tile_n))
        )
        accumulator_mn = _reshape_acc_to_mn(accumulator)
        coordinates_mn = _reshape_acc_to_mn(coordinates)

        # Each four-lane MMA group first reduces the fragment columns it owns.
        for local_row_slot in cutlass.range_constexpr(
            cute.size(accumulator_mn, mode=[0])
        ):
            accumulator_row = accumulator_mn[local_row_slot, None]
            coordinate_row = coordinates_mn[local_row_slot, None]
            local_row = coordinate_row[0][0]
            row = block_m * self.tile_m + local_row
            target_id = cutlass.Int32(-1)
            target_logit = cutlass.Float32(0.0)
            if row < mA.shape[0]:
                target_id = mTargetIds[row]
                target_logit = mTargetLogits[row]

            row_max = -cutlass.Float32.inf
            rank_count = cutlass.Int32(0)
            for element in cutlass.range_constexpr(cute.size(accumulator_row)):
                local_vocab_id = vocab_block * self.tile_n + coordinate_row[element][1]
                if row < mA.shape[0] and local_vocab_id < valid_vocab_size:
                    logit = accumulator_row[element]
                    if target_id == local_vocab_id:
                        logit = target_logit
                    row_max = cute.arch.fmax(row_max, logit)
                    if logit >= target_logit:
                        rank_count += 1

            row_max = cute.arch.warp_reduction_max(
                row_max,
                threads_in_group=4,
            )
            rank_count = cute.arch.warp_reduction_sum(
                rank_count,
                threads_in_group=4,
            )
            row_sum_exp = cutlass.Float32(0.0)
            for element in cutlass.range_constexpr(cute.size(accumulator_row)):
                local_vocab_id = vocab_block * self.tile_n + coordinate_row[element][1]
                if row < mA.shape[0] and local_vocab_id < valid_vocab_size:
                    logit = accumulator_row[element]
                    if target_id == local_vocab_id:
                        logit = target_logit
                    row_sum_exp += cute.math.exp(
                        logit - row_max,
                        fastmath=True,
                    )
            row_sum_exp = cute.arch.warp_reduction_sum(
                row_sum_exp,
                threads_in_group=4,
            )
            if lane % 4 == 0 and row < mA.shape[0]:
                stats_column = warp_n * 3
                sC[local_row, stats_column] = row_max
                sC[local_row, stats_column + 1] = row_sum_exp
                sC[local_row, stats_column + 2] = cutlass.Float32(rank_count)

        cute.arch.sync_threads()
        # Merge per-N-warp statistics into one state for each row in this tile.
        threads_per_row = self.num_threads // self.tile_m
        if tidx < self.tile_m * threads_per_row:
            row_in_tile = tidx // threads_per_row
            lane_in_row = tidx % threads_per_row
            row = block_m * self.tile_m + row_in_tile
            if lane_in_row == 0 and row < mA.shape[0]:
                tile_max = -cutlass.Float32.inf
                tile_sum_exp = cutlass.Float32(0.0)
                tile_rank_count = cutlass.Int32(0)
                for n_warp in cutlass.range_constexpr(self.atom_layout_mnk[1]):
                    stats_column = n_warp * 3
                    partial_max = sC[row_in_tile, stats_column]
                    partial_sum_exp = sC[row_in_tile, stats_column + 1]
                    if partial_max != -cutlass.Float32.inf:
                        if tile_max == -cutlass.Float32.inf:
                            tile_max = partial_max
                            tile_sum_exp = partial_sum_exp
                        else:
                            merged_max = cute.arch.fmax(tile_max, partial_max)
                            tile_sum_exp = tile_sum_exp * cute.math.exp(
                                tile_max - merged_max,
                                fastmath=True,
                            ) + partial_sum_exp * cute.math.exp(
                                partial_max - merged_max,
                                fastmath=True,
                            )
                            tile_max = merged_max
                    tile_rank_count += cutlass.Int32(sC[row_in_tile, stats_column + 2])

                previous_max = running_max[0]
                merged_max = cute.arch.fmax(previous_max, tile_max)
                running_sum_exp[0] = running_sum_exp[0] * cute.math.exp(
                    previous_max - merged_max,
                    fastmath=True,
                ) + tile_sum_exp * cute.math.exp(
                    tile_max - merged_max,
                    fastmath=True,
                )
                running_max[0] = merged_max
                running_rank_count[0] += tile_rank_count

    def _make_smem_layout(self, dtype, smem_tiler):
        major_size = min(smem_tiler[1], 128 * 8 // dtype.width)
        swizzle_bits = min(int(math.log2(major_size * dtype.width // 128)), 3)
        base_bits = int(math.log2(128 // 8))
        shift_bits = int(math.log2(128 // dtype.width))
        swizzle = cute.make_swizzle(swizzle_bits, base_bits, shift_bits)
        atom = cute.make_layout((8, major_size), stride=(major_size, 1))
        return cute.tile_to_shape(atom, smem_tiler, (0, 1, 2)), swizzle

    def _make_gmem_tiled_copy(self, copy_atom, dtype):
        copy_elems = 128 // dtype.width
        threads_k = self.tile_k // copy_elems
        thread_layout = cute.make_layout(
            (self.num_threads // threads_k, threads_k),
            stride=(threads_k, 1),
        )
        value_layout = cute.make_layout((1, copy_elems))
        return cute.make_tiled_copy_tv(copy_atom, thread_layout, value_layout)


@cache
def _compile_lm_head_logprobs(
    hidden_size: int,
    local_vocab_size: int,
    num_vocab_blocks: int,
    device_index: int,
    num_topk: int,
    group_n: int,
):
    _validate_device_environment(device_index)
    # Keep M symbolic so one compiled specialization serves all prompt chunks.
    num_rows = cute.sym_int()
    num_partials = triton.cdiv(num_vocab_blocks, group_n)
    hidden = make_fake_tensor(
        cutlass.BFloat16,
        (num_rows, hidden_size),
        stride=(hidden_size, 1),
        assumed_align=16,
    )
    weight = make_fake_tensor(
        cutlass.BFloat16,
        (local_vocab_size, hidden_size),
        stride=(hidden_size, 1),
        assumed_align=16,
    )
    target_ids = make_fake_tensor(
        cutlass.Int32, (num_rows,), stride=(1,), assumed_align=4
    )
    target_logits = make_fake_tensor(
        cutlass.Float32, (num_rows,), stride=(1,), assumed_align=4
    )
    partial_max = make_fake_tensor(
        cutlass.Float32,
        (num_rows, num_partials),
        stride=(num_partials, 1),
        assumed_align=4,
    )
    partial_sum_exp = make_fake_tensor(
        cutlass.Float32,
        (num_rows, num_partials),
        stride=(num_partials, 1),
        assumed_align=4,
    )
    partial_rank_count = make_fake_tensor(
        cutlass.Int32,
        (num_rows, num_partials),
        stride=(num_partials, 1),
        assumed_align=4,
    )
    # Preserve one compiled call signature without constructing zero-width
    # fake top-K tensors for the statistics-only specialization.
    if num_topk == 0:
        partial_topk_values = partial_max
        partial_topk_ids = partial_rank_count
    else:
        partial_topk_values = make_fake_tensor(
            cutlass.Float32,
            (num_rows, num_partials, num_topk),
            stride=(num_partials * num_topk, num_topk, 1),
            assumed_align=4,
        )
        partial_topk_ids = make_fake_tensor(
            cutlass.Int32,
            (num_rows, num_partials, num_topk),
            stride=(num_partials * num_topk, num_topk, 1),
            assumed_align=4,
        )
    op = _LMHeadLogprobsCpAsync(
        _TILE_M,
        _TILE_N,
        _TILE_K,
        _NUM_STAGES,
        _ATOM_LAYOUT_M,
        _ATOM_LAYOUT_N,
        num_topk,
        group_n,
    )
    return cute.compile(
        op,
        hidden,
        weight,
        target_ids,
        target_logits,
        partial_max,
        partial_sum_exp,
        partial_rank_count,
        partial_topk_values,
        partial_topk_ids,
        cutlass.Int32(local_vocab_size),
        cutlass.Int32(0),
        cutlass.Int32(0),
        cutlass.Int32(num_vocab_blocks),
        make_fake_stream(),
        options="--enable-tvm-ffi",
    )
