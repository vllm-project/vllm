# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# RoutedExpertGemm is modified from LowLatencyBlockscaledGemmKernel in
# FlashInfer (flashinfer/gemm/kernels/cute_dsl/low_latency_blockscaled_gemm.py,
# flashinfer-python 0.7.1), which carries the notice below: its operand and
# scale-factor layouts, TMA partitioning, load warps, MMA warp and accumulator
# load. Modified by the vLLM contributors: the kernel is persistent over the
# token tiles of a routing record, A and its scales are loaded at each tile's
# expert, the accumulator is double-buffered in tensor memory, and the
# epilogue is each GEMM's own.

# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Routed-expert GEMMs of the DeepSeek-V4.1 low-batch MoE in CuTe DSL.

The low-batch MoE (``routed_expert_gemm.py``) multiplies at most 192 token
rows by the weights of the experts they chose. Its routing record
(``routed_expert_route_cute.py``) lists, for each 8-row token tile, the
tile's expert and its row limit (the end of the expert's rows within the
tile). A GEMM here computes 128 x 8 output tiles from MXFP4 weights (operand
A, 128 rows of one expert) and MXFP8 activations (operand B, 8 rows) with
``tcgen05`` block-scaled MMAs, as FlashInfer's low-latency block-scaled GEMM
does, over the work items (weight row tile, token tile) of the tiles the
record uses:

* the kernel is persistent: CTA ``b`` of ``G`` takes items ``b, b + G, ..``,
  so the weight loads stream without a gap from one item to the next, and no
  CTA works on a token tile the call does not use;
* the warp that loads A loads it at the item's expert ``record[4 + c]``;
* the MMAs run in ascending K order without split-K, the order of the
  TRT-LLM-gen ``t128x8`` tactics, so every output is bitwise theirs;
* the accumulator has two tensor-memory stages, so the epilogue of one item
  overlaps the MMAs of the next.

Warps: 0 loads A and its scales, 1 loads B and its scales, 2 issues the MMAs,
4-7 run the epilogue (warp 3 is idle).

Programmatic dependent launch. GEMM1 needs the record to pick its experts, so
all its warps wait for the kernel before it first; it then triggers its
dependents at once. GEMM2 never waits for the GEMM1 grid. It reads the record
without waiting (the record comes from the kernel before GEMM1, and GEMM1
triggers only after its own wait for that kernel). Its CTAs take the SMs that
GEMM1's CTAs leave (one CTA of either fills an SM's shared memory) and stream
their weights at once. Its B load warp waits for each item's token tile
alone: GEMM1 counts a tile's finished items in the record's ready words
(``record[ready + c]``, zeroed with the record). After an item's epilogue
stores, GEMM1's four epilogue warps meet at a barrier and one thread adds 1
with release semantics at GPU scope. GEMM2's B load warp spins with acquire
loads until tile ``c`` counts all of GEMM1's m tiles, then fences the async
proxy before its TMA loads of H and HS. So GEMM2 works on the tiles of
GEMM1's first waves while GEMM1's last, partial wave runs. A record serves
one GEMM1 and GEMM2 pair: the ready words are zeroed only when the record is
built. GEMM2 completes only after every GEMM1 item's epilogue, after which
GEMM1 reads and writes no global memory (its CTAs read the record's tile
count before triggering, and a release fence precedes the trigger); so a
kernel that waits for GEMM2 also sees GEMM1's work done.

The weights are those vLLM prepares for the TRT-LLM-gen backend, whose rows are
shuffled for 128-row epilogue tiles: in each block of 32, shuffled row
``s0 + 8 * s1`` (``s0 < 8``, ``s1 < 4``) holds row ``4 * s0 + s1``. Their
scales are in the 128x4 layout the kernel reads for A.

GEMM1 (:class:`RoutedExpertGemm1`) gathers each tile's tokens from the
routed input (``record[4 + 2C + 8c + r]``, the token of tile row ``r``) and
multiplies them by the tile's expert: clamped SwiGLU of the interleaved
up/gate rows, quantized to MXFP8 (H and its block scales). The epilogue's
arithmetic follows the GEMM1 kernel from Cake (vllm-project/vllm#59725).

GEMM2 (:class:`RoutedExpertGemm2`) reads token tile ``c`` of the intermediate
H, the contiguous rows ``8c .. 8c + 7``, and writes BF16 rows of the output in
the same permuted order. H's block scales use the layout the kernel reads for
B: one 128-row scale atom per 8-row tile, of which rows 0-7 are used
(``HS[c * K + (k // 128) * 512 + r * 16 + (k // 32) % 4]`` for the scale of
row ``8c + r`` at column ``k``, ``K`` the bytes of one tile's atoms).
"""

from __future__ import annotations

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass import Float32, Int32, Int64, Uint8, Uint16, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineAsync,
    PipelineUmmaAsync,
    PipelineUserType,
    make_pipeline_state,
    pipeline_init_arrive,
    pipeline_init_wait,
)

TILE = 8  # token rows per tile
TMEM_COLUMNS = 128
# L2 eviction priority of the streamed weights: evict first.
EVICT_FIRST = 0x12F0000000000000


def tile_scale_bytes(k: int) -> int:
    """Bytes of one 8-row tile's block scales over ``k`` columns in the
    per-tile layout B's scales use (one 512-byte atom per 128 columns)."""
    return k // 128 * 512


class RoutedExpertGemm:
    """Persistent block-scaled GEMM over the work items of a routing record
    (see the module docstring). A subclass builds the operands in
    ``__call__`` and supplies the epilogue (``store``)."""

    # GEMM1 reads the record only after its grid dependency wait.
    route_before_wait = True
    # GEMM1's B load warp gathers its rows (``gather_b_warp``) instead of
    # loading B tiles with TMA.
    gather_b = False
    # GEMM1 counts each token tile's finished items in the record's ready
    # words and triggers its dependents right after its wait; GEMM2 waits,
    # item by item, for its tile's count to reach ``ready_target`` (GEMM1's
    # m tiles) instead of for the GEMM1 grid (module docstring).
    signal_ready = False
    ready_target = 0
    # Operand types, which a subclass's ``__call__`` sets.
    a_dtype: type[cutlass.Numeric]
    b_dtype: type[cutlass.Numeric]
    sf_dtype: type[cutlass.Numeric]

    def __init__(
        self, m: int, m_tiles: int, k: int, cta_k: int, num_ab_stage: int
    ) -> None:
        self.m, self.m_tiles, self.k = m, m_tiles, k
        self.acc_dtype = cutlass.Float32
        self.cta_m, self.cta_n, self.cta_k = 128, TILE, cta_k
        self.k_tiles = k // cta_k
        self.num_ab_stage = num_ab_stage
        self.num_acc_stage = 2
        self.sf_vec_size = 32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        self.threads_per_cta = 256
        self.epilogue_warp = 4

    def _setup_attributes(self) -> None:
        """Derived configuration (FlashInfer's, with staged accumulators)."""
        tiled_mma = self._tiled_mma((self.cta_m, self.cta_n))
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = self.cta_k // mma_inst_shape_k
        self.num_acc_tmem_cols = self.cta_n * self.num_acc_stage
        self.num_sfa_tmem_cols = (self.cta_m // 32) * self.mma_inst_tile_k
        num_sfb_tmem_cols = self.mma_inst_tile_k * 4
        assert (
            self.num_acc_tmem_cols + self.num_sfa_tmem_cols + num_sfb_tmem_cols
            <= TMEM_COLUMNS
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        tile_mnk = (self.cta_m, self.cta_n, self.cta_k)
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, tile_mnk, self.smem_alloc_a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, tile_mnk, self.smem_alloc_b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, tile_mnk, self.sf_vec_size, self.num_ab_stage
        )
        # B's scales: one 128-row atom per 8-row tile.
        self.mma_tiler_sfb = (self.cta_m, 128, self.cta_k)
        self.tiled_mma_sfb = self._tiled_mma((self.cta_m, 128))
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            self.tiled_mma_sfb, self.mma_tiler_sfb, self.sf_vec_size, self.num_ab_stage
        )

    def _tiled_mma(self, mn: tuple) -> cute.TiledMma:
        """The block-scaled MMA of an (M, N) tile."""
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            mn,
        )

    @cute.jit
    def _launch(
        self,
        a: cute.Tensor,
        sfa: cute.Tensor,
        b: cute.Tensor,
        sfb: Optional[cute.Tensor],  # noqa: UP045 (the DSL's annotations)
        route: cute.Tensor,
        epi: tuple,
        b_side: tuple,
        cta_words: Int32,
        ready: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """Build the TMA atoms and launch ``ctas`` persistent CTAs. ``sfb``
        None: the B load warp writes B's scales itself (``b_side``).
        ``ready``: the record's first ready word."""
        self.mxf8f6f4 = self.a_dtype.width != self.b_dtype.width
        self.smem_alloc_a_dtype = (
            cutlass.Int8 if self.mxf8f6f4 and self.a_dtype.width < 8 else self.a_dtype
        )
        self.smem_alloc_b_dtype = (
            cutlass.Int8 if self.mxf8f6f4 and self.b_dtype.width < 8 else self.b_dtype
        )
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self._setup_attributes()
        tiled_mma = self._tiled_mma((self.cta_m, self.cta_n))
        tile_mnk = (self.cta_m, self.cta_n, self.cta_k)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            tile_mnk,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_alloc_a_dtype
            if self.mxf8f6f4 and self.a_dtype.width < 8
            else None,
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            tile_mnk,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_alloc_b_dtype
            if self.mxf8f6f4 and self.b_dtype.width < 8
            else None,
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            sfa,
            sfa_smem_layout,
            tile_mnk,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        sfb_copy_size = 0
        tma_atom_sfb = None
        tma_tensor_sfb = None
        if cutlass.const_expr(sfb is not None):
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sm100_utils.cluster_shape_to_tma_atom_B(
                    self.cluster_shape_mn, self.tiled_mma_sfb.thr_id
                ),
                sfb,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                self.tiled_mma_sfb,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
            sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        # Count input bytes before unpacking mixed-width operands.
        self.tma_bytes_a = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        self.tma_bytes_b = (
            cute.size_in_bytes(self.b_dtype, b_smem_layout) + sfb_copy_size
        )
        self.kernel(
            tiled_mma,
            self.tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfb,
            tma_tensor_sfb,
            route,
            epi,
            b_side,
            cta_words,
            ready,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
        ).launch(
            grid=(ctas, 1, 1),
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfb: Optional[cute.CopyAtom],  # noqa: UP045
        mSFB_nkl: Optional[cute.Tensor],  # noqa: UP045
        mRoute: cute.Tensor,
        epi: tuple,
        b_side: tuple,
        cta_words: Int32,
        ready: Int32,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
    ):
        """The persistent warp-specialized GEMM (module docstring)."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_b)
            if cutlass.const_expr(tma_atom_sfb is not None):
                cpasync.prefetch_descriptor(tma_atom_sfb)

        mma_tile_coord_v = Int32(0)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        AB_Stage = self.num_ab_stage
        Acc_Stage = self.num_acc_stage

        @cute.struct
        class SharedStorage:
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, AB_Stage * 2]
            acc_pipeline_bars: cute.struct.MemRange[cutlass.Int64, Acc_Stage * 2]
            tmem_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            # Exchange between the epilogue warps (GEMM1's block amax), one
            # buffer per item parity.
            epi_exchange: cute.struct.MemRange[cutlass.Float32, 2 * 4 * TILE]

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        # The A and B load warps each arrive once per stage; the MMA warp
        # releases a stage with one commit.
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(),
            num_stages=AB_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 2),
            consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True,
        )
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage
        acc_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.acc_pipeline_bars.data_ptr(),
            num_stages=Acc_Stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        tmem_bar = storage.tmem_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        if warp_idx == 0:
            with cute.arch.elect_one():
                # The MMA warp allocates; the epilogue reads, then both
                # arrive again before the MMA warp frees it.
                cute.arch.mbarrier_init(tmem_bar, 32 + 128)
        pipeline_init_arrive(cluster_shape_mn=None, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=None)

        sA = smem.allocate_tensor(
            element_type=self.smem_alloc_a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.smem_alloc_b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        # This CTA's work items: b, b + G, .. below tiles * m_tiles.
        if cutlass.const_expr(not self.route_before_wait):
            cute.arch.griddepcontrol_wait()
        total = mRoute[1] * self.m_tiles
        count = Int32(0)
        if bid < total:
            count = (total - 1 - bid) // grid + 1
        if cutlass.const_expr(self.signal_ready):
            # Every warp has read the tile count before warp 0 triggers
            # GEMM2, which may finish before a CTA without items exits.
            cute.arch.barrier()

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        if warp_idx == 0:
            # A and its scales, at each item's expert.
            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            gA_mkl = cute.local_tile(
                mA_mkl, (self.cta_m, self.cta_k), (None, None, None)
            )
            tAsA, tAgA = cpasync.tma_partition(
                tma_atom_a,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gA_mkl), 0, 3),
            )
            gSFA_mkl = cute.local_tile(
                mSFA_mkl, (self.cta_m, self.cta_k), (None, None, None)
            )
            tAsSFA, tAgSFA = cpasync.tma_partition(
                tma_atom_sfa,
                block_in_cluster_coord_vmnk[2],
                a_cta_layout,
                cute.group_modes(sSFA, 0, 3),
                cute.group_modes(thr_mma.partition_A(gSFA_mkl), 0, 3),
            )
            tAsSFA = cute.filter_zeros(tAsSFA)
            tAgSFA = cute.filter_zeros(tAgSFA)
            if cutlass.const_expr(self.signal_ready):
                # A release fence first, so that GEMM2's start synchronizes
                # with the record (and its zeroed ready words) GEMM1 waited for.
                cute.arch.fence_acq_rel_gpu()
                cute.arch.griddepcontrol_launch_dependents()
            policy = Int64(EVICT_FIRST)
            state = make_pipeline_state(PipelineUserType.Producer, AB_Stage)
            for i in cutlass.range(count, unroll=1):
                item = bid + i * grid
                n_tile = item // self.m_tiles
                m_tile = item - n_tile * self.m_tiles
                expert = mRoute[4 + n_tile]
                tAgA_item = tAgA[(None, m_tile, None, expert)]
                tAgSFA_item = tAgSFA[(None, m_tile, None, expert)]
                for k_tile in cutlass.range(self.k_tiles, unroll=1):
                    cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            ab_full_bar + state.index, self.tma_bytes_a
                        )
                    cute.copy(
                        tma_atom_a,
                        tAgA_item[(None, k_tile)],
                        tAsA[(None, state.index)],
                        tma_bar_ptr=ab_full_bar + state.index,
                        cache_policy=policy,
                    )
                    cute.copy(
                        tma_atom_sfa,
                        tAgSFA_item[(None, k_tile)],
                        tAsSFA[(None, state.index)],
                        tma_bar_ptr=ab_full_bar + state.index,
                        cache_policy=policy,
                    )
                    state.advance()
            if cutlass.const_expr(not self.signal_ready):
                cute.arch.griddepcontrol_launch_dependents()
            # Keep shared memory alive until the MMA warp releases every stage.
            for _ in cutlass.range(AB_Stage, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                state.advance()
        elif warp_idx == 1:
            if cutlass.const_expr(self.route_before_wait and self.ready_target == 0):
                cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(self.gather_b):
                # B and its scales, gathered from the previous kernel's rows.
                self.gather_b_warp(
                    sB,
                    sSFB,
                    ab_full_bar,
                    ab_empty_bar,
                    mRoute,
                    b_side,
                    cta_words,
                    count,
                )
            else:
                # B and its scales, from the previous kernel.
                self.load_b_warp(
                    tma_atom_b,
                    mB_nkl,
                    tma_atom_sfb,
                    mSFB_nkl,
                    tiled_mma_sfb,
                    thr_mma,
                    cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk,
                    sB,
                    sSFB,
                    ab_full_bar,
                    ab_empty_bar,
                    mRoute,
                    ready,
                    count,
                )
        elif warp_idx == 2:
            self.mma_warp(
                ab_full_bar,
                ab_empty_bar,
                acc_pipeline,
                tmem_bar,
                tmem_base_smem_ptr,
                tiled_mma,
                sA,
                sB,
                sSFA,
                sSFB,
                sfa_smem_layout_staged,
                sfb_smem_layout_staged,
                count,
            )
        elif warp_idx >= self.epilogue_warp:
            epi_tid = tidx - 32 * self.epilogue_warp
            cute.arch.mbarrier_arrive(tmem_bar)
            cute.arch.mbarrier_wait(tmem_bar, 0)
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                self.acc_dtype, 16, tmem_base_smem_ptr
            )
            tCtAcc_base = cute.make_tensor(tmem_ptr, self._acc_layout(tiled_mma))
            # The TMEM load FlashInfer's epilogue uses for M-major BF16 C
            # (16x256b): each thread holds rows s, s + 8, s + 16, s + 24 of
            # its warp's 32 (s = lane / 4), two columns each.
            copy_atom_t2r = sm100_utils.get_tmem_load_op(
                (self.cta_m, self.cta_n, self.cta_k),
                cutlass.utils.LayoutEnum.COL_MAJOR,
                cutlass.BFloat16,
                self.acc_dtype,
                (self.cta_m, self.cta_n),
                False,
            )
            tiled_copy_t2r = tcgen05.make_tmem_copy(
                copy_atom_t2r, tCtAcc_base[((None, None), 0, 0, 0)]
            )
            thr_copy_t2r = tiled_copy_t2r.get_slice(epi_tid)
            tDtAcc_base = thr_copy_t2r.partition_S(
                tCtAcc_base[((None, None), 0, 0, None)]
            )
            coord_c = cute.make_identity_tensor(
                (self.m, cute.size(mB_nkl, mode=[0]), 1)
            )
            tCcC = thr_mma.partition_C(
                cute.local_tile(coord_c, (self.cta_m, self.cta_n), (None, None, None))
            )
            tDcC_all = thr_copy_t2r.partition_D(
                tCcC[((None, None), 0, 0, None, None, None)]
            )
            tDrAcc = cute.make_rmem_tensor(
                tDcC_all[(None, None, None, 0, 0, 0)].shape, self.acc_dtype
            )
            exchange = cute.make_tensor(
                storage.epi_exchange.data_ptr(), cute.make_layout(2 * 4 * TILE)
            )
            state = make_pipeline_state(PipelineUserType.Consumer, Acc_Stage)
            for i in cutlass.range(count, unroll=1):
                item = bid + i * grid
                n_tile = item // self.m_tiles
                m_tile = item - n_tile * self.m_tiles
                acc_pipeline.consumer_wait(state)
                cute.copy(
                    tiled_copy_t2r,
                    tDtAcc_base[(None, None, None, state.index)],
                    tDrAcc,
                )
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(state)
                state.advance()
                tDcC = tDcC_all[
                    (None, None, None, m_tile, self.b_tile_index(n_tile), 0)
                ]
                self.store(
                    tDrAcc, tDcC, epi, mRoute, cta_words, n_tile, m_tile, i, exchange
                )
                if cutlass.const_expr(self.signal_ready):
                    # Every epilogue warp's stores of the item, then one
                    # release of its tile's count (module docstring).
                    cute.arch.barrier(barrier_id=2, number_of_threads=128)
                    if epi_tid == 0:
                        cute.arch.fence_proxy("async.global")
                        red_add_release((mRoute.iterator + ready + n_tile).toint())
            cute.arch.mbarrier_arrive(tmem_bar)
            cute.arch.mbarrier_wait(tmem_bar, 1)
        cute.arch.barrier()

    def _acc_layout(self, tiled_mma: cute.TiledMma) -> cute.Layout:
        """Tensor-memory layout of the staged accumulators."""
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        return tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        ).layout

    @cute.jit
    def mma_warp(
        self,
        ab_full_bar,
        ab_empty_bar,
        acc_pipeline,
        tmem_bar,
        tmem_base_smem_ptr,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        count: Int32,
    ):
        """Allocate tensor memory; for every item, wait for an accumulator
        stage and issue the block-scaled MMAs of its K tiles (scales copied
        from shared to tensor memory first); free the tensor memory."""
        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        cute.arch.alloc_tmem(TMEM_COLUMNS, tmem_base_smem_ptr)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.relinquish_tmem_alloc_permit()
        tmem_ptr = cute.arch.retrieve_tmem_ptr(self.acc_dtype, 16, tmem_base_smem_ptr)
        tCtAcc_base = cute.make_tensor(tmem_ptr, self._acc_layout(tiled_mma))

        sfa_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self.num_acc_tmem_cols, dtype=self.sf_dtype
        )
        tCtSFA = cute.make_tensor(
            sfa_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                tiled_mma.shape_mnk,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            ),
        )
        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self.num_acc_tmem_cols + self.num_sfa_tmem_cols,
            dtype=self.sf_dtype,
        )
        tCtSFB = cute.make_tensor(
            sfb_tmem_ptr,
            blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                tiled_mma.shape_mnk,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            ),
        )
        copy_sfa, tCsSFA_s2t, tCtSFA_s2t = self._s2t_copy(sSFA, tCtSFA)
        copy_sfb, tCsSFB_s2t, tCtSFB_s2t = self._s2t_copy(sSFB, tCtSFB)

        ab_state = make_pipeline_state(PipelineUserType.Consumer, self.num_ab_stage)
        acc_state = make_pipeline_state(PipelineUserType.Producer, self.num_acc_stage)
        for _ in cutlass.range(count, unroll=1):
            acc_pipeline.producer_acquire(acc_state)
            tCtAcc = tCtAcc_base[(None, None, None, acc_state.index)]
            for k_tile in cutlass.range(self.k_tiles, unroll=1):
                cute.arch.mbarrier_wait(ab_full_bar + ab_state.index, ab_state.phase)
                stage = (None, None, None, None, ab_state.index)
                cute.copy(copy_sfa, tCsSFA_s2t[stage], tCtSFA_s2t)
                cute.copy(copy_sfb, tCsSFB_s2t[stage], tCtSFB_s2t)
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                operand = (None, None, None, ab_state.index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    [tCrA[operand], tCtSFA],
                    [tCrB[operand], tCtSFB],
                    tCtAcc,
                )
                with cute.arch.elect_one():
                    tcgen05.commit(ab_empty_bar + ab_state.index, None, self.cta_group)
                ab_state.advance()
            acc_pipeline.producer_commit(acc_state)
            acc_state.advance()

        # Free the tensor memory once the epilogue has read it. Every thread
        # waits for each phase it arrived on before arriving again.
        cute.arch.mbarrier_wait(tmem_bar, 0)
        cute.arch.mbarrier_arrive(tmem_bar)
        cute.arch.mbarrier_wait(tmem_bar, 1)
        cute.arch.dealloc_tmem(tmem_ptr, TMEM_COLUMNS)

    def _s2t_copy(self, sSF: cute.Tensor, tSF: cute.Tensor):
        """Shared-to-tensor-memory copy of scale factors, partitioned."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
        )
        tiled_copy = tcgen05.make_s2t_copy(copy_atom, tCtSF_compact)
        thr_copy = tiled_copy.get_slice(0)
        tCsSF = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy, thr_copy.partition_S(tCsSF_compact)
        )
        return tiled_copy, tCsSF, thr_copy.partition_D(tCtSF_compact)

    @cute.jit
    def b_tile_index(self, n_tile: Int32) -> Int32:
        """B's 8-row tile for token tile ``n_tile``."""
        return n_tile

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
        """Write an item's accumulator tile (the subclass's epilogue)."""
        raise NotImplementedError

    @cute.jit
    def load_b_warp(
        self,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tiled_mma_sfb: cute.TiledMma,
        thr_mma: cute.ThrMma,
        cluster_layout_vmnk: cute.Layout,
        block_in_cluster_coord_vmnk: tuple,
        sB: cute.Tensor,
        sSFB: cute.Tensor,
        ab_full_bar: cute.Pointer,
        ab_empty_bar: cute.Pointer,
        mRoute: cute.Tensor,
        ready: Int32,
        count: Int32,
    ):
        """The B load warp: TMA of each item's B tile and its scales, each
        item's once its token tile's ready count reaches ``ready_target``
        (when set)."""
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        b_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
        )
        gB_nkl = cute.local_tile(mB_nkl, (self.cta_n, self.cta_k), (None, None, None))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sB, 0, 3),
            cute.group_modes(thr_mma.partition_B(gB_nkl), 0, 3),
        )
        gSFB_nkl = cute.local_tile(mSFB_nkl, (128, self.cta_k), (None, None, None))
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            block_in_cluster_coord_vmnk[1],
            b_cta_layout,
            cute.group_modes(sSFB, 0, 3),
            cute.group_modes(
                tiled_mma_sfb.get_slice(Int32(0)).partition_B(gSFB_nkl), 0, 3
            ),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)
        state = make_pipeline_state(PipelineUserType.Producer, self.num_ab_stage)
        for i in cutlass.range(count, unroll=1):
            item = bid + i * grid
            n_tile = item // self.m_tiles
            if cutlass.const_expr(self.ready_target > 0):
                wait_ready(
                    (mRoute.iterator + ready + n_tile).toint(), self.ready_target
                )
            b_tile = self.b_tile_index(n_tile)
            tBgB_item = tBgB[(None, b_tile, None, 0)]
            tBgSFB_item = tBgSFB[(None, b_tile, None, 0)]
            for k_tile in cutlass.range(self.k_tiles, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        ab_full_bar + state.index, self.tma_bytes_b
                    )
                cute.copy(
                    tma_atom_b,
                    tBgB_item[(None, k_tile)],
                    tBsB[(None, state.index)],
                    tma_bar_ptr=ab_full_bar + state.index,
                )
                cute.copy(
                    tma_atom_sfb,
                    tBgSFB_item[(None, k_tile)],
                    tBsSFB[(None, state.index)],
                    tma_bar_ptr=ab_full_bar + state.index,
                )
                state.advance()
        for _ in cutlass.range(self.num_ab_stage, unroll=1):
            cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
            state.advance()

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
        """The B load warp of a subclass that gathers B's rows itself
        (``gather_b``)."""
        raise NotImplementedError


@cute.jit
def unshuffled_row(row: Int32) -> Int32:
    """The weight row that shuffled row ``row`` holds (module docstring)."""
    return (row & Int32(-32)) | ((row & Int32(7)) << Int32(2)) | ((row >> 3) & 3)


class RoutedExpertGemm2(RoutedExpertGemm):
    """GEMM2: the BF16 rows ``h @ w2[expert]^T`` of each token tile, in the
    tile's permuted order."""

    def __init__(
        self,
        hidden: int,
        intermediate: int,
        gemm1_m_tiles: int,
        cta_k: int = 128,
        num_ab_stage: int = 10,
    ) -> None:
        """``gemm1_m_tiles``: GEMM1's items per token tile, the ready count
        at which a tile's H is complete."""
        super().__init__(hidden, hidden // 128, intermediate, cta_k, num_ab_stage)
        self.ready_target = gemm1_m_tiles

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w2 (hidden, intermediate, experts), K-major; B: H (rows,
        intermediate), K-major, ``rows`` the padded rows; C: (rows, hidden)
        BF16; ``ctas`` persistent CTAs."""
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
        # The ready words follow the token table (one row per padded row).
        ready = 4 + 2 * cta_words + rows
        route = cute.make_tensor(route_ptr, cute.make_layout(ready + cta_words))
        self._launch(a, sfa, b, sfb, route, (c,), (), cta_words, ready, ctas, stream)

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
        """Round to BF16 (nearest even) and store the tile's rows below its
        limit at their unshuffled hidden index."""
        (mC,) = epi
        row_limit = mRoute[4 + cta_words + n_tile]
        tDrC = cute.make_rmem_tensor(tDrAcc.shape, cutlass.BFloat16)
        tDrC.store(tDrAcc.load().to(cutlass.BFloat16))
        for i in cutlass.range_constexpr(cute.size(tDrC)):
            coord = tDcC[i]
            if coord[1] < row_limit:
                mC[unshuffled_row(coord[0]), coord[1]] = tDrC[i]


# GEMM1's epilogue arithmetic, spelled out instruction by instruction so that H,
# its scales and GEMM2's rows are bitwise those of the TRT-LLM-gen MoE.
_SWIGLU_X2 = """{
.reg .pred sw_p;
.reg .f32 sw_u0, sw_u1, sw_g0, sw_g1, sw_x0, sw_x1;
.reg .b32 sw_r0, sw_r1;
.reg .b64 sw_up, sw_gate, sw_one, sw_zero, sw_l2e, sw_neg, sw_f, sw_a, sw_d;
setp.lt.ftz.f32 sw_p, $6, $2;
selp.f32 sw_u0, $6, $2, sw_p;
setp.lt.ftz.f32 sw_p, $2, $7;
selp.f32 sw_u0, $7, sw_u0, sw_p;
setp.lt.ftz.f32 sw_p, $6, $3;
selp.f32 sw_u1, $6, $3, sw_p;
setp.lt.ftz.f32 sw_p, $3, $7;
selp.f32 sw_u1, $7, sw_u1, sw_p;
setp.lt.ftz.f32 sw_p, $6, $4;
selp.f32 sw_g0, $6, $4, sw_p;
setp.lt.ftz.f32 sw_p, $4, 0fFF7FFFFF;
selp.f32 sw_g0, 0fFF7FFFFF, sw_g0, sw_p;
setp.lt.ftz.f32 sw_p, $6, $5;
selp.f32 sw_g1, $6, $5, sw_p;
setp.lt.ftz.f32 sw_p, $5, 0fFF7FFFFF;
selp.f32 sw_g1, 0fFF7FFFFF, sw_g1, sw_p;
mov.b32 sw_r0, sw_u0;
mov.b32 sw_r1, sw_u1;
mov.b64 sw_up, {sw_r0, sw_r1};
mov.b32 sw_r0, sw_g0;
mov.b32 sw_r1, sw_g1;
mov.b64 sw_gate, {sw_r0, sw_r1};
mov.b64 sw_one, 4575657222473777152;
mov.b64 sw_zero, 0;
mov.b64 sw_l2e, 4591606991553079867;
mov.b64 sw_neg, -4647714812233515008;
fma.rn.f32x2 sw_f, sw_up, sw_one, sw_zero;
mul.rn.f32x2 sw_a, sw_gate, sw_l2e;
mul.rn.f32x2 sw_a, sw_neg, sw_a;
mov.b64 {sw_r0, sw_r1}, sw_a;
mov.b32 sw_x0, sw_r0;
mov.b32 sw_x1, sw_r1;
ex2.approx.ftz.f32 sw_x0, sw_x0;
ex2.approx.ftz.f32 sw_x1, sw_x1;
mov.b32 sw_r0, sw_x0;
mov.b32 sw_r1, sw_x1;
mov.b64 sw_d, {sw_r0, sw_r1};
add.f32x2 sw_d, sw_one, sw_d;
mov.b64 {sw_r0, sw_r1}, sw_d;
mov.b32 sw_x0, sw_r0;
mov.b32 sw_x1, sw_r1;
rcp.approx.ftz.f32 sw_x0, sw_x0;
rcp.approx.ftz.f32 sw_x1, sw_x1;
mov.b32 sw_r0, sw_x0;
mov.b32 sw_r1, sw_x1;
mov.b64 sw_d, {sw_r0, sw_r1};
mul.rn.f32x2 sw_d, sw_gate, sw_d;
mul.rn.f32x2 sw_d, sw_f, sw_d;
mov.b64 {sw_r0, sw_r1}, sw_d;
mov.b32 $0, sw_r0;
mov.b32 $1, sw_r1;
}"""

# Two blocks' amax (this warp's half, the partner warp's half) to the bits of
# amax / 448, an all-zero block counting as amax 448.
_SCALES_X2 = """{
.reg .pred sc_p;
.reg .f32 sc_a, sc_b;
.reg .b32 sc_r0, sc_r1;
.reg .b64 sc_x, sc_c;
max.ftz.f32 sc_a, $2, $3;
setp.eq.ftz.f32 sc_p, sc_a, 0f00000000;
selp.f32 sc_a, 0f43E00000, sc_a, sc_p;
max.ftz.f32 sc_b, $4, $5;
setp.eq.ftz.f32 sc_p, sc_b, 0f00000000;
selp.f32 sc_b, 0f43E00000, sc_b, sc_p;
mov.b32 sc_r0, sc_a;
mov.b32 sc_r1, sc_b;
mov.b64 sc_x, {sc_r0, sc_r1};
mov.b64 sc_c, 4256545022072211749;
mul.rn.f32x2 sc_x, sc_x, sc_c;
mov.b64 {$0, $1}, sc_x;
}"""


def _asm(result, operands: list, text: str, constraints: str, loc, ip):
    """A side-effect-free PTX inline assembly operation."""
    return llvm.inline_asm(
        result,
        operands,
        text,
        constraints,
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_16(dst: Int32, src: Int64, *, loc=None, ip=None) -> None:
    """``cp.async`` of 16 bytes from global address ``src`` to shared
    address ``dst`` (cached in L2 only)."""
    llvm.inline_asm(
        None,
        [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_4(dst: Int32, src: Int64, *, loc=None, ip=None) -> None:
    """``cp.async`` of 4 bytes from global address ``src`` to shared address
    ``dst``."""
    llvm.inline_asm(
        None,
        [Int32(dst).ir_value(loc=loc, ip=ip), Int64(src).ir_value(loc=loc, ip=ip)],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_release(addr: Int64, *, loc=None, ip=None) -> None:
    """Add 1 to the 32-bit word at global address ``addr`` with release
    semantics at GPU scope."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip)],
        "red.release.gpu.global.add.u32 [$0], 1;",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def load_acquire(addr: Int64, *, loc=None, ip=None) -> Int32:
    """The 32-bit word at global address ``addr``, loaded with acquire
    semantics at GPU scope (never hoisted out of a polling loop)."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.acquire.gpu.global.b32 $0, [$1];",
            "=r,l",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def nanosleep(ns: int, *, loc=None, ip=None) -> None:
    """Suspend the calling thread for about ``ns`` nanoseconds."""
    llvm.inline_asm(
        None,
        [],
        f"nanosleep.u32 {ns};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@cute.jit
def wait_ready(addr: Int64, target: cutlass.Constexpr[int]):
    """Spin until the ready count at ``addr`` reaches ``target`` (acquire
    loads), then fence the async proxy, so that the caller's later TMA loads
    see what the count's releasing writers stored (module docstring)."""
    seen = load_acquire(addr)
    while seen < target:
        nanosleep(100)
        seen = load_acquire(addr)
    cute.arch.fence_proxy("async.global")


@cute.jit
def publish_stage(full_bar: cute.Pointer):
    """Publish the calling warp's completed shared-memory writes of a stage to
    the MMA warp: fence them to the async proxy, through which the MMA reads
    shared memory, then one lane arrives on the stage's full barrier."""
    cute.arch.fence_proxy("async.shared", space="cta")
    cute.arch.sync_warp()
    with cute.arch.elect_one():
        cute.arch.mbarrier_arrive(full_bar)


@dsl_user_op
def clamped_swiglu_x2(up0, up1, gate0, gate1, limit, neg_limit, *, loc=None, ip=None):
    """``h`` of two tokens: ``up`` clamped to [-limit, limit] and ``gate`` to
    at most ``limit``, then ``up * gate * sigmoid(gate)``."""
    result = _asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [
            Float32(v).ir_value(loc=loc, ip=ip)
            for v in (up0, up1, gate0, gate1, limit, neg_limit)
        ],
        _SWIGLU_X2,
        "=f,=f,f,f,f,f,f,f",
        loc,
        ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [index], loc=loc, ip=ip))
        for index in range(2)
    )


@dsl_user_op
def abs_max_x2(lo, hi, *, loc=None, ip=None) -> Float32:
    """``max(|lo|, |hi|, 0)``, subnormals flushed."""
    return Float32(
        _asm(
            T.f32(),
            [Float32(v).ir_value(loc=loc, ip=ip) for v in (lo, hi)],
            "{\n.reg .f32 am_a, am_b;\nabs.ftz.f32 am_a, $1;\nabs.ftz.f32 am_b, $2;\n"
            "max.ftz.f32 am_a, am_a, am_b;\nabs.ftz.f32 am_a, am_a;\n"
            "max.ftz.f32 $0, am_a, 0f00000000;\n}",
            "=f,f,f",
            loc,
            ip,
        )
    )


@dsl_user_op
def scale_bits_x2(mine0, partner0, mine1, partner1, *, loc=None, ip=None):
    """The ``_SCALES_X2`` bits of two blocks."""
    result = _asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Float32(v).ir_value(loc=loc, ip=ip)
            for v in (mine0, partner0, mine1, partner1)
        ],
        _SCALES_X2,
        "=r,=r,f,f,f,f",
        loc,
        ip,
    )
    return tuple(
        Uint32(llvm.extractvalue(T.i32(), result, [index], loc=loc, ip=ip))
        for index in range(2)
    )


@dsl_user_op
def quantize_e4m3x2(lo, hi, inverse_bits, *, loc=None, ip=None) -> Uint16:
    """``lo`` and ``hi`` times the inverse scale (subnormals flushed) as two
    saturating E4M3 bytes, ``lo`` first."""
    return Uint16(
        _asm(
            T.i16(),
            [
                Float32(lo).ir_value(loc=loc, ip=ip),
                Float32(hi).ir_value(loc=loc, ip=ip),
                Uint32(inverse_bits).ir_value(loc=loc, ip=ip),
            ],
            "{\n.reg .f32 qz_i, qz_a, qz_b;\nmov.b32 qz_i, $3;\n"
            "mul.ftz.f32 qz_a, $1, qz_i;\nmul.ftz.f32 qz_b, $2, qz_i;\n"
            "cvt.rn.satfinite.e4m3x2.f32 $0, qz_b, qz_a;\n}",
            "=h,f,f,r",
            loc,
            ip,
        )
    )


@cute.jit
def ue8m0_code(bits: Uint32):
    """The UE8M0 code of a positive scale (rounded up to a power of two) and
    the bits of the FP32 inverse scale."""
    exponent = (bits >> Uint32(23)) & Uint32(255)
    mantissa = bits & Uint32(0x7FFFFF)
    code = exponent
    if mantissa != Uint32(0) and exponent != Uint32(254):
        if exponent != Uint32(0):
            code = exponent + Uint32(1)
        elif mantissa > Uint32(0x400000):
            code = Uint32(1)
    if exponent == Uint32(255):
        code = Uint32(255)
    inverse = (Uint32(254) - code) << Uint32(23)
    if code == Uint32(254):
        inverse = Uint32(0x400000)
    return code, inverse


class RoutedExpertGemm1(RoutedExpertGemm):
    """GEMM1: ``h = MXFP8(clamped SwiGLU(x @ w13[expert]^T))`` of each token
    tile, in the tile's permuted rows.

    B is tile ``c``'s rows of the routed input ``x``: tile row ``r`` below
    the tile's limit is token ``record[4 + 2 C + 8 c + r]``. The B load warp
    gathers those rows and their linear UE8M0 scales into the layout the MMA
    reads, so accumulator column ``r`` is tile row ``r``, and the epilogue
    writes the columns below the limit. w13's rows interleave up and
    gate, so each thread pairs them, clamps them, applies SwiGLU, scales each
    32-column block by its amax / 448 rounded up to a power of two (UE8M0),
    and stores E4M3 values with the block scales in the per-tile layout GEMM2
    reads. The m tile holding the slice's last columns also zeroes the padding
    columns of H and gives their blocks scale 1.
    """

    route_before_wait = False
    gather_b = True
    signal_ready = True

    def __init__(
        self,
        hidden: int,
        slice_: int,
        padded: int,
        cta_k: int = 256,
        num_ab_stage: int = 6,
    ) -> None:
        super().__init__(2 * padded, 2 * slice_ // 128, hidden, cta_k, num_ab_stage)
        self.slice, self.padded = slice_, padded

    @cute.jit
    def __call__(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        b_scale_ptr: cute.Pointer,
        h_ptr: cute.Pointer,
        hs_ptr: cute.Pointer,
        route_ptr: cute.Pointer,
        clamp_ptr: cute.Pointer,
        rows: Int32,
        experts: Int32,
        cta_words: Int32,
        h_rows: Int32,
        ctas: Int32,
        stream: cuda.CUstream,
    ):
        """A: w13 (2 x padded, hidden, experts), K-major; B: x (rows,
        hidden), ``b_scale_ptr`` its scales as Uint32 words (rows, hidden /
        128); H (``h_rows``, padded) as Uint16 pairs; HS in the per-tile
        layout; ``clamp`` FP32 per expert; ``ctas`` persistent CTAs."""
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
        words = self.k // 128
        b_scale = cute.make_tensor(
            b_scale_ptr, cute.make_layout((rows, words), stride=(words, 1))
        )
        h = cute.make_tensor(h_ptr, cute.make_layout(h_rows * (self.padded // 2)))
        hs = cute.make_tensor(
            hs_ptr,
            cute.make_layout(
                (h_rows + TILE - 1) // TILE * tile_scale_bytes(self.padded)
            ),
        )
        clamp = cute.make_tensor(clamp_ptr, cute.make_layout(experts))
        # The ready words follow the token table (one row per padded row).
        ready = 4 + 2 * cta_words + h_rows
        route = cute.make_tensor(route_ptr, cute.make_layout(ready + cta_words))
        self._launch(
            a,
            sfa,
            b,
            None,
            route,
            (h, hs, clamp),
            (b, b_scale),
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
        """Gather each item's tile rows of ``x`` and their scales into B.

        Per stage, lane ``l`` copies the 16-byte chunk ``l % 16`` of tile rows
        ``l // 16 + 2 j`` (``j < 4``) with ``cp.async``, to its address under
        the 128-byte swizzle that TMA and the MMA apply to shared addresses,
        and lane ``l < 8 x atoms`` copies one 32-bit word (four K blocks) of
        row ``l // atoms``'s scales. Rows at or past the tile's limit are not
        loaded; their accumulator columns are never stored. The warp never
        waits for a copy it has just issued: each stage is published two
        stages late (after issuing a stage's copies, the warp waits for those
        of two stages before, fences them to the async proxy and arrives on
        that stage's full barrier)."""
        x, b_scale = b_side
        bid, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        lane = cute.arch.lane_idx()
        atoms = self.cta_k // 128  # 128-column swizzle atoms and scale words per row
        b_stage_bytes = TILE * self.cta_k
        sf_stage_bytes = 512 * atoms
        b_base = sB.iterator.toint()
        sf_base = sSFB.iterator.toint()
        chunk = lane % 16
        # The chunk's offset in a row of an 8-row stage: atoms of 8 x 128 bytes.
        chunk_offset = (chunk // 8) * (TILE * 128) + (chunk % 8) * 16
        sf_row = lane // atoms
        sf_atom = lane - sf_row * atoms
        # Row r < 32 of a 128-row scale atom: byte 16 r, four K blocks.
        sf_offset = sf_atom * 512 + sf_row * 16
        token = cute.make_rmem_tensor((4,), Int32)
        state = make_pipeline_state(PipelineUserType.Producer, self.num_ab_stage)
        older = Int32(-1)  # stages issued but not yet published, oldest first
        newer = Int32(-1)
        for i in cutlass.range(count, unroll=1):
            item = bid + i * grid
            n_tile = item // self.m_tiles
            row0 = n_tile * TILE
            valid = mRoute[4 + cta_words + n_tile] - row0
            for j in cutlass.range_constexpr(4):
                token[j] = Int32(-1)
                if lane // 16 + 2 * j < valid:
                    token[j] = mRoute[4 + 2 * cta_words + row0 + lane // 16 + 2 * j]
            sf_token = Int32(-1)
            if lane < TILE * atoms and sf_row < valid:
                sf_token = mRoute[4 + 2 * cta_words + row0 + sf_row]
            for k_tile in cutlass.range(self.k_tiles, unroll=1):
                cute.arch.mbarrier_wait(ab_empty_bar + state.index, state.phase)
                if sf_token >= 0:
                    cp_async_4(
                        sf_base + state.index * sf_stage_bytes + sf_offset,
                        (
                            b_scale.iterator
                            + sf_token * cute.size(b_scale, mode=[1])
                            + k_tile * atoms
                            + sf_atom
                        ).toint(),
                    )
                for j in cutlass.range_constexpr(4):
                    if token[j] >= 0:
                        linear = (
                            b_base
                            + state.index * b_stage_bytes
                            + (lane // 16 + 2 * j) * 128
                            + chunk_offset
                        )
                        cp_async_16(
                            linear ^ (((linear >> 7) & 7) << 4),
                            (
                                x.iterator
                                + token[j] * self.k
                                + k_tile * self.cta_k
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
        """H and HS of the item (class docstring). The thread holds, for tile
        rows ``t, t + 1`` (``t`` = 2 x lane % 4), the up and gate rows of H
        columns ``p`` (registers 0-3) and ``p + 1`` (4-7), ``p`` = 64 x m_tile
        + 16 x warp + 2 x lane / 4."""
        mH, mHS, mClamp = epi
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - self.epilogue_warp
        lane = cute.arch.lane_idx()
        token = (lane % 4) * 2
        pair = warp * 16 + (lane // 4) * 2
        limit = mClamp[mRoute[4 + n_tile]]
        row0 = n_tile * TILE
        valid = mRoute[4 + cta_words + n_tile] - row0
        # The thread's two tile rows (-1: past the tile's limit).
        row_a = Int32(-1)
        row_b = Int32(-1)
        if token < valid:
            row_a = Int32(token)
        if token + 1 < valid:
            row_b = Int32(token + 1)

        lo_a, lo_b = clamped_swiglu_x2(
            tDrAcc[0], tDrAcc[1], tDrAcc[2], tDrAcc[3], limit, -limit
        )
        hi_a, hi_b = clamped_swiglu_x2(
            tDrAcc[4], tDrAcc[5], tDrAcc[6], tDrAcc[7], limit, -limit
        )
        amax_a = abs_max_x2(lo_a, hi_a)
        amax_b = abs_max_x2(lo_b, hi_b)
        for shift in cutlass.range_constexpr(4, 1, -1):
            amax_a = cute.arch.fmax(
                amax_a, cute.arch.shuffle_sync_bfly(amax_a, 1 << shift), ftz=True
            )
            amax_b = cute.arch.fmax(
                amax_b, cute.arch.shuffle_sync_bfly(amax_b, 1 << shift), ftz=True
            )
        # A 32-column block spans warps 2j and 2j + 1.
        base = (item % 2) * (4 * TILE)
        if lane < 4:
            exchange[base + warp * TILE + token] = amax_a
            exchange[base + warp * TILE + token + 1] = amax_b
        cute.arch.barrier(barrier_id=1, number_of_threads=128)
        partner = base + (warp ^ 1) * TILE + token
        bits_a, bits_b = scale_bits_x2(
            amax_a, exchange[partner], amax_b, exchange[partner + 1]
        )
        code_a, inverse_a = ue8m0_code(bits_a)
        code_b, inverse_b = ue8m0_code(bits_b)

        h_row = self.padded // 2
        column = (m_tile * 64 + pair) // 2
        if row_a >= 0:
            mH[(row0 + row_a) * h_row + column] = quantize_e4m3x2(lo_a, hi_a, inverse_a)
        if row_b >= 0:
            mH[(row0 + row_b) * h_row + column] = quantize_e4m3x2(lo_b, hi_b, inverse_b)
        tile_bytes = tile_scale_bytes(self.padded)
        if warp % 2 == 0 and lane < TILE:
            block = m_tile * 2 + warp // 2
            row = row_a
            code = code_a
            if lane >= 4:
                row = row_b
                code = code_b
            if row >= 0:
                mHS[n_tile * tile_bytes + (block // 4) * 512 + row * 16 + block % 4] = (
                    Uint8(code)
                )
        if m_tile == self.m_tiles - 1:
            # Zero H's padding columns; their blocks get scale 1 (code 127).
            thread = warp * 32 + lane
            pad_row = thread // 16
            if pad_row < valid:
                pad = (row0 + pad_row) * h_row + (self.slice + (thread % 16) * 4) // 2
                mH[pad] = Uint16(0)
                mH[pad + 1] = Uint16(0)
            if thread < 2 * TILE:
                pad_row = thread // 2
                pad_block = self.slice // 32 + thread % 2
                if pad_row < valid:
                    mHS[
                        n_tile * tile_bytes
                        + (pad_block // 4) * 512
                        + pad_row * 16
                        + pad_block % 4
                    ] = Uint8(127)
