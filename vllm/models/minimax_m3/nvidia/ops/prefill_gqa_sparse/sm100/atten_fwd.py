# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# mypy: ignore-errors
"""SM100 sparse attention forward kernel.

This kernel implements the delivered Sparse Attention / Sparse Page Attention
forward contract:
- CSR sparse metadata (`k2q_row_ptr`, `k2q_q_indices`)
- varlen Q metadata via `cu_seqlens_q`
- Sparse Attention with flat varlen K/V
- Sparse Page Attention with paged K/V
"""

import math
from functools import partial

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as cutlass_pipeline
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from quack import copy_utils

from ..common import blackwell_helpers as sm100_helpers
from ..common import mma_sm100_desc as sm100_desc
from ..common import pipeline
from ..common.cute_dsl_utils import assume_tensor_aligned
from ..common.mask import mask_r2p_lambda, r2p_bitmask_below
from ..common.named_barrier import NamedBarrierFwdSm100

# Shared raw PTX helpers and layout conversions used by the lean kernel.
from ..common.paged_kv import PagedKVManager
from ..common.softmax import SoftmaxSm100
from ..common.tma_utils import (
    TMA_CACHE_EVICT_FIRST,
    TMA_CACHE_EVICT_LAST,
    make_16x256b_tensor_mn_view,
    prefetch_tma_desc_raw,
    real_col_to_stg128_fake_col,
    real_col_to_stg128_half_fake_col,
    stg_128_bf16_cs,
    stg_128_cs,
    stg_128_f16_cs,
    tma_gather4_cached,
    tma_gather4_prefetch,
    tma_tile_load,
    tma_tile_load_cached,
    tma_tile_prefetch,
)


class SparseAttentionForwardSm100:
    """SM100 sparse attention forward kernel."""

    k_tile = 64  # UTCMMA bf16 K-tile (matches sparse_fwd_utcmma.py)

    def __init__(
        self,
        head_dim: int = 128,
        qheadperkv: int = 16,
        m_block_size: int = 128,
        n_block_size: int = 128,
        paged_kv: bool = False,
        page_size: int | None = None,
        has_seqused_k: bool = False,
        causal: bool = False,
    ):
        if head_dim != 128:
            raise NotImplementedError(
                f"SparseAttentionForwardSm100 currently supports only D=128, got D={head_dim}"
            )
        self.head_dim = 128
        self.qheadperkv = qheadperkv
        self.use_q_gather4 = qheadperkv in (4, 2, 1)
        if qheadperkv not in (16, 8, 4, 2, 1):
            raise ValueError(
                "SparseAttentionForwardSm100 supports qheadperkv in "
                f"{{1, 2, 4, 8, 16}}, got {qheadperkv}"
            )
        self.tokens_per_gather4 = 4 // qheadperkv if self.use_q_gather4 else 0
        self.m_block_size = m_block_size  # 128 packed Q heads
        self.n_block_size = n_block_size  # 128 KV-block width
        self.paged_kv = paged_kv
        self.page_size = page_size
        self.has_seqused_k = has_seqused_k
        self.causal = causal
        if self.paged_kv:
            if page_size is None:
                raise ValueError("page_size must be provided when paged_kv=True")
            if page_size < 8:
                raise ValueError(
                    f"page_size must be >= 8 for paged TMA, got {page_size}"
                )
            if page_size % n_block_size == 0:
                self.kv_segment_rows = n_block_size
                self.kv_segments_per_block = 1
                self.kv_blocks_per_page = page_size // n_block_size
            elif n_block_size % page_size == 0:
                self.kv_segment_rows = page_size
                self.kv_segments_per_block = n_block_size // page_size
                self.kv_blocks_per_page = 1
            else:
                raise ValueError(
                    f"page_size ({page_size}) must divide blk_kv ({n_block_size}) "
                    "or be divisible by it"
                )
            self.paged_block_tma = page_size >= n_block_size
            self.paged_segment_tma = page_size < n_block_size
        else:
            self.kv_segment_rows = n_block_size
            self.kv_segments_per_block = 1
            self.kv_blocks_per_page = 1
            self.paged_block_tma = False
            self.paged_segment_tma = False
        self.use_raw_kv_desc = self.paged_segment_tma
        self.q_tokens_per_group = m_block_size // qheadperkv  # 8

        self.mma_tiler_qk = (m_block_size, n_block_size, self.head_dim)
        self.mma_tiler_pv = (m_block_size, self.head_dim, n_block_size)
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32

        # Pipeline configuration — deeper Q prefetch ring plus 2-slot S/O rings.
        self.q_stage = 2
        self.s_stage = 2
        self.o_stage = 2
        self.kv_stage = 1
        # Sparse q_idx metadata ring bridging load -> epilogue. Sized larger
        # than the in-flight group distance so epilogue can reuse q_idx
        # without rereading mK2qIndices.
        self.qidx_meta_stages = 16

        self.k_stages = 2
        self.q_stage_stride_bytes = m_block_size * self.head_dim * 2
        self.k_tile_stride_bytes = m_block_size * self.k_tile * 2
        self.token_stride_bytes = qheadperkv * self.k_tile * 2

        # Warp layout: two softmax WGs, one Q-load/epilogue WG, one
        # MMA issue warp, two K/V load warps, and one empty warp.
        self.warps_per_group = 4
        self.softmax0_warp_base = 0
        self.softmax1_warp_base = self.softmax0_warp_base + self.warps_per_group
        self.store_warp_base = self.softmax1_warp_base + self.warps_per_group
        self.mma_warp_id = self.store_warp_base + self.warps_per_group
        self.load_warp_base = self.mma_warp_id + 1
        self.q_load_warp_base = self.store_warp_base
        self.kv_load_warp_base = self.load_warp_base
        self.num_kv_load_warps = 2
        self.num_q_load_warps = self.warps_per_group
        self.total_warps = 16
        self.threads_per_cta = cute.arch.WARP_SIZE * self.total_warps  # 512

        # TMEM layout follows FA SM100:
        #   S0/S1: [0:128], [128:256]
        #   O0/O1: [256:384], [384:512] for hdim_v=128
        #   P is bf16 and starts halfway into each S tile.
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        self.tmem_s_offset = 0
        self.tmem_stage_stride = n_block_size
        self.tmem_o_stage_stride = self.head_dim
        self.tmem_o_offset = self.s_stage * n_block_size
        self.tmem_s_to_p_offset = n_block_size // 2
        self.tmem_p_offset = self.tmem_s_offset + self.tmem_s_to_p_offset
        raw_tmem_total = self.tmem_o_offset + self.o_stage * self.tmem_o_stage_stride
        # SM100 TMEM allocation requires a power-of-two column count.  The
        # 128-wide path naturally uses 512 columns; 64-wide KV blocks use 384
        # columns and must round the allocation up while keeping the same
        # logical offsets.
        self.tmem_total = 1 << (raw_tmem_total - 1).bit_length()

        # Current WS kernel keeps the simpler full-P handoff.  Releasing the
        # S/P slot immediately after the final P store already gives overlap
        # with row_sum/stats without needing mid-PV waits.
        self.split_P_arrive = 0

        # Register allocation per role.  The causal hdim128 split gives the
        # epilogue enough room for partial-O/LSE address generation while the
        # two softmax WGs still have enough registers to avoid S/P spills.
        self.num_regs_softmax = 176 if causal else 192
        self.num_regs_store = 112 if causal else 80
        self.num_regs_other = 512 - self.num_regs_softmax * 2 - self.num_regs_store
        self.num_regs_mma = self.num_regs_other
        self.num_regs_load = self.num_regs_other
        self.num_regs_empty = self.num_regs_other
        self.store_reg_decrease = self.num_regs_store <= 128
        self.ex2_emu_freq = 16 if causal else 0
        self.ex2_emu_start_frg = 1
        self.buffer_align_bytes = 1024

        # Blackwell config.
        self.use_2cta_instrs = False
        self.cta_group_size = 1
        self.cluster_shape_mn = (1, 1)
        self.cluster_shape_mnk = (1, 1, 1)

        self.arch = BaseDSL._get_dsl().get_arch_enum()

    @cute.jit
    def _batch_q_offset(
        self,
        batch_idx: Int32,
        mCuSeqlensQ,
    ) -> Int32:
        return mCuSeqlensQ[batch_idx]

    @cute.jit
    def _logical_seqlen_k(
        self,
        batch_idx: Int32,
        mPageTable,
        mSeqUsedK,
        mCuSeqlensK,
    ) -> Int32:
        if const_expr(self.has_seqused_k):
            return mSeqUsedK[batch_idx]
        if const_expr(self.paged_kv):
            return Int32(mPageTable.shape[1]) * Int32(self.page_size)
        return mCuSeqlensK[batch_idx + Int32(1)] - mCuSeqlensK[batch_idx]

    @cute.jit
    def _valid_cols_in_block(
        self,
        batch_idx: Int32,
        kv_block_idx: Int32,
        mPageTable,
        mSeqUsedK,
        mCuSeqlensK,
    ) -> Int32:
        seqlen_k = self._logical_seqlen_k(batch_idx, mPageTable, mSeqUsedK, mCuSeqlensK)
        block_start = kv_block_idx * Int32(self.n_block_size)
        remaining = seqlen_k - block_start
        remaining = cutlass.max(remaining, Int32(0))
        return cutlass.min(remaining, Int32(self.n_block_size))

    @cute.jit
    def _load_q_idx(
        self,
        mK2qIndices: cute.Tensor,
        head_kv_idx: Int32,
        row_start: Int32,
        qi: Int32,
    ) -> Int32:
        return mK2qIndices[head_kv_idx, row_start + qi]

    @cute.jit
    def _load_qsplit_idx(
        self,
        mK2qQSplitIndices: cute.Tensor,
        head_kv_idx: Int32,
        row_start: Int32,
        qi: Int32,
    ) -> Int32:
        return mK2qQSplitIndices[head_kv_idx, row_start + qi]

    @cute.jit
    def _decode_q_idx_from_qsplit(self, qsplit: Int32) -> Int32:
        return qsplit & Int32(0x00FF_FFFF)

    @cute.jit
    def _decode_split_idx_from_qsplit(self, qsplit: Int32) -> Int32:
        return (qsplit >> Int32(24)) & Int32(0xFF)

    @cute.jit
    def _lower_bound_q_idx(
        self,
        mK2qIndices: cute.Tensor,
        head_kv_idx: Int32,
        row_start: Int32,
        count: Int32,
        q_value: Int32,
    ) -> Int32:
        left = Int32(0)
        right = count
        # k2q_q_indices is sorted by q_idx within each CSR row.  A fixed
        # 32-step loop covers int32-sized rows and avoids a 128-element
        # serial probe on every causal CTA.
        for _ in cutlass.range(32, unroll=1):
            if left < right:
                mid = (left + right) // Int32(2)
                q_idx = self._load_q_idx(
                    mK2qIndices,
                    head_kv_idx,
                    row_start,
                    mid,
                )
                if q_idx < q_value:
                    left = mid + Int32(1)
                else:
                    right = mid
        return left

    # ------------------------------------------------------------------
    # Host-side: TMA descriptors, SMEM layout, launch
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        mK: cute.Tensor,  # Sparse Attention: [total_k, head_kv, dim] / Sparse Page Attention: prepared paged KV tensor
        mV: cute.Tensor,  # Sparse Attention: [total_k, head_kv, dim] / Sparse Page Attention: prepared paged KV tensor
        mK2qIndices: cute.Tensor,  # csr payload: [head_kv, nnz]
        mK2qQSplitIndices: cute.Tensor,  # csr payload: [head_kv, nnz] packed q_idx/split slot
        mK2qCounts: cute.Tensor,  # csr row_ptr: [head_kv, total_rows + 1]
        mSchedulerMetadata: cute.Tensor | None,
        mWorkCount: cute.Tensor | None,
        mO_partial: cute.Tensor,  # fp32 O_partial buffer (kept alive)
        mLSE_partial: cute.Tensor,  # fp32 LSE_partial
        mLSE_temperature_partial: cute.Tensor
        | None,  # fp32 temperature-scaled LSE_partial
        mQ_flat: cute.Tensor,  # [batch*Sq*head_q, dim] bf16, pre-flattened
        mQ_gather4_desc: cute.Tensor
        | None,  # [128] uint8 tensor map for gather4 Q load
        mK_raw_desc,
        mV_raw_desc,
        mPageTable,
        mSeqUsedK,
        mCuSeqlensQ,
        mCuSeqlensK,
        softmax_scale: Float32,
        lse_temperature_scale: Float32,
        num_kv_blocks: Int32,
        num_heads_kv: Int32,
        seq_len_q: Int32,
        work_capacity: Int32,
        stream=None,
    ):
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.q_dtype = self.k_dtype
        self.o_dtype = mO_partial.element_type
        if const_expr(self.o_dtype not in [Float32, cutlass.BFloat16, cutlass.Float16]):
            raise TypeError(f"Unsupported O_partial dtype: {self.o_dtype}")
        mK, mV = [assume_tensor_aligned(t) for t in (mK, mV)]

        if const_expr(self.use_raw_kv_desc):
            # Sparse Page Attention with page_size < blk_kv stays on the raw
            # descriptor path. Host-side descriptors are built from per-head
            # 2D views.
            pass
        elif const_expr(not self.paged_kv):
            # Flat varlen K/V use CUTE-managed TMA descriptors, matching FA:
            # K: [total_k, h, d] -> [total_k, d, h].
            # V: [total_k, h, d] -> [d, total_k, h] for MN-major PV.
            layout_t = [0, 2, 1]
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=layout_t))
            mV_kv = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=layout_t))
            mV = cute.make_tensor(
                mV_kv.iterator, cute.select(mV_kv.layout, mode=[1, 0, 2])
            )
        else:
            # Sparse Page Attention with page-sized blocks can use the blocked
            # paged TMA layout directly.
            layout_t = [1, 3, 2, 0]
            mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=layout_t))
            mV_kv = cute.make_tensor(mV.iterator, cute.select(mV.layout, mode=layout_t))
            # V: (s,d,h,b) -> (d,s,h,b) for MN-major
            mV = cute.make_tensor(
                mV_kv.iterator, cute.select(mV_kv.layout, mode=[1, 0, 2, 3])
            )

        # ------------------------------------------------------------------
        #  UTCMMA TiledMma: QK^T and PV
        # ------------------------------------------------------------------
        cta_group = tcgen05.CtaGroup.ONE
        tiled_mma_qk = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            Float32,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN,
            Float32,
            cta_group,
            self.mma_tiler_pv[:2],
            tcgen05.OperandSource.TMEM,
        )

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        # ------------------------------------------------------------------
        #  SMEM layouts: sQ/sK/sV only. O_partial is written directly from
        #  registers to GMEM in the epilogue.
        # ------------------------------------------------------------------
        total_q_stages = self.q_stage
        sQ_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, total_q_stages
        )
        # Per-sub-tile load layout: total_q_stages * q_tokens_per_group * k_stages slots
        num_subtiles_total = total_q_stages * self.q_tokens_per_group * self.k_stages
        sQ_load_layout = sm100_utils.make_smem_layout(
            tcgen05.OperandMajorMode.K,
            (self.qheadperkv, self.k_tile),
            self.q_dtype,
            num_subtiles_total,
        )
        sK_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        sV_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        # P SMEM layout metadata (no actual SMEM allocation — P lives in TMEM,
        # overlaying the S region; this layout is only used to compute the PV
        # A-operand TMEM descriptor shape at the MMA issue site.)
        tP_layout = sm100_utils.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )

        # ------------------------------------------------------------------
        #  TMA atoms
        # ------------------------------------------------------------------
        kv_tma_bytes = cute.size_in_bytes(
            self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2])
        ) + cute.size_in_bytes(self.v_dtype, cute.select(sV_layout, mode=[0, 1, 2]))
        q_tma_bytes = cute.size_in_bytes(
            self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2])
        )
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        if const_expr(self.use_raw_kv_desc):
            tma_atom_K = None
            tma_atom_V = None
        else:
            tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mK,
                cute.select(sK_layout, mode=[0, 1, 2]),
                self.mma_tiler_qk,
                tiled_mma_qk,
                cta_layout_vmnk.shape,
            )
            tma_atom_V, mV = cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                mV,
                cute.select(sV_layout, mode=[0, 1, 2]),
                self.mma_tiler_pv,
                tiled_mma_pv,
                cta_layout_vmnk.shape,
            )

        # Q per-sub-tile TMA atom: box (qheadperkv=16, k_tile=64)
        mQ_flat = assume_tensor_aligned(mQ_flat)
        mQ_2d = cute.make_tensor(
            mQ_flat.iterator, cute.select(mQ_flat.layout, mode=[0, 1])
        )
        if const_expr(self.use_q_gather4):
            # Placeholder atom for unified kernel signature. Small-GQA Q load
            # uses raw gather4 and keeps mQ_2d as a plain row-major GMEM tensor.
            tma_atom_Q = tma_atom_V
        else:
            tma_atom_Q, mQ_2d = cpasync.make_tiled_tma_atom(
                tma_load_op,
                mQ_2d,
                cute.select(sQ_load_layout, mode=[0, 1]),
                (self.qheadperkv, self.k_tile),
            )
        q_subtile_bytes = cute.size_in_bytes(
            self.q_dtype, cute.select(sQ_load_layout, mode=[0, 1])
        )

        softmax_scale_log2 = softmax_scale * Float32(math.log2(math.e))
        lse_temperature_scale_log2 = softmax_scale_log2 * lse_temperature_scale

        # ------------------------------------------------------------------
        #  SharedStorage — lean: just the mbars and tiles we actually use.
        #
        #  Mbarriers (all storage rings stay below the 64-per-CTA limit):
        #    mbar_kv           [2]  one-shot K/V load handshake (full + empty)
        #    mbar_q            [q_stage * 2]  Q producer/consumer ring
        #    mbar_s            [2]  QK UTCMMA -> softmax (full + empty)
        #    mbar_o            [2]  PV UTCMMA -> epilogue (full + empty)
        #    mbar_p_lastsplit  [s_stage * 2]  softmax partial-P arrive -> PV
        #                      (unused when ``self.split_P_arrive == 0``;
        #                      allocated unconditionally to keep storage
        #                      layout stable across configurations)
        #    mbar_sm_stats     [s_stage * 2]  softmax row_sum/row_max
        #                      publish -> epilogue consumer read.  In lean
        #                      1-WG topology the producer and consumer are
        #                      the same 128 WG_C threads, but we keep the
        #                      barrier for structural parity with FA so the
        #                      softmax body reads identically.
        # ------------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            mbar_k: cute.struct.MemRange[Int64, 2]
            mbar_v: cute.struct.MemRange[Int64, 2]
            mbar_q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_s: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_p: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_o: cute.struct.MemRange[Int64, self.o_stage * 2]
            mbar_sm_stats: cute.struct.MemRange[Int64, self.o_stage * 2]
            tmem_dealloc_mbar_ptr: Int64
            tmem_holding_buf: Int32
            # Per-row softmax stats cache (for epilogue LSE + rescale):
            #   [0 : m_block_size)            row_sum
            #   [m_block_size : 2*m_block_size) row_max
            sScale: cute.struct.MemRange[Float32, self.o_stage * self.m_block_size * 2]
            # Per-row temperature LSE row_sum cache. The row_max is shared with
            # sScale because lse_temperature_scale is positive.
            sScaleTemperature: cute.struct.MemRange[
                Float32, self.o_stage * self.m_block_size
            ]
            # Per-token split_id from prepare-time per-edge metadata.
            sSplitIdx: cute.struct.MemRange[
                Int32, self.o_stage * self.q_tokens_per_group
            ]
            # Per-token q_idx cache to avoid reloading sparse indices in epilogue.
            sQIdx: cute.struct.MemRange[Int32, self.o_stage * self.q_tokens_per_group]
            # Number of diagonal-block queries for this KV block. Because k2q
            # is q_idx-ascending, these form a prefix and only that prefix
            # needs block-internal causal masking.
            sDiagQCount: cute.struct.MemRange[Int32, 1]
            # CTA-wide row metadata, published once by tidx 0 and reused by
            # all warp-specialized roles:
            #   [0] batch_idx
            #   [1] kv_block_idx
            #   [2] row_start
            #   [3] count_raw
            #   [4] kv_valid_cols
            #   [5] q_batch_offset
            #   [6] k_batch_offset
            #   [7] causal_q_offset = seqlen_k - seqlen_q
            sRowMeta: cute.struct.MemRange[Int32, 8]
            sPagedKvIdx: cute.struct.MemRange[Int32, max(1, self.kv_segments_per_block)]
            sQLoadMIdx: cute.struct.MemRange[
                Int32, self.q_stage * self.q_tokens_per_group
            ]
            # Packed per-edge q/split metadata:
            #   low 24 bits = q_idx, high 8 bits = split slot.
            sQIdxMeta: cute.struct.MemRange[
                Int32, self.qidx_meta_stages * self.q_tokens_per_group
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(sV_layout)],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(sQ_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        num_ctas = work_capacity

        self.kernel(
            mK,
            mV,
            mK2qIndices,
            mK2qQSplitIndices,
            mK2qCounts,
            mSchedulerMetadata,
            mWorkCount,
            mO_partial,
            mLSE_partial,
            mLSE_temperature_partial,
            mQ_2d,
            mQ_gather4_desc,
            mK_raw_desc,
            mV_raw_desc,
            mPageTable,
            mSeqUsedK,
            mCuSeqlensQ,
            mCuSeqlensK,
            softmax_scale_log2,
            lse_temperature_scale_log2,
            lse_temperature_scale,
            sQ_layout,
            sQ_load_layout,
            sK_layout,
            sV_layout,
            tP_layout,
            tma_atom_K,
            tma_atom_V,
            tma_atom_Q,
            tiled_mma_qk,
            tiled_mma_pv,
            kv_tma_bytes,
            q_tma_bytes,
            q_subtile_bytes,
            num_kv_blocks,
            num_heads_kv,
            seq_len_q,
            work_capacity,
        ).launch(
            grid=(num_ctas,),
            block=[self.threads_per_cta, 1, 1],
            smem=max(SharedStorage.size_in_bytes(), 49152),
            stream=stream,
            min_blocks_per_mp=1,
        )

    # ------------------------------------------------------------------
    # Device-side: kernel entry, dispatch by warpgroup
    # ------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        # Runtime tensors
        tma_K: cute.Tensor,
        tma_V: cute.Tensor,
        mK2qIndices: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        mK2qCounts: cute.Tensor,
        mSchedulerMetadata: cute.Tensor | None,
        mWorkCount: cute.Tensor | None,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: cute.Tensor | None,
        mQ_2d: cute.Tensor,
        mQ_gather4_desc: cute.Tensor | None,
        mK_raw_desc,
        mV_raw_desc,
        mPageTable,
        mSeqUsedK,
        mCuSeqlensQ,
        mCuSeqlensK,
        # Scalars
        softmax_scale_log2: Float32,
        lse_temperature_scale_log2: Float32,
        lse_temperature_scale: Float32,
        # Layouts
        sQ_layout: cute.ComposedLayout,
        sQ_load_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        # TMA atoms
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_Q: cute.CopyAtom,
        # MMA
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        # Transfer sizes
        kv_tma_bytes: cutlass.Constexpr[int],
        q_tma_bytes: cutlass.Constexpr[int],
        q_subtile_bytes: cutlass.Constexpr[int],
        # Iteration bounds
        num_kv_blocks: Int32,
        num_heads_kv: Int32,
        seq_len_q: Int32,
        work_capacity: Int32,
    ):
        # ------------------------------------------------------------------
        #  Thread / warp identity, CTA coordinate
        # ------------------------------------------------------------------
        bidx, _, _ = cute.arch.block_idx()
        row_linear = Int32(0)
        head_kv_idx = Int32(0)
        batch_idx = Int32(0)
        kv_block_idx = Int32(0)
        work_q_begin = Int32(0)
        work_q_count = Int32(0)
        cta_valid_work = True
        work_idx = bidx
        cta_valid_work = work_idx < mWorkCount[Int32(0)]
        if cta_valid_work:
            head_kv_idx = mSchedulerMetadata[work_idx, Int32(0)]
            row_linear = mSchedulerMetadata[work_idx, Int32(1)]
            work_q_begin = mSchedulerMetadata[work_idx, Int32(2)]
            work_q_count = mSchedulerMetadata[work_idx, Int32(3)]
            batch_idx = mSchedulerMetadata[work_idx, Int32(4)]
            kv_block_idx = mSchedulerMetadata[work_idx, Int32(5)]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        head_q = num_heads_kv * Int32(self.qheadperkv)
        paged_kv_manager = (
            PagedKVManager.create(
                mPageTable,
                page_size=self.page_size,
                n_block_size=self.n_block_size,
            )
            if const_expr(self.paged_kv)
            else None
        )

        # Prefetch TMA descriptors (warp 0 once).
        if warp_idx == 0:
            if const_expr(not self.use_raw_kv_desc):
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
            else:
                with cute.arch.elect_one():
                    prefetch_tma_desc_raw(
                        mK_raw_desc.iterator + head_kv_idx * Int32(128)
                    )
                    prefetch_tma_desc_raw(
                        mV_raw_desc.iterator + head_kv_idx * Int32(128)
                    )
            if const_expr(not self.use_q_gather4):
                cpasync.prefetch_descriptor(tma_atom_Q)
            else:
                with cute.arch.elect_one():
                    prefetch_tma_desc_raw(mQ_gather4_desc.iterator)
        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk), (tiled_mma_qk.thr_id.shape,)
        )

        # ------------------------------------------------------------------
        #  SMEM allocation (all warps — same SharedStorage type from __call__)
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQ_load = storage.sQ.get_tensor(
            sQ_load_layout.outer, swizzle=sQ_load_layout.inner
        )
        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.o_stage * self.m_block_size * 2)
        )
        sScaleTemperature = storage.sScaleTemperature.get_tensor(
            cute.make_layout(self.o_stage * self.m_block_size)
        )
        sSplitIdx = storage.sSplitIdx.get_tensor(
            cute.make_layout((self.o_stage * self.q_tokens_per_group,))
        )
        sQIdx = storage.sQIdx.get_tensor(
            cute.make_layout((self.o_stage * self.q_tokens_per_group,))
        )
        sDiagQCount = storage.sDiagQCount.get_tensor(cute.make_layout((1,)))
        sRowMeta = storage.sRowMeta.get_tensor(cute.make_layout((8,)))
        sPagedKvIdx = storage.sPagedKvIdx.get_tensor(
            cute.make_layout((max(1, self.kv_segments_per_block),))
        )
        sQLoadMIdx = storage.sQLoadMIdx.get_tensor(
            cute.make_layout((self.q_stage * self.q_tokens_per_group,))
        )
        sQIdxMeta = storage.sQIdxMeta.get_tensor(
            cute.make_layout((self.qidx_meta_stages * self.q_tokens_per_group,))
        )
        mbar_k_ptr = storage.mbar_k.data_ptr()
        mbar_v_ptr = storage.mbar_v.data_ptr()

        # ------------------------------------------------------------------
        #  TMEM allocator — allocator warp 0 serves the whole CTA.
        # ------------------------------------------------------------------
        tmem_alloc_warps: cutlass.Constexpr[int] = self.warps_per_group * 2 + 1
        tmem_alloc_threads = cute.arch.WARP_SIZE * tmem_alloc_warps
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=tmem_alloc_threads,
        )
        tmem = cutlass.utils.TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr,
        )

        # ------------------------------------------------------------------
        #  Warp-specialized pipelines.
        # ------------------------------------------------------------------
        ThreadCooperativeGroup = partial(
            cutlass_pipeline.CooperativeGroup, cutlass_pipeline.Agent.Thread
        )
        tma_thread = ThreadCooperativeGroup(1)
        mma_thread = ThreadCooperativeGroup(1)
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * self.warps_per_group
        )
        epilogue_threads = softmax_threads

        pipeline_q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=tma_thread,
            consumer_group=mma_thread,
            tx_count=q_tma_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_s = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_s.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_thread,
            consumer_group=softmax_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_p = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_p.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=mma_thread,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_o = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_o.data_ptr(),
            num_stages=self.o_stage,
            producer_group=mma_thread,
            consumer_group=epilogue_threads,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )
        pipeline_sm_stats = pipeline.PipelineAsync.create(
            barrier_storage=storage.mbar_sm_stats.data_ptr(),
            num_stages=self.o_stage,
            producer_group=softmax_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )
        # Cluster sync (no-op for 1CTA cluster).
        pipeline_init_arrive(cluster_shape_mn=cta_layout_vmnk, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cta_layout_vmnk)

        # ------------------------------------------------------------------
        #  Work count: how many Q tokens reference this CTA's KV block
        # ------------------------------------------------------------------
        k_tma_bytes = cute.size_in_bytes(
            self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2])
        )
        v_tma_bytes = cute.size_in_bytes(
            self.v_dtype, cute.select(sV_layout, mode=[0, 1, 2])
        )
        if tidx == 0:
            row_batch_idx = batch_idx
            row_kv_block_idx = kv_block_idx
            base_row_start = mK2qCounts[head_kv_idx, row_linear]
            row_start = base_row_start
            count_raw = mK2qCounts[head_kv_idx, row_linear + Int32(1)] - base_row_start
            row_start = base_row_start + work_q_begin
            count_raw = work_q_count
            kv_valid_cols = self._valid_cols_in_block(
                row_batch_idx,
                row_kv_block_idx,
                mPageTable,
                mSeqUsedK,
                mCuSeqlensK,
            )
            q_batch_offset = self._batch_q_offset(row_batch_idx, mCuSeqlensQ)
            k_batch_offset = (
                Int32(0) if const_expr(self.paged_kv) else mCuSeqlensK[row_batch_idx]
            )
            sRowMeta[0] = row_batch_idx
            sRowMeta[1] = row_kv_block_idx
            sRowMeta[2] = row_start
            sRowMeta[3] = count_raw
            sRowMeta[4] = kv_valid_cols
            sRowMeta[5] = q_batch_offset
            sRowMeta[6] = k_batch_offset
            causal_q_offset = Int32(0)
            if const_expr(self.causal):
                seqlen_q = mCuSeqlensQ[row_batch_idx + Int32(1)] - q_batch_offset
                seqlen_k = self._logical_seqlen_k(
                    row_batch_idx,
                    mPageTable,
                    mSeqUsedK,
                    mCuSeqlensK,
                )
                causal_q_offset = seqlen_k - seqlen_q
            sRowMeta[7] = causal_q_offset
            if const_expr(self.paged_kv):
                if const_expr(self.paged_block_tma):
                    sPagedKvIdx[0] = paged_kv_manager.physical_block_index(
                        row_batch_idx, row_kv_block_idx
                    )
                else:
                    for seg_idx_c in cutlass.range_constexpr(
                        self.kv_segments_per_block
                    ):
                        seg_idx = Int32(seg_idx_c)
                        sPagedKvIdx[seg_idx] = paged_kv_manager.physical_page_index(
                            row_batch_idx, row_kv_block_idx, seg_idx
                        )
            cute.arch.mbarrier_init(mbar_k_ptr, 1)
            cute.arch.mbarrier_expect_tx(mbar_k_ptr, k_tma_bytes)
            cute.arch.mbarrier_init(mbar_v_ptr, 1)
            cute.arch.mbarrier_expect_tx(mbar_v_ptr, v_tma_bytes)
            diag_q_count = Int32(0)
            row_has_work = (count_raw > Int32(0)) & (kv_valid_cols > Int32(0))
            if const_expr(self.causal):
                if row_has_work:
                    q_block_end = (row_kv_block_idx + Int32(1)) * Int32(
                        self.n_block_size
                    )
                    q_threshold = q_block_end - causal_q_offset
                    diag_q_count = self._lower_bound_q_idx(
                        mK2qIndices,
                        head_kv_idx,
                        row_start,
                        count_raw,
                        q_threshold,
                    )
                    diag_q_count = cutlass.min(
                        diag_q_count,
                        Int32(self.n_block_size),
                    )
            sDiagQCount[0] = diag_q_count
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_base = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tStS = cute.make_tensor(
            tStS_base.iterator,
            cute.append(
                tStS_base.layout,
                cute.make_layout((self.s_stage,), stride=(self.tmem_stage_stride,)),
            ),
        )
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        tP_width_ratio = Float32.width // self.v_dtype.width
        tP_stage_stride = self.tmem_stage_stride * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        tmem_cols = self.tmem_total

        load_wg_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.LoadWG),
            num_threads=cute.arch.WARP_SIZE * self.num_q_load_warps,
        )
        sm_stats_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        epilogue_barrier = pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.StoreEpilogue),
            num_threads=cute.arch.WARP_SIZE * self.warps_per_group,
        )
        if warp_idx == Int32(self.total_warps - 1):
            cute.arch.setmaxregister_decrease(self.num_regs_empty)

        q_load_thread_base = Int32(self.q_load_warp_base * cute.arch.WARP_SIZE)
        q_load_thread_end = Int32(
            (self.q_load_warp_base + self.num_q_load_warps) * cute.arch.WARP_SIZE
        )
        is_q_load_thread = tidx >= q_load_thread_base and tidx < q_load_thread_end
        if is_q_load_thread and cta_valid_work:
            if self.store_reg_decrease:
                cute.arch.setmaxregister_decrease(self.num_regs_store)
            else:
                cute.arch.setmaxregister_increase(self.num_regs_store)
            row_start_load = sRowMeta[2]
            count_raw_load = sRowMeta[3]
            kv_valid_cols_load = sRowMeta[4]
            q_batch_offset_load = sRowMeta[5]
            has_work_load = count_raw_load > Int32(0)
            has_work_load = has_work_load & (kv_valid_cols_load > Int32(0))
            num_q_groups_load = (
                count_raw_load + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            if const_expr(self.use_q_gather4):
                self._wg_load_q_gather4(
                    mQ_2d,
                    mQ_gather4_desc,
                    mK2qQSplitIndices,
                    sQIdxMeta,
                    sQ,
                    pipeline_q,
                    load_wg_barrier,
                    num_q_groups_load,
                    count_raw_load,
                    has_work_load,
                    head_kv_idx,
                    row_start_load,
                    q_batch_offset_load,
                    num_heads_kv,
                )
            else:
                self._wg_load_q_tma(
                    tma_atom_Q,
                    mQ_2d,
                    mK2qQSplitIndices,
                    sQLoadMIdx,
                    sQIdxMeta,
                    sQ_load,
                    pipeline_q,
                    load_wg_barrier,
                    num_q_groups_load,
                    count_raw_load,
                    has_work_load,
                    head_kv_idx,
                    row_start_load,
                    q_batch_offset_load,
                    num_heads_kv,
                )

        if (
            warp_idx >= Int32(self.kv_load_warp_base)
            and warp_idx < Int32(self.kv_load_warp_base + self.num_kv_load_warps)
            and cta_valid_work
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            kv_block_idx_load = sRowMeta[1]
            kv_valid_cols_load = sRowMeta[4]
            k_batch_offset_load = sRowMeta[6]
            has_work_load = sRowMeta[3] > Int32(0)
            has_work_load = has_work_load & (kv_valid_cols_load > Int32(0))
            self._wg_load_kv(
                tma_atom_K,
                tma_atom_V,
                tma_K,
                tma_V,
                mK_raw_desc,
                mV_raw_desc,
                sPagedKvIdx,
                sK,
                sV,
                tiled_mma_qk,
                tiled_mma_pv,
                mbar_k_ptr,
                mbar_v_ptr,
                has_work_load,
                head_kv_idx,
                kv_block_idx_load,
                k_batch_offset_load,
                kv_valid_cols_load,
            )

        if warp_idx == Int32(self.mma_warp_id) and cta_valid_work:
            cute.arch.setmaxregister_decrease(self.num_regs_mma)
            count_raw_mma = sRowMeta[3]
            kv_valid_cols_mma = sRowMeta[4]
            has_work_mma = count_raw_mma > Int32(0)
            has_work_mma = has_work_mma & (kv_valid_cols_mma > Int32(0))
            num_q_groups_mma = (
                count_raw_mma + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            tmem.allocate(tmem_cols)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self._wg_mma_issue(
                tiled_mma_qk,
                tiled_mma_pv,
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOrP,
                sK,
                sV,
                sQ,
                pipeline_q,
                pipeline_s,
                pipeline_p,
                pipeline_o,
                pipeline_sm_stats,
                mbar_k_ptr,
                mbar_v_ptr,
                num_q_groups_mma,
                has_work_mma,
            )
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr, num_columns=tmem_cols)
            cute.arch.griddepcontrol_launch_dependents()

        if (
            warp_idx >= Int32(self.softmax0_warp_base)
            and warp_idx < Int32(self.softmax1_warp_base)
            and cta_valid_work
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            kv_block_idx_softmax = sRowMeta[1]
            count_raw_softmax = sRowMeta[3]
            kv_valid_cols_softmax = sRowMeta[4]
            causal_q_offset_softmax = sRowMeta[7]
            has_work_softmax = count_raw_softmax > Int32(0)
            has_work_softmax = has_work_softmax & (kv_valid_cols_softmax > Int32(0))
            num_q_groups_softmax = (
                count_raw_softmax + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            diag_q_count_softmax = sDiagQCount[0]
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self._wg_softmax(
                0,
                tiled_mma_qk,
                tiled_mma_pv,
                tStS,
                sScale,
                sScaleTemperature,
                sSplitIdx,
                sQIdx,
                sQIdxMeta,
                pipeline_s,
                pipeline_p,
                pipeline_o,
                pipeline_sm_stats,
                sm_stats_barrier,
                epilogue_barrier,
                mO_partial,
                mLSE_partial,
                mLSE_temperature_partial,
                softmax_scale_log2,
                lse_temperature_scale_log2,
                lse_temperature_scale,
                kv_block_idx_softmax,
                kv_valid_cols_softmax,
                diag_q_count_softmax,
                num_q_groups_softmax,
                count_raw_softmax,
                has_work_softmax,
                causal_q_offset_softmax,
                sRowMeta[0],
                head_kv_idx,
                seq_len_q,
                head_q,
                num_heads_kv,
                sRowMeta[5],
                mQ_2d,
            )
            tmem_alloc_barrier.arrive()

        if (
            warp_idx >= Int32(self.softmax1_warp_base)
            and warp_idx < Int32(self.store_warp_base)
            and cta_valid_work
        ):
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            kv_block_idx_softmax = sRowMeta[1]
            count_raw_softmax = sRowMeta[3]
            kv_valid_cols_softmax = sRowMeta[4]
            causal_q_offset_softmax = sRowMeta[7]
            has_work_softmax = count_raw_softmax > Int32(0)
            has_work_softmax = has_work_softmax & (kv_valid_cols_softmax > Int32(0))
            num_q_groups_softmax = (
                count_raw_softmax + Int32(self.q_tokens_per_group - 1)
            ) // Int32(self.q_tokens_per_group)
            diag_q_count_softmax = sDiagQCount[0]
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self._wg_softmax(
                1,
                tiled_mma_qk,
                tiled_mma_pv,
                tStS,
                sScale,
                sScaleTemperature,
                sSplitIdx,
                sQIdx,
                sQIdxMeta,
                pipeline_s,
                pipeline_p,
                pipeline_o,
                pipeline_sm_stats,
                sm_stats_barrier,
                epilogue_barrier,
                mO_partial,
                mLSE_partial,
                mLSE_temperature_partial,
                softmax_scale_log2,
                lse_temperature_scale_log2,
                lse_temperature_scale,
                kv_block_idx_softmax,
                kv_valid_cols_softmax,
                diag_q_count_softmax,
                num_q_groups_softmax,
                count_raw_softmax,
                has_work_softmax,
                causal_q_offset_softmax,
                sRowMeta[0],
                head_kv_idx,
                seq_len_q,
                head_q,
                num_heads_kv,
                sRowMeta[5],
                mQ_2d,
            )
            tmem_alloc_barrier.arrive()

    # ------------------------------------------------------------------
    # Warp-specialized helpers
    # ------------------------------------------------------------------

    @cute.jit
    def _wg_load_kv(
        self,
        tma_atom_K,
        tma_atom_V,
        tma_K: cute.Tensor,
        tma_V: cute.Tensor,
        mK_raw_desc,
        mV_raw_desc,
        sPagedKvIdx: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mbar_k_ptr,
        mbar_v_ptr,
        has_work: Int32,
        head_kv_idx: Int32,
        kv_block_idx: Int32,
        k_batch_offset: Int32,
        kv_valid_cols: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx - Int32(self.kv_load_warp_base)
        valid_segments = (
            (kv_valid_cols + Int32(self.kv_segment_rows - 1))
            // Int32(self.kv_segment_rows)
            if const_expr(self.paged_segment_tma)
            else Int32(1)
        )

        if has_work:
            if warp_idx_in_wg == Int32(0):
                if const_expr(self.use_raw_kv_desc):
                    desc_ptr = mK_raw_desc.iterator + head_kv_idx * Int32(128)
                    sK_ptr = sK.iterator
                    seg_byte_stride = Int32(self.kv_segment_rows * self.k_tile * 2)
                    half_tile_stage_bytes = Int32(self.n_block_size * self.k_tile * 2)
                    with cute.arch.elect_one():
                        if const_expr(self.paged_segment_tma):
                            for seg_idx_c in cutlass.range_constexpr(
                                self.kv_segments_per_block
                            ):
                                seg_idx = Int32(seg_idx_c)
                                if seg_idx < valid_segments:
                                    row_idx = sPagedKvIdx[seg_idx] * Int32(
                                        self.page_size
                                    )
                                    seg_byte_off = seg_idx * seg_byte_stride
                                    for ks_c in cutlass.range_constexpr(self.k_stages):
                                        ks = Int32(ks_c)
                                        if const_expr(ks_c + 1 < self.k_stages):
                                            tma_tile_prefetch(
                                                desc_ptr,
                                                Int32((ks_c + 1) * self.k_tile),
                                                row_idx,
                                                TMA_CACHE_EVICT_FIRST,
                                            )
                                        tma_tile_load_cached(
                                            sK_ptr,
                                            ks * half_tile_stage_bytes + seg_byte_off,
                                            desc_ptr,
                                            Int32(ks_c * self.k_tile),
                                            row_idx,
                                            mbar_k_ptr,
                                            TMA_CACHE_EVICT_FIRST,
                                        )
                        else:
                            row_idx = k_batch_offset + kv_block_idx * Int32(
                                self.n_block_size
                            )
                            for ks in cutlass.range_constexpr(self.k_stages):
                                tma_tile_load(
                                    sK_ptr,
                                    ks * half_tile_stage_bytes,
                                    desc_ptr,
                                    Int32(ks * self.k_tile),
                                    row_idx,
                                    mbar_k_ptr,
                                )
                        cute.arch.mbarrier_arrive(mbar_k_ptr)
                else:
                    thr_mma_qk = tiled_mma_qk.get_slice(0)
                    if const_expr(self.paged_kv):
                        mK_cur = tma_K[None, None, head_kv_idx, None]
                        gK = cute.local_tile(
                            mK_cur,
                            cute.select(self.mma_tiler_qk, mode=[1, 2]),
                            (None, 0, None),
                        )
                    else:
                        mK_cur = cute.domain_offset(
                            (k_batch_offset, 0),
                            tma_K[None, None, head_kv_idx],
                        )
                        gK = cute.local_tile(
                            mK_cur,
                            cute.select(self.mma_tiler_qk, mode=[1, 2]),
                            (None, 0),
                        )
                    tSgK = thr_mma_qk.partition_B(gK)
                    tKsK, tKgK = cpasync.tma_partition(
                        tma_atom_K,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sK, 0, 3),
                        cute.group_modes(tSgK, 0, 3),
                    )
                    gmem_k_idx = (
                        sPagedKvIdx[0] if const_expr(self.paged_kv) else kv_block_idx
                    )
                    cute.copy(
                        tma_atom_K,
                        tKgK[(None, 0, gmem_k_idx)]
                        if const_expr(self.paged_kv)
                        else tKgK[(None, gmem_k_idx)],
                        tKsK[(None, 0)],
                        tma_bar_ptr=mbar_k_ptr,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_k_ptr)

            if warp_idx_in_wg == Int32(1):
                if const_expr(self.use_raw_kv_desc):
                    desc_ptr = mV_raw_desc.iterator + head_kv_idx * Int32(128)
                    sV_ptr = sV.iterator
                    seg_byte_stride = Int32(self.kv_segment_rows * self.k_tile * 2)
                    half_tile_stage_bytes = Int32(self.n_block_size * self.k_tile * 2)
                    with cute.arch.elect_one():
                        if const_expr(self.paged_segment_tma):
                            for seg_idx_c in cutlass.range_constexpr(
                                self.kv_segments_per_block
                            ):
                                seg_idx = Int32(seg_idx_c)
                                if seg_idx < valid_segments:
                                    row_idx = sPagedKvIdx[seg_idx] * Int32(
                                        self.page_size
                                    )
                                    seg_byte_off = seg_idx * seg_byte_stride
                                    for ks_c in cutlass.range_constexpr(self.k_stages):
                                        ks = Int32(ks_c)
                                        if const_expr(ks_c + 1 < self.k_stages):
                                            tma_tile_prefetch(
                                                desc_ptr,
                                                Int32((ks_c + 1) * self.k_tile),
                                                row_idx,
                                                TMA_CACHE_EVICT_FIRST,
                                            )
                                        tma_tile_load_cached(
                                            sV_ptr,
                                            ks * half_tile_stage_bytes + seg_byte_off,
                                            desc_ptr,
                                            Int32(ks_c * self.k_tile),
                                            row_idx,
                                            mbar_v_ptr,
                                            TMA_CACHE_EVICT_FIRST,
                                        )
                        else:
                            row_idx = k_batch_offset + kv_block_idx * Int32(
                                self.n_block_size
                            )
                            for ks in cutlass.range_constexpr(self.k_stages):
                                tma_tile_load(
                                    sV_ptr,
                                    ks * half_tile_stage_bytes,
                                    desc_ptr,
                                    Int32(ks * self.k_tile),
                                    row_idx,
                                    mbar_v_ptr,
                                )
                        cute.arch.mbarrier_arrive(mbar_v_ptr)
                else:
                    thr_mma_pv = tiled_mma_pv.get_slice(0)
                    if const_expr(self.paged_kv):
                        mV_cur = tma_V[None, None, head_kv_idx, None]
                        gV = cute.local_tile(
                            mV_cur,
                            cute.select(self.mma_tiler_pv, mode=[1, 2]),
                            (0, None, None),
                        )
                    else:
                        mV_cur = cute.domain_offset(
                            (0, k_batch_offset),
                            tma_V[None, None, head_kv_idx],
                        )
                        gV = cute.local_tile(
                            mV_cur,
                            cute.select(self.mma_tiler_pv, mode=[1, 2]),
                            (0, None),
                        )
                    tOgV = thr_mma_pv.partition_B(gV)
                    tVsV, tVgV = cpasync.tma_partition(
                        tma_atom_V,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sV, 0, 3),
                        cute.group_modes(tOgV, 0, 3),
                    )
                    gmem_v_idx = (
                        sPagedKvIdx[0] if const_expr(self.paged_kv) else kv_block_idx
                    )
                    cute.copy(
                        tma_atom_V,
                        tVgV[(None, 0, gmem_v_idx)]
                        if const_expr(self.paged_kv)
                        else tVgV[(None, gmem_v_idx)],
                        tVsV[(None, 0)],
                        tma_bar_ptr=mbar_v_ptr,
                    )
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(mbar_v_ptr)

    @cute.jit
    def _wg_load_q_gather4(
        self,
        mQ_2d: cute.Tensor,
        mQ_gather4_desc: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        sQ: cute.Tensor,
        pipeline_q,
        load_wg_barrier,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        head_kv_idx: Int32,
        row_start: Int32,
        q_batch_offset: Int32,
        num_heads_kv: Int32,
    ):
        tidx = cute.arch.thread_idx()[0]
        q_load_thread_base = Int32(self.q_load_warp_base * cute.arch.WARP_SIZE)
        group_tidx = tidx - q_load_thread_base
        producer_warp_idx_in_wg = cute.arch.make_warp_uniform(
            group_tidx // Int32(cute.arch.WARP_SIZE)
        )
        q_oob_m_idx = mQ_2d.shape[0] // Int32(self.qheadperkv)
        gathers_per_warp: cutlass.Constexpr[int] = self.m_block_size // (
            self.num_q_load_warps * 4
        )

        if has_work:
            for qi_group in cutlass.range(num_q_groups, unroll=1):
                slot = qi_group % Int32(self.q_stage)
                phase = (qi_group // Int32(self.q_stage)) & Int32(1)
                producer_phase = phase ^ Int32(1)
                if producer_warp_idx_in_wg == Int32(0):
                    pipeline_q.producer_acquire_w_index_phase(slot, producer_phase)
                load_wg_barrier.arrive_and_wait()

                group_tidx = tidx - q_load_thread_base
                warp_idx_in_wg = cute.arch.make_warp_uniform(
                    group_tidx // Int32(cute.arch.WARP_SIZE)
                )
                lane_idx = group_tidx % Int32(cute.arch.WARP_SIZE)
                mbar_ptr = pipeline_q.sync_object_full.get_barrier(slot)
                qidx_meta_slot = (qi_group & Int32(self.qidx_meta_stages - 1)) * Int32(
                    self.q_tokens_per_group
                )

                meta_iters: cutlass.Constexpr[int] = (
                    self.q_tokens_per_group
                    + self.num_q_load_warps * cute.arch.WARP_SIZE
                    - 1
                ) // (self.num_q_load_warps * cute.arch.WARP_SIZE)
                for meta_iter in cutlass.range_constexpr(meta_iters):
                    tok_idx_g4 = (
                        Int32(meta_iter) * Int32(self.num_q_load_warps) + warp_idx_in_wg
                    ) * Int32(cute.arch.WARP_SIZE) + lane_idx
                    if tok_idx_g4 < Int32(self.q_tokens_per_group):
                        qi = qi_group * Int32(self.q_tokens_per_group) + tok_idx_g4
                        if qi < count_raw:
                            sQIdxMeta[qidx_meta_slot + tok_idx_g4] = (
                                self._load_qsplit_idx(
                                    mK2qQSplitIndices, head_kv_idx, row_start, qi
                                )
                            )
                        else:
                            sQIdxMeta[qidx_meta_slot + tok_idx_g4] = Int32(0)
                load_wg_barrier.arrive_and_wait()

                with cute.arch.elect_one():
                    q_desc_ptr = mQ_gather4_desc.iterator
                    sQ_ptr = sQ.iterator
                    for gather_slot in cutlass.range_constexpr(gathers_per_warp):
                        gather_idx = (
                            Int32(gather_slot) * Int32(self.num_q_load_warps)
                            + warp_idx_in_wg
                        )
                        tok_base = gather_idx * Int32(self.tokens_per_gather4)
                        if const_expr(self.qheadperkv == 1):
                            qi0 = qi_group * Int32(self.q_tokens_per_group) + tok_base
                            qi1 = qi0 + Int32(1)
                            qi2 = qi0 + Int32(2)
                            qi3 = qi0 + Int32(3)
                            row0 = q_oob_m_idx
                            row1 = q_oob_m_idx
                            row2 = q_oob_m_idx
                            row3 = q_oob_m_idx
                            if qi0 < count_raw:
                                q_idx0 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base]
                                )
                                row0 = (
                                    q_batch_offset + q_idx0
                                ) * num_heads_kv + head_kv_idx
                            if qi1 < count_raw:
                                q_idx1 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base + Int32(1)]
                                )
                                row1 = (
                                    q_batch_offset + q_idx1
                                ) * num_heads_kv + head_kv_idx
                            if qi2 < count_raw:
                                q_idx2 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base + Int32(2)]
                                )
                                row2 = (
                                    q_batch_offset + q_idx2
                                ) * num_heads_kv + head_kv_idx
                            if qi3 < count_raw:
                                q_idx3 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base + Int32(3)]
                                )
                                row3 = (
                                    q_batch_offset + q_idx3
                                ) * num_heads_kv + head_kv_idx
                        elif const_expr(self.qheadperkv == 2):
                            qi0 = qi_group * Int32(self.q_tokens_per_group) + tok_base
                            qi1 = qi0 + Int32(1)
                            row_base0 = q_oob_m_idx * Int32(self.qheadperkv)
                            row_base1 = q_oob_m_idx * Int32(self.qheadperkv)
                            if qi0 < count_raw:
                                q_idx0 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base]
                                )
                                row_base0 = (
                                    (q_batch_offset + q_idx0) * num_heads_kv
                                    + head_kv_idx
                                ) * Int32(self.qheadperkv)
                            if qi1 < count_raw:
                                q_idx1 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base + Int32(1)]
                                )
                                row_base1 = (
                                    (q_batch_offset + q_idx1) * num_heads_kv
                                    + head_kv_idx
                                ) * Int32(self.qheadperkv)
                            row0 = row_base0
                            row1 = row_base0 + Int32(1)
                            row2 = row_base1
                            row3 = row_base1 + Int32(1)
                        else:
                            qi0 = qi_group * Int32(self.q_tokens_per_group) + tok_base
                            row_base0 = q_oob_m_idx * Int32(self.qheadperkv)
                            if qi0 < count_raw:
                                q_idx0 = self._decode_q_idx_from_qsplit(
                                    sQIdxMeta[qidx_meta_slot + tok_base]
                                )
                                row_base0 = (
                                    (q_batch_offset + q_idx0) * num_heads_kv
                                    + head_kv_idx
                                ) * Int32(self.qheadperkv)
                            row0 = row_base0
                            row1 = row_base0 + Int32(1)
                            row2 = row_base0 + Int32(2)
                            row3 = row_base0 + Int32(3)
                        group_byte_off = gather_idx * Int32(4 * self.k_tile * 2)
                        for ks_c in cutlass.range_constexpr(self.k_stages):
                            stage_idx = slot * Int32(self.k_stages) + Int32(ks_c)
                            stage_byte_off = stage_idx * Int32(self.k_tile_stride_bytes)
                            if const_expr(ks_c + 1 < self.k_stages):
                                tma_gather4_prefetch(
                                    q_desc_ptr,
                                    Int32((ks_c + 1) * self.k_tile),
                                    row0,
                                    row1,
                                    row2,
                                    row3,
                                    TMA_CACHE_EVICT_LAST,
                                )
                            tma_gather4_cached(
                                sQ_ptr,
                                stage_byte_off + group_byte_off,
                                q_desc_ptr,
                                Int32(ks_c * self.k_tile),
                                row0,
                                row1,
                                row2,
                                row3,
                                mbar_ptr,
                                TMA_CACHE_EVICT_LAST,
                            )
                load_wg_barrier.arrive_and_wait()

            if producer_warp_idx_in_wg == Int32(0):
                next_slot = num_q_groups % Int32(self.q_stage)
                next_phase = ((num_q_groups // Int32(self.q_stage)) & Int32(1)) ^ Int32(
                    1
                )
                pipeline_q.producer_acquire_w_index_phase(next_slot, next_phase)

    @cute.jit
    def _wg_load_q_tma(
        self,
        tma_atom_Q,
        mQ_2d: cute.Tensor,
        mK2qQSplitIndices: cute.Tensor,
        sQLoadMIdx: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        sQ_load: cute.Tensor,
        pipeline_q,
        load_wg_barrier,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        head_kv_idx: Int32,
        row_start: Int32,
        q_batch_offset: Int32,
        num_heads_kv: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        warp_idx_in_wg = warp_idx - Int32(self.q_load_warp_base)
        gQ_k0 = cute.local_tile(mQ_2d, (self.qheadperkv, self.k_tile), (None, 0))
        gQ_k1 = cute.local_tile(mQ_2d, (self.qheadperkv, self.k_tile), (None, 1))
        load_Q_fn_k0, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Q, 0, cute.make_layout(1), gQ_k0, sQ_load
        )
        load_Q_fn_k1, _, _ = copy_utils.tma_get_copy_fn(
            tma_atom_Q, 0, cute.make_layout(1), gQ_k1, sQ_load
        )
        q_oob_m_idx = mQ_2d.shape[0] // Int32(self.qheadperkv)
        tokens_per_warp: cutlass.Constexpr[int] = (
            self.q_tokens_per_group + self.num_q_load_warps - 1
        ) // self.num_q_load_warps

        if has_work:
            for qi_group in cutlass.range(num_q_groups, unroll=1):
                slot = qi_group % Int32(self.q_stage)
                phase = (qi_group // Int32(self.q_stage)) & Int32(1)
                producer_phase = phase ^ Int32(1)
                if warp_idx_in_wg == Int32(0):
                    pipeline_q.producer_acquire_w_index_phase(slot, producer_phase)
                load_wg_barrier.arrive_and_wait()

                mbar_ptr = pipeline_q.sync_object_full.get_barrier(slot)
                sub_stage_base = slot * Int32(self.q_tokens_per_group * self.k_stages)
                load_meta_slot = slot * Int32(self.q_tokens_per_group)
                qidx_meta_slot = (qi_group & Int32(self.qidx_meta_stages - 1)) * Int32(
                    self.q_tokens_per_group
                )

                if warp_idx_in_wg == Int32(0) and lane_idx < Int32(
                    self.q_tokens_per_group
                ):
                    tok_idx = lane_idx
                    qi = qi_group * Int32(self.q_tokens_per_group) + tok_idx
                    if qi < count_raw:
                        qsplit = self._load_qsplit_idx(
                            mK2qQSplitIndices, head_kv_idx, row_start, qi
                        )
                        q_idx = self._decode_q_idx_from_qsplit(qsplit)
                        q_abs = q_batch_offset + q_idx
                        sQIdxMeta[qidx_meta_slot + tok_idx] = qsplit
                        sQLoadMIdx[load_meta_slot + tok_idx] = (
                            q_abs * num_heads_kv + head_kv_idx
                        )
                    else:
                        sQIdxMeta[qidx_meta_slot + tok_idx] = Int32(0)
                        sQLoadMIdx[load_meta_slot + tok_idx] = q_oob_m_idx
                load_wg_barrier.arrive_and_wait()

                for qi_slot in cutlass.range_constexpr(tokens_per_warp):
                    tok_idx = warp_idx_in_wg * Int32(tokens_per_warp) + Int32(qi_slot)
                    if tok_idx < Int32(self.q_tokens_per_group):
                        m_tile_idx = sQLoadMIdx[load_meta_slot + tok_idx]
                        load_Q_fn_k0(
                            src_idx=m_tile_idx,
                            dst_idx=sub_stage_base + tok_idx,
                            tma_bar_ptr=mbar_ptr,
                        )
                        load_Q_fn_k1(
                            src_idx=m_tile_idx,
                            dst_idx=(
                                sub_stage_base
                                + Int32(self.q_tokens_per_group)
                                + tok_idx
                            ),
                            tma_bar_ptr=mbar_ptr,
                        )

            if warp_idx_in_wg == Int32(0):
                next_slot = num_q_groups % Int32(self.q_stage)
                next_phase = ((num_q_groups // Int32(self.q_stage)) & Int32(1)) ^ Int32(
                    1
                )
                pipeline_q.producer_acquire_w_index_phase(next_slot, next_phase)

    @cute.jit
    def _wg_mma_issue(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        thr0_qk: cute.core.ThrMma,
        thr0_pv: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOrP: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sQ: cute.Tensor,
        pipeline_q,
        pipeline_s,
        pipeline_p,
        pipeline_o,
        pipeline_sm_stats,
        mbar_k_ptr,
        mbar_v_ptr,
        num_q_groups: Int32,
        has_work: Int32,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_mma_warp = warp_idx == Int32(self.mma_warp_id)

        if is_mma_warp:
            if has_work:
                tSrQ = tiled_mma_qk.make_fragment_A(sQ)
                tSrK = tiled_mma_qk.make_fragment_B(sK)
                tSrQ0 = tSrQ[None, None, None, 0]
                tSrK0 = tSrK[None, None, None, 0]
                tOrV = tiled_mma_pv.make_fragment_B(sV)
                tOrV0 = tOrV[None, None, None, 0]
                sV0 = sV[None, None, None, 0]
                pv_mma_op = tiled_mma_pv.op
                qk_mma_op = tiled_mma_qk.op
                q_smem_base = sm100_desc.smem_desc_base_from_tensor(
                    sQ, sm100_desc.Major.K
                )
                k_smem_base = sm100_desc.smem_desc_base_from_tensor(
                    sK, sm100_desc.Major.K
                )
                k_smem_start = sm100_desc.make_smem_desc_start_addr(
                    sK[None, None, None, 0].iterator
                )
                q_smem_start = sm100_desc.make_smem_desc_start_addr(
                    sQ[None, None, None, self.q_stage - 1].iterator
                )
                sm100_helpers.declare_ptx_smem_desc(
                    q_smem_start,
                    q_smem_base,
                    tSrQ0.layout,
                    var_name_prefix="lean_q_desc",
                )
                sm100_helpers.declare_ptx_idesc(qk_mma_op, var_name="lean_qk_idesc")
                sQ_stage_stride = (
                    sQ.layout.stride[-1] * sQ.element_type.width // 8
                ) >> 4
                if const_expr(self.q_stage == 1):
                    sQ_stage_stride = 0
                q_wrap_offset = -(self.q_stage - 1) * sQ_stage_stride
                q_advance_offset = sQ_stage_stride
                gemm_qk_s0_wrap = partial(
                    sm100_helpers.gemm_ptx_precomputed_varname,
                    Int32(self.tmem_s_offset),
                    smem_desc_base_b=k_smem_base,
                    tCrB_layout=tSrK0.layout,
                    smem_var_name_prefix="lean_q_desc",
                    idesc_var_name="lean_qk_idesc",
                    smem_offset=q_wrap_offset,
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                gemm_qk_s0_advance = partial(
                    sm100_helpers.gemm_ptx_precomputed_varname,
                    Int32(self.tmem_s_offset),
                    smem_desc_base_b=k_smem_base,
                    tCrB_layout=tSrK0.layout,
                    smem_var_name_prefix="lean_q_desc",
                    idesc_var_name="lean_qk_idesc",
                    smem_offset=q_advance_offset,
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                gemm_qk_s1_wrap = partial(
                    sm100_helpers.gemm_ptx_precomputed_varname,
                    Int32(self.tmem_stage_stride + self.tmem_s_offset),
                    smem_desc_base_b=k_smem_base,
                    tCrB_layout=tSrK0.layout,
                    smem_var_name_prefix="lean_q_desc",
                    idesc_var_name="lean_qk_idesc",
                    smem_offset=q_wrap_offset,
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                gemm_qk_s1_advance = partial(
                    sm100_helpers.gemm_ptx_precomputed_varname,
                    Int32(self.tmem_stage_stride + self.tmem_s_offset),
                    smem_desc_base_b=k_smem_base,
                    tCrB_layout=tSrK0.layout,
                    smem_var_name_prefix="lean_q_desc",
                    idesc_var_name="lean_qk_idesc",
                    smem_offset=q_advance_offset,
                    zero_init=True,
                    cta_group=self.cta_group_size,
                )
                gemm_pv_0 = partial(
                    sm100_helpers.gemm_ptx_partial,
                    pv_mma_op,
                    Int32(self.tmem_o_offset),
                    tOrP[None, None, None, 0],
                    sA=None,
                    split_arrive=(
                        self.split_P_arrive if self.split_P_arrive > 0 else None
                    ),
                    tA_addr=Int32(self.tmem_p_offset),
                    cta_group=self.cta_group_size,
                )
                gemm_pv_1 = partial(
                    sm100_helpers.gemm_ptx_partial,
                    pv_mma_op,
                    Int32(self.tmem_o_offset + self.tmem_o_stage_stride),
                    tOrP[None, None, None, 1],
                    sA=None,
                    split_arrive=(
                        self.split_P_arrive if self.split_P_arrive > 0 else None
                    ),
                    tA_addr=Int32(self.tmem_stage_stride + self.tmem_p_offset),
                    cta_group=self.cta_group_size,
                )

                cute.arch.mbarrier_wait(mbar_k_ptr, 0)
                # Issue order:
                #   Q0K, Q1K, P0V, Q2K, P1V, Q3K, ...
                # This reuses each slot as soon as its previous PV drains,
                # instead of batching both PVs after both QKs of a pair.
                # The schedule is still 2-slot safe:
                #   - QK(qi) consumes slot qi&1
                #   - PV(qi-2) frees the same slot before QK(qi) reuses it
                #   - phases still toggle every 2 groups per slot

                # Prologue: issue up to the first two QK tiles. Q slots come
                # from the q_stage ring; S slots remain a 2-slot ring.
                pipeline_q.consumer_wait_w_index_phase(Int32(0), Int32(0))
                pipeline_s.producer_acquire_w_index_phase(Int32(0), Int32(1))
                gemm_qk_s0_wrap(smem_desc_start_b=k_smem_start)
                pipeline_s.producer_commit_w_index(Int32(0))
                pipeline_q.consumer_release_w_index(Int32(0))

                if num_q_groups > Int32(1):
                    pipeline_q.consumer_wait_w_index_phase(Int32(1), Int32(0))
                    pipeline_s.producer_acquire_w_index_phase(Int32(1), Int32(1))
                    gemm_qk_s1_advance(smem_desc_start_b=k_smem_start)
                    pipeline_s.producer_commit_w_index(Int32(1))
                    pipeline_q.consumer_release_w_index(Int32(1))

                cute.arch.mbarrier_wait(mbar_v_ptr, 0)

                # Steady-state: for qi >= 2, drain PV(qi-2) before reusing
                # that slot for QK(qi).
                for qi in cutlass.range(Int32(2), num_q_groups, unroll=1):
                    pv_qi = qi - Int32(2)
                    pv_slot = pv_qi & Int32(1)
                    pv_phase = (pv_qi // Int32(2)) & Int32(1)
                    pipeline_p.consumer_wait_w_index_phase(pv_slot, pv_phase)
                    pipeline_o.producer_acquire_w_index_phase(
                        pv_slot, pv_phase ^ Int32(1)
                    )
                    if pv_slot == Int32(0):
                        gemm_pv_0(
                            tCrB=tOrV0,
                            sB=sV0,
                            mbar_ptr=(
                                pipeline_sm_stats.sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0
                                else None
                            ),
                            mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                            zero_init=True,
                        )
                    else:
                        gemm_pv_1(
                            tCrB=tOrV0,
                            sB=sV0,
                            mbar_ptr=(
                                pipeline_sm_stats.sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0
                                else None
                            ),
                            mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                            zero_init=True,
                        )
                    pipeline_o.producer_commit_w_index(pv_slot)
                    pipeline_p.consumer_release_w_index(pv_slot)

                    q_slot = qi % Int32(self.q_stage)
                    q_phase = (qi // Int32(self.q_stage)) & Int32(1)
                    s_slot = qi & Int32(1)
                    s_phase = (qi // Int32(2)) & Int32(1)
                    pipeline_q.consumer_wait_w_index_phase(q_slot, q_phase)
                    pipeline_s.producer_acquire_w_index_phase(
                        s_slot, s_phase ^ Int32(1)
                    )
                    if s_slot == Int32(0):
                        if q_slot == Int32(0):
                            gemm_qk_s0_wrap(smem_desc_start_b=k_smem_start)
                        else:
                            gemm_qk_s0_advance(smem_desc_start_b=k_smem_start)
                    else:
                        if q_slot == Int32(0):
                            gemm_qk_s1_wrap(smem_desc_start_b=k_smem_start)
                        else:
                            gemm_qk_s1_advance(smem_desc_start_b=k_smem_start)
                    pipeline_s.producer_commit_w_index(s_slot)
                    pipeline_q.consumer_release_w_index(q_slot)

                # Drain the remaining one or two PV tiles.
                drain_begin = (
                    Int32(0) if num_q_groups == Int32(1) else num_q_groups - Int32(2)
                )
                for pv_qi in cutlass.range(drain_begin, num_q_groups, unroll=1):
                    pv_slot = pv_qi & Int32(1)
                    pv_phase = (pv_qi // Int32(2)) & Int32(1)
                    pipeline_p.consumer_wait_w_index_phase(pv_slot, pv_phase)
                    pipeline_o.producer_acquire_w_index_phase(
                        pv_slot, pv_phase ^ Int32(1)
                    )
                    if pv_slot == Int32(0):
                        gemm_pv_0(
                            tCrB=tOrV0,
                            sB=sV0,
                            mbar_ptr=(
                                pipeline_sm_stats.sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0
                                else None
                            ),
                            mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                            zero_init=True,
                        )
                    else:
                        gemm_pv_1(
                            tCrB=tOrV0,
                            sB=sV0,
                            mbar_ptr=(
                                pipeline_sm_stats.sync_object_full.get_barrier(pv_slot)
                                if self.split_P_arrive > 0
                                else None
                            ),
                            mbar_phase=(pv_phase if self.split_P_arrive > 0 else None),
                            zero_init=True,
                        )
                    pipeline_o.producer_commit_w_index(pv_slot)
                    pipeline_p.consumer_release_w_index(pv_slot)

    @cute.jit
    def _softmax_step(
        self,
        slot: cutlass.Constexpr[int],
        s_consumer_phase: Int32,
        p_producer_phase: Int32,
        sm_stats_producer_phase: Int32,
        softmax: SoftmaxSm100,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        pipeline_s,
        pipeline_p,
        pipeline_sm_stats,
        sm_stats_barrier,
        stats_barrier_idx: Int32,
        thr_tmem_load,
        thr_tmem_store,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        tScS_t2r: cute.Tensor,
        tScP_shape,
        sQIdxMeta: cute.Tensor,
        qidx_meta_slot: Int32,
        group_tidx: Int32,
        masked_tok_count: Int32,
        kv_block_col_start: Int32,
        causal_q_offset: Int32,
        kv_valid_cols: Int32,
        lse_temperature_scale: Float32,
        return_temperature_lse: cutlass.Constexpr[bool],
        apply_causal_mask: cutlass.Constexpr[bool] = False,
        signal_stats_barrier: cutlass.Constexpr[bool] = True,
    ):
        slot_rt = Int32(slot)

        pipeline_s.consumer_wait_w_index_phase(slot_rt, s_consumer_phase)

        tSrS_t2r = cute.make_rmem_tensor(tScS_t2r.shape, self.qk_acc_dtype)
        cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)

        col_limit = kv_valid_cols
        if const_expr(self.causal):
            if cutlass.const_expr(apply_causal_mask):
                tok_idx = group_tidx // Int32(self.qheadperkv)
                q_idx = self._decode_q_idx_from_qsplit(
                    sQIdxMeta[qidx_meta_slot + tok_idx]
                )
                causal_limit = q_idx + causal_q_offset - kv_block_col_start + Int32(1)
                col_limit = cutlass.select_(
                    tok_idx < masked_tok_count,
                    cutlass.min(col_limit, causal_limit),
                    col_limit,
                )
        mask_r2p_lambda(
            tSrS_t2r,
            lambda s: r2p_bitmask_below(col_limit, s),
            rank1=True,
        )

        # Each sparse CTA computes exactly one KV block for the current Q group,
        # so full-tile softmax is always the first and only online-softmax step.
        row_max, _ = softmax.update_row_max(tSrS_t2r.load(), True)
        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        if const_expr(return_temperature_lse):
            lse_temperature_row_sum = softmax.compute_scaled_exp2_row_sum(
                tSrS_t2r,
                lse_temperature_scale,
            )

        pipeline_p.producer_acquire_w_index_phase(slot_rt, p_producer_phase)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape,
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )

        for k in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, k], tStP_r2t[None, None, k]
            )
            if cutlass.const_expr(self.split_P_arrive > 0):
                split_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive
                    // self.n_block_size
                )
                if cutlass.const_expr(k + 1 == split_idx):
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_p.producer_commit_w_index(slot_rt)
        cute.arch.fence_view_async_tmem_store()
        if cutlass.const_expr(self.split_P_arrive == 0):
            pipeline_p.producer_commit_w_index(slot_rt)

        pipeline_sm_stats.producer_acquire_w_index_phase(
            slot_rt, sm_stats_producer_phase
        )
        softmax.update_row_sum(tSrS_t2r.load(), Float32(0.0), True)
        del tSrS_t2r
        sScale_slot = cute.make_tensor(
            sScale.iterator + slot_rt * Int32(self.m_block_size * 2),
            cute.make_layout(self.m_block_size * 2),
        )
        sScale_slot[group_tidx] = softmax.row_sum[0]
        sScale_slot[group_tidx + Int32(self.m_block_size)] = softmax.row_max[0]
        if const_expr(return_temperature_lse):
            sScale_temperature_slot = cute.make_tensor(
                sScaleTemperature.iterator + slot_rt * Int32(self.m_block_size),
                cute.make_layout(self.m_block_size),
            )
            sScale_temperature_slot[group_tidx] = lse_temperature_row_sum
        cute.arch.fence_view_async_shared()

        if const_expr(signal_stats_barrier):
            sm_stats_barrier.arrive_w_index(index=stats_barrier_idx)
        pipeline_s.consumer_release_w_index(slot_rt)

    @cute.jit
    def _wg_softmax(
        self,
        stage: cutlass.Constexpr[int],
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tStS: cute.Tensor,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        sSplitIdx: cute.Tensor,
        sQIdx: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        pipeline_s,
        pipeline_p,
        pipeline_o,
        pipeline_sm_stats,
        sm_stats_barrier,
        epilogue_barrier,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: cute.Tensor | None,
        softmax_scale_log2: Float32,
        lse_temperature_scale_log2: Float32,
        lse_temperature_scale: Float32,
        kv_block_idx: Int32,
        kv_valid_cols: Int32,
        diag_q_count: Int32,
        num_q_groups: Int32,
        count_raw: Int32,
        has_work: Int32,
        causal_q_offset: Int32,
        batch_idx: Int32,
        head_kv_idx: Int32,
        seq_len_q: Int32,
        head_q: Int32,
        num_heads_kv: Int32,
        q_batch_offset: Int32,
        mQ_2d: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        warp_idx_in_wg = warp_idx % Int32(self.warps_per_group)
        group_tidx = warp_idx_in_wg * Int32(cute.arch.WARP_SIZE) + tidx % Int32(
            cute.arch.WARP_SIZE
        )
        stats_barrier_idx = Int32(stage) * Int32(self.warps_per_group) + warp_idx_in_wg

        thr0_qk = tiled_mma_qk.get_slice(0)
        tScS = thr0_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr0_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScP_shape = (cta_qk_tiler[0], tilePlikeFP32)
        tSAcc = tStS[(None, None), 0, 0, stage]

        softmax = SoftmaxSm100.create(softmax_scale_log2)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.qk_acc_dtype
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(
            group_tidx
        )
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)
        tScS_t2r = thr_tmem_load.partition_D(tScS)
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(
            group_tidx
        )
        tStP_r2t = thr_tmem_store.partition_D(tStP)

        total_q = mQ_2d.shape[0] // head_q
        thr0_pv = tiled_mma_pv.get_slice(0)
        pv_acc_shape = thr0_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO_base = thr0_pv.make_fragment_C(pv_acc_shape)
        corr_tile_size = 64
        tOcO = thr0_pv.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        o_tmem_copy_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.pv_acc_dtype
        )

        if has_work:
            kv_block_col_start = Int32(0)
            if const_expr(self.causal):
                kv_block_col_start = kv_block_idx * Int32(self.n_block_size)

            num_stage_groups = (num_q_groups + Int32(1 - stage)) // Int32(2)
            for qi_iter in cutlass.range(num_stage_groups, unroll=1):
                qi_group = qi_iter * Int32(2) + Int32(stage)
                phase = qi_iter & Int32(1)
                producer_phase = phase ^ Int32(1)
                qidx_meta_slot = (qi_group & Int32(self.qidx_meta_stages - 1)) * Int32(
                    self.q_tokens_per_group
                )

                softmax.reset()

                if const_expr(self.causal):
                    qi_group_start = qi_group * Int32(self.q_tokens_per_group)
                    masked_tok_count = cutlass.max(
                        Int32(0),
                        cutlass.min(
                            Int32(self.q_tokens_per_group),
                            diag_q_count - qi_group_start,
                        ),
                    )
                    self._softmax_step(
                        stage,
                        phase,
                        producer_phase,
                        producer_phase,
                        softmax,
                        sScale,
                        sScaleTemperature,
                        pipeline_s,
                        pipeline_p,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        stats_barrier_idx,
                        thr_tmem_load,
                        thr_tmem_store,
                        tStS_t2r,
                        tStP_r2t,
                        tScS_t2r,
                        tScP_shape,
                        sQIdxMeta,
                        qidx_meta_slot,
                        group_tidx,
                        masked_tok_count,
                        kv_block_col_start,
                        causal_q_offset,
                        kv_valid_cols,
                        lse_temperature_scale,
                        const_expr(mLSE_temperature_partial is not None),
                        True,
                        False,
                    )
                else:
                    self._softmax_step(
                        stage,
                        phase,
                        producer_phase,
                        producer_phase,
                        softmax,
                        sScale,
                        sScaleTemperature,
                        pipeline_s,
                        pipeline_p,
                        pipeline_sm_stats,
                        sm_stats_barrier,
                        stats_barrier_idx,
                        thr_tmem_load,
                        thr_tmem_store,
                        tStS_t2r,
                        tStP_r2t,
                        tScS_t2r,
                        tScP_shape,
                        sQIdxMeta,
                        qidx_meta_slot,
                        group_tidx,
                        Int32(0),
                        kv_block_col_start,
                        Int32(0),
                        kv_valid_cols,
                        lse_temperature_scale,
                        const_expr(mLSE_temperature_partial is not None),
                        False,
                        False,
                    )
                epilogue_barrier.arrive_and_wait_w_index(index=Int32(stage))
                self._epilogue_step(
                    qi_group,
                    group_tidx,
                    warp_idx_in_wg,
                    tOtO_base,
                    tOcO_i,
                    o_tmem_copy_atom,
                    sScale,
                    sScaleTemperature,
                    sSplitIdx,
                    sQIdx,
                    sQIdxMeta,
                    pipeline_o,
                    pipeline_sm_stats,
                    sm_stats_barrier,
                    epilogue_barrier,
                    mO_partial,
                    mLSE_partial,
                    mLSE_temperature_partial,
                    softmax_scale_log2,
                    lse_temperature_scale_log2,
                    count_raw,
                    batch_idx,
                    head_kv_idx,
                    seq_len_q,
                    head_q,
                    num_heads_kv,
                    q_batch_offset,
                    total_q,
                    False,
                    stage,
                )

    @cute.jit
    def _store_o_partial_vec4(
        self,
        ptr: cute.Pointer,
        v0: Float32,
        v1: Float32,
        v2: Float32,
        v3: Float32,
    ):
        stg_128_cs(ptr, v0, v1, v2, v3)

    @cute.jit
    def _store_o_partial_vec8_half(
        self,
        ptr: cute.Pointer,
        v0: Float32,
        v1: Float32,
        v2: Float32,
        v3: Float32,
        v4: Float32,
        v5: Float32,
        v6: Float32,
        v7: Float32,
    ):
        if cutlass.const_expr(self.o_dtype is cutlass.BFloat16):
            stg_128_bf16_cs(ptr, v0, v1, v2, v3, v4, v5, v6, v7)
        else:
            stg_128_f16_cs(ptr, v0, v1, v2, v3, v4, v5, v6, v7)

    @cute.jit
    def _epilogue_step(
        self,
        qi_group: Int32,
        group_tidx: Int32,
        warp_idx_in_wg: Int32,
        tOtO_base: cute.Tensor,
        tOcO_i: cute.Tensor,
        o_tmem_copy_atom,
        sScale: cute.Tensor,
        sScaleTemperature: cute.Tensor,
        sSplitIdx: cute.Tensor,
        sQIdx: cute.Tensor,
        sQIdxMeta: cute.Tensor,
        pipeline_o,
        pipeline_sm_stats,
        sm_stats_barrier,
        epilogue_barrier,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mLSE_temperature_partial: cute.Tensor | None,
        softmax_scale_log2: Float32,
        lse_temperature_scale_log2: Float32,
        count_raw: Int32,
        batch_idx: Int32,
        head_kv_idx: Int32,
        seq_len_q: Int32,
        head_q: Int32,
        num_heads_kv: Int32,
        q_batch_offset: Int32,
        total_q: Int32,
        use_stats_barrier: cutlass.Constexpr[bool],
        softmax_stage: cutlass.Constexpr[int],
    ):
        slot = qi_group & Int32(1)
        phase = (qi_group // Int32(2)) & Int32(1)
        stage_base = slot * Int32(self.tmem_o_stage_stride)
        corr_tile_size = 64
        sScale_slot = cute.make_tensor(
            sScale.iterator + slot * Int32(self.m_block_size * 2),
            cute.make_layout(self.m_block_size * 2),
        )
        sScale_temperature_slot = cute.make_tensor(
            sScaleTemperature.iterator + slot * Int32(self.m_block_size),
            cute.make_layout(self.m_block_size),
        )
        sSplitIdx_slot = cute.make_tensor(
            sSplitIdx.iterator + slot * Int32(self.q_tokens_per_group),
            cute.make_layout((self.q_tokens_per_group,)),
        )
        sQIdx_slot = cute.make_tensor(
            sQIdx.iterator + slot * Int32(self.q_tokens_per_group),
            cute.make_layout((self.q_tokens_per_group,)),
        )
        qidx_meta_slot = (qi_group & Int32(self.qidx_meta_stages - 1)) * Int32(
            self.q_tokens_per_group
        )

        pipeline_o.consumer_wait_w_index_phase(slot, phase)
        if const_expr(use_stats_barrier):
            sm_stats_barrier.arrive_and_wait_w_index(
                index=slot * Int32(self.warps_per_group) + warp_idx_in_wg
            )

        if group_tidx < Int32(self.q_tokens_per_group):
            tok = group_tidx
            qi = qi_group * Int32(self.q_tokens_per_group) + tok
            if qi < count_raw:
                qsplit = sQIdxMeta[qidx_meta_slot + tok]
                q_idx = self._decode_q_idx_from_qsplit(qsplit)
                sQIdx_slot[tok] = q_idx
                sSplitIdx_slot[tok] = self._decode_split_idx_from_qsplit(qsplit)
        epilogue_barrier.arrive_and_wait_w_index(index=Int32(softmax_stage))

        tOtO = cute.make_tensor(
            tOtO_base.iterator + stage_base + Int32(self.tmem_o_offset),
            tOtO_base.layout,
        )
        for col_pass_idx in cutlass.range(Int32(2), unroll=1):
            col_pass = col_pass_idx * Int32(corr_tile_size)
            tOtO_pass_ptr = cute.make_ptr(
                self.pv_acc_dtype,
                tOtO.iterator.toint() + col_pass,
                cute.AddressSpace.tmem,
                assumed_align=8,
            )
            tOtO_pass = cute.make_tensor(tOtO_pass_ptr, tOtO.layout)
            tOtO_pass_i = cute.logical_divide(
                tOtO_pass, cute.make_layout((self.m_block_size, corr_tile_size))
            )
            tiled_tmem_load_pass = tcgen05.make_tmem_copy(
                o_tmem_copy_atom, tOtO_pass_i[(None, None), 0]
            )
            thr_tmem_load_pass = tiled_tmem_load_pass.get_slice(group_tidx)
            tOtO_t2r_pass = thr_tmem_load_pass.partition_S(
                tOtO_pass_i[(None, None), None]
            )
            tOcO_t2r_pass = thr_tmem_load_pass.partition_D(tOcO_i[(None, None), None])

            tOtO_t2r_i = tOtO_t2r_pass[None, None, None, 0]
            tOcO_t2r_i = tOcO_t2r_pass[None, None, None, 0]
            tOrO_frg = cute.make_rmem_tensor_like(tOcO_t2r_i, self.pv_acc_dtype)
            cute.copy(tiled_tmem_load_pass, tOtO_t2r_i, tOrO_frg)

            tOrO_mn = make_16x256b_tensor_mn_view(tOrO_frg)
            tOrO_mn = cute.make_tensor(
                tOrO_mn.iterator, cute.select(tOrO_mn.layout, mode=[0, 1])
            )
            tOcO_mn = make_16x256b_tensor_mn_view(tOcO_t2r_i)
            tOcO_mn = cute.make_tensor(
                tOcO_mn.iterator, cute.select(tOcO_mn.layout, mode=[0, 1])
            )
            num_rows = cute.size(tOrO_mn, mode=[0])
            num_cols = cute.size(tOrO_mn, mode=[1])

            for r in cutlass.range_constexpr(num_rows):
                if const_expr(self.o_dtype is Float32):
                    for c4 in cutlass.range_constexpr(num_cols // 4):
                        c_base = Int32(c4) * Int32(4)
                        row_col = tOcO_mn[r, c_base]
                        row = row_col[0]
                        col = row_col[1] + col_pass
                        if row < Int32(self.m_block_size):
                            tok = row // Int32(self.qheadperkv)
                            row_in_tok = row - tok * Int32(self.qheadperkv)
                            qi = qi_group * Int32(self.q_tokens_per_group) + tok
                            if qi < count_raw:
                                q_idx = sQIdx_slot[tok]
                                split = sSplitIdx_slot[tok]
                                q_abs = q_batch_offset + q_idx
                                flat_row = (
                                    Int64(split) * Int64(total_q) * Int64(head_q)
                                    + Int64(q_abs) * Int64(head_q)
                                    + Int64(head_kv_idx) * Int64(self.qheadperkv)
                                    + Int64(row_in_tok)
                                )
                                row_sum_val = sScale_slot[row]
                                is_zero_or_nan = (
                                    row_sum_val == Float32(0.0)
                                    or row_sum_val != row_sum_val
                                )
                                row_scale = cute.arch.rcp_approx(
                                    row_sum_val if not is_zero_or_nan else Float32(1.0)
                                )
                                row_base_ptr = flat_row * Int64(self.head_dim)
                                o0 = tOrO_mn[r, c_base]
                                o1 = tOrO_mn[r, c_base + Int32(1)]
                                o2 = tOrO_mn[r, c_base + Int32(2)]
                                o3 = tOrO_mn[r, c_base + Int32(3)]
                                scale_pair = (row_scale, row_scale)
                                o0, o1 = cute.arch.mul_packed_f32x2(
                                    (o0, o1), scale_pair
                                )
                                o2, o3 = cute.arch.mul_packed_f32x2(
                                    (o2, o3), scale_pair
                                )
                                fake_col = real_col_to_stg128_fake_col(col)
                                ptr = (
                                    mO_partial.iterator + row_base_ptr + Int64(fake_col)
                                )
                                self._store_o_partial_vec4(
                                    ptr,
                                    o0,
                                    o1,
                                    o2,
                                    o3,
                                )
                else:
                    assert num_cols % 8 == 0, (
                        "half O_partial STG.128 requires the epilogue "
                        "TMEM fragment column count to be a multiple of 8"
                    )
                    for c8 in cutlass.range_constexpr(num_cols // 8):
                        c_base = Int32(c8) * Int32(8)
                        row_col = tOcO_mn[r, c_base]
                        row = row_col[0]
                        col = row_col[1] + col_pass
                        if row < Int32(self.m_block_size):
                            tok = row // Int32(self.qheadperkv)
                            row_in_tok = row - tok * Int32(self.qheadperkv)
                            qi = qi_group * Int32(self.q_tokens_per_group) + tok
                            if qi < count_raw:
                                q_idx = sQIdx_slot[tok]
                                split = sSplitIdx_slot[tok]
                                q_abs = q_batch_offset + q_idx
                                flat_row = (
                                    Int64(split) * Int64(total_q) * Int64(head_q)
                                    + Int64(q_abs) * Int64(head_q)
                                    + Int64(head_kv_idx) * Int64(self.qheadperkv)
                                    + Int64(row_in_tok)
                                )
                                row_sum_val = sScale_slot[row]
                                is_zero_or_nan = (
                                    row_sum_val == Float32(0.0)
                                    or row_sum_val != row_sum_val
                                )
                                row_scale = cute.arch.rcp_approx(
                                    row_sum_val if not is_zero_or_nan else Float32(1.0)
                                )
                                row_base_ptr = flat_row * Int64(self.head_dim)
                                o0 = tOrO_mn[r, c_base]
                                o1 = tOrO_mn[r, c_base + Int32(1)]
                                o2 = tOrO_mn[r, c_base + Int32(2)]
                                o3 = tOrO_mn[r, c_base + Int32(3)]
                                o4 = tOrO_mn[r, c_base + Int32(4)]
                                o5 = tOrO_mn[r, c_base + Int32(5)]
                                o6 = tOrO_mn[r, c_base + Int32(6)]
                                o7 = tOrO_mn[r, c_base + Int32(7)]
                                scale_pair = (row_scale, row_scale)
                                o0, o1 = cute.arch.mul_packed_f32x2(
                                    (o0, o1), scale_pair
                                )
                                o2, o3 = cute.arch.mul_packed_f32x2(
                                    (o2, o3), scale_pair
                                )
                                o4, o5 = cute.arch.mul_packed_f32x2(
                                    (o4, o5), scale_pair
                                )
                                o6, o7 = cute.arch.mul_packed_f32x2(
                                    (o6, o7), scale_pair
                                )
                                fake_col = real_col_to_stg128_half_fake_col(col)
                                ptr = (
                                    mO_partial.iterator + row_base_ptr + Int64(fake_col)
                                )
                                self._store_o_partial_vec8_half(
                                    ptr,
                                    o0,
                                    o1,
                                    o2,
                                    o3,
                                    o4,
                                    o5,
                                    o6,
                                    o7,
                                )
        cute.arch.fence_view_async_tmem_load()

        tok_local = Int32(group_tidx) // Int32(self.qheadperkv)
        h_local = Int32(group_tidx) % Int32(self.qheadperkv)
        qi_lse = qi_group * Int32(self.q_tokens_per_group) + tok_local
        if qi_lse < count_raw:
            row_sum_val = sScale_slot[group_tidx]
            row_max_val = sScale_slot[group_tidx + Int32(self.m_block_size)]
            is_zero_or_nan = row_sum_val == Float32(0.0) or row_sum_val != row_sum_val
            LN2 = Float32(math.log(2.0))
            lse_cur = (
                (
                    row_max_val * softmax_scale_log2
                    + cute.math.log2(row_sum_val, fastmath=True)
                )
                * LN2
                if not is_zero_or_nan
                else -Float32.inf
            )
            q_idx_lse = sQIdx_slot[tok_local]
            h_abs = head_kv_idx * Int32(self.qheadperkv) + h_local
            split_lse = sSplitIdx_slot[tok_local]
            q_abs_lse = q_batch_offset + q_idx_lse
            mLSE_partial[split_lse, q_abs_lse, h_abs] = lse_cur
            if const_expr(mLSE_temperature_partial is not None):
                row_sum_temperature_val = sScale_temperature_slot[group_tidx]
                is_temperature_zero_or_nan = (
                    row_sum_temperature_val == Float32(0.0)
                    or row_sum_temperature_val != row_sum_temperature_val
                )
                lse_temperature_cur = (
                    (
                        row_max_val * lse_temperature_scale_log2
                        + cute.math.log2(row_sum_temperature_val, fastmath=True)
                    )
                    * LN2
                    if not is_temperature_zero_or_nan
                    else -Float32.inf
                )
                mLSE_temperature_partial[split_lse, q_abs_lse, h_abs] = (
                    lse_temperature_cur
                )
        epilogue_barrier.arrive_and_wait_w_index(index=Int32(softmax_stage))

        pipeline_sm_stats.consumer_release_w_index(slot)
        pipeline_o.consumer_release_w_index(slot)
