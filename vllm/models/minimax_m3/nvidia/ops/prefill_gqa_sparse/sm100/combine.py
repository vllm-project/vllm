# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# mypy: ignore-errors
"""Sparse forward combine kernel and public launcher.

This keeps the local fake-layout -> real-layout epilogue needed by the lean
sparse forward path.
"""

# Modified Step 7: O_out write with SMEM fake->real column permutation.
# O_partial dim is in STG.128 fake layout; O_out dim is real layout.
import math
from functools import partial

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
from cutlass import Boolean, Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from cutlass.cute.nvgpu import cpasync

from ..common import utils
from ..common.copy_utils import sts_32_bf16, sts_32_f16
from ..common.cute_dsl_utils import assume_tensor_aligned, torch2cute_dtype_map
from ..common.pack_gqa import PackGQAComb
from ..common.seqlen_info import SeqlenInfo
from ..common.tma_utils import (
    stg128_fake_col_to_real_col,
    stg128_half_fake_col_to_real_col,
)


class SparseAttentionForwardCombine:
    def __init__(
        self,
        dtype: type[cutlass.Numeric],
        dtype_partial: type[cutlass.Numeric],
        head_dim: int,
        tile_m: int = 8,
        k_block_size: int = 64,
        log_max_splits: int = 4,
        num_threads: int = 256,
        stages: int = 4,
        use_pdl: bool = False,
    ):
        """
        Forward combine kernel for split attention computation.

        :param dtype: output data type
        :param dtype_partial: partial accumulation data type
        :param head_dim: head dimension
        :param tile_m: m block size
        :param k_block_size: k block size
        :param log_max_splits: log2 of maximum splits
        :param num_threads: number of threads
        :param varlen: whether using variable length sequences
        :param stages: number of pipeline stages
        """
        self.dtype = dtype
        self.dtype_partial = dtype_partial
        self.head_dim = head_dim
        self.tile_m = tile_m
        self.k_block_size = k_block_size
        self.max_splits = 1 << log_max_splits
        self.num_threads = num_threads
        self.is_even_k = head_dim % k_block_size == 0
        self.stages = stages
        self.use_pdl = use_pdl
        self.use_stg128_half_layout = dtype_partial in (
            cutlass.BFloat16,
            cutlass.Float16,
        )

    @staticmethod
    def can_implement(
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        num_threads,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters."""
        if dtype not in [cutlass.Float16, cutlass.BFloat16, cutlass.Float32]:
            return False
        if dtype_partial not in [cutlass.Float16, cutlass.BFloat16, Float32]:
            return False
        if head_dim % 8 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if tile_m % 8 != 0:
            return False
        max_splits = 1 << log_max_splits
        if max_splits > 256:
            return False
        if (tile_m * max_splits) % num_threads != 0:
            return False
        return True

    def _setup_attributes(self):
        # GMEM copy setup for O partial
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype_partial.width
        assert self.k_block_size % async_copy_elems == 0

        k_block_gmem = (
            128
            if self.k_block_size % 128 == 0
            else (64 if self.k_block_size % 64 == 0 else 32)
        )
        gmem_threads_per_row = k_block_gmem // async_copy_elems
        assert self.num_threads % gmem_threads_per_row == 0

        # Async copy atom for O partial load
        atom_async_copy_partial = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype_partial,
            num_bits_per_copy=universal_copy_bits,
        )
        tOpartial_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row, gmem_threads_per_row),
            order=(1, 0),
        )
        vOpartial_layout = cute.make_layout((1, async_copy_elems))
        self.gmem_tiled_copy_O_partial = cute.make_tiled_copy_tv(
            atom_async_copy_partial, tOpartial_layout, vOpartial_layout
        )

        # GMEM copy setup for final O (use universal copy for store)
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=async_copy_elems * self.dtype.width,
        )
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy,
            tOpartial_layout,
            vOpartial_layout,
        )
        # LSE copy setup with async copy (alignment = 1)
        lse_copy_bits = Float32.width  # 1 element per copy, width is in bits
        m_block_smem = (
            128
            if self.tile_m % 128 == 0
            else (
                64
                if self.tile_m % 64 == 0
                else (
                    32
                    if self.tile_m % 32 == 0
                    else (16 if self.tile_m % 16 == 0 else 8)
                )
            )
        )
        gmem_threads_per_row_lse = m_block_smem
        assert self.num_threads % gmem_threads_per_row_lse == 0

        # Async copy atom for LSE load
        atom_async_copy_lse = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.ALWAYS),
            Float32,
            num_bits_per_copy=lse_copy_bits,
        )
        tLSE_layout = cute.make_ordered_layout(
            (self.num_threads // gmem_threads_per_row_lse, gmem_threads_per_row_lse),
            order=(1, 0),
        )
        vLSE_layout = cute.make_layout(1)
        self.gmem_tiled_copy_LSE = cute.make_tiled_copy_tv(
            atom_async_copy_lse, tLSE_layout, vLSE_layout
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory
        # ///////////////////////////////////////////////////////////////////////////////

        # Shared memory to register copy for LSE
        self.smem_threads_per_col_lse = self.num_threads // m_block_smem
        assert 32 % self.smem_threads_per_col_lse == 0  # Must divide warp size

        s2r_layout_atom_lse = cute.make_ordered_layout(
            (
                self.smem_threads_per_col_lse,
                self.num_threads // self.smem_threads_per_col_lse,
            ),
            order=(0, 1),
        )
        self.s2r_tiled_copy_LSE = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32),
            s2r_layout_atom_lse,
            cute.make_layout(1),
        )

        # LSE shared memory layout with swizzling to avoid bank conflicts
        # This works for kBlockMSmem = 8, 16, 32, 64, 128, no bank conflicts
        if const_expr(m_block_smem == 8):
            smem_lse_swizzle = cute.make_swizzle(5, 0, 5)
        elif const_expr(m_block_smem == 16):
            smem_lse_swizzle = cute.make_swizzle(4, 0, 4)
        else:
            smem_lse_swizzle = cute.make_swizzle(3, 2, 3)
        smem_layout_atom_lse = cute.make_composed_layout(
            smem_lse_swizzle,
            0,
            cute.make_ordered_layout((8, m_block_smem), order=(1, 0)),
        )
        self.smem_layout_lse = cute.tile_to_shape(
            smem_layout_atom_lse, (self.max_splits, self.tile_m), (0, 1)
        )

        # O_partial staging layout.
        if const_expr(self.dtype_partial in [cutlass.Float16, cutlass.BFloat16]):
            smem_layout_atom_o = _get_cpasync_smem_layout_atom(
                self.dtype_partial, self.k_block_size
            )
            self.smem_layout_o = cute.tile_to_shape(
                smem_layout_atom_o,
                (self.tile_m, self.k_block_size, self.stages),
                (0, 1, 2),
            )
        else:
            self.smem_layout_o = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size, self.stages), order=(1, 0, 2)
            )

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None = None,
        mLSE_temperature_partial: cute.Tensor | None = None,
        mLSE_temperature: cute.Tensor | None = None,
        cu_seqlens: cute.Tensor | None = None,
        seqused: cute.Tensor | None = None,
        num_splits_dynamic_ptr: cute.Tensor | None = None,
        varlen_batch_idx: cute.Tensor | None = None,
        semaphore_to_reset: cute.Tensor | None = None,
        mSplitCounts: cute.Tensor | None = None,
        qhead_per_kvhead: Int32 = Int32(1),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        # Type checking
        if const_expr(not (mO_partial.element_type == self.dtype_partial)):
            raise TypeError("O partial tensor must match dtype_partial")
        if const_expr(not (mO.element_type == self.dtype)):
            raise TypeError("O tensor must match dtype")
        if const_expr(mLSE_partial.element_type not in [Float32]):
            raise TypeError("LSE partial tensor must be Float32")
        if const_expr(mLSE is not None and mLSE.element_type not in [Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(
            mLSE_temperature_partial is not None
            and mLSE_temperature_partial.element_type not in [Float32]
        ):
            raise TypeError("temperature LSE partial tensor must be Float32")
        if const_expr(
            mLSE_temperature is not None
            and mLSE_temperature.element_type not in [Float32]
        ):
            raise TypeError("temperature LSE tensor must be Float32")
        if const_expr((mLSE_temperature_partial is None) != (mLSE_temperature is None)):
            raise ValueError(
                "temperature LSE partial and output tensors must either both be provided or both be None"
            )

        # Shape validation - input tensors are in user format, need to be converted to kernel format
        if const_expr(len(mO_partial.shape) not in [4, 5]):
            raise ValueError(
                "O partial tensor must have 4 or 5 dimensions: (num_splits, batch, seqlen, nheads, headdim) or (num_splits, total_q, nheads, headdim)"
            )
        if const_expr(len(mLSE_partial.shape) not in [3, 4]):
            raise ValueError(
                "LSE partial tensor must have 3 or 4 dimensions: (num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(len(mO.shape) not in [3, 4]):
            raise ValueError(
                "O tensor must have 3 or 4 dimensions: (batch, seqlen, nheads, headdim) or (total_q, nheads, headdim)"
            )
        if const_expr(mLSE is not None and len(mLSE.shape) not in [2, 3]):
            raise ValueError(
                "LSE tensor must have 2 or 3 dimensions: (batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(
            mLSE_temperature_partial is not None
            and len(mLSE_temperature_partial.shape) not in [3, 4]
        ):
            raise ValueError(
                "temperature LSE partial tensor must have 3 or 4 dimensions: "
                "(num_splits, batch, seqlen, nheads) or (num_splits, total_q, nheads)"
            )
        if const_expr(
            mLSE_temperature is not None and len(mLSE_temperature.shape) not in [2, 3]
        ):
            raise ValueError(
                "temperature LSE tensor must have 2 or 3 dimensions: "
                "(batch, seqlen, nheads) or (total_q, nheads)"
            )
        if const_expr(mSplitCounts is not None):
            if const_expr(mSplitCounts.element_type not in [Int32]):
                raise TypeError("split_counts tensor must be Int32")
            if const_expr(len(mSplitCounts.shape) != 3):
                raise ValueError(
                    "split_counts tensor must have shape (batch, seqlen, nheads_kv)"
                )

        mO_partial, mO = [assume_tensor_aligned(t) for t in (mO_partial, mO)]
        # (num_splits, b, seqlen, h, d) -> (seqlen, d, num_splits, h, b)
        # or (num_splits, total_q, h, d) -> (total_q, d, num_splits, h)
        O_partial_layout_transpose = (
            [2, 4, 0, 3, 1] if const_expr(cu_seqlens is None) else [1, 3, 0, 2]
        )
        # (b, seqlen, h, d) -> (seqlen, d, h, b) or (total_q, h, d) -> (total_q, d, h)
        mO_partial = cute.make_tensor(
            mO_partial.iterator,
            cute.select(mO_partial.layout, mode=O_partial_layout_transpose),
        )
        O_layout_transpose = (
            [1, 3, 2, 0] if const_expr(cu_seqlens is None) else [0, 2, 1]
        )
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        # (num_splits, b, h, seqlen) -> (seqlen, num_splits, h, b)
        # Input is pre-transposed: [topK, B, Hq, Sq] with Sq innermost for K2-friendly reads.
        # or (num_splits, total_q, h) -> (total_q, num_splits, h)
        LSE_partial_layout_transpose = (
            [3, 0, 2, 1] if const_expr(cu_seqlens is None) else [1, 0, 2]
        )
        mLSE_partial = cute.make_tensor(
            mLSE_partial.iterator,
            cute.select(mLSE_partial.layout, mode=LSE_partial_layout_transpose),
        )
        # (b, seqlen, h) -> (seqlen, h, b) or (total_q, h) -> (total_q, h)
        LSE_layout_transpose = [1, 2, 0] if const_expr(cu_seqlens is None) else [0, 1]
        mLSE = (
            cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
            if mLSE is not None
            else None
        )
        mLSE_temperature_partial = (
            cute.make_tensor(
                mLSE_temperature_partial.iterator,
                cute.select(
                    mLSE_temperature_partial.layout, mode=LSE_partial_layout_transpose
                ),
            )
            if mLSE_temperature_partial is not None
            else None
        )
        mLSE_temperature = (
            cute.make_tensor(
                mLSE_temperature.iterator,
                cute.select(mLSE_temperature.layout, mode=LSE_layout_transpose),
            )
            if mLSE_temperature is not None
            else None
        )

        # Determine if we have variable length sequences
        varlen = const_expr(cu_seqlens is not None or seqused is not None)

        self._setup_attributes()

        # Output-dtype permutation buffer for Step 7 (tile_m × k_block_size).
        # Accumulation stays fp32; the final dtype conversion happens before
        # the fake→real SMEM scatter to reduce half-output SMEM pressure.
        if const_expr(self.dtype in [cutlass.Float16, cutlass.BFloat16]):
            smem_layout_perm = cute.make_layout(
                (self.tile_m, self.k_block_size),
                stride=(self.k_block_size + 16, 1),
            )
        else:
            smem_layout_perm = cute.make_ordered_layout(
                (self.tile_m, self.k_block_size), order=(1, 0)
            )

        @cute.struct
        class SharedStorage:
            sLSE: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sLSETemperature: cute.struct.Align[
                cute.struct.MemRange[Float32, cute.cosize(self.smem_layout_lse)], 128
            ]
            sMaxValidSplit: cute.struct.Align[
                cute.struct.MemRange[Int32, self.tile_m], 128
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[
                    self.dtype_partial, cute.cosize(self.smem_layout_o)
                ],
                128,
            ]
            sO_perm: cute.struct.Align[
                cute.struct.MemRange[self.dtype, cute.cosize(smem_layout_perm)], 128
            ]

        smem_size = SharedStorage.size_in_bytes()

        # Grid: (ceil(seqlen/tile_m), ceil(dim/k_block), num_head * batch)
        # Head separated from seqlen → enables future TMA (contiguous Sq tiles)
        seqlen = mO_partial.shape[0]
        num_head = mO_partial.shape[3]
        batch_size = (
            mO_partial.shape[4]
            if const_expr(cu_seqlens is None)
            else Int32(cu_seqlens.shape[0] - 1)
        )

        seqlen_divmod = FastDivmodDivisor(seqlen)
        head_divmod = FastDivmodDivisor(num_head)

        grid_dim = (
            cute.ceil_div(seqlen * num_head, self.tile_m),
            cute.ceil_div(self.head_dim, self.k_block_size),
            batch_size,
        )

        self.kernel(
            mO_partial,
            mLSE_partial,
            mO,
            mLSE,
            mLSE_temperature_partial,
            mLSE_temperature,
            cu_seqlens,
            seqused,
            num_splits_dynamic_ptr,
            varlen_batch_idx,
            semaphore_to_reset,
            mSplitCounts,
            qhead_per_kvhead,
            SharedStorage,
            self.smem_layout_lse,
            self.smem_layout_o,
            smem_layout_perm,
            self.gmem_tiled_copy_O_partial,
            self.gmem_tiled_copy_O,
            self.gmem_tiled_copy_LSE,
            self.s2r_tiled_copy_LSE,
            seqlen_divmod,
            head_divmod,
            self.use_pdl,
            varlen,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=smem_size,
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.jit
    def decode_flat_row_idx(
        self,
        idx: Int32,
        head_divmod: FastDivmodDivisor,
    ):
        """Decode flattened tile rows under the H_q-innermost contract."""
        q_idx_local, head_idx = divmod(idx, head_divmod)
        return q_idx_local, head_idx

    @cute.kernel
    def kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mO: cute.Tensor,
        mLSE: cute.Tensor | None,
        mLSE_temperature_partial: cute.Tensor | None,
        mLSE_temperature: cute.Tensor | None,
        cu_seqlens: cute.Tensor | None,
        seqused: cute.Tensor | None,
        num_splits_dynamic_ptr: cute.Tensor | None,
        varlen_batch_idx: cute.Tensor | None,
        semaphore_to_reset: cute.Tensor | None,
        mSplitCounts: cute.Tensor | None,
        qhead_per_kvhead: Int32,
        SharedStorage: cutlass.Constexpr,
        smem_layout_lse: cute.Layout | cute.ComposedLayout,
        smem_layout_o: cute.Layout | cute.ComposedLayout,
        smem_layout_perm: cute.Layout,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        gmem_tiled_copy_LSE: cute.TiledCopy,
        s2r_tiled_copy_LSE: cute.TiledCopy,
        seqlen_divmod: FastDivmodDivisor,
        head_divmod: FastDivmodDivisor,
        use_pdl: cutlass.Constexpr[bool],
        varlen: cutlass.Constexpr[bool],
    ):
        # Thread and block indices
        tidx, _, _ = cute.arch.thread_idx()
        m_block, k_block, maybe_virtual_batch = cute.arch.block_idx()

        batch_idx = (
            varlen_batch_idx[maybe_virtual_batch]
            if const_expr(varlen_batch_idx is not None)
            else maybe_virtual_batch
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sLSE = storage.sLSE.get_tensor(smem_layout_lse)
        sLSE_temperature = storage.sLSETemperature.get_tensor(smem_layout_lse)
        sMaxValidSplit = storage.sMaxValidSplit.get_tensor((self.tile_m,))
        sO = storage.sO.get_tensor(smem_layout_o)
        sO_perm_buf = storage.sO_perm.get_tensor(smem_layout_perm)

        # Handle semaphore reset — wait for dependent grids first
        if const_expr(use_pdl and semaphore_to_reset is not None):
            if (
                tidx == 0
                and m_block == cute.arch.grid_dim()[0] - 1
                and k_block == cute.arch.grid_dim()[1] - 1
                and maybe_virtual_batch == cute.arch.grid_dim()[2] - 1
            ):
                cute.arch.griddepcontrol_wait()
                semaphore_to_reset[0] = 0

        # Get number of splits (use maybe_virtual_batch for per-batch-slot splits)
        num_splits = (
            num_splits_dynamic_ptr[maybe_virtual_batch]
            if const_expr(num_splits_dynamic_ptr is not None)
            else mLSE_partial.shape[1]
        )
        # Handle variable length sequences using SeqlenInfo
        seqlen_info = SeqlenInfo.create(
            batch_idx=batch_idx,
            seqlen_static=mO_partial.shape[0],
            cu_seqlens=cu_seqlens,
            seqused=seqused,
            # Don't need to pass in tile size since we won't use offset_padded
        )
        seqlen, offset = seqlen_info.seqlen, seqlen_info.offset

        num_head = mO_partial.shape[3]
        max_idx = seqlen * num_head

        # Early exit for single split if dynamic
        if (const_expr(num_splits_dynamic_ptr is None) or num_splits > 1) and (
            const_expr(not varlen) or m_block * self.tile_m < max_idx
        ):
            # Wait for dependent grids (e.g., the main attention kernel that produces O_partial/LSE_partial)
            if const_expr(use_pdl):
                cute.arch.griddepcontrol_wait()

            # ===============================
            # Step 1: Load LSE_partial from gmem to shared memory
            # ===============================
            # `cLSE` (identity tensor for row/split coord tracking) is reused
            # later in steps 4-5, so it must be defined on both branches.
            cLSE = cute.make_identity_tensor((self.max_splits, self.tile_m))
            # Reshape mLSE_partial to PackGQA packed layout and delegate the
            # tile load to PackGQAComb.load_LSE. The packed form folds (H_q, Sq)
            # into one compound dim with H_q innermost (stride 1), so thread
            # rows that vary along h_pos produce one-sector coalesced reads.
            # Non-varlen path only — varlen keeps the original inline loop.
            if const_expr(not varlen):
                mLSE_partial_cur = seqlen_info.offset_batch(
                    mLSE_partial, batch_idx, dim=3
                )
                # mLSE_partial_cur: (H_q, topK, Sq) — after initial transpose
                # [3,0,2,1] on [topK,B,Sq,H_q] and dropping B.
                # Reorder to (H_q, Sq, topK) then group modes 0..1 for packed dim:
                mLSE_partial_reord = cute.make_tensor(
                    mLSE_partial_cur.iterator,
                    cute.select(mLSE_partial_cur.layout, mode=[0, 2, 1]),
                )
                mLSE_partial_packed = cute.group_modes(mLSE_partial_reord, 0, 2)
                # shape ((H_q, Sq), topK) with H_q innermost.
                packgqa = PackGQAComb(
                    m_block_size=self.tile_m,
                    head_dim_padded=0,  # unused for LSE load
                    check_hdim_oob=False,  # unused for LSE load
                    qhead_per_kvhead=1,  # unused; num_heads_divmod is passed explicitly
                )
                packgqa.load_LSE(
                    mLSE_partial_packed,
                    sLSE,
                    self.max_splits,
                    gmem_tiled_copy_LSE,
                    tidx,
                    m_block,
                    num_splits,
                    seqlen,
                    head_divmod,
                    mSplitCounts,
                    batch_idx,
                    qhead_per_kvhead,
                )
                if const_expr(mLSE_temperature_partial is not None):
                    mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                        mLSE_temperature_partial, batch_idx, dim=3
                    )
                    mLSE_temperature_partial_reord = cute.make_tensor(
                        mLSE_temperature_partial_cur.iterator,
                        cute.select(
                            mLSE_temperature_partial_cur.layout, mode=[0, 2, 1]
                        ),
                    )
                    mLSE_temperature_partial_packed = cute.group_modes(
                        mLSE_temperature_partial_reord, 0, 2
                    )
                    packgqa.load_LSE(
                        mLSE_temperature_partial_packed,
                        sLSE_temperature,
                        self.max_splits,
                        gmem_tiled_copy_LSE,
                        tidx,
                        m_block,
                        num_splits,
                        seqlen,
                        head_divmod,
                        mSplitCounts,
                        batch_idx,
                        qhead_per_kvhead,
                    )
            else:
                # Varlen path keeps the same H_q-innermost flat-row contract:
                # after transpose [1, 0, 2], mLSE_partial_cur is
                # (q_local, split, head).
                # mSplitCounts is the authoritative valid-split count per
                # (batch, q_local, kv_head); masked splits stay at -inf and
                # therefore drop out of the final kernel LSE_out reduction.
                mLSE_partial_cur = seqlen_info.offset_batch(
                    mLSE_partial, batch_idx, dim=3
                )
                mLSE_partial_copy = cute.tiled_divide(mLSE_partial_cur, (1,))
                gmem_thr_copy_LSE = gmem_tiled_copy_LSE.get_slice(tidx)
                tLSEsLSE = gmem_thr_copy_LSE.partition_D(sLSE)
                tLSEsLSE_temperature = gmem_thr_copy_LSE.partition_D(sLSE_temperature)
                tLSEcLSE = gmem_thr_copy_LSE.partition_S(cLSE)
                if const_expr(mLSE_temperature_partial is not None):
                    mLSE_temperature_partial_cur = seqlen_info.offset_batch(
                        mLSE_temperature_partial, batch_idx, dim=3
                    )
                    mLSE_temperature_partial_copy = cute.tiled_divide(
                        mLSE_temperature_partial_cur, (1,)
                    )

                for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
                    mi = tLSEcLSE[0, 0, m][1]
                    idx = m_block * self.tile_m + mi
                    if idx < max_idx:
                        m_idx, head_idx = self.decode_flat_row_idx(idx, head_divmod)
                        row_count = (
                            mSplitCounts[batch_idx, m_idx, head_idx // qhead_per_kvhead]
                            if const_expr(mSplitCounts is not None)
                            else num_splits
                        )
                        mLSE_partial_cur_copy = mLSE_partial_copy[
                            None, m_idx, None, head_idx
                        ]
                        if const_expr(mLSE_temperature_partial is not None):
                            mLSE_temperature_partial_cur_copy = (
                                mLSE_temperature_partial_copy[
                                    None, m_idx, None, head_idx
                                ]
                            )
                        for s in cutlass.range(
                            cute.size(tLSEcLSE, mode=[1]), unroll_full=True
                        ):
                            si = tLSEcLSE[0, s, 0][0]
                            if si < num_splits and si < row_count:
                                cute.copy(
                                    gmem_thr_copy_LSE,
                                    mLSE_partial_cur_copy[None, si],
                                    tLSEsLSE[None, s, m],
                                )
                                if const_expr(mLSE_temperature_partial is not None):
                                    cute.copy(
                                        gmem_thr_copy_LSE,
                                        mLSE_temperature_partial_cur_copy[None, si],
                                        tLSEsLSE_temperature[None, s, m],
                                    )
                            else:
                                tLSEsLSE[None, s, m].fill(-Float32.inf)
                                if const_expr(mLSE_temperature_partial is not None):
                                    tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
                    else:
                        for s in cutlass.range(
                            cute.size(tLSEcLSE, mode=[1]), unroll_full=True
                        ):
                            tLSEsLSE[None, s, m].fill(-Float32.inf)
                            if const_expr(mLSE_temperature_partial is not None):
                                tLSEsLSE_temperature[None, s, m].fill(-Float32.inf)
            cute.arch.cp_async_commit_group()

            # ===============================
            # Step 2: Load O_partial for pipeline stages
            # ===============================

            gmem_thr_copy_O_partial = gmem_tiled_copy_O_partial.get_slice(tidx)
            cO = cute.make_identity_tensor((self.tile_m, self.k_block_size))
            tOcO = gmem_thr_copy_O_partial.partition_D(cO)
            tOsO_partial = gmem_thr_copy_O_partial.partition_D(sO)
            mO_partial_cur = seqlen_info.offset_batch(mO_partial, batch_idx, dim=4)

            # Precompute per-row values for flattened (q_local, head) tiles.
            num_rows = const_expr(cute.size(tOcO, mode=[1]))
            tOmidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOhidx = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOSplitCount = cute.make_rmem_tensor(num_rows, cutlass.Int32)
            tOrOptr = cute.make_rmem_tensor(num_rows, cutlass.Int64)
            for m in cutlass.range(num_rows, unroll_full=True):
                mi = tOcO[0, m, 0][0]  # m coordinate in tile
                idx = m_block * self.tile_m + mi
                if idx >= max_idx:
                    tOhidx[m] = -1
                    tOmidx[m] = 0
                    tOSplitCount[m] = 0
                    tOrOptr[m] = cutlass.Int64(0)
                else:
                    tOmidx[m], tOhidx[m] = self.decode_flat_row_idx(idx, head_divmod)
                    tOSplitCount[m] = (
                        mSplitCounts[
                            batch_idx, tOmidx[m], tOhidx[m] // qhead_per_kvhead
                        ]
                        if const_expr(mSplitCounts is not None)
                        else num_splits
                    )
                    tOrOptr[m] = utils.elem_pointer(
                        mO_partial_cur,
                        (tOmidx[m], k_block * self.k_block_size, 0, tOhidx[m]),
                    ).toint()

            tOpO = None
            if const_expr(not self.is_even_k):
                tOpO = cute.make_rmem_tensor(cute.size(tOcO, mode=[2]), Boolean)
                for k in cutlass.range(cute.size(tOpO), unroll_full=True):
                    tOpO[k] = (
                        tOcO[0, 0, k][1]
                        < mO_partial.shape[1] - k_block * self.k_block_size
                    )
                # if cute.arch.thread_idx()[0] == 0 and k_block == 1: cute.print_tensor(tOpO)

            load_O_partial = partial(
                self.load_O_partial,
                gmem_tiled_copy_O_partial,
                tOrOptr,
                tOsO_partial,
                tOhidx,
                tOSplitCount,
                tOpO,
                tOcO,
                mO_partial_cur.layout,
            )

            # Load first few stages of O_partial
            for stage in cutlass.range(self.stages - 1, unroll_full=True):
                if stage < num_splits:
                    load_O_partial(stage, stage)
                cute.arch.cp_async_commit_group()

            # ===============================
            # Step 3: Load and transpose LSE from smem to registers
            # ===============================

            # Wait for LSE and initial O partial stages to complete
            cute.arch.cp_async_wait_group(self.stages - 1)
            cute.arch.sync_threads()
            # if cute.arch.thread_idx()[0] == 0:
            #     # cute.print_tensor(sLSE)
            #     for i in range(64):
            #         cute.printf("sLSE[%d, 0] = %f", i, sLSE[i, 0])
            # cute.arch.sync_threads()

            s2r_thr_copy_LSE = s2r_tiled_copy_LSE.get_slice(tidx)
            ts2rsLSE = s2r_thr_copy_LSE.partition_S(sLSE)
            ts2rrLSE = cute.make_rmem_tensor_like(ts2rsLSE)
            cute.copy(s2r_tiled_copy_LSE, ts2rsLSE, ts2rrLSE)
            if const_expr(mLSE_temperature_partial is not None):
                ts2rsLSE_temperature = s2r_thr_copy_LSE.partition_S(sLSE_temperature)
                ts2rrLSE_temperature = cute.make_rmem_tensor_like(ts2rsLSE_temperature)
                cute.copy(
                    s2r_tiled_copy_LSE,
                    ts2rsLSE_temperature,
                    ts2rrLSE_temperature,
                )

            # ===============================
            # Step 4: Compute final LSE along split dimension
            # ===============================

            final_lse = cute.make_rmem_tensor(cute.size(ts2rrLSE, mode=[2]), Float32)
            ts2rcLSE = s2r_thr_copy_LSE.partition_D(cLSE)
            # We compute the max valid split for each row to short-circuit the computation later
            max_valid_split = cute.make_rmem_tensor(
                cute.size(ts2rrLSE, mode=[2]), Int32
            )
            assert cute.size(ts2rrLSE, mode=[0]) == 1
            # Compute max, scales, and final LSE for each row. Invalid splits
            # have already been filled with -inf, so Step 5 can write the
            # kernel-native LSE_out directly.
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                # Find max LSE value across splits
                threads_per_col = const_expr(self.smem_threads_per_col_lse)
                lse_max = cute.arch.warp_reduction_max(
                    ts2rrLSE[None, None, m]
                    .load()
                    .reduce(
                        cute.ReductionOp.MAX, init_val=-Float32.inf, reduction_profile=0
                    ),
                    threads_in_group=threads_per_col,
                )
                # if cute.arch.thread_idx()[0] == 0: cute.printf(lse_max)
                # Find max valid split index
                max_valid_idx = -1
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    if ts2rrLSE[0, s, m] != -Float32.inf:
                        max_valid_idx = ts2rcLSE[0, s, 0][0]  # Get split coordinate
                # if cute.arch.thread_idx()[0] < 32: cute.printf(max_valid_idx)
                max_valid_split[m] = cute.arch.warp_reduction_max(
                    max_valid_idx, threads_in_group=threads_per_col
                )
                # Compute exp scales and sum
                lse_max_cur = (
                    0.0 if lse_max == -Float32.inf else lse_max
                )  # In case all local LSEs are -inf
                LOG2_E = math.log2(math.e)
                lse_sum_cur = 0.0
                for s in cutlass.range(cute.size(ts2rrLSE, mode=[1]), unroll_full=True):
                    scale = cute.math.exp2(
                        ts2rrLSE[0, s, m] * LOG2_E - (lse_max_cur * LOG2_E),
                        fastmath=True,
                    )
                    lse_sum_cur += scale
                    ts2rrLSE[0, s, m] = scale  # Store scale for later use
                lse_sum_cur = cute.arch.warp_reduction_sum(
                    lse_sum_cur, threads_in_group=threads_per_col
                )
                # Normalize scales
                inv_sum = 0.0
                if (
                    max_valid_split[m] < 0
                    or lse_sum_cur == 0.0
                    or lse_sum_cur != lse_sum_cur
                ):
                    final_lse[m] = -Float32.inf
                else:
                    final_lse[m] = cute.math.log(lse_sum_cur, fastmath=True) + lse_max
                    inv_sum = 1.0 / lse_sum_cur
                ts2rrLSE[None, None, m].store(ts2rrLSE[None, None, m].load() * inv_sum)
            # Store the scales exp(lse - lse_logsum) back to smem
            cute.copy(s2r_tiled_copy_LSE, ts2rrLSE, ts2rsLSE)

            if const_expr(mLSE_temperature_partial is not None):
                final_lse_temperature = cute.make_rmem_tensor(
                    cute.size(ts2rrLSE_temperature, mode=[2]), Float32
                )
                for m in cutlass.range(
                    cute.size(ts2rrLSE_temperature, mode=[2]), unroll_full=True
                ):
                    threads_per_col = const_expr(self.smem_threads_per_col_lse)
                    lse_temperature_max = cute.arch.warp_reduction_max(
                        ts2rrLSE_temperature[None, None, m]
                        .load()
                        .reduce(
                            cute.ReductionOp.MAX,
                            init_val=-Float32.inf,
                            reduction_profile=0,
                        ),
                        threads_in_group=threads_per_col,
                    )
                    lse_temperature_max_cur = (
                        0.0
                        if lse_temperature_max == -Float32.inf
                        else lse_temperature_max
                    )
                    LOG2_E = math.log2(math.e)
                    lse_temperature_sum_cur = 0.0
                    for s in cutlass.range(
                        cute.size(ts2rrLSE_temperature, mode=[1]), unroll_full=True
                    ):
                        scale = cute.math.exp2(
                            ts2rrLSE_temperature[0, s, m] * LOG2_E
                            - (lse_temperature_max_cur * LOG2_E),
                            fastmath=True,
                        )
                        lse_temperature_sum_cur += scale
                    lse_temperature_sum_cur = cute.arch.warp_reduction_sum(
                        lse_temperature_sum_cur, threads_in_group=threads_per_col
                    )
                    if (
                        max_valid_split[m] < 0
                        or lse_temperature_sum_cur == 0.0
                        or lse_temperature_sum_cur != lse_temperature_sum_cur
                    ):
                        final_lse_temperature[m] = -Float32.inf
                    else:
                        final_lse_temperature[m] = (
                            cute.math.log(lse_temperature_sum_cur, fastmath=True)
                            + lse_temperature_max
                        )

            # Store max valid split to smem
            for m in cutlass.range(cute.size(ts2rrLSE, mode=[2]), unroll_full=True):
                if ts2rcLSE[0, 0, m][0] == 0:  # Only thread responsible for s=0 writes
                    mi = ts2rcLSE[0, 0, m][1]
                    if mi < self.tile_m:
                        sMaxValidSplit[mi] = max_valid_split[m]

            # ===============================
            # Step 5: Store final LSE to gmem
            # This writeback is the authoritative LSE_out returned by the
            # public Sparse Attention / Sparse Page Attention interface.
            # ===============================

            if const_expr(mLSE is not None):
                if const_expr(cu_seqlens is None):
                    mLSE_cur = mLSE[None, None, batch_idx]
                else:
                    mLSE_cur = cute.domain_offset((offset, 0), mLSE)
                if const_expr(mLSE_temperature is not None):
                    if const_expr(cu_seqlens is None):
                        mLSE_temperature_cur = mLSE_temperature[None, None, batch_idx]
                    else:
                        mLSE_temperature_cur = cute.domain_offset(
                            (offset, 0), mLSE_temperature
                        )
                if k_block == 0:  # Only first k_block writes LSE when mLSE is provided
                    for m in cutlass.range(
                        cute.size(ts2rrLSE, mode=[2]), unroll_full=True
                    ):
                        if (
                            ts2rcLSE[0, 0, m][0] == 0
                        ):  # Only thread responsible for s=0 writes
                            mi = ts2rcLSE[0, 0, m][1]
                            idx = m_block * self.tile_m + mi
                            if idx < max_idx:
                                m_idx, head_idx = self.decode_flat_row_idx(
                                    idx, head_divmod
                                )
                                mLSE_cur[m_idx, head_idx] = final_lse[m]
                                if const_expr(mLSE_temperature is not None):
                                    mLSE_temperature_cur[m_idx, head_idx] = (
                                        final_lse_temperature[m]
                                    )

            # ===============================
            # Step 6: Read O_partial and accumulate final O
            # ===============================

            cute.arch.sync_threads()

            # Get max valid split for this thread
            thr_max_valid_split = sMaxValidSplit[tOcO[0, 0, 0][0]]
            for m in cutlass.range(1, cute.size(tOcO, mode=[1]), unroll_full=True):
                thr_max_valid_split = max(
                    thr_max_valid_split, sMaxValidSplit[tOcO[0, m, 0][0]]
                )

            tOrO_partial = cute.make_rmem_tensor_like(tOsO_partial[None, None, None, 0])
            tOrO = cute.make_rmem_tensor_like(tOrO_partial, Float32)
            tOrO.fill(0.0)

            stage_load = self.stages - 1
            stage_compute = 0

            # Main accumulation loop
            for s in cutlass.range(thr_max_valid_split + 1, unroll=4):
                # Get scales for this split
                scale = cute.make_rmem_tensor(num_rows, Float32)
                for m in cutlass.range(num_rows, unroll_full=True):
                    scale[m] = sLSE[s, tOcO[0, m, 0][0]]  # Get scale from smem

                # Load next stage if needed
                split_to_load = s + self.stages - 1
                if split_to_load <= thr_max_valid_split:
                    load_O_partial(split_to_load, stage_load)
                cute.arch.cp_async_commit_group()
                stage_load = 0 if stage_load == self.stages - 1 else stage_load + 1

                # Wait for the current stage to be ready
                cute.arch.cp_async_wait_group(self.stages - 1)
                # We don't need __syncthreads() because each thread is just reading its own data from smem
                # Copy from smem to registers
                cute.autovec_copy(
                    tOsO_partial[None, None, None, stage_compute], tOrO_partial
                )
                stage_compute = (
                    0 if stage_compute == self.stages - 1 else stage_compute + 1
                )

                # Accumulate scaled partial results
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0 and scale[m] > 0.0:
                        tOrO[None, m, None].store(
                            tOrO[None, m, None].load()
                            + scale[m] * tOrO_partial[None, m, None].load().to(Float32)
                        )

            # Flush any outstanding async-copy groups before the local Step-7
            # permutation buffer is read on the tail of the kernel.
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()

            # ===============================
            # Step 7: Write final O to gmem (fake→real via SMEM)
            # ===============================

            mO_cur = seqlen_info.offset_batch(mO, batch_idx, dim=3)
            if const_expr(cu_seqlens is None):
                mO_cur = mO[None, None, None, batch_idx]
            else:
                mO_cur = cute.domain_offset((offset, 0, 0), mO)
            mO_cur = utils.domain_offset_aligned(
                (0, k_block * self.k_block_size, 0), mO_cur
            )
            num_vals = const_expr(cute.size(tOcO, mode=[0]))
            if const_expr(not use_pdl):
                # Direct / standalone calls don't participate in the K1->K2
                # dependency chain. Use a simple per-element real-column store
                # path here to keep mixed-shape launches stable.
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0:
                        for k in cutlass.range(
                            cute.size(tOcO, mode=[2]), unroll_full=True
                        ):
                            if const_expr(self.is_even_k) or tOpO[k]:
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(
                                            fake_col
                                        )
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    mO_cur[tOmidx[m], real_col, tOhidx[m]] = tOrO[
                                        v, m, k
                                    ].to(self.dtype)
            else:
                # 7a: fp32 accumulator -> output dtype SMEM with fake→real
                # permutation. The dedicated permutation buffer stays separate
                # from the O_partial pipeline staging buffer.
                sO_perm = sO_perm_buf

                if const_expr(
                    self.use_stg128_half_layout
                    and self.dtype in [cutlass.BFloat16, cutlass.Float16]
                ):
                    # Half STG.128 fake layout maps each adjacent fake pair to
                    # adjacent real columns. Store each pair as one 32-bit SMEM
                    # transaction instead of emitting two STS.U16 instructions.
                    assert num_vals % 2 == 0
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(
                                cute.size(tOcO, mode=[2]), unroll_full=True
                            ):
                                for v_pair in cutlass.range(
                                    num_vals // 2, unroll_full=True
                                ):
                                    v = v_pair * 2
                                    fake_col = tOcO[v, 0, k][1]
                                    real_col = stg128_half_fake_col_to_real_col(
                                        fake_col
                                    )
                                    smem_ptr = utils.elem_pointer(
                                        sO_perm, (row_local, real_col)
                                    )
                                    if const_expr(self.dtype is cutlass.BFloat16):
                                        sts_32_bf16(
                                            smem_ptr,
                                            tOrO[v, m, k],
                                            tOrO[v + 1, m, k],
                                        )
                                    else:
                                        sts_32_f16(
                                            smem_ptr,
                                            tOrO[v, m, k],
                                            tOrO[v + 1, m, k],
                                        )
                else:
                    # 7a: iterate over ALL val elements in mode[0].
                    # tOcO[v, m, k][1] gives different fake_col for each v.
                    for m in cutlass.range(num_rows, unroll_full=True):
                        row_local = tOcO[0, m, 0][0]
                        if tOhidx[m] >= 0:
                            for k in cutlass.range(
                                cute.size(tOcO, mode=[2]), unroll_full=True
                            ):
                                for v in cutlass.range(num_vals, unroll_full=True):
                                    fake_col = tOcO[v, 0, k][1]
                                    if const_expr(self.use_stg128_half_layout):
                                        real_col = stg128_half_fake_col_to_real_col(
                                            fake_col
                                        )
                                    else:
                                        real_col = stg128_fake_col_to_real_col(fake_col)
                                    sO_perm[row_local, real_col] = tOrO[v, m, k].to(
                                        self.dtype
                                    )

                cute.arch.sync_threads()

                # 7b: SMEM (real order, output dtype) → GMEM
                rO = cute.make_rmem_tensor_like(tOrO, self.dtype)
                elems_per_store = const_expr(
                    cute.size(gmem_tiled_copy_O.layout_tv_tiled[1])
                )
                gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)

                # Read output dtype from SMEM (now in real column order).
                for m in cutlass.range(num_rows, unroll_full=True):
                    row_local = tOcO[0, m, 0][0]
                    for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                        for v in cutlass.range(num_vals, unroll_full=True):
                            col = tOcO[v, 0, k][1]
                            rO[v, m, k] = sO_perm[row_local, col]

                # Write bf16 to GMEM using gmem_tiled_copy_O (same as original FA Step 7)
                for m in cutlass.range(num_rows, unroll_full=True):
                    if tOhidx[m] >= 0:
                        mO_cur_copy = cute.tiled_divide(
                            mO_cur[tOmidx[m], None, tOhidx[m]], (elems_per_store,)
                        )
                        for k in cutlass.range(
                            cute.size(tOcO, mode=[2]), unroll_full=True
                        ):
                            k_idx = tOcO[0, 0, k][1] // elems_per_store
                            if const_expr(self.is_even_k) or tOpO[k]:
                                cute.copy(
                                    gmem_thr_copy_O,
                                    rO[None, m, k],
                                    mO_cur_copy[None, k_idx],
                                )

    @cute.jit
    def load_O_partial(
        self,
        gmem_tiled_copy_O_partial: cute.TiledCopy,
        tOrOptr: cute.Tensor,
        tOsO_partial: cute.Tensor,
        tOhidx: cute.Tensor,
        tOSplitCount: cute.Tensor,
        tOpO: cute.Tensor | None,
        tOcO: cute.Tensor,
        mO_cur_partial_layout: cute.Layout,
        split: Int32,
        stage: Int32,
    ) -> None:
        elems_per_load = const_expr(
            cute.size(gmem_tiled_copy_O_partial.layout_tv_tiled[1])
        )
        tOsO_partial_cur = tOsO_partial[None, None, None, stage]
        for m in cutlass.range(cute.size(tOcO, [1]), unroll_full=True):
            if tOhidx[m] >= 0:
                o_gmem_ptr = cute.make_ptr(
                    tOsO_partial.element_type,
                    tOrOptr[m],
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
                mO_partial_cur = cute.make_tensor(
                    o_gmem_ptr, cute.slice_(mO_cur_partial_layout, (0, None, None, 0))
                )
                mO_partial_cur_copy = cute.tiled_divide(
                    mO_partial_cur, (elems_per_load,)
                )
                for k in cutlass.range(cute.size(tOcO, mode=[2]), unroll_full=True):
                    k_idx = tOcO[0, 0, k][1] // elems_per_load
                    if split < tOSplitCount[m] and (
                        const_expr(tOpO is None) or tOpO[k]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O_partial,
                            mO_partial_cur_copy[None, k_idx, split],
                            tOsO_partial_cur[None, m, k],
                        )
                    else:
                        tOsO_partial_cur[None, m, k].fill(0)


def _get_cutlass_dtype(torch_dtype: torch.dtype):
    if torch_dtype not in torch2cute_dtype_map:
        raise TypeError(f"Unsupported dtype: {torch_dtype}")
    return torch2cute_dtype_map[torch_dtype]


_combine_compile_cache = {}


def _get_cpasync_smem_layout_atom(
    dtype: type[cutlass.Numeric], k_dim: int
) -> cute.ComposedLayout:
    dtype_byte = const_expr(dtype.width // 8)
    bytes_per_row = const_expr(k_dim * dtype_byte)
    smem_k_block_size = (
        const_expr(
            128
            if bytes_per_row % 128 == 0
            else (
                64
                if bytes_per_row % 64 == 0
                else (32 if bytes_per_row % 32 == 0 else 16)
            )
        )
        // dtype_byte
    )
    swizzle_bits = (
        4
        if smem_k_block_size == 128
        else (3 if smem_k_block_size == 64 else (2 if smem_k_block_size == 32 else 1))
    )
    swizzle_base = 2 if dtype_byte == 4 else (3 if dtype_byte == 2 else 4)
    return cute.make_composed_layout(
        cute.make_swizzle(swizzle_bits, swizzle_base, swizzle_base),
        0,
        cute.make_ordered_layout(
            (8 if const_expr(k_dim % 32 == 0) else 16, smem_k_block_size),
            order=(1, 0),
        ),
    )


def combine(
    o_partial_fake,
    lse_partial,
    o_out,
    lse_out,
    *,
    lse_temperature_partial=None,
    lse_temperature_out=None,
    cu_seqlens=None,
    seqused=None,
    split_counts=None,
    use_pdl=False,
):
    """K2: merge sparse forward split partials into the final output.

    STG.128 fake-layout handling remains an internal implementation detail.
    When lse_out is provided, the kernel writes the final authoritative
    log-sum-exp for each query row/head directly into that tensor.

    Args:
        o_partial_fake:
            Batched: [num_splits, batch, Sq, head_q, dim]
            Varlen:  [num_splits, total_q, head_q, dim]
        lse_partial:
            Batched: [num_splits, batch, Sq, head_q]
            Varlen:  [num_splits, total_q, head_q]
        o_out:
            Batched: [batch, Sq, head_q, dim]
            Varlen:  [total_q, head_q, dim]
        lse_out:
            Batched: [batch, Sq, head_q]
            Varlen:  [total_q, head_q]
        lse_temperature_partial:
            Optional temperature-scaled LSE partial with the same shape as
            lse_partial.
        lse_temperature_out:
            Optional temperature-scaled final LSE with the same shape as
            lse_out.
        cu_seqlens: Optional [batch + 1] int32 for varlen-Q combine.
        seqused: Optional [batch] int32 effective lengths for combine.
        split_counts: Optional [batch, max_seqlen_q, head_kv] int32 rowwise
            valid split counts prepared from q2k metadata.
        use_pdl: When True, wait on PDL dependencies from the producer K1
            kernel. When False, launch without PDL waits.
    """
    D = o_partial_fake.shape[-1]
    num_splits = o_partial_fake.shape[0]
    return_temperature_lse = (
        lse_temperature_partial is not None or lse_temperature_out is not None
    )
    if (lse_temperature_partial is None) != (lse_temperature_out is None):
        raise ValueError(
            "lse_temperature_partial and lse_temperature_out must either both be provided or both be None"
        )
    if (
        lse_temperature_partial is not None
        and lse_temperature_partial.shape != lse_partial.shape
    ):
        raise ValueError(
            "lse_temperature_partial must have the same shape as lse_partial, "
            f"got {lse_temperature_partial.shape} vs {lse_partial.shape}"
        )
    if lse_temperature_out is not None:
        if lse_out is None:
            raise ValueError("lse_temperature_out requires lse_out")
        if lse_temperature_out.shape != lse_out.shape:
            raise ValueError(
                "lse_temperature_out must have the same shape as lse_out, "
                f"got {lse_temperature_out.shape} vs {lse_out.shape}"
            )
        if (
            lse_temperature_out.dtype != torch.float32
            or lse_temperature_partial.dtype != torch.float32
        ):
            raise TypeError("temperature LSE tensors must be torch.float32")

    partial_dtype = _get_cutlass_dtype(o_partial_fake.dtype)
    out_dtype = _get_cutlass_dtype(o_out.dtype)
    if split_counts is not None:
        if split_counts.dtype != torch.int32:
            raise TypeError(
                f"split_counts must be torch.int32, got {split_counts.dtype}"
            )
        if split_counts.ndim != 3:
            raise ValueError(
                f"split_counts must have shape [batch, seqlen, head_kv], got {split_counts.shape}"
            )
        if o_out.ndim == 4:
            if split_counts.shape[:2] != o_out.shape[:2]:
                raise ValueError(
                    f"split_counts shape {split_counts.shape} must match batch/seqlen of o_out {o_out.shape}"
                )
        else:
            if cu_seqlens is None:
                raise ValueError("split_counts with varlen output requires cu_seqlens")
            if split_counts.shape[0] != cu_seqlens.shape[0] - 1:
                raise ValueError(
                    f"split_counts batch ({split_counts.shape[0]}) must match cu_seqlens batch "
                    f"({cu_seqlens.shape[0] - 1})"
                )
        if o_out.shape[-2] % split_counts.shape[2] != 0:
            raise ValueError(
                f"o_out heads ({o_out.shape[-2]}) must be divisible by split_counts heads ({split_counts.shape[2]})"
            )
        qheadperkv = o_out.shape[-2] // split_counts.shape[2]
    else:
        qheadperkv = 1
    if cu_seqlens is not None:
        if cu_seqlens.dtype != torch.int32:
            raise TypeError(f"cu_seqlens must be torch.int32, got {cu_seqlens.dtype}")
        if cu_seqlens.ndim != 1:
            raise ValueError(f"cu_seqlens must be rank-1, got {cu_seqlens.shape}")
        if not cu_seqlens.is_contiguous():
            raise ValueError("cu_seqlens must be contiguous")
    if seqused is not None:
        if seqused.dtype != torch.int32:
            raise TypeError(f"seqused must be torch.int32, got {seqused.dtype}")
        if seqused.ndim != 1:
            raise ValueError(f"seqused must be rank-1, got {seqused.shape}")
        if not seqused.is_contiguous():
            raise ValueError("seqused must be contiguous")

    k_block_size = 128 if D > 64 else 64
    tile_m = 64
    log_max_splits = max(math.ceil(math.log2(max(num_splits, 2))), 4)
    has_cu_seqlens = cu_seqlens is not None
    has_seqused = seqused is not None
    has_lse = lse_out is not None
    has_split_counts = split_counts is not None

    key = (
        "combine",
        D,
        k_block_size,
        tile_m,
        log_max_splits,
        partial_dtype,
        out_dtype,
        has_cu_seqlens,
        has_seqused,
        has_lse,
        bool(return_temperature_lse),
        has_split_counts,
        use_pdl,
    )
    if key not in _combine_compile_cache:
        from quack.compile_utils import make_fake_tensor

        kernel = SparseAttentionForwardCombine(
            dtype=out_dtype,
            dtype_partial=partial_dtype,
            head_dim=D,
            tile_m=tile_m,
            k_block_size=k_block_size,
            log_max_splits=log_max_splits,
            use_pdl=use_pdl,
            # stages=2 halves per-block SMEM (168 KB -> 103 KB) -> 2 blocks/SM,
            # theoretical occupancy 12.5% -> 25%. NCU DRAM throughput 76.35%
            # -> 88.64%. Runtime latency within noise (kernel already at HBM
            # bandwidth ceiling in practice) but the cleaner SOL profile
            # matters for downstream NCU comparison.
            stages=2,
        )
        div = 128 // partial_dtype.width
        if has_cu_seqlens:
            ns, total_q, nheads = (cute.sym_int64() for _ in range(3))
            mO_partial = make_fake_tensor(
                partial_dtype, (ns, total_q, nheads, D), divisibility=div
            )
            mLSE_partial = make_fake_tensor(
                Float32, (ns, total_q, nheads), divisibility=1, leading_dim=2
            )
            mO = make_fake_tensor(
                out_dtype, (total_q, nheads, D), divisibility=128 // out_dtype.width
            )
            mLSE = (
                make_fake_tensor(
                    Float32, (total_q, nheads), divisibility=1, leading_dim=1
                )
                if has_lse
                else None
            )
            mLSE_temperature_partial = (
                make_fake_tensor(
                    Float32, (ns, total_q, nheads), divisibility=1, leading_dim=2
                )
                if return_temperature_lse
                else None
            )
            mLSE_temperature = (
                make_fake_tensor(
                    Float32, (total_q, nheads), divisibility=1, leading_dim=1
                )
                if return_temperature_lse
                else None
            )
        else:
            ns, batch, sq, nheads = (cute.sym_int64() for _ in range(4))
            mO_partial = make_fake_tensor(
                partial_dtype, (ns, batch, sq, nheads, D), divisibility=div
            )
            mLSE_partial = make_fake_tensor(
                Float32, (ns, batch, sq, nheads), divisibility=1, leading_dim=3
            )
            mO = make_fake_tensor(
                out_dtype, (batch, sq, nheads, D), divisibility=128 // out_dtype.width
            )
            mLSE = (
                make_fake_tensor(
                    Float32, (batch, sq, nheads), divisibility=1, leading_dim=2
                )
                if has_lse
                else None
            )
            mLSE_temperature_partial = (
                make_fake_tensor(
                    Float32, (ns, batch, sq, nheads), divisibility=1, leading_dim=3
                )
                if return_temperature_lse
                else None
            )
            mLSE_temperature = (
                make_fake_tensor(
                    Float32, (batch, sq, nheads), divisibility=1, leading_dim=2
                )
                if return_temperature_lse
                else None
            )
        if not has_split_counts:
            mSplitCounts = None
        elif has_cu_seqlens:
            batch_ctr, sq_ctr, nheads_kv = (cute.sym_int64() for _ in range(3))
            mSplitCounts = make_fake_tensor(
                Int32, (batch_ctr, sq_ctr, nheads_kv), divisibility=1, leading_dim=2
            )
        else:
            nheads_kv = cute.sym_int64()
            mSplitCounts = make_fake_tensor(
                Int32, (batch, sq, nheads_kv), divisibility=1, leading_dim=2
            )
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

        _combine_compile_cache[key] = cute.compile(
            kernel,
            mO_partial,
            mLSE_partial,
            mO,
            mLSE,
            mLSE_temperature_partial,
            mLSE_temperature,
            None
            if cu_seqlens is None
            else make_fake_tensor(
                Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0
            ),
            None
            if seqused is None
            else make_fake_tensor(
                Int32, (cute.sym_int64(),), divisibility=1, leading_dim=0
            ),
            None,
            None,
            None,
            mSplitCounts,
            Int32(qheadperkv),
            stream,
            options="--enable-tvm-ffi",
        )

    with torch.cuda.nvtx.range("K2_Combine"):
        _combine_compile_cache[key](
            o_partial_fake,
            lse_partial,
            o_out,
            lse_out,
            lse_temperature_partial,
            lse_temperature_out,
            cu_seqlens,
            seqused,
            None,
            None,
            None,
            split_counts,
            qheadperkv,
        )
