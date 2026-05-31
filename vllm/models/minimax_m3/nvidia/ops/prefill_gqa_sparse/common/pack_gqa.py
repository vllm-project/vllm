# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# mypy: ignore-errors
"""PackGQA primitives for GQA (grouped-query attention) tile layouts.

Contains:
- ``pack_gqa_layout`` / ``unpack_gqa_layout``: fold/unfold ``qhead_per_kvhead``
  into the seqlen dimension of a tensor layout (zero-copy view).
- ``PackGQA``: base class with ``compute_ptr`` / ``load_Q`` / ``store_LSE`` /
  ``store_O`` helpers for kernels that treat ``(qhead_per_kvhead × seqlen_q)``
  as a single packed row dimension.
- ``PackGQAComb``: subclass used by the K2 combine kernel; adds ``load_LSE``
  for coalesced GMEM→SMEM async copies when LSE_partial is laid out with H_q
  innermost (stride-1).
"""

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute import FastDivmodDivisor
from quack import layout_utils

from . import utils


def pack_gqa_layout(T, qhead_per_kvhead, nheads_kv, head_idx):
    """Reshape a tensor to fold qhead_per_kvhead into the seqlen dimension (mode 0).

    The head dimension is at mode ``head_idx``.  Modes before it (1..head_idx-1)
    are kept as-is (e.g. headdim for Q/O tensors), and modes after it are kept
    as-is (e.g. batch).

    For Q/O tensors (head_idx=2):
        (seqlen_q, headdim, nheads, batch, ...) -> ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch, ...)
    For LSE tensors (head_idx=1):
        (seqlen_q, nheads, batch, ...) -> ((qhead_per_kvhead, seqlen_q), nheads_kv, batch, ...)
    """
    head_stride = T.stride[head_idx]
    shape_packed = (
        (qhead_per_kvhead, T.shape[0]),
        *[T.shape[i] for i in range(1, head_idx)],
        nheads_kv,
        *[T.shape[i] for i in range(head_idx + 1, len(T.shape))],
    )
    stride_packed = (
        (head_stride, T.stride[0]),
        *[T.stride[i] for i in range(1, head_idx)],
        head_stride * qhead_per_kvhead,
        *[T.stride[i] for i in range(head_idx + 1, len(T.shape))],
    )
    return cute.make_tensor(
        T.iterator, cute.make_layout(shape_packed, stride=stride_packed)
    )


def unpack_gqa_layout(T, qhead_per_kvhead, head_idx):
    """Reverse of pack_gqa_layout: unfold qhead_per_kvhead from the seqlen dimension (mode 0).

    The head dimension is at mode ``head_idx``.  Modes before it (1..head_idx-1)
    are kept as-is (e.g. headdim for Q/O tensors), and modes after it are kept
    as-is (e.g. batch).

    For Q/O tensors (head_idx=2):
        ((qhead_per_kvhead, seqlen_q), headdim, nheads_kv, batch, ...) -> (seqlen_q, headdim, nheads, batch, ...)
    For LSE tensors (head_idx=1):
        ((qhead_per_kvhead, seqlen_q), nheads_kv, batch, ...) -> (seqlen_q, nheads, batch, ...)
    """
    seqlen_stride = T.stride[0][1]
    head_stride = T.stride[0][0]
    shape_unpacked = (
        T.shape[0][1],
        *[T.shape[i] for i in range(1, head_idx)],
        T.shape[head_idx] * qhead_per_kvhead,
        *[T.shape[i] for i in range(head_idx + 1, len(T.shape))],
    )
    stride_unpacked = (
        seqlen_stride,
        *[T.stride[i] for i in range(1, head_idx)],
        head_stride,
        *[T.stride[i] for i in range(head_idx + 1, len(T.shape))],
    )
    return cute.make_tensor(
        T.iterator, cute.make_layout(shape_unpacked, stride=stride_unpacked)
    )


@dataclass
class PackGQA:
    m_block_size: cutlass.Constexpr[int]
    head_dim_padded: cutlass.Constexpr[int]
    check_hdim_oob: cutlass.Constexpr[bool]
    qhead_per_kvhead: cutlass.Constexpr[bool]

    @cute.jit
    def compute_ptr(
        self,
        tensor: cute.Tensor,
        cRows: cute.Tensor,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        threads_per_row: cutlass.Constexpr[int],
        num_threads: cutlass.Constexpr[int],
    ):
        num_ptr_per_thread = cute.ceil_div(cute.size(cRows), threads_per_row)
        tPrPtr = cute.make_rmem_tensor(num_ptr_per_thread, cutlass.Int64)
        for i in cutlass.range_constexpr(num_ptr_per_thread):
            row = i * num_threads + cRows[tidx % threads_per_row][0]
            idx = block * self.m_block_size + row
            m_idx = idx // self.qhead_per_kvhead
            h_idx = idx - m_idx * self.qhead_per_kvhead
            tPrPtr[i] = utils.elem_pointer(tensor, ((h_idx, m_idx),)).toint()
        return tPrPtr

    @cute.jit
    def load_Q(
        self,
        mQ: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        sQ: cute.Tensor,  # (m_block_size, head_dim_padded)
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cQ = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tQsQ = gmem_thr_copy.partition_D(sQ)
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=mQ.shape[1])
        tQcQ_row = tQcQ[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, (
            "threads_per_row must divide WARP_SIZE"
        )
        num_threads = gmem_tiled_copy.size
        tPrQPtr = self.compute_ptr(
            mQ[None, 0], tQcQ_row, tidx, block, threads_per_row, num_threads
        )
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            q_ptr_i64 = utils.shuffle_sync(
                tPrQPtr[m // threads_per_row],
                m % threads_per_row,
                width=threads_per_row,
            )
            q_gmem_ptr = cute.make_ptr(
                mQ.element_type, q_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if (
                t0QcQ[0, m, 0][0]
                < seqlen * self.qhead_per_kvhead
                - block * self.m_block_size
                - tQcQ_row[0][0]
            ):
                mQ_cur = cute.make_tensor(q_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tQsQ.shape[0][0])
                mQ_cur_copy = cute.tiled_divide(mQ_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tQsQ.shape[2])):
                    ki = tQcQ[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        mQ_cur_copy[None, ki],
                        tQsQ[None, m, k],
                        pred=tQpQ[None, m, k]
                        if cutlass.const_expr(self.check_hdim_oob)
                        else None,
                    )

    @cute.jit
    def store_LSE(
        self,
        mLSE: cute.Tensor,  # (qhead_per_kvhead, seqlen_q)
        tLSErLSE: cute.Tensor,  # (m_block_size, head_dim_padded)
        tiled_mma: cute.TiledMma,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        thr_mma = tiled_mma.get_slice(tidx)
        caccO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        taccOcO = thr_mma.partition_C(caccO)
        taccOcO_row = layout_utils.reshape_acc_to_mn(taccOcO)[None, 0]
        assert cute.size(tLSErLSE) == cute.size(taccOcO_row)
        threads_per_row = tiled_mma.tv_layout_C.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, (
            "threads_per_row must divide WARP_SIZE"
        )
        assert cute.size(tLSErLSE) <= threads_per_row
        num_threads = tiled_mma.size
        tPrLSEPtr = self.compute_ptr(
            mLSE, taccOcO_row, tidx, block, threads_per_row, num_threads
        )
        for m in cutlass.range_constexpr(cute.size(tLSErLSE)):
            lse_ptr_i64 = utils.shuffle_sync(
                tPrLSEPtr[m // threads_per_row],
                m % threads_per_row,
                width=threads_per_row,
            )
            lse_gmem_ptr = cute.make_ptr(
                mLSE.element_type, lse_ptr_i64, cute.AddressSpace.gmem, assumed_align=4
            )
            row = block * self.m_block_size + taccOcO_row[m][0]
            # Only the thread corresponding to column 0 writes out the lse to gmem
            if taccOcO[0][1] == 0 and row < seqlen * self.qhead_per_kvhead:
                mLSE_copy = cute.make_tensor(lse_gmem_ptr, (1,))
                mLSE_copy[0] = tLSErLSE[m]

    @cute.jit
    def store_O(
        self,
        mO: cute.Tensor,  # ((qhead_per_kvhead, seqlen_q), headdim)
        tOrO: cute.Tensor,  # (m_block_size, head_dim_padded) split across threads according to gmem_tiled_copy
        gmem_tiled_copy: cute.TiledCopy,
        tidx: cutlass.Int32,
        block: cutlass.Int32,
        seqlen: cutlass.Int32,
    ):
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cO = cute.make_identity_tensor((self.m_block_size, self.head_dim_padded))
        tOcO = gmem_thr_copy.partition_S(cO)
        t0OcO = gmem_thr_copy.get_slice(0).partition_S(cO)
        tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
        tOcO_row = tOcO[0, None, 0]
        threads_per_row = gmem_tiled_copy.layout_tv_tiled.shape[0][0]
        assert cute.arch.WARP_SIZE % threads_per_row == 0, (
            "threads_per_row must divide WARP_SIZE"
        )
        num_threads = gmem_tiled_copy.size
        tPrOPtr = self.compute_ptr(
            mO[None, 0], tOcO_row, tidx, block, threads_per_row, num_threads
        )
        for m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
            o_ptr_i64 = utils.shuffle_sync(
                tPrOPtr[m // threads_per_row],
                m % threads_per_row,
                width=threads_per_row,
            )
            o_gmem_ptr = cute.make_ptr(
                mO.element_type, o_ptr_i64, cute.AddressSpace.gmem, assumed_align=16
            )
            if (
                t0OcO[0, m, 0][0]
                < seqlen * self.qhead_per_kvhead
                - block * self.m_block_size
                - tOcO_row[0][0]
            ):
                mO_cur = cute.make_tensor(o_gmem_ptr, (self.head_dim_padded,))
                elems_per_load = cute.size(tOrO.shape[0][0])
                mO_cur_copy = cute.tiled_divide(mO_cur, (elems_per_load,))
                for k in cutlass.range_constexpr(cute.size(tOrO.shape[2])):
                    ki = tOcO[0, 0, k][1] // elems_per_load
                    cute.copy(
                        gmem_thr_copy,
                        tOrO[None, m, k],
                        mO_cur_copy[None, ki],
                        pred=tOpO[None, m, k]
                        if cutlass.const_expr(self.check_hdim_oob)
                        else None,
                    )


@dataclass
class PackGQAComb(PackGQA):
    """PackGQA subclass for the K2 combine kernel.

    Inherits ``compute_ptr`` / ``load_Q`` / ``store_LSE`` / ``store_O`` from
    ``PackGQA``. Adds ``load_LSE`` for coalesced GMEM→SMEM async copies when
    LSE_partial is laid out with H_q innermost.

    K2 combine treats each query head independently (no GQA grouping in combine
    itself), so ``qhead_per_kvhead`` is set to ``num_heads_q`` by the caller —
    all heads are folded into one "group" per Sq position.
    """

    @cute.jit
    def load_LSE(
        self,
        mLSE_partial: cute.Tensor,
        # Packed layout after caller-side reshape:
        #   shape  ((qhead_per_kvhead, seqlen_q), num_splits)
        #   stride ((1, qhead_per_kvhead), ...)
        # — H_q is the innermost (stride-1) element of the packed first dim.
        sLSE: cute.Tensor,
        # SMEM destination: ``(max_splits, m_block_size)`` fp32.
        max_splits: cutlass.Constexpr[int],
        # Explicit max_splits so the identity tensor shape is a plain int,
        # avoiding compound-shape traps from sLSE.shape[0] after tile_to_shape.
        gmem_tiled_copy: cute.TiledCopy,
        tidx: Int32,
        block: Int32,
        num_splits: Int32,
        seqlen: Int32,
        num_heads_divmod: FastDivmodDivisor,
        mCounter: cute.Tensor | None = None,
        batch_idx: Int32 | None = None,
        qhead_per_kvhead: Int32 = Int32(1),
        # divmod for ``m_pos = idx // qhead_per_kvhead``; passed explicitly so
        # caller controls whether the divisor is constexpr or a runtime value.
    ):
        """Coalesced GMEM→SMEM async load of LSE_partial for one tile.

        For each (split, row) slot this thread owns in the tile, compute the
        GMEM coordinate ``(h_pos, m_pos)`` via PackGQA divmod and copy one fp32.
        Out-of-bounds rows (``m_pos >= seqlen``) and splits (``si >= num_splits``)
        are filled with ``-inf`` so they flow cleanly through downstream reductions.

        Coalescing: adjacent thread rows correspond to adjacent ``h_pos`` values
        (head varies fast under ``divmod(idx, qhead_per_kvhead)``), which map to
        adjacent GMEM addresses when H_q is stride-1 — one sector per warp.
        """
        gmem_thr_copy = gmem_tiled_copy.get_slice(tidx)
        cLSE = cute.make_identity_tensor((max_splits, self.m_block_size))
        tLSEcLSE = gmem_thr_copy.partition_S(cLSE)
        tLSEsLSE = gmem_thr_copy.partition_D(sLSE)

        for m in cutlass.range(cute.size(tLSEcLSE, mode=[2]), unroll_full=True):
            mi = tLSEcLSE[0, 0, m][1]
            idx = block * self.m_block_size + mi
            m_pos, h_pos = divmod(idx, num_heads_divmod)

            if m_pos < seqlen:
                row_count = (
                    mCounter[batch_idx, m_pos, h_pos // qhead_per_kvhead]
                    if const_expr(mCounter is not None)
                    else num_splits
                )
                for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                    si = tLSEcLSE[0, s, 0][0]
                    if si < num_splits and si < row_count:
                        # Build a 1-element GMEM tensor at ((h_pos, m_pos), si),
                        # matching PackGQA.store_LSE's ptr pattern so cute.copy
                        # receives a proper Tensor, not a scalar.
                        src_ptr_i64 = utils.elem_pointer(
                            mLSE_partial, ((h_pos, m_pos), si)
                        ).toint()
                        src_ptr = cute.make_ptr(
                            Float32,
                            src_ptr_i64,
                            cute.AddressSpace.gmem,
                            assumed_align=4,
                        )
                        src_t = cute.make_tensor(src_ptr, (1,))
                        cute.copy(gmem_thr_copy, src_t, tLSEsLSE[None, s, m])
                    else:
                        tLSEsLSE[None, s, m].fill(-Float32.inf)
            else:
                for s in cutlass.range(cute.size(tLSEcLSE, mode=[1]), unroll_full=True):
                    tLSEsLSE[None, s, m].fill(-Float32.inf)
