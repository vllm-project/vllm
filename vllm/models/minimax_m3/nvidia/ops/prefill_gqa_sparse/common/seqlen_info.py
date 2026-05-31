# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
# mypy: ignore-errors
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Int32, const_expr
from quack import copy_utils

"""
This consolidates all the info related to sequence length. This is so that we can do all
the gmem reads once at the beginning of each tile, rather than having to repeat these reads
to compute various things like n_block_min, n_block_max, etc.
"""


@dataclass(frozen=True)
class SeqlenInfo:
    offset: Int32
    offset_padded: Int32
    seqlen: Int32
    has_cu_seqlens: cutlass.Constexpr[bool] = False

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_static: Int32,
        cu_seqlens: cute.Tensor | None = None,
        seqused: cute.Tensor | None = None,
        tile: cutlass.Constexpr[int] = 128,
    ):
        offset = 0 if const_expr(cu_seqlens is None) else cu_seqlens[batch_idx]
        offset_padded = (
            0
            if const_expr(cu_seqlens is None)
            # Add divby so that the compiler knows the alignment when moving by offset_padded
            else cute.assume((offset + batch_idx * tile) // tile * tile, divby=tile)
        )
        if const_expr(seqused is not None):
            seqlen = seqused[batch_idx]
        elif const_expr(cu_seqlens is not None):
            seqlen = cu_seqlens[batch_idx + 1] - cu_seqlens[batch_idx]
        else:
            seqlen = seqlen_static
        return SeqlenInfo(
            offset, offset_padded, seqlen, has_cu_seqlens=cu_seqlens is not None
        )

    def offset_batch(
        self,
        mT: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        """Offset a tensor by batch index. batch dim is at position `dim`, seqlen is at dim=0."""
        if const_expr(not self.has_cu_seqlens):
            idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mT) - 1 - dim)
            return mT[idx]
        else:
            off = multiple * (
                self.offset if const_expr(not padded) else self.offset_padded
            )
            offset = off if const_expr(cute.rank(mT.shape[0]) == 1) else (0, off)
            idx = (offset,) + (None,) * (cute.rank(mT) - 1)
            return cute.domain_offset(idx, mT)


@dataclass(frozen=True)
class SeqlenInfoQK:
    offset_q: Int32
    offset_k: Int32
    padded_offset_q: Int32
    padded_offset_k: Int32
    seqlen_q: Int32
    seqlen_k: Int32
    has_cu_seqlens_q: cutlass.Constexpr[bool]
    has_cu_seqlens_k: cutlass.Constexpr[bool]
    has_seqused_q: cutlass.Constexpr[bool]
    has_seqused_k: cutlass.Constexpr[bool]

    @staticmethod
    def create(
        batch_idx: Int32,
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
        mCuSeqlensQ: cute.Tensor | None = None,
        mCuSeqlensK: cute.Tensor | None = None,
        mSeqUsedQ: cute.Tensor | None = None,
        mSeqUsedK: cute.Tensor | None = None,
        tile_m: cutlass.Constexpr[Int32] = 128,
        tile_n: cutlass.Constexpr[Int32] = 128,
    ):
        offset_q = 0 if const_expr(mCuSeqlensQ is None) else mCuSeqlensQ[batch_idx]
        offset_k = 0 if const_expr(mCuSeqlensK is None) else mCuSeqlensK[batch_idx]
        padded_offset_q = (
            0
            if const_expr(mCuSeqlensQ is None)
            else cute.assume(
                (offset_q + batch_idx * tile_m) // tile_m * tile_m, divby=tile_m
            )
        )
        padded_offset_k = (
            0
            if const_expr(mCuSeqlensK is None)
            else cute.assume(
                (offset_k + batch_idx * tile_n) // tile_n * tile_n, divby=tile_n
            )
        )
        if const_expr(mSeqUsedQ is not None):
            seqlen_q = mSeqUsedQ[batch_idx]
        else:
            seqlen_q = (
                seqlen_q_static
                if const_expr(mCuSeqlensQ is None)
                else mCuSeqlensQ[batch_idx + 1] - offset_q
            )
        if const_expr(mSeqUsedK is not None):
            seqlen_k = mSeqUsedK[batch_idx]
        else:
            seqlen_k = (
                seqlen_k_static
                if const_expr(mCuSeqlensK is None)
                else mCuSeqlensK[batch_idx + 1] - offset_k
            )
        return SeqlenInfoQK(
            offset_q,
            offset_k,
            padded_offset_q,
            padded_offset_k,
            seqlen_q,
            seqlen_k,
            has_cu_seqlens_q=mCuSeqlensQ is not None,
            has_cu_seqlens_k=mCuSeqlensK is not None,
            has_seqused_q=mSeqUsedQ is not None,
            has_seqused_k=mSeqUsedK is not None,
        )

    def offset_batch_Q(
        self,
        mQ: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        ragged: cutlass.Constexpr[bool] = False,
    ) -> cute.Tensor:
        """Seqlen must be the first dimension of mQ"""
        if const_expr(not ragged):
            if const_expr(not self.has_cu_seqlens_q):
                idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
                return mQ[idx]
            else:
                offset_q = (
                    self.offset_q if const_expr(not padded) else self.padded_offset_q
                )
                offset_q = (
                    offset_q
                    if const_expr(cute.rank(mQ.shape[0]) == 1)
                    else (None, offset_q)
                )
                idx = (offset_q,) + (None,) * (cute.rank(mQ) - 1)
                return cute.domain_offset(idx, mQ)
        else:
            if const_expr(not self.has_cu_seqlens_q):
                offset_q = 0
                idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mQ) - 1 - dim)
                mQ = mQ[idx]
            else:
                offset_q = (
                    self.offset_q if const_expr(not padded) else self.padded_offset_q
                )
            if const_expr(cute.rank(mQ.shape[0]) == 1):
                return copy_utils.offset_ragged_tensor(
                    mQ, offset_q, self.seqlen_q, ragged_dim=0, ptr_shift=True
                )
            else:  # PackGQA
                assert cute.rank(mQ.shape[0]) == 2
                # Unpack before calling offset_ragged_tensor, then pack
                idx = ((None, None),) + (None,) * (cute.rank(mQ) - 1)
                mQ = mQ[idx]
                mQ = copy_utils.offset_ragged_tensor(
                    mQ, offset_q, self.seqlen_q, ragged_dim=1, ptr_shift=True
                )
                return cute.group_modes(mQ, 0, 2)

    def offset_batch_K(
        self,
        mK: cute.Tensor,
        batch_idx: Int32,
        dim: int,
        padded: cutlass.Constexpr[bool] = False,
        ragged: cutlass.Constexpr[bool] = False,
        multiple: int = 1,
    ) -> cute.Tensor:
        """Seqlen must be the first dimension of mK"""
        if const_expr(not ragged):
            if const_expr(not self.has_cu_seqlens_k):
                idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
                return mK[idx]
            else:
                offset_k = (
                    self.offset_k if const_expr(not padded) else self.padded_offset_k
                )
                offset_k *= multiple
                idx = (offset_k,) + (None,) * (cute.rank(mK) - 1)
                return cute.domain_offset(idx, mK)
        else:
            if const_expr(not self.has_cu_seqlens_k):
                offset_k = 0
                idx = (None,) * dim + (batch_idx,) + (None,) * (cute.rank(mK) - 1 - dim)
                mK = mK[idx]
            else:
                offset_k = (
                    self.offset_k if const_expr(not padded) else self.padded_offset_k
                )
                offset_k *= multiple
            return copy_utils.offset_ragged_tensor(
                mK, offset_k, self.seqlen_k, ragged_dim=0, ptr_shift=True
            )
