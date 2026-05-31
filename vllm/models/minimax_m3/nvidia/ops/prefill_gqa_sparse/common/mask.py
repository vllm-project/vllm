# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa
from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, const_expr

from . import utils
from .seqlen_info import SeqlenInfoQK

MaskGenFn: TypeAlias = Callable[[int], Uint32]
MASK_R2P_CHUNK_SIZE: int = 32


@cute.jit
def r2p_bitmask_below(limit: Int32, s: int) -> Uint32:
    m = max((s + 1) * MASK_R2P_CHUNK_SIZE - limit, 0)
    return utils.shr_u32(Uint32(0xFFFFFFFF), Uint32(m))


@cute.jit
def r2p_bitmask_above(limit: Int32, s: int) -> Uint32:
    n = max(limit - s * MASK_R2P_CHUNK_SIZE, 0)
    return utils.shl_u32(Uint32(0xFFFFFFFF), Uint32(n))


@cute.jit
def mask_r2p_lambda(
    X: cute.Tensor,
    mask_gen_fn: cutlass.Constexpr[MaskGenFn],
    rank1: bool = False,
) -> None:
    ncol = const_expr(
        cute.size(X.shape[cute.rank(X) - 1]) if not rank1 else cute.size(X.shape)
    )
    for s in cutlass.range_constexpr(cute.ceil_div(ncol, MASK_R2P_CHUNK_SIZE)):
        mask = mask_gen_fn(s)
        for i in cutlass.range_constexpr(
            min(MASK_R2P_CHUNK_SIZE, ncol - s * MASK_R2P_CHUNK_SIZE)
        ):
            in_bound = cutlass.Boolean(mask & (Uint32(1) << i))
            c = s * MASK_R2P_CHUNK_SIZE + i
            if const_expr(rank1):
                X[c] = X[c] if in_bound else -Float32.inf
            else:
                for r in cutlass.range_constexpr(cute.size(X.shape[0])):
                    X[r, c] = X[r, c] if in_bound else -Float32.inf


@cute.jit
def row_to_r2p_idx(x: Int32, num_rep: int, num_wg: int) -> Int32:
    return x // (num_rep * num_wg) * num_rep + min(x % (num_rep * num_wg), num_rep)


@dataclass(frozen=True)
class AttentionMask:
    tile_m: cutlass.Constexpr[int]
    tile_n: cutlass.Constexpr[int]
    seqlen_info: SeqlenInfoQK
    qhead_per_kvhead_packgqa: cutlass.Constexpr[int] = 1
    swap_AB: cutlass.Constexpr[bool] = False

    @property
    def seqlen_q(self) -> Int32:
        return self.seqlen_info.seqlen_q

    @property
    def seqlen_k(self) -> Int32:
        return self.seqlen_info.seqlen_k

    @cute.jit
    def apply_mask_sm100_transposed(
        self,
        acc_S: cute.Tensor,
        tScS_t2r: cute.Tensor,
        t0ScS_t2r: cute.Tensor,
        m_block: cutlass.Int32,
        n_block: cutlass.Int32,
        mask_seqlen: cutlass.Constexpr,
        mask_causal: cutlass.Constexpr,
        is_full_block: bool = False,
        check_m_boundary: bool = True,
    ) -> None:
        del is_full_block, check_m_boundary
        del t0ScS_t2r
        row_axis = 0 if const_expr(not self.swap_AB) else 1
        col_axis = 1 if const_expr(not self.swap_AB) else 0
        thr_col_offset = tScS_t2r[0][col_axis]
        seqlenk_col_limit = self.seqlen_k - n_block * self.tile_n - thr_col_offset

        if const_expr(not mask_causal):
            if const_expr(mask_seqlen) and seqlenk_col_limit <= 0:
                for i in cutlass.range(cute.size(acc_S.shape), unroll_full=True):
                    acc_S[i] = -cutlass.Float32.inf
            return

        thr_row_offset = tScS_t2r[0][row_axis]
        seqlenq_row_limit = self.seqlen_q - m_block * self.tile_m - thr_row_offset
        row_limit_top = seqlenq_row_limit - seqlenk_col_limit
        if const_expr(mask_seqlen) and seqlenk_col_limit <= 0:
            row_limit_top = self.tile_m
        num_rep = cute.size(tScS_t2r, mode=[0])
        row_limit = row_to_r2p_idx(row_limit_top, num_rep, 2)
        mask_r2p_lambda(
            acc_S,
            lambda s: r2p_bitmask_above(row_limit, s),
            rank1=True,
        )
