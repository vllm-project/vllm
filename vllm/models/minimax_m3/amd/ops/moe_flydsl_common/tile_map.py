# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""gemm1's expert-major tile order as a device kernel (replaces the ~75 us torch
construction in the benches; one CTA per 256 blocks, a few microseconds).

Sorted rows come in expert order, so the valid m-blocks of an expert are one
contiguous run ``[lo, hi)`` of ``sorted_expert_ids``. gemm1's work list puts every
(m, n) tile of an expert together, n-major inside the expert so consecutive
blocks share the W13 slab::

    tile_map[NB_N*lo + n*(hi - lo) + (m - lo)] = m << 3 | n      for valid m, n < NB_N
    tile_map[valid_blocks*NB_N .. grid)        = -1               (idle blocks)
    tile_map[grid]                              = valid_blocks*NB_N

Thread ``tid`` of CTA ``bx`` handles block ``bx*256 + tid``; ``lo`` / ``hi`` come
from two binary searches over the (non-decreasing) expert ids, branch-free. The
idle tail of the map is filled by all CTAs together.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import range_constexpr

_THREADS = 256
_MAX_BLOCKS = (
    4096  # >= max_sorted / BM: 1409 at 32768 tokens (BM 128) and at 65536 (BM 256)
)
_SEARCH_STEPS = 13  # 2^13 > _MAX_BLOCKS


def compile_tile_map(*, I: int, BM: int = 128):  # noqa: E741
    NB_N = I // 128
    N_CTAS = _MAX_BLOCKS // _THREADS
    TAIL_ITERS = (_MAX_BLOCKS * NB_N + 8) // (_THREADS * N_CTAS) + 1

    @flyc.kernel
    def kernel_tile_map(
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        tile_map: fx.Tensor,
        grid_entries: fx.Int32,
    ):
        tid = fx.thread_idx.x
        bx = fx.block_idx.x
        gtid = bx * _THREADS + tid
        vb = fx.Int32(num_valid_ids[0]) // BM  # valid blocks (num_valid is BM-padded)

        def _eid(i):
            return fx.Int32(sorted_expert_ids[i])

        def _min(a, b):
            return (a < b).select(a, b)

        def _bound(e, upper):
            """first index in [0, vb) whose expert id is > e (upper) / >= e (lower)"""
            lo = fx.Int32(0)
            n = vb
            for _ in range_constexpr(_SEARCH_STEPS):
                half = n // 2
                mid = lo + half
                probe = _eid(_min(mid, vb - 1))
                go_right = (n > 0) & ((probe <= e) if upper else (probe < e))
                lo = go_right.select(mid + 1, lo)
                n = go_right.select(n - half - 1, half)
            return lo

        m = gtid
        if m < vb:
            e = _eid(m)
            lo = _bound(e, False)
            hi = _bound(e, True)
            cnt = hi - lo
            base = lo * NB_N + (m - lo)
            for n in range_constexpr(NB_N):
                tile_map[base + cnt * n] = (m << 3) | n
        tail0 = vb * NB_N
        for it in range_constexpr(TAIL_ITERS):
            i = tail0 + gtid + it * (_THREADS * N_CTAS)
            if i < grid_entries:
                tile_map[i] = fx.Int32(-1)
        if gtid == 0:
            tile_map[grid_entries] = tail0

    @flyc.jit
    def launch_tile_map(
        sorted_expert_ids: fx.Tensor,
        num_valid_ids: fx.Tensor,
        tile_map: fx.Tensor,
        grid_entries: fx.Int32,
        stream: fx.Stream,
    ):
        kernel_tile_map(
            sorted_expert_ids, num_valid_ids, tile_map, grid_entries
        ).launch(grid=(N_CTAS, 1, 1), block=(_THREADS, 1, 1), stream=stream)

    return launch_tile_map


def tile_map_grid(num_m_blocks: int, I: int) -> int:  # noqa: E741
    """gemm1 grid for the table: every (m, n) tile of the allocation, rounded to 8
    XCDs"""
    return (num_m_blocks * (I // 128) + 7) // 8 * 8
