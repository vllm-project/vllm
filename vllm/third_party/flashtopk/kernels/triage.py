# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""Recovery triage: from the selection's health, which rows recovery rescores, decided and laid out on the device.

A call is healthy, or it has a few bad rows (at most RECOVERY_ROWS, recorded by the selection), or many. One launch:

  * CTA s < RECOVERY_ROWS is slot s of per-row recovery. With a few bad rows, slot s takes the s-th smallest bad row
    (sorted, so the recovery's work does not depend on the order the rows went bad): its output row in row_map, its
    window, its q rows and weights gathered (q in the SM90 kernels' operand layout on SM90). Otherwise its map entry
    is -1 and its window empty.
  * The other CTAs cover the call's rows for whole-call recovery: with many bad rows every row maps to itself and
    keeps its window; otherwise the map is -1 and the windows are empty.

Empty windows make the dense scans that follow nearly free, and a -1 map entry makes the selection skip the slot, so
a healthy call pays a few short launches and no host read.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32

from .common import NEXT_N, NUM_HEADS, select_i32
from .selection import H_BAD, H_OVERFLOW, H_UNDERFILL, RECOVERY_ROWS
from .simt import THREADS


class Triage:
    def __init__(self, words_per_head: int, sm90: bool):
        self.words_per_head = words_per_head
        self.sm90 = sm90

    @cute.jit
    def __call__(
        self,
        health: cute.Tensor,  # [HEALTH_WORDS] i64
        bad_rows: cute.Tensor,  # [RECOVERY_ROWS] i32
        ks: cute.Tensor,  # [num_q] i32
        ke: cute.Tensor,
        q_words: cute.Tensor,  # [num_q, NUM_HEADS * words_per_head] i32: q's bytes
        weights: cute.Tensor,  # [num_q, NUM_HEADS] f32
        row_map: cute.Tensor,  # [RECOVERY_ROWS] i32 out
        slot_ks: cute.Tensor,  # [RECOVERY_ROWS] i32 out
        slot_ke: cute.Tensor,
        q_gather: cute.Tensor,  # [RECOVERY_ROWS * NUM_HEADS * words_per_head] i32 out
        w_gather: cute.Tensor,  # [RECOVERY_ROWS, NUM_HEADS] f32 out
        whole_map: cute.Tensor,  # [num_q] i32 out
        whole_ks: cute.Tensor,  # [num_q] i32 out
        whole_ke: cute.Tensor,
        num_q: Int32,
        stream: cuda.CUstream,
    ):
        grid = RECOVERY_ROWS + (num_q + THREADS - 1) // THREADS
        self.kernel(health, bad_rows, ks, ke, q_words, weights, row_map, slot_ks, slot_ke, q_gather, w_gather,
                    whole_map, whole_ks, whole_ke, num_q).launch(grid=(grid, 1, 1), block=[THREADS, 1, 1],
                                                                 stream=stream)

    @cute.kernel
    def kernel(self, mHealth, mBad, mKs, mKe, mQ, mW, mMap, mSks, mSke, mQg, mWg, mWhole, mWks, mWke, num_q: Int32):
        wph = self.words_per_head
        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        smem = utils.SmemAllocator()
        sRow = smem.allocate_tensor(Int32, cute.make_layout((1,)), 16)

        failed = (mHealth[H_OVERFLOW] != cutlass.Int64(0)) | (mHealth[H_UNDERFILL] != cutlass.Int64(0))
        bad = mHealth[H_BAD]
        by_row = failed & (bad > cutlass.Int64(0)) & (bad <= cutlass.Int64(RECOVERY_ROWS))
        if bid < RECOVERY_ROWS:
            slot = bid
            nbad = Int32(min(bad, cutlass.Int64(RECOVERY_ROWS)))
            if tid == 0:
                sRow[0] = Int32(-1)
            cute.arch.sync_threads()
            if by_row & (slot < nbad) & (tid < nbad):
                mine = mBad[tid]
                rank = Int32(0)
                for j in cutlass.range(nbad, unroll=1):
                    if mBad[j] < mine:
                        rank = rank + 1
                if rank == slot:
                    sRow[0] = mine
            cute.arch.sync_threads()
            row = sRow[0]
            if tid == 0:
                mMap[slot] = row
                s = Int32(0)
                e = Int32(0)
                if row >= 0:
                    s = mKs[row]
                    e = mKe[row]
                mSks[slot] = s
                mSke[slot] = e
            if row >= 0:
                for i in cutlass.range(tid, NUM_HEADS * wph, THREADS, unroll=4):
                    dst = slot * (NUM_HEADS * wph) + i
                    if cutlass.const_expr(self.sm90):
                        # scan_sm90.q_n_of: row 2b + a, head 2c + e of an 8-row tile is operand row
                        # 128a + 8c + 2b + e.
                        head = i // wph
                        r = slot % NEXT_N
                        op_row = (slot // NEXT_N) * (NEXT_N * NUM_HEADS) + 128 * (r % 2) + 8 * (head // 2) \
                            + 2 * (r // 2) + head % 2
                        dst = op_row * wph + (i - head * wph)
                    mQg[dst] = mQ[(row, i)]
                if tid < NUM_HEADS:
                    mWg[(slot, tid)] = mW[(row, tid)]
        else:
            r = (bid - RECOVERY_ROWS) * THREADS + tid
            if r < num_q:
                whole = failed & ((bad <= cutlass.Int64(0)) | (bad > cutlass.Int64(RECOVERY_ROWS)))
                mWhole[r] = select_i32(whole, r, Int32(-1))
                mWks[r] = select_i32(whole, mKs[r], Int32(0))
                mWke[r] = select_i32(whole, mKe[r], Int32(0))
