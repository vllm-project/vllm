# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 the FlashTopK authors.
# Licensed under the Apache License, Version 2.0.
"""SM90 FlashTopK key pruning in CuTe DSL (FP8 E4M3, 32 heads): the score upper bound,
the first of pruning's two kernels.

The SM100 pruning work split (prune_sm100.py) on scan_sm90.py's pipeline: 16-row tiles
of the pruning operand (16 pruning columns a row), one M=64 x N=256 WGMMA per 64-key
block, the two math warpgroups taking alternate blocks in turn
(sm90_utils.wgmma_in_turn), warps 8 and 9 the TMA of each group's ring (KV, kv scales,
key norms). A thread of the
accumulator holds columns 8i + 2l + e (l = lane % 4) of two keys, so the operand's rows
are permuted (``n_of``) for lane l to hold rows 4l .. 4l + 3 whole: each lane forms its
rows' bounds alone, no lane exchange,

    s_k * (sum_{c < 14} relu(x_c) w_c - relu(x_14 w_14 + x_15 w_15))
        + allowance * (s_k ||k||)

as two interleaved fma(max(x, 0), w, .) chains over the retained columns plus the
aggregate's relu. FP32 rounding in any order is inside plan.SCREEN_ROUNDING. A key is
kept for an 8-row half when some row's bound passes its raw gate (NaN keeps) within its
window: each warp covers 16 consecutive keys, so it writes its half of a mask word
directly (mask[2 * tile + half][block][4], one bit per key, as prune_sm100.py). The
CTA's range starts on a 128-key block and covers whole blocks, so every word the refine
reads is written. Like the scan, a group issues its next block's WGMMA once the partials
are formed, before reducing them.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_wait

from .common import (
    BLOCK_KV,
    HEAD_DIM,
    add_rn,
    fma_rn,
    fmax,
    loose_raw_gate,
    mul_rn,
    select_i32,
)
from .common import PRUNE_BLOCK_Q as BLOCK_Q
from .common import PRUNE_COLS as COLS
from .sm90_utils import (
    CTA_THREADS,
    MATH_BARRIER,
    MATH_THREADS,
    PRODUCER_WARP,
    WARPGROUP_THREADS,
    WARPS_PER_GROUP,
    WGMMA_KEYS,
    allocate_kv_q,
    cta_kv_range,
    group_blocks,
    issue_kv_block,
    load4,
    make_kv_q_operands,
    make_tma_pipelines,
    partition_kv_tma,
    partition_q_tma,
    produce_kv_and_q,
    store_row_window,
)

N = BLOCK_Q * COLS  # the WGMMA's N: the tile's rows x pruning columns
ROW_FLOATS = COLS + 2  # the row's weights by column, its allowance, its raw gate


def n_of(r: int, c: int) -> int:
    """The operand row (accumulator column) of tile row r's pruning column c: thread
    lane % 4 == l holds columns 8i + 2l + e, so rows 4l .. 4l + 3 go to lane l, column
    pairs (c, c + 1) to one column group.

    Reference only, not called by the kernel: the host side builds the permuted
    operand (plan.py) and this documents the permutation."""
    return 8 * ((r % 4) * 8 + c // 2) + 2 * (r // 4) + c % 2


@dsl_user_op
def store_global_b16(address, value, *, loc=None, ip=None) -> None:
    llvm.inline_asm(
        None,
        [
            cutlass.Int64(address).ir_value(loc=loc, ip=ip),
            Int32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.b16 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class PruneSm90:
    def __init__(self):
        self.num_stages = 6
        self.math_regs = 232
        self.other_regs = 40

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [seq_len_kv, 128] uint8 (FP8 E4M3 bits)
        # [num_q * 16, 128, 1] uint8 (FP8 E4M3 bits): the pruning operand, row by row
        q_rows: cute.Tensor,
        w_prune: cute.Tensor,  # [num_q, 16] f32
        kv_scales: cute.Tensor,  # [>= seq_len_kv] f32
        knsc: cute.Tensor,  # [>= seq_len_kv] f32
        row_start: cute.Tensor,
        row_end: cute.Tensor,
        origin: cute.Tensor,
        inv_delta: cute.Tensor,
        th_bucket: cute.Tensor,
        allowance: cute.Tensor,  # [num_q] f32
        mask: cute.Tensor,  # [2 * tiles * mask_blocks * 4] i32
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        stream: cuda.CUstream,
    ):
        (
            tiled_mma,
            a_smem_staged,
            b_smem_staged,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
        ) = make_kv_q_operands(kv, q_rows, seq_len_kv, self.num_stages, N)
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        w_layout = cute.make_layout((BLOCK_Q, COLS), stride=(COLS, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_op, w_prune, w_layout, (BLOCK_Q, COLS)
        )
        vec = cute.make_layout((WGMMA_KEYS,))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_op, kv_scales, vec, (WGMMA_KEYS,)
        )
        tma_atom_n, tma_tensor_n = cpasync.make_tiled_tma_atom(
            tma_op, knsc, vec, (WGMMA_KEYS,)
        )
        # A stage: the block's K rows (one byte an element), kv scales and key norms;
        # the pruning operand and its weights.
        self.kv_tx_bytes = WGMMA_KEYS * HEAD_DIM + 2 * WGMMA_KEYS * 4
        self.q_tx_bytes = N * HEAD_DIM + BLOCK_Q * COLS * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]

        num_tiles = (num_q + BLOCK_Q - 1) // BLOCK_Q
        self.kernel(
            tiled_mma,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_w,
            tma_tensor_w,
            tma_atom_s,
            tma_tensor_s,
            tma_atom_n,
            tma_tensor_n,
            row_start,
            row_end,
            origin,
            inv_delta,
            th_bucket,
            allowance,
            mask,
            num_q,
            seq_len_kv,
            mask_blocks,
            num_splits,
            a_smem_staged,
            b_smem_staged,
            w_layout,
            SharedStorage,
        ).launch(
            grid=(num_tiles, num_splits, 1),
            block=[CTA_THREADS, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB: cute.Tensor,
        tma_atom_w: cute.CopyAtom,
        mW: cute.Tensor,
        tma_atom_s: cute.CopyAtom,
        mS: cute.Tensor,
        tma_atom_n: cute.CopyAtom,
        mN: cute.Tensor,
        mRowStart: cute.Tensor,
        mRowEnd: cute.Tensor,
        mOrigin: cute.Tensor,
        mInvDelta: cute.Tensor,
        mThBucket: cute.Tensor,
        mAllowance: cute.Tensor,
        mMask: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
        mask_blocks: Int32,
        num_splits: Int32,
        a_smem_staged: cute.ComposedLayout,
        b_smem_staged: cute.ComposedLayout,
        w_layout: cute.Layout,
        SharedStorage: cutlass.Constexpr,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, _ = cute.arch.block_idx()
        gdimx, _, _ = cute.arch.grid_dim()
        S = self.num_stages
        q_tile = gdimx - 1 - bidx
        query_start = q_tile * BLOCK_Q

        split_lo, split_hi, range_lo, range_hi, has_work = cta_kv_range(
            BLOCK_KV,
            BLOCK_Q,
            query_start,
            bidy,
            num_q,
            seq_len_kv,
            num_splits,
            mRowStart,
            mRowEnd,
        )
        # Whole mask blocks from a mask-block boundary: every word the refine reads is
        # written.
        kv_start = range_lo // BLOCK_KV * BLOCK_KV
        num_kv = Int32(0)
        if has_work:
            num_kv = (
                (range_hi - kv_start + BLOCK_KV - 1)
                // BLOCK_KV
                * (BLOCK_KV // WGMMA_KEYS)
            )

        if warp_idx == PRODUCER_WARP:
            for atom in (tma_atom_a, tma_atom_b, tma_atom_w, tma_atom_s, tma_atom_n):
                cpasync.prefetch_descriptor(atom)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        q_pipeline, kv_pipeline = make_tma_pipelines(
            storage, S, self.q_tx_bytes, self.kv_tx_bytes, tidx, warp_idx
        )

        sKV, sQ = allocate_kv_q(smem, a_smem_staged, b_smem_staged)
        sW = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout((N,)), byte_alignment=128
        )
        # Per stage: kv scales, then key norms.
        sVec = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout(
                (WGMMA_KEYS, 2 * S, 2), stride=(1, WGMMA_KEYS, 2 * S * WGMMA_KEYS)
            ),
            byte_alignment=128,
        )
        sRow = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((BLOCK_Q, ROW_FLOATS), stride=(ROW_FLOATS, 1)),
            byte_alignment=16,
        )
        sWin = smem.allocate_tensor(
            Int32, cute.make_layout((BLOCK_Q, 2), stride=(2, 1)), 16
        )
        # The weights of lane l's row 4l + rr, columns 4k .. 4k + 3, at (rr, k, l): the
        # four lanes of a load read adjacent 16 bytes.
        sTab = smem.allocate_tensor(
            Float32, cute.make_layout((4, 4, 4, 4), stride=(64, 16, 4, 1)), 128
        )

        if tidx < BLOCK_Q:
            row_q = query_start + tidx
            row = min(row_q, num_q - 1)
            # Review fix (changes codegen): the planner leaves the allowances unwritten
            # when it decides not to prune (then no CTA has work), so read them only
            # with work, as prune_sm100.py does. The allowance column is used only by
            # the math loop, which runs no block without work.
            if has_work:
                sRow[(tidx, COLS)] = mAllowance[row]
            sRow[(tidx, COLS + 1)] = loose_raw_gate(
                mInvDelta[row], mOrigin[row], mThBucket[row]
            )
            store_row_window(
                sWin,
                tidx,
                row_q,
                num_q,
                seq_len_kv,
                split_lo,
                split_hi,
                mRowStart,
                mRowEnd,
            )

        tAsA, tAgA = partition_kv_tma(tma_atom_a, mA, sKV, kv_start)
        sS = sVec[(None, None, 0)]
        sN = sVec[(None, None, 1)]
        gS = cute.local_tile(
            cute.domain_offset((kv_start,), mS), (WGMMA_KEYS,), (None,)
        )
        gN = cute.local_tile(
            cute.domain_offset((kv_start,), mN), (WGMMA_KEYS,), (None,)
        )
        tSsS, tSgS = cpasync.tma_partition(
            tma_atom_s, 0, cute.make_layout(1), cute.group_modes(sS, 0, 1), gS
        )
        tNsN, tNgN = cpasync.tma_partition(
            tma_atom_n, 0, cute.make_layout(1), cute.group_modes(sN, 0, 1), gN
        )
        q_tma = partition_q_tma(
            tma_atom_b,
            mB,
            sQ,
            tma_atom_w,
            mW,
            sW,
            w_layout,
            query_start,
            COLS,
            N,
            (BLOCK_Q, COLS),
        )

        pipeline_init_wait(cluster_shape_mn=(1, 1))
        cute.arch.sync_threads()

        if warp_idx >= PRODUCER_WARP:
            cute.arch.setmaxregister_decrease(self.other_regs)
            # Warps 8 and 9 fill rings 0 and 1 (warp 8 first loads Q); 10 and 11 idle.
            if warp_idx < PRODUCER_WARP + 2:
                produce_kv_and_q(
                    q_pipeline,
                    q_tma,
                    kv_pipeline,
                    tma_atom_a,
                    tAgA,
                    tAsA,
                    [tma_atom_s, tma_atom_n],
                    [tSgS, tNgN],
                    [tSsS, tNsN],
                    has_work,
                    num_kv,
                    S,
                    warp_idx - PRODUCER_WARP,
                )
        else:
            cute.arch.setmaxregister_increase(self.math_regs)
            self._math(
                tiled_mma,
                sKV,
                sQ,
                sW,
                sVec,
                sRow,
                sWin,
                sTab,
                q_pipeline,
                kv_pipeline,
                has_work,
                num_kv,
                kv_start,
                q_tile,
                tidx,
                warp_idx,
                mask_blocks,
                mMask,
            )

    @cute.jit
    def _math(
        self,
        tiled_mma,
        sKV,
        sQ,
        sW,
        sVec,
        sRow,
        sWin,
        sTab,
        q_pipeline,
        kv_pipeline,
        has_work,
        num_kv,
        kv_start,
        q_tile,
        tidx,
        warp_idx,
        mask_blocks,
        mMask,
    ):
        S = self.num_stages
        g = warp_idx // WARPS_PER_GROUP
        tid_wg = tidx % WARPGROUP_THREADS
        lane = tidx % 32
        q4 = lane % 4
        warp_in_wg = warp_idx % WARPS_PER_GROUP
        thr_mma = tiled_mma.get_slice(tid_wg)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sKV))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sQ))
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((WGMMA_KEYS, N)), Float32)
        num_k = cute.size(tCrA.shape[2])
        key_lo = 16 * warp_in_wg + lane // 4
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        kv_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, S)
        if has_work:
            q_pipeline.consumer_wait(q_cons)
            rr, k, lq, j = tidx // 64, (tidx // 16) % 4, (tidx // 4) % 4, tidx % 4
            sTab[(rr, k, lq, j)] = sW[(4 * lq + rr) * COLS + 4 * k + j]
            cute.arch.barrier(barrier_id=MATH_BARRIER, number_of_threads=MATH_THREADS)
        rows = [4 * q4 + rr for rr in range(4)]
        allows = [sRow[(r, COLS)] for r in rows]
        gates = [sRow[(r, COLS + 1)] for r in rows]
        win_lo = [sWin[(r, 0)] for r in rows]
        win_len = [sWin[(r, 1)] for r in rows]
        # Mask half-words: plane 2 * q_tile + half, 8 half-words per 128-key block.
        plane_hw = cutlass.Int64(2 * q_tile) * mask_blocks * 8
        count = group_blocks(num_kv, g)
        count_other = group_blocks(num_kv, 1 - g)
        stage = Int32(0)
        if count > 0:
            stage = issue_kv_block(
                tiled_mma,
                acc,
                tCrA,
                tCrB,
                num_k,
                kv_pipeline,
                kv_cons,
                S,
                g,
                Int32(0),
                count_other,
            )
        for t in cutlass.range(count, unroll=1):
            blk = 2 * t + g
            warpgroup.wait_group(0)
            scales = [sVec[(key_lo + 8 * i, stage, 0)] for i in range(2)]
            norms = [sVec[(key_lo + 8 * i, stage, 1)] for i in range(2)]
            kv_pipeline.consumer_release(kv_cons)
            kv_cons.advance()
            parts = self._rows(acc, sTab, q4)
            if t + 1 < count:
                stage = issue_kv_block(
                    tiled_mma,
                    acc,
                    tCrA,
                    tCrB,
                    num_k,
                    kv_pipeline,
                    kv_cons,
                    S,
                    g,
                    t + 1,
                    count_other,
                )
            key0 = kv_start + blk * WGMMA_KEYS + key_lo
            surv = [cutlass.Boolean(False), cutlass.Boolean(False)]
            for rr in cutlass.range_constexpr(4):
                for i in cutlass.range_constexpr(2):
                    bound = fma_rn(
                        allows[rr], norms[i], mul_rn(scales[i], parts[rr][i])
                    )
                    in_win = cutlass.Uint32(key0 + 8 * i - win_lo[rr]) < cutlass.Uint32(
                        win_len[rr]
                    )
                    # Kept unless the bound is at most the gate (a NaN in either keeps).
                    surv[i] = surv[i] | ((~(bound <= gates[rr])) & in_win)
            # Half h's rows are lanes 2h and 2h + 1 of each quad; a quad's key at bit
            # 4g, then the 8 quads packed.
            hws = [Int32(0), Int32(0)]
            for i in cutlass.range_constexpr(2):
                b = cutlass.Uint32(cute.arch.vote_ballot_sync(surv[i]))
                for h in cutlass.range_constexpr(2):
                    m = (
                        (b >> cutlass.Uint32(2 * h)) | (b >> cutlass.Uint32(2 * h + 1))
                    ) & cutlass.Uint32(0x11111111)
                    m = (m | (m >> cutlass.Uint32(3))) & cutlass.Uint32(0x03030303)
                    m = (m | (m >> cutlass.Uint32(6))) & cutlass.Uint32(0x000F000F)
                    m = (m | (m >> cutlass.Uint32(12))) & cutlass.Uint32(0xFF)
                    hws[h] = hws[h] | (Int32(m) << (8 * i))
            if lane < 2:
                hw = (
                    plane_hw
                    + cutlass.Int64(lane) * mask_blocks * 8
                    + cutlass.Int64((kv_start + blk * WGMMA_KEYS) // 16 + warp_in_wg)
                )
                store_global_b16(
                    mMask.iterator.toint() + hw * 2,
                    select_i32(lane == 0, hws[0], hws[1]),
                )
        if has_work:
            q_pipeline.consumer_release(q_cons)

    @cute.jit
    def _rows(self, acc, sTab, q4):
        """Lane q4's rows 4 q4 + rr, both key slots, before the scale, [rr][slot]: the
        retained columns' relu times weight in two interleaved chains, less the relu of
        the aggregate."""
        parts = []
        for rr in cutlass.range_constexpr(4):
            w = [load4(sTab.iterator, rr * 64 + k * 16 + q4 * 4) for k in range(4)]
            row = []
            for i in cutlass.range_constexpr(2):
                x = [acc[c % 2 + 2 * i + 4 * (rr * 8 + c // 2)] for c in range(COLS)]
                chains = [Float32(0.0), Float32(0.0)]
                for c in cutlass.range_constexpr(COLS - 2):
                    wc = w[c // 4][c % 4]
                    if cutlass.const_expr(c < 2):
                        chains[c % 2] = mul_rn(fmax(x[c], Float32(0.0)), wc)
                    else:
                        chains[c % 2] = fma_rn(
                            fmax(x[c], Float32(0.0)), wc, chains[c % 2]
                        )
                agg = fma_rn(x[COLS - 1], w[3][3], mul_rn(x[COLS - 2], w[3][2]))
                row.append(add_rn(chains[0], chains[1]) - fmax(agg, Float32(0.0)))
            parts.append(row)
        return parts
