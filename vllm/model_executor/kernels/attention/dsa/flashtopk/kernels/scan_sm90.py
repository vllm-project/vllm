# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 the FlashTopK authors.
# Licensed under the Apache License, Version 2.0.
"""SM90 FlashTopK scan in CuTe DSL: FP8 E4M3 with per-key scales, 32 heads, head_dim
128, 8-row query tiles.

A CTA scans one 8-row query tile over the union of its rows' KV windows, clipped to its
KV split, and appends every key whose score passes the row's raw gate to the row's
candidate buffer as (-score, kv index). Warp roles (384 threads):

  * warps 0-7, two math warpgroups: group g takes the 64-key blocks 2i + g from its own
    KV ring (stages g*S .. g*S + S - 1 of one staged buffer), runs one M=64 x N=256 x
    K=128 WGMMA per block (8 rows x 32 heads, accumulators in registers), scores the
    block's keys and appends the hits;
  * warp 8, TMA: Q and the weights once, then group 0's KV blocks and kv scales; warp 9
    group 1's.

Scores come in a fixed order, so every run gives the same bits. A thread of the WGMMA
accumulator holds columns 8i + 2l + e (l = lane % 4) of two keys; the Q operand's rows
are permuted (``q_n_of``) so that lane l holds rows 2l and 2l + 1 with all 32 heads, and
forms each (row, key) score alone (``score_owned``): heads 0, 2, .., 30 and 1, 3, .., 31
in two fma(max(x, 0), w_h, .) chains, their sum, then the key's kv scale -- no lane
exchange. The SM90 refine, the shared-sample scoring and recovery (the dense variant)
use the same functions. Hits are appended with one atomic per row and warp: four ballots
(row parity x key slot) give every row's hits at its owner lanes, lane r reserves row
r's, and each lane writes its own hits at their popcount positions. With one KV split
the CTA owns its rows, so the reservation is a shared-memory atomic and the counts are
written once at the end; with KV splits it is the row's global count. The two groups
take turns issuing their WGMMAs (named barriers 2 and 3, as FlashAttention-3 does): a
group issues only after the other has issued its previous one, so the tensor cores run
the two in order and one group scores while the other's WGMMA runs, rather than both
waiting on interleaved WGMMAs at once (sm90_utils.wgmma_in_turn).

The dense variant scores the shared calibration sample: every key of a row's window,
written to cand_val[row, key] (no gate, no counts), in the scan's bits.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.pipeline import pipeline_init_wait

from .common import (
    HEAD_DIM,
    NEXT_N,
    NUM_HEADS,
    add_rn,
    atomic_add_global_i32,
    fma_rn,
    fmax,
    fp32_to_i32_bits,
    loose_raw_gate,
    mul_rn,
    select_i32,
    store_global_b32,
)
from .simt import atomic_add_shared_u32
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
    load_row_count,
    make_kv_q_operands,
    make_tma_pipelines,
    partition_kv_tma,
    partition_q_tma,
    produce_kv_and_q,
    store_row_counts,
    store_row_window,
)

N = NEXT_N * NUM_HEADS  # the WGMMA's N: the tile's rows x heads


def q_n_of(r: int, h: int) -> int:
    """The Q operand row (accumulator column) of tile row r's head h: thread lane % 4 ==
    l holds columns 8i + 2l + e, so rows 2l and 2l + 1 go to lane l, head pairs (h, h +
    1) to one column group.

    Reference only, not called by the kernels: the host side builds the permuted
    operand (launch.py, triage.py) and this documents the permutation."""
    return 8 * ((r % 2) * 16 + h // 2) + 2 * (r // 2) + h % 2


def head_table_layout():
    """The weights of lane l's row 2l + t, heads 4k .. 4k + 3, at (t, k, l): the four
    lanes of a load read adjacent 16 bytes."""
    return cute.make_layout((2, 8, 4, 4), stride=(128, 16, 4, 1))


@cute.jit
def fill_head_table(sW, sWq, tidx):
    """One entry per math thread (tidx < MATH_THREADS)."""
    t, k, q, j = tidx // 128, (tidx // 16) % 8, (tidx // 4) % 4, tidx % 4
    sWq[(t, k, q, j)] = sW[(2 * q + t) * NUM_HEADS + 4 * k + j]


def score_chain(acc, w, base, scale):
    """One (row, key) score from the accumulator values acc[base + h % 2 + 4 (h // 2)]
    of heads h and the weights w[h // 4][h % 4]: heads 0, 2, .., 30 and 1, 3, .., 31 in
    two fma(max(x, 0), w_h, .) chains, their sum, then the kv scale. Every SM90 CuTe
    scorer forms its scores here."""
    chains = [Float32(0.0), Float32(0.0)]
    for h in range(NUM_HEADS):
        x = fmax(acc[base + h % 2 + 4 * (h // 2)], Float32(0.0))
        if h < 2:
            chains[h % 2] = mul_rn(x, w[h // 4][h % 4])
        else:
            chains[h % 2] = fma_rn(x, w[h // 4][h % 4], chains[h % 2])
    return mul_rn(scale, add_rn(chains[0], chains[1]))


@cute.jit
def score_owned(acc, sWq, lane, scale0, scale1):
    """Lane 4g + q's scores, [t][slot]: rows 2q + t of its key slots (keys key_lo and
    key_lo + 8), each formed by this lane alone."""
    q4 = lane % 4
    raws = []
    for t in cutlass.range_constexpr(2):
        w = [load4(sWq.iterator, t * 128 + k * 16 + q4 * 4) for k in range(8)]
        raws.append(
            [
                score_chain(acc, w, 2 * slot + 64 * t, scale0 if slot == 0 else scale1)
                for slot in range(2)
            ]
        )
    return raws


class ScanSm90:
    """The SM90 FlashTopK scan (FP8 E4M3 operands with per-key scales)."""

    def __init__(self, *, single_split: bool, dense: bool = False):
        self.single_split = single_split and not dense
        self.dense = dense
        # Per ring: 8 KB FP8 stages; with the 32 KB Q tile well under 227 KB.
        self.num_stages = 6
        self.math_regs = 232
        self.other_regs = 40

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [seq_len_kv, 128] uint8 (FP8 E4M3 bits)
        q: cute.Tensor,  # [num_q * 32, 128, 1] uint8 (FP8 E4M3 bits)
        weights: cute.Tensor,  # [num_q, 32] f32
        kv_scales: cute.Tensor,  # [>= seq_len_kv] f32, padded to 4
        row_start: cute.Tensor,
        row_end: cute.Tensor,
        origin: cute.Tensor,
        inv_delta: cute.Tensor,
        th_bucket: cute.Tensor,
        cand_val: cute.Tensor,
        cand_idx: cute.Tensor,
        cand_cnt: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
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
        ) = make_kv_q_operands(kv, q, seq_len_kv, self.num_stages, N)
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        w_layout = cute.make_layout((NEXT_N, NUM_HEADS), stride=(NUM_HEADS, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(
            tma_op, weights, w_layout, (NEXT_N, NUM_HEADS)
        )
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_op, kv_scales, cute.make_layout((WGMMA_KEYS,)), (WGMMA_KEYS,)
        )
        # A stage: the block's K rows (one byte an element) and kv scales; the Q tile
        # and the weights.
        self.kv_tx_bytes = WGMMA_KEYS * HEAD_DIM + WGMMA_KEYS * 4
        self.q_tx_bytes = N * HEAD_DIM + N * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]

        num_tiles = (num_q + NEXT_N - 1) // NEXT_N
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
            row_start,
            row_end,
            origin,
            inv_delta,
            th_bucket,
            cand_val,
            cand_idx,
            cand_cnt,
            num_q,
            seq_len_kv,
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
        mRowStart: cute.Tensor,
        mRowEnd: cute.Tensor,
        mOrigin: cute.Tensor,
        mInvDelta: cute.Tensor,
        mThBucket: cute.Tensor,
        mCandVal: cute.Tensor,
        mCandIdx: cute.Tensor,
        mCandCnt: cute.Tensor,
        num_q: Int32,
        seq_len_kv: Int32,
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
        # Longest windows first: under causal masking the last tile's rows see the most
        # keys.
        q_tile = gdimx - 1 - bidx
        query_start = q_tile * NEXT_N

        # The scan splits KV in WGMMA_KEYS-key blocks.
        split_lo, split_hi, range_lo, range_hi, has_work = cta_kv_range(
            WGMMA_KEYS,
            NEXT_N,
            query_start,
            bidy,
            num_q,
            seq_len_kv,
            num_splits,
            mRowStart,
            mRowEnd,
        )
        # The scale TMA's inner coordinate stays 16-byte aligned.
        kv_start = range_lo // 4 * 4
        num_kv = Int32(0)
        if has_work:
            num_kv = (range_hi - kv_start + WGMMA_KEYS - 1) // WGMMA_KEYS

        if warp_idx == PRODUCER_WARP:
            for atom in (tma_atom_a, tma_atom_b, tma_atom_w, tma_atom_s):
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
        sWq = smem.allocate_tensor(
            element_type=Float32, layout=head_table_layout(), byte_alignment=128
        )
        sScales = smem.allocate_tensor(
            element_type=Float32,
            layout=cute.make_layout((WGMMA_KEYS, 2 * S), stride=(1, WGMMA_KEYS)),
            byte_alignment=128,
        )
        sGate = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout((NEXT_N,)), byte_alignment=16
        )
        sWin = smem.allocate_tensor(
            Int32, cute.make_layout((NEXT_N, 2), stride=(2, 1)), 16
        )
        sCnt = smem.allocate_tensor(Int32, cute.make_layout((NEXT_N,)), 16)

        if tidx < NEXT_N:
            row_q = query_start + tidx
            row = min(row_q, num_q - 1)
            load_row_count(sCnt, tidx, row_q, num_q, mCandCnt, self.single_split)
            sGate[tidx] = loose_raw_gate(mInvDelta[row], mOrigin[row], mThBucket[row])
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

        # TMA partitions, rebased on this CTA's query tile and KV start.
        tAsA, tAgA = partition_kv_tma(tma_atom_a, mA, sKV, kv_start)
        gS = cute.local_tile(
            cute.domain_offset((kv_start,), mS), (WGMMA_KEYS,), (None,)
        )
        tSsS, tSgS = cpasync.tma_partition(
            tma_atom_s, 0, cute.make_layout(1), cute.group_modes(sScales, 0, 1), gS
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
            NUM_HEADS,
            N,
            (NEXT_N, NUM_HEADS),
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
                    [tma_atom_s],
                    [tSgS],
                    [tSsS],
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
                sWq,
                sScales,
                sGate,
                sWin,
                sCnt,
                q_pipeline,
                kv_pipeline,
                has_work,
                num_kv,
                kv_start,
                query_start,
                tidx,
                warp_idx,
                mCandVal,
                mCandIdx,
                mCandCnt,
            )
            if cutlass.const_expr(self.single_split):
                store_row_counts(sCnt, tidx, query_start, num_q, mCandCnt, NEXT_N)

    @cute.jit
    def _math(
        self,
        tiled_mma,
        sKV,
        sQ,
        sW,
        sWq,
        sScales,
        sGate,
        sWin,
        sCnt,
        q_pipeline,
        kv_pipeline,
        has_work,
        num_kv,
        kv_start,
        query_start,
        tidx,
        warp_idx,
        mCandVal,
        mCandIdx,
        mCandCnt,
    ):
        S = self.num_stages
        g = warp_idx // WARPS_PER_GROUP
        tid_wg = tidx % WARPGROUP_THREADS
        lane = tidx % 32
        warp_in_wg = warp_idx % WARPS_PER_GROUP
        thr_mma = tiled_mma.get_slice(tid_wg)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sKV))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sQ))
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((WGMMA_KEYS, N)), Float32)
        num_k = cute.size(tCrA.shape[2])
        cap = mCandVal.layout.shape[1]
        lt_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        key_lo = 16 * warp_in_wg + lane // 4
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        kv_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, S)
        if has_work:
            q_pipeline.consumer_wait(q_cons)
            fill_head_table(sW, sWq, tidx)
            cute.arch.barrier(barrier_id=MATH_BARRIER, number_of_threads=MATH_THREADS)
        gates, win_lo, win_len = owned_rows(sGate, sWin, lane)
        count = group_blocks(num_kv, g)
        count_other = group_blocks(num_kv, 1 - g)
        # Software pipelined: block t's hits are written while block t + 1's WGMMA runs.
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
            scale0 = sScales[(key_lo, stage)]
            scale1 = sScales[(key_lo + 8, stage)]
            kv_pipeline.consumer_release(kv_cons)
            kv_cons.advance()
            key0 = kv_start + blk * WGMMA_KEYS + key_lo
            key1 = key0 + 8
            raws = score_owned(acc, sWq, lane, scale0, scale1)
            if cutlass.const_expr(self.dense):
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
                store_owned(
                    raws, lane, key0, key1, win_lo, win_len, query_start, mCandVal
                )
            else:
                takes, ballots = owned_hits(raws, key0, key1, gates, win_lo, win_len)
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
                append_owned(
                    raws,
                    takes,
                    ballots,
                    lane,
                    lt_mask,
                    key0,
                    key1,
                    sCnt,
                    query_start,
                    cap,
                    mCandVal,
                    mCandIdx,
                    mCandCnt,
                    self.single_split,
                )


@cute.jit
def owned_hits(raws, key0, key1, gates, win_lo, win_len):
    """Whether each owned score passes its row's gate inside the row's window
    ([t][slot]; gates and windows are rows 2q + t's), and the warp's ballots of that:
    row 2q + t's hits are ballots[t][slot] & (0x11111111 << q)."""
    takes, ballots = [], []
    for t in cutlass.range_constexpr(2):
        take0 = (raws[t][0] > gates[t]) & (
            cutlass.Uint32(key0 - win_lo[t]) < cutlass.Uint32(win_len[t])
        )
        take1 = (raws[t][1] > gates[t]) & (
            cutlass.Uint32(key1 - win_lo[t]) < cutlass.Uint32(win_len[t])
        )
        takes.append((take0, take1))
        ballots.append(
            (
                cutlass.Uint32(cute.arch.vote_ballot_sync(take0)),
                cutlass.Uint32(cute.arch.vote_ballot_sync(take1)),
            )
        )
    return takes, ballots


@cute.jit
def append_owned(
    raws,
    takes,
    ballots,
    lane,
    lt_mask,
    key0,
    key1,
    sCnt,
    query_start,
    cap,
    mCandVal,
    mCandIdx,
    mCandCnt,
    single_split: cutlass.Constexpr,
):
    """Append the hits: lane r < 8 reserves row r's (one round trip for the warp), then
    every lane writes its own, slot-0 hits of a row before its slot-1 hits, each by
    popcount among the row's owner lanes."""
    any_hits = (
        ballots[0][0] | ballots[0][1] | ballots[1][0] | ballots[1][1]
    ) != cutlass.Uint32(0)
    if any_hits:
        odd = (lane & 1) != 0
        lane_rows = cutlass.Uint32(0x11111111) << cutlass.Uint32((lane >> 1) & 3)
        c0 = (
            cutlass.Uint32(select_i32(odd, Int32(ballots[1][0]), Int32(ballots[0][0])))
            & lane_rows
        )
        c1 = (
            cutlass.Uint32(select_i32(odd, Int32(ballots[1][1]), Int32(ballots[0][1])))
            & lane_rows
        )
        mine = select_i32(
            lane < NEXT_N,
            Int32(cute.arch.popc(c0)) + Int32(cute.arch.popc(c1)),
            Int32(0),
        )
        base_lane = Int32(0)
        if mine > 0:
            if cutlass.const_expr(single_split):
                base_lane = atomic_add_shared_u32(
                    sCnt.iterator.toint() + lane * 4, mine
                )
            else:
                base_lane = atomic_add_global_i32(
                    mCandCnt.iterator.toint() + cutlass.Int64(query_start + lane) * 4,
                    mine,
                )
        q4 = lane % 4
        my_rows = cutlass.Uint32(0x11111111) << cutlass.Uint32(q4)
        for t in cutlass.range_constexpr(2):
            row = 2 * q4 + t
            base = cute.arch.shuffle_sync(base_lane, row)
            m0 = ballots[t][0] & my_rows
            m1 = ballots[t][1] & my_rows
            out_row = cutlass.Int64(query_start + row)
            val_row = (
                mCandVal.iterator.toint() + out_row * mCandVal.layout.stride[0] * 4
            )
            idx_row = (
                mCandIdx.iterator.toint() + out_row * mCandIdx.layout.stride[0] * 4
            )
            pos0 = base + Int32(cute.arch.popc(m0 & lt_mask))
            pos1 = (
                base + Int32(cute.arch.popc(m0)) + Int32(cute.arch.popc(m1 & lt_mask))
            )
            if takes[t][0] & (pos0 < cap):
                store_global_b32(
                    val_row + cutlass.Int64(pos0) * 4, fp32_to_i32_bits(-raws[t][0])
                )
                store_global_b32(idx_row + cutlass.Int64(pos0) * 4, key0)
            if takes[t][1] & (pos1 < cap):
                store_global_b32(
                    val_row + cutlass.Int64(pos1) * 4, fp32_to_i32_bits(-raws[t][1])
                )
                store_global_b32(idx_row + cutlass.Int64(pos1) * 4, key1)


@cute.jit
def store_owned(raws, lane, key0, key1, win_lo, win_len, query_start, mOut):
    """The owned scores of keys inside their row's window to mOut[row, key] (the shared
    sample's calibration)."""
    q4 = lane % 4
    for t in cutlass.range_constexpr(2):
        row = query_start + 2 * q4 + t
        if cutlass.Uint32(key0 - win_lo[t]) < cutlass.Uint32(win_len[t]):
            mOut[(row, key0)] = raws[t][0]
        if cutlass.Uint32(key1 - win_lo[t]) < cutlass.Uint32(win_len[t]):
            mOut[(row, key1)] = raws[t][1]


def owned_rows(sGate, sWin, lane):
    """The gates and windows of the lane's rows 2q + t: [gate_t], [lo_t], [len_t]."""
    q4 = lane % 4
    gates = [sGate[2 * q4 + t] for t in range(2)]
    win_lo = [sWin[(2 * q4 + t, 0)] for t in range(2)]
    win_len = [sWin[(2 * q4 + t, 1)] for t in range(2)]
    return gates, win_lo, win_len
