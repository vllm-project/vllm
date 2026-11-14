# Copyright (c) 2026 the FlashTopK authors. Licensed under the Apache License, Version 2.0.
"""SM90 FlashTopK scan in CuTe DSL: FP8 or BF16, 32 heads, head_dim 128, 8-row query tiles.

A CTA scans one 8-row query tile over the union of its rows' KV windows, clipped to its KV split, and appends every
key whose score passes the row's raw gate to the row's candidate buffer as (-score, kv index). Warp roles (384
threads):

  * warps 0-7, two math warpgroups: group g takes the 64-key blocks 2i + g from its own KV ring (stages g*S ..
    g*S + S - 1 of one staged buffer), runs one M=64 x N=256 x K=128 WGMMA per block (8 rows x 32 heads, accumulators
    in registers), scores the block's keys and appends the hits;
  * warp 8, TMA: Q and the weights once, then group 0's KV blocks and kv scales; warp 9 group 1's.

Scores come in a fixed order, so every run gives the same bits. A thread of the WGMMA accumulator holds columns
8i + 2l + e (l = lane % 4) of two keys; the Q operand's rows are permuted (``q_n_of``) so that lane l holds rows 2l and
2l + 1 with all 32 heads, and forms each (row, key) score alone (``score_owned``): heads 0, 2, .., 30 and 1, 3, .., 31
in two fma(max(x, 0), w_h, .) chains, their sum, then the key's kv scale -- no lane exchange. The SM90 refine,
the shared-sample scoring and recovery (the dense variant) use the same functions. Hits are appended with one atomic
per row and
warp: four ballots (row parity x key slot) give every row's hits at its owner lanes, lane r reserves row r's, and each
lane writes its own hits at their popcount positions. With one KV split the CTA owns its rows, so the reservation is
a shared-memory atomic and the counts are written once at the end; with KV splits it is the row's global count.
The two groups take turns issuing their WGMMAs (named barriers 2 and 3, as FlashAttention-3 does): a
group issues only after the other has issued its previous one, so the tensor cores run the two in order and one
group scores while the other's WGMMA runs, rather than both waiting on interleaved WGMMAs at once.

The dense variant scores the shared calibration sample: every key of a row's window, written to cand_val[row, key]
(no gate, no counts), in the scan's bits.
"""

from __future__ import annotations

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warpgroup
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from .common import (
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

BLOCK_KV = 64
NEXT_N = 8
NUM_HEADS = 32
HEAD_DIM = 128
N = NEXT_N * NUM_HEADS


@dsl_user_op
def atomic_add_shared_i32(smem_addr, value, *, loc=None, ip=None) -> Int32:
    return Int32(
        llvm.inline_asm(
            Int32.mlir_type,
            [Int32(smem_addr).ir_value(loc=loc, ip=ip), Int32(value).ir_value(loc=loc, ip=ip)],
            "atom.shared.add.u32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


def q_n_of(r: int, h: int) -> int:
    """The Q operand row (accumulator column) of tile row r's head h: thread lane % 4 == l holds columns 8i + 2l + e,
    so rows 2l and 2l + 1 go to lane l, head pairs (h, h + 1) to one column group."""
    return 8 * ((r % 2) * 16 + h // 2) + 2 * (r // 2) + h % 2


def head_table_layout():
    """The weights of lane l's row 2l + t, heads 4k .. 4k + 3, at (t, k, l): the four lanes of a load read adjacent
    16 bytes."""
    return cute.make_layout((2, 8, 4, 4), stride=(128, 16, 4, 1))


@cute.jit
def fill_head_table(sW, sWq, tidx):
    """One entry per math thread (tidx < 256)."""
    t, k, q, j = tidx // 128, (tidx // 16) % 8, (tidx // 4) % 4, tidx % 4
    sWq[(t, k, q, j)] = sW[(2 * q + t) * NUM_HEADS + 4 * k + j]


@cute.jit
def load4(base, offset):
    """Four floats of shared memory at base + offset (a multiple of 4) into registers, one 16-byte load."""
    ptr = cute.make_ptr(Float32, (base + offset).llvm_ptr, cute.AddressSpace.smem, assumed_align=16)
    dst = cute.make_rmem_tensor((4,), Float32)
    cute.autovec_copy(cute.make_tensor(ptr, cute.make_layout((4,))), dst)
    return dst


def score_chain(acc, w, base, scale):
    """One (row, key) score from the accumulator values acc[base + h % 2 + 4 (h // 2)] of heads h and the weights
    w[h // 4][h % 4]: heads 0, 2, .., 30 and 1, 3, .., 31 in two fma(max(x, 0), w_h, .) chains, their sum, then the
    kv scale. Every SM90 CuTe scorer forms its scores here."""
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
    """Lane 4g + q's scores, [t][slot]: rows 2q + t of its key slots (keys key_lo and key_lo + 8), each formed by
    this lane alone."""
    q4 = lane % 4
    raws = []
    for t in cutlass.range_constexpr(2):
        w = [load4(sWq.iterator, t * 128 + k * 16 + q4 * 4) for k in range(8)]
        raws.append([score_chain(acc, w, 2 * slot + 64 * t, scale0 if slot == 0 else scale1) for slot in range(2)])
    return raws


class ScanSm90:
    """The SM90 FlashTopK scan, for FP8 (with per-key scales) or BF16 operands."""

    def __init__(
        self, *, bf16: bool, single_split: bool, dense: bool = False, pingpong: bool = True
    ):
        self.bf16 = bf16
        self.single_split = single_split and not dense
        self.dense = dense
        # The two groups issue their WGMMAs in turn (named barriers 2 and 3); off, each issues once its stage
        # is in.
        self.pingpong = pingpong
        # Per ring: FP8 stages are 8 KB, BF16 16 KB; with the Q tile (32 / 64 KB) both stay well under 227 KB.
        self.num_stages = 4 if bf16 else 6
        self.threads_per_cta = 384
        self.math_regs = 232
        self.other_regs = 40
        self.acc_dtype = Float32
        self.mma_tiler = (BLOCK_KV, N, HEAD_DIM)

    @cute.jit
    def __call__(
        self,
        kv: cute.Tensor,  # [seq_len_kv, 128] uint8 (FP8 E4M3 bits) or bf16
        q: cute.Tensor,  # [num_q * 32, 128, 1] uint8 or bf16
        weights: cute.Tensor,  # [num_q, 32] f32
        kv_scales: cute.Tensor,  # [>= seq_len_kv] f32 (ones for BF16), padded to 4
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
        if cutlass.const_expr(self.bf16):
            kv_typed, b = kv, q
        else:
            kv_typed = cute.recast_tensor(kv, cutlass.Float8E4M3FN)
            b = cute.recast_tensor(q, cutlass.Float8E4M3FN)
        a = cute.make_tensor(
            kv_typed.iterator, cute.make_layout((seq_len_kv, HEAD_DIM, 1), stride=(kv_typed.layout.stride[0], 1, 0))
        )
        self.a_dtype, self.b_dtype = a.element_type, b.element_type
        a_layout = utils.LayoutEnum.from_tensor(a)
        b_layout = utils.LayoutEnum.ROW_MAJOR
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            a.element_type, b.element_type, OperandMajorMode.K, OperandMajorMode.K, self.acc_dtype, (1, 1, 1),
            (BLOCK_KV, N),
        )
        a_smem_staged = sm90_utils.make_smem_layout_a(a_layout, self.mma_tiler, a.element_type, 2 * self.num_stages)
        b_smem_staged = sm90_utils.make_smem_layout_b(b_layout, self.mma_tiler, b.element_type, 1)
        tma_op = cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            tma_op, a, cute.slice_(a_smem_staged, (None, None, 0)), (BLOCK_KV, HEAD_DIM)
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            tma_op, b, cute.slice_(b_smem_staged, (None, None, 0)), (N, HEAD_DIM)
        )
        w_layout = cute.make_layout((NEXT_N, NUM_HEADS), stride=(NUM_HEADS, 1))
        tma_atom_w, tma_tensor_w = cpasync.make_tiled_tma_atom(tma_op, weights, w_layout, (NEXT_N, NUM_HEADS))
        tma_atom_s, tma_tensor_s = cpasync.make_tiled_tma_atom(
            tma_op, kv_scales, cute.make_layout((BLOCK_KV,)), (BLOCK_KV,)
        )
        elem_bytes = a.element_type.width // 8
        self.kv_tx_bytes = BLOCK_KV * HEAD_DIM * elem_bytes + BLOCK_KV * 4
        self.q_tx_bytes = N * HEAD_DIM * elem_bytes + N * 4

        @cute.struct
        class SharedStorage:
            kv_mbar: cute.struct.MemRange[cutlass.Int64, 2 * self.num_stages * 2]
            q_mbar: cute.struct.MemRange[cutlass.Int64, 2]

        num_tiles = (num_q + NEXT_N - 1) // NEXT_N
        self.kernel(
            tiled_mma, tma_atom_a, tma_tensor_a, tma_atom_b, tma_tensor_b, tma_atom_w, tma_tensor_w, tma_atom_s,
            tma_tensor_s, row_start, row_end, origin, inv_delta, th_bucket, cand_val, cand_idx, cand_cnt, num_q,
            seq_len_kv, num_splits, a_smem_staged, b_smem_staged, w_layout, SharedStorage,
        ).launch(grid=(num_tiles, num_splits, 1), block=[self.threads_per_cta, 1, 1], min_blocks_per_mp=1,
                 stream=stream)

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
        # Longest windows first: under causal masking the last tile's rows see the most keys.
        tile = gdimx - 1 - bidx
        query_start = tile * NEXT_N

        total_blocks = (seq_len_kv + BLOCK_KV - 1) // BLOCK_KV
        blocks_per_split = (total_blocks + num_splits - 1) // num_splits
        split_lo = bidy * blocks_per_split * BLOCK_KV
        split_hi = min((bidy + 1) * blocks_per_split * BLOCK_KV, seq_len_kv)
        range_lo = seq_len_kv
        range_hi = Int32(0)
        for r in cutlass.range_constexpr(NEXT_N):
            row = query_start + r
            s = seq_len_kv
            e = Int32(0)
            if row < num_q:
                s = max(min(mRowStart[row], seq_len_kv), split_lo)
                e = min(min(mRowEnd[row], seq_len_kv), split_hi)
            nonempty = e > s
            range_lo = min(range_lo, select_i32(nonempty, s, seq_len_kv))
            range_hi = max(range_hi, select_i32(nonempty, e, 0))
        has_work = range_hi > range_lo
        # The scale TMA's inner coordinate stays 16-byte aligned.
        kv_start = range_lo // 4 * 4
        num_kv = Int32(0)
        if has_work:
            num_kv = (range_hi - kv_start + BLOCK_KV - 1) // BLOCK_KV

        if warp_idx == 8:
            for atom in (tma_atom_a, tma_atom_b, tma_atom_w, tma_atom_s):
                cpasync.prefetch_descriptor(atom)

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        q_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.q_mbar.data_ptr(), num_stages=1, producer_group=prod_group,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8), tx_count=self.q_tx_bytes,
            tidx=tidx, defer_sync=True,
        )
        kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
        # Both rings' barriers are initialized through their own pipeline objects; then each warp works through one
        # pointing at its ring (group g's math warps consume ring g, TMA warp 8 + g fills it).
        for r in cutlass.range_constexpr(2):
            pipeline.PipelineTmaAsync.create(
                barrier_storage=storage.kv_mbar.data_ptr() + r * S * 2, num_stages=S, producer_group=prod_group,
                consumer_group=kv_cons_group, tx_count=self.kv_tx_bytes, tidx=tidx, defer_sync=True,
            )
        ring = select_i32((warp_idx >= 4) & (warp_idx != 8), Int32(1), Int32(0))
        kv_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr() + ring * S * 2, num_stages=S, producer_group=prod_group,
            consumer_group=kv_cons_group, tx_count=self.kv_tx_bytes, tidx=tidx, defer_sync=True,
        )
        pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)

        sKV = smem.allocate_tensor(
            element_type=self.a_dtype, layout=a_smem_staged.outer, byte_alignment=1024, swizzle=a_smem_staged.inner
        )
        sQ = smem.allocate_tensor(
            element_type=self.b_dtype, layout=b_smem_staged.outer, byte_alignment=1024, swizzle=b_smem_staged.inner
        )
        sW = smem.allocate_tensor(element_type=Float32, layout=cute.make_layout((N,)), byte_alignment=128)
        sWq = smem.allocate_tensor(element_type=Float32, layout=head_table_layout(), byte_alignment=128)
        sScales = smem.allocate_tensor(
            element_type=Float32, layout=cute.make_layout((BLOCK_KV, 2 * S), stride=(1, BLOCK_KV)), byte_alignment=128
        )
        sGate = smem.allocate_tensor(element_type=Float32, layout=cute.make_layout((NEXT_N,)), byte_alignment=16)
        sWin = smem.allocate_tensor(Int32, cute.make_layout((NEXT_N, 2), stride=(2, 1)), 16)
        sCnt = smem.allocate_tensor(Int32, cute.make_layout((NEXT_N,)), 16)

        if tidx < NEXT_N:
            row_q = query_start + tidx
            row = min(row_q, num_q - 1)
            sCnt[tidx] = Int32(0)
            if cutlass.const_expr(self.single_split):
                # cand_cnt accumulates: a row's hits go after what an earlier kernel left (adaptive pruning
                # runs the scan, then the refine on the other rows).
                if row_q < num_q:
                    sCnt[tidx] = mCandCnt[row_q]
            sGate[tidx] = loose_raw_gate(mInvDelta[row], mOrigin[row], mThBucket[row])
            s = seq_len_kv
            e = Int32(0)
            if row_q < num_q:
                s = max(min(mRowStart[row_q], seq_len_kv), split_lo)
                e = min(min(mRowEnd[row_q], seq_len_kv), split_hi)
            sWin[(tidx, 0)] = s
            sWin[(tidx, 1)] = max(e - s, Int32(0))

        # TMA partitions, rebased on this CTA's query tile and KV start.
        gA = cute.local_tile(cute.domain_offset((kv_start, 0, 0), mA), (BLOCK_KV, HEAD_DIM), (None, None, None))
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a, 0, cute.make_layout(1), cute.group_modes(sKV, 0, 2), cute.group_modes(gA, 0, 2)
        )
        gS = cute.local_tile(cute.domain_offset((kv_start,), mS), (BLOCK_KV,), (None,))
        tSsS, tSgS = cpasync.tma_partition(
            tma_atom_s, 0, cute.make_layout(1), cute.group_modes(sScales, 0, 1), gS
        )
        gB = cute.local_tile(cute.domain_offset((query_start * NUM_HEADS, 0, 0), mB), (N, HEAD_DIM), (None, None, None))
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b, 0, cute.make_layout(1), cute.group_modes(sQ, 0, 2), cute.group_modes(gB, 0, 2)
        )
        sW2 = cute.make_tensor(sW.iterator, cute.append(w_layout, cute.make_layout(1)))
        gW = cute.local_tile(cute.domain_offset((query_start, 0), mW), (NEXT_N, NUM_HEADS), (None, None))
        tWsW, tWgW = cpasync.tma_partition(
            tma_atom_w, 0, cute.make_layout(1), cute.group_modes(sW2, 0, 2), cute.group_modes(gW, 0, 2)
        )

        pipeline_init_wait(cluster_shape_mn=(1, 1))
        cute.arch.sync_threads()

        if warp_idx >= 8:
            cute.arch.setmaxregister_decrease(self.other_regs)
            if warp_idx < 10:
                g = warp_idx - 8
                if g == 0:
                    if has_work:
                        q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
                        q_pipeline.producer_acquire(q_prod)
                        q_bar = q_pipeline.producer_get_barrier(q_prod)
                        cute.copy(tma_atom_b, tBgB[(None, 0, 0, 0)], tBsB[(None, 0)], tma_bar_ptr=q_bar)
                        cute.copy(tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar)
                kv_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, S)
                count = (num_kv - g + 1) // 2
                for t in cutlass.range(count, unroll=1):
                    kv_pipeline.producer_acquire(kv_prod)
                    bar = kv_pipeline.producer_get_barrier(kv_prod)
                    stage = g * S + kv_prod.index
                    blk = 2 * t + g
                    cute.copy(tma_atom_a, tAgA[(None, blk, 0, 0)], tAsA[(None, stage)], tma_bar_ptr=bar)
                    cute.copy(tma_atom_s, tSgS[(None, blk)], tSsS[(None, stage)], tma_bar_ptr=bar)
                    kv_prod.advance()
        else:
            cute.arch.setmaxregister_increase(self.math_regs)
            self._math(
                tiled_mma, sKV, sQ, sW, sWq, sScales, sGate, sWin, sCnt, q_pipeline, kv_pipeline, has_work, num_kv,
                kv_start, query_start, tidx, warp_idx, mCandVal, mCandIdx, mCandCnt,
            )
            if cutlass.const_expr(self.single_split):
                cute.arch.barrier(barrier_id=1, number_of_threads=256)
                if tidx < NEXT_N:
                    if query_start + tidx < num_q:
                        mCandCnt[query_start + tidx] = sCnt[tidx]

    @cute.jit
    def _math(
        self, tiled_mma, sKV, sQ, sW, sWq, sScales, sGate, sWin, sCnt, q_pipeline, kv_pipeline, has_work, num_kv,
        kv_start, query_start, tidx, warp_idx, mCandVal, mCandIdx, mCandCnt,
    ):
        S = self.num_stages
        g = warp_idx // 4
        tid_wg = tidx % 128
        lane = tidx % 32
        warp_in_wg = warp_idx % 4
        thr_mma = tiled_mma.get_slice(tid_wg)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sKV))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sQ))
        acc = cute.make_rmem_tensor(thr_mma.partition_shape_C((BLOCK_KV, N)), Float32)
        num_k = cute.size(tCrA.shape[2])
        cap = mCandVal.layout.shape[1]
        lt_mask = (cutlass.Uint32(1) << cutlass.Uint32(lane)) - cutlass.Uint32(1)
        key_lo = 16 * warp_in_wg + lane // 4
        q_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 1)
        kv_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, S)
        if has_work:
            q_pipeline.consumer_wait(q_cons)
            fill_head_table(sW, sWq, tidx)
            cute.arch.barrier(barrier_id=1, number_of_threads=256)
        gates, win_lo, win_len = owned_rows(sGate, sWin, lane)
        count = (num_kv - g + 1) // 2
        count_other = (num_kv - (1 - g) + 1) // 2
        # Software pipelined: block t's hits are written while block t + 1's WGMMA runs.
        stage = Int32(0)
        if count > 0:
            stage = self._issue(tiled_mma, acc, tCrA, tCrB, num_k, kv_pipeline, kv_cons, g, Int32(0), count_other)
        for t in cutlass.range(count, unroll=1):
            blk = 2 * t + g
            warpgroup.wait_group(0)
            scale0 = sScales[(key_lo, stage)]
            scale1 = sScales[(key_lo + 8, stage)]
            kv_pipeline.consumer_release(kv_cons)
            kv_cons.advance()
            key0 = kv_start + blk * BLOCK_KV + key_lo
            key1 = key0 + 8
            raws = score_owned(acc, sWq, lane, scale0, scale1)
            if cutlass.const_expr(self.dense):
                if t + 1 < count:
                    stage = self._issue(tiled_mma, acc, tCrA, tCrB, num_k, kv_pipeline, kv_cons, g, t + 1,
                                        count_other)
                store_owned(raws, lane, key0, key1, win_lo, win_len, query_start, mCandVal)
            else:
                takes, ballots = owned_hits(raws, key0, key1, gates, win_lo, win_len)
                if t + 1 < count:
                    stage = self._issue(tiled_mma, acc, tCrA, tCrB, num_k, kv_pipeline, kv_cons, g, t + 1,
                                        count_other)
                append_owned(raws, takes, ballots, lane, lt_mask, key0, key1, sCnt, query_start, cap,
                             mCandVal, mCandIdx, mCandCnt, self.single_split)

    @cute.jit
    def _issue(self, tiled_mma, acc, tCrA, tCrB, num_k: cutlass.Constexpr, kv_pipeline, kv_cons, g, t, count_other):
        """Wait for block t's KV stage and issue its WGMMA in turn with the other group; returns the stage."""
        kv_pipeline.consumer_wait(kv_cons)
        stage = g * self.num_stages + kv_cons.index
        # My turn: group 0 waits for group 1's previous issue (from its second block), group 1 for group 0's.
        if cutlass.const_expr(self.pingpong):
            if (g == 1) | (t > 0):
                cute.arch.barrier(barrier_id=2 + g, number_of_threads=256)
        warpgroup.fence()
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, False)
        for k in cutlass.range_constexpr(num_k):
            cute.gemm(mma_atom, acc, tCrA[None, None, k, stage], tCrB[None, None, k, 0], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        # The other group's turn, if it has a block left to issue after this one.
        if cutlass.const_expr(self.pingpong):
            if t + g < count_other:
                cute.arch.barrier_arrive(barrier_id=3 - g, number_of_threads=256)
        return stage


@cute.jit
def owned_hits(raws, key0, key1, gates, win_lo, win_len):
    """Whether each owned score passes its row's gate inside the row's window ([t][slot]; gates and windows are
    rows 2q + t's), and the warp's ballots of that: row 2q + t's hits are ballots[t][slot] & (0x11111111 << q)."""
    takes, ballots = [], []
    for t in cutlass.range_constexpr(2):
        take0 = (raws[t][0] > gates[t]) & (cutlass.Uint32(key0 - win_lo[t]) < cutlass.Uint32(win_len[t]))
        take1 = (raws[t][1] > gates[t]) & (cutlass.Uint32(key1 - win_lo[t]) < cutlass.Uint32(win_len[t]))
        takes.append((take0, take1))
        ballots.append((cutlass.Uint32(cute.arch.vote_ballot_sync(take0)),
                        cutlass.Uint32(cute.arch.vote_ballot_sync(take1))))
    return takes, ballots


@cute.jit
def append_owned(raws, takes, ballots, lane, lt_mask, key0, key1, sCnt, query_start, cap, mCandVal, mCandIdx,
                 mCandCnt, single_split: cutlass.Constexpr):
    """Append the hits: lane r < 8 reserves row r's (one round trip for the warp), then every lane writes its own,
    slot-0 hits of a row before its slot-1 hits, each by popcount among the row's owner lanes."""
    any_hits = (ballots[0][0] | ballots[0][1] | ballots[1][0] | ballots[1][1]) != cutlass.Uint32(0)
    if any_hits:
        odd = (lane & 1) != 0
        lane_rows = cutlass.Uint32(0x11111111) << cutlass.Uint32((lane >> 1) & 3)
        c0 = cutlass.Uint32(select_i32(odd, Int32(ballots[1][0]), Int32(ballots[0][0]))) & lane_rows
        c1 = cutlass.Uint32(select_i32(odd, Int32(ballots[1][1]), Int32(ballots[0][1]))) & lane_rows
        mine = select_i32(lane < NEXT_N, Int32(cute.arch.popc(c0)) + Int32(cute.arch.popc(c1)), Int32(0))
        base_lane = Int32(0)
        if mine > 0:
            if cutlass.const_expr(single_split):
                base_lane = atomic_add_shared_i32(sCnt.iterator.toint() + lane * 4, mine)
            else:
                base_lane = atomic_add_global_i32(
                    mCandCnt.iterator.toint() + cutlass.Int64(query_start + lane) * 4, mine
                )
        q4 = lane % 4
        my_rows = cutlass.Uint32(0x11111111) << cutlass.Uint32(q4)
        for t in cutlass.range_constexpr(2):
            row = 2 * q4 + t
            base = cute.arch.shuffle_sync(base_lane, row)
            m0 = ballots[t][0] & my_rows
            m1 = ballots[t][1] & my_rows
            out_row = cutlass.Int64(query_start + row)
            val_row = mCandVal.iterator.toint() + out_row * mCandVal.layout.stride[0] * 4
            idx_row = mCandIdx.iterator.toint() + out_row * mCandIdx.layout.stride[0] * 4
            pos0 = base + Int32(cute.arch.popc(m0 & lt_mask))
            pos1 = base + Int32(cute.arch.popc(m0)) + Int32(cute.arch.popc(m1 & lt_mask))
            if takes[t][0] & (pos0 < cap):
                store_global_b32(val_row + cutlass.Int64(pos0) * 4, fp32_to_i32_bits(-raws[t][0]))
                store_global_b32(idx_row + cutlass.Int64(pos0) * 4, key0)
            if takes[t][1] & (pos1 < cap):
                store_global_b32(val_row + cutlass.Int64(pos1) * 4, fp32_to_i32_bits(-raws[t][1]))
                store_global_b32(idx_row + cutlass.Int64(pos1) * 4, key1)


@cute.jit
def store_owned(raws, lane, key0, key1, win_lo, win_len, query_start, mOut):
    """The owned scores of keys inside their row's window to mOut[row, key] (the shared sample's calibration)."""
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
