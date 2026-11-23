# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2026 the FlashTopK authors.
# Licensed under the Apache License, Version 2.0.
"""What the SM90 kernels (scan_sm90.py, prune_sm90.py, refine_sm90.py) share: the CTA
layout, the per-CTA KV range, the TMA operand and pipeline setup, and the two math
warpgroups' WGMMA turn-taking.

Every kernel runs 384 threads: warps 0-7 are two math warpgroups (group g = warps
4g .. 4g + 3), warps 8-11 a producer warpgroup (TMA, or the refine's gather). The
@cute.jit helpers inline at trace time, so the kernels generate the same code as
with the bodies written in place.
"""

from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as hopper_helpers
from cutlass import Float32, Int32
from cutlass.cute.nvgpu import OperandMajorMode, cpasync, warpgroup
from cutlass.pipeline import pipeline_init_arrive

from .common import HEAD_DIM, select_i32

WGMMA_KEYS = 64  # keys of one WGMMA (its M): a KV ring stage or a refine compact tile
WARPGROUP_THREADS = 128
WARPS_PER_GROUP = 4
MATH_THREADS = 2 * WARPGROUP_THREADS  # warps 0-7: the two math warpgroups
CTA_THREADS = 3 * WARPGROUP_THREADS  # plus the producer warpgroup, warps 8-11
PRODUCER_WARP = 8  # the producer warpgroup's first warp
# Named barriers. 0 is sync_threads.
MATH_BARRIER = 1  # the two math warpgroups
# WGMMA turn-taking: group g waits on TURN_BARRIER + g and lets the other group go
# with TURN_BARRIER + 1 - g.
TURN_BARRIER = 2


@cute.jit
def load4(base, offset):
    """Four floats of shared memory at base + offset (a multiple of 4) into registers,
    one 16-byte load."""
    ptr = cute.make_ptr(
        Float32, (base + offset).llvm_ptr, cute.AddressSpace.smem, assumed_align=16
    )
    dst = cute.make_rmem_tensor((4,), Float32)
    cute.autovec_copy(cute.make_tensor(ptr, cute.make_layout((4,))), dst)
    return dst


@cute.jit
def clip_window(row, num_q, seq_len_kv, split_lo, split_hi, mRowStart, mRowEnd):
    """Row `row`'s KV window clipped to the CTA's split, [s, e) (empty, e <= s, for a
    row past num_q)."""
    s = seq_len_kv
    e = Int32(0)
    if row < num_q:
        s = max(min(mRowStart[row], seq_len_kv), split_lo)
        e = min(min(mRowEnd[row], seq_len_kv), split_hi)
    return s, e


@cute.jit
def cta_kv_range(
    split_block: cutlass.Constexpr,
    rows: cutlass.Constexpr,
    query_start,
    bidy,
    num_q,
    seq_len_kv,
    num_splits,
    mRowStart,
    mRowEnd,
):
    """KV split bidy (whole split_block-key blocks) and the union of the tile's rows'
    windows inside it: split_lo, split_hi, range_lo, range_hi, has_work."""
    total_blocks = (seq_len_kv + split_block - 1) // split_block
    blocks_per_split = (total_blocks + num_splits - 1) // num_splits
    split_lo = bidy * blocks_per_split * split_block
    split_hi = min((bidy + 1) * blocks_per_split * split_block, seq_len_kv)
    range_lo = seq_len_kv
    range_hi = Int32(0)
    for r in cutlass.range_constexpr(rows):
        s, e = clip_window(
            query_start + r, num_q, seq_len_kv, split_lo, split_hi, mRowStart, mRowEnd
        )
        nonempty = e > s
        range_lo = min(range_lo, select_i32(nonempty, s, seq_len_kv))
        range_hi = max(range_hi, select_i32(nonempty, e, 0))
    has_work = range_hi > range_lo
    return split_lo, split_hi, range_lo, range_hi, has_work


@cute.jit
def store_row_window(
    sWin, tidx, row_q, num_q, seq_len_kv, split_lo, split_hi, mRowStart, mRowEnd
):
    """Row tidx's clipped window to sWin[tidx] as (start, length)."""
    s, e = clip_window(row_q, num_q, seq_len_kv, split_lo, split_hi, mRowStart, mRowEnd)
    sWin[(tidx, 0)] = s
    sWin[(tidx, 1)] = max(e - s, Int32(0))


@cute.jit
def load_row_count(sCnt, tidx, row_q, num_q, mCandCnt, single_split: cutlass.Constexpr):
    """Row tidx's candidate count to sCnt[tidx]: with one KV split the CTA owns its
    rows and appends after what an earlier kernel left in cand_cnt (adaptive pruning
    runs the scan, then the refine on the other rows); else from 0 (unused)."""
    sCnt[tidx] = Int32(0)
    if cutlass.const_expr(single_split):
        if row_q < num_q:
            sCnt[tidx] = mCandCnt[row_q]


@cute.jit
def store_row_counts(sCnt, tidx, query_start, num_q, mCandCnt, rows: cutlass.Constexpr):
    """One KV split: the rows' counts back to cand_cnt, once the math warps are done
    appending (math threads only)."""
    cute.arch.barrier(barrier_id=MATH_BARRIER, number_of_threads=MATH_THREADS)
    if tidx < rows:
        if query_start + tidx < num_q:
            mCandCnt[query_start + tidx] = sCnt[tidx]


def make_kv_q_operands(kv, q, seq_len_kv, num_stages, n):
    """Host side of the TMA-fed scan and pruning: the FP8 KV ([seq_len_kv, 128] E4M3
    bits) as the WGMMA's A (two rings of num_stages WGMMA_KEYS-key stages), the
    n-row Q operand ([rows, 128, 1] E4M3 bits) as its B (one stage), the tiled MMA,
    and both TMA atoms. Returns tiled_mma, a_smem_staged, b_smem_staged, tma_atom_a,
    tma_tensor_a, tma_atom_b, tma_tensor_b."""
    kv_typed = cute.recast_tensor(kv, cutlass.Float8E4M3FN)
    b = cute.recast_tensor(q, cutlass.Float8E4M3FN)
    a = cute.make_tensor(
        kv_typed.iterator,
        cute.make_layout(
            (seq_len_kv, HEAD_DIM, 1), stride=(kv_typed.layout.stride[0], 1, 0)
        ),
    )
    mma_tiler = (WGMMA_KEYS, n, HEAD_DIM)
    tiled_mma = hopper_helpers.make_trivial_tiled_mma(
        a.element_type,
        b.element_type,
        OperandMajorMode.K,
        OperandMajorMode.K,
        Float32,
        (1, 1, 1),
        (WGMMA_KEYS, n),
    )
    a_smem_staged = hopper_helpers.make_smem_layout_a(
        utils.LayoutEnum.from_tensor(a), mma_tiler, a.element_type, 2 * num_stages
    )
    b_smem_staged = hopper_helpers.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, mma_tiler, b.element_type, 1
    )
    tma_op = cpasync.CopyBulkTensorTileG2SOp()
    tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
        tma_op, a, cute.slice_(a_smem_staged, (None, None, 0)), (WGMMA_KEYS, HEAD_DIM)
    )
    tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
        tma_op, b, cute.slice_(b_smem_staged, (None, None, 0)), (n, HEAD_DIM)
    )
    return (
        tiled_mma,
        a_smem_staged,
        b_smem_staged,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
    )


@cute.jit
def make_tma_pipelines(
    storage,
    num_stages: cutlass.Constexpr,
    q_tx_bytes: cutlass.Constexpr,
    kv_tx_bytes: cutlass.Constexpr,
    tidx,
    warp_idx,
):
    """The Q pipeline (one stage, consumed by the eight math warps) and this warp's KV
    ring: both rings' barriers are initialized through their own pipeline objects, then
    each warp works through one pointing at its ring (group g's math warps consume ring
    g, TMA warp PRODUCER_WARP + g fills it). Ends with the barrier-init arrive."""
    prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    q_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.q_mbar.data_ptr(),
        num_stages=1,
        producer_group=prod_group,
        consumer_group=pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 2 * WARPS_PER_GROUP
        ),
        tx_count=q_tx_bytes,
        tidx=tidx,
        defer_sync=True,
    )
    kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, WARPS_PER_GROUP)
    for r in cutlass.range_constexpr(2):
        pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.kv_mbar.data_ptr() + r * num_stages * 2,
            num_stages=num_stages,
            producer_group=prod_group,
            consumer_group=kv_cons_group,
            tx_count=kv_tx_bytes,
            tidx=tidx,
            defer_sync=True,
        )
    ring = select_i32(
        (warp_idx >= WARPS_PER_GROUP) & (warp_idx != PRODUCER_WARP), Int32(1), Int32(0)
    )
    kv_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.kv_mbar.data_ptr() + ring * num_stages * 2,
        num_stages=num_stages,
        producer_group=prod_group,
        consumer_group=kv_cons_group,
        tx_count=kv_tx_bytes,
        tidx=tidx,
        defer_sync=True,
    )
    pipeline_init_arrive(cluster_shape_mn=(1, 1), is_relaxed=True)
    return q_pipeline, kv_pipeline


@cute.jit
def allocate_kv_q(smem, a_smem_staged, b_smem_staged):
    """The KV rings, then the Q tile (FP8, 1024-byte aligned for the swizzle)."""
    sKV = smem.allocate_tensor(
        element_type=cutlass.Float8E4M3FN,
        layout=a_smem_staged.outer,
        byte_alignment=1024,
        swizzle=a_smem_staged.inner,
    )
    sQ = smem.allocate_tensor(
        element_type=cutlass.Float8E4M3FN,
        layout=b_smem_staged.outer,
        byte_alignment=1024,
        swizzle=b_smem_staged.inner,
    )
    return sKV, sQ


@cute.jit
def partition_kv_tma(tma_atom_a, mA, sKV, kv_start):
    """The KV TMA partition from key kv_start, in WGMMA_KEYS-key blocks:
    (tAsA, tAgA)."""
    gA = cute.local_tile(
        cute.domain_offset((kv_start, 0, 0), mA),
        (WGMMA_KEYS, HEAD_DIM),
        (None, None, None),
    )
    return cpasync.tma_partition(
        tma_atom_a,
        0,
        cute.make_layout(1),
        cute.group_modes(sKV, 0, 2),
        cute.group_modes(gA, 0, 2),
    )


@cute.jit
def partition_q_tma(
    tma_atom_b,
    mB,
    sQ,
    tma_atom_w,
    mW,
    sW,
    w_layout,
    query_start,
    operand_rows: cutlass.Constexpr,
    n: cutlass.Constexpr,
    w_tile: cutlass.Constexpr,
):
    """The TMA partitions of the CTA's Q tile (n operand rows, operand_rows a query
    row) and weights (a w_tile tile), for issue_q_tma: (tma_atom_b, tBgB, tBsB,
    tma_atom_w, tWgW, tWsW)."""
    gB = cute.local_tile(
        cute.domain_offset((query_start * operand_rows, 0, 0), mB),
        (n, HEAD_DIM),
        (None, None, None),
    )
    tBsB, tBgB = cpasync.tma_partition(
        tma_atom_b,
        0,
        cute.make_layout(1),
        cute.group_modes(sQ, 0, 2),
        cute.group_modes(gB, 0, 2),
    )
    sW2 = cute.make_tensor(sW.iterator, cute.append(w_layout, cute.make_layout(1)))
    gW = cute.local_tile(cute.domain_offset((query_start, 0), mW), w_tile, (None, None))
    tWsW, tWgW = cpasync.tma_partition(
        tma_atom_w,
        0,
        cute.make_layout(1),
        cute.group_modes(sW2, 0, 2),
        cute.group_modes(gW, 0, 2),
    )
    return tma_atom_b, tBgB, tBsB, tma_atom_w, tWgW, tWsW


@cute.jit
def issue_q_tma(q_tma, q_bar):
    """The Q tile and the weights onto mbarrier q_bar."""
    tma_atom_b, tBgB, tBsB, tma_atom_w, tWgW, tWsW = q_tma
    cute.copy(tma_atom_b, tBgB[(None, 0, 0, 0)], tBsB[(None, 0)], tma_bar_ptr=q_bar)
    cute.copy(tma_atom_w, tWgW[(None, 0, 0)], tWsW[(None, 0)], tma_bar_ptr=q_bar)


def group_blocks(n, g):
    """How many of n blocks math group g takes (blocks g, g + 2, ..)."""
    return (n - g + 1) // 2


@cute.jit
def produce_kv_and_q(
    q_pipeline,
    q_tma,
    kv_pipeline,
    tma_atom_a,
    tAgA,
    tAsA,
    vec_atoms,
    vec_gmem,
    vec_smem,
    has_work,
    num_kv,
    num_stages: cutlass.Constexpr,
    g,
):
    """TMA warp PRODUCER_WARP + g: warp PRODUCER_WARP first loads Q and the weights
    (if the CTA has work), then fills ring g with blocks 2t + g, each its WGMMA_KEYS
    K rows and the per-key vectors (vec_atoms[i] from vec_gmem[i] to vec_smem[i])."""
    if g == 0:
        if has_work:
            q_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 1)
            q_pipeline.producer_acquire(q_prod)
            q_bar = q_pipeline.producer_get_barrier(q_prod)
            issue_q_tma(q_tma, q_bar)
    kv_prod = pipeline.make_pipeline_state(
        pipeline.PipelineUserType.Producer, num_stages
    )
    count = group_blocks(num_kv, g)
    for t in cutlass.range(count, unroll=1):
        kv_pipeline.producer_acquire(kv_prod)
        bar = kv_pipeline.producer_get_barrier(kv_prod)
        stage = g * num_stages + kv_prod.index
        blk = 2 * t + g
        cute.copy(
            tma_atom_a, tAgA[(None, blk, 0, 0)], tAsA[(None, stage)], tma_bar_ptr=bar
        )
        for i in cutlass.range_constexpr(len(vec_atoms)):
            cute.copy(
                vec_atoms[i],
                vec_gmem[i][(None, blk)],
                vec_smem[i][(None, stage)],
                tma_bar_ptr=bar,
            )
        kv_prod.advance()


@cute.jit
def wgmma_in_turn(
    tiled_mma, acc, tCrA, tCrB, num_k: cutlass.Constexpr, stage, g, t, count_other
):
    """Math group g's WGMMA of its t-th block (operand A stage `stage`), in turn with
    the other group, as FlashAttention-3 does: group 0 waits for group 1's previous
    issue (from its second block), group 1 for group 0's; after committing, it lets
    the other group go if that one has a block left to issue. The tensor cores then
    run the two groups' WGMMAs in order, and one group scores while the other's runs.
    The caller waits for its operands first and for the result (wait_group) after."""
    if (g == 1) | (t > 0):
        cute.arch.barrier(barrier_id=TURN_BARRIER + g, number_of_threads=MATH_THREADS)
    warpgroup.fence()
    mma_atom = cute.make_mma_atom(tiled_mma.op)
    mma_atom.set(warpgroup.Field.ACCUMULATE, False)
    for k in cutlass.range_constexpr(num_k):
        cute.gemm(
            mma_atom, acc, tCrA[None, None, k, stage], tCrB[None, None, k, 0], acc
        )
        mma_atom.set(warpgroup.Field.ACCUMULATE, True)
    warpgroup.commit_group()
    if t + g < count_other:
        cute.arch.barrier_arrive(
            barrier_id=TURN_BARRIER + 1 - g, number_of_threads=MATH_THREADS
        )


@cute.jit
def issue_kv_block(
    tiled_mma,
    acc,
    tCrA,
    tCrB,
    num_k: cutlass.Constexpr,
    kv_pipeline,
    kv_cons,
    num_stages: cutlass.Constexpr,
    g,
    t,
    count_other,
):
    """Wait for group g's t-th KV stage and issue its WGMMA in turn with the other
    group; returns the stage (scan and pruning)."""
    kv_pipeline.consumer_wait(kv_cons)
    stage = g * num_stages + kv_cons.index
    wgmma_in_turn(tiled_mma, acc, tCrA, tCrB, num_k, stage, g, t, count_other)
    return stage
