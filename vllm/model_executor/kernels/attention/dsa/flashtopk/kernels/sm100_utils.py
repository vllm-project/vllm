# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright (c) 2025 by FlashInfer team.
# Copyright (c) 2026 the FlashTopK authors.
# Licensed under the Apache License, Version 2.0.
#
# The UMMA, TMEM and pipeline set-up below is moved out of scan_sm100.py,
# prune_sm100.py and refine_sm100.py, which derive from FlashInfer's selective-logits
# candidate kernel (flashinfer/attn_scores/kernels/fp8_paged_mqa_logits.py), itself
# adapted from TensorRT-LLM's CuTe DSL FP8 paged MQA logits kernel (Copyright (c) 2026
# NVIDIA CORPORATION & AFFILIATES).
"""What the SM100 kernels (scan_sm100.py, prune_sm100.py, refine_sm100.py) share:
their warp-role constants, the UMMA, TMEM and pipeline set-up, the CTA's KV range, and
the head arithmetic of a score. Each helper emits the operations of the code it
replaced, in the same order."""

from __future__ import annotations

from typing import Any

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as blackwell
from cutlass import Float32, Int32
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import tcgen05
from cutlass.cutlass_dsl import dsl_user_op

from .common import BLOCK_KV, HEAD_DIM, abs_f32, fp32_to_i32_bits, select_i32
from .simt import bits_f32

RND_RN = "rn"  # the rounding of the packed FP32 instructions
CLUSTER_SHAPE_MN = (1, 1)
NUM_MATH_WG = 2  # math warpgroup w takes the blocks (or compact tiles) 2i + w
NUM_MATH_WARPS = 4 * NUM_MATH_WG
NUM_MATH_THREADS = 32 * NUM_MATH_WARPS
NUM_EPI_SUBTILES = 4  # TMEM is read in four 64-column subtiles
NUM_W_IN_REG = 16  # heads whose weights a math thread keeps in registers
# Named barriers. 0 is __syncthreads.
TMEM_ALLOC_BARRIER_ID = 1  # the math warps and the UMMA warp retrieve the allocation
TMEM_ALLOC_THREADS = NUM_MATH_THREADS + 32
MATH_BARRIER_ID = 2  # the math warps only


@dsl_user_op
def named_barrier_sync_unaligned(count, *, loc=None, ip=None) -> None:
    """Sync the TMEM-retrieve barrier without the warp-aligned requirement."""
    llvm.inline_asm(
        None,
        [Int32(count).ir_value(loc=loc, ip=ip)],
        f"barrier.cta.sync {TMEM_ALLOC_BARRIER_ID}, $0;",
        "r,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


class UmmaKernel:
    """The attributes setup_umma sets on an SM100 kernel (declared for type checking;
    no runtime state of its own)."""

    N: int
    a_dtype: Any
    b_dtype: Any
    mma_tiler: tuple[int, int, int]
    cluster_layout_vmnk: Any
    a_smem_layout_staged: Any
    b_smem_layout_staged: Any
    num_tmem_alloc_cols: int


def setup_umma(kernel: UmmaKernel, a_dtype, b_dtype, a_major, b_major, num_a_stages):
    """The kernel's M=BLOCK_KV x N=kernel.N x K=HEAD_DIM MMA on one CTA: sets the
    operand types, the tiler, the trivial cluster layout, the staged A and single-stage
    B shared-memory layouts and one accumulator's TMEM columns on `kernel`, and returns
    the tiled MMA."""
    kernel.a_dtype, kernel.b_dtype = a_dtype, b_dtype
    tiled_mma = blackwell.make_trivial_tiled_mma(
        a_dtype,
        b_dtype,
        a_major,
        b_major,
        Float32,
        tcgen05.CtaGroup.ONE,
        (BLOCK_KV, kernel.N),
    )
    k = cute.size(tiled_mma.shape_mnk, mode=[2])
    kernel.mma_tiler = (BLOCK_KV, kernel.N, k * (HEAD_DIM // k))
    kernel.cluster_layout_vmnk = cute.tiled_divide(
        cute.make_layout((*CLUSTER_SHAPE_MN, 1)), (tiled_mma.thr_id.shape,)
    )
    kernel.a_smem_layout_staged = blackwell.make_smem_layout_a(
        tiled_mma, kernel.mma_tiler, a_dtype, num_a_stages
    )
    kernel.b_smem_layout_staged = blackwell.make_smem_layout_b(
        tiled_mma, kernel.mma_tiler, b_dtype, 1
    )
    acc_shape = tiled_mma.partition_shape_C(kernel.mma_tiler[:2])
    kernel.num_tmem_alloc_cols = utils.get_num_tmem_alloc_cols(
        tiled_mma.make_fragment_C(acc_shape)
    )
    return tiled_mma


def make_kv_q_operands(kernel: UmmaKernel, kv, q, seq_len_kv, num_kv_stages: int):
    """Host TMA setup shared by ScanSm100 and PruneSm100: tiled UMMA, staged KV and
    single-stage Q TMA atoms and tensors, and the Q SMEM slice."""
    kv_typed = cute.recast_tensor(kv, cutlass.Float8E4M3FN)
    b = cute.recast_tensor(q, cutlass.Float8E4M3FN)
    a = cute.make_tensor(
        kv_typed.iterator,
        cute.make_layout(
            (seq_len_kv, HEAD_DIM, 1), stride=(kv_typed.layout.stride[0], 1, 0)
        ),
    )
    tiled_mma = setup_umma(
        kernel,
        a.element_type,
        b.element_type,
        utils.LayoutEnum.from_tensor(a).mma_major_mode(),
        utils.LayoutEnum.ROW_MAJOR.mma_major_mode(),
        num_kv_stages,
    )
    tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
        blackwell.cluster_shape_to_tma_atom_A(CLUSTER_SHAPE_MN, tiled_mma.thr_id),
        a,
        kernel.a_smem_layout_staged,
        kernel.mma_tiler,
        tiled_mma,
        kernel.cluster_layout_vmnk.shape,
    )
    b_smem_layout = cute.slice_(kernel.b_smem_layout_staged, (None, None, None, 0))
    tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
        blackwell.cluster_shape_to_tma_atom_B(CLUSTER_SHAPE_MN, tiled_mma.thr_id),
        b,
        b_smem_layout,
        kernel.mma_tiler,
        tiled_mma,
        kernel.cluster_layout_vmnk.shape,
    )
    return (
        tiled_mma,
        tma_atom_a,
        tma_tensor_a,
        tma_atom_b,
        tma_tensor_b,
        b.element_type,
        b_smem_layout,
    )


@cute.jit
def make_q_kv_umma_pipelines(
    storage,
    num_kv_stages: cutlass.Constexpr,
    q_tx_bytes: cutlass.Constexpr,
    kv_tx_bytes: cutlass.Constexpr,
    cluster_layout_vmnk,
    tidx,
    warpgroup_idx,
):
    """Single-stage Q TMA pipeline, per-group KV TMA pipelines, and per-group UMMA
    pipelines for ScanSm100 and PruneSm100."""
    prod_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
    q_pipeline = pipeline.PipelineTmaAsync.create(
        barrier_storage=storage.q_mbar.data_ptr(),
        num_stages=1,
        producer_group=prod_group,
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 8),
        tx_count=q_tx_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
        tidx=tidx,
        defer_sync=True,
    )
    kv_cons_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 4)
    kv_0, kv_1, kv_math, math_ring = make_group_pipelines(
        pipeline.PipelineTmaAsync.create,
        storage.kv_mbar.data_ptr,
        lambda g: g * num_kv_stages * 2,
        lambda: min(warpgroup_idx, Int32(1)),
        num_stages=num_kv_stages,
        producer_group=prod_group,
        consumer_group=kv_cons_group,
        tx_count=kv_tx_bytes,
        cta_layout_vmnk=cluster_layout_vmnk,
        tidx=tidx,
        defer_sync=True,
    )
    umma_cons_group = pipeline.CooperativeGroup(
        pipeline.Agent.Thread, NUM_MATH_THREADS // 2
    )
    umma_0, umma_1, umma_math, _ = make_group_pipelines(
        create_umma_pipeline,
        storage.umma_mbar.data_ptr,
        lambda g: g * 2,
        math_ring,
        consumer_group=umma_cons_group,
        cta_layout_vmnk=cluster_layout_vmnk,
    )
    return (
        q_pipeline,
        (kv_0, kv_1),
        kv_math,
        (umma_0, umma_1),
        umma_math,
        math_ring,
    )


def make_tmem_allocator(holding_buf):
    """Warp 0 allocates; the math warps and the UMMA warp retrieve the address."""
    return utils.TmemAllocator(
        holding_buf,
        barrier_for_retrieve=pipeline.NamedBarrier(
            barrier_id=TMEM_ALLOC_BARRIER_ID, num_threads=TMEM_ALLOC_THREADS
        ),
        allocator_warp_id=0,
        is_two_cta=False,
    )


def make_group_pipelines(
    create, barriers, group_offset, math_ring, *, math_kwargs=None, **kwargs
):
    """Group 0's and group 1's pipelines, then the math warpgroup's view of its own
    group's, created in this order. barriers() is group 0's first barrier,
    group_offset(g) group g's offset from it; math_ring is the math warpgroup's group,
    or a thunk computing it once group 1's pipeline exists. Returns the three pipelines
    and math_ring."""
    pipe_0 = create(barrier_storage=barriers(), **kwargs)
    pipe_1 = create(barrier_storage=barriers() + group_offset(1), **kwargs)
    if callable(math_ring):
        math_ring = math_ring()
    pipe_math = create(
        barrier_storage=barriers() + group_offset(math_ring),
        **{**kwargs, **(math_kwargs or {})},
    )
    return pipe_0, pipe_1, pipe_math, math_ring


def create_umma_pipeline(**kwargs):
    """A single-stage UMMA -> math pipeline (one producer thread)."""
    return pipeline.PipelineUmmaAsync.create(
        num_stages=1,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
        defer_sync=True,
        **kwargs,
    )


def make_tmem_load_atom(mma_tiler):
    """The TMEM load of a 32-column chunk of an EPI subtile."""
    return blackwell.get_tmem_load_op(
        mma_tiler,
        utils.LayoutEnum.ROW_MAJOR,
        Float32,
        Float32,
        (BLOCK_KV, 32),
        False,
    )


def partition_tmem_readout(copy_atom_t2r, tCtAcc, epi_sub_mn, thr_idx):
    """The accumulator's TMEM -> register copy by EPI subtiles: the tiled copy, this
    thread's slice of it and its source partition (subtile and stage modes last)."""
    tAcc_epi = cute.flat_divide(tCtAcc[((None, None), 0, 0, None)], epi_sub_mn)
    tiled_copy_t2r = tcgen05.make_tmem_copy(
        copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
    )
    thr_copy_t2r = tiled_copy_t2r.get_slice(thr_idx)
    return tiled_copy_t2r, thr_copy_t2r, thr_copy_t2r.partition_S(tAcc_epi)


@cute.jit
def clip_window(mRowStart, mRowEnd, row, seq_len_kv, split_lo, split_hi):
    """Row `row`'s KV window [s, e), clipped to the keys and the split (empty if e <=
    s)."""
    s = max(min(mRowStart[row], seq_len_kv), split_lo)
    e = min(min(mRowEnd[row], seq_len_kv), split_hi)
    return s, e


@cute.jit
def row_window(mRowStart, mRowEnd, row, num_q, seq_len_kv, split_lo, split_hi):
    """clip_window, empty (seq_len_kv, 0) for a padding row past num_q."""
    s = seq_len_kv
    e = Int32(0)
    if row < num_q:
        s, e = clip_window(mRowStart, mRowEnd, row, seq_len_kv, split_lo, split_hi)
    return s, e


@cute.jit
def cta_kv_range(
    mRowStart,
    mRowEnd,
    query_start,
    num_rows: cutlass.Constexpr,
    num_q,
    seq_len_kv,
    split,
    num_splits,
    with_cover: cutlass.Constexpr,
):
    """The CTA's KV split [split_lo, split_hi) (whole 128-key blocks), the union
    [range_lo, range_hi) of its rows' clipped windows and, with_cover, the range
    [cover_lo, cover_hi) every row with a nonempty window covers (blocks inside it need
    no per-key window check)."""
    total_blocks = (seq_len_kv + BLOCK_KV - 1) // BLOCK_KV
    blocks_per_split = (total_blocks + num_splits - 1) // num_splits
    split_lo = split * blocks_per_split * BLOCK_KV
    split_hi = min((split + 1) * blocks_per_split * BLOCK_KV, seq_len_kv)
    range_lo = seq_len_kv
    range_hi = Int32(0)
    cover_lo = Int32(0)
    cover_hi = seq_len_kv
    for r in cutlass.range_constexpr(num_rows):
        s, e = row_window(
            mRowStart, mRowEnd, query_start + r, num_q, seq_len_kv, split_lo, split_hi
        )
        nonempty = e > s
        range_lo = min(range_lo, select_i32(nonempty, s, seq_len_kv))
        range_hi = max(range_hi, select_i32(nonempty, e, 0))
        if cutlass.const_expr(with_cover):
            cover_lo = max(cover_lo, select_i32(nonempty, s, seq_len_kv))
            cover_hi = min(cover_hi, select_i32(nonempty, e, 0))
    return split_lo, split_hi, range_lo, range_hi, cover_lo, cover_hi


def is_int_relu_head(h):
    """Heads 8-11, 24-27 take relu(x) by integer max; the others x + |x| = 2 relu(x)
    (weights halved). Works on a constant head or a lane index."""
    return (h // 4) % 4 == 2


def make_weight_cache(num_rows):
    """Registers for the first NUM_W_IN_REG heads' weights of each row."""
    return cute.make_rmem_tensor(
        cute.make_layout((num_rows, NUM_W_IN_REG), stride=(NUM_W_IN_REG, 1)), Float32
    )


@cute.jit
def replicate_weights(
    sW, sWarpWeights, w_cache, warp, lane, num_rows: cutlass.Constexpr
):
    """A warp-private replica of the tile's weights (lane = head), scaled so that
    either head activation times it is exactly relu(x) * w, and its first NUM_W_IN_REG
    heads in registers."""
    factor = Float32(0.5)
    if is_int_relu_head(lane):
        factor = Float32(1.0)
    for r in cutlass.range_constexpr(num_rows):
        sWarpWeights[(warp, r, lane)] = sW[(r, lane, 0)] * factor
    cute.arch.sync_warp()
    for r in cutlass.range_constexpr(num_rows):
        for h in cutlass.range_constexpr(NUM_W_IN_REG):
            w_cache[(r, h)] = sWarpWeights[(warp, r, h)]


@cute.jit
def head_weight(w_cache, sWarpWeights, warp, t, h: cutlass.Constexpr):
    """Row t's (replicated) weight of head h: from registers or the warp's replica."""
    if cutlass.const_expr(h < NUM_W_IN_REG):
        return w_cache[(t, h)]
    return sWarpWeights[(warp, t, h)]


@cute.jit
def head_quad_activations(acc, n0, h: cutlass.Constexpr):
    """Heads h .. h + 3 of acc[n0 ..] as two packed pairs: x + |x|, or relu(x) by an
    integer max on the bits on the integer-ReLU heads."""
    if cutlass.const_expr(not is_int_relu_head(h)):
        a01 = cute.arch.add_packed_f32x2(
            (acc[n0], acc[n0 + 1]), (abs_f32(acc[n0]), abs_f32(acc[n0 + 1]))
        )
        a23 = cute.arch.add_packed_f32x2(
            (acc[n0 + 2], acc[n0 + 3]),
            (abs_f32(acc[n0 + 2]), abs_f32(acc[n0 + 3])),
        )
    else:
        a01 = tuple(
            bits_f32(cutlass.max(fp32_to_i32_bits(acc[n0 + i]), Int32(0)))
            for i in (0, 1)
        )
        a23 = tuple(
            bits_f32(cutlass.max(fp32_to_i32_bits(acc[n0 + i]), Int32(0)))
            for i in (2, 3)
        )
    return a01, a23
