# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from FlashInfer 0.7.3.dev20261030 (Apache-2.0):
# flashinfer/gemm/kernels/dense_bf16_gemm_warp_splitk.py
"""FlashInfer's cp.async-pipelined BF16 GEMM for M <= 32 whose four compute warps hold
split-K partials (warp split-K), kernel unchanged; the autotune heuristics are dropped
for per-shape tactics chosen by ``decode.py`` and compiled before capture.

The weight streams through two cp.async loader warps with an L2 evict_first hint
(one tile is issued before the PDL wait), the activations through one or two loader
warps after it; four mma.sync warps each own a K quarter of every stage and reduce
through shared memory. Small output tiles (16 or 32 weight rows) give many CTAs that
co-reside on an SM, which is what keeps a decode projection's weight stream at HBM
speed."""

from __future__ import annotations

import dataclasses
import functools
import math

import cuda.bindings.driver as _cuda
import cutlass
import cutlass.cute as cute
import torch as _torch
from cutlass import const_expr
from cutlass._mlir.dialects import llvm
from cutlass.cute import experimental as cute_ext
from cutlass.cute.nvgpu import warp
from cutlass.cute.runtime import from_dlpack
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

_MMA_SHAPE = (16, 8, 16)
_COMPUTE_WARPS = 4
_A_LOADER_WARPS = 2
_A_LOADER_THREADS = _A_LOADER_WARPS * 32
_MAX_M = 32
# Ring depth cap: cold-read bandwidth saturates around 12 stages (96 KB in
# flight), so deeper rings only enlarge the tactic space.
_MAX_STAGES = 16
# The consumer's K loop is fully unrolled; bound the tile count so JIT time and
# code size stay proportionate (64 tiles of 256 elements = K 16384).
_MAX_K_TILES = 64
_SUPPORTED_OUTPUT_TILES = (16, 32)
_SUPPORTED_TOKEN_TILES = (8, 16, 32)
_SUPPORTED_K_TILES = (128, 256)
_SUPPORTED_B_LOADER_WARPS = (1, 2)
_SMEM_CAPACITY = cutlass.utils.get_smem_capacity_in_bytes("sm_100")
# Per-SM shared memory is the per-CTA maximum plus the 1 KB the driver reserves
# for every resident CTA; residency estimates must include that reservation.
_CTA_RESERVED_SMEM = 1024
_SM_SMEM_BYTES = _SMEM_CAPACITY + _CTA_RESERVED_SMEM
_MAILBOX_GROUP_VALUES = 32
_MAILBOX_PADDED_GROUP_VALUES = 34


# Pad each mailbox group by two FP32 words to reduce bank conflicts.
def _partial_offset(feature, token, token_tile: int):
    group_rows = _MAILBOX_GROUP_VALUES // token_tile
    return (
        (feature % group_rows) * token_tile
        + (feature // group_rows) * _MAILBOX_PADDED_GROUP_VALUES
        + token
    )


def _a_fragment_offset(lane, compute_warp, m_iter, phase, k_tile):
    """Element offset of one lane's A ``ldmatrix.x4`` address within a stage."""
    linear = lane % 16 + (lane // 16) * 128 + phase * 256
    row = linear % _MMA_SHAPE[0] + m_iter * _MMA_SHAPE[0]
    col = linear // _MMA_SHAPE[0] + compute_warp * (k_tile // _COMPUTE_WARPS)
    return row * k_tile + (col ^ ((row % 8) * 8))


def _b_pair_offset(lane, compute_warp, n_iter, pair, token_tile, k_tile):
    """Element offset of one lane's B ``ldmatrix.x4`` address within a stage."""
    b_atom = lane % 8 + ((lane % 32) // 8) * (token_tile * 8)
    linear = b_atom + pair * 2 * token_tile * 16 + n_iter * 8
    row = linear % token_tile
    col = linear // token_tile + compute_warp * (k_tile // _COMPUTE_WARPS)
    return row * k_tile + (col ^ ((row % 8) * 8))


@dsl_user_op
def _ldmatrix_x4(smem, offset, *, loc=None, ip=None):
    """``ldmatrix.x4`` of the 16-bit fragment ``offset`` elements into ``smem``."""
    return prims.ldmatrix(
        smem.iterator.raw_ptr() + offset,
        num=4,
        layout=prims.MMALayout.ROW,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _l2_policy_evict_first(*, loc=None, ip=None):
    """64-bit L2 evict_first policy (fraction 1.0).

    Must stay a runtime ``createpolicy``: with the equivalent constant, ptxas
    13.0 emits the invalid ``LDGSTS ... [R+UR], desc[UR1]`` encoding.
    """
    return cutlass.Int64(
        llvm.inline_asm(
            cutlass.Int64.mlir_type,
            [],
            "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
            "=l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def _cp_async_16_with_policy(dst, src, policy, *, loc=None, ip=None):
    """16-byte cp.async.cg global->shared carrying an L2 cache-policy hint."""
    llvm.inline_asm(
        None,
        [
            dst.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            policy.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global.L2::cache_hint [$0], [$1], 16, $2;",
        "r,l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def _issue_weight_tile(
    smem,
    weight,
    tidx,
    output_tile_idx,
    k_tile,
    stage,
    policy,
    output_tile,
    k_tile_size,
    with_policy,
    *,
    loc=None,
    ip=None,
):
    """Issue one A-loader thread's cp.async copies for weight K tile ``k_tile``.

    Plain Python control flow only: this helper is not preprocessed by the DSL.
    ``with_policy`` selects the ``.L2::cache_hint`` form; the pre-barrier
    prologue tile must use the plain form because ptxas 13.0 misencodes the
    hinted copy in that position (``LDGSTS ... [R+UR], desc[UR<odd>]``).
    """
    copy_count = output_tile * k_tile_size // (8 * _A_LOADER_THREADS)
    for copy_idx in range(copy_count):
        linear = tidx * 8 + copy_idx * _A_LOADER_THREADS * 8
        row = linear // k_tile_size
        col = linear % k_tile_size
        dst_offset = (
            stage * output_tile * k_tile_size
            + row * k_tile_size
            + (col ^ ((row % 8) * 8))
        )
        src_offset = weight.layout(
            (
                output_tile_idx * output_tile + row,
                k_tile * k_tile_size + col,
            )
        )
        if with_policy:
            # CuTe pointers: ``toint()`` yields the 32-bit shared address for
            # SMEM and the 64-bit global address for GMEM, as the PTX expects.
            _cp_async_16_with_policy(
                smem.iterator + dst_offset,
                weight.iterator + src_offset,
                policy,
                loc=loc,
                ip=ip,
            )
        else:
            prims.cp_async_shared_global(
                smem.iterator.raw_ptr() + dst_offset,
                weight.iterator.raw_ptr() + src_offset,
                16,
                "cg",
                loc=loc,
                ip=ip,
            )


@dataclasses.dataclass(frozen=True)
class WarpSplitKTactic:
    output_tile: int
    token_tile: int
    k_tile: int
    stages: int
    b_loader_warps: int


def _align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _smem_bytes(tactic: WarpSplitKTactic) -> int:
    cursor = tactic.output_tile * tactic.k_tile * 2 * tactic.stages
    cursor = _align_up(cursor, 1024)
    cursor += tactic.token_tile * tactic.k_tile * 2 * tactic.stages
    cursor = _align_up(cursor, 16)
    partial_stride = (
        tactic.output_tile * tactic.token_tile // _MAILBOX_GROUP_VALUES
    ) * _MAILBOX_PADDED_GROUP_VALUES
    cursor += _COMPUTE_WARPS * partial_stride * 4
    cursor = _align_up(cursor, 8)
    cursor += tactic.stages * 8
    cursor = _align_up(cursor, 8)
    return cursor + tactic.stages * 8


def _make_smem_layout(dtype, rows: int, cols: int, stages: int):
    copy_elems = 128 // dtype.width
    major_size = min(cols, 128 * 8 // dtype.width)
    outer = cute.tile_to_shape(
        cute.make_layout((8, major_size), stride=(major_size, 1)),
        (rows, cols, stages),
        (0, 1, 2),
    )
    return cute.make_composed_layout(
        cute.make_swizzle(
            min(int(math.log2(major_size // copy_elems)), 3),
            4,
            int(math.log2(copy_elems)),
        ),
        0,
        outer,
    )


class CpAsyncWarpSplitKKernel:
    def __init__(self, tactic: WarpSplitKTactic, use_pdl: bool, has_bias: bool) -> None:
        self.output_tile = tactic.output_tile
        self.token_tile = tactic.token_tile
        self.k_tile = tactic.k_tile
        self.stages = tactic.stages
        self.b_loader_warps = tactic.b_loader_warps
        self.compute_warp_base = _A_LOADER_WARPS + self.b_loader_warps
        self.threads = (self.compute_warp_base + _COMPUTE_WARPS) * 32
        self.load_threads = self.compute_warp_base * 32
        self.use_pdl = use_pdl
        self.has_bias = has_bias
        self.smem_bytes = _smem_bytes(tactic)

    @cute.experimental.jit
    def __call__(self, weight, activation, bias, output, stream: _cuda.CUstream):
        sA_layout = _make_smem_layout(
            weight.element_type, self.output_tile, self.k_tile, self.stages
        )
        sB_layout = _make_smem_layout(
            activation.element_type, self.token_tile, self.k_tile, self.stages
        )
        tiled_mma = cute.make_tiled_mma(
            warp.MmaF16BF16Op(weight.element_type, cutlass.Float32, _MMA_SHAPE),
            cute.make_layout((1, 1, _COMPUTE_WARPS)),
            permutation_mnk=(self.output_tile, self.token_tile, self.k_tile),
        )
        self.kernel(
            weight, activation, bias, output, sA_layout, sB_layout, tiled_mma
        ).launch(
            grid=(
                cute.ceil_div(weight.shape[0], self.output_tile),
                cute.ceil_div(activation.shape[0], self.token_tile),
                1,
            ),
            block=(self.threads, 1, 1),
            smem=cute.Int64(self.smem_bytes),
            stream=stream,
            use_pdl=self.use_pdl,
        )

    @cute.experimental.kernel
    def kernel(
        self,
        weight: cute.Tensor,
        activation: cute.Tensor,
        bias: cute.Tensor,  # (N,) broadcast bias; unread when has_bias is False
        output: cute.Tensor,
        sA_layout: cute.ComposedLayout,
        sB_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        output_tile_idx, token_tile_idx, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = tidx % cute.arch.WARP_SIZE
        reuse_stages = cute.size(weight, mode=[1]) > self.stages * self.k_tile

        sA = cute_ext.allocate(
            weight.element_type,
            cute.AddressSpace.smem,
            sA_layout,
            alignment=1024,
        )
        sB = cute_ext.allocate(
            activation.element_type,
            cute.AddressSpace.smem,
            sB_layout,
            alignment=1024,
        )
        values_per_lane = self.output_tile * self.token_tile // cute.arch.WARP_SIZE
        partial_stride = (
            self.output_tile * self.token_tile // _MAILBOX_GROUP_VALUES
        ) * _MAILBOX_PADDED_GROUP_VALUES
        partials = cute_ext.allocate(
            cutlass.Float32,
            cute.AddressSpace.smem,
            cute.make_layout(_COMPUTE_WARPS * partial_stride),
            alignment=16,
        )
        bar_full_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        bar_empty_arr = cute_ext.allocate(
            cutlass.Int64,
            cute.AddressSpace.smem,
            cute.make_layout(self.stages),
            alignment=8,
        )
        bar_full = bar_full_arr.iterator
        bar_empty = bar_empty_arr.iterator

        k_tile_count = cute.size(weight, mode=[1]) // self.k_tile

        if warp_idx < _A_LOADER_WARPS:
            # Tile 0 has no mbarrier dependency: issue it before the init handshake
            # so its DRAM latency overlaps init and the CTA barrier.
            _issue_weight_tile(
                sA,
                weight,
                tidx,
                output_tile_idx,
                0,
                0,
                cutlass.Int64(0),
                self.output_tile,
                self.k_tile,
                False,
            )
        bias_value = cutlass.Float32(0.0)
        if warp_idx == self.compute_warp_base:  # noqa: SIM102 (DSL: no dynamic and)
            # The first compute warp is idle here; it initializes the mbarriers so
            # the loader warps are never held behind the init.
            if lane_idx < self.stages:
                cute.arch.mbarrier_init(bar_full + lane_idx, self.load_threads)
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_init(bar_empty + lane_idx, _COMPUTE_WARPS)
        cute.arch.mbarrier_init_fence()
        cute.arch.barrier()

        if warp_idx < _A_LOADER_WARPS:
            # Weights stream through L2 evict_first so the activation rows every
            # CTA re-reads stay resident. Only tile 0's arrive may follow the init.
            weight_policy = _l2_policy_evict_first()
            empty_phase = cutlass.Int32(1)
            prims.cp_async_mbarrier_arrive(bar_full.llvm_ptr, noinc=True)
            for k_tile in cutlass.range(1, k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                _issue_weight_tile(
                    sA,
                    weight,
                    tidx,
                    output_tile_idx,
                    k_tile,
                    stage,
                    weight_policy,
                    self.output_tile,
                    self.k_tile,
                    True,
                )
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
        elif warp_idx < self.compute_warp_base:
            if const_expr(self.use_pdl):
                cute.arch.griddepcontrol_wait()
            b_loader_threads = self.b_loader_warps * 32
            local_tid = tidx - _A_LOADER_WARPS * 32
            copy_count = self.token_tile * self.k_tile // (8 * b_loader_threads)
            empty_phase = cutlass.Int32(1)
            for k_tile in cutlass.range(k_tile_count, unroll=1):
                stage = k_tile % self.stages
                if const_expr(reuse_stages):
                    cute.arch.mbarrier_wait(bar_empty + stage, empty_phase)
                for copy_idx in cutlass.range_constexpr(copy_count):
                    linear = local_tid * 8 + copy_idx * b_loader_threads * 8
                    row = linear // self.k_tile
                    col = linear % self.k_tile
                    global_row = token_tile_idx * self.token_tile + row
                    # Out-of-range token rows are discarded by the output predicate.
                    if global_row < cute.size(activation, mode=[0]):
                        dst = (
                            sB.iterator.raw_ptr()
                            + stage * self.token_tile * self.k_tile
                            + row * self.k_tile
                            + (col ^ ((row % 8) * 8))
                        )
                        src = activation.iterator.raw_ptr() + activation.layout(
                            (
                                global_row,
                                k_tile * self.k_tile + col,
                            )
                        )
                        prims.cp_async_shared_global(dst, src, 16, "cg")
                prims.cp_async_mbarrier_arrive((bar_full + stage).llvm_ptr, noinc=True)
                if stage == self.stages - 1:
                    empty_phase = empty_phase ^ 1
            if const_expr(self.use_pdl):
                # This grid-level hint is idempotent across B-loader warps.
                cute.arch.griddepcontrol_launch_dependents()
        else:
            if const_expr(self.has_bias):
                # Each reduce lane owns one output feature (32 % output_tile == 0).
                # Issue the load here, after the CTA barrier: waiting on the first
                # full stage hides its latency, whereas a load before the barrier
                # would hold the loader warps behind a cold DRAM read.
                bias_value = bias[
                    output_tile_idx * self.output_tile + lane_idx % self.output_tile
                ].to(cutlass.Float32)
            mma_tid = tidx - self.compute_warp_base * cute.arch.WARP_SIZE
            compute_warp = warp_idx - self.compute_warp_base
            thr_mma = tiled_mma.get_slice(mma_tid)
            acc = cute.make_rmem_tensor(
                thr_mma.partition_shape_C((self.output_tile, self.token_tile)),
                cutlass.Float32,
            )
            acc.fill(0.0)
            k_phases = self.k_tile // (_COMPUTE_WARPS * _MMA_SHAPE[2])
            m_iters = self.output_tile // _MMA_SHAPE[0]
            n_iters = self.token_tile // _MMA_SHAPE[1]
            # Fragment addresses are loop-invariant per lane apart from the stage
            # base; hoist them so the K loop issues ldmatrix from one add each.
            a_offsets = tuple(
                tuple(
                    _a_fragment_offset(
                        lane_idx, compute_warp, m_iter, phase, self.k_tile
                    )
                    for phase in range(k_phases)
                )
                for m_iter in range(m_iters)
            )
            b_offsets = tuple(
                tuple(
                    _b_pair_offset(
                        lane_idx,
                        compute_warp,
                        n_iter,
                        pair,
                        self.token_tile,
                        self.k_tile,
                    )
                    for pair in range(k_phases // 2)
                )
                for n_iter in range(n_iters)
            )
            a_stage_elems = self.output_tile * self.k_tile
            b_stage_elems = self.token_tile * self.k_tile
            full_phase = cutlass.Int32(0)
            # The K-tile count is static: unrolling fully folds `stage`, barrier
            # addresses and phases to constants, so every ldmatrix is
            # `[R_lane + imm]` (a rolled loop costs 3-16% on K=7168).
            for k_tile in cutlass.range(k_tile_count, unroll_full=True):
                stage = k_tile % self.stages
                cute.arch.mbarrier_wait(bar_full + stage, full_phase)
                a_base = stage * a_stage_elems
                b_base = stage * b_stage_elems
                a_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sA, a_base + a_offsets[m_iter][phase])
                        for phase in range(k_phases)
                    )
                    for m_iter in range(m_iters)
                )
                b_registers = tuple(
                    tuple(
                        _ldmatrix_x4(sB, b_base + b_offsets[n_iter][pair])
                        for pair in range(k_phases // 2)
                    )
                    for n_iter in range(n_iters)
                )
                for phase in cutlass.range_constexpr(k_phases):
                    for n_iter in cutlass.range_constexpr(n_iters):
                        packed_b = b_registers[n_iter][phase // 2]
                        b_offset = (phase % 2) * 2
                        for m_iter in cutlass.range_constexpr(m_iters):
                            acc_base = (m_iter * n_iters + n_iter) * 4
                            result = prims.mma_sync(
                                llvm.StructType.get_literal(
                                    [cutlass.Float32.mlir_type] * 4
                                ),
                                shape=_MMA_SHAPE,
                                layout_a=prims.MMALayout.ROW,
                                layout_b=prims.MMALayout.COL,
                                operand_a=[
                                    a_registers[m_iter][phase][i].ir_value()
                                    for i in range(4)
                                ],
                                operand_b=[
                                    packed_b[b_offset + i].ir_value() for i in range(2)
                                ],
                                operand_c=[
                                    acc[acc_base + i].ir_value() for i in range(4)
                                ],
                                multiplicand_a_ptx_type=prims.MMAType.BF16,
                                multiplicand_b_ptx_type=prims.MMAType.BF16,
                            )
                            for i in cutlass.range_constexpr(4):
                                acc[acc_base + i] = cutlass.Float32(
                                    llvm.extractvalue(
                                        cutlass.Float32.mlir_type,
                                        result,
                                        [i],
                                    )
                                )
                if const_expr(reuse_stages):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(bar_empty + stage)
                if stage == self.stages - 1:
                    full_phase = full_phase ^ 1
            for value in cutlass.range_constexpr(values_per_lane):
                atom = value % 4
                fragment = value // 4
                m_iter = fragment // n_iters
                n_iter = fragment % n_iters
                feature = lane_idx // 4 + 8 * (atom // 2) + m_iter * _MMA_SHAPE[0]
                token = 8 * n_iter + 2 * (lane_idx % 4) + atom % 2
                partials[
                    compute_warp * partial_stride
                    + _partial_offset(feature, token, self.token_tile)
                ] = acc[value]
            prims.barrier_cta_sync(1, thread_count=_COMPUTE_WARPS * 32)

        if warp_idx == self.compute_warp_base:  # noqa: SIM102 (DSL: no dynamic and)
            final_acc = cute.make_rmem_tensor(
                cute.make_layout(values_per_lane), cutlass.Float32
            )
            for value in cutlass.range_constexpr(values_per_lane):
                linear = value * cute.arch.WARP_SIZE + lane_idx
                feature = linear % self.output_tile
                token = linear // self.output_tile
                total = cutlass.Float32(0)
                for peer in cutlass.range_constexpr(_COMPUTE_WARPS):
                    total = (
                        total
                        + partials[
                            peer * partial_stride
                            + _partial_offset(feature, token, self.token_tile)
                        ]
                    )
                if const_expr(self.has_bias):
                    total = total + bias_value
                final_acc[value] = total

            output_base = output_tile_idx * self.output_tile
            token_base = token_tile_idx * self.token_tile
            for value in cutlass.range_constexpr(values_per_lane):
                linear = value * cute.arch.WARP_SIZE + lane_idx
                feature = linear % self.output_tile
                token = linear // self.output_tile
                if token_base + token < cute.size(output, mode=[0]):
                    output[token_base + token, output_base + feature] = final_acc[
                        value
                    ].to(output.element_type)


def _view(tensor, *, dynamic_m=False):
    view = from_dlpack(tensor.detach(), assumed_align=32)
    if dynamic_m:
        # Only M varies; the row stride and N/K extents stay static.
        view = view.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
    return view


def check_tactic(tactic: WarpSplitKTactic, n: int, k: int) -> None:
    if tactic.output_tile not in _SUPPORTED_OUTPUT_TILES or n % tactic.output_tile:
        raise ValueError(f"output_tile {tactic.output_tile} for N={n}")
    if tactic.token_tile not in _SUPPORTED_TOKEN_TILES:
        raise ValueError(f"unsupported token_tile={tactic.token_tile}")
    if tactic.k_tile not in _SUPPORTED_K_TILES or k % tactic.k_tile:
        raise ValueError(f"k_tile {tactic.k_tile} for K={k}")
    tiles = k // tactic.k_tile
    if tiles > _MAX_K_TILES or (tactic.k_tile == 256 and tiles == 2):
        raise ValueError(f"K={k} in tiles of {tactic.k_tile}")
    if not (1 if tiles <= 1 else 2) <= tactic.stages <= min(_MAX_STAGES, tiles):
        raise ValueError(f"invalid stages={tactic.stages}")
    if tactic.b_loader_warps not in _SUPPORTED_B_LOADER_WARPS:
        raise ValueError(f"unsupported b_loader_warps={tactic.b_loader_warps}")
    if _smem_bytes(tactic) > _SMEM_CAPACITY:
        raise ValueError(f"{tactic} needs {_smem_bytes(tactic)} B of shared memory")


@functools.cache
def compile_warp_splitk(device_index: int, n: int, k: int, tactic: WarpSplitKTactic):
    """Code for an [n, k] weight with dynamic M (at most 32 rows), PDL on."""
    check_tactic(tactic, n, k)
    dtype = _torch.bfloat16
    with _torch.cuda.device(device_index):
        shapes: tuple[tuple[tuple[int, ...], bool], ...] = (
            ((n, k), False),
            ((_MAX_M, k), True),
            ((n,), False),
        )
        shapes += (((_MAX_M, n), True),)
        args = tuple(
            _view(_torch.empty(s, device="cuda", dtype=dtype), dynamic_m=d)
            for s, d in shapes
        )
        stream = _cuda.CUstream(_torch.cuda.current_stream().cuda_stream)
        return cute_ext.compile(
            CpAsyncWarpSplitKKernel(tactic, True, False), *args, stream
        )


class PreparedWarpSplitK:
    """``out[M, N] = x[M, K] @ weight[N, K]^T`` for M <= 32 with a tactic per M;
    ``x`` and ``out`` packed row-major, 32 B aligned."""

    def __init__(self, weight, tactics):
        n, k = weight.shape
        if weight.dtype != _torch.bfloat16 or weight.stride() != (k, 1):
            raise ValueError("warp split-K weights are packed BF16 [N, K]")
        if weight.data_ptr() % 32 or any(m > _MAX_M for m in tactics):
            raise ValueError("warp split-K: 32 B aligned weight, M <= 32")
        self.weight, self.tactics = weight, dict(tactics)
        index = weight.device.index
        self.codes = {
            m: compile_warp_splitk(index, n, k, t) for m, t in self.tactics.items()
        }
        self.weight_view = _view(weight)

    def run(self, x, out):
        code = self.codes[x.shape[0]]
        stream = _cuda.CUstream(_torch.cuda.current_stream(x.device).cuda_stream)
        # The kernel never reads its bias argument; a row of ``out`` fills it.
        bias = _view(out[0])
        code(self.weight_view, _view(x, dynamic_m=True), bias,
             _view(out, dynamic_m=True), stream)  # fmt: skip
        return out
