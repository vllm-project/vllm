# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
# The dual norm's arithmetic, thread mapping and cluster reductions are adapted
# from FlashInfer (flashinfer/norm/kernels/gated_residual_next_rmsnorm.py, from
# flashinfer-ai/flashinfer#6544, and the cluster row sum of
# flashinfer/comm/mnnvl_cutedsl/kernel_ll/column_gather.py,
# flashinfer-ai/flashinfer@14ba7aaa84718abe7a5648184fa9ad15362e2880,
# Apache-2.0).
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".

"""The dual norm at every Muse Spark residual boundary.

Muse keeps an FP32 residual stream; each branch (attention or MoE) is
post-normed, added with per-channel gates, and the sum is pre-normed for the
next branch::

    b = branch * (post_weight + post_gain_center)  # or branch
    r1 = rsqrt(mean(b**2) + post_eps)
    stream = alpha * stream + (beta * b) * r1  # FP32, in place
    r2 = rsqrt(mean(stream**2) + next_eps)
    out = bf16(stream * r2 * (next_weight + next_gain_center))

``dual_norm_fragment`` is the device function: one thread's 8 columns of one
row, with the two row statistics supplied by a cluster reduction. The
standalone kernel (``DualNorm``) uses the reduction of the public
``gated_residual_next_rmsnorm`` (``tree_row_sum``: a butterfly over the
cluster's 32 warp sums); the boundary all-gathers use the one of the public
all-gather dual norm (``serial_row_sum``: the 32 warp sums added in order).
Each matches its reference bit for bit; the two differ in their last bits.

Both use grid (rows, 8), cluster (1, 8, 1), 128 threads, 8 columns per thread
(H = 8192). The stream may be updated in place: each thread reads its 8
columns before it writes them.
"""

from typing import NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cu_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, const_expr
from cutlass.cute.runtime import make_fake_compact_tensor

from .host import assert_not_capturing
from .primitives import (
    VEC_BF16,
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    load_bf16x8_as_f32,
    load_f32x8,
    make_fake_dynamic_compact_tensor,
    map_shared_to_peer,
    store_f32x8,
    store_global_u32x4,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic,
)

H = 8192
CLUSTER_N = 8
THREADS = 128
VEC = H // (CLUSTER_N * THREADS)
NWARPS = THREADS // 32
NSLOT = NWARPS * CLUSTER_N
INV_H = 1.0 / H
# Two mbarriers and a [4, 2 * NSLOT] FP32 buffer, one half per cluster round.
SMEM_BYTES = 2048

assert NSLOT <= 32
assert VEC == VEC_BF16
assert CLUSTER_N * THREADS * VEC == H


class DualNormConstants(NamedTuple):
    """Compile-time constants of one boundary (bound at build, never per step)."""

    post_eps: float
    next_eps: float
    next_gain_center: float
    # None: the post-norm has no weight.
    post_gain_center: float | None
    # None: alpha is a [H] FP32 tensor; else the scalar stream gate.
    alpha_scalar: float | None = None


def boundary_constants(post_norm, connection, pre_norm) -> DualNormConstants:
    """The constants of the stock boundary ``post_norm`` -> ``connection`` ->
    ``pre_norm`` (MuseSpark1xRMSNorm, MuseSpark1xNormPreservingResidualConnection
    with materialized coefficients, MuseSpark1xRMSNorm), as the stock forward
    passes them to the dual-norm kernel."""
    if not getattr(connection, "_coefficients_ready", False):
        raise ValueError("residual coefficients are not materialized")
    if post_norm.has_weight and not post_norm.applies_weight_before_norm:
        raise ValueError("the post-norm weight must scale the branch before the norm")
    if not pre_norm.has_weight or pre_norm.applies_weight_before_norm:
        raise ValueError("the pre-norm must apply its weight after the norm")
    alpha = connection.alpha
    return DualNormConstants(
        post_eps=float(post_norm.variance_epsilon),
        next_eps=float(pre_norm.variance_epsilon),
        next_gain_center=float(pre_norm.fused_gain_center),
        post_gain_center=(
            float(post_norm.fused_gain_center) if post_norm.has_weight else None
        ),
        alpha_scalar=None if isinstance(alpha, torch.Tensor) else float(alpha),
    )


@cute.jit
def _store4_remote(ptr, mbar, peer, x0, x1, x2, x3):
    """One transaction-counted 16-byte store into a peer CTA's SMEM."""
    remote_ptr = cute.arch.map_dsmem_ptr(ptr, peer)
    remote_bar = cute.arch.map_dsmem_ptr(mbar, peer)
    asm = (
        "{\n\t.reg .v4 .f32 q;\n\t"
        "mov.f32 q.x, {$r2};\n\t"
        "mov.f32 q.y, {$r3};\n\t"
        "mov.f32 q.z, {$r4};\n\t"
        "mov.f32 q.w, {$r5};\n\t"
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.v4.f32 "
        "[{$r0}], q, [{$r1}];\n}"
    )
    cute.arch.inline_ptx(
        asm,
        read_only_args=[
            remote_ptr.ir_value(),
            remote_bar.ir_value(),
            x0,
            x1,
            x2,
            x3,
        ],
    )


@cute.jit
def _warp_sum4(v0, v1, v2, v3):
    """Interleave four independent butterfly trees to expose shuffle ILP."""
    w0, w1, w2, w3 = v0, v1, v2, v3
    for offset in (16, 8, 4, 2, 1):
        w0 = w0 + cute.arch.shuffle_sync_bfly(w0, offset=offset)
        w1 = w1 + cute.arch.shuffle_sync_bfly(w1, offset=offset)
        w2 = w2 + cute.arch.shuffle_sync_bfly(w2, offset=offset)
        w3 = w3 + cute.arch.shuffle_sync_bfly(w3, offset=offset)
    return w0, w1, w2, w3


@cute.jit
def _row_sum4(v0, v1, v2, v3, buf, mbar, tidx, lane, warp, rank):
    w0, w1, w2, w3 = _warp_sum4(v0, v1, v2, v3)
    if const_expr(CLUSTER_N > 1):
        if tidx == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(mbar, NSLOT * 16)
        slot = rank * NWARPS + warp
        if lane < CLUSTER_N:
            p = buf[(None, slot)].iterator
            _store4_remote(p, mbar, lane, w0, w1, w2, w3)
        cute.arch.mbarrier_wait(mbar, 0)
        f = cute.make_fragment_like(buf[(None, lane)])
        cute.autovec_copy(buf[(None, lane)], f)
        x = f.load()
        return _warp_sum4(x[0], x[1], x[2], x[3])
    else:
        if lane == 0:
            buf[(0, warp)] = w0
            buf[(1, warp)] = w1
            buf[(2, warp)] = w2
            buf[(3, warp)] = w3
        cute.arch.barrier()
        x0 = Float32(0.0)
        x1 = Float32(0.0)
        x2 = Float32(0.0)
        x3 = Float32(0.0)
        if lane < NWARPS:
            x0 = buf[(0, lane)]
            x1 = buf[(1, lane)]
            x2 = buf[(2, lane)]
            x3 = buf[(3, lane)]
        return _warp_sum4(x0, x1, x2, x3)


@cute.jit
def cluster_row_sum(
    value: Float32,
    slots: cute.Pointer,
    warps: cutlass.Constexpr[int],
    cluster_size: cutlass.Constexpr[int],
    cluster_rank: Int32,
    warp: Int32,
    lane: Int32,
) -> Float32:
    """Sum one value per thread over the cluster; every thread gets the total.

    Each warp scatters its sum into ``slots[cluster_rank, warp]`` of every peer
    CTA, then the slots are added in order. The caller must complete one
    cluster barrier before the first call (a remote store to a peer CTA that
    has not started faults).
    """
    for offset in (16, 8, 4, 2, 1):
        value = value + cute.arch.shuffle_sync_bfly(value, offset=offset)
    if lane < cluster_size:
        remote_slot = map_shared_to_peer(slots + (cluster_rank * warps + warp), lane)
        store_shared_cluster_f32(remote_slot, value)
    cute.arch.cluster_arrive()
    cute.arch.cluster_wait()
    total = Float32(0.0)
    for slot in cutlass.range_constexpr(cluster_size * warps):
        total = total + cute.arch.load((slots + slot).llvm_ptr, Float32)
    return total


def tree_row_sum(buf, mbar, tidx, lane, warp, rank):
    """The row reduction of the standalone kernel: two cluster rounds through
    halves of ``buf`` ([4, 2 * NSLOT] FP32 SMEM) and mbarriers ``mbar`` and
    ``mbar + 1``. Before round 0: the mbarriers initialized and fenced and one
    cluster barrier completed; after round 1: one more cluster barrier."""
    halves = (
        cute.make_tensor(buf.iterator, cute.make_layout((4, NSLOT), stride=(1, 4))),
        cute.make_tensor(
            buf.iterator + 4 * NSLOT, cute.make_layout((4, NSLOT), stride=(1, 4))
        ),
    )
    zero = Float32(0.0)

    def row_sum(value, round_index):
        total, _, _, _ = _row_sum4(
            value,
            zero,
            zero,
            zero,
            halves[round_index],
            mbar + round_index,
            tidx,
            lane,
            warp,
            rank,
        )
        return total

    return row_sum


def serial_row_sum(post_slots, next_slots, cluster_rank, warp, lane):
    """The row reduction of the boundary all-gathers: ``cluster_row_sum`` into
    ``post_slots`` then ``next_slots`` (NSLOT FP32 each, SMEM)."""
    slots = (post_slots, next_slots)

    def row_sum(value, round_index):
        return cluster_row_sum(
            value, slots[round_index], NWARPS, CLUSTER_N, cluster_rank, warp, lane
        )

    return row_sum


def dual_norm_fragment(
    branch,
    stream,
    alpha,
    beta,
    post_gamma,
    next_gamma,
    post_eps: float,
    next_eps: float,
    row_sum,
):
    """One thread's 8 columns of the dual norm (traced inline).

    ``branch`` (the BF16 branch widened), ``stream``, ``beta`` and
    ``next_gamma`` = next_weight + next_gain_center are FP32x8; ``alpha`` is
    FP32x8, a Float32 scalar or None (1.0); ``post_gamma`` = post_weight +
    post_gain_center is FP32x8 or None (no post weight). ``row_sum(value,
    round_index)`` sums one value per thread over the row (round 0: the
    branch's squares, round 1: the stream's). Returns (stream, out), the FP32
    updated stream and the unrounded FP32 next-norm output.
    """
    b = branch
    if post_gamma is not None:
        b = b * post_gamma
    u = stream if alpha is None else alpha * stream
    v = beta * b
    post_sum = row_sum((b * b).reduce(cute.ReductionOp.ADD, 0.0, 0), 0)
    r1 = cute.math.rsqrt(post_sum * INV_H + post_eps, fastmath=True)
    res = u + v * r1
    # The next norm reduces the FP32 stream itself: expanding its square loses
    # the small result when the stream and the normalized branch cancel.
    next_sum = row_sum((res * res).reduce(cute.ReductionOp.ADD, 0.0, 0), 1)
    r2 = cute.math.rsqrt(next_sum * INV_H + next_eps, fastmath=True)
    out = (res * r2) * next_gamma
    return res, out


class _DualNormKernel:
    """Grid (m, CLUSTER_N), cluster (1, CLUSTER_N, 1), THREADS threads."""

    def __init__(
        self, constants: DualNormConstants, router: bool, enable_pdl: bool
    ) -> None:
        self.constants = constants
        self.router = router
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        branch: cute.Tensor,
        stream_in: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        stream_out: cute.Tensor,
        out: cute.Tensor,
        router_out: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            branch,
            stream_in,
            alpha,
            beta,
            post_weight,
            next_weight,
            stream_out,
            out,
            router_out,
        ).launch(
            grid=(m, CLUSTER_N, 1),
            block=(THREADS, 1, 1),
            cluster=(1, CLUSTER_N, 1),
            smem=SMEM_BYTES,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        branch: cute.Tensor,
        stream_in: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor,
        next_weight: cute.Tensor,
        stream_out: cute.Tensor,
        out: cute.Tensor,
        router_out: cute.Tensor,
    ) -> None:
        constants = self.constants
        tidx, _, _ = cute.arch.thread_idx()
        token, cy, _ = cute.arch.block_idx()
        smem = cu_utils.SmemAllocator()
        mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
        buf = smem.allocate_tensor(
            Float32, cute.make_layout((4, NSLOT * 2), stride=(1, 4)), byte_alignment=16
        )
        fragment = cy * THREADS + tidx
        column = Int64(fragment) * VEC_BF16
        row_element = Int64(token) * H + column

        # Gates and weights do not change between calls: load before the PDL wait.
        alpha_values = None
        if cutlass.const_expr(constants.alpha_scalar is None):
            alpha_values = load_f32x8(alpha, column)
        elif cutlass.const_expr(constants.alpha_scalar != 1.0):
            alpha_values = Float32(constants.alpha_scalar)
        beta_values = load_f32x8(beta, column)
        next_gamma = load_bf16x8_as_f32(next_weight, column) + Float32(
            constants.next_gain_center
        )
        post_gamma = None
        if cutlass.const_expr(constants.post_gain_center is not None):
            post_gamma = load_bf16x8_as_f32(post_weight, column) + Float32(
                constants.post_gain_center
            )
        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_init(mbar + 1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()
        lane = cute.arch.lane_idx()
        warp = cute.arch.warp_idx()
        rank = cute.arch.block_idx_in_cluster()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()
        branch_values = load_bf16x8_as_f32(branch, row_element)
        stream_values = load_f32x8(stream_in, row_element)

        cute.arch.cluster_wait()
        res, result = dual_norm_fragment(
            branch_values,
            stream_values,
            alpha_values,
            beta_values,
            post_gamma,
            next_gamma,
            constants.post_eps,
            constants.next_eps,
            tree_row_sum(buf, mbar, tidx, lane, warp, rank),
        )
        # No CTA may exit while a peer still reads its shared memory.
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()
        store_f32x8(stream_out, row_element, res)
        store_global_u32x4(
            Int64((out.iterator + row_element).toint()),
            bf16x8_to_packed_u32x4(result.to(BFloat16)),
        )
        if cutlass.const_expr(self.router):
            store_f32x8(router_out, row_element, result)
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class DualNorm:
    """The dual norm's standalone kernel for one boundary: constants bound and
    the kernel compiled at construction (build time, before CUDA graph
    capture).

    ``run`` is graph-capturable and allocates nothing. Every tensor is
    contiguous on one device: branch and out BF16 [b, H], stream FP32 [b, H]
    (updated in place), post_weight and next_weight BF16 [H], alpha (unless
    bound as a scalar) and beta FP32 [H], router_input FP32 [b, H] when the
    kernel was built with ``router``. Rows (b) are a runtime value.
    """

    def __init__(
        self,
        constants: DualNormConstants,
        *,
        router: bool = False,
        enable_pdl: bool = True,
    ) -> None:
        assert_not_capturing("DualNorm")
        self.constants, self.router = constants, router
        weight = make_fake_compact_tensor(BFloat16, (H,), assumed_align=16)
        gate = make_fake_compact_tensor(Float32, (H,), assumed_align=32)
        rows_f32 = make_fake_dynamic_compact_tensor(
            Float32, alignment=32, divisibility=H
        )
        rows_bf16 = make_fake_dynamic_compact_tensor(
            BFloat16, alignment=16, divisibility=H
        )
        self._compiled = cute.compile(
            _DualNormKernel(constants, router, enable_pdl),
            rows_bf16,
            rows_f32,
            gate,
            gate,
            weight,
            weight,
            rows_f32,
            rows_bf16,
            rows_f32,
            Int32(1),
            current_cu_stream(),
        )

    def run(
        self,
        branch: torch.Tensor,
        stream: torch.Tensor,
        post_weight: torch.Tensor | None,
        alpha: torch.Tensor | float,
        beta: torch.Tensor,
        next_weight: torch.Tensor,
        out: torch.Tensor,
        router_input: torch.Tensor | None = None,
    ) -> None:
        rows = branch.shape[0]
        if rows == 0:
            return
        stream_rows = to_cute_dynamic(stream.flatten(), 32, divisibility=H)
        self._compiled(
            to_cute_dynamic(branch.flatten(), 16, divisibility=H),
            stream_rows,
            to_cute(alpha if isinstance(alpha, torch.Tensor) else beta, 32),
            to_cute(beta, 32),
            to_cute(post_weight if post_weight is not None else next_weight, 16),
            to_cute(next_weight, 16),
            stream_rows,
            to_cute_dynamic(out.flatten(), 16, divisibility=H),
            (
                to_cute_dynamic(router_input.flatten(), 32, divisibility=H)
                if router_input is not None
                else stream_rows
            ),
            Int32(rows),
            current_cu_stream(),
        )
