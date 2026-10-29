# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
# The publish step, the consumer's poll and re-arm and the rank-order sum are
# adapted from FlashInfer (flashinfer/comm/mnnvl_cutedsl/kernel_ll/
# device_kernels.py and column_gather.py,
# flashinfer-ai/flashinfer@14ba7aaa84718abe7a5648184fa9ad15362e2880,
# Apache-2.0); the ticket stage and the boundary arithmetic are not.
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".

"""The DSpark draft's residual boundary with its branch all-reduce, in one
CuTe kernel on the thread mapping of the target's attention boundary
(all_reduce_dual_norm.py): grid (rows, 8), cluster (1, 8, 1), 128 threads, one
16-byte fragment (8 columns) of one row per thread.

Each thread multicasts its fragment of this rank's branch partial (the draft
o_proj or down_proj output; -0.0 stored as +0.0) into every rank's Lamport
mailbox, polls the ``tp`` slots of that fragment until none holds the sentinel,
re-arms them and sums them in the order of the group's stock all-reduce
(``rank_order_sum``). The stage is a ticket word: every CTA takes one, the last
advances the stage.

The boundary then runs on the BF16 sum with the draft boundary's arithmetic and
row-sum order (``boundary_fragments`` of cutedsl/common/draft_boundary.py), bit
for bit, one fragment per thread.

With ``published`` the branch's GEMM has already multicast this rank's partial
(the package GEMM's zero-copy epilogue on the same ticket): the kernel neither
reads the partial nor publishes it.

Every CTA releases its dependents right after its PDL wait; the next kernel
waits for this grid's end before it reads the output.

Every rank must call with the same row count, one call at a time; a poll that
sees no data for ``timeout_ns`` traps instead of hanging the group.
"""

import functools
from dataclasses import dataclass

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as cu_utils
import torch
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, const_expr
from cutlass.cute.runtime import make_fake_compact_tensor

from ..common.draft_boundary import (
    FRAGMENTS,
    BoundaryOperands,
    DraftBoundaryConstants,
    boundary_fragments,
    fake_weights,
    load_gates,
)
from ..common.host import assert_not_capturing
from ..common.mailbox import SPIN_TIMEOUT_NS, spin_deadline
from ..common.policy import SUM_ORDERS
from ..common.primitives import (
    VEC_BF16,
    atomic_add1_acq_rel_u32,
    current_cu_stream,
    fragment_has_negative_zero,
    load_bf16x8_as_f32,
    load_f32x8,
    load_global_u32x4,
    make_fake_dynamic_compact_tensor,
    regs,
    round_bf16,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_f32x8,
    store_global_u32,
    store_global_u32x4,
    store_lamport_sentinel_u32x4,
    to_cute,
    to_cute_dynamic,
)
from ..common.row_norm import (
    CHUNK,
    WARP,
    WARPS,
    H,
    chunk_pair,
    thread_sum,
    warp_sum,
    warps_sum,
)
from .all_reduce_dual_norm import rank_order_sum, stock_one_shot_rows

CLUSTER_N = WARPS  # CTAs per row: the reference kernel's warps
THREADS = FRAGMENTS * WARP  # warp k: the reference thread's k-th 2048 columns

assert CLUSTER_N * THREADS * VEC_BF16 == H


def boundary_unsupported(constants: DraftBoundaryConstants) -> str | None:
    """Why this kernel does not take a boundary that ``DraftBoundary`` takes,
    or None. It takes BF16 or FP32 norm weights, with the pre-norm's applied
    after the norm."""
    if constants.pre_input_scale:
        return "an input-scaled pre-norm"
    if constants.post_weight_fp16 or constants.pre_weight_fp16:
        return "FP16 norm weights"
    return None


@cute.jit
def _send_sum(slot: cute.Pointer, mbar: cute.Pointer, peer: Int32, value: Float32):
    """Store ``value`` into ``slot`` of CTA ``peer``'s shared memory, counted
    as 4 bytes on that CTA's ``mbar``."""
    remote_slot = cute.arch.map_dsmem_ptr(slot, peer)
    remote_bar = cute.arch.map_dsmem_ptr(mbar, peer)
    cute.arch.inline_ptx(
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 "
        "[{$r0}], {$r2}, [{$r1}];",
        read_only_args=[remote_slot.ir_value(), remote_bar.ir_value(), value],
    )


@cute.jit
def _row_sum(
    even: Float32,
    odd: Float32,
    pairs: cute.Pointer,
    sums: cute.Pointer,
    mbar: cute.Pointer,
    tidx: Int32,
    cy: Int32,
    warp: Int32,
    lane: Int32,
) -> Float32:
    """The row's sum of squares in the single-tile norm's order
    (cutedsl/common/row_norm.py) from each thread's (even, odd) fragment pair
    (``chunk_pair``); every thread gets it. ``pairs``:
    2 * THREADS FP32 of shared memory, ``sums``: CLUSTER_N FP32 and ``mbar``
    an mbarrier (initialized for one arrival), each used by one call. The
    cluster must have passed one barrier since every CTA initialized its
    mbarriers."""
    slot = (warp * WARP + lane) * 2
    cute.arch.store((pairs + slot).llvm_ptr, even)
    cute.arch.store((pairs + (slot + 1)).llvm_ptr, odd)
    if tidx == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, CLUSTER_N * 4)
    cute.arch.barrier()
    if warp == Int32(0):
        blocks = []
        for k in cutlass.range_constexpr(FRAGMENTS):
            base = (k * WARP + lane) * 2
            blocks.append(
                (
                    cute.arch.load((pairs + base).llvm_ptr, Float32),
                    cute.arch.load((pairs + (base + 1)).llvm_ptr, Float32),
                )
            )
        total = warp_sum(thread_sum(blocks))
        if lane < Int32(CLUSTER_N):
            _send_sum(sums + cy, mbar, lane, total)
    cute.arch.mbarrier_wait(mbar, 0)
    return warps_sum(
        [cute.arch.load((sums + i).llvm_ptr, Float32) for i in range(CLUSTER_N)]
    )


@dataclass(frozen=True)
class _ClusterRowSum:
    """``RowSum`` of a cluster that holds the row, one fragment per thread:
    each round its own ``pairs`` (2 * THREADS FP32), ``sums`` (CLUSTER_N FP32)
    and ``mbar``. Round 0 first waits for the cluster barrier after the
    mbarriers' initialization."""

    pairs: cute.Pointer
    sums: cute.Pointer
    mbar: cute.Pointer
    tidx: Int32
    cy: Int32
    warp: Int32
    lane: Int32

    def __call__(self, fragments: list, fused: bool, round: int) -> Float32:
        if round == 0:
            cute.arch.cluster_wait()
        (fragment,) = fragments
        even, odd = chunk_pair(fragment, fused)
        return _row_sum(
            even,
            odd,
            self.pairs + round * 2 * THREADS,
            self.sums + round * CLUSTER_N,
            self.mbar + round,
            self.tidx,
            self.cy,
            self.warp,
            self.lane,
        )


@dataclass(frozen=True)
class _StreamStore:
    """Stores the thread's fragment of the new stream at ``row_element``."""

    stream_out: cute.Tensor
    row_element: Int64

    def __call__(self, k: int, values) -> None:
        store_f32x8(self.stream_out, self.row_element, values)


class _DraftAllReduceBoundaryKernel:
    """Grid (m, CLUSTER_N), cluster (1, CLUSTER_N, 1), THREADS threads."""

    def __init__(
        self,
        *,
        tp: int,
        capacity: int,
        generations: int,
        sum_order: str,
        multicast: bool,
        constants: DraftBoundaryConstants,
        stream_bf16: bool,
        store_stream: bool,
        timeout_ns: int,
        published: bool = False,
    ) -> None:
        self.tp, self.capacity = tp, capacity
        self.generations = generations
        self.sum_order = sum_order
        self.one_shot_rows = stock_one_shot_rows(sum_order, tp, capacity)
        self.multicast = multicast
        self.constants = constants
        self.stream_bf16 = stream_bf16
        self.store_stream = store_stream
        self.timeout_ns = timeout_ns
        self.published = published

    @cute.jit
    def __call__(
        self,
        partial: cute.Tensor,
        mailbox: cute.Tensor,
        multicast_address: Int64,
        rank: Int32,
        ticket: cute.Tensor,
        stream_in: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor | None,
        pre_weight: cute.Tensor | None,
        stream_out: cute.Tensor | None,
        out: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            partial,
            mailbox,
            multicast_address,
            rank,
            ticket,
            stream_in,
            alpha,
            beta,
            post_weight,
            pre_weight,
            stream_out,
            out,
        ).launch(
            grid=(m, CLUSTER_N, 1),
            block=(THREADS, 1, 1),
            cluster=(1, CLUSTER_N, 1),
            stream=stream,
            use_pdl=True,
        )

    @cute.kernel
    def kernel(
        self,
        partial: cute.Tensor,
        mailbox: cute.Tensor,
        multicast_address: Int64,
        rank: Int32,
        ticket: cute.Tensor,
        stream_in: cute.Tensor,
        alpha: cute.Tensor,
        beta: cute.Tensor,
        post_weight: cute.Tensor | None,
        pre_weight: cute.Tensor | None,
        stream_out: cute.Tensor | None,
        out: cute.Tensor,
    ) -> None:
        constants = self.constants
        tidx, _, _ = cute.arch.thread_idx()
        token, cy, _ = cute.arch.block_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()
        smem = cu_utils.SmemAllocator()
        mbar = smem.allocate_array(cutlass.Int64, num_elems=2)
        stage_slot = smem.allocate_array(Int32, 4)
        # Two calls of _row_sum: 2 * THREADS pairs and CLUSTER_N sums each.
        pairs = smem.allocate_array(Float32, 4 * THREADS)
        sums = smem.allocate_array(Float32, 2 * CLUSTER_N)
        column = Int64(warp) * CHUNK + Int64(cy) * (WARP * VEC_BF16) + Int64(lane) * 8
        row_element = Int64(token) * H + column

        # Gates and weights do not change between calls: load before the PDL wait.
        gates = load_gates(constants, alpha, beta, post_weight, pre_weight, column)
        # Peers store into this CTA's shared memory (counted on its mbarriers)
        # after one cluster barrier.
        if tidx == 0:
            cute.arch.mbarrier_init(mbar, 1)
            cute.arch.mbarrier_init(mbar + 1, 1)
        cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

        # The next kernel may launch now: it waits for this grid's end before
        # it reads the output.
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        if tidx == 0:
            # Every CTA takes a ticket; the last of the call advances the stage.
            word = atomic_add1_acq_rel_u32(ticket.iterator)
            taken = word >> Uint32(16)
            ctas = Uint32(cute.arch.grid_dim()[0]) * Uint32(CLUSTER_N)
            if (word & Uint32(0xFFFF)) == ctas - Uint32(1):
                store_global_u32(
                    ticket.iterator,
                    ((taken + Uint32(1)) % Uint32(self.generations)) << Uint32(16),
                )
            cute.arch.store(stage_slot.llvm_ptr, Int32(taken))
        own = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
        if const_expr(not self.published):
            own.store(
                sanitize_negative_zero_u32x4(
                    load_global_u32x4(
                        cute.make_ptr(
                            BFloat16,
                            (partial.iterator + row_element).llvm_ptr,
                            cute.AddressSpace.gmem,
                            assumed_align=16,
                        )
                    )
                )
            )
        if const_expr(self.stream_bf16):
            stream_values = regs(load_bf16x8_as_f32(stream_in, row_element))
        else:
            stream_values = regs(load_f32x8(stream_in, row_element))
        cute.arch.barrier()
        stage = Int64(cute.arch.load(stage_slot.llvm_ptr, Int32))

        # Slot [stage, r, token] of the fragment: first + r * rank_stride.
        first = (stage * (self.tp * self.capacity) + Int64(token)) * H + column
        rank_stride = self.capacity * H

        # Publish this rank's fragment (unless its GEMM has).
        if const_expr(not self.published):
            mine = first + Int64(rank) * rank_stride
            if const_expr(self.multicast):
                stmc_bf16x8(multicast_address + mine * 2, own.load())
            else:
                store_global_u32x4(Int64((mailbox.iterator + mine).toint()), own.load())

        # Poll every rank's fragment, then re-arm. Every pass re-polls all of
        # them.
        lane_packed = cute.make_rmem_tensor(cute.make_layout((self.tp, 4)), Uint32)
        lane_packed.fill(Uint32(0))
        deadline = Int64(0)
        # Always true; a runtime value keeps the poll a device loop.
        dirty = tidx >= Int32(0)
        while dirty:
            dirty = False
            for r in cutlass.range_constexpr(self.tp):
                packed = load_global_u32x4(
                    cute.make_ptr(
                        BFloat16,
                        (mailbox.iterator + (first + r * rank_stride)).llvm_ptr,
                        cute.AddressSpace.gmem,
                        assumed_align=16,
                    ),
                    volatile=True,
                )
                dirty = dirty | fragment_has_negative_zero(packed)
                for w in cutlass.range_constexpr(4):
                    lane_packed[r, w] = packed[w]
            if dirty:
                deadline = spin_deadline(deadline, self.timeout_ns)
        for r in cutlass.range_constexpr(self.tp):
            store_lamport_sentinel_u32x4(
                Int64((mailbox.iterator + (first + r * rank_stride)).toint())
            )

        # Rank-order sum, as the stock all-reduce, rounded to BF16.
        total = rank_order_sum(
            lane_packed, self.tp, self.sum_order, self.one_shot_rows, True
        )
        x = [round_bf16(total[i]) for i in range(VEC_BF16)]

        # The boundary on the sum, as the draft boundary's.
        store = None
        if const_expr(self.store_stream):
            store = _StreamStore(stream_out, row_element)
        (words,) = boundary_fragments(
            constants,
            [x],
            [stream_values],
            [gates],
            _ClusterRowSum(pairs, sums, mbar, tidx, cy, warp, lane),
            store,
        )
        store_global_u32x4(Int64((out.iterator + row_element).toint()), words.load())


@functools.cache
def compile_draft_all_reduce_boundary(
    *,
    tp: int,
    capacity: int,
    generations: int,
    mailbox_numel: int,
    sum_order: str,
    multicast: bool,
    constants: DraftBoundaryConstants,
    stream_bf16: bool,
    store_stream: bool,
    timeout_ns: int,
    published: bool = False,
):
    """cute.compile one kernel; the rank and the row count are runtime
    arguments (a weight None where its norm has none, ``stream_out`` None
    without ``store_stream``). Call before CUDA graph capture."""
    assert_not_capturing("the draft all-reduce boundary")
    gate = make_fake_compact_tensor(Float32, (H,), assumed_align=32)
    rows_f32 = make_fake_dynamic_compact_tensor(Float32, alignment=32, divisibility=H)
    rows_bf16 = make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=H)
    return cute.compile(
        _DraftAllReduceBoundaryKernel(
            tp=tp,
            capacity=capacity,
            generations=generations,
            sum_order=sum_order,
            multicast=multicast,
            constants=constants,
            stream_bf16=stream_bf16,
            store_stream=store_stream,
            timeout_ns=timeout_ns,
            published=published,
        ),
        rows_bf16,
        make_fake_compact_tensor(BFloat16, (mailbox_numel,), assumed_align=16),
        Int64(0),
        Int32(0),
        make_fake_compact_tensor(Int32, (1,), assumed_align=4),
        rows_bf16 if stream_bf16 else rows_f32,
        gate,
        gate,
        *fake_weights(constants),
        rows_f32 if store_stream else None,
        rows_bf16,
        Int32(1),
        current_cu_stream(),
    )


class DraftAllReduceBoundary:
    """One TP group's draft boundaries with the branch all-reduce fused in:
    ``mailbox`` ([generations, tp, capacity, 8192] BF16, sentinel-filled, the
    rank's copy of a LamportMailbox), its multicast address (None: publish into
    this device's copy only, a single-device check), the stage ticket (int32
    [1], stage << 16, zero before the first call) and the sum order of the
    group's stock all-reduce (``one_shot_rows``: the rows above which it
    runs its two-shot form). Kernels compile per boundary constant set,
    stream dtype, whether the stream is stored and whether the branch's GEMM
    publishes the partial (``prepare``, before CUDA graph capture)."""

    def __init__(
        self,
        mailbox: torch.Tensor,
        multicast: int | None,
        rank: int,
        *,
        sum_order: str,
        ticket: torch.Tensor | None = None,
        timeout_ns: int = SPIN_TIMEOUT_NS,
    ) -> None:
        if sum_order not in SUM_ORDERS:
            raise ValueError(f"unknown sum order {sum_order!r}")
        generations, tp, capacity, width = mailbox.shape
        if (
            mailbox.dtype != torch.bfloat16
            or not mailbox.is_contiguous()
            or width != H
            or not 0 <= rank < tp
        ):
            raise ValueError(f"the mailbox must be a contiguous BF16 [G, tp, C, {H}]")
        self.mailbox, self.multicast, self.rank = mailbox, multicast, rank
        self.tp, self.capacity, self.generations = tp, capacity, generations
        self.sum_order = sum_order
        self.one_shot_rows = stock_one_shot_rows(sum_order, tp, capacity)
        self.timeout_ns = timeout_ns
        self.ticket = (
            ticket
            if ticket is not None
            else torch.zeros(1, dtype=torch.int32, device=mailbox.device)
        )
        self._mailbox_view = to_cute(mailbox.flatten(), 16)
        self._ticket_view = to_cute(self.ticket, 4)
        self._address = Int64(multicast or 0)

    def prepare(
        self,
        constants: DraftBoundaryConstants,
        stream_bf16: bool,
        store_stream: bool,
        published: bool = False,
    ):
        """The compiled kernel of one boundary variant (before CUDA graph
        capture; compiled once per process for every group of this shape)."""
        return compile_draft_all_reduce_boundary(
            tp=self.tp,
            capacity=self.capacity,
            generations=self.generations,
            mailbox_numel=self.mailbox.numel(),
            sum_order=self.sum_order,
            multicast=self.multicast is not None,
            constants=constants,
            stream_bf16=stream_bf16,
            store_stream=store_stream,
            timeout_ns=self.timeout_ns,
            published=published,
        )

    def bind(
        self,
        constants: DraftBoundaryConstants,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        post_weight: torch.Tensor | None,
        pre_weight: torch.Tensor | None,
    ) -> "BoundDraftBoundary":
        """One boundary with its constant operands checked and viewed once
        (before CUDA graph capture): ``run`` on it launches without
        rebuilding them."""
        return BoundDraftBoundary(self, constants, alpha, beta, post_weight, pre_weight)

    def run(
        self,
        partial: torch.Tensor,
        stream: torch.Tensor,
        constants: DraftBoundaryConstants,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        post_weight: torch.Tensor | None,
        pre_weight: torch.Tensor | None,
        store_stream: bool,
        published: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """``(new_stream or None, out)`` of the boundary of the all-reduce of
        every rank's ``partial`` (this rank's BF16 [m, 8192] branch partial),
        ``stream`` FP32 or BF16 [m, 8192]. With ``published`` the partial's
        GEMM has multicast it into this call's slot (the package GEMM on the
        ticket); the kernel does not read it. Graph-capturable once
        prepared. Binds the boundary on every call: a caller that runs one
        boundary repeatedly binds it once (``bind``)."""
        bound = self.bind(constants, alpha, beta, post_weight, pre_weight)
        return bound.run(partial, stream, store_stream, published)

    def check_rows(self, partial: torch.Tensor, stream: torch.Tensor) -> None:
        """Refuse rows the kernel does not take (before anything launches)."""
        m = partial.shape[0]
        if not (
            1 <= m <= self.capacity
            and partial.shape == (m, H)
            and partial.dtype == torch.bfloat16
            and partial.is_contiguous()
            and stream.shape == (m, H)
            and stream.dtype in (torch.float32, torch.bfloat16)
            and stream.is_contiguous()
        ):
            raise ValueError(
                f"partial (BF16) and stream must be dense [m, {H}], "
                f"1 <= m <= {self.capacity}"
            )

    def launch(
        self,
        kernel,
        partial: torch.Tensor,
        stream: torch.Tensor,
        operands: tuple[cute.Tensor, ...],
        store_stream: bool,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """Launch a prepared ``kernel`` on rows ``check_rows`` took and a
        bound boundary's ``operands`` (alpha, beta, post and pre weight
        views)."""
        m = partial.shape[0]
        out = torch.empty((m, H), dtype=torch.bfloat16, device=partial.device)
        new_stream = (
            torch.empty((m, H), dtype=torch.float32, device=partial.device)
            if store_stream
            else None
        )
        stream_rows = to_cute_dynamic(
            stream.flatten(),
            16 if stream.dtype == torch.bfloat16 else 32,
            divisibility=H,
        )
        kernel(
            to_cute_dynamic(partial.flatten(), 16, divisibility=H),
            self._mailbox_view,
            self._address,
            Int32(self.rank),
            self._ticket_view,
            stream_rows,
            *operands,
            (
                to_cute_dynamic(new_stream.flatten(), 32, divisibility=H)
                if new_stream is not None
                else None
            ),
            to_cute_dynamic(out.flatten(), 16, divisibility=H),
            Int32(m),
            current_cu_stream(),
        )
        return new_stream, out


class BoundDraftBoundary:
    """One draft boundary of a ``DraftAllReduceBoundary``, bound at build:
    its constants and its constant operands (``BoundaryOperands``); its
    kernels by (stream BF16, store stream, published) come from the
    process-wide compile cache (``prepare`` them before CUDA graph
    capture)."""

    def __init__(
        self,
        op: DraftAllReduceBoundary,
        constants: DraftBoundaryConstants,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        post_weight: torch.Tensor | None,
        pre_weight: torch.Tensor | None,
    ) -> None:
        reason = boundary_unsupported(constants)
        if reason is not None:
            raise ValueError(f"The fused all-reduce boundary does not take {reason}")
        self.op, self.constants = op, constants
        self.operands = BoundaryOperands(
            constants, alpha, beta, post_weight, pre_weight
        )

    def prepare(self, stream_bf16: bool, store_stream: bool, published: bool = False):
        """The kernel of one variant (``DraftAllReduceBoundary.prepare``)."""
        return self.op.prepare(self.constants, stream_bf16, store_stream, published)

    def run(
        self,
        partial: torch.Tensor,
        stream: torch.Tensor,
        store_stream: bool,
        published: bool = False,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        """``DraftAllReduceBoundary.run`` of this boundary."""
        self.op.check_rows(partial, stream)
        kernel = self.prepare(stream.dtype == torch.bfloat16, store_stream, published)
        return self.op.launch(
            kernel, partial, stream, self.operands.views, store_stream
        )
