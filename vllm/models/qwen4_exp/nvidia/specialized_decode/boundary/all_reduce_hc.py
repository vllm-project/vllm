# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Sublayer boundary of the Qwen4Exp specialized decode in one CuTe DSL kernel.

Every decoder block of Qwen4Exp (attention, MoE) ends at a *boundary*: under
tensor parallelism each rank holds a partial `[rows, 2560]` block output
whose sum over the ranks is the block output `y`. The stock decode then runs

1. FlashInfer's MNNVL one-shot all-reduce of the partials (TP > 1), and
2. the next hyper-connection read's Triton combine-norm: the four residual
   streams take the block output, `streams_s += 2 sigmoid(inj_s / 4) y` with
   the injection logits `inj` of the read that fed the block, and each
   stream of 2560 values is normalized by a Gemma RMSNorm into `xn`.

After a MoE block the shared expert's gate (`sigmoid` of a one-column
projection), its product with the shared expert's output and the add of the
routed output come first, as three more PyTorch kernels.

`SublayerBoundary` does all of it in one kernel per boundary; the read's
projections (`..hc`) then continue from `xn`.

Protocol (TP > 1). The partials travel through a Lamport mailbox in symmetric
memory, `[generation][source rank][token][2560]` BF16 on every rank, as in
FlashInfer's low-latency MNNVL all-reduce: each thread stores its fragment of
the partial through the NVLink multicast address into the same slot of every
rank's mailbox, then polls its own copy until all ranks' slots of that
fragment hold data, and resets them. An empty slot holds BF16 negative zero in
every value; senders turn a real negative zero into a positive one, so data
and flag are the same bytes.

Generations. A rank keeps two device words: the generation its calls publish
into and an arrival counter. In every CTA, thread 0 reads the generation,
stores it in shared memory and then adds one to the counter (the add depends
on the value read); every other thread takes thread 0's value from shared
memory after a CTA barrier and never reads the device word. The thread 0 that
finds the counter at `rows * cluster - 1` is the last to arrive: every CTA's
thread 0 has read the generation before it, and it stores the next generation
and a zero counter at its end, so all threads of a call use the same
generation. The next call reads the new one because its grid-dependency wait
returns only after this call has completed. That holds as long as every kernel
of the programmatic-dependent-launch chain between two calls runs
`griddepcontrol.wait` before it completes (each wait returns after the
previous kernel completed); a kernel of the chain without that wait would let
the next call read the generation word early.

Two generations suffice. A rank starts call `n + 1` only after its call
`n` completed, which needs every rank's call `n` data; each rank publishes
that data after its own call `n - 1` completed, including the resets of the
slots it read. So ranks are at most one call apart, and a rank publishing call
`n + 1` into generation `(n + 1) % 2` finds every rank's slots there read
and reset by call `n - 1`. The protocol test
(`tests/models/qwen4_exp/specialized_decode/test_boundary_protocol.py`)
checks this, over every interleaving of a small model of threads, CTAs, ranks
and consecutive calls, together with defects that break it: a generation read
by every thread, one generation, a missing reset.

Stalls. A thread that polls for longer than `stall_timeout` (a rank that
skipped a call or called in a different order) stores a diagnostic word in
host memory (`SublayerBoundary.stall_report`) and traps, so the step raises
a CUDA error instead of hanging. The healthy path pays one timer read per
polling iteration.

Liveness (TP > 1). A thread publishes its fragment before it polls, and it
polls only for the same fragment of the same token from the other ranks, which
the CTA of the same grid index publishes on every rank. A grid of `rows *
cluster` CTAs (1,280 at TP4 for 256 rows with three draft tokens) can exceed
what a GPU holds at once; the kernel then relies on each GPU starting a grid's
CTAs in index order, which CUDA does not promise but its hardware does. Under
that order the lowest-index unfinished CTAs are resident on every rank, their
peers of the same index are resident or done, so they finish and free their
slots. A rank that started them out of order would stall into the timeout
above, not hang.

Work split. A token's 320 sixteen-byte fragments (8 BF16 values each) are
shared by a cluster of CTAs (five of 64 threads by default), one fragment per
thread; the
thread publishes, polls, sums and combines that fragment of `y` for all
four streams. The per-stream sums of squares meet in shared memory and then,
through distributed shared memory, in every CTA of the cluster::

    partial ──multicast──► mailbox[gen][rank][token] ── poll, rank-order sum ─► y
    streams, inj ─► streams_s + 2 sigmoid(inj_s/4) y ─► streams_out
                                       └─► sum of squares ─► cluster sum ─► xn

Distributed shared memory follows two rules: every CTA arrives on the cluster
barrier at entry and waits on it before its first store into a peer's shared
memory, so no CTA writes into one that has not started; and no CTA accesses a
peer's shared memory after the cluster barrier that follows those stores, so
a CTA may exit once it has passed that barrier.

Numerics. The rank sum is FP32 from rank 0 up, rounded once to BF16, the order
and rounding of the stock all-reduce, and every rank computes the same bits.
The combine repeats the stock kernel's instruction sequence, so the streams
match it bit for bit; the norm's sum of squares runs in a different order than
Triton's reduction, so `xn` can differ from the stock kernel's in the last
bit. After a MoE block the contribution is
`bf16(bf16(shared * bf16(sigmoid(logit))) + routed)`, the stock roundings.

Routed top-k reduction (`moe_folded`). The routed experts can stop after
their second GEMM and leave the weighted sum over each token's experts (the
*finalize*, FlashInfer's `moe::dev::finalize` kernel) to this kernel. The
experts then hand over their unweighted GEMM2 rows `[permuted rows, 2560]`
BF16, each token's top-10 routing weights `[rows, 10]` BF16 and the row of
each of its top-10 experts in the GEMM2 output `[rows, 10]` int32 (-1 for an
expert another rank holds, under expert parallelism). Each thread sums its
fragment of the token's rows, FP32 products of the widened weight and value,
and rounds the sum once to BF16: the routed output the finalize kernel would
have written, bit for bit, in its order. FlashInfer runs two kernels:

- up to 118 tokens (fewer than 1184 CTAs of 256 columns), the small-batch
  kernel splits a token's ten experts over four thread groups (expert `k`
  to group `k % 4`), each accumulating its experts in order from zero by
  fused multiply-adds, and adds the four groups' sums in group order;
- from 119 tokens, the vectorized kernel accumulates the ten experts in
  order from zero.

The product of two BF16 values is exact in FP32, so a fused multiply-add and
a rounded product plus a rounded add give the same bits; only the order of the
additions matters, and the kernel keeps FlashInfer's. `finalize_groups`
gives the group count for a row count; the kernel takes it as a compile-time
variant. An expert another rank holds is skipped, as there.
The routing outputs and the GEMM2 rows are read after the grid-dependency
wait, like every other kernel input.

Scheduling. Every kernel of the chain launches with programmatic dependent
launch. The kernel copies its norm weight into shared memory (`cp.async`)
before the grid-dependency wait. By default it lets its dependent (the read's
down projection) launch once the block output is known, so the dependent's
CTAs load their weights during the combine and norm without competing with the
polling exchange for SM slots; `early_trigger` launches the dependent at
entry instead (measured slower to `xn` at 4 and 16 rows, even at 64).

At TP 1 the same kernel runs without the mailbox: the partial is the block
output.

Row split (`split`, TP > 1). The one-shot exchange gives every rank the
block output of every row, and every rank then combines, normalizes and reads
all rows although the next block needs only the read's block input. In a
split call the rows are divided among the ranks in contiguous runs of
`rows / tp` (rank `q` owns rows `[q R, q R + R)`), and the residual
streams and injection logits a rank holds are those of its own rows only:

    partial row t ──peer store──► owner(t) mailbox[gen][source][t - owner R]
    owner: poll the three peers' slots, rank-order sum ─► y ─► combine, norm

Each CTA still computes one row's contribution, as in the one-shot kernel (the
grid is unchanged); a CTA whose row another rank owns stores it into the
owner's mailbox and exits, the owner's CTA keeps its own contribution in
registers, waits for the three others and runs the combine and norm of that
row. The sum is the one-shot sum (FP32 in rank order from zero over the
sanitized contributions, one BF16 rounding), so a split call writes, for the
owner's rows, the bits the one-shot call writes. A rank's mailbox is
`[generation][source rank][rows / tp][2560]` (its own source slots unused);
the generation and arrival words are the split mailbox's own, with the same
protocol: every rank owns rows of every call (`rows` is a multiple of
`tp`), so a rank's call `n` completes only after every peer published its
call `n` data, and the two-generation argument above holds unchanged. The
reference decode server's two-shot boundary splits a row's columns among
owners and all-gathers the sums; here the rows are split and the all-gather
moves to the end of the read (`..hc.up_mix`, `HcUpMixGather`), where it
carries the 2560-wide block input instead of the reduced block output.

Scope: 1 to `max_rows` token rows per call (tested up to 256 at TP 4 and
TP 1, the executor's largest steps), BF16, the hyper-connection shape of
Qwen3.8-Flash-Next. One instance serves all boundaries of a forward; ranks must
issue the same calls in the same order.
"""

from functools import cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_stream

from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .. import HC_HIDDEN, HC_STREAMS
from ..hc._ptx import (
    copy_async_16,
    copy_async_wait_all,
    load_bf16,
    load_shared_f32,
    load_shared_x4,
    load_x4,
    store_shared_f32,
)
from ..logits._ptx import load_u64, store_x4_peer
from ._ptx import (
    SENTINEL_BF16X2,
    add_rn,
    arrive_u32,
    empty_halves_x4,
    fma_if,
    fma_rn,
    global_timer,
    injection_scale,
    load_s32,
    load_shared_u32,
    load_volatile_u32,
    load_volatile_x4,
    load_x4_if,
    mul_rn,
    multicast_store_x4,
    norm_scale,
    pack_bf16x2,
    report_stall,
    round_bf16,
    sanitize_bf16x2,
    store_cluster_f32,
    store_shared_u32,
    store_u32,
    store_x4,
    unpack_bf16x2,
)

HIDDEN = HC_HIDDEN
STREAMS = HC_STREAMS
WIDTH = STREAMS * HIDDEN
VALUES = 8  # BF16 values of a 16-byte fragment
FRAGMENTS = HIDDEN // VALUES  # 320 per token
CLUSTER = 5  # CTAs of a token (default)
GENERATIONS = 2
# The device words of the protocol: the generation to publish into, and the
# arrival counter of the current call.
GENERATION_WORD = 0
ARRIVALS_WORD = 1
# Seconds a thread polls before it reports a stall and traps (default).
STALL_TIMEOUT_S = 30.0
# Routed experts per token whose top-k reduction the MoE kernel can fold in.
TOP_K = 10
# FlashInfer's finalize (moe::dev::finalize::run): the small-batch kernel runs
# below 1184 CTAs of 256 columns, 4 < top_k <= 64, with top-k groups of at most
# four experts each (the fewest power-of-two count).
FINALIZE_CTA_LIMIT = 1184
FINALIZE_COLUMNS = 256
FINALIZE_LOADS = 4


def finalize_groups(rows: int, top_k: int = TOP_K) -> int:
    """The thread groups over which FlashInfer's finalize splits each token's
    `top_k` experts for a call of `rows` tokens (1: the vectorized kernel,
    which sums them in order)."""
    ctas = -(-HIDDEN // FINALIZE_COLUMNS) * min(8192, rows)
    if ctas >= FINALIZE_CTA_LIMIT or not FINALIZE_LOADS < top_k <= 64:
        return 1
    groups = 1
    while groups * FINALIZE_LOADS < top_k:
        groups *= 2
    return groups


def _shared_gate(logit: Float32) -> Float32:
    """`bf16(sigmoid(logit))` as PyTorch's BF16 sigmoid computes it:
    `1 / (1 + exp(-x))` in FP32 with the precise exponential."""
    one = Float32(1.0)
    return round_bf16(one / (one + cute.math.exp(-logit, fastmath=False)))


class BoundaryKernel:
    """The boundary kernel of one rank and contribution kind.

    Args:
        tp: Ranks of the all-reduce (1: no exchange).
        rank: This rank in the group.
        capacity: Token rows of a mailbox generation (the largest call).
        moe: The contribution is a MoE block's (routed output plus the gated
            shared expert output) rather than a partial block output.
        early_trigger: Let dependents launch at entry (else once the block
            output is known, before the combine).
        cluster: CTAs of a token (1, 2 or 5), one 16-byte fragment per
            thread.
        stall_ns: Nanoseconds a thread polls before it reports a stall and
            traps.
        fold: For `moe`, 0 when the routed output is given; else the routed
            experts' unweighted GEMM2 rows are, and this is the number of
            thread groups of the finalize whose top-k reduction the kernel
            repeats (`finalize_groups`).
        split: Reduce-scatter the rows to their owner ranks and combine and
            normalize only this rank's rows (module docstring, Row split);
            `capacity` is then a rank's rows of a mailbox generation.

    """

    def __init__(
        self,
        tp: int,
        rank: int,
        capacity: int,
        moe: bool,
        early_trigger: bool,
        cluster: int,
        stall_ns: int,
        fold: int = 0,
        split: bool = False,
    ) -> None:
        assert tp >= 1 and 0 <= rank < tp and capacity >= 1 and stall_ns > 0
        assert cluster in (1, 2, 5)
        assert fold == 0 or (moe and 1 <= fold <= TOP_K)
        assert not split or (tp > 1 and not early_trigger)
        self.tp = tp
        self.rank = rank
        self.capacity = capacity
        self.moe = moe
        self.fold = fold
        self.split = split
        self.early_trigger = early_trigger
        self.cluster = cluster
        self.stall_ns = stall_ns
        self.threads = FRAGMENTS // cluster
        self.warps = self.threads // 32
        assert self.threads % 32 == 0 and self.threads >= cluster * STREAMS
        # The norm weights ([stream][thread] 16-byte fragments), the
        # [warp][stream] and [cluster CTA][stream] sums of squares, then the
        # generation (one word in a 16-byte block).
        self.shared_bytes = (
            self.threads * STREAMS * 16 + (self.warps + cluster) * STREAMS * 4 + 16
        )

    @cute.jit
    def __call__(
        self,
        partial: Int64,
        shared: Int64,
        gate: Int64,
        expert_weights: Int64,
        expert_slots: Int64,
        streams: Int64,
        injection: Int64,
        injection_stride: Int32,
        norm: Int64,
        streams_out: Int64,
        normed: Int64,
        mailbox: Int64,
        multicast: Int64,
        peers: Int64,
        state: Int64,
        stall: Int64,
        eps: Float32,
        rows: Int32,
        stream: cuda.CUstream,
    ):
        """Launch on `stream` for `rows` tokens: BF16 addresses of
        `partial [rows, 2560]` (the routed output for `moe`; with `fold`
        the routed experts' GEMM2 rows `[any, 2560]`), `shared [rows,
        2560]` and `gate [rows]` (`moe` only), `expert_weights [rows,
        10]` BF16 and `expert_slots [rows, 10]` int32 (`fold` only),
        `streams [rows, 10240]`, `injection [rows, >= 4]` (row stride
        `injection_stride`), `norm [10240]` and the outputs `streams_out`
        and `normed [rows, 10240]` (`split`: `rows / tp` rows, this
        rank's); the mailbox's local and multicast addresses, the device array
        of every rank's mailbox address (`split` only), the protocol words
        `state` and the host-memory stall word `stall` (`tp > 1`
        only)."""
        if cutlass.const_expr(self.split):
            self.split_kernel(
                partial,
                shared,
                gate,
                expert_weights,
                expert_slots,
                streams,
                injection,
                injection_stride,
                norm,
                streams_out,
                normed,
                mailbox,
                peers,
                state,
                stall,
                eps,
            ).launch(
                grid=(rows, self.cluster, 1),
                block=(self.threads, 1, 1),
                cluster=(1, self.cluster, 1),
                smem=self.shared_bytes,
                stream=stream,
                use_pdl=True,
            )
        else:
            self.kernel(
                partial,
                shared,
                gate,
                expert_weights,
                expert_slots,
                streams,
                injection,
                injection_stride,
                norm,
                streams_out,
                normed,
                mailbox,
                multicast,
                state,
                stall,
                eps,
            ).launch(
                grid=(rows, self.cluster, 1),
                block=(self.threads, 1, 1),
                cluster=(1, self.cluster, 1),
                smem=self.shared_bytes,
                stream=stream,
                use_pdl=True,
            )

    @cute.kernel
    def kernel(
        self,
        partial: Int64,
        shared: Int64,
        gate: Int64,
        expert_weights: Int64,
        expert_slots: Int64,
        streams: Int64,
        injection: Int64,
        injection_stride: Int32,
        norm: Int64,
        streams_out: Int64,
        normed: Int64,
        mailbox: Int64,
        multicast: Int64,
        state: Int64,
        stall: Int64,
        eps: Float32,
    ):
        """One CTA: a `1 / cluster` share of one token's hidden columns, all
        four streams."""
        tp = self.tp
        tid, _, _ = cute.arch.thread_idx()
        token, part, _ = cute.arch.block_idx()  # part == rank in the cluster
        rows, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        column = (part * self.threads + tid) * VALUES

        allocator = cutlass.memory.SmemAllocator()
        weights = allocator.allocate_array(
            Uint32, self.threads * STREAMS * 4, byte_alignment=16
        ).toint()
        # The four streams' norm weights at this fragment, copied to shared
        # memory before the wait so they arrive while the previous kernel
        # runs. Asynchronous copies stay above the wait; register loads of
        # constant data would be sunk below it.
        for s in cutlass.range_constexpr(STREAMS):
            copy_async_16(
                weights + (s * self.threads + tid) * 16,
                norm + Int64(s * HIDDEN + column) * 2,
            )

        warp_sums = allocator.allocate_array(
            Float32, self.warps * STREAMS, byte_alignment=16
        ).toint()
        cluster_sums = allocator.allocate_array(
            Float32, self.cluster * STREAMS, byte_alignment=16
        ).toint()
        generation_shared = allocator.allocate_array(
            Uint32, 1, byte_alignment=16
        ).toint()
        if cutlass.const_expr(self.cluster > 1):
            # This CTA has started: peers may store into its shared memory
            # once they pass the matching wait (module docstring).
            cute.arch.cluster_arrive_relaxed()

        # Dependents may launch at any point: they read this kernel's outputs
        # only after their own grid-dependency wait, and each kernel of the
        # chain ends only after its own wait returned.
        if cutlass.const_expr(self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()
        cute.arch.griddepcontrol_wait()

        generation = Int32(0)
        arrival = Uint32(0)
        if cutlass.const_expr(tp > 1):  # noqa: SIM102
            # Thread 0 reads the generation for the CTA, then arrives; the
            # last arriver advances it at its end (module docstring).
            if tid == 0:
                word = load_volatile_u32(state + GENERATION_WORD * 4)
                store_shared_u32(generation_shared, word)
                arrival = arrive_u32(state + ARRIVALS_WORD * 4, word)

        # The token's streams and injection logits (written by earlier kernels).
        row = streams + Int64(token * WIDTH + column) * 2
        residual = []
        scales = []
        for s in cutlass.range_constexpr(STREAMS):
            residual.append(load_x4(row + s * HIDDEN * 2))
            scales.append(
                injection_scale(
                    load_bf16(injection + Int64(token * injection_stride + s) * 2)
                )
            )

        # This rank's contribution: four words of eight BF16 values.
        source = Int64(token * HIDDEN + column) * 2
        contribution = []
        if cutlass.const_expr(self.moe):
            routed_values = []
            if cutlass.const_expr(self.fold > 0):
                routed_values = self._fold(
                    partial, expert_weights, expert_slots, token, column
                )
            else:
                routed = load_x4(partial + source)
                for i in cutlass.range_constexpr(4):
                    r_lo, r_hi = unpack_bf16x2(routed[i])
                    routed_values.append(r_lo)
                    routed_values.append(r_hi)
            expert = load_x4(shared + source)
            g = _shared_gate(load_bf16(gate + Int64(token) * 2))
            for i in cutlass.range_constexpr(4):
                e_lo, e_hi = unpack_bf16x2(expert[i])
                contribution.append(
                    pack_bf16x2(
                        add_rn(round_bf16(mul_rn(e_lo, g)), routed_values[2 * i]),
                        add_rn(round_bf16(mul_rn(e_hi, g)), routed_values[2 * i + 1]),
                    )
                )
        else:
            contribution = list(load_x4(partial + source))

        # The block output at this fragment.
        block = []
        if cutlass.const_expr(tp > 1):
            # Every thread uses thread 0's generation.
            cute.arch.sync_threads()
            generation = Int32(load_shared_u32(generation_shared))
            published = []
            for i in cutlass.range_constexpr(4):
                published.append(sanitize_bf16x2(contribution[i]))
            slot = []
            for r in cutlass.range_constexpr(tp):
                slot.append(
                    Int64(
                        ((generation * tp + r) * self.capacity + token) * HIDDEN
                        + column
                    )
                    * 2
                )
            multicast_store_x4(multicast + slot[self.rank], tuple(published))
            # Poll every rank's slot (this rank's too, so that the reset below
            # follows its multicast store) until none holds the sentinel.
            received = cute.make_rmem_tensor(cute.make_layout((tp, 4)), Uint32)
            start = global_timer()
            pending = Int32(1)
            while pending != Int32(0):
                pending = Int32(0)
                for r in cutlass.range_constexpr(tp):
                    words = load_volatile_x4(mailbox + slot[r])
                    for i in cutlass.range_constexpr(4):
                        received[r, i] = words[i]
                    pending = pending | empty_halves_x4(words)
                if pending != Int32(0):  # noqa: SIM102
                    if global_timer() - start > Int64(self.stall_ns):
                        # The ranks whose slot is still empty, this rank, the
                        # generation, the token and the fragment.
                        missing = Int64(0)
                        for r in cutlass.range_constexpr(tp):
                            empty = empty_halves_x4(load_volatile_x4(mailbox + slot[r]))
                            missing = missing | (Int64(empty) << r)
                        report_stall(
                            stall,
                            (Int64(1) << 62)
                            | (missing << 48)
                            | (Int64(self.rank) << 40)
                            | (Int64(generation) << 32)
                            | (Int64(token) << 16)
                            | Int64(part * self.threads + tid),
                        )
            sentinel = Uint32(SENTINEL_BF16X2)
            for r in cutlass.range_constexpr(tp):
                store_x4(mailbox + slot[r], (sentinel, sentinel, sentinel, sentinel))
            # FP32 in rank order from zero, one BF16 rounding.
            for i in cutlass.range_constexpr(4):
                lo = Float32(0.0)
                hi = Float32(0.0)
                for r in cutlass.range_constexpr(tp):
                    v_lo, v_hi = unpack_bf16x2(received[r, i])
                    lo = add_rn(lo, v_lo)
                    hi = add_rn(hi, v_hi)
                block.append(round_bf16(lo))
                block.append(round_bf16(hi))
        else:
            for i in cutlass.range_constexpr(4):
                v_lo, v_hi = unpack_bf16x2(contribution[i])
                block.append(v_lo)
                block.append(v_hi)
        if cutlass.const_expr(not self.early_trigger):
            cute.arch.griddepcontrol_launch_dependents()

        # The combine, stored as the new streams, and its sums of squares.
        combined = []
        squares = []
        out_row = streams_out + Int64(token * WIDTH + column) * 2
        for s in cutlass.range_constexpr(STREAMS):
            values = []
            words = []
            square_sum = Float32(0.0)
            for i in cutlass.range_constexpr(4):
                r_lo, r_hi = unpack_bf16x2(residual[s][i])
                lo = round_bf16(fma_rn(block[2 * i], scales[s], r_lo))
                hi = round_bf16(fma_rn(block[2 * i + 1], scales[s], r_hi))
                values.append(lo)
                values.append(hi)
                words.append(pack_bf16x2(lo, hi))
                square_sum = fma_rn(lo, lo, square_sum)
                square_sum = fma_rn(hi, hi, square_sum)
            store_x4(out_row + s * HIDDEN * 2, tuple(words))
            combined.append(values)
            squares.append(square_sum)

        # Sums of squares: warp, then CTA in warp order, then cluster in CTA
        # order (every CTA of the cluster adds the same values in the same
        # order).
        for s in cutlass.range_constexpr(STREAMS):
            value = squares[s]
            for k in cutlass.range_constexpr(5):
                value = value + cute.arch.shuffle_sync_bfly(
                    value, offset=16 >> k, mask=-1, mask_and_clamp=31
                )
            squares[s] = value
        if lane == 0:
            for s in cutlass.range_constexpr(STREAMS):
                store_shared_f32(warp_sums + (warp * STREAMS + s) * 4, squares[s])
        cute.arch.sync_threads()
        if cutlass.const_expr(self.cluster > 1):
            # Every CTA of the cluster has started (it arrived at entry).
            cute.arch.cluster_wait()
        if tid < self.cluster * STREAMS:
            stream_index = tid % STREAMS
            cta_sum = load_shared_f32(warp_sums + stream_index * 4)
            for w in cutlass.range_constexpr(1, self.warps):
                cta_sum = cta_sum + load_shared_f32(
                    warp_sums + (w * STREAMS + stream_index) * 4
                )
            store_cluster_f32(
                cluster_sums + (part * STREAMS + stream_index) * 4,
                tid // STREAMS,
                cta_sum,
            )
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        normed_row = normed + Int64(token * WIDTH + column) * 2
        copy_async_wait_all()
        for s in cutlass.range_constexpr(STREAMS):
            weight = load_shared_x4(weights + (s * self.threads + tid) * 16)
            full_sum = load_shared_f32(cluster_sums + s * 4)
            for p in cutlass.range_constexpr(1, self.cluster):
                full_sum = full_sum + load_shared_f32(
                    cluster_sums + (p * STREAMS + s) * 4
                )
            scale = norm_scale(full_sum, eps)
            normed_words = []
            for i in cutlass.range_constexpr(4):
                w_lo, w_hi = unpack_bf16x2(weight[i])
                n_lo = mul_rn(combined[s][2 * i], scale)
                n_hi = mul_rn(combined[s][2 * i + 1], scale)
                normed_words.append(
                    pack_bf16x2(fma_rn(n_lo, w_lo, n_lo), fma_rn(n_hi, w_hi, n_hi))
                )
            store_x4(normed_row + s * HIDDEN * 2, tuple(normed_words))

        if cutlass.const_expr(tp > 1):  # noqa: SIM102
            # The last CTA to arrive advances the generation for the next call.
            if tid == 0:  # noqa: SIM102
                if arrival == Uint32(rows * self.cluster - 1):
                    store_u32(
                        state + GENERATION_WORD * 4,
                        Uint32((generation + 1) % GENERATIONS),
                    )
                    store_u32(state + ARRIVALS_WORD * 4, Uint32(0))

    @cute.kernel
    def split_kernel(
        self,
        partial: Int64,
        shared: Int64,
        gate: Int64,
        expert_weights: Int64,
        expert_slots: Int64,
        streams: Int64,
        injection: Int64,
        injection_stride: Int32,
        norm: Int64,
        streams_out: Int64,
        normed: Int64,
        mailbox: Int64,
        peers: Int64,
        state: Int64,
        stall: Int64,
        eps: Float32,
    ):
        """One CTA of a split call (module docstring, Row split): a `1 /
        cluster` share of one token's hidden columns. A CTA of a row another
        rank owns stores its contribution into the owner's mailbox; the
        owner's CTA reduces the row and combines and normalizes its four
        streams."""
        tp = self.tp
        tid, _, _ = cute.arch.thread_idx()
        token, part, _ = cute.arch.block_idx()
        rows, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        column = (part * self.threads + tid) * VALUES
        # Rank q owns rows [q owned, q owned + owned); `local` is the row's
        # index among its owner's rows (the same for every CTA of a cluster).
        owned = rows // tp
        owner = token // owned
        local = token - owner * owned

        allocator = cutlass.memory.SmemAllocator()
        weights = allocator.allocate_array(
            Uint32, self.threads * STREAMS * 4, byte_alignment=16
        ).toint()
        if owner == self.rank:
            # As in the one-shot kernel: the norm weights before the wait.
            for s in cutlass.range_constexpr(STREAMS):
                copy_async_16(
                    weights + (s * self.threads + tid) * 16,
                    norm + Int64(s * HIDDEN + column) * 2,
                )
        warp_sums = allocator.allocate_array(
            Float32, self.warps * STREAMS, byte_alignment=16
        ).toint()
        cluster_sums = allocator.allocate_array(
            Float32, self.cluster * STREAMS, byte_alignment=16
        ).toint()
        generation_shared = allocator.allocate_array(
            Uint32, 1, byte_alignment=16
        ).toint()
        if cutlass.const_expr(self.cluster > 1):
            cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_wait()

        # Thread 0 reads the generation for the CTA, then arrives; the last
        # arriver advances it at its end.
        arrival = Uint32(0)
        if tid == 0:
            word = load_volatile_u32(state + GENERATION_WORD * 4)
            store_shared_u32(generation_shared, word)
            arrival = arrive_u32(state + ARRIVALS_WORD * 4, word)

        contribution = self._contribution(
            partial, shared, gate, expert_weights, expert_slots, token, column
        )
        published = []
        for i in cutlass.range_constexpr(4):
            published.append(sanitize_bf16x2(contribution[i]))
        cute.arch.sync_threads()
        generation = Int32(load_shared_u32(generation_shared))

        if owner != self.rank:
            # This rank's slot of the row in the owner's mailbox.
            slot = (
                Int64(((generation * tp + self.rank) * self.capacity + local) * HIDDEN)
                + Int64(column)
            ) * 2
            store_x4_peer(load_u64(peers + Int64(owner) * 8) + slot, tuple(published))
            cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(self.cluster > 1):
                # Matches the arrival at entry; no peer touches this CTA's
                # shared memory (the whole cluster takes this branch).
                cute.arch.cluster_wait()
        else:
            # The owner's streams and injection logits of the row.
            row = streams + Int64(local * WIDTH + column) * 2
            residual = []
            scales = []
            for s in cutlass.range_constexpr(STREAMS):
                residual.append(load_x4(row + s * HIDDEN * 2))
                scales.append(
                    injection_scale(
                        load_bf16(injection + Int64(local * injection_stride + s) * 2)
                    )
                )
            slots = []
            for r in cutlass.range_constexpr(tp):
                slots.append(
                    (
                        Int64(((generation * tp + r) * self.capacity + local) * HIDDEN)
                        + Int64(column)
                    )
                    * 2
                )
            # Poll the peers' slots until none holds the sentinel.
            received = cute.make_rmem_tensor(cute.make_layout((tp, 4)), Uint32)
            start = global_timer()
            pending = Int32(1)
            while pending != Int32(0):
                pending = Int32(0)
                for r in cutlass.range_constexpr(tp):
                    if cutlass.const_expr(r != self.rank):
                        words = load_volatile_x4(mailbox + slots[r])
                        for i in cutlass.range_constexpr(4):
                            received[r, i] = words[i]
                        pending = pending | empty_halves_x4(words)
                if pending != Int32(0):  # noqa: SIM102
                    if global_timer() - start > Int64(self.stall_ns):
                        missing = Int64(0)
                        for r in cutlass.range_constexpr(tp):
                            if cutlass.const_expr(r != self.rank):
                                empty = empty_halves_x4(
                                    load_volatile_x4(mailbox + slots[r])
                                )
                                missing = missing | (Int64(empty) << r)
                        report_stall(
                            stall,
                            (Int64(1) << 62)
                            | (missing << 48)
                            | (Int64(self.rank) << 40)
                            | (Int64(generation) << 32)
                            | (Int64(token) << 16)
                            | Int64(part * self.threads + tid),
                        )
            sentinel = Uint32(SENTINEL_BF16X2)
            for r in cutlass.range_constexpr(tp):
                if cutlass.const_expr(r != self.rank):
                    store_x4(
                        mailbox + slots[r], (sentinel, sentinel, sentinel, sentinel)
                    )
            # FP32 in rank order from zero, one BF16 rounding: the one-shot sum.
            block = []
            for i in cutlass.range_constexpr(4):
                lo = Float32(0.0)
                hi = Float32(0.0)
                for r in cutlass.range_constexpr(tp):
                    word = published[i]
                    if cutlass.const_expr(r != self.rank):
                        word = received[r, i]
                    v_lo, v_hi = unpack_bf16x2(word)
                    lo = add_rn(lo, v_lo)
                    hi = add_rn(hi, v_hi)
                block.append(round_bf16(lo))
                block.append(round_bf16(hi))
            cute.arch.griddepcontrol_launch_dependents()

            # The combine and norm of the one-shot kernel, at the row's local
            # index.
            combined = []
            squares = []
            out_row = streams_out + Int64(local * WIDTH + column) * 2
            for s in cutlass.range_constexpr(STREAMS):
                values = []
                words = []
                square_sum = Float32(0.0)
                for i in cutlass.range_constexpr(4):
                    r_lo, r_hi = unpack_bf16x2(residual[s][i])
                    lo = round_bf16(fma_rn(block[2 * i], scales[s], r_lo))
                    hi = round_bf16(fma_rn(block[2 * i + 1], scales[s], r_hi))
                    values.append(lo)
                    values.append(hi)
                    words.append(pack_bf16x2(lo, hi))
                    square_sum = fma_rn(lo, lo, square_sum)
                    square_sum = fma_rn(hi, hi, square_sum)
                store_x4(out_row + s * HIDDEN * 2, tuple(words))
                combined.append(values)
                squares.append(square_sum)
            for s in cutlass.range_constexpr(STREAMS):
                value = squares[s]
                for k in cutlass.range_constexpr(5):
                    value = value + cute.arch.shuffle_sync_bfly(
                        value, offset=16 >> k, mask=-1, mask_and_clamp=31
                    )
                squares[s] = value
            if lane == 0:
                for s in cutlass.range_constexpr(STREAMS):
                    store_shared_f32(warp_sums + (warp * STREAMS + s) * 4, squares[s])
            cute.arch.sync_threads()
            if cutlass.const_expr(self.cluster > 1):
                cute.arch.cluster_wait()
            if tid < self.cluster * STREAMS:
                stream_index = tid % STREAMS
                cta_sum = load_shared_f32(warp_sums + stream_index * 4)
                for w in cutlass.range_constexpr(1, self.warps):
                    cta_sum = cta_sum + load_shared_f32(
                        warp_sums + (w * STREAMS + stream_index) * 4
                    )
                store_cluster_f32(
                    cluster_sums + (part * STREAMS + stream_index) * 4,
                    tid // STREAMS,
                    cta_sum,
                )
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()

            normed_row = normed + Int64(local * WIDTH + column) * 2
            copy_async_wait_all()
            for s in cutlass.range_constexpr(STREAMS):
                weight = load_shared_x4(weights + (s * self.threads + tid) * 16)
                full_sum = load_shared_f32(cluster_sums + s * 4)
                for p in cutlass.range_constexpr(1, self.cluster):
                    full_sum = full_sum + load_shared_f32(
                        cluster_sums + (p * STREAMS + s) * 4
                    )
                scale = norm_scale(full_sum, eps)
                normed_words = []
                for i in cutlass.range_constexpr(4):
                    w_lo, w_hi = unpack_bf16x2(weight[i])
                    n_lo = mul_rn(combined[s][2 * i], scale)
                    n_hi = mul_rn(combined[s][2 * i + 1], scale)
                    normed_words.append(
                        pack_bf16x2(fma_rn(n_lo, w_lo, n_lo), fma_rn(n_hi, w_hi, n_hi))
                    )
                store_x4(normed_row + s * HIDDEN * 2, tuple(normed_words))

        # The last CTA to arrive advances the generation for the next call.
        if tid == 0:  # noqa: SIM102
            if arrival == Uint32(rows * self.cluster - 1):
                store_u32(
                    state + GENERATION_WORD * 4,
                    Uint32((generation + 1) % GENERATIONS),
                )
                store_u32(state + ARRIVALS_WORD * 4, Uint32(0))

    def _contribution(
        self,
        partial: Int64,
        shared: Int64,
        gate: Int64,
        expert_weights: Int64,
        expert_slots: Int64,
        token: Int32,
        column: Int32,
    ) -> list[Uint32]:
        """This rank's contribution at the thread's eight values of `token`,
        four words of BF16 pairs, as the one-shot kernel computes it: the
        partial, or for `moe` the gated shared expert output plus the routed
        output. Part of the kernel body (traced inline, no dynamic control
        flow); runs after the grid-dependency wait."""
        source = Int64(token * HIDDEN + column) * 2
        if not self.moe:
            return list(load_x4(partial + source))
        routed_values = []
        if self.fold > 0:
            routed_values = self._fold(
                partial, expert_weights, expert_slots, token, column
            )
        else:
            routed = load_x4(partial + source)
            for i in range(4):
                r_lo, r_hi = unpack_bf16x2(routed[i])
                routed_values.append(r_lo)
                routed_values.append(r_hi)
        expert = load_x4(shared + source)
        g = _shared_gate(load_bf16(gate + Int64(token) * 2))
        contribution = []
        for i in range(4):
            e_lo, e_hi = unpack_bf16x2(expert[i])
            contribution.append(
                pack_bf16x2(
                    add_rn(round_bf16(mul_rn(e_lo, g)), routed_values[2 * i]),
                    add_rn(round_bf16(mul_rn(e_hi, g)), routed_values[2 * i + 1]),
                )
            )
        return contribution

    def _fold(
        self,
        rows: Int64,
        expert_weights: Int64,
        expert_slots: Int64,
        token: Int32,
        column: Int32,
    ) -> list[Float32]:
        """The routed output at this thread's eight values of `token`: the
        finalize's weighted sum of the token's GEMM2 rows, in its order
        (module docstring), rounded to BF16, as eight FP32 values. Part of the
        kernel body (traced inline, no dynamic control flow); runs after the
        grid-dependency wait."""
        slots = []
        weights = []
        for k in range(TOP_K):
            entry = Int64(token * TOP_K + k)
            slots.append(load_s32(expert_slots + entry * 4))
            weights.append(load_bf16(expert_weights + entry * 2))
        values = []
        for k in range(TOP_K):
            values.append(
                load_x4_if(rows + (Int64(slots[k]) * HIDDEN + column) * 2, slots[k])
            )
        # One FP32 sum per group and value; expert k adds to group k % fold.
        sums = []
        for _ in range(self.fold):
            sums.append([Float32(0.0)] * VALUES)
        for k in range(TOP_K):
            group = sums[k % self.fold]
            for i in range(4):
                v_lo, v_hi = unpack_bf16x2(values[k][i])
                group[2 * i] = fma_if(weights[k], v_lo, group[2 * i], slots[k])
                group[2 * i + 1] = fma_if(weights[k], v_hi, group[2 * i + 1], slots[k])
        routed = []
        for v in range(VALUES):
            total = sums[0][v]
            for g in range(1, self.fold):
                total = add_rn(total, sums[g][v])
            routed.append(round_bf16(total))
        return routed


@cache
def _compiled(
    device_index: int,
    tp: int,
    rank: int,
    capacity: int,
    moe: bool,
    early: bool,
    cluster: int,
    stall_ns: int,
    fold: int = 0,
    split: bool = False,
):
    """`BoundaryKernel` compiled for a device, rank and contribution."""
    with torch.accelerator.device_index(device_index):
        return cute.compile(
            BoundaryKernel(
                tp, rank, capacity, moe, early, cluster, stall_ns, fold, split
            ),
            *(Int64(0),) * 7,
            Int32(STREAMS),
            *(Int64(0),) * 8,
            Float32(1e-6),
            Int32(1),
            make_fake_stream(),
        )


class SublayerBoundary:
    """The sublayer boundaries of one model replica on one rank.

    Construction is collective over `group` (every rank builds its mailbox
    and maps the others'); `group` None or of one rank builds the local
    variant. Calls are captured into CUDA graphs; every rank must make the
    same calls in the same order.

    Args:
        group: The tensor-parallel process group, or None at TP 1.
        max_rows: Token rows of the largest call.
        device: This rank's GPU.
        early_trigger: Let the next kernel launch at entry (it then loads its
            weights during the exchange) rather than once the block output is
            known.
        stall_timeout: Seconds a thread polls for the other ranks' data
            before it reports a stall (`stall_report`) and traps.
        split: Also build the row-split calls' mailbox (module docstring,
            Row split; TP > 1 only).

    """

    def __init__(
        self,
        group: dist.ProcessGroup | None,
        max_rows: int,
        device: torch.device,
        early_trigger: bool = False,
        stall_timeout: float = STALL_TIMEOUT_S,
        split: bool = False,
    ) -> None:
        self.group = group
        self.tp = 1 if group is None else dist.get_world_size(group)
        self.rank = 0 if group is None else dist.get_rank(group)
        self.max_rows = max_rows
        self.early_trigger = early_trigger
        self.stall_ns = int(stall_timeout * 1e9)
        self.device = torch.device(device)
        self._multicast = 0
        # The mailbox and protocol words (unused at TP 1).
        self._mailbox = torch.empty(0, dtype=torch.bfloat16, device=self.device)
        self._state = torch.zeros(2, dtype=torch.int32, device=self.device)
        # The split calls' mailbox, its protocol words and every rank's
        # mailbox address (``split`` only).
        self.split = split and self.tp > 1
        self.split_capacity = -(-max_rows // self.tp)
        self._split_mailbox = torch.empty(0, dtype=torch.bfloat16, device=self.device)
        self._split_state = torch.zeros(2, dtype=torch.int32, device=self.device)
        self._split_peers = torch.zeros(1, dtype=torch.int64, device=self.device)
        # The stall word, in pinned host memory that the host can still read
        # after the kernel trapped, and the device's view of it.
        self._stall_host = torch.zeros(1, dtype=torch.int64)
        self._stall = torch.zeros(1, dtype=torch.int64, device=self.device)
        if self.tp > 1:
            assert group is not None
            self._stall_host = self._stall_host.pin_memory()
            self._stall = get_accelerator_view_from_cpu_tensor(self._stall_host)
            self._mailbox = symm_mem.empty(
                (GENERATIONS, self.tp, max_rows, HIDDEN),
                dtype=torch.bfloat16,
                device=self.device,
            )
            handle = symm_mem.rendezvous(self._mailbox, group)
            multicast = int(handle.multicast_ptr or 0)
            if not multicast or multicast % 16:
                raise RuntimeError("NVLink multicast mapping is unavailable")
            self._multicast = multicast
            # Every slot starts empty: BF16 negative zero.
            self._mailbox.view(torch.int16).fill_(-32768)
            if self.split:
                # [generation][source rank][owned row][2560].
                self._split_mailbox = symm_mem.empty(
                    (GENERATIONS, self.tp, self.split_capacity, HIDDEN),
                    dtype=torch.bfloat16,
                    device=self.device,
                )
                split_handle = symm_mem.rendezvous(self._split_mailbox, group)
                self._split_peers = torch.tensor(
                    [int(split_handle.buffer_ptrs[r]) for r in range(self.tp)],
                    dtype=torch.int64,
                    device=self.device,
                )
                self._split_mailbox.view(torch.int16).fill_(-32768)
            torch.accelerator.synchronize(self.device)
            dist.barrier(group=group)

    def stall_report(self) -> str | None:
        """What a stalled call reported before it trapped, or None. Reads host
        memory only, so it works after the CUDA error."""
        word = int(self._stall_host[0])
        if not word:
            return None
        missing = [r for r in range(self.tp) if word >> (48 + r) & 1]
        return (
            f"rank {word >> 40 & 0xFF} timed out in generation {word >> 32 & 0xFF} "
            f"waiting for the data of rank(s) {missing} (token {word >> 16 & 0xFFFF}, "
            f"fragment {word & 0xFFFF})"
        )

    def warm_up(
        self,
        norm_weight: torch.Tensor,
        eps: float,
        attention: bool,
        moe: bool,
        folded: bool = False,
    ) -> None:
        """Compile the attention and/or MoE variants (`folded`: also the
        folded MoE variants of every row count up to `max_rows`), wait for
        every rank, then run each once on one row of zeros. Collective over
        the group: no rank starts polling while another still compiles."""
        kinds = [(False, 0)] if attention else []
        if moe:
            kinds.append((True, 0))
        if folded:
            folds = {finalize_groups(rows) for rows in range(1, self.max_rows + 1)}
            kinds.extend((True, fold) for fold in sorted(folds))
        for kind, fold in kinds:
            self._kernel(kind, fold)
            if self.split:
                self._kernel(kind, fold, True)
        if self.tp > 1:
            torch.accelerator.synchronize(self.device)
            dist.barrier(group=self.group)
        # One call of one row (split: one row per rank, of tp rows).
        for split in (False, True) if self.split else (False,):
            rows = self.tp if split else 1
            streams = norm_weight.new_zeros(1, WIDTH)
            partial = norm_weight.new_zeros(rows, HIDDEN)
            injection = norm_weight.new_zeros(1, STREAMS)
            logit = norm_weight.new_zeros(rows, 1)
            # Tokens whose experts are all held by other ranks: no row is read.
            weights = norm_weight.new_zeros(rows, TOP_K)
            slots = torch.full((rows, TOP_K), -1, dtype=torch.int32, device=self.device)
            for kind, fold in kinds:
                self._launch(
                    kind,
                    partial,
                    partial if kind else None,
                    logit if kind else None,
                    streams,
                    injection,
                    norm_weight,
                    eps,
                    (weights, slots, fold) if fold else None,
                    split,
                )
        torch.accelerator.synchronize(self.device)

    def _kernel(self, moe: bool, fold: int = 0, split: bool = False):
        """The compiled kernel of a contribution kind for this rank (`split`:
        the row-split kernel)."""
        return _compiled(
            self.device.index,
            self.tp,
            self.rank,
            self.split_capacity if split else self.max_rows,
            moe,
            False if split else self.early_trigger,
            CLUSTER,
            self.stall_ns,
            fold,
            split,
        )

    def __call__(
        self,
        partial: torch.Tensor,
        streams: torch.Tensor,
        injection: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        split: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The boundary after an attention block.

        Args:
            partial: This rank's partial block output, contiguous BF16
                `[rows, 2560]`.
            streams: The residual streams, contiguous BF16 `[rows, 10240]`.
            injection: The injection logits of the read that fed the block,
                BF16 `[rows, 4]` with unit column stride.
            norm_weight: The next read's grouped norm weight, BF16 `[10240]`.
            eps: Its epsilon.
            split: A row-split call (module docstring, Row split): `streams`
                and `injection` hold this rank's `rows / tp` rows only.

        Returns:
            The combined streams and their normalized copy `xn`, BF16
            `[rows, 10240]`: what `hc_combine_norm` returns, after the
            all-reduce of `partial` (`split`: of this rank's rows).

        """
        return self._launch(
            False,
            partial,
            None,
            None,
            streams,
            injection,
            norm_weight,
            eps,
            None,
            split,
        )

    def moe(
        self,
        routed: torch.Tensor,
        shared: torch.Tensor,
        gate_logit: torch.Tensor,
        streams: torch.Tensor,
        injection: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        split: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The boundary after a MoE block, with the shared expert's gate, its
        product and the routed add folded in.

        Args:
            routed: This rank's routed expert output, contiguous BF16
                `[rows, 2560]`.
            shared: This rank's shared expert output before its gate,
                contiguous BF16 `[rows, 2560]`.
            gate_logit: The shared expert gate's logits, BF16 `[rows, 1]`.
            streams: As in `__call__`.
            injection: As in `__call__`.
            norm_weight: As in `__call__`.
            eps: As in `__call__`.
            split: As in `__call__`.

        Returns:
            As `__call__` for the partial
            `sigmoid(gate_logit) * shared + routed`.

        """
        assert shared.shape == routed.shape and shared.is_contiguous()
        assert gate_logit.numel() == routed.shape[0] and gate_logit.is_contiguous()
        return self._launch(
            True,
            routed,
            shared,
            gate_logit,
            streams,
            injection,
            norm_weight,
            eps,
            None,
            split,
        )

    def moe_folded(
        self,
        expert_rows: torch.Tensor,
        expert_weights: torch.Tensor,
        expert_slots: torch.Tensor,
        shared: torch.Tensor,
        gate_logit: torch.Tensor,
        streams: torch.Tensor,
        injection: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        split: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The boundary after a MoE block whose routed experts stopped after
        their second GEMM: as `moe`, with the routed output the finalize
        would have written from these operands (module docstring).

        Args:
            expert_rows: The routed experts' unweighted GEMM2 rows, contiguous
                BF16 `[permuted rows, 2560]`.
            expert_weights: Each token's top-10 routing weights, contiguous
                BF16 `[rows, 10]` (or flat).
            expert_slots: The GEMM2 row of each token's top-10 experts (-1 for
                an expert another rank holds), contiguous int32 `[rows, 10]`
                (or flat).
            shared: As in `moe`.
            gate_logit: As in `moe`.
            streams: As in `__call__`.
            injection: As in `__call__`.
            norm_weight: As in `__call__`.
            eps: As in `__call__`.
            split: As in `__call__`.

        Returns:
            As `moe` for the routed output those operands finalize to.

        """
        rows = shared.shape[0]
        assert expert_rows.dtype == torch.bfloat16 and expert_rows.is_contiguous()
        assert expert_rows.dim() == 2 and expert_rows.shape[1] == HIDDEN
        assert expert_weights.dtype == torch.bfloat16
        assert expert_weights.numel() == rows * TOP_K
        assert expert_weights.is_contiguous()
        assert expert_slots.dtype == torch.int32
        assert expert_slots.numel() == rows * TOP_K and expert_slots.is_contiguous()
        assert shared.is_contiguous()
        assert gate_logit.numel() == rows and gate_logit.is_contiguous()
        return self._launch(
            True,
            expert_rows,
            shared,
            gate_logit,
            streams,
            injection,
            norm_weight,
            eps,
            (expert_weights, expert_slots, finalize_groups(rows)),
            split,
        )

    def _launch(
        self,
        moe: bool,
        partial: torch.Tensor,
        shared: torch.Tensor | None,
        gate: torch.Tensor | None,
        streams: torch.Tensor,
        injection: torch.Tensor,
        norm_weight: torch.Tensor,
        eps: float,
        fold: tuple[torch.Tensor, torch.Tensor, int] | None = None,
        split: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Check the operands, allocate the outputs and launch the kernel;
        `fold` is the folded top-k reduction's weights, slots and group
        count, with `partial` the GEMM2 rows; `split` launches the
        row-split kernel, `streams` and `injection` holding this rank's
        `rows / tp` rows."""
        assert not split or self.split, "built without the split mailbox"
        owned = streams.shape[0]
        rows = owned * self.tp if split else owned
        assert 1 <= rows <= self.max_rows, f"{rows} rows, at most {self.max_rows}"
        assert not split or owned <= self.split_capacity
        assert partial.dtype == streams.dtype == norm_weight.dtype == torch.bfloat16
        assert partial.shape[1:] == (HIDDEN,) and partial.is_contiguous()
        assert fold is not None or partial.shape[0] == rows
        assert shared is None or shared.shape == (rows, HIDDEN)
        assert streams.shape == (owned, WIDTH) and streams.is_contiguous()
        assert injection.shape == (owned, STREAMS) and injection.stride(1) == 1
        assert norm_weight.numel() == WIDTH and norm_weight.is_contiguous()
        weights, slots, groups = (None, None, 0) if fold is None else fold
        # The outputs are allocated while every input is referenced, so the
        # allocator cannot hand them an input's memory.
        streams_out = torch.empty_like(streams)
        normed = torch.empty_like(streams)
        mailbox = self._split_mailbox if split else self._mailbox
        state = self._split_state if split else self._state
        self._kernel(moe, groups, split)(
            Int64(partial.data_ptr()),
            Int64(0 if shared is None else shared.data_ptr()),
            Int64(0 if gate is None else gate.data_ptr()),
            Int64(0 if weights is None else weights.data_ptr()),
            Int64(0 if slots is None else slots.data_ptr()),
            Int64(streams.data_ptr()),
            Int64(injection.data_ptr()),
            Int32(injection.stride(0)),
            Int64(norm_weight.data_ptr()),
            Int64(streams_out.data_ptr()),
            Int64(normed.data_ptr()),
            Int64(mailbox.data_ptr()),
            Int64(self._multicast),
            Int64(self._split_peers.data_ptr()),
            Int64(state.data_ptr()),
            Int64(self._stall.data_ptr()),
            Float32(eps),
            Int32(rows),
            cuda.CUstream(torch.cuda.current_stream().cuda_stream),
        )
        return streams_out, normed
