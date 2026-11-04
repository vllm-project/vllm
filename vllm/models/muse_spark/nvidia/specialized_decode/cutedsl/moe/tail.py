# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team
# The finalize publisher is FlashInfer's (_NormFinalizePublishDeviceKernel in
# flashinfer/comm/mnnvl_cutedsl/kernel_ll/moe_finalize_norm_all_reduce.py) and
# the reduction is its Lamport reduce (_LamportResidualRMSNormDeviceKernel in
# flashinfer/comm/mnnvl_cutedsl/kernel_ll/device_kernels.py) in the no-norm form
# the MoE finalize compiles, with its spin bounded here. The TP tail's
# arithmetic, decomposition and register helpers are those of FlashInfer's
# single-launch token-owner TP finalize (single_kernel and its helpers in
# flashinfer/comm/tp_moe_finalize_rmsnorm.py); its transport, the in-CTA
# expert combine and the PDL wiring are not. All from
# flashinfer-ai/flashinfer@14ba7aaa84718abe7a5648184fa9ad15362e2880,
# Apache-2.0.
# FlashInfer NOTICE: "FlashInfer / Copyright 2025-2026 NVIDIA / Copyright
# 2023-2026 FlashInfer community (https://flashinfer.ai/)".

"""The MoE tail of the Muse Spark specialized decode.

With expert parallelism every rank holds the GEMM2 rows of its local experts
for the replicated tokens (``UnfinalizedMoEOutput``). The tail (``MoETail``)
RMS-normalizes each local route's row over the pooled width (post_expert_norm),
weights it by its route weight, sums a token's routes in FP32, and all-reduces
the per-rank BF16 partial sums (the EP combine). Per token, over the routes
``r`` local to a rank with row ``x_r`` and weight ``w_r``::

    s_r = x_r * (gamma + gain_center)       # weight_before_norm
    partial = bf16(sum_r w_r * s_r * rsqrt(mean(s_r**2) + eps))
    out = bf16(sum over ranks, in rank order, of partial)  # FP32 sum

Without ``weight_before_norm`` the weight multiplies after the norm. The output
is bitwise the stock MNNVL MoE finalize's for the same rows, identical on every
rank. It takes two kernels over one Lamport mailbox: a publisher that
multicasts each token's partial, and a reduction that sums the ``tp`` slots.

Under expert TP (``TpTail``) every rank holds a K-partial of every route's
GEMM2 row instead. The rows are summed across ranks before the norm, bitwise
as the stock TP finalize (``tp_moe_finalize_rmsnorm``) on its single-launch
token-owner path::

    x_r = bf16(sum over ranks, in rank order, of the K-partials)  # FP32 sum
    out = bf16(sum_r fp32(w_r * norm(x_r)))    # routes in order, from 0.0

See ``_TpTailKernel``.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, NamedTuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import torch
import torch.distributed as dist
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.runtime import make_fake_compact_tensor

from ..common.host import assert_not_capturing
from ..common.mailbox import SPIN_TIMEOUT_NS, LamportMailbox, poll, poll_at, rearm
from ..common.policy import MAILBOX_CAPACITY
from ..common.primitives import (
    ACTIVE_STAGE,
    LAMPORT_GENERATIONS,
    NEXT_STAGE,
    VEC_BF16,
    WARP_SIZE,
    bf16x2_to_f32x2,
    bf16x8_to_packed_u32x4,
    current_cu_stream,
    load_bf16x8_as_f32,
    load_global_u32x4,
    load_volatile_u32,
    make_fake_dynamic_compact_tensor,
    map_shared_to_peer,
    mul_rn,
    pack_bf16x2_rn,
    packed_u32x4_to_bf16x8,
    sanitize_negative_zero_u32x4,
    stmc_bf16x8,
    store_global_u32,
    store_global_u32x4,
    store_shared_cluster_f32,
    to_cute,
    to_cute_dynamic,
)

if TYPE_CHECKING:
    from vllm.model_executor.layers.fused_moe.moe_output import UnfinalizedMoEOutput

# The pooled width the experts work in (the post-expert norm's hidden size).
POOLED = 4096
TOP_K = 8
# The publisher's CTA and the reduction's CTA and cluster (the stock shapes).
THREADS = 128
REDUCE_CLUSTER = 4


# The vendored publisher, unchanged (the tests compare it with the reference).
class _NormFinalizePublishDeviceKernel:
    def __init__(
        self,
        *,
        hidden: int,
        row_stride: int,
        topk: int,
        tp: int,
        rank: int,
        capacity_m: int,
        threads: int,
        weight_before_norm: bool,
        eps: float,
        gain_center: float,
        enable_pdl: bool,
    ) -> None:
        if threads % WARP_SIZE or hidden % (threads * VEC_BF16):
            raise ValueError("hidden must split evenly into 8-element fragments")
        self.hidden = hidden
        self.row_stride = row_stride
        self.topk = topk
        self.tp = tp
        self.rank = rank
        self.capacity_m = capacity_m
        self.threads = threads
        self.warps = threads // WARP_SIZE
        self.trips = hidden // (threads * VEC_BF16)
        self.weight_before_norm = weight_before_norm
        self.eps = eps
        self.gain_center = gain_center
        self.enable_pdl = enable_pdl

    @cute.jit
    def __call__(
        self,
        routed_rows: cute.Tensor,
        route_weights: cute.Tensor,
        permuted_idx: cute.Tensor,
        norm_weight: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        m: Int32,
        num_rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            routed_rows,
            route_weights,
            permuted_idx,
            norm_weight,
            stage_state,
            contribution_mailbox_multicast_address,
            num_rows,
        ).launch(
            grid=(m, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.topk * self.warps * 4,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        routed_rows: cute.Tensor,
        route_weights: cute.Tensor,
        permuted_idx: cute.Tensor,
        norm_weight: cute.Tensor,
        stage_state: cute.Tensor,
        contribution_mailbox_multicast_address: Int64,
        num_rows: Int32,
    ) -> None:
        token, _, _ = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = cute.arch.lane_idx()

        frag_layout = cute.make_layout((self.trips, VEC_BF16), stride=(VEC_BF16, 1))
        gamma = cute.make_rmem_tensor(frag_layout, Float32)
        for trip in cutlass.range_constexpr(self.trips):
            column = Int64(tidx + trip * self.threads) * VEC_BF16
            gamma[trip, None].store(
                load_bf16x8_as_f32(norm_weight, column) + Float32(self.gain_center)
            )

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        smem = cutlass.utils.SmemAllocator()
        # One slot set per route avoids a second barrier per route.
        warp_sums = smem.allocate_array(Float32, self.topk * self.warps)
        scaled = cute.make_rmem_tensor(frag_layout, Float32)
        acc = cute.make_rmem_tensor(frag_layout, Float32)
        acc.fill(Float32(0.0))
        for k in cutlass.range_constexpr(self.topk):
            route = Int64(token) * self.topk + k
            row = Int32(permuted_idx[route])
            is_local = row >= Int32(0) and row < num_rows
            # All threads in this token's CTA have the same route index.
            if is_local:
                for trip in cutlass.range_constexpr(self.trips):
                    column = Int64(tidx + trip * self.threads) * VEC_BF16
                    values = load_bf16x8_as_f32(
                        routed_rows, Int64(row) * self.row_stride + column
                    )
                    if cutlass.const_expr(self.weight_before_norm):
                        values = values * gamma[trip, None].load()
                    scaled[trip, None].store(values)
                sum_sq = (scaled.load() * scaled.load()).reduce(
                    cute.ReductionOp.ADD, init_val=Float32(0.0), reduction_profile=0
                )
                for offset in (16, 8, 4, 2, 1):
                    sum_sq = sum_sq + cute.arch.shuffle_sync_bfly(sum_sq, offset=offset)
                if lane == 0:
                    cute.arch.store(
                        (warp_sums + (k * self.warps + warp)).llvm_ptr, sum_sq
                    )
                cute.arch.barrier()
                total = Float32(0.0)
                for w in cutlass.range_constexpr(self.warps):
                    total = total + cute.arch.load(
                        (warp_sums + (k * self.warps + w)).llvm_ptr, Float32
                    )
                inv_rms = cute.math.rsqrt(
                    total * Float32(1.0 / self.hidden) + Float32(self.eps),
                    fastmath=True,
                )
                weight = Float32(route_weights[route])
                for trip in cutlass.range_constexpr(self.trips):
                    normalized = scaled[trip, None].load() * inv_rms
                    if cutlass.const_expr(not self.weight_before_norm):
                        normalized = normalized * gamma[trip, None].load()
                    acc[trip, None].store(acc[trip, None].load() + normalized * weight)

        stage = load_volatile_u32(stage_state.iterator + NEXT_STAGE)
        for trip in cutlass.range_constexpr(self.trips):
            column = Int64(tidx + trip * self.threads) * VEC_BF16
            mailbox_element = (
                (Int64(stage) * self.tp + self.rank) * self.capacity_m + Int64(token)
            ) * self.hidden + column
            stmc_bf16x8(
                contribution_mailbox_multicast_address + mailbox_element * 2,
                sanitize_negative_zero_u32x4(
                    bf16x8_to_packed_u32x4(acc[trip, None].load().to(BFloat16))
                ),
            )
        if token == 0 and tidx == 0:
            store_global_u32(stage_state.iterator + ACTIVE_STAGE, stage)
        cute.arch.barrier()
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()


class _TailReduceKernel:
    """Sum the ``tp`` partials of every token from the mailbox into ``output``.

    Each 8-column fragment waits for all ``tp`` slots (bounded), adds them in
    rank order in FP32 from 0.0 and stores the BF16 sum; then the stage
    advances, dependents are released and the slots are re-armed.
    """

    def __init__(
        self,
        *,
        hidden: int,
        tp: int,
        capacity_m: int,
        cluster_size: int,
        threads: int,
        enable_pdl: bool,
        timeout_ns: int = SPIN_TIMEOUT_NS,
    ) -> None:
        if threads <= 0 or threads % WARP_SIZE:
            raise ValueError("threads must be a positive multiple of the warp size")
        if hidden <= 0 or hidden % (cluster_size * threads * VEC_BF16):
            raise ValueError("hidden must split evenly into the cluster's fragments")
        if not 1 <= tp <= 32:
            raise ValueError("tp must be between 1 and 32")
        self.hidden = hidden
        self.tp = tp
        self.capacity_m = capacity_m
        self.cluster_size = cluster_size
        self.threads = threads
        self.enable_pdl = enable_pdl
        self.timeout_ns = timeout_ns
        self.fragment_stride = cluster_size * threads
        self.trips = hidden // (self.fragment_stride * VEC_BF16)
        # Elements between one rank's slot and the next.
        self.rank_stride = capacity_m * hidden

    @cute.jit
    def __call__(
        self,
        contribution_mailbox: cute.Tensor,
        output: cute.Tensor,
        stage_state: cute.Tensor,
        m: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(contribution_mailbox, output, stage_state).launch(
            grid=(m, self.cluster_size, 1),
            block=(self.threads, 1, 1),
            cluster=(1, self.cluster_size, 1),
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    @cute.kernel
    def kernel(
        self,
        contribution_mailbox: cute.Tensor,
        output: cute.Tensor,
        stage_state: cute.Tensor,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        token, _, _ = cute.arch.block_idx()
        cluster_rank = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        base_fragment = cluster_rank * self.threads + tidx

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        active_stage = load_volatile_u32(stage_state.iterator + ACTIVE_STAGE)
        stage_base = (Int64(active_stage) * self.tp) * self.capacity_m + Int64(token)
        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            source_pointer = cute.make_ptr(
                BFloat16,
                (
                    contribution_mailbox.iterator
                    + (stage_base * self.hidden + Int64(fragment) * VEC_BF16)
                ).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            slots = poll(
                source_pointer, Int64(self.rank_stride), self.tp, self.timeout_ns
            )
            lane_sum = cute.make_rmem_tensor(cute.make_layout((VEC_BF16,)), Float32)
            lane_sum.fill(Float32(0.0))
            for wave in cutlass.range_constexpr(self.tp):
                packed = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for word in cutlass.range_constexpr(4):
                    packed[word] = slots[wave, word]
                lane_sum.store(
                    lane_sum.load() + packed_u32x4_to_bf16x8(packed.load()).to(Float32)
                )
            output_element = Int64(token) * self.hidden + Int64(fragment) * VEC_BF16
            store_global_u32x4(
                Int64((output.iterator + output_element).toint()),
                bf16x8_to_packed_u32x4(lane_sum.load().to(BFloat16)),
            )

        if token == 0 and cluster_rank == 0 and tidx == 0:
            store_global_u32(
                stage_state.iterator + NEXT_STAGE,
                (active_stage + Uint32(1)) % Uint32(LAMPORT_GENERATIONS),
            )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        for trip in cutlass.range_constexpr(self.trips):
            fragment = base_fragment + trip * self.fragment_stride
            source_pointer = cute.make_ptr(
                BFloat16,
                (
                    contribution_mailbox.iterator
                    + (stage_base * self.hidden + Int64(fragment) * VEC_BF16)
                ).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            for wave in cutlass.range_constexpr(self.tp):
                rearm(source_pointer + Int64(wave * self.rank_stride))


# ---------------------------------------------------------------- expert TP
# The TP finalize's split rule: CTAs per token band grow until the padded
# token grid covers the SM array, at most this many.
TP_MAX_SPLIT = 16


def tp_split(m: int, tp: int, nsm: int) -> int:
    """H bands per token of the TP tail for ``m`` tokens on ``nsm`` SMs, as
    the stock TP finalize picks them; the split fixes the sum-of-squares tree,
    so the tails agree bitwise."""
    ctas = -(-m // tp) * tp
    split = 1
    while split < TP_MAX_SPLIT and ctas * split * 2 <= nsm:
        split *= 2
    return split


class _TpTailKernel:
    """The expert-TP MoE tail, one launch: grid (split, slots, tp), clusters of
    the ``split`` H bands of one (slot, d), ``TOP_K`` x ``band_threads``
    threads; CTA (b, g, d) works on H band b of token t = g * tp + d. Rank
    ``t % tp`` owns token ``t``.

    - Send: every rank sends its K-partial rows of token t's routes to the
      owner's route mailbox (unicast), -0.0 as +0.0.
    - Sum: the owner sums each route's partials across ranks in rank order,
      RMS-normalizes and weights them, sums the routes in order and
      multicasts the finished BF16 row to every rank's token mailbox.
    - Copy: every rank copies the rows it does not own from its token mailbox
      into the output.

    The sum's arithmetic, order and BF16 roundings are the stock TP
    finalize's (``tp_split`` gives its sum-of-squares tree).

    Each CTA reads its generation word(s) after griddepcontrol.wait and
    advances them at exit; the words are per finest band (``TP_MAX_SPLIT``),
    so every word of a slot advances once per launch whatever the split and
    all ranks agree on it. A slot is re-armed by its reader right after the
    read, and its next writer is at least one launch later.
    """

    def __init__(
        self,
        *,
        tp: int,
        rank: int,
        split: int,
        slots: int,
        route_peers: tuple[int, ...],
        constants: "TailConstants",
        weight_dtype: torch.dtype,
        enable_pdl: bool,
        timeout_ns: int = SPIN_TIMEOUT_NS,
    ) -> None:
        if not 2 <= tp <= 32 or len(route_peers) != tp:
            raise ValueError("the TP tail needs 2-32 ranks and every peer")
        if split < 2 or TP_MAX_SPLIT % split:
            raise ValueError(f"split must divide {TP_MAX_SPLIT} and exceed 1")
        band_threads = POOLED // (VEC_BF16 * split)
        if band_threads % WARP_SIZE:
            raise ValueError("a band must split into whole warps")
        self.tp, self.rank, self.split, self.slots = tp, rank, split, slots
        self.route_peers = route_peers
        self.weight_fp32 = weight_dtype == torch.float32
        self.weight_before_norm = constants.weight_before_norm
        self.eps = constants.eps
        self.gain_center = constants.gain_center
        self.enable_pdl = enable_pdl
        self.timeout_ns = timeout_ns
        self.band_threads = band_threads
        self.threads = TOP_K * band_threads
        self.group_warps = band_threads // WARP_SIZE
        self.band = band_threads * VEC_BF16
        # Route mailbox [3, tp, slots * TOP_K, P]; token mailbox [3, tp, slots, P].
        self.route_rows = slots * TOP_K
        self.route_rank_stride = self.route_rows * POOLED
        self.fine = TP_MAX_SPLIT // split
        self.smem_bytes = 4 * (
            TOP_K * self.group_warps + TOP_K + split * TOP_K + TOP_K * self.band
        )

    @cute.jit
    def __call__(
        self,
        rows: cute.Tensor,
        route_weights: cute.Tensor,
        permuted_idx: cute.Tensor,
        norm_weight: cute.Tensor,
        out: cute.Tensor,
        generations: cute.Tensor,
        route_mailbox: cute.Tensor,
        token_mailbox: cute.Tensor,
        token_multicast: Int64,
        m: Int32,
        num_rows: Int32,
        stream: cuda.CUstream,
    ) -> None:
        self.kernel(
            rows,
            route_weights,
            permuted_idx,
            norm_weight,
            out,
            generations,
            route_mailbox,
            token_mailbox,
            token_multicast,
            m,
            num_rows,
        ).launch(
            grid=(self.split, (m + self.tp - 1) // self.tp, self.tp),
            block=(self.threads, 1, 1),
            cluster=(self.split, 1, 1),
            smem=self.smem_bytes + 64,
            stream=stream,
            use_pdl=self.enable_pdl,
        )

    # Straight-line helpers: plain Python, they emit into the kernel's trace.
    def gains(self, words):
        """The norm weight's 8 columns plus the gain center, FP32."""
        gains = []
        for q in range(4):
            low, high = bf16x2_to_f32x2(words[q])
            gains.append(low + self.gain_center)
            gains.append(high + self.gain_center)
        return gains

    def warp_sum_of_squares(self, packed, gains):
        """The warp's sum of squares of one BF16 route row (scaled by the
        gains first with weight_before_norm)."""
        total = Float32(0.0)
        for q in range(4):
            low, high = bf16x2_to_f32x2(packed[q])
            if self.weight_before_norm:
                low = low * gains[2 * q]
                high = high * gains[2 * q + 1]
            total = total + low * low + high * high
        for shift in range(5):
            total = total + cute.arch.shuffle_sync_bfly(total, offset=1 << shift)
        return total

    def normalized(self, packed, inv_rms: Float32, gains):
        """One BF16 route row RMS-normalized and scaled by the gains, FP32."""
        values = []
        for q in range(4):
            low, high = bf16x2_to_f32x2(packed[q])
            if self.weight_before_norm:
                low = low * gains[2 * q]
                high = high * gains[2 * q + 1]
            low = low * inv_rms
            high = high * inv_rms
            if not self.weight_before_norm:
                low = low * gains[2 * q]
                high = high * gains[2 * q + 1]
            values.append(low)
            values.append(high)
        return values

    @cute.kernel
    def kernel(
        self,
        rows: cute.Tensor,
        route_weights: cute.Tensor,
        permuted_idx: cute.Tensor,
        norm_weight: cute.Tensor,
        out: cute.Tensor,
        generations: cute.Tensor,
        route_mailbox: cute.Tensor,
        token_mailbox: cute.Tensor,
        token_multicast: Int64,
        m: Int32,
        num_rows: Int32,
    ) -> None:
        tidx, _, _ = cute.arch.thread_idx()
        band, slot, d = cute.arch.block_idx()
        route_k = tidx // self.band_threads
        group_lane = tidx % self.band_threads
        group_warp = group_lane // WARP_SIZE
        lane = cute.arch.lane_idx()
        column = Int64(band * self.band + group_lane * VEC_BF16)
        token = slot * self.tp + d
        own_token = slot * self.tp + self.rank
        owner = d == Int32(0) and own_token < m

        smem = cutlass.utils.SmemAllocator()
        sums = smem.allocate_array(Float32, TOP_K * self.group_warps)
        inv_rms = smem.allocate_array(Float32, TOP_K)
        exchange = smem.allocate_array(Float32, self.split * TOP_K)
        products = smem.allocate_array(Float32, TOP_K * self.band)

        if owner:
            # Every peer must have started before the sums of squares store
            # into its shared memory; the matching wait precedes them.
            cute.arch.cluster_arrive_relaxed()

        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_wait()

        word = Int64((slot * self.tp + d) * TP_MAX_SPLIT + band * self.fine)
        generation = Int64(generations[word])

        # Send: this rank's partial of route k of token t, to its owner d.
        if token < m:
            push_route = Int64(token) * TOP_K + route_k
            push_row = Int32(permuted_idx[push_route])
            push_local = Int32(0)
            if push_row >= Int32(0) and push_row < num_rows:
                push_local = Int32(1)
            push_source = cute.make_ptr(
                BFloat16,
                (rows.iterator + (Int64(push_row) * POOLED + column)).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            partial = sanitize_negative_zero_u32x4(
                load_global_u32x4(push_source, push_local)
            )
            element = (
                (generation * self.tp + self.rank) * self.route_rows
                + Int64(slot * TOP_K + route_k)
            ) * POOLED + column
            # The words leave the vector here, outside the per-peer branches.
            words = [Uint32(partial[q]) for q in range(4)]
            # The owner reads its own rows from its GEMM2 output: tp - 1 peers.
            for other in cutlass.range_constexpr(self.tp - 1):
                dest = other + int(other >= self.rank)
                if d == Int32(dest):
                    store_global_u32x4(
                        Int64(self.route_peers[dest]) + element * 2, words
                    )
        if cutlass.const_expr(self.enable_pdl):
            cute.arch.griddepcontrol_launch_dependents()

        # Sum: the owned token's route k, then the routes' sum.
        if owner:
            gain_words = load_global_u32x4(
                cute.make_ptr(
                    BFloat16,
                    (norm_weight.iterator + column).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                )
            )
            own_route = Int64(own_token) * TOP_K + route_k
            weight = Float32(route_weights[own_route])
            own_row = Int32(permuted_idx[own_route])
            own_local = Int32(0)
            if own_row >= Int32(0) and own_row < num_rows:
                own_local = Int32(1)
            own = load_global_u32x4(
                cute.make_ptr(
                    BFloat16,
                    (rows.iterator + (Int64(own_row) * POOLED + column)).llvm_ptr,
                    cute.AddressSpace.gmem,
                    assumed_align=16,
                ),
                own_local,
            )
            first = (
                generation * self.tp * self.route_rows + Int64(slot * TOP_K + route_k)
            ) * POOLED + column
            base = cute.make_ptr(
                BFloat16,
                (route_mailbox.iterator + first).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            sources = [src for src in range(self.tp) if src != self.rank]
            offsets = tuple(src * self.route_rank_stride for src in sources)
            peers = poll_at(base, offsets, self.timeout_ns)
            for i in cutlass.range_constexpr(len(offsets)):
                rearm(base + Int64(offsets[i]))

            summed = [Float32(0.0) for _ in range(VEC_BF16)]
            for src in cutlass.range_constexpr(self.tp):
                for q in cutlass.range_constexpr(4):
                    if cutlass.const_expr(src == self.rank):
                        low, high = bf16x2_to_f32x2(own[q])
                    else:
                        low, high = bf16x2_to_f32x2(peers[sources.index(src), q])
                    summed[2 * q] = summed[2 * q] + low
                    summed[2 * q + 1] = summed[2 * q + 1] + high
            packed = [
                pack_bf16x2_rn(summed[2 * q], summed[2 * q + 1]) for q in range(4)
            ]

            gains = self.gains(gain_words)
            total = self.warp_sum_of_squares(packed, gains)
            if lane == 0:
                cute.arch.store(
                    (sums + (route_k * self.group_warps + group_warp)).llvm_ptr, total
                )
            cute.arch.barrier()
            # Fold each route's sums: the CTA's warps in order, then the
            # cluster's bands in order, on every band.
            cute.arch.cluster_wait()
            if tidx < TOP_K:
                partial_sum = Float32(0.0)
                for w in cutlass.range_constexpr(self.group_warps):
                    partial_sum = partial_sum + cute.arch.load(
                        (sums + (tidx * self.group_warps + w)).llvm_ptr, Float32
                    )
                mine = exchange + (band * TOP_K + tidx)
                for peer in cutlass.range_constexpr(self.split):
                    store_shared_cluster_f32(
                        map_shared_to_peer(mine, Int32(peer)), partial_sum
                    )
            cute.arch.barrier()
            cute.arch.cluster_arrive()
            cute.arch.cluster_wait()
            if tidx < TOP_K:
                total_sum = Float32(0.0)
                for peer in cutlass.range_constexpr(self.split):
                    total_sum = total_sum + cute.arch.load(
                        (exchange + (peer * TOP_K + tidx)).llvm_ptr, Float32
                    )
                cute.arch.store(
                    (inv_rms + tidx).llvm_ptr,
                    cute.rsqrt(total_sum * (1.0 / POOLED) + self.eps),
                )
            cute.arch.barrier()

            scale = cute.arch.load((inv_rms + route_k).llvm_ptr, Float32)
            values = self.normalized(packed, scale, gains)
            slot_base = route_k * self.band + group_lane * VEC_BF16
            # Each product rounded alone (mul_rn): no FMA with the route sum.
            for i in cutlass.range_constexpr(VEC_BF16):
                cute.arch.store(
                    (products + (slot_base + i)).llvm_ptr, mul_rn(values[i], weight)
                )
            cute.arch.barrier()
            if route_k == 0:
                acc = [Float32(0.0) for _ in range(VEC_BF16)]
                for k in cutlass.range_constexpr(TOP_K):
                    first_product = k * self.band + group_lane * VEC_BF16
                    for i in cutlass.range_constexpr(VEC_BF16):
                        acc[i] = acc[i] + cute.arch.load(
                            (products + (first_product + i)).llvm_ptr, Float32
                        )
                rounded = cute.make_rmem_tensor(cute.make_layout((4,)), Uint32)
                for q in cutlass.range_constexpr(4):
                    rounded[q] = pack_bf16x2_rn(acc[2 * q], acc[2 * q + 1])
                result = sanitize_negative_zero_u32x4(rounded.load())
                store_global_u32x4(
                    Int64(
                        (out.iterator + (Int64(own_token) * POOLED + column)).toint()
                    ),
                    result,
                )
                published = (
                    (generation * self.tp + self.rank) * self.slots + Int64(slot)
                ) * POOLED + column
                stmc_bf16x8(token_multicast + published * 2, result)

        # Copy: token t's row from its owner d.
        if token < m and d != Int32(self.rank) and route_k == 0:
            token_source = cute.make_ptr(
                BFloat16,
                (
                    token_mailbox.iterator
                    + ((generation * self.tp + Int64(d)) * self.slots + Int64(slot))
                    * POOLED
                    + column
                ).llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            )
            copied = poll_at(token_source, (0,), self.timeout_ns)
            store_global_u32x4(
                Int64((out.iterator + (Int64(token) * POOLED + column)).toint()),
                copied[0, None].load(),
            )
            rearm(token_source)

        cute.arch.barrier()
        if tidx == 0:
            following = Int32((generation + Int64(1)) % Int64(LAMPORT_GENERATIONS))
            for i in cutlass.range_constexpr(self.fine):
                generations[word + i] = following


class TailConstants(NamedTuple):
    """Compile-time constants of one layer's post-expert norm (bound at build)."""

    eps: float
    gain_center: float
    weight_before_norm: bool


def tail_constants(norm) -> TailConstants:
    """The constants the stock post-expert finalize passes for ``norm`` (a
    MuseSpark1xRMSNorm over the pooled width)."""
    if norm.hidden_size != POOLED:
        raise ValueError(f"the post-expert norm must span {POOLED} columns")
    if norm.variance_size_override is not None:
        raise ValueError("the post-expert norm must normalize its full width")
    return TailConstants(
        eps=float(norm.variance_epsilon),
        gain_center=float(norm.fused_gain_center),
        weight_before_norm=bool(norm.applies_weight_before_norm),
    )


def tail_weight(norm, device: torch.device) -> torch.Tensor:
    """The norm weight as the stock finalize hands it to the kernel: BF16 [P],
    ones without a weight (the gain center is added in FP32 in the kernel)."""
    if norm.has_weight:
        weight = norm.weight.detach()
        if tuple(weight.shape) != (POOLED,):
            raise ValueError(f"the post-expert norm weight must be [{POOLED}]")
    else:
        weight = torch.ones(POOLED)
    return weight.to(dtype=torch.bfloat16, device=device).contiguous()


def compile_reduce(*, tp: int, capacity: int, enable_pdl: bool):
    """Compile the reduction for a ``[3, tp, capacity, P]`` mailbox."""
    assert_not_capturing("the MoE tail reduction")
    reduce = _TailReduceKernel(
        hidden=POOLED,
        tp=tp,
        capacity_m=capacity,
        cluster_size=REDUCE_CLUSTER,
        threads=THREADS,
        enable_pdl=enable_pdl,
    )
    return cute.compile(
        reduce,
        make_fake_compact_tensor(
            BFloat16, (LAMPORT_GENERATIONS * tp * capacity * POOLED,), assumed_align=16
        ),
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=POOLED),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int32(capacity),
        current_cu_stream(),
    )


def compile_publisher(
    constants: TailConstants,
    weight_dtype: torch.dtype,
    *,
    tp: int,
    rank: int,
    capacity: int,
    enable_pdl: bool,
):
    """Compile the finalize publisher for one norm and route-weight dtype."""
    assert_not_capturing("the MoE tail publisher")
    if weight_dtype not in (torch.bfloat16, torch.float32):
        raise ValueError("route weights must be BF16 or FP32")
    device_kernel = _NormFinalizePublishDeviceKernel(
        hidden=POOLED,
        row_stride=POOLED,
        topk=TOP_K,
        tp=tp,
        rank=rank,
        capacity_m=capacity,
        threads=THREADS,
        weight_before_norm=constants.weight_before_norm,
        eps=constants.eps,
        gain_center=constants.gain_center,
        enable_pdl=enable_pdl,
    )
    weight_type = Float32 if weight_dtype == torch.float32 else BFloat16
    return cute.compile(
        device_kernel,
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=POOLED),
        make_fake_dynamic_compact_tensor(
            weight_type, alignment=weight_type.width // 8, divisibility=TOP_K
        ),
        make_fake_dynamic_compact_tensor(Int32, alignment=4, divisibility=TOP_K),
        make_fake_compact_tensor(BFloat16, (POOLED,), assumed_align=16),
        make_fake_compact_tensor(Int32, (2,), assumed_align=4),
        Int64(0),
        Int32(capacity),
        Int32(1),
        current_cu_stream(),
    )


ROUTE_WEIGHT_DTYPES = (torch.bfloat16, torch.float32)


def _check_tail_args(
    capacity: int, unfinalized: "UnfinalizedMoEOutput", out: torch.Tensor
) -> None:
    """Refuse a call either tail cannot take (before anything launches: the
    other ranks would wait for this one). ``out`` must be dense: the kernels
    write through its flat view."""
    m = out.shape[0]
    rows = unfinalized.gemm2_permuted
    route_weights = unfinalized.expert_weights
    permuted_idx = unfinalized.expanded_idx_to_permuted_idx
    if (
        m > capacity
        or tuple(out.shape) != (m, POOLED)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or rows.dim() != 2
        or rows.shape[1] != POOLED
        or rows.dtype != torch.bfloat16
        or tuple(route_weights.shape) != (m, TOP_K)
        or tuple(permuted_idx.shape) != (m, TOP_K)
        or permuted_idx.dtype != torch.int32
    ):
        raise ValueError("unsupported MoE tail arguments")
    if route_weights.dtype not in ROUTE_WEIGHT_DTYPES:
        raise ValueError("route weights must be BF16 or FP32")


def _route_views(unfinalized: "UnfinalizedMoEOutput") -> tuple:
    """The routed rows, the route weights and the route-to-row index as the
    tails' kernels take them."""
    route_weights = unfinalized.expert_weights
    return (
        to_cute_dynamic(unfinalized.gemm2_permuted.flatten(), 16, divisibility=POOLED),
        to_cute_dynamic(
            route_weights.flatten(), route_weights.element_size(), divisibility=TOP_K
        ),
        to_cute_dynamic(
            unfinalized.expanded_idx_to_permuted_idx.flatten(), 4, divisibility=TOP_K
        ),
    )


class MoETail:
    """The MoE tail of one executor: its Lamport mailbox and compiled kernels.

    Built once, collectively over the TP group, before CUDA graph capture. Every
    MoE layer binds its post-expert norm with :meth:`bind`; the layers share the
    mailbox (their calls are sequential on every rank, as the stock workspace's
    are). ``tp`` >= 2 (multicast); ``capacity`` rows per call at most.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        tp: int,
        rank: int,
        device: torch.device,
        capacity: int = MAILBOX_CAPACITY,
        enable_pdl: bool = True,
    ) -> None:
        assert_not_capturing("MoETail")
        if tp < 2:
            raise ValueError("the MoE tail needs at least two ranks")
        self.tp, self.rank, self.capacity = tp, rank, capacity
        self.device = device
        self.enable_pdl = enable_pdl
        self.mailbox = LamportMailbox(
            group, tp=tp, rank=rank, capacity=capacity, width=POOLED, device=device
        )
        self._multicast = Int64(self.mailbox.multicast)
        self._reduce = compile_reduce(tp=tp, capacity=capacity, enable_pdl=enable_pdl)
        self._publishers: dict[tuple, Any] = {}

    def select(self, constants: TailConstants, weight_dtype: torch.dtype, m: int):
        """The finalize publisher for one norm and route-weight dtype (cached;
        every row count takes the same one)."""
        key = (constants, weight_dtype)
        publisher = self._publishers.get(key)
        if publisher is None:
            publisher = compile_publisher(
                constants,
                weight_dtype,
                tp=self.tp,
                rank=self.rank,
                capacity=self.capacity,
                enable_pdl=self.enable_pdl,
            )
            self._publishers[key] = publisher
        return publisher

    def bind(self, norm) -> "BoundTail":
        """One layer's tail: ``norm`` is its post_expert_norm. Both route-weight
        dtypes are compiled here, before capture."""
        constants = tail_constants(norm)
        for dtype in ROUTE_WEIGHT_DTYPES:
            self.select(constants, dtype, self.capacity)
        return BoundTail(self, constants, tail_weight(norm, self.device))

    def launch(
        self,
        publisher,
        weight_view: cute.Tensor,
        unfinalized: "UnfinalizedMoEOutput",
        out: torch.Tensor,
    ) -> None:
        m = out.shape[0]
        stream = current_cu_stream()
        publisher(
            *_route_views(unfinalized),
            weight_view,
            self.mailbox.stage_view,
            self._multicast,
            Int32(m),
            Int32(unfinalized.gemm2_permuted.shape[0]),
            stream,
        )
        self._reduce(
            self.mailbox.view,
            to_cute_dynamic(out.flatten(), 16, divisibility=POOLED),
            self.mailbox.stage_view,
            Int32(m),
            stream,
        )


class BoundTail:
    """One MoE layer's tail on ``tail`` (a ``MoETail``, or a ``TpTail`` for
    the expert-TP layers); ``run`` is graph-capturable and allocates nothing."""

    def __init__(self, tail: Any, constants: TailConstants, weight: torch.Tensor):
        self.tail = tail
        self.constants = constants
        self.weight = weight
        self._weight_view = to_cute(weight, 16)

    def run(self, unfinalized: "UnfinalizedMoEOutput", out: torch.Tensor) -> None:
        """Finalize the experts' output into ``out`` (BF16 [b, P],
        contiguous): the MoE output, identical on every rank. Every rank of
        the group calls it with the same b; b = 0 does nothing on every rank."""
        m = out.shape[0]
        if m == 0:
            return
        _check_tail_args(self.tail.capacity, unfinalized, out)
        kernel = self.tail.select(self.constants, unfinalized.expert_weights.dtype, m)
        self.tail.launch(kernel, self._weight_view, unfinalized, out)


def compile_tp_tail(
    constants: TailConstants,
    weight_dtype: torch.dtype,
    *,
    tp: int,
    rank: int,
    split: int,
    slots: int,
    route_peers: tuple[int, ...],
    generations: int,
    enable_pdl: bool,
    timeout_ns: int = SPIN_TIMEOUT_NS,
):
    """Compile the TP tail for one norm, route-weight dtype and band split."""
    assert_not_capturing("the expert-TP MoE tail")
    if weight_dtype not in ROUTE_WEIGHT_DTYPES:
        raise ValueError("route weights must be BF16 or FP32")
    device_kernel = _TpTailKernel(
        tp=tp,
        rank=rank,
        split=split,
        slots=slots,
        route_peers=route_peers,
        constants=constants,
        weight_dtype=weight_dtype,
        enable_pdl=enable_pdl,
        timeout_ns=timeout_ns,
    )
    weight_type = Float32 if weight_dtype == torch.float32 else BFloat16
    return cute.compile(
        device_kernel,
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=POOLED),
        make_fake_dynamic_compact_tensor(
            weight_type, alignment=weight_type.width // 8, divisibility=TOP_K
        ),
        make_fake_dynamic_compact_tensor(Int32, alignment=4, divisibility=TOP_K),
        make_fake_compact_tensor(BFloat16, (POOLED,), assumed_align=16),
        make_fake_dynamic_compact_tensor(BFloat16, alignment=16, divisibility=POOLED),
        make_fake_compact_tensor(Int32, (generations,), assumed_align=4),
        make_fake_compact_tensor(
            BFloat16,
            (LAMPORT_GENERATIONS * tp * slots * TOP_K * POOLED,),
            assumed_align=16,
        ),
        make_fake_compact_tensor(
            BFloat16, (LAMPORT_GENERATIONS * tp * slots * POOLED,), assumed_align=16
        ),
        Int64(0),
        Int32(slots * tp),
        Int32(1),
        current_cu_stream(),
    )


class TpTail:
    """The MoE tail of the expert-TP layers of one executor: the route and
    token mailboxes, the generation words and the compiled kernels.

    Built once, collectively over the TP group, before CUDA graph capture; the
    layers share it (their calls are sequential on every rank). Every rank
    holds a K-partial of every route row; the output is bitwise the stock TP
    finalize's on its token-owner path for the same partials (``tp_split``).
    ``tp`` >= 2 (multicast); ``capacity`` tokens per call at most. ``batches``:
    the token counts a captured call runs (the executor's), whose band splits
    every bind compiles; other counts compile theirs on their first call.
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        *,
        tp: int,
        rank: int,
        device: torch.device,
        capacity: int = MAILBOX_CAPACITY,
        batches: Sequence[int] = (8,),
        enable_pdl: bool = True,
        timeout_ns: int = SPIN_TIMEOUT_NS,
    ) -> None:
        assert_not_capturing("TpTail")
        if tp < 2:
            raise ValueError("the TP tail needs at least two ranks")
        if not 1 <= min(batches) <= max(batches) <= capacity:
            raise ValueError(f"batches must lie in [1, {capacity}]")
        self.tp, self.rank, self.capacity = tp, rank, capacity
        self.device = device
        self.enable_pdl = enable_pdl
        self.timeout_ns = timeout_ns
        self.slots = -(-capacity // tp)
        # Rank d's route mailbox receives every rank's partial rows of the
        # tokens d owns; its token mailbox, every owner's finished rows.
        self.routes = LamportMailbox(
            group,
            tp=tp,
            rank=rank,
            capacity=self.slots * TOP_K,
            width=POOLED,
            device=device,
        )
        self.tokens = LamportMailbox(
            group, tp=tp, rank=rank, capacity=self.slots, width=POOLED, device=device
        )
        self.generations = torch.zeros(
            self.slots * tp * TP_MAX_SPLIT, dtype=torch.int32, device=device
        )
        self._generations_view = to_cute(self.generations, 4)
        self._multicast = Int64(self.tokens.multicast)
        self.nsm = torch.cuda.get_device_properties(device).multi_processor_count
        self.batches = tuple(batches)
        self._kernels: dict[tuple, Any] = {}

    def split(self, m: int) -> int:
        return tp_split(m, self.tp, self.nsm)

    def select(self, constants: TailConstants, weight_dtype: torch.dtype, m: int):
        """The tail for one norm and route-weight dtype at ``m`` tokens (cached
        per band split; compiled on a miss, which must precede capture)."""
        split = self.split(m)
        key = (constants, weight_dtype, split)
        kernel = self._kernels.get(key)
        if kernel is None:
            kernel = compile_tp_tail(
                constants,
                weight_dtype,
                tp=self.tp,
                rank=self.rank,
                split=split,
                slots=self.slots,
                route_peers=self.routes.peers,
                generations=self.generations.numel(),
                enable_pdl=self.enable_pdl,
                timeout_ns=self.timeout_ns,
            )
            self._kernels[key] = kernel
        return kernel

    def bind(self, norm) -> BoundTail:
        """One layer's tail: ``norm`` is its post_expert_norm. The splits of
        ``batches`` are compiled here for both route-weight dtypes, before
        capture."""
        constants = tail_constants(norm)
        for dtype in ROUTE_WEIGHT_DTYPES:
            for m in self.batches:
                self.select(constants, dtype, m)
        return BoundTail(self, constants, tail_weight(norm, self.device))

    def launch(
        self,
        kernel,
        weight_view: cute.Tensor,
        unfinalized: "UnfinalizedMoEOutput",
        out: torch.Tensor,
    ) -> None:
        kernel(
            *_route_views(unfinalized),
            weight_view,
            to_cute_dynamic(out.flatten(), 16, divisibility=POOLED),
            self._generations_view,
            self.routes.view,
            self.tokens.view,
            self._multicast,
            Int32(out.shape[0]),
            Int32(unfinalized.gemm2_permuted.shape[0]),
            current_cu_stream(),
        )
