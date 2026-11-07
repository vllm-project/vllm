# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent KdaWide: each CTA walks a contiguous range of (request, head) items and
streams the next item's state page and operands into shared memory while it runs
the current item's recurrence.

KdaWide (one CTA per item) spends most of a large step outside the recurrence
(TP1 R128 q5: ~360 of 557 us): every CTA waits for its 64 KB state page, its slots
and its operands (dependent global round trips) before it can start, and all CTAs
of a wave wait together. Here a CTA loads item n + 1 (cp.async: the state page into
thread-private slots, the convolution window, the step's x / gate / F_A rows; the
slots and beta into registers) right after item n's state has moved into
registers, so those loads overlap item n's recurrence and epilogue.

Per item the math is KdaWide's, instruction for instruction (same operands, cache
semantics, BF16 roundings and FP32 reduction order): the convolution taps, inputs,
output gates and F_A columns are only read from shared memory instead of global."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.primitives import (
    alloc,
    at,
    copy_g2s,
    fma,
    fma2,
    launch,
    load_words,
    mma_into,
    mul2,
    split,
    store_f32x4,
    warp_sum,
    zeros,
)
from .kernel import mul_rn, reciprocal, record_row, sqrt_approx, triton_sigmoid
from .wide import (
    ACCUMULATORS,
    CTAS_PER_SM,
    KCHUNKS,
    LANES,
    ROWS,
    THREADS,
    KdaWide,
    load_shared_f32x4,
    load_weight_bf16x4,
)

GRANULE = 16  # bytes per cp.async


def load_shared_bf16x4(pointer):
    return load_words(pointer, count=2, shared=True, mutable=True, word_type="u32")


class KdaStream(KdaWide):
    """KdaWide's operands and semantics; ``ctas`` persistent CTAs (two per SM),
    ``conv_dim_first`` the convolution cache layout ([channel, window] when true,
    else [window, channel] per slot) and ``window`` its length."""

    def __init__(
        self,
        heads,
        tokens,
        record,
        gate_mode,
        has_bias,
        conv_dim_first,
        window,
        ctas,
    ):
        super().__init__(heads, tokens, record, gate_mode, has_bias, 0)
        self.conv_dim_first, self.window, self.ctas = conv_dim_first, window, ctas

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=CTAS_PER_SM)

    @cute.jit
    def item_slots(self, args, item):
        """(head, request, state slot, conv slot, window offset) of an item."""
        slots = args[8]
        head, request = item % self.heads, item // self.heads
        if cutlass.const_expr(self.tokens == 1):
            slot = slots[request]
            conv_slot, offset = slot, Int32(0)
        else:
            offset = args[17][request] - 1
            slot, conv_slot = slots[request, offset], slots[request, 0]
        return head, request, Int32(slot), Int32(conv_slot), Int32(offset)

    @cute.jit
    def stage_state(self, args, staged, head, slot):
        """cp.async the item's [128, 128] FP32 state page into the thread-private
        slots [chunk, thread] (KdaWide's layout)."""
        tid, _, _ = cute.arch.thread_idx()
        base_row, lane_in = tid // LANES, tid % LANES
        page = args[9][Int64(slot), None, None, None]
        for r in cutlass.range_constexpr(ROWS):
            for c in cutlass.range_constexpr(KCHUNKS):
                source = at(page, head, base_row + 64 * r, 4 * (lane_in + LANES * c))
                copy_g2s(staged[r * KCHUNKS + c, tid, None].iterator, source)

    @cute.jit
    def stage_conv(self, args, convs, head, conv_slot, offset):
        """cp.async the item's convolution window: [part, channel x window] when
        the cache is dim-first, else the three tap rows [part, tap, channel]."""
        tid, _, _ = cute.arch.thread_idx()
        conv_state = args[3]
        dim: cutlass.Constexpr = self.heads * 128
        size: cutlass.Constexpr = conv_state.element_type.width // 8
        per: cutlass.Constexpr = GRANULE // size  # elements per granule
        page = conv_state[Int64(conv_slot), None, None]
        if cutlass.const_expr(self.conv_dim_first):
            part_granules: cutlass.Constexpr = 128 * self.window // per
            total: cutlass.Constexpr = 3 * part_granules
            for j in cutlass.range_constexpr((total + THREADS - 1) // THREADS):
                g = tid + THREADS * j
                if g < total:
                    part, k = g // part_granules, g % part_granules
                    source = at(page, part * dim + head * 128, 0) + per * k
                    copy_g2s(at(convs, part, per * k), source)
        else:
            row_granules: cutlass.Constexpr = 128 // per
            total: cutlass.Constexpr = 9 * row_granules  # type: ignore[no-redef]
            for j in cutlass.range_constexpr((total + THREADS - 1) // THREADS):
                g = tid + THREADS * j
                if g < total:
                    row, k = g // row_granules, g % row_granules
                    part, tap = row // 3, row % 3
                    channel = part * dim + head * 128 + per * k
                    source = at(page, channel, offset + tap)
                    copy_g2s(at(convs, part, tap * 128 + per * k), source)

    @cute.jit
    def stage_inputs(self, args, xs, gs, fas, head, request):
        """cp.async the item's projection rows: x (q, k, v), output gate, F_A."""
        tid, _, _ = cute.arch.thread_idx()
        x, f_a, gate = args[0], args[4], args[11]
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * 128
        token = request * T
        total: cutlass.Constexpr = 5 * 16 * T  # x: 3 parts, gate, F_A
        for j in cutlass.range_constexpr((total + THREADS - 1) // THREADS):
            g = tid + THREADS * j
            if g < total:
                row, k = g // 16, g % 16
                t, kind = row // 5, row % 5
                if kind < 3:
                    source = at(x, token + t, kind * dim + head * 128 + 8 * k)
                    copy_g2s(at(xs, t, kind, 8 * k), source)
                elif kind == 3:
                    source = at(gate, token + t, head * 128 + 8 * k)
                    copy_g2s(at(gs, t, 8 * k), source)
                else:
                    copy_g2s(at(fas, t, 8 * k), at(f_a, token + t, 8 * k))

    @cute.kernel
    def kernel(self, args):
        x, weights, bias, conv_state, f_a, beta, a_log, dt_bias, slots = args[:9]
        state, output, gate, norm_weight, lower_bound, scale, eps, f_b = args[9:17]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        dim: cutlass.Constexpr = H * 128
        requests: cutlass.Constexpr = output.shape[0] // T
        first, count = split(H * requests, cta, self.ctas)

        smem = cutlass.utils.SmemAllocator()
        staged = alloc(smem, Float32, (ROWS * KCHUNKS, THREADS, 4), align=128)
        conv_type: cutlass.Constexpr = conv_state.element_type
        if cutlass.const_expr(self.conv_dim_first):
            convs = alloc(smem, conv_type, (3, 128 * self.window), align=16)
        else:
            convs = alloc(smem, conv_type, (3, 3 * 128), align=16)
        xs = alloc(smem, BFloat16, (T, 3, 128), align=16)
        gs = alloc(smem, BFloat16, (T, 128), align=16)
        fas = alloc(smem, BFloat16, (T, 128), align=16)
        shared_q, shared_k, shared_decay, shared_v, shared_o = (
            alloc(smem, Float32, (T, 128)) for _ in range(5)
        )
        shared_beta = alloc(smem, Float32, 8)
        sums = alloc(smem, Float32, (T, 2, 4))  # q, k squares per warp
        out_sums = alloc(smem, Float32, (T, 4))

        channel = tid % 128
        section = tid // 128  # 0: q and k, 1: v
        base_row, lane_in = tid // LANES, tid % LANES
        group, quarter = lane >> 2, lane & 3
        nw = norm_weight[channel]

        # Prologue: the first item's state and window (predecessor-independent),
        # then its projection rows.
        head, request, slot, conv_slot, offset = self.item_slots(args, first)
        beta_value = Float32(0)
        if count > 0:  # noqa: SIM102
            if slot > 0:
                self.stage_state(args, staged, head, slot)
                self.stage_conv(args, convs, head, conv_slot, offset)
        cute.arch.griddepcontrol_wait()
        if count > 0:  # noqa: SIM102
            if slot > 0:
                self.stage_inputs(args, xs, gs, fas, head, request)
                if (section == 1) & (channel < T):
                    beta_value = Float32(beta[request * T + channel, head])
        cute.arch.cp_async_commit_group()
        cute.arch.griddepcontrol_launch_dependents()

        for i in cutlass.range(count, unroll=1):
            head, request = (first + i) % H, (first + i) // H
            token = request * T
            # Item n + 1's slots (registers), issued now so they land during item n.
            has_next = i + 1 < count
            following = first + i + 1
            if following >= H * requests:  # read a valid item's slots
                following = first + i
            next_head, next_request, next_slot, next_conv, next_offset = (
                self.item_slots(args, following)
            )
            cute.arch.cp_async_wait_group(0)
            cute.arch.barrier()
            if slot <= 0:
                if tid < 128:
                    for t in cutlass.range_constexpr(T):
                        output[token + t, head * 128 + tid] = BFloat16(0)
                if has_next:  # noqa: SIM102
                    if next_slot > 0:
                        self.stage_state(args, staged, next_head, next_slot)
                        self.stage_conv(args, convs, next_head, next_conv, next_offset)
                        self.stage_inputs(args, xs, gs, fas, next_head, next_request)
                        if (section == 1) & (channel < T):
                            beta_value = Float32(
                                beta[next_request * T + channel, next_head]
                            )
                cute.arch.cp_async_commit_group()
            else:
                # Per-head parameters (L2-resident weights).
                taps = zeros((2, 3), conv_type)
                coeff = zeros((2, 4), Float32)
                accum = zeros((2, T), Float32)
                for s in cutlass.range_constexpr(2):
                    part = 2 * section + s  # 0 q, 1 k, 2 v
                    if (section == 0) | (s == 0):
                        for t in cutlass.range_constexpr(4):
                            coeff[s, t] = Float32(
                                weights[part, t, head * 128 + channel]
                            )
                        if cutlass.const_expr(self.has_bias):
                            for t in cutlass.range_constexpr(T):
                                accum[s, t] = Float32(
                                    bias[part * dim + head * 128 + channel]
                                )
                a = a_log[head]
                dt = [
                    dt_bias[head * 128 + 16 * warp + lane // 4 + 8 * e]
                    for e in range(2)
                ]
                wr = cute.make_rmem_tensor((8, 4), Uint32)
                for chunk in cutlass.range_constexpr(8):
                    feature = head * 128 + 16 * warp + group
                    col = 16 * chunk + 4 * quarter
                    low = load_weight_bf16x4(at(f_b, feature, col))
                    high = load_weight_bf16x4(at(f_b, feature + 8, col))
                    wr[chunk, 0], wr[chunk, 2] = low
                    wr[chunk, 1], wr[chunk, 3] = high

                # Staged operands.
                for s in cutlass.range_constexpr(2):
                    part = 2 * section + s
                    if (section == 0) | (s == 0):
                        for t in cutlass.range_constexpr(3):
                            if cutlass.const_expr(self.conv_dim_first):
                                taps[s, t] = convs[
                                    part, channel * self.window + offset + t
                                ]
                            else:
                                taps[s, t] = convs[part, t * 128 + channel]
                inputs = cute.make_rmem_tensor((2, T), BFloat16)
                gates = []  # the output gate of channel c (threads 0-127)
                for t in cutlass.range_constexpr(T):
                    inputs[0, t] = xs[t, 2 * section, channel]
                    inputs[1, t] = xs[t, 1, channel]  # k (section 0)
                    gates.append(gs[t, channel])
                if (section == 1) & (channel < T):
                    shared_beta[channel] = triton_sigmoid(beta_value)
                activations = cute.make_rmem_tensor((8, 2), Uint32)
                for chunk in cutlass.range_constexpr(8):  # column n = group: token n
                    column = at(fas, group % T, 16 * chunk + 4 * quarter)
                    activations[chunk, 0], activations[chunk, 1] = load_shared_bf16x4(
                        column
                    )

                # The convolution (stock tap order) and SiLU, BF16-rounded.
                for s in cutlass.range_constexpr(2):
                    if (section == 0) | (s == 0):
                        for t in cutlass.range_constexpr(T):
                            for e in cutlass.range_constexpr(4):
                                w = t + e
                                term = taps[s, w] if w < 3 else inputs[s, w - 3]
                                accum[s, t] = fma(
                                    Float32(term), coeff[s, e], accum[s, t]
                                )
                for t in cutlass.range_constexpr(T):
                    first_value = self.silu(accum[0, t])
                    if section == 0:
                        second = self.silu(accum[1, t])
                        squares_q, squares_k = (
                            warp_sum(first_value * first_value),
                            warp_sum(second * second),
                        )
                        if lane == 0:
                            sums[t, 0, warp] = squares_q
                            sums[t, 1, warp] = squares_k
                        shared_q[t, channel] = first_value
                        shared_k[t, channel] = second
                        if cutlass.const_expr(self.record):
                            args[19][record_row(args, request), head, t, channel] = (
                                BFloat16(second)
                            )
                    else:
                        shared_v[t, channel] = first_value

                # Gate projection: D[feature, token] over 8 K chunks of 16.
                chains: cutlass.Constexpr = 4 if self.gate_mode == 1 else 1
                acc = zeros((chains, 4), Float32)
                for chunk in cutlass.range_constexpr(8):
                    chain: cutlass.Constexpr = chunk // 2 if self.gate_mode == 1 else 0
                    wa = tuple(wr[chunk, e] for e in range(4))
                    ab = (activations[chunk, 0], activations[chunk, 1])
                    mma_into(acc[chain, None], wa, ab, bf16=True)
                for e in cutlass.range_constexpr(4):
                    t = 2 * quarter + e % 2
                    feature = 16 * warp + group + 8 * (e // 2)
                    value = acc[0, e]
                    if cutlass.const_expr(self.gate_mode == 1):
                        value = Float32(0) + (
                            ((value + acc[1, e]) + acc[2, e]) + acc[3, e]
                        )
                    raw_g = Float32(BFloat16(value))
                    if t < T:
                        shared_decay[t, feature] = self.decay(
                            a, raw_g, dt[e // 2], lower_bound
                        )
                        if cutlass.const_expr(self.record):
                            args[19][
                                record_row(args, request), head, t, 128 + feature
                            ] = BFloat16(raw_g)
                cute.arch.barrier()

                # Normalized q (with the 1/sqrt(K) scale) and k.
                if section == 0:
                    for t in cutlass.range_constexpr(T):
                        qn, kn = (
                            self.quad_sum(sums[t, e, None]) + 1.0e-6 for e in (0, 1)
                        )
                        fq = reciprocal(sqrt_approx(qn))
                        fk = reciprocal(sqrt_approx(kn))
                        q_n = mul_rn(mul_rn(shared_q[t, channel], fq), scale)
                        shared_q[t, channel] = q_n
                        shared_k[t, channel] = mul_rn(shared_k[t, channel], fk)

                history = cute.make_rmem_tensor((ROWS, KCHUNKS, 4), Float32)
                for r in cutlass.range_constexpr(ROWS):
                    for c in cutlass.range_constexpr(KCHUNKS):
                        values = load_shared_f32x4(
                            staged[r * KCHUNKS + c, tid, None].iterator
                        )
                        for e in cutlass.range_constexpr(4):
                            history[r, c, e] = values[e]
                cute.arch.barrier()  # the stage is free: item n + 1 streams in

                if has_next:  # noqa: SIM102
                    if next_slot > 0:
                        self.stage_state(args, staged, next_head, next_slot)
                        self.stage_conv(args, convs, next_head, next_conv, next_offset)
                        self.stage_inputs(args, xs, gs, fas, next_head, next_request)
                        if (section == 1) & (channel < T):
                            beta_value = Float32(
                                beta[next_request * T + channel, next_head]
                            )
                cute.arch.cp_async_commit_group()

                # The recurrence (KdaWide's).
                for t in cutlass.range_constexpr(T):

                    def chunk_of(buffer, c, t=t):
                        return load_shared_f32x4(
                            at(buffer, t, 4 * (lane_in + LANES * c))
                        )

                    dots = zeros((ROWS, ACCUMULATORS, 2), Float32)
                    for c in cutlass.range_constexpr(KCHUNKS):
                        d4, k4 = chunk_of(shared_decay, c), chunk_of(shared_k, c)
                        for r in cutlass.range_constexpr(ROWS):
                            for p in cutlass.range_constexpr(2):
                                j = (2 * c + p) % ACCUMULATORS
                                pair = (history[r, c, 2 * p], history[r, c, 2 * p + 1])
                                pair = mul2(pair, (d4[2 * p], d4[2 * p + 1]))
                                history[r, c, 2 * p], history[r, c, 2 * p + 1] = pair
                                acc2 = (dots[r, j, 0], dots[r, j, 1])
                                acc2 = fma2(pair, (k4[2 * p], k4[2 * p + 1]), acc2)
                                dots[r, j, 0], dots[r, j, 1] = acc2
                    v_new = []
                    for r in cutlass.range_constexpr(ROWS):
                        dot = self.lane_sum(self.tree_sum(dots[r, None, None]))
                        value_row = base_row + 64 * r
                        v = (shared_v[t, value_row] - dot) * shared_beta[t]
                        v_new.append(v)
                        if cutlass.const_expr(self.record):  # noqa: SIM102
                            if lane_in == 0:
                                args[18][
                                    record_row(args, request), head, t, value_row
                                ] = v
                    outs = zeros((ROWS, ACCUMULATORS, 2), Float32)
                    for c in cutlass.range_constexpr(KCHUNKS):
                        k4, q4 = chunk_of(shared_k, c), chunk_of(shared_q, c)
                        for r in cutlass.range_constexpr(ROWS):
                            for p in cutlass.range_constexpr(2):
                                j = (2 * c + p) % ACCUMULATORS
                                pair = (history[r, c, 2 * p], history[r, c, 2 * p + 1])
                                kp = (k4[2 * p], k4[2 * p + 1])
                                pair = fma2(kp, (v_new[r], v_new[r]), pair)
                                history[r, c, 2 * p], history[r, c, 2 * p + 1] = pair
                                acc2 = (outs[r, j, 0], outs[r, j, 1])
                                acc2 = fma2(pair, (q4[2 * p], q4[2 * p + 1]), acc2)
                                outs[r, j, 0], outs[r, j, 1] = acc2
                    for r in cutlass.range_constexpr(ROWS):
                        o = self.lane_sum(self.tree_sum(outs[r, None, None]))
                        if lane_in == 0:
                            shared_o[t, base_row + 64 * r] = Float32(BFloat16(o))
                    if cutlass.const_expr(not self.record):
                        target = slot
                        if cutlass.const_expr(T > 1):
                            target = slots[request, t]
                        if target > 0:
                            destination = state[Int64(target), None, None, None]
                            for r in cutlass.range_constexpr(ROWS):
                                for c in cutlass.range_constexpr(KCHUNKS):
                                    row_values = [history[r, c, e] for e in range(4)]
                                    key = 4 * (lane_in + LANES * c)
                                    pointer = at(
                                        destination, head, base_row + 64 * r, key
                                    )
                                    store_f32x4(pointer, row_values)
                cute.arch.barrier()

                # Gated RMSNorm (threads 0-127: channel c of each token) and the
                # convolution window [tap 1, tap 2, x_0, ..., x_q-1] of every channel.
                if section == 0:
                    for t in cutlass.range_constexpr(T):
                        o = shared_o[t, channel]
                        total = warp_sum(o * o)
                        if lane == 0:
                            out_sums[t, warp] = total
                conv_page = conv_state[Int64(conv_slot), None, None]
                for s in cutlass.range_constexpr(2):
                    if (section == 0) | (s == 0):
                        conv_channel = (2 * section + s) * dim + head * 128 + channel
                        conv_page[conv_channel, 0] = conv_type(taps[s, 1])
                        conv_page[conv_channel, 1] = conv_type(taps[s, 2])
                        for t in cutlass.range_constexpr(T):
                            conv_page[conv_channel, 2 + t] = conv_type(inputs[s, t])
                cute.arch.barrier()
                if section == 0:
                    for t in cutlass.range_constexpr(T):
                        sumsq = self.quad_sum(out_sums[t, None])
                        variance = sumsq * (1.0 / 128) + eps
                        rstd = reciprocal(sqrt_approx(variance))
                        sigmoid_gate = triton_sigmoid(Float32(gates[t]))
                        result = shared_o[t, channel] * rstd * nw * sigmoid_gate
                        output[token + t, head * 128 + channel] = BFloat16(result)
            slot, conv_slot, offset = next_slot, next_conv, next_offset
