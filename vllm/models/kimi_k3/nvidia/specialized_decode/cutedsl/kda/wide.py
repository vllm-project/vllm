# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Throughput KDA core: one CTA per (request, head) holds the head's whole FP32 state
in registers (8 warps, a quarter of two state rows per thread) through every token of
the step.

KdaSplit splits a head over a cluster to spread a few requests over the GPU; with
hundreds of (request, head) items that split only repeats the convolution, the
gate projection and the cluster exchanges per CTA. Here each item reads its state
once (and, on plain verify steps, writes each token's state once), the gate
projection runs on mma.sync (one 16-feature tile per warp) and the recurrence on
packed FP32 FMAs, so the step runs at the HBM or FP32-pipe bound.

Same operands, cache semantics and precision as `KdaSplit` (see its docstring):
the Triton formulas and instructions of the stock composed decode, BF16-rounded
convolution outputs, raw gates and recurrence outputs; only FP32 reduction order
differs."""

from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int64, Uint32

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
    prefetch_bytes,
    store_f32x4,
    warp_sum,
    zeros,
)
from .kernel import (
    div_full,
    mul_rn,
    reciprocal,
    record_row,
    sqrt_approx,
    triton_exp,
    triton_sigmoid,
)

load_weight_bf16x4 = partial(load_words, count=2, cache_policy="nc", word_type="u32")
load_activation_bf16x4 = partial(load_words, count=2, word_type="u32", mutable=True)
load_shared_f32x4 = partial(
    load_words, count=4, shared=True, mutable=True, result_type=Float32, word_type="f32"
)

THREADS = 256  # 8 warps; thread 2 i + h owns keys 8 c + 4 h + [0, 4) of state row i
ROWS = 2  # state rows per thread: g and g + 64
LANES = 4  # threads per row pair
KCHUNKS = 32 // LANES  # 4-key chunks a thread owns per row
ACCUMULATORS = 4  # independent FMA chains per dot product
CTAS_PER_SM = 2


class KdaWide:
    """`tokens` q (1: single-token decode with slots [B]; > 1: verify with slots
    [B, >= q] and accepted [B]); `record`: RecoverSSM verify (checkpoint slot,
    records out); `gate_mode` as KdaSplit's F_B reduction order (0: one chain of
    eight MMAs, 1: four chains of two)."""

    def __init__(
        self, heads, tokens=1, record=False, gate_mode=0, has_bias=False, ahead=0
    ):
        if not 1 <= tokens <= 8:
            raise ValueError(f"KdaWide supports 1-8 tokens per request; got {tokens}")
        assert gate_mode in (0, 1) and (tokens > 1 or not record)
        self.heads, self.tokens, self.record = heads, tokens, record
        self.gate_mode, self.has_bias = gate_mode, has_bias
        # CTAs resident at once: each prefetches into L2 the state (and conv rows)
        # of the item that many CTAs later, which runs in the next wave.
        self.ahead = ahead

    @cute.jit
    def __call__(self, *args):
        grid = (self.heads, args[0].shape[0] // self.tokens, 1)
        launch(self.kernel, args, grid, THREADS, min_blocks_per_mp=CTAS_PER_SM)

    @cute.kernel
    def kernel(self, args):
        x, weights, bias, conv_state, f_a, beta, a_log, dt_bias, slots = args[:9]
        state, output, gate, norm_weight, lower_bound, scale, eps, f_b = args[9:17]
        T: cutlass.Constexpr = self.tokens
        tid, _, _ = cute.arch.thread_idx()
        head, request, _ = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        dim: cutlass.Constexpr = self.heads * 128
        token = request * T
        if cutlass.const_expr(T == 1):
            slot = slots[request]
            conv_slot, offset = slot, 0
        else:
            offset = args[17][request] - 1
            slot, conv_slot = slots[request, offset], slots[request, 0]
        if slot <= 0:
            cute.arch.griddepcontrol_wait()
            if tid < 128:
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * 128 + tid] = BFloat16(0)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            smem = cutlass.utils.SmemAllocator()
            shared_q, shared_k, shared_decay, shared_v, shared_o = (
                alloc(smem, Float32, (T, 128)) for _ in range(5)
            )
            shared_beta = alloc(smem, Float32, 8)
            sums = alloc(smem, Float32, (T, 2, 4))  # q, k squares per warp
            out_sums = alloc(smem, Float32, (T, 4))

            if cutlass.const_expr(self.ahead > 0):  # noqa: SIM102
                if tid == 0:
                    self.prefetch_next(args, head, request)

            # State rows first, independent of the predecessor: cp.async into
            # thread-private slots [chunk, thread] (conflict-free reads later), so
            # the registers stay free for the input phase.
            base_row, lane_in = tid // LANES, tid % LANES
            page = state[Int64(slot), None, None, None]
            staged = alloc(smem, Float32, (ROWS * KCHUNKS, THREADS, 4), align=128)
            for r in cutlass.range_constexpr(ROWS):
                for c in cutlass.range_constexpr(KCHUNKS):
                    source = at(
                        page, head, base_row + 64 * r, 4 * (lane_in + LANES * c)
                    )
                    copy_g2s(staged[r * KCHUNKS + c, tid, None].iterator, source)
            cute.arch.cp_async_commit_group()

            # Convolution operands: threads 0-127 the q and k channels, 128-255 v.
            channel = tid % 128
            section = tid // 128  # 0: q and k, 1: v
            conv_page = conv_state[Int64(conv_slot), None, None]
            conv_type: cutlass.Constexpr = conv_state.element_type
            taps = zeros((2, 3), conv_type)
            coeff = zeros((2, 4), Float32)
            accum = zeros((2, T), Float32)
            for s in cutlass.range_constexpr(2):
                part = 2 * section + s  # 0 q, 1 k, 2 v
                if (section == 0) | (s == 0):
                    conv_channel = part * dim + head * 128 + channel
                    for t in cutlass.range_constexpr(3):
                        taps[s, t] = conv_page[conv_channel, offset + t]
                    for t in cutlass.range_constexpr(4):
                        coeff[s, t] = Float32(weights[part, t, head * 128 + channel])
                    if cutlass.const_expr(self.has_bias):
                        for t in cutlass.range_constexpr(T):
                            accum[s, t] = Float32(bias[conv_channel])
            nw = norm_weight[channel]
            a = a_log[head]
            dt = [dt_bias[head * 128 + 16 * warp + lane // 4 + 8 * i] for i in range(2)]

            # Gate projection weights (warp w: features 16 w ...): A fragments.
            group, quarter = lane >> 2, lane & 3
            wr = cute.make_rmem_tensor((8, 4), Uint32)
            for chunk in cutlass.range_constexpr(8):
                feature = head * 128 + 16 * warp + group
                col = 16 * chunk + 4 * quarter
                low = load_weight_bf16x4(at(f_b, feature, col))
                high = load_weight_bf16x4(at(f_b, feature + 8, col))
                wr[chunk, 0], wr[chunk, 2] = low
                wr[chunk, 1], wr[chunk, 3] = high

            cute.arch.griddepcontrol_wait()
            inputs = cute.make_rmem_tensor((2, T), BFloat16)
            gates = []  # the output gate of channel c (threads 0-127)
            for t in cutlass.range_constexpr(T):
                inputs[0, t] = x[token + t, 2 * section * dim + head * 128 + channel]
                inputs[1, t] = x[token + t, dim + head * 128 + channel]  # k (section 0)
                gates.append(gate[token + t, head * 128 + channel])
            if (section == 1) & (channel < T):
                shared_beta[channel] = triton_sigmoid(
                    Float32(beta[token + channel, head])
                )
            activations = cute.make_rmem_tensor((8, 2), Uint32)
            for chunk in cutlass.range_constexpr(8):  # column n = group: token n
                column = at(f_a, token + group % T, 16 * chunk + 4 * quarter)
                activations[chunk, 0], activations[chunk, 1] = load_activation_bf16x4(
                    column
                )
            cute.arch.griddepcontrol_launch_dependents()

            # The convolution (stock tap order) and SiLU, BF16-rounded.
            for s in cutlass.range_constexpr(2):
                if (section == 0) | (s == 0):
                    for t in cutlass.range_constexpr(T):
                        for i in cutlass.range_constexpr(4):
                            w = t + i
                            term = taps[s, w] if w < 3 else inputs[s, w - 3]
                            accum[s, t] = fma(Float32(term), coeff[s, i], accum[s, t])
            for t in cutlass.range_constexpr(T):
                first = self.silu(accum[0, t])
                if section == 0:
                    second = self.silu(accum[1, t])
                    squares_q, squares_k = (
                        warp_sum(first * first),
                        warp_sum(second * second),
                    )
                    if lane == 0:
                        sums[t, 0, warp] = squares_q
                        sums[t, 1, warp] = squares_k
                    shared_q[t, channel] = first
                    shared_k[t, channel] = second
                    if cutlass.const_expr(self.record):
                        args[19][record_row(args, request), head, t, channel] = (
                            BFloat16(second)
                        )
                else:
                    shared_v[t, channel] = first

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
                    value = Float32(0) + (((value + acc[1, e]) + acc[2, e]) + acc[3, e])
                raw_g = Float32(BFloat16(value))
                if t < T:
                    shared_decay[t, feature] = self.decay(
                        a, raw_g, dt[e // 2], lower_bound
                    )
                    if cutlass.const_expr(self.record):
                        args[19][record_row(args, request), head, t, 128 + feature] = (
                            BFloat16(raw_g)
                        )
            cute.arch.barrier()

            # Normalized q (with the 1/sqrt(K) scale) and k.
            if section == 0:
                for t in cutlass.range_constexpr(T):
                    qn, kn = (self.quad_sum(sums[t, i, None]) + 1.0e-6 for i in (0, 1))
                    fq = reciprocal(sqrt_approx(qn))
                    fk = reciprocal(sqrt_approx(kn))
                    q_n = mul_rn(mul_rn(shared_q[t, channel], fq), scale)
                    shared_q[t, channel] = q_n
                    shared_k[t, channel] = mul_rn(shared_k[t, channel], fk)
            cute.arch.barrier()

            # The recurrence: thread (g, l) owns keys 4 (l + LANES c) + [0, 4) of
            # rows g and g + 64; each shared operand load serves both rows.
            cute.arch.cp_async_wait_group(0)
            history = cute.make_rmem_tensor((ROWS, KCHUNKS, 4), Float32)
            for r in cutlass.range_constexpr(ROWS):
                for c in cutlass.range_constexpr(KCHUNKS):
                    values = load_shared_f32x4(
                        staged[r * KCHUNKS + c, tid, None].iterator
                    )
                    for e in cutlass.range_constexpr(4):
                        history[r, c, e] = values[e]
            for t in cutlass.range_constexpr(T):

                def chunk_of(buffer, c, t=t):
                    return load_shared_f32x4(at(buffer, t, 4 * (lane_in + LANES * c)))

                dots = zeros((ROWS, ACCUMULATORS, 2), Float32)
                for c in cutlass.range_constexpr(KCHUNKS):
                    d4, k4 = chunk_of(shared_decay, c), chunk_of(shared_k, c)
                    for r in cutlass.range_constexpr(ROWS):
                        for p in cutlass.range_constexpr(2):
                            j = (2 * c + p) % ACCUMULATORS
                            pair = (history[r, c, 2 * p], history[r, c, 2 * p + 1])
                            pair = mul2(pair, (d4[2 * p], d4[2 * p + 1]))
                            history[r, c, 2 * p], history[r, c, 2 * p + 1] = pair
                            acc = (dots[r, j, 0], dots[r, j, 1])
                            acc = fma2(pair, (k4[2 * p], k4[2 * p + 1]), acc)
                            dots[r, j, 0], dots[r, j, 1] = acc
                v_new = []
                for r in cutlass.range_constexpr(ROWS):
                    dot = self.lane_sum(self.tree_sum(dots[r, None, None]))
                    value_row = base_row + 64 * r
                    v = (shared_v[t, value_row] - dot) * shared_beta[t]
                    v_new.append(v)
                    if cutlass.const_expr(self.record):  # noqa: SIM102
                        if lane_in == 0:
                            args[18][record_row(args, request), head, t, value_row] = v
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
                            acc = (outs[r, j, 0], outs[r, j, 1])
                            acc = fma2(pair, (q4[2 * p], q4[2 * p + 1]), acc)
                            outs[r, j, 0], outs[r, j, 1] = acc
                for r in cutlass.range_constexpr(ROWS):
                    o = self.lane_sum(self.tree_sum(outs[r, None, None]))
                    if lane_in == 0:
                        shared_o[t, base_row + 64 * r] = Float32(BFloat16(o))  # BF16
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
                                pointer = at(destination, head, base_row + 64 * r, key)
                                store_f32x4(pointer, row_values)
            cute.arch.barrier()
            cute.arch.barrier()

            # Gated RMSNorm (threads 0-127: channel c of each token) and the
            # convolution window [tap 1, tap 2, x_0, ..., x_q-1] of every channel.
            if section == 0:
                for t in cutlass.range_constexpr(T):
                    o = shared_o[t, channel]
                    total = warp_sum(o * o)
                    if lane == 0:
                        out_sums[t, warp] = total
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

    @cute.jit
    def prefetch_next(self, args, head, request):
        """L2 prefetch of the state page of item `ahead` CTAs on."""
        slots, state = args[8], args[9]
        _, requests, _ = cute.arch.grid_dim()
        item = head + self.heads * request + self.ahead
        if item < self.heads * requests:
            h, r = item % self.heads, item // self.heads
            slot = slots[r]
            if cutlass.const_expr(self.tokens > 1):
                slot = slots[r, args[17][r] - 1]
            if slot > 0:
                page = state[Int64(slot), None, None, None]
                prefetch_bytes(at(page, h, 0, 0), 128 * 128 * 4)

    @cute.jit
    def lane_sum(self, value):
        """Sum over the LANES threads of a row."""
        for shift in cutlass.range_constexpr(LANES.bit_length() - 2, -1, -1):
            value += cute.arch.shuffle_sync_bfly(value, 1 << shift)
        return value

    @staticmethod
    def tree_sum(partials):
        """Pairwise sum of [ACCUMULATORS, 2] partial dot products."""
        values = [partials[i // 2, i % 2] for i in range(2 * ACCUMULATORS)]
        while len(values) > 1:
            values = [values[i] + values[i + 1] for i in range(0, len(values), 2)]
        return values[0]

    @staticmethod
    def quad_sum(r):
        """The four warps' partial sums, as KdaSplit adds them."""
        return (r[0] + r[2]) + (r[1] + r[3])

    @cute.jit
    def silu(self, value):
        """The convolution's SiLU: x / (1 + exp(-x)), stored BF16."""
        silu = div_full(value, Float32(1) + triton_exp(-value))
        return Float32(BFloat16(silu))

    @cute.jit
    def decay(self, a_log, raw_g, dt, lower_bound):
        """exp(lower_bound * sigmoid(exp(A_log) * (raw_g + dt_bias)))."""
        gate = lower_bound * triton_sigmoid(triton_exp(a_log) * (raw_g + dt))
        return triton_exp(gate)
