# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Cluster-split KDA recurrence; an optional fifth warp projects F_B and multicasts the
decays while the main warps run the convolution and query/key normalization."""

from functools import partial

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32
from cutlass.cute.arch import store_async_dsmem

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    dot,
    float_bits,
    fma,
    launch,
    load_f32x4,
    load_words,
    mma_into,
    push_float,
    sigmoid,
    smem_words,
    store_f32x4,
    unpack_bf16,
    wait,
    warp_sum,
    zeros,
)

load_bf16x4 = partial(load_words, count=2, cache_policy="nc", word_type="u32")
load_weight_bf16x8 = partial(load_words, count=4, cache_policy="nc", word_type="u32")
load_activation_bf16x4 = partial(load_words, count=2, word_type="u32", mutable=True)
load_activation_bf16x8 = partial(load_words, count=4, word_type="u32", mutable=True)
load_staged = partial(
    load_words, count=4, shared=True, mutable=True, result_type=Float32, word_type="f32"
)


@asm((Float32, Int32), pure=True)
def reciprocal_guess(denominator):
    """``sigmoid_reciprocal``'s refined fast path (IEEE mode) and whether it holds."""
    code = "{ .reg .f32 r,error,neg_error; .reg .b32 bits; .reg .pred regular; "
    code += "mov.b32 bits,$2; add.u32 bits,bits,0x01800000; "
    code += "and.b32 bits,bits,0x7f800000; setp.gt.u32 regular,bits,0x01ffffff; "
    code += "rcp.approx.ftz.f32 r,$2; fma.rn.f32 error,$2,r,0fbf800000; "
    code += "neg.f32 neg_error,error; fma.rn.f32 $0,r,neg_error,r; "
    return code + "selp.u32 $1,1,0,regular; }", "=f,=r,f", (denominator,)


@asm(Float32, pure=True)
def reciprocal_rn(value):
    return "div.rn.f32 $0,0f3f800000,$1;", "=f,f", (value,)


@cute.jit
def sigmoids(values):
    """``sigmoid(v, False)`` of every value, bit for bit, with all fast paths ahead
    of the division fallbacks' branches (verify tokens get ILP)."""
    count: cutlass.Constexpr = len(values)
    den = [Float32(1) + cute.math.exp(-Float32(v), fastmath=True) for v in values]
    result, regular = zeros(count, Float32), zeros(count, Int32)
    irregular = Int32(0)
    for i in cutlass.range_constexpr(count):
        result[i], regular[i] = reciprocal_guess(den[i])
        irregular |= regular[i] ^ 1
    if irregular != 0:  # rare: one branch for the whole batch
        for i in cutlass.range_constexpr(count):
            if regular[i] == 0:
                result[i] = reciprocal_rn(den[i])
    return result


def advance(base, n):
    """``base + n``, or ``base`` itself for a static zero ``n`` (keeps q = 1 code)."""
    return base + n if n else base


@asm()
def store_evict_f32x4(pointer, values):
    code = "{ .reg .b64 pol; createpolicy.fractional.L2::evict_first.b64 pol,1.0; "
    code += "st.global.L2::cache_hint.v4.f32 [$0],{$1,$2,$3,$4},pol; }"
    return code, "l,f,f,f,f", (pointer, *values)


class KdaSplit:
    """128 main threads and C value partitions. Conv state [slot, 3 * heads * 128, 3]
    and recurrent state [slot, head, value, key] keep their strides; slots <= 0 are
    padding (never written) and active rows own distinct slots. Weights must be ready
    at launch; each run waits for prior cache restoration and slot publication.

    ``schedule`` = (early, lanes per state row, minimum CTAs per SM, CTAs per head);
    zeros keep the batch defaults. Two CTAs per head give each CTA 64 state rows and
    each gate lane two decays. Early issues every load the predecessor does not
    produce (slot, state and conv rows, conv / F_B / norm weights) before the PDL
    wait, converts operands at first use, and syncs the cluster only before each
    role's first DSMEM send. Safe because those loads read only weights and data a
    graph or non-PDL boundary orders before this launch (states and conv rows of this
    layer's previous call or a fresh prompt's non-PDL fill, the host's slot table),
    never same-step producer output, which is read after the wait; PDL-chained calls
    without such a boundary must not share state slots. Early 2 also bulk-copies
    the CTA's (contiguous) state rows to shared memory and updates four passes at a
    time, which frees the registers that held them across the input phase; early 3
    (two CTAs per head) also stages the gate warp's F_B rows.

    ``tokens`` q > 1 (speculative verify; ``batch`` counts requests, rows b * q + t)
    runs each request's q tokens in order with vLLM's spec cache semantics: slots
    [B, >= q] and accepted [B] (an extra last tensor, read early like the slots); the
    state comes from slot[b, accepted - 1] and token t's state goes to slot[b, t] if
    > 0; conv slot[b, 0] holds [dim, q + 2] with taps at accepted - 1 ..., rewritten to
    [c_accepted, c_accepted + 1, x_0, ..., x_q-1]. Token t's F_B row is MMA column t;
    CTA t % C normalizes token t."""

    def __init__(self, heads, batch, gate_mode, has_bias, schedule=(0, 0, 0), tokens=1):
        # The gate stage sends decays for tokens 2 * quarter + column (at most 8), while
        # the decay barrier expects 512 bytes per token: more tokens would never arrive.
        if not 1 <= tokens <= 8:
            raise ValueError(f"KdaSplit supports 1-8 tokens per request; got {tokens}")
        self.heads, self.gate_mode, self.has_bias = heads, gate_mode, has_bias
        early, lanes, self.occupancy, cluster, evict = (*schedule, 0, 0)[:5]
        self.early, self.staged = bool(early), early >= 2
        self.gate_staged, self.tokens = early == 3, tokens
        assert gate_mode != 2 or not (early or cluster)
        assert tokens == 1 or not (self.staged or gate_mode == 2 or cluster == 16)
        # TP8 (12 heads) uses fast math. Release the o-proj at entry so its weight loads
        # overlap KDA: TP8 B1-8 (the producer waits before reading KDA's output), TP16 B1
        # only: TP16's fused o-proj reads the AttnRes prefix/blocks at launch, which only
        # the B1-2 skinny projection (it triggers after its PDL wait) guarantees complete;
        # the B3-8 split-K projection triggers before its wait.
        self.fast_math = heads == 12
        self.trigger_at_entry = batch == 1 or (heads == 12 and batch <= 8)
        self.cluster = cluster or (8 if batch == 1 else 4)
        self.lanes_per_row = lanes or (8 if batch <= (4 if heads == 6 else 2) else 16)
        self.rows, self.groups = 128 // self.cluster, 128 // self.lanes_per_row
        self.passes = (self.rows + self.groups - 1) // self.groups
        self.wave = min(4, self.passes) if self.staged else self.passes
        self.chunks = 32 // self.lanes_per_row
        assert not self.gate_staged or self.rows >= 64
        self.evict = bool(evict)  # staged state traffic marked L2 evict-first
        assert self.staged or not self.evict

    @cute.jit
    def __call__(self, *args):
        grid = (self.cluster, self.heads, args[0].shape[0] // self.tokens)
        bound = dict(min_blocks_per_mp=self.occupancy) if self.occupancy else {}
        launch(self.kernel, args, grid, 160, self.cluster, **bound)

    @cute.kernel
    def kernel(self, args):
        x, weights, bias, conv_state, raw_gate, beta, a_log, dt_bias, slots = args[:9]
        state, output, gate, norm_weight, lower_bound, scale, eps, f_b_weight = args[
            9:17
        ]
        T: cutlass.Constexpr = self.tokens
        if cutlass.const_expr(not self.early):
            cute.arch.griddepcontrol_wait()
        thread, _, _ = cute.arch.thread_idx()
        partition, head, token = cute.arch.block_idx()
        if cutlass.const_expr(T == 1):
            slot = slots[token]
            conv_slot, offset, writes = slot, 0, [slot]
        else:
            request, token = token, token * T
            offset = args[17][request] - 1
            slot, conv_slot = slots[request, offset], slots[request, 0]
            writes = [slots[request, t] for t in range(T)]
        if slot <= 0:
            cute.arch.griddepcontrol_wait()
            if (partition == 0) & (thread < 128):
                for t in cutlass.range_constexpr(T):
                    output[advance(token, t), head * 128 + thread] = BFloat16(0)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            lane, warp = thread % 32, thread // 32
            group = thread // self.lanes_per_row
            group_lane = thread % self.lanes_per_row
            dim: cutlass.Constexpr = self.heads * 128
            value_base, qk_channel = partition * self.rows, head * 128 + thread
            value_channel = head * 128 + value_base + thread

            smem = cutlass.utils.SmemAllocator()
            shared_q, shared_k, shared_decay, shared_v, shared_o, reductions = (
                alloc(smem, Float32, size * T)
                for size in (128, 128, 128, self.rows, 128, 8)
            )
            barrier, gate_barrier = (smem.allocate(cutlass.Uint64, 8) for _ in range(2))
            owned: cutlass.Constexpr = -(-T // self.cluster)  # tokens a CTA normalizes
            if cutlass.const_expr(owned > 1):  # one output barrier per owned token
                barrier = smem.allocate(8 * owned, 8)
                barrier = cute.recast_ptr(barrier, dtype=cutlass.Uint64)
            if cutlass.const_expr(self.staged):
                staged = alloc(smem, Float32, (self.rows, 128), align=128)
                state_barrier = smem.allocate(cutlass.Uint64, 8)
            gate_rows = None
            if cutlass.const_expr(self.gate_staged):  # [copy, lane] 16-byte slots
                gate_rows = alloc(smem, Uint32, (self.rows // 2, 32, 4), align=128)
            if cutlass.const_expr(self.trigger_at_entry):
                cute.arch.griddepcontrol_launch_dependents()
            if cutlass.const_expr(not self.early):
                self.setup(barrier, gate_barrier, partition, thread)
                cute.arch.cluster_wait()
            if thread >= 128:
                if cutlass.const_expr(self.early):
                    cute.arch.cluster_arrive_relaxed()
                coords = (head, token, value_base, lane)
                self.gate_warp(args, shared_decay, gate_barrier, coords, gate_rows)

            if thread < 128:
                if cutlass.const_expr(self.early):  # thread 0 arms; barrier 1 orders it
                    self.setup(barrier, gate_barrier, partition, thread)
                # 64-bit page bases (slot x page stride passes Int32), 32-bit offsets.
                page = lambda s: state[Int64(s), None, None, None]
                state_page = page(slot)
                conv_page = conv_state[Int64(conv_slot), None, None]
                state_row = lambda r, c, s=state_page: at(s, head, value_base + r, c)
                history = zeros((self.passes, self.chunks, 4), Float32)
                if cutlass.const_expr(self.staged):  # noqa: SIM102
                    if thread == 0:
                        cute.arch.mbarrier_init(state_barrier, 1)
                        cute.arch.mbarrier_init_fence()
                        size = self.rows * 512
                        cute.arch.mbarrier_arrive_and_expect_tx(state_barrier, size)
                        src, policy = state_row(0, 0), dict(evict_first=self.evict)
                        copy_g2s(staged.iterator, src, size, state_barrier, **policy)
                for p in cutlass.range_constexpr(0 if self.staged else self.passes):
                    row = p * self.groups + group
                    if row < self.rows:
                        for j in cutlass.range_constexpr(self.chunks):
                            column = 4 * (group_lane + self.lanes_per_row * j)
                            values = load_f32x4(state_row(row, column))
                            for e in cutlass.range_constexpr(4):
                                history[p, j, e] = values[e]
                taps = zeros((3, 3), BFloat16)
                coeff = zeros((3, 4), weights.element_type if self.early else Float32)
                accum = zeros(3 * T, Float32)
                for s in cutlass.range_constexpr(3):
                    channel = value_channel if s == 2 else qk_channel
                    if (s != 2) | (thread < self.rows):
                        for t in cutlass.range_constexpr(3):
                            taps[s, t] = conv_page[s * dim + channel, offset + t]
                        for t in cutlass.range_constexpr(4):
                            coeff[s, t] = weights[s, t, channel]
                        if cutlass.const_expr(self.has_bias):
                            accum[s] = bias[s * dim + channel]
                            for t in cutlass.range_constexpr(1, T):
                                accum[3 * t + s] = accum[s]
                nw = norm_weight[thread]

                cute.arch.griddepcontrol_wait()
                inputs = cute.make_rmem_tensor(3 * T, BFloat16)
                betas, gates = [], []
                for t in cutlass.range_constexpr(T):
                    row = advance(token, t)
                    inputs[3 * t], inputs[3 * t + 1] = (
                        x[row, qk_channel],
                        x[row, dim + qk_channel],
                    )
                    inputs[3 * t + 2] = BFloat16(0)
                    if cutlass.const_expr(t == T - 1):  # noqa: SIM102 (all tokens)
                        if thread < self.rows:
                            for u in cutlass.range_constexpr(T):
                                channel = 2 * dim + value_channel
                                inputs[3 * u + 2] = x[advance(token, u), channel]
                    if cutlass.const_expr(self.early):  # converted at first use
                        betas.append(beta[row, head])
                        gates.append(gate[row, qk_channel])
                    else:
                        betas.append(Float32(beta[row, head]))
                        gates.append(Float32(gate[row, qk_channel]))
                cute.arch.griddepcontrol_launch_dependents()

                vs, squares, qks = [], [], []
                for t in cutlass.range_constexpr(T):  # window: 3 taps, then the inputs
                    for s in cutlass.range_constexpr(3):
                        for i in cutlass.range_constexpr(4):
                            w = t + i
                            term = taps[s, w] if w < 3 else inputs[3 * (w - 3) + s]
                            term = Float32(term)
                            cf = Float32(coeff[s, i]) if self.early else coeff[s, i]
                            acc = accum[3 * t + s]
                            accum[3 * t + s] = fma(term, cf, acc, ftz=self.fast_math)
                act = lambda i, value: sigmoid(value, self.fast_math)
                if cutlass.const_expr(T > 1 and not self.fast_math):
                    raw = [accum[i] for i in range(3 * T)] + betas + gates
                    act = partial(lambda batch, i, value: batch[i], sigmoids(raw))
                for t in cutlass.range_constexpr(T):
                    q = accum[3 * t] * act(3 * t, accum[3 * t])
                    k = accum[3 * t + 1] * act(3 * t + 1, accum[3 * t + 1])
                    v = accum[3 * t + 2] * act(3 * t + 2, accum[3 * t + 2])
                    betas[t] = act(3 * T + t, Float32(betas[t]))
                    gates[t] = act(4 * T + t, Float32(gates[t]))
                    shared_q[advance(thread, 128 * t)] = q
                    shared_k[advance(thread, 128 * t)] = k
                    vs.append(v)
                    qks += [q, k]
                    if cutlass.const_expr(t == T - 1):  # noqa: SIM102
                        if thread < self.rows:
                            for u in cutlass.range_constexpr(T):
                                shared_v[advance(thread, self.rows * u)] = vs[u]
                    squares += [warp_sum(q * q), warp_sum(k * k)]
                    if cutlass.const_expr(t == T - 1):  # noqa: SIM102
                        if lane == 0:
                            for u in cutlass.range_constexpr(T):
                                reductions[advance(warp, 8 * u)] = squares[2 * u]
                                reductions[advance(warp + 4, 8 * u)] = squares[
                                    2 * u + 1
                                ]
                cute.arch.barrier(barrier_id=1, number_of_threads=128)

                def factors(t):
                    r = lambda i: reductions[8 * t + i]
                    qn = (r(0) + r(2)) + (r(1) + r(3))
                    kn = (r(4) + r(6)) + (r(5) + r(7))
                    fq = cute.math.rsqrt(qn + 1.0e-6, fastmath=True) * scale
                    return fq, cute.math.rsqrt(kn + 1.0e-6, fastmath=True)

                if cutlass.const_expr(T > 1):  # normalize each token's q, k once
                    for t in cutlass.range_constexpr(T):
                        fq, fk = factors(t)
                        shared_q[128 * t + thread] = qks[2 * t] * fq
                        shared_k[128 * t + thread] = qks[2 * t + 1] * fk
                    cute.arch.barrier(barrier_id=1, number_of_threads=128)

                def operands(t):
                    fq, fk = factors(t) if T == 1 else (None, None)
                    qr = cute.make_rmem_tensor((self.chunks, 4), Float32)
                    kr = cute.make_rmem_tensor((self.chunks, 4), Float32)
                    for j in cutlass.range_constexpr(self.chunks):
                        column = 4 * (group_lane + self.lanes_per_row * j)
                        q4 = load_f32x4(
                            advance(shared_q.iterator, 128 * t) + column, shared=True
                        )
                        k4 = load_f32x4(
                            advance(shared_k.iterator, 128 * t) + column, shared=True
                        )
                        for e in cutlass.range_constexpr(4):
                            qr[j, e] = q4[e] * fq if T == 1 else q4[e]
                            kr[j, e] = k4[e] * fk if T == 1 else k4[e]
                    if cutlass.const_expr(t == 0):
                        wait(gate_barrier)
                    decays = []
                    for j in cutlass.range_constexpr(self.chunks):
                        column = 4 * (group_lane + self.lanes_per_row * j)
                        decay_ptr = advance(shared_decay.iterator, 128 * t) + column
                        decays.append(load_f32x4(decay_ptr, shared=True))
                    values = shared_v
                    if cutlass.const_expr(T > 1):
                        pointer = shared_v.iterator + self.rows * t
                        values = cute.make_tensor(pointer, cute.make_layout(self.rows))
                    return kr, qr, decays, values, betas[t]

                ready = operands(0)
                for t in cutlass.range_constexpr(T):
                    current, ready = ready, operands(t + 1) if t + 1 < T else None
                    # Butterfly order: interleave rows per level, publish before state.
                    dots = cute.make_rmem_tensor(self.passes, Float32)
                    if cutlass.const_expr(self.staged):
                        wait(state_barrier)
                    source = staged if self.staged else None
                    owner, bar = t % self.cluster, advance(barrier, t // self.cluster)
                    for start in cutlass.range_constexpr(0, self.passes, self.wave):
                        stop = min(start + self.wave, self.passes)
                        lanes = (group, group_lane)
                        self.update(
                            history, dots, current, lanes, (start, stop), source
                        )
                        if cutlass.const_expr(self.early and start == 0 and t == 0):
                            cute.arch.cluster_wait()
                        for p in cutlass.range_constexpr(start, stop):
                            row = p * self.groups + group
                            send = (row < self.rows) & (group_lane == 0)
                            if cutlass.const_expr(T > 1):  # predicated: no branches
                                position = self.output_position(value_base + row)
                                dst = advance(shared_o.iterator, 128 * t) + position
                                push_float(dst, dots[p], bar, owner, when=send)
                            elif send:
                                position = self.output_position(value_base + row)
                                dst = shared_o.iterator + position
                                store_async_dsmem(dst, float_bits(dots[p]), bar, owner)
                        rows = partial(state_row, s=page(writes[t]))
                        when = None if T == 1 else writes[t] > 0
                        self.store_rows(history, rows, (start, stop), lanes, when)
                if thread < self.rows:
                    channel = 2 * dim + value_channel
                    window = (
                        taps[2, 1],
                        taps[2, 2],
                        *(inputs[3 * t + 2] for t in range(T)),
                    )
                    for i, value in enumerate(window):
                        conv_page[channel, i] = value

                for t in cutlass.range_constexpr(T):
                    if partition == t % self.cluster:
                        wait(advance(barrier, t // self.cluster))
                        raw_o = shared_o[advance(self.output_position(thread), 128 * t)]
                        sums = cute.make_rmem_tensor(4, Float32)
                        for g in cutlass.range_constexpr(4):
                            base = lane % 8 * 16 + g * 4
                            o_row = advance(shared_o.iterator, 128 * t)
                            o4 = load_f32x4(o_row + base, shared=True)
                            s0, s1, s2, s3 = (o * o for o in o4)
                            sums[g] = warp_sum((s0 + s2) + (s1 + s3), 8)
                        sumsq = (sums[0] + sums[2]) + (sums[1] + sums[3])
                        rstd = cute.math.rsqrt(sumsq * (1.0 / 128) + eps, fastmath=True)
                        result = raw_o * rstd * nw * gates[t]
                        output[advance(token, t), qk_channel] = BFloat16(result)
                        if cutlass.const_expr(t == 0):
                            for section in cutlass.range_constexpr(2):
                                channel = section * dim + qk_channel
                                window = (taps[section, 1], taps[section, 2])
                                window += tuple(
                                    inputs[3 * u + section] for u in range(T)
                                )
                                for i, value in enumerate(window):
                                    conv_page[channel, i] = value

    @cute.jit
    def store_rows(self, history, state_row, span, lanes, when=None):
        """Store the passes in ``span`` of the CTA's state rows (``when``: predicated
        on it, without branches)."""
        (start, stop), (group, group_lane) = span, lanes
        for p in cutlass.range_constexpr(start, stop):
            row = p * self.groups + group
            if cutlass.const_expr(when is not None):
                for j in cutlass.range_constexpr(self.chunks):
                    column = 4 * (group_lane + self.lanes_per_row * j)
                    values = [history[p, j, e] for e in range(4)]
                    pointer = state_row(row % self.rows, column)
                    store_f32x4(pointer, values, when=when & (row < self.rows))
            elif row < self.rows:
                for j in cutlass.range_constexpr(self.chunks):
                    column = 4 * (group_lane + self.lanes_per_row * j)
                    pointer = state_row(row, column)
                    row_values = [history[p, j, e] for e in range(4)]
                    if cutlass.const_expr(self.evict):
                        store_evict_f32x4(pointer, row_values)
                    else:
                        store_f32x4(pointer, row_values)

    @cute.jit
    def update(self, history, dots, operands, lanes, span, staged):
        """Decay, delta-rule update and query dot of the passes in ``span``, whose rows
        ``staged`` (if given) first supplies from shared memory."""
        kr, qr, decays, shared_v, beta = operands
        (group, lane), (start, stop) = lanes, span
        if cutlass.const_expr(staged is not None):
            for p in cutlass.range_constexpr(start, stop):
                row = (p * self.groups + group) % self.rows
                for j in cutlass.range_constexpr(self.chunks):
                    column = 4 * (lane + self.lanes_per_row * j)
                    values = load_staged(staged.iterator + (row * 128 + column))
                    for e in cutlass.range_constexpr(4):
                        history[p, j, e] = values[e]
        for p in cutlass.range_constexpr(start, stop):
            for j in cutlass.range_constexpr(self.chunks):
                for e in cutlass.range_constexpr(4):
                    history[p, j, e] *= decays[j][e]
            dots[p] = self.row_dot(history, kr, p)
        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
            for p in cutlass.range_constexpr(start, stop):
                dots[p] += cute.arch.shuffle_sync_bfly(dots[p], 1 << shift)
        for p in cutlass.range_constexpr(start, stop):
            safe_row = (p * self.groups + group) % self.rows
            v_new = (shared_v[safe_row] - dots[p]) * beta
            for j in cutlass.range_constexpr(self.chunks):
                for e in cutlass.range_constexpr(4):
                    history[p, j, e] = fma(
                        kr[j, e], v_new, history[p, j, e], ftz=self.fast_math
                    )
            dots[p] = self.row_dot(history, qr, p)
        for shift in reversed(range(self.lanes_per_row.bit_length() - 1)):
            for p in cutlass.range_constexpr(start, stop):
                dots[p] += cute.arch.shuffle_sync_bfly(dots[p], 1 << shift)

    @cute.jit
    def gate_warp(self, args, shared_decay, barrier, coords, staged_rows):
        """Project one value partition and multicast its decays to every CTA. Modes fix
        the F_B reduction order: 0 = one chain of eight MMAs, 1 = four chains of two,
        2 = sixteen 8-FMA partials summed in a butterfly. Projection math stays IEEE.
        ``staged_rows``: shared [copy, lane, 4] words; each lane cp.asyncs its own
        16-byte F_B operands there before the PDL wait and reads them back after it."""
        f_a, a_log, dt_bias, lower_bound, weight = (args[i] for i in (4, 6, 7, 13, 16))
        head, token, value_base, lane = coords
        rows, mode = self.rows, self.gate_mode
        per: cutlass.Constexpr = max(1, rows // 32)  # decays per lane
        a = a_log[head]
        if cutlass.const_expr(mode == 2):
            feature = lane % rows
            row = head * 128 + value_base + feature
            # uint4 weights before the wait, two uint2 activation loads per 8-FMA
            # partial after it; keep that granularity and the low-then-high FMA order.
            wr = [load_weight_bf16x8(at(weight, row, 8 * i)) for i in range(16)]
            dt = dt_bias[row]
            cute.arch.griddepcontrol_wait()
            partial = cute.make_rmem_tensor(16, Float32)
            for i in cutlass.range_constexpr(16):
                a0 = load_activation_bf16x4(at(f_a, token, 8 * i))
                a1 = load_activation_bf16x4(at(f_a, token, 8 * i + 4))
                av = (*a0, *a1)
                acc = Float32(0)
                for e in cutlass.range_constexpr(4):
                    x0, x1 = unpack_bf16(av[e])
                    w0, w1 = unpack_bf16(wr[i][e])
                    acc = fma(x0, w0, acc)
                    acc = fma(x1, w1, acc)
                partial[i] = acc
            for shift in cutlass.range_constexpr(3, -1, -1):
                for i in cutlass.range_constexpr(1 << shift):
                    partial[i] += partial[i | (1 << shift)]
            raw_g = [Float32(BFloat16(partial[0]))]
            dt = [dt]
        else:
            tiles: cutlass.Constexpr = rows // 16
            chains: cutlass.Constexpr = 4 if mode == 1 else 1
            group, quarter = lane >> 2, lane & 3
            feature = 16 * (quarter >> 1) + group + 8 * (quarter & 1)
            dt = [Float32(0) for _ in range(per)]
            if cutlass.const_expr(self.tokens > 1):  # token column, its features
                token += group % self.tokens
                features = [
                    16 * (i // 2) + group + 8 * (i % 2) for i in range(rows // 8)
                ]
                dt = [dt_bias[head * 128 + value_base + f] for f in features]
            for r in cutlass.range_constexpr(per if self.tokens == 1 else 0):
                if feature + 32 * r < rows:
                    dt[r] = dt_bias[head * 128 + value_base + 32 * r + feature]
            # Two decays per lane load 16 bytes per row: chunk c then covers columns
            # 32 (c // 2) + 8 quarter + 4 (c % 2) + [0, 4) instead of 16 c + 4 quarter.
            wide: cutlass.Constexpr = per >= 2
            wr = cute.make_rmem_tensor((tiles, 8, 4), Uint32)
            if cutlass.const_expr(staged_rows is not None):
                for tile in cutlass.range_constexpr(tiles):
                    row = head * 128 + value_base + 16 * tile + group
                    for chunk in cutlass.range_constexpr(4):
                        col = 32 * chunk + 8 * quarter
                        for half in cutlass.range_constexpr(2):
                            slot = staged_rows[
                                (tile * 4 + chunk) * 2 + half, lane, None
                            ]
                            source = at(weight, row + 8 * half, col)
                            copy_g2s(slot.iterator, source)
                cute.arch.cp_async_commit_group()
            else:
                for tile in cutlass.range_constexpr(tiles):
                    row = head * 128 + value_base + 16 * tile + group
                    for chunk in cutlass.range_constexpr(4 if wide else 8):
                        if cutlass.const_expr(wide):
                            col = 32 * chunk + 8 * quarter
                            low = load_weight_bf16x8(at(weight, row, col))
                            high = load_weight_bf16x8(at(weight, row + 8, col))
                            self.split_pairs(wr, tile, chunk, low, high)
                        else:
                            col = 16 * chunk + 4 * quarter
                            low = load_bf16x4(at(weight, row, col))
                            high = load_bf16x4(at(weight, row + 8, col))
                            wr[tile, chunk, 0], wr[tile, chunk, 2] = low
                            wr[tile, chunk, 1], wr[tile, chunk, 3] = high
            cute.arch.griddepcontrol_wait()
            if cutlass.const_expr(wide):
                pairs = [
                    load_activation_bf16x8(at(f_a, token, 32 * c + 8 * quarter))
                    for c in range(4)
                ]
                ar = [pairs[c // 2][2 * (c % 2) : 2 * (c % 2) + 2] for c in range(8)]
            else:
                ar = [
                    load_activation_bf16x4(at(f_a, token, 16 * c + 4 * quarter))
                    for c in range(8)
                ]
            if cutlass.const_expr(staged_rows is not None):
                cute.arch.cp_async_wait_group(0)
                for tile in cutlass.range_constexpr(tiles):
                    for chunk in cutlass.range_constexpr(4):
                        copy = (tile * 4 + chunk) * 2
                        low = smem_words(staged_rows[copy, lane, None].iterator)
                        high = smem_words(staged_rows[copy + 1, lane, None].iterator)
                        self.split_pairs(wr, tile, chunk, low, high)
            acc = zeros((chains, tiles, 4), Float32)
            for chunk in cutlass.range_constexpr(8):
                chain: cutlass.Constexpr = chunk // 2 if mode == 1 else 0
                for tile in cutlass.range_constexpr(tiles):
                    wa = tuple(wr[tile, chunk, e] for e in range(4))
                    mma_into(acc[chain, tile, None], wa, ar[chunk], bf16=True)
            if cutlass.const_expr(self.tokens > 1):
                gates = (a, lower_bound, dt, features)
                targets = (shared_decay, barrier, value_base)
                return self.multicast(acc, gates, targets, lane)
            raw_g = []
            for r in cutlass.range_constexpr(per):
                partial = cute.make_rmem_tensor(chains, Float32)
                for chain in cutlass.range_constexpr(chains):
                    selected = acc[chain, 2 * r, 0]
                    if quarter & 1:
                        selected = acc[chain, 2 * r, 2]
                    if cutlass.const_expr(tiles > 2 * r + 1):  # noqa: SIM102
                        if quarter >> 1:
                            selected = acc[chain, 2 * r + 1, 0]
                            if quarter & 1:
                                selected = acc[chain, 2 * r + 1, 2]
                    partial[chain] = selected
                value = partial[0]
                if cutlass.const_expr(mode == 1):
                    value = Float32(0) + (
                        ((value + partial[1]) + partial[2]) + partial[3]
                    )
                raw_g.append(Float32(BFloat16(value)))

        decay = []
        for r in cutlass.range_constexpr(per):
            gate = cute.math.exp(a, fastmath=True) * (raw_g[r] + dt[r])
            gate = sigmoid(gate, self.fast_math)
            decay.append(cute.math.exp(lower_bound * gate, fastmath=True))
        packets: cutlass.Constexpr = rows // 4
        packet, destination = lane % packets, lane // packets
        values = cute.make_rmem_tensor(4, Float32)
        for e in cutlass.range_constexpr(4):
            source = 4 * packet + e
            if cutlass.const_expr(per == 2):
                feature = source & 31
                source = (feature & 7) << 2 | (feature >> 4) << 1 | feature >> 3 & 1
                low = cute.arch.shuffle_sync(decay[0], source)
                high = cute.arch.shuffle_sync(decay[1], source)
                values[e] = low
                if packet >= 8:
                    values[e] = high
            elif cutlass.const_expr(per > 2):  # one CTA per head: 128 features
                feature = source & 31
                source = (feature & 7) << 2 | (feature >> 4) << 1 | feature >> 3 & 1
                shuffled = [cute.arch.shuffle_sync(d, source) for d in decay]
                values[e] = shuffled[0]
                for r in cutlass.range_constexpr(1, per):
                    if packet >> 3 == r:
                        values[e] = shuffled[r]
            else:
                if cutlass.const_expr(mode != 2):
                    source = (source & 7) << 2 | (source >> 4) << 1 | source >> 3 & 1
                values[e] = cute.arch.shuffle_sync(decay[0], source)
        offset = shared_decay.iterator + value_base + 4 * packet
        pointer = cute.make_ptr(
            Float32, offset.llvm_ptr, cute.AddressSpace.smem, assumed_align=16
        )
        words = tuple(float_bits(values[e]) for e in range(4))
        if cutlass.const_expr(self.early):
            cute.arch.cluster_wait()
        store_async_dsmem(pointer, words, barrier, destination)

    @cute.jit
    def multicast(self, acc, gates, targets, lane):
        """Decays (q > 1) of the lane's D fragments (feature rows, token columns 2
        quarter + {0, 1}) to every CTA's shared [token, 128] decays."""
        (a, lower_bound, dt, features), (shared_decay, barrier, value_base) = (
            gates,
            targets,
        )
        quarter = lane & 3
        if cutlass.const_expr(self.early):
            cute.arch.cluster_wait()
        for i in cutlass.range_constexpr(len(features)):
            tile, e = i // 2, (i % 2) * 2
            for column in cutlass.range_constexpr(2):
                value = acc[0, tile, e + column]
                if cutlass.const_expr(self.gate_mode == 1):
                    parts = [acc[c, tile, e + column] for c in range(4)]
                    value = Float32(0) + (((value + parts[1]) + parts[2]) + parts[3])
                raw_g = Float32(BFloat16(value))
                gate = cute.math.exp(a, fastmath=True) * (raw_g + dt[i])
                gate = sigmoid(gate, self.fast_math)
                decay = float_bits(cute.math.exp(lower_bound * gate, fastmath=True))
                token = 2 * quarter + column
                offset = token * 128 + value_base + features[i]
                pointer = shared_decay.iterator + offset
                if token < self.tokens:
                    for rank in cutlass.range_constexpr(self.cluster):
                        store_async_dsmem(pointer, decay, barrier, rank)

    @staticmethod
    def split_pairs(wr, tile, chunk, low, high):
        """16 bytes of rows r and r + 8 hold the A fragments of chunks 2c and 2c + 1."""
        for h in range(2):
            c = 2 * chunk + h
            wr[tile, c, 0], wr[tile, c, 2] = low[2 * h : 2 * h + 2]
            wr[tile, c, 1], wr[tile, c, 3] = high[2 * h : 2 * h + 2]

    @cute.jit
    def setup(self, barrier, gate_barrier, partition, thread):
        """Thread 0 arms the output (leader) and decay barriers; the CTA arrives on the
        cluster barrier peers wait on before their sends."""
        tokens, cluster = self.tokens, self.cluster
        owned = [(barrier, partition == 0, 512)]  # token t's outputs: CTA t % cluster
        if cutlass.const_expr(tokens > 1):
            owned = [
                (barrier + k, partition + k * cluster < tokens, 512)
                for k in range(-(-tokens // cluster))
            ]
        for bar, cta, size in (*owned, (gate_barrier, True, 512 * tokens)):
            if cta & (thread == 0):
                cute.arch.mbarrier_init(bar, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bar, size)
                cute.arch.mbarrier_init_fence()
        cute.arch.cluster_arrive_relaxed()

    @cute.jit
    def row_dot(self, history, vector, p):
        partial = cute.make_rmem_tensor(self.chunks, Float32)
        for j in cutlass.range_constexpr(self.chunks):
            row = (history[p, j, e] for e in range(4))
            partial[j] = dot(row, (vector[j, e] for e in range(4)), ftz=self.fast_math)
        for shift in cutlass.range_constexpr(self.chunks.bit_length() - 2, -1, -1):
            for j in cutlass.range_constexpr(1 << shift):
                partial[j] += partial[j + (1 << shift)]
        return partial[0]

    @cute.jit
    def output_position(self, row: Int32):
        return ((row & 7) << 4) | ((row >> 5) << 2) | ((row & 31) >> 3)
