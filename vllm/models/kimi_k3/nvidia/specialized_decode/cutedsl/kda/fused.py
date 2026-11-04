# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused replay verify (KdaFused): the RecoverSSM commit of the previous step and the
verify of this step in one pass over the FP32 state, the verify in chunked form.

Deferred RecoverSSM. After sampling, the runner no longer folds the accepted tokens
into the checkpoint; it records per request (its request-state slot) how many of the
last verify's tokens were accepted (``pending``). The verify records of a request
stay in its request-indexed record slot. The next verify of that request, per
(request, head):

1. reads the checkpoint S_0 and replays the ``n = pending`` accepted tokens of the
   previous step from its records in the commit's closed form,

       S_n = S_0 * D_n + sum_{t<n} u_t (kn_t * P_t),  P_t = prod_{t<s<n} d_s,

   (``u_t`` the recorded corrections, ``kn_t`` the normalized recorded keys, ``d_t``
   the recorded gates' decays, D_n = prod_{s<n} d_s), and writes S_n back as the
   new checkpoint (nothing to write for n = 0); the convolution window is read at
   the offset n - 1 the commit's compaction would have rolled it by;
2. verifies the step's q tokens from S_n in chunked (WY) form:
   with Gamma_t = prod_{r<=t} d_r (this step's decays) and P(s, t) = Gamma_t /
   Gamma_s = prod_{s<r<=t} d_r,

       w_t = S_n (Gamma_t * k_t),         y_t = S_n (Gamma_t * q_t),
       A_ts = k_s . (P(s, t) * k_t) (s < t),  B_ts = k_s . (P(s, t) * q_t) (s <= t),
       u_t = beta_t (v_t - w_t - sum_{s<t} A_ts u_s),
       o_t = y_t + sum_{s<=t} B_ts u_s,

   which is the stock recurrence (S <- S diag(d_t); u_t = beta_t (v_t - S k_t);
   S <- S + u_t k_t^T; o_t = S q_t) rewritten: the state is only read (2q
   dot products per row, no dependent chain over the tokens), the q x q systems are
   solved per state row from the A / B pair products;
3. writes the new records (u_t per row, the key and raw gate per token) into the same
   request slot (every old record is read before any is written).

So each state byte is read once and written at most once per step (the separate
commit read and wrote it again), and the recurrence's per-token chain is gone.

Precision: FP32 state and math; BF16 roundings at the stock points (convolution /
SiLU outputs, raw gates, recurrence outputs, conv window); only the FP32 rounding of
the recurrence differs from stock (chunked form, as the user allowed).

Layout: a cluster of ``split`` CTAs per (request, head) splits the state rows (the
rows of the chunked form are independent): each CTA streams its 128 / split rows and
recomputes the shared per-item operands (convolution, norms, gate projection, pair
products); the RMSNorm's sum of squares is exchanged over DSMEM. split = 1 is one
CTA per item (two per SM). Thread (g, l): rows g + G r (r < rows per thread),
keys 4 (l + LANES c) + [0, 4)."""

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
    mma_into,
    push_float,
    scatter_reduce,
    store_f32x4,
    wait,
    warp_sum,
    zeros,
)
from .kernel import (
    div_full,
    mul_rn,
    reciprocal,
    sqrt_approx,
    triton_exp,
    triton_sigmoid,
)
from .wide import load_activation_bf16x4, load_shared_f32x4, load_weight_bf16x4

K = 128
THREADS = 256


def pair_count(tokens):
    """A (s < t) and B (s <= t) pair products: tokens^2."""
    return tokens * tokens


def a_index(t, s):
    return t * (t - 1) // 2 + s


def b_index(tokens, t, s):
    return tokens * (tokens - 1) // 2 + t * (t + 1) // 2 + s


class KdaFused:
    """``tokens`` q, ``gate_mode`` the F_B chain order (KdaWide's), ``split`` CTAs
    per (request, head). Operands: PreparedKda's RecoverSSM verify operands, then
    ``record_rows`` [B] (each request's record / pending slot; < 0: none) and
    ``pending`` [slots] (accepted tokens of the last verify not yet in the
    checkpoint)."""

    def __init__(self, heads, tokens, gate_mode, has_bias, split):
        assert 2 <= tokens <= 8 and gate_mode in (0, 1) and split in (1, 2, 4, 8)
        self.heads, self.tokens, self.gate_mode = heads, tokens, gate_mode
        self.has_bias, self.split = has_bias, split
        self.rows = K // split  # state rows per CTA
        self.rpt = max(1, self.rows // 64)  # rows per thread
        self.lanes = THREADS * self.rpt // self.rows  # threads per row
        self.kch = K // self.lanes // 4  # 4-key chunks per thread and row
        self.groups = self.rows // self.rpt

    @cute.jit
    def __call__(self, *args):
        grid = (self.split, self.heads, args[0].shape[0] // self.tokens)
        cluster = (self.split, 1, 1) if self.split > 1 else None
        launch(self.kernel, args, grid, THREADS, cluster, min_blocks_per_mp=2)

    @cute.kernel
    def kernel(self, args):
        x, weights, bias, conv_state, f_a, beta, a_log, dt_bias, slots = args[:9]
        state, output, gate, norm_weight, lower_bound, scale, eps, f_b = args[9:17]
        corrections, key_gates, record_rows, pending = args[18:22]
        T: cutlass.Constexpr = self.tokens
        S: cutlass.Constexpr = self.split
        RC: cutlass.Constexpr = self.rows
        RPT: cutlass.Constexpr = self.rpt
        LANES: cutlass.Constexpr = self.lanes
        KCH: cutlass.Constexpr = self.kch
        G: cutlass.Constexpr = self.groups
        RQ: cutlass.Constexpr = corrections.shape[2]  # tokens a record holds
        NP: cutlass.Constexpr = pair_count(T)
        P2: cutlass.Constexpr = 1 << (NP - 1).bit_length()
        tid, _, _ = cute.arch.thread_idx()
        rank, head, request = cute.arch.block_idx()
        lane, warp = tid % 32, tid // 32
        dim: cutlass.Constexpr = self.heads * K
        token = request * T
        row0 = rank * RC
        slot = slots[request, 0]
        if slot <= 0:
            cute.arch.griddepcontrol_wait()
            if tid < RC:
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * K + row0 + tid] = BFloat16(0)
            cute.arch.griddepcontrol_launch_dependents()
        else:
            smem = cutlass.utils.SmemAllocator()
            staged = alloc(smem, Float32, (RPT * KCH, THREADS, 4), align=128)
            xvec = alloc(smem, Float32, (2 * T, K))  # Gamma_t k_t, Gamma_t q_t
            kp = alloc(smem, Float32, (RQ, K))  # replay: kn_t * P_t
            decay_n = alloc(smem, Float32, K)  # replay: D_n
            decays = alloc(smem, Float32, (T, K))  # this step's d_t
            raw_gates = alloc(smem, Float32, (T, K))  # BF16 values, for the records
            shared_v = alloc(smem, Float32, (T, K))
            shared_o = alloc(smem, Float32, (T, RC))
            dots = alloc(smem, Float32, (RC, 2 * T + 1))
            pairs = alloc(smem, Float32, (P2, 4))
            pair_sums = alloc(smem, Float32, P2)
            sums = alloc(smem, Float32, (8, 3, 4))  # q, k squares; recorded k squares
            shared_beta = alloc(smem, Float32, 8)
            out_sums = alloc(smem, Float32, (T, 4))
            # Every CTA's RMSNorm sums of the item (pushed by each peer, by rank).
            received = alloc(smem, Float32, (S, T))
            exchange = smem.allocate(cutlass.Uint64, 1)
            if cutlass.const_expr(S > 1):
                if tid == 0:
                    cute.arch.mbarrier_init(exchange, 1)
                    cute.arch.mbarrier_arrive_and_expect_tx(exchange, S * T * 4)
                    cute.arch.mbarrier_init_fence()
                cute.arch.cluster_arrive_relaxed()  # waited before the first push

            # Loads that need only the request: n (the pending count) and the
            # record slot gate the replay operands, which load regardless.
            rec = record_rows[request]
            live = rec >= 0
            n = Int32(0)
            if live:
                n = pending[rec]
            if n > RQ:
                n = Int32(RQ)
            if n < 0:
                n = Int32(0)

            # This CTA's state rows: cp.async into thread-private slots.
            base_row, lane_in = tid // LANES, tid % LANES
            page = state[Int64(slot), None, None, None]
            for r in cutlass.range_constexpr(RPT):
                for c in cutlass.range_constexpr(KCH):
                    source = at(
                        page, head, row0 + base_row + G * r, 4 * (lane_in + LANES * c)
                    )
                    copy_g2s(staged[r * KCH + c, tid, None].iterator, source)
            cute.arch.cp_async_commit_group()

            # The recorded corrections of this thread's rows (replay; 0 past n).
            u_old = zeros((RPT, RQ), Float32)
            if live:
                for r in cutlass.range_constexpr(RPT):
                    for t in cutlass.range_constexpr(RQ):
                        u_old[r, t] = corrections[rec, head, t, row0 + base_row + G * r]

            # Convolution operands: threads 0-127 the q and k channels, 128-255 v.
            channel = tid % K
            section = tid // K  # 0: q and k, 1: v
            conv_page = conv_state[Int64(slot), None, None]
            conv_type: cutlass.Constexpr = conv_state.element_type
            # The window [tap 1, tap 2, x_0, ..., x_q-1] of the last verify: the
            # history after n accepted tokens starts at n - 1 (0 if none pending).
            W: cutlass.Constexpr = RQ + 2
            window = zeros((2, W), conv_type)
            taps = zeros((2, 3), conv_type)
            coeff = zeros((2, 4), Float32)
            accum = zeros((2, T), Float32)
            for s in cutlass.range_constexpr(2):
                part = 2 * section + s  # 0 q, 1 k, 2 v
                if (section == 0) | (s == 0):
                    conv_channel = part * dim + head * K + channel
                    for i in cutlass.range_constexpr(W):
                        window[s, i] = conv_page[conv_channel, i]
                    for t in cutlass.range_constexpr(4):
                        coeff[s, t] = Float32(weights[part, t, head * K + channel])
                    if cutlass.const_expr(self.has_bias):
                        for t in cutlass.range_constexpr(T):
                            accum[s, t] = Float32(bias[conv_channel])
            a = a_log[head]
            dt = [dt_bias[head * K + 16 * warp + lane // 4 + 8 * i] for i in range(2)]
            dt_channel = Float32(dt_bias[head * K + channel])
            # The recorded keys and raw gates (section 1, channel c).
            old_keys = zeros(RQ, Float32)
            old_gates = zeros(RQ, Float32)
            if live & (section == 1):
                for t in cutlass.range_constexpr(RQ):
                    old_keys[t] = Float32(key_gates[rec, head, t, channel])
                    old_gates[t] = Float32(key_gates[rec, head, t, K + channel])

            # Gate projection weights (warp w: features 16 w ...): A fragments.
            group, quarter = lane >> 2, lane & 3
            wr = cute.make_rmem_tensor((8, 4), Uint32)
            for chunk in cutlass.range_constexpr(8):
                feature = head * K + 16 * warp + group
                col = 16 * chunk + 4 * quarter
                low = load_weight_bf16x4(at(f_b, feature, col))
                high = load_weight_bf16x4(at(f_b, feature + 8, col))
                wr[chunk, 0], wr[chunk, 2] = low
                wr[chunk, 1], wr[chunk, 3] = high

            cute.arch.griddepcontrol_wait()
            inputs = cute.make_rmem_tensor((2, T), BFloat16)
            for t in cutlass.range_constexpr(T):
                inputs[0, t] = x[token + t, 2 * section * dim + head * K + channel]
                inputs[1, t] = x[token + t, dim + head * K + channel]  # k (section 0)
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

            offset = Int32(0)
            if n > 0:
                offset = n - 1
            for s in cutlass.range_constexpr(2):
                for i in cutlass.range_constexpr(3):
                    taps[s, i] = window[s, i]
                    for o in cutlass.range_constexpr(1, RQ):
                        if offset == o:
                            taps[s, i] = window[s, o + i]
            for r in cutlass.range_constexpr(RPT):
                for t in cutlass.range_constexpr(RQ):
                    if t >= n:
                        u_old[r, t] = Float32(0)
            old_decays = zeros(RQ, Float32)
            if section == 1:
                a_exp = triton_exp(Float32(a))
                for t in cutlass.range_constexpr(RQ):
                    if t >= n:
                        old_keys[t] = Float32(0)
                    gated = lower_bound * triton_sigmoid(
                        a_exp * (old_gates[t] + dt_channel)
                    )
                    old_decays[t] = triton_exp(gated)

            # The convolution (stock tap order) and SiLU, BF16-rounded.
            for s in cutlass.range_constexpr(2):
                if (section == 0) | (s == 0):
                    for t in cutlass.range_constexpr(T):
                        for i in cutlass.range_constexpr(4):
                            w = t + i
                            term = taps[s, w] if w < 3 else inputs[s, w - 3]
                            accum[s, t] = fma(Float32(term), coeff[s, i], accum[s, t])
            values_q = zeros(T, Float32)
            values_k = zeros(T, Float32)
            for t in cutlass.range_constexpr(T):
                first = self.silu(accum[0, t])
                if section == 0:
                    second = self.silu(accum[1, t])
                    squares_q = warp_sum(first * first)
                    squares_k = warp_sum(second * second)
                    if lane == 0:
                        sums[t, 0, warp] = squares_q
                        sums[t, 1, warp] = squares_k
                    values_q[t] = first
                    values_k[t] = second
                else:
                    shared_v[t, channel] = first
            if section == 1:
                for t in cutlass.range_constexpr(RQ):
                    if t < n:
                        squares = warp_sum(old_keys[t] * old_keys[t])
                        if lane == 0:
                            sums[t, 2, warp - 4] = squares

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
                    decays[t, feature] = self.decay(a, raw_g, dt[e // 2], lower_bound)
                    raw_gates[t, feature] = raw_g
            cute.arch.barrier()

            if section == 0:
                # Normalized q (with the 1/sqrt(K) scale) and k; the chunk vectors
                # and this channel's pair products.
                qs = zeros(T, Float32)
                ks = zeros(T, Float32)
                gamma = Float32(1)
                for t in cutlass.range_constexpr(T):
                    qn, kn = (self.quad_sum(sums[t, i, None]) + 1.0e-6 for i in (0, 1))
                    fq = reciprocal(sqrt_approx(qn))
                    fk = reciprocal(sqrt_approx(kn))
                    qs[t] = mul_rn(mul_rn(values_q[t], fq), scale)
                    ks[t] = mul_rn(values_k[t], fk)
                    gamma = gamma * decays[t, channel]
                    xvec[t, channel] = gamma * ks[t]
                    xvec[T + t, channel] = gamma * qs[t]
                products = [Float32(0) for _ in range(P2)]
                for s in cutlass.range_constexpr(T):
                    products[b_index(T, s, s)] = ks[s] * qs[s]
                    p = Float32(1)
                    for t in cutlass.range_constexpr(s + 1, T):
                        p = p * decays[t, channel]
                        weighted = ks[s] * p
                        products[a_index(t, s)] = weighted * ks[t]
                        products[b_index(T, t, s)] = weighted * qs[t]
                for g in cutlass.range_constexpr(-(-P2 // 32)):
                    count: cutlass.Constexpr = min(32, P2)
                    chunk_values = products[32 * g : 32 * g + count]
                    total, selected = scatter_reduce(chunk_values, lane, count)
                    if lane % (32 // count) == 0:
                        pairs[32 * g + selected, warp] = total
            else:
                # The replay operands: normalized recorded keys times the suffix
                # decay products (the commit's association), and D_n.
                # Tokens past n replay as zero keys (and zero corrections) with
                # unit decay, so the scan has no per-token branch.
                fd = Float32(1)
                for r in cutlass.range_constexpr(RQ):
                    t = RQ - 1 - r
                    weighted = Float32(0)
                    if t < n:
                        total = (sums[t, 2, 0] + sums[t, 2, 1]) + (
                            sums[t, 2, 2] + sums[t, 2, 3]
                        )
                        kn = old_keys[t] * cute.math.rsqrt(total + Float32(1e-6))
                        weighted = kn * fd
                        fd = fd * old_decays[t]
                    kp[t, channel] = weighted
                decay_n[channel] = fd
            cute.arch.barrier()
            if tid < NP:
                pair_sums[tid] = self.quad_sum(pairs[tid, None])

            # Replay and the chunked products of this thread's rows.
            cute.arch.cp_async_wait_group(0)
            dot = zeros((RPT, 2 * T, 2), Float32)
            for c in cutlass.range_constexpr(KCH):
                key = 4 * (lane_in + LANES * c)
                rows_now = zeros((RPT, 4), Float32)
                for r in cutlass.range_constexpr(RPT):
                    values = load_shared_f32x4(staged[r * KCH + c, tid, None].iterator)
                    for e in cutlass.range_constexpr(4):
                        rows_now[r, e] = values[e]
                # The replay (n = 0: S_0 unchanged, D = 1, no keys; not stored).
                d4 = load_shared_f32x4(decay_n.iterator + key)
                final = zeros((RPT, 4), Float32)
                for t in cutlass.range_constexpr(RQ):
                    k4 = load_shared_f32x4(at(kp, t, key))
                    for r in cutlass.range_constexpr(RPT):
                        u = u_old[r, t]
                        for pp in cutlass.range_constexpr(2):
                            pair = fma2(
                                (u, u),
                                (k4[2 * pp], k4[2 * pp + 1]),
                                (final[r, 2 * pp], final[r, 2 * pp + 1]),
                            )
                            final[r, 2 * pp], final[r, 2 * pp + 1] = pair
                for r in cutlass.range_constexpr(RPT):
                    for pp in cutlass.range_constexpr(2):
                        pair = fma2(
                            (rows_now[r, 2 * pp], rows_now[r, 2 * pp + 1]),
                            (d4[2 * pp], d4[2 * pp + 1]),
                            (final[r, 2 * pp], final[r, 2 * pp + 1]),
                        )
                        rows_now[r, 2 * pp], rows_now[r, 2 * pp + 1] = pair
                    row = row0 + base_row + G * r
                    store_f32x4(
                        at(page, head, row, key),
                        [rows_now[r, e] for e in range(4)],
                        n > 0,
                    )
                for j in cutlass.range_constexpr(2 * T):
                    x4 = load_shared_f32x4(at(xvec, j, key))
                    for r in cutlass.range_constexpr(RPT):
                        for pp in cutlass.range_constexpr(2):
                            pair = fma2(
                                (rows_now[r, 2 * pp], rows_now[r, 2 * pp + 1]),
                                (x4[2 * pp], x4[2 * pp + 1]),
                                (dot[r, j, 0], dot[r, j, 1]),
                            )
                            dot[r, j, 0], dot[r, j, 1] = pair
            for r in cutlass.range_constexpr(RPT):
                for j in cutlass.range_constexpr(2 * T):
                    value = dot[r, j, 0] + dot[r, j, 1]
                    for shift in cutlass.range_constexpr((LANES - 1).bit_length()):
                        value += cute.arch.shuffle_sync_bfly(value, 1 << shift)
                    if lane_in == j % LANES:
                        dots[base_row + G * r, j] = value
            cute.arch.barrier()

            # Per state row: the q x q forward substitution and the outputs.
            if tid < RC:
                row = row0 + tid
                us = zeros(T, Float32)
                for t in cutlass.range_constexpr(T):
                    total = dots[tid, t]
                    for s in cutlass.range_constexpr(t):
                        total = fma(pair_sums[a_index(t, s)], us[s], total)
                    us[t] = (shared_v[t, row] - total) * shared_beta[t]
                for t in cutlass.range_constexpr(T):
                    o = dots[tid, T + t]
                    for s in cutlass.range_constexpr(t + 1):
                        o = fma(pair_sums[b_index(T, t, s)], us[s], o)
                    shared_o[t, tid] = Float32(BFloat16(o))
                    if rec >= 0:
                        corrections[rec, head, t, row] = us[t]
            cute.arch.barrier()
            # The gated RMSNorm's sums of squares (over this CTA's rows).
            if tid < (RC + 31) // 32 * 32:
                for t in cutlass.range_constexpr(T):
                    o = Float32(0)
                    if tid < RC:
                        o = shared_o[t, tid]
                    total = warp_sum(o * o)
                    if lane == 0:
                        out_sums[t, warp] = total
            cute.arch.barrier()
            WR: cutlass.Constexpr = (RC + 31) // 32
            if cutlass.const_expr(S > 1):
                cute.arch.cluster_wait()  # every peer's exchange barrier is armed
                if tid < T:
                    local = out_sums[tid, 0]
                    for i in cutlass.range_constexpr(1, WR):
                        local = local + out_sums[tid, i]
                    for peer in cutlass.range_constexpr(S):
                        push_float(
                            received.iterator + rank * T + tid, local, exchange, peer
                        )
                wait(exchange, 0)
            if tid < RC:
                row = row0 + tid
                nw = norm_weight[row]
                for t in cutlass.range_constexpr(T):
                    if cutlass.const_expr(S == 1):
                        sumsq = self.quad_sum(out_sums[t, None])
                    else:
                        sumsq = received[0, t]
                        for peer in cutlass.range_constexpr(1, S):
                            sumsq = sumsq + received[peer, t]
                    variance = sumsq * (1.0 / 128) + eps
                    rstd = reciprocal(sqrt_approx(variance))
                    sigmoid_gate = triton_sigmoid(
                        Float32(gate[token + t, head * K + row])
                    )
                    result = shared_o[t, tid] * rstd * nw * sigmoid_gate
                    output[token + t, head * K + row] = BFloat16(result)
            # The new records' keys and gates and the convolution window: written
            # once per item (rank 0) after every CTA of the item read the old ones.
            if rank == 0:
                if rec >= 0:
                    for t in cutlass.range_constexpr(T):
                        if section == 0:
                            key_gates[rec, head, t, channel] = BFloat16(values_k[t])
                        else:
                            key_gates[rec, head, t, K + channel] = BFloat16(
                                raw_gates[t, channel]
                            )
                for s in cutlass.range_constexpr(2):
                    if (section == 0) | (s == 0):
                        conv_channel = (2 * section + s) * dim + head * K + channel
                        conv_page[conv_channel, 0] = conv_type(taps[s, 1])
                        conv_page[conv_channel, 1] = conv_type(taps[s, 2])
                        for t in cutlass.range_constexpr(T):
                            conv_page[conv_channel, 2 + t] = conv_type(inputs[s, t])

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
