# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent warp-specialized fused replay verify (KdaFusedWS): KdaFused's math
(deferred RecoverSSM: the replay of the pending accepted tokens, the checkpoint
write and the chunked verify, see `fused`) on KdaWS's schedule.

One CTA per SM walks every `ctas`-th (request, head) item (strided, see kernel):

- two **prep warpgroups** (thread c = channel c; warpgroup w takes items w, w + 2,
  ...) stream each item in (projection rows and conv window by cp.async, the
  item's old records into their staging), run the convolution, SiLU, norms, F_B
  gate projection and decays, and fill one of four operand blocks: the chunk
  vectors Gamma_t k_t and Gamma_t q_t, the replay keys kn_t P_t and D_n, the old
  corrections, v, beta, the output gate and the q x q pair sums;
- the **scan group** (8 warps) takes the item's state page (bulk copy) into
  registers (thread (g, l): rows g + 32 r, keys 4 (l + 8 c) + [0, 4)), replays and
  stores the checkpoint, forms the 2q row dot products, solves each row's q x q
  system (thread = row), writes the corrections and runs the gated RMSNorm.

Synchronization as KdaWS (state_full / state_empty / ops_full[4] / done[4])."""

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
    scatter_reduce,
    store_f32x4,
    wait,
    warp_sum,
    zeros,
)
from .fused import a_index, b_index, pair_count
from .kernel import (
    div_full,
    mul_rn,
    reciprocal,
    sqrt_approx,
    triton_exp,
    triton_sigmoid,
)
from .specialized import (
    fence_async_shared,
    load_shared_u32x2,
    set_max_registers,
    store_bulk,
    sync_threads,
    wait_bulk_reads,
)
from .wide import load_weight_bf16x4

K = 128
GROUP_THREADS = 256
PREP_THREADS = 128
THREADS = GROUP_THREADS + 2 * PREP_THREADS
ROWS = 4  # state rows per scan thread: g + 32 r
LANES = 8  # scan threads per row
CHUNKS = 4  # 4-key chunks per thread and row
SCAN_REGISTERS, PREP_REGISTERS = 168, 88
PAGE_BYTES = K * K * 4
COPY_BYTES = 16384
SMEM_LIMIT = 227 * 1024
META = 8  # live, request, head, token, n, record slot, state slot


def load_shared_f32x4(pointer):
    return load_words(
        pointer,
        count=4,
        shared=True,
        mutable=True,
        result_type=Float32,
        word_type="f32",
    )


def operand_bytes(tokens, rq):
    """One operand block: chunk vectors (2q, the decays and later the outputs
    alias them), replay keys (rq) and D_n, old corrections (rq), v (FP32), the
    output gate (BF16), beta, pair sums, meta."""
    p2 = 1 << (pair_count(tokens) - 1).bit_length()
    floats = (2 * tokens + rq + 1 + rq + tokens) * K + 8 + p2
    return floats * 4 + tokens * K * 2 + META * 4


def staging_bytes(tokens, rq, conv_dim_first, window, conv_size):
    rows = 5 * tokens * K * 2  # x (3 parts), gate, F_A
    conv = 3 * K * window * conv_size if conv_dim_first else 9 * K * conv_size
    records = rq * 2 * K * 2 + rq * K * 4  # old key / gate, old corrections
    return rows + conv + records


def shared_bytes(tokens, rq, conv_dim_first, window, conv_size):
    staging = staging_bytes(tokens, rq, conv_dim_first, window, conv_size)
    sums = 2 * 64 * 4 * 4 + 2 * 8 * 4 * 4
    dots = K * (2 * tokens + 1) * 4
    blocks = 4 * operand_bytes(tokens, rq)
    return PAGE_BYTES + blocks + 2 * staging + sums + dots + 12 * 8 + 1024


class KdaFusedWS:
    """KdaFused's operands and semantics on `ctas` persistent CTAs."""

    strided = True  # item order (see kernel)
    registers = (SCAN_REGISTERS, PREP_REGISTERS)

    def __init__(
        self, heads, tokens, gate_mode, has_bias, conv_dim_first, window, ctas
    ):
        assert 2 <= tokens <= 8 and gate_mode in (0, 1)
        self.heads, self.tokens, self.gate_mode = heads, tokens, gate_mode
        self.has_bias = has_bias
        self.conv_dim_first, self.window, self.ctas = conv_dim_first, window, ctas

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1)

    # ---------------------------------------------------------------- helpers
    @cute.jit
    def item_info(self, args, item):
        """(head, request, state slot, record slot, pending n, window offset)."""
        slots, record_rows, pending = args[8], args[20], args[21]
        RQ: cutlass.Constexpr = args[18].shape[2]
        head, request = item % self.heads, item // self.heads
        slot = Int32(slots[request, 0])
        rec = Int32(record_rows[request])
        n = Int32(0)
        if rec >= 0:
            n = Int32(pending[rec])
        if n > RQ:
            n = Int32(RQ)
        if n < 0:
            n = Int32(0)
        offset = Int32(0)
        if n > 0:
            offset = n - 1
        return head, request, slot, rec, n, offset

    @cute.jit
    def stage_conv(self, args, convs, p, head, conv_slot, offset):
        conv_state = args[3]
        dim: cutlass.Constexpr = self.heads * K
        size: cutlass.Constexpr = conv_state.element_type.width // 8
        per: cutlass.Constexpr = 16 // size
        page = conv_state[Int64(conv_slot), None, None]
        if cutlass.const_expr(self.conv_dim_first):
            part_granules: cutlass.Constexpr = K * self.window // per
            for part in cutlass.range_constexpr(3):
                base = at(page, part * dim + head * K, 0)
                for j in cutlass.range_constexpr(-(-part_granules // PREP_THREADS)):
                    k = p + PREP_THREADS * j
                    if k < part_granules:
                        copy_g2s(at(convs, part, per * k), base + per * k)
        else:
            row_granules: cutlass.Constexpr = K // per
            if p < row_granules:
                for part in cutlass.range_constexpr(3):
                    for tap in cutlass.range_constexpr(3):
                        source = at(page, part * dim + head * K + per * p, offset + tap)
                        copy_g2s(at(convs, part, tap * K + per * p), source)

    @cute.jit
    def stage_rows(self, args, xs, gs, fas, p, head, request):
        x, f_a, gate = args[0], args[4], args[11]
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * K
        token = request * T
        k, kind = p % 16, p // 16
        if kind < 5:
            for t in cutlass.range_constexpr(T):
                if kind < 3:
                    source = at(x, token + t, kind * dim + head * K + 8 * k)
                    copy_g2s(at(xs, t, kind, 8 * k), source)
                elif kind == 3:
                    copy_g2s(at(gs, t, 8 * k), at(gate, token + t, head * K + 8 * k))
                else:
                    copy_g2s(at(fas, t, 8 * k), at(f_a, token + t, 8 * k))

    @cute.jit
    def stage_records(self, args, kgs, us, p, head, rec):
        """cp.async an item's old records: key / raw gate [RQ, 2K] (BF16) and the
        corrections [RQ, K] (FP32), 32 granules per token each."""
        corrections, key_gates = args[18], args[19]
        RQ: cutlass.Constexpr = corrections.shape[2]
        for j in cutlass.range_constexpr(-(-(RQ * 32) // PREP_THREADS)):
            g = p + PREP_THREADS * j
            if g < RQ * 32:
                t, k = g // 32, g % 32
                copy_g2s(at(kgs, t, 8 * k), at(key_gates, rec, head, t, 8 * k))
                copy_g2s(at(us, t, 4 * k), at(corrections, rec, head, t, 4 * k))

    @cute.jit
    def stage(self, args, staging, p, item):
        convs, xs, gs, fas, _, _, kgs, us = staging
        head, request, slot, rec, n, offset = self.item_info(args, item)
        if slot > 0:
            self.stage_conv(args, convs, p, head, slot, offset)
            self.stage_rows(args, xs, gs, fas, p, head, request)
            if rec >= 0:
                self.stage_records(args, kgs, us, p, head, rec)

    @cute.jit
    def block_sums(self, values, sums, p, count: cutlass.Constexpr):
        lane, warp = p % 32, p // 32
        total, selected = scatter_reduce(values, lane, count)
        if lane % (32 // count) == 0:
            sums[selected, warp] = total

    @staticmethod
    def quad_sum(r):
        return (r[0] + r[2]) + (r[1] + r[3])

    @cute.jit
    def silu(self, value):
        silu = div_full(value, Float32(1) + triton_exp(-value))
        return Float32(BFloat16(silu))

    @cute.jit
    def decay(self, a, raw_g, dt, lower_bound):
        """`a` = exp(A_log)."""
        gate = lower_bound * triton_sigmoid(a * (raw_g + dt))
        return triton_exp(gate)

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, args):
        conv_state, output, corrections = args[3], args[10], args[18]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        RQ: cutlass.Constexpr = corrections.shape[2]
        P2: cutlass.Constexpr = 1 << (pair_count(T) - 1).bit_length()
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = output.shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        # CTA c walks items c, c + ctas, ... (strided: CTAs running side by side
        # take neighbouring items, so different heads and state pages, instead of
        # all taking item j of equal contiguous ranges: with ranges a multiple of
        # the head count every CTA reads the same head's F_B rows and the same
        # page offset at once).
        first, stride = Int32(cta), Int32(self.ctas)
        count = Int32(0)
        if cutlass.const_expr(self.strided):
            count = (Int32(total_items) - first + stride - 1) // stride
        else:
            first = Int32(total_items * cta // self.ctas)
            count = Int32(total_items * (cta + 1) // self.ctas) - first
            stride = Int32(1)

        smem = cutlass.utils.SmemAllocator()
        page = alloc(smem, Float32, K * K, align=1024)
        conv_type: cutlass.Constexpr = conv_state.element_type
        xv_s = alloc(
            smem, Float32, (4, 2 * T, K)
        )  # decays -> Gamma k, Gamma q; outputs
        kp_s = alloc(smem, Float32, (4, RQ, K))
        dn_s = alloc(smem, Float32, (4, K))
        u_s = alloc(smem, Float32, (4, RQ, K))
        v_s = alloc(smem, Float32, (4, T, K))
        g_s = alloc(smem, BFloat16, (4, T, K))
        beta_s = alloc(smem, Float32, (4, 8))
        pr_s = alloc(smem, Float32, (4, P2))
        meta = alloc(smem, Int32, (4, META))
        if cutlass.const_expr(self.conv_dim_first):
            convs = alloc(smem, conv_type, (2, 3, K * self.window), align=16)
        else:
            convs = alloc(smem, conv_type, (2, 3, 3 * K), align=16)
        xs = alloc(smem, BFloat16, (2, T, 3, K), align=16)
        gs = alloc(smem, BFloat16, (2, T, K), align=16)
        fas = alloc(smem, BFloat16, (2, T, K), align=16)
        kgs = alloc(smem, BFloat16, (2, RQ, 2 * K), align=16)
        us = alloc(smem, Float32, (2, RQ, K), align=16)
        sums = alloc(smem, Float32, (2, 32, 4))
        psums = alloc(smem, Float32, (2, 64, 4))
        out_sums = alloc(smem, Float32, (2, 8, 4))
        dots = alloc(smem, Float32, (K, 2 * T + 1))
        bars = smem.allocate(cutlass.Uint64, 12)
        state_full, state_empty, ops_full, done = bars, bars + 2, bars + 4, bars + 8

        if tid == 0:
            cute.arch.mbarrier_init(state_full, 1)
            cute.arch.mbarrier_init(state_empty, GROUP_THREADS)
            for blk in cutlass.range_constexpr(4):
                cute.arch.mbarrier_init(ops_full + blk, PREP_THREADS)
                cute.arch.mbarrier_init(done + blk, GROUP_THREADS)
            cute.arch.mbarrier_init_fence()
        cute.arch.barrier()
        cute.arch.griddepcontrol_wait()
        cute.arch.griddepcontrol_launch_dependents()
        ops = (xv_s, kp_s, dn_s, u_s, v_s, g_s, beta_s, pr_s, meta)

        if tid >= GROUP_THREADS:
            set_max_registers(self.registers[1], increase=False)
            w = (tid - GROUP_THREADS) // PREP_THREADS
            p = (tid - GROUP_THREADS) % PREP_THREADS
            staging = (
                convs[w, None, None], xs[w, None, None, None], gs[w, None, None],
                fas[w, None, None], sums[w, None, None], psums[w, None, None],
                kgs[w, None, None], us[w, None, None],
            )  # fmt: skip
            self.prep(args, w, p, first, stride, count, staging, ops, ops_full, done)
        else:
            set_max_registers(self.registers[0], increase=True)
            self.scan(args, tid, first, stride, count, page, ops, dots, out_sums,
                      state_full, state_empty, ops_full, done)  # fmt: skip

    @cute.jit
    def load_page(self, args, page, item, state_full):
        head, request = item % self.heads, item // self.heads
        slot = args[8][request, 0]
        if slot > 0:
            cute.arch.mbarrier_arrive_and_expect_tx(state_full, PAGE_BYTES)
            source = args[9][Int64(slot), None, None, None]
            for i in cutlass.range_constexpr(PAGE_BYTES // COPY_BYTES):
                rows: cutlass.Constexpr = COPY_BYTES // (K * 4)
                copy_g2s(
                    page.iterator + i * rows * K,
                    at(source, head, i * rows, 0),
                    COPY_BYTES,
                    state_full,
                )
        else:
            cute.arch.mbarrier_arrive_and_expect_tx(state_full, 0)

    # -------------------------------------------------------------- scan role
    @cute.jit
    def scan(self, args, t_g, first, stride, count, page, ops, dots, out_sums, state_full,  # noqa: E501
             state_empty, ops_full, done):  # fmt: skip
        xv_s, kp_s, dn_s, u_s, v_s, g_s, beta_s, pr_s, meta = ops
        state, corrections = args[9], args[18]
        T: cutlass.Constexpr = self.tokens
        RQ: cutlass.Constexpr = corrections.shape[2]
        lane_in, base_row = t_g % LANES, t_g // LANES
        nw = args[12][t_g % K]
        if t_g == 0:
            self.load_page(args, page, first, state_full)
        for j in cutlass.range(count, unroll=1):
            b = 2 * (j % 2) + (j // 2) % 2  # operand block
            following = first + (j + 1) * stride
            wait(state_full, j % 2, scope="cta")
            history = cute.make_rmem_tensor((ROWS, CHUNKS, 4), Float32)
            for r in cutlass.range_constexpr(ROWS):
                for c in cutlass.range_constexpr(CHUNKS):
                    row = base_row + 32 * r
                    values = load_shared_f32x4(
                        page.iterator + row * K + 4 * (lane_in + LANES * c)
                    )
                    for e in cutlass.range_constexpr(4):
                        history[r, c, e] = values[e]
            cute.arch.mbarrier_arrive(state_empty)
            if t_g == 0:
                wait(state_empty, j % 2, scope="cta")
                if j + 1 < count:
                    self.load_page(args, page, following, state_full)
            wait(ops_full + b, (j // 4) % 2, scope="cta")
            if meta[b, 0] != 0:
                head, token, n = meta[b, 2], meta[b, 3], meta[b, 4]
                rec, slot = meta[b, 5], meta[b, 6]
                destination = state[Int64(slot), None, None, None]
                # Replay: S_n = S_0 D_n + sum_t u_t (kn_t P_t), stored when n > 0.
                u = cute.make_rmem_tensor((ROWS, RQ), Float32)
                for r in cutlass.range_constexpr(ROWS):
                    for t in cutlass.range_constexpr(RQ):
                        u[r, t] = u_s[b, t, base_row + 32 * r]
                for c in cutlass.range_constexpr(CHUNKS):
                    key = 4 * (lane_in + LANES * c)
                    d4 = load_shared_f32x4(at(dn_s, b, key))
                    final = zeros((ROWS, 4), Float32)
                    for t in cutlass.range_constexpr(RQ):
                        k4 = load_shared_f32x4(at(kp_s, b, t, key))
                        for r in cutlass.range_constexpr(ROWS):
                            for pp in cutlass.range_constexpr(2):
                                pair = fma2(
                                    (u[r, t], u[r, t]),
                                    (k4[2 * pp], k4[2 * pp + 1]),
                                    (final[r, 2 * pp], final[r, 2 * pp + 1]),
                                )
                                final[r, 2 * pp], final[r, 2 * pp + 1] = pair
                    for r in cutlass.range_constexpr(ROWS):
                        for pp in cutlass.range_constexpr(2):
                            pair = fma2(
                                (history[r, c, 2 * pp], history[r, c, 2 * pp + 1]),
                                (d4[2 * pp], d4[2 * pp + 1]),
                                (final[r, 2 * pp], final[r, 2 * pp + 1]),
                            )
                            history[r, c, 2 * pp], history[r, c, 2 * pp + 1] = pair
                        store_f32x4(
                            at(destination, head, base_row + 32 * r, key),
                            [history[r, c, e] for e in range(4)],
                            n > 0,
                        )
                # The 2q dot products of each row, two rows at a time.
                for half in cutlass.range_constexpr(2):
                    acc = zeros((2, 2 * T, 2), Float32)
                    for c in cutlass.range_constexpr(CHUNKS):
                        key = 4 * (lane_in + LANES * c)
                        for jj in cutlass.range_constexpr(2 * T):
                            x4 = load_shared_f32x4(at(xv_s, b, jj, key))
                            for rr in cutlass.range_constexpr(2):
                                r = 2 * half + rr
                                for pp in cutlass.range_constexpr(2):
                                    pair = fma2(
                                        (
                                            history[r, c, 2 * pp],
                                            history[r, c, 2 * pp + 1],
                                        ),
                                        (x4[2 * pp], x4[2 * pp + 1]),
                                        (acc[rr, jj, 0], acc[rr, jj, 1]),
                                    )
                                    acc[rr, jj, 0], acc[rr, jj, 1] = pair
                    for rr in cutlass.range_constexpr(2):
                        red = [Float32(0) for _ in range(16)]
                        for jj in cutlass.range_constexpr(2 * T):
                            red[jj] = acc[rr, jj, 0] + acc[rr, jj, 1]
                        selected = Int32(0)
                        size = 16
                        for offset in (4, 2, 1):
                            size //= 2
                            upper = (lane_in & offset) != 0
                            for i in cutlass.range_constexpr(size):
                                send = red[i] if upper else red[i + size]
                                keep = red[i + size] if upper else red[i]
                                red[i] = keep + cute.arch.shuffle_sync_bfly(
                                    send, offset
                                )
                            if upper:
                                selected += size
                        row = base_row + 32 * (2 * half + rr)
                        for i in cutlass.range_constexpr(2):
                            if selected + i < 2 * T:
                                dots[row, selected + i] = red[i]
                sync_threads(3, GROUP_THREADS)
                # Per row (thread t_g < 128): the q x q substitution and the outputs
                # (into the block's chunk vectors, read by now).
                if t_g < K:
                    us_new = zeros(T, Float32)
                    for t in cutlass.range_constexpr(T):
                        total = dots[t_g, t]
                        for s in cutlass.range_constexpr(t):
                            total = fma(pr_s[b, a_index(t, s)], us_new[s], total)
                        us_new[t] = (v_s[b, t, t_g] - total) * beta_s[b, t]
                    for t in cutlass.range_constexpr(T):
                        o = dots[t_g, T + t]
                        for s in cutlass.range_constexpr(t + 1):
                            o = fma(pr_s[b, b_index(T, t, s)], us_new[s], o)
                        xv_s[b, t, t_g] = Float32(BFloat16(o))
                        if rec >= 0:
                            corrections[rec, head, t, t_g] = us_new[t]
                self.epilogue(args, t_g, j, b, head, token, xv_s, g_s,
                              out_sums[j % 2, None, None], nw)  # fmt: skip
            cute.arch.mbarrier_arrive(done + b)

    # -------------------------------------------------------------- prep role
    @cute.jit
    def prep(self, args, w, p, first, stride, count, staging, ops, ops_full, done):
        x, weights, bias, conv_state, f_a, beta, a_log, dt_bias, slots = args[:9]
        state, output, gate, norm_weight, lower_bound, scale, eps, f_b = args[9:17]
        key_gates = args[19]
        xv_s, kp_s, dn_s, u_s, v_s, g_s, beta_s, pr_s, meta = ops
        convs, xs, gs, fas, sums, psums, kgs, us = staging
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        RQ: cutlass.Constexpr = args[18].shape[2]
        NP: cutlass.Constexpr = pair_count(T)
        P2: cutlass.Constexpr = 1 << (NP - 1).bit_length()
        NS: cutlass.Constexpr = 1 << (2 * T + RQ - 1).bit_length()  # norms
        dim: cutlass.Constexpr = H * K
        lane, warp = p % 32, p // 32
        group_id, quarter = lane >> 2, lane & 3
        c = p
        bar = 1 + w
        conv_type: cutlass.Constexpr = conv_state.element_type

        start = first + w * stride
        if w >= count:
            start = first
        if w < count:
            self.stage(args, staging, p, start)
        cute.arch.cp_async_commit_group()

        for j in cutlass.range(w, count, 2, unroll=1):
            item = first + j * stride
            head, request, slot, rec, n, offset = self.item_info(args, item)
            token = request * T
            blk = 2 * w + (j // 2) % 2
            cute.arch.cp_async_wait_group(0)
            sync_threads(bar, PREP_THREADS)
            if j >= 4:
                wait(done + blk, ((j - 4) // 4) % 2, scope="cta")
            if slot > 0:
                wr = cute.make_rmem_tensor((8, 4), Uint32)
                for chunk in cutlass.range_constexpr(8):
                    feature = head * K + 32 * warp + group_id
                    col = 16 * chunk + 4 * quarter
                    low = load_weight_bf16x4(at(f_b, feature, col))
                    high = load_weight_bf16x4(at(f_b, feature + 8, col))
                    wr[chunk, 0], wr[chunk, 2] = low
                    wr[chunk, 1], wr[chunk, 3] = high
                beta_value = Float32(0)
                if p < T:
                    beta_value = Float32(beta[token + p, head])
                a = triton_exp(Float32(a_log[head]))

                # Convolution + SiLU of q, k, v (thread c: channel c of each part).
                values = cute.make_rmem_tensor((3, T), Float32)
                conv_page = conv_state[Int64(slot), None, None]
                for part in cutlass.range_constexpr(3):
                    channel = part * dim + head * K + c
                    taps = cute.make_rmem_tensor(3, conv_type)
                    for t in cutlass.range_constexpr(3):
                        if cutlass.const_expr(self.conv_dim_first):
                            taps[t] = convs[part, c * self.window + offset + t]
                        else:
                            taps[t] = convs[part, t * K + c]
                    coeff = [Float32(weights[part, i, head * K + c]) for i in range(4)]
                    inputs = [xs[t, part, c] for t in range(T)]
                    for t in cutlass.range_constexpr(T):
                        accum = Float32(0)
                        if cutlass.const_expr(self.has_bias):
                            accum = Float32(bias[channel])
                        for i in cutlass.range_constexpr(4):
                            tap = t + i
                            term = taps[tap] if tap < 3 else inputs[tap - 3]
                            accum = fma(Float32(term), coeff[i], accum)
                        values[part, t] = self.silu(accum)
                    if cutlass.const_expr(self.conv_dim_first):
                        row = c * self.window
                        convs[part, row] = conv_type(taps[1])
                        convs[part, row + 1] = conv_type(taps[2])
                        for t in cutlass.range_constexpr(T):
                            convs[part, row + 2 + t] = conv_type(inputs[t])
                    else:
                        conv_page[channel, 0] = conv_type(taps[1])
                        conv_page[channel, 1] = conv_type(taps[2])
                        for t in cutlass.range_constexpr(T):
                            conv_page[channel, 2 + t] = conv_type(inputs[t])
                if cutlass.const_expr(self.conv_dim_first):
                    fence_async_shared()
                for t in cutlass.range_constexpr(T):
                    v_s[blk, t, c] = values[2, t]
                    g_s[blk, t, c] = gs[t, c]
                # The old records (staged) are read; the new key records go out.
                old_keys = cute.make_rmem_tensor(RQ, Float32)
                old_gates = cute.make_rmem_tensor(RQ, Float32)
                for t in cutlass.range_constexpr(RQ):
                    live_t = t < n
                    old_keys[t] = Float32(0)
                    old_gates[t] = Float32(0)
                    u_old = Float32(0)
                    if live_t:
                        old_keys[t] = Float32(kgs[t, c])
                        old_gates[t] = Float32(kgs[t, K + c])
                        u_old = us[t, c]
                    u_s[blk, t, c] = u_old
                if rec >= 0:
                    for t in cutlass.range_constexpr(T):
                        key_gates[rec, head, t, c] = BFloat16(values[1, t])

                # q, k and old key norms (one reduction).
                squares = [Float32(0) for _ in range(NS)]
                for t in cutlass.range_constexpr(T):
                    squares[2 * t] = values[0, t] * values[0, t]
                    squares[2 * t + 1] = values[1, t] * values[1, t]
                for t in cutlass.range_constexpr(RQ):
                    squares[2 * T + t] = old_keys[t] * old_keys[t]
                self.block_sums(squares, sums, p, NS)

                # Gate projection: the decays (into the chunk vector rows) and raw
                # gates (records).
                for m in cutlass.range_constexpr(2):
                    if cutlass.const_expr(m == 1):
                        for chunk in cutlass.range_constexpr(8):
                            feature = head * K + 32 * warp + 16 + group_id
                            col = 16 * chunk + 4 * quarter
                            low = load_weight_bf16x4(at(f_b, feature, col))
                            high = load_weight_bf16x4(at(f_b, feature + 8, col))
                            wr[chunk, 0], wr[chunk, 2] = low
                            wr[chunk, 1], wr[chunk, 3] = high
                    chains: cutlass.Constexpr = 4 if self.gate_mode == 1 else 1
                    acc = zeros((chains, 4), Float32)
                    for chunk in cutlass.range_constexpr(8):
                        chain: cutlass.Constexpr = (
                            chunk // 2 if self.gate_mode == 1 else 0
                        )
                        wa = tuple(wr[chunk, e] for e in range(4))
                        ab = load_shared_u32x2(
                            at(fas, group_id % T, 16 * chunk + 4 * quarter)
                        )
                        mma_into(acc[chain, None], wa, ab, bf16=True)
                    for e in cutlass.range_constexpr(4):
                        t = 2 * quarter + e % 2
                        feature = 32 * warp + 16 * m + group_id + 8 * (e // 2)
                        value = acc[0, e]
                        if cutlass.const_expr(self.gate_mode == 1):
                            value = Float32(0) + (
                                ((value + acc[1, e]) + acc[2, e]) + acc[3, e]
                            )
                        raw_g = Float32(BFloat16(value))
                        if t < T:
                            dt = dt_bias[head * K + feature]
                            xv_s[blk, t, feature] = self.decay(
                                a, raw_g, dt, lower_bound
                            )
                            if rec >= 0:
                                key_gates[rec, head, t, K + feature] = BFloat16(raw_g)
                if p < T:
                    beta_s[blk, p] = triton_sigmoid(beta_value)
                dt_c = Float32(dt_bias[head * K + c])
                old_decays = cute.make_rmem_tensor(RQ, Float32)
                for t in cutlass.range_constexpr(RQ):
                    old_decays[t] = self.decay(a, old_gates[t], dt_c, lower_bound)
                sync_threads(bar, PREP_THREADS)  # norm sums, decays
                if cutlass.const_expr(self.conv_dim_first):  # noqa: SIM102
                    if p == 0:
                        size: cutlass.Constexpr = K * self.window * conv_type.width // 8
                        for part in cutlass.range_constexpr(3):
                            channel0 = part * dim + head * K
                            store_bulk(
                                at(conv_page, channel0, 0), at(convs, part, 0), size
                            )

                # Normalized q, k; the chunk vectors (over this channel's decays).
                ds = cute.make_rmem_tensor(T, Float32)
                for t in cutlass.range_constexpr(T):
                    ds[t] = xv_s[blk, t, c]
                qs = cute.make_rmem_tensor(T, Float32)
                ks = cute.make_rmem_tensor(T, Float32)
                gamma = Float32(1)
                for t in cutlass.range_constexpr(T):
                    qn = self.quad_sum(sums[2 * t, None]) + 1.0e-6
                    kn = self.quad_sum(sums[2 * t + 1, None]) + 1.0e-6
                    fq = reciprocal(sqrt_approx(qn))
                    fk = reciprocal(sqrt_approx(kn))
                    qs[t] = mul_rn(mul_rn(values[0, t], fq), scale)
                    ks[t] = mul_rn(values[1, t], fk)
                    gamma = gamma * ds[t]
                    xv_s[blk, t, c] = gamma * ks[t]
                    xv_s[blk, T + t, c] = gamma * qs[t]
                # The replay keys (zero past n) and D_n.
                fd = Float32(1)
                for r in cutlass.range_constexpr(RQ):
                    t = RQ - 1 - r
                    weighted = Float32(0)
                    if t < n:
                        total = self.quad_sum(sums[2 * T + t, None])
                        kn_old = old_keys[t] * cute.math.rsqrt(total + Float32(1e-6))
                        weighted = kn_old * fd
                        fd = fd * old_decays[t]
                    kp_s[blk, t, c] = weighted
                dn_s[blk, c] = fd
                # The pair products, 32 at a time.
                for g in cutlass.range_constexpr(-(-P2 // 32)):
                    size_g: cutlass.Constexpr = min(32, P2)
                    products = [Float32(0) for _ in range(size_g)]
                    for s in cutlass.range_constexpr(T):
                        p_st = Float32(1)
                        for t in cutlass.range_constexpr(s, T):
                            if cutlass.const_expr(t > s):
                                p_st = p_st * ds[t]
                            weighted = ks[s] * p_st
                            if cutlass.const_expr(t > s):
                                idx = a_index(t, s)
                                if cutlass.const_expr(32 * g <= idx < 32 * g + size_g):
                                    products[idx - 32 * g] = weighted * ks[t]
                            idx = b_index(T, t, s)
                            if cutlass.const_expr(32 * g <= idx < 32 * g + size_g):
                                products[idx - 32 * g] = weighted * qs[t]
                    total, selected = scatter_reduce(products, lane, size_g)
                    if lane % (32 // size_g) == 0:
                        psums[32 * g + selected, warp] = total
                sync_threads(bar, PREP_THREADS)
                if p < P2:
                    pr_s[blk, p] = self.quad_sum(psums[p, None])
                if p == 0:
                    meta[blk, 0] = 1
                    meta[blk, 1] = request
                    meta[blk, 2] = head
                    meta[blk, 3] = token
                    meta[blk, 4] = n
                    meta[blk, 5] = rec
                    meta[blk, 6] = slot
            else:
                if p == 0:
                    meta[blk, 0] = 0
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * K + c] = BFloat16(0)
            cute.arch.mbarrier_arrive(ops_full + blk)

            if cutlass.const_expr(self.conv_dim_first):  # noqa: SIM102
                if p == 0:
                    wait_bulk_reads()
            sync_threads(bar, PREP_THREADS)
            if j + 2 < count:
                self.stage(args, staging, p, item + 2 * stride)
            cute.arch.cp_async_commit_group()

    @cute.jit
    def epilogue(self, args, t_g, j, b, head, token, o_s, g_s, out_sums, nw):
        output, eps = args[10], args[15]
        T: cutlass.Constexpr = self.tokens
        lane, warp = t_g % 32, t_g // 32
        c, half = t_g % K, t_g // K
        sync_threads(3, GROUP_THREADS)
        for t in cutlass.range_constexpr(T):
            if t % 2 == half:
                o = o_s[b, t, c]
                total = warp_sum(o * o)
                if lane == 0:
                    out_sums[t, warp % 4] = total
        sync_threads(3, GROUP_THREADS)
        for t in cutlass.range_constexpr(T):
            if t % 2 == half:
                sumsq = self.quad_sum(out_sums[t, None])
                variance = sumsq * (1.0 / 128) + eps
                rstd = reciprocal(sqrt_approx(variance))
                sigmoid_gate = triton_sigmoid(Float32(g_s[b, t, c]))
                result = o_s[b, t, c] * rstd * nw * sigmoid_gate
                output[token + t, head * K + c] = BFloat16(result)
