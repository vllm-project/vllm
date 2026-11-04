# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent warp-specialized KDA core for wide steps (KdaWS).

One CTA per SM walks a contiguous range of (request, head) items with two roles:

- the **scan group** (8 warps) holds one item's [128, 128] FP32 state in registers
  (thread (g, l): rows g + 32 r, r < 4, keys 4 (l + 8 c) + [0, 4), c < 4) and runs
  the recurrence of the CTA's items one after another;
- two **prep warpgroups** (4 warps each, thread c = channel c; warpgroup w takes
  items w, w + 2, ...) stream each item in: the state page by bulk copies into the
  shared-memory page (once the scan holds the previous item in registers), the
  projection rows and the convolution window by cp.async; they run the convolution,
  SiLU, q / k normalization, the F_B gate projection (mma.sync) and the decays into
  one of their two operand blocks; the scan group then runs the item's gated
  RMSNorm and output from its recurrence outputs.

So the per-item input and output phases (half the instructions of KdaWide) run on
their own warps, overlapped with the recurrence, and the scan warps only scan
(``setmaxnreg``: 168 registers for scan threads, 88 for prep).

Per item the math is KdaWide's: the stock Triton formulas and instructions, BF16
roundings of the convolution outputs, raw gates and recurrence outputs, FP32 state;
only FP32 reduction order differs (the recurrence's dot products and the q / k / out
norms are summed in another order).

Synchronization (mbarriers, phase = use count & 1 of each barrier):
  state_full     prep -> scan (bulk copy bytes)
  state_empty    scan -> scan thread 0 (item j is in registers: it bulk-copies item
                 j + 1's page)
  ops_full[blk]  prep -> scan (operand block blk = 2 w + (j // 2) % 2 ready)
  done[blk]      scan -> prep (item scanned: the operand block is free)
"""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32

from ..common.primitives import (
    alloc,
    asm,
    at,
    copy_g2s,
    fma,
    fma2,
    launch,
    load_words,
    mma_into,
    mul2,
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
    record_row,
    sqrt_approx,
    triton_exp,
    triton_sigmoid,
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
COPY_BYTES = 16384  # per bulk copy of the state page
SMEM_LIMIT = 227 * 1024
META = 16  # per operand block: live, request, head, token, targets[T]


def load_shared_f32x4(pointer):
    return load_words(
        pointer,
        count=4,
        shared=True,
        mutable=True,
        result_type=Float32,
        word_type="f32",
    )


def load_shared_u32x2(pointer):
    return load_words(pointer, count=2, shared=True, mutable=True, word_type="u32")


@asm()
def sync_threads(barrier_id, count):
    """bar.sync with a runtime barrier id."""
    return "bar.sync $0, $1;", "r,r,~{memory}", (Int32(barrier_id), Int32(count))


@asm((Float32, Float32), pure=True)
def add2(a, b):
    """Lanewise a + b of FP32 pairs in one packed instruction (IEEE per lane)."""
    code = "{ .reg .b64 a,b,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    code += "add.rn.f32x2 d,a,b; mov.b64 {$0,$1},d; }"
    return code, "=f,=f,f,f,f,f", (*a, *b)


def pair_total(acc):
    """(a0 + b0) + (a1 + b1) of a row's two accumulator pairs, one packed add."""
    low, high = add2((acc[0, 0], acc[0, 1]), (acc[1, 0], acc[1, 1]))
    return low + high


@asm()
def fence_async_shared():
    """Order this thread's shared-memory writes before later bulk copies."""
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm()
def store_bulk(dst, src, size):
    """Shared -> global bulk copy (one bulk group)."""
    code = "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2; "
    code += "cp.async.bulk.commit_group;"
    return code, "l,r,r,~{memory}", (dst, src, Int32(size))


@asm()
def wait_bulk_reads():
    """The issued bulk stores have read their shared-memory sources."""
    return "cp.async.bulk.wait_group.read 0;", "~{memory}", ()


@asm()
def set_max_registers(count, *, increase):
    op = "inc" if increase else "dec"
    return f"setmaxnreg.{op}.sync.aligned.u32 {count};", "", ()


def operand_bytes(tokens):
    """One operand block: q, k, v, decay, out (FP32), the output gate (BF16), beta,
    meta."""
    return 5 * tokens * K * 4 + tokens * K * 2 + 8 * 4 + META * 4


def staging_bytes(tokens, conv_dim_first, window, conv_size):
    rows = 5 * tokens * K * 2  # x (3 parts), gate, F_A
    conv = 3 * K * window * conv_size if conv_dim_first else 9 * K * conv_size
    return rows + conv


def shared_bytes(tokens, conv_dim_first, window, conv_size):
    staging = staging_bytes(tokens, conv_dim_first, window, conv_size)
    sums = 2 * 16 * 4 * 4 + 2 * 8 * 4 * 4
    blocks = 4 * operand_bytes(tokens)
    return PAGE_BYTES + blocks + 2 * staging + sums + 12 * 8 + 1024  # + alignment


class KdaWS:
    """KdaWide's operands and semantics (``tokens`` q, ``record``: RecoverSSM verify,
    ``gate_mode`` the F_B chain order); ``ctas`` persistent CTAs (one per SM)."""

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
        assert 1 <= tokens <= 8 and (tokens > 1 or not record)
        self.heads, self.tokens, self.record = heads, tokens, record
        self.gate_mode, self.has_bias = gate_mode, has_bias
        self.conv_dim_first, self.window, self.ctas = conv_dim_first, window, ctas

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, self.ctas, THREADS, min_blocks_per_mp=1)

    # ---------------------------------------------------------------- helpers
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
    def stage_conv(self, args, convs, p, head, conv_slot, offset):
        """cp.async the item's convolution window (prep thread ``p``)."""
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
        """cp.async the item's projection rows: x (q, k, v), output gate, F_A."""
        x, f_a, gate = args[0], args[4], args[11]
        T: cutlass.Constexpr = self.tokens
        dim: cutlass.Constexpr = self.heads * K
        token = request * T
        k, kind = (
            p % 16,
            p // 16,
        )  # 16-byte granule k of row kind (x q / k / v, gate, F_A)
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
    def block_sums(self, values, sums, p, count: cutlass.Constexpr):
        """Per-value totals over the 128 prep threads (each thread one channel):
        ``values`` (a power-of-two list) -> sums[value, warp]; the caller syncs."""
        lane, warp = p % 32, p // 32
        total, selected = scatter_reduce(values, lane, count)
        if lane % (32 // count) == 0:
            sums[selected, warp] = total

    @staticmethod
    def quad_sum(r):
        return (r[0] + r[2]) + (r[1] + r[3])

    @cute.jit
    def silu(self, value):
        """The convolution's SiLU: x / (1 + exp(-x)), stored BF16."""
        silu = div_full(value, Float32(1) + triton_exp(-value))
        return Float32(BFloat16(silu))

    @cute.jit
    def decay(self, a, raw_g, dt, lower_bound):
        """``a`` = exp(A_log) (tl.exp)."""
        gate = lower_bound * triton_sigmoid(a * (raw_g + dt))
        return triton_exp(gate)

    @cute.jit
    def row_sum(self, partials, lane_in):
        """The four rows' partials of 8 lanes -> the full sum of row lane_in // 2."""
        upper = (lane_in & 4) != 0
        a = [Float32(0), Float32(0)]
        for i in cutlass.range_constexpr(2):
            send = partials[i] if upper else partials[i + 2]
            keep = partials[i + 2] if upper else partials[i]
            a[i] = keep + cute.arch.shuffle_sync_bfly(send, 4)
        upper = (lane_in & 2) != 0
        send = a[0] if upper else a[1]
        keep = a[1] if upper else a[0]
        b = keep + cute.arch.shuffle_sync_bfly(send, 2)
        return b + cute.arch.shuffle_sync_bfly(b, 1)

    # ----------------------------------------------------------------- kernel
    @cute.kernel
    def kernel(self, args):
        conv_state, output = args[3], args[10]
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        requests: cutlass.Constexpr = output.shape[0] // T
        total_items: cutlass.Constexpr = H * requests
        first = Int32(total_items * cta // self.ctas)
        count = Int32(total_items * (cta + 1) // self.ctas) - first

        smem = cutlass.utils.SmemAllocator()
        page = alloc(smem, Float32, K * K, align=1024)
        conv_type: cutlass.Constexpr = conv_state.element_type
        # Per prep warpgroup w: its operand block, staging buffers and sums.
        q_s = alloc(smem, Float32, (4, T, K))
        k_s = alloc(smem, Float32, (4, T, K))
        v_s = alloc(smem, Float32, (4, T, K))
        d_s = alloc(smem, Float32, (4, T, K))
        o_s = alloc(smem, Float32, (4, T, K))
        g_s = alloc(smem, BFloat16, (4, T, K))
        beta_s = alloc(smem, Float32, (4, 8))
        meta = alloc(smem, Int32, (4, META))
        if cutlass.const_expr(self.conv_dim_first):
            convs = alloc(smem, conv_type, (2, 3, K * self.window), align=16)
        else:
            convs = alloc(smem, conv_type, (2, 3, 3 * K), align=16)
        xs = alloc(smem, BFloat16, (2, T, 3, K), align=16)
        gs = alloc(smem, BFloat16, (2, T, K), align=16)
        fas = alloc(smem, BFloat16, (2, T, K), align=16)
        sums = alloc(smem, Float32, (2, 16, 4))
        out_sums = alloc(smem, Float32, (2, 8, 4))  # the scan's, by item parity
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
        ops = (q_s, k_s, v_s, d_s, o_s, g_s, beta_s, meta)

        if tid >= GROUP_THREADS:
            set_max_registers(PREP_REGISTERS, increase=False)
            w = (tid - GROUP_THREADS) // PREP_THREADS
            p = (tid - GROUP_THREADS) % PREP_THREADS
            staging = (
                convs[w, None, None], xs[w, None, None, None], gs[w, None, None],
                fas[w, None, None], sums[w, None, None],
            )  # fmt: skip
            self.prep(args, w, p, first, count, page, staging, ops,
                      state_full, state_empty, ops_full, done)  # fmt: skip
        else:
            set_max_registers(SCAN_REGISTERS, increase=True)
            self.scan(args, tid, first, count, page, ops, out_sums, state_full,
                      state_empty, ops_full, done)  # fmt: skip

    @cute.jit
    def load_page(self, args, page, item, state_full):
        """Bulk-copy an item's state page (or complete the phase of a padded row)."""
        head, _, slot, _, _ = self.item_slots(args, item)
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
    def scan(self, args, t_g, first, count, page, ops, out_sums, state_full,
             state_empty, ops_full, done):  # fmt: skip
        q_s, k_s, v_s, d_s, o_s, g_s, beta_s, meta = ops
        state = args[9]
        T: cutlass.Constexpr = self.tokens
        lane = t_g % 32
        lane_in, base_row = t_g % LANES, t_g // LANES
        own_row = base_row + 32 * (lane_in // 2)  # the row this lane reduces
        nw = args[12][t_g % K]
        if t_g == 0:
            self.load_page(args, page, first, state_full)
        for j in cutlass.range(count, unroll=1):
            b = 2 * (j % 2) + (j // 2) % 2  # operand block
            following = first + j + 1
            if j + 1 >= count:  # a valid item's slots (not loaded)
                following = first + j
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
            if t_g == 0:  # once every thread holds item j: item j + 1's page
                wait(state_empty, j % 2, scope="cta")
                if j + 1 < count:
                    self.load_page(args, page, following, state_full)
            wait(ops_full + b, (j // 4) % 2, scope="cta")
            if meta[b, 0] != 0:
                request, head = meta[b, 1], meta[b, 2]
                for t in cutlass.range_constexpr(T):

                    def chunk_of(buffer, c, t=t):
                        return load_shared_f32x4(
                            at(buffer, b, t, 4 * (lane_in + LANES * c))  # noqa: B023 (called in its iteration)
                        )

                    dots = zeros((ROWS, 2, 2), Float32)
                    k4s = [chunk_of(k_s, c) for c in range(CHUNKS)]  # both passes
                    for c in cutlass.range_constexpr(CHUNKS):
                        d4, k4 = chunk_of(d_s, c), k4s[c]
                        for r in cutlass.range_constexpr(ROWS):
                            for pp in cutlass.range_constexpr(2):
                                pair = (
                                    history[r, c, 2 * pp],
                                    history[r, c, 2 * pp + 1],
                                )
                                pair = mul2(pair, (d4[2 * pp], d4[2 * pp + 1]))
                                history[r, c, 2 * pp], history[r, c, 2 * pp + 1] = pair
                                acc = (dots[r, pp, 0], dots[r, pp, 1])
                                acc = fma2(pair, (k4[2 * pp], k4[2 * pp + 1]), acc)
                                dots[r, pp, 0], dots[r, pp, 1] = acc
                    partials = [pair_total(dots[r, None, None]) for r in range(ROWS)]
                    dot = self.row_sum(partials, lane_in)
                    v = (v_s[b, t, own_row] - dot) * beta_s[b, t]
                    if cutlass.const_expr(self.record):  # noqa: SIM102
                        if lane_in % 2 == 0:
                            args[18][record_row(args, request), head, t, own_row] = v
                    v_new = [
                        cute.arch.shuffle_sync(v, (lane & ~7) | (2 * r))
                        for r in range(ROWS)
                    ]
                    outs = zeros((ROWS, 2, 2), Float32)
                    for c in cutlass.range_constexpr(CHUNKS):
                        k4, q4 = k4s[c], chunk_of(q_s, c)
                        for r in cutlass.range_constexpr(ROWS):
                            for pp in cutlass.range_constexpr(2):
                                pair = (
                                    history[r, c, 2 * pp],
                                    history[r, c, 2 * pp + 1],
                                )
                                kp = (k4[2 * pp], k4[2 * pp + 1])
                                pair = fma2(kp, (v_new[r], v_new[r]), pair)
                                history[r, c, 2 * pp], history[r, c, 2 * pp + 1] = pair
                                acc = (outs[r, pp, 0], outs[r, pp, 1])
                                acc = fma2(pair, (q4[2 * pp], q4[2 * pp + 1]), acc)
                                outs[r, pp, 0], outs[r, pp, 1] = acc
                    partials = [pair_total(outs[r, None, None]) for r in range(ROWS)]
                    o = self.row_sum(partials, lane_in)
                    if lane_in % 2 == 0:
                        o_s[b, t, own_row] = Float32(BFloat16(o))
                    if cutlass.const_expr(not self.record):
                        target = meta[b, 4 + t]
                        if target > 0:
                            destination = state[Int64(target), None, None, None]
                            for r in cutlass.range_constexpr(ROWS):
                                for c in cutlass.range_constexpr(CHUNKS):
                                    row_values = [history[r, c, e] for e in range(4)]
                                    key = 4 * (lane_in + LANES * c)
                                    row = base_row + 32 * r
                                    store_f32x4(
                                        at(destination, head, row, key), row_values
                                    )
                self.epilogue(args, t_g, j, b, head, meta[b, 3], o_s, g_s,
                              out_sums[j % 2, None, None], nw)  # fmt: skip
            cute.arch.mbarrier_arrive(done + b)

    # -------------------------------------------------------------- prep role
    @cute.jit
    def prep(self, args, w, p, first, count, page, staging, ops, state_full,
             state_empty, ops_full, done):  # fmt: skip
        x, weights, bias, conv_state, f_a, beta, a_log, dt_bias, slots = args[:9]
        state, output, gate, norm_weight, lower_bound, scale, eps, f_b = args[9:17]
        q_s, k_s, v_s, d_s, _, g_s, beta_s, meta = ops
        convs, xs, gs, fas, sums = staging
        T: cutlass.Constexpr = self.tokens
        H: cutlass.Constexpr = self.heads
        dim: cutlass.Constexpr = H * K
        P2: cutlass.Constexpr = 1 << (2 * T - 1).bit_length()  # q and k norms
        lane, warp = p % 32, p // 32
        group_id, quarter = lane >> 2, lane & 3
        c = p  # this thread's channel
        bar = 1 + w  # this warpgroup's named barrier
        conv_type: cutlass.Constexpr = conv_state.element_type

        # Prologue: the first item's window and rows.
        start = first + w
        if w >= count:  # read a valid item's slots
            start = first
        head, request, slot, conv_slot, offset = self.item_slots(args, start)
        if w < count:  # noqa: SIM102
            if slot > 0:
                self.stage_conv(args, convs, p, head, conv_slot, offset)
                self.stage_rows(args, xs, gs, fas, p, head, request)
        cute.arch.cp_async_commit_group()

        for j in cutlass.range(w, count, 2, unroll=1):
            item = first + j
            head, request = item % H, item // H
            token = request * T
            blk = 2 * w + (j // 2) % 2
            cute.arch.cp_async_wait_group(0)
            sync_threads(bar, PREP_THREADS)
            if j >= 4:  # the block's previous item (j - 4) is scanned
                wait(done + blk, ((j - 4) // 4) % 2, scope="cta")
            if slot > 0:
                # Early: the F_B rows of tile 0 (L2) and beta.
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
                conv_page = conv_state[Int64(conv_slot), None, None]
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
                    # The window [tap 1, tap 2, x_0, ..., x_q-1]: dim-first, into the
                    # staged block (written back by one bulk copy per part).
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
                    if cutlass.const_expr(self.record):
                        args[19][record_row(args, request), head, t, c] = BFloat16(
                            values[1, t]
                        )

                # q and k norms over the 128 channels (one reduction for all 2T).
                squares = [Float32(0) for _ in range(P2)]
                for t in cutlass.range_constexpr(T):
                    squares[2 * t] = values[0, t] * values[0, t]
                    squares[2 * t + 1] = values[1, t] * values[1, t]
                self.block_sums(squares, sums, p, P2)

                # Gate projection: D[feature, token], tile m of this warp's 32
                # features; the decays and raw gates.
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
                            d_s[blk, t, feature] = self.decay(a, raw_g, dt, lower_bound)
                            if cutlass.const_expr(self.record):
                                args[19][
                                    record_row(args, request), head, t, K + feature
                                ] = BFloat16(raw_g)
                if p < T:
                    beta_s[blk, p] = triton_sigmoid(beta_value)
                sync_threads(bar, PREP_THREADS)  # the norms' partial sums
                if cutlass.const_expr(self.conv_dim_first):  # noqa: SIM102
                    if p == 0:
                        size: cutlass.Constexpr = K * self.window * conv_type.width // 8
                        for part in cutlass.range_constexpr(3):
                            channel0 = part * dim + head * K
                            store_bulk(
                                at(conv_page, channel0, 0), at(convs, part, 0), size
                            )

                for t in cutlass.range_constexpr(T):
                    qn = self.quad_sum(sums[2 * t, None]) + 1.0e-6
                    kn = self.quad_sum(sums[2 * t + 1, None]) + 1.0e-6
                    fq = reciprocal(sqrt_approx(qn))
                    fk = reciprocal(sqrt_approx(kn))
                    q_s[blk, t, c] = mul_rn(mul_rn(values[0, t], fq), scale)
                    k_s[blk, t, c] = mul_rn(values[1, t], fk)
                if p == 0:
                    meta[blk, 0] = 1
                    meta[blk, 1] = request
                    meta[blk, 2] = head
                    meta[blk, 3] = token
                    if cutlass.const_expr(not self.record and T > 1):
                        for t in cutlass.range_constexpr(T):
                            meta[blk, 4 + t] = slots[request, t]
                    if cutlass.const_expr(not self.record and T == 1):
                        meta[blk, 4] = slot
            else:
                if p == 0:
                    meta[blk, 0] = 0
                for t in cutlass.range_constexpr(T):
                    output[token + t, head * K + c] = BFloat16(0)
            cute.arch.mbarrier_arrive(ops_full + blk)

            # This block's next item's window and rows (the staging is free).
            if cutlass.const_expr(self.conv_dim_first):  # noqa: SIM102
                if p == 0:
                    wait_bulk_reads()
            sync_threads(bar, PREP_THREADS)
            if j + 2 < count:
                head, request, slot, conv_slot, offset = self.item_slots(args, item + 2)
                if slot > 0:
                    self.stage_conv(args, convs, p, head, conv_slot, offset)
                    self.stage_rows(args, xs, gs, fas, p, head, request)
            cute.arch.cp_async_commit_group()

    @cute.jit
    def epilogue(self, args, t_g, j, b, head, token, o_s, g_s, out_sums, nw):
        """Gated RMSNorm and output of the scanned item (scan threads: channel
        t_g % 128 of tokens t_g // 128, + 2, ...)."""
        output, eps = args[10], args[15]
        T: cutlass.Constexpr = self.tokens
        lane, warp = t_g % 32, t_g // 32
        c, half = t_g % K, t_g // K  # half: tokens t = half (mod 2)
        sync_threads(3, GROUP_THREADS)  # every row's outputs are in o_s
        for t in cutlass.range_constexpr(T):
            if t % 2 == half:  # warp-uniform
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
