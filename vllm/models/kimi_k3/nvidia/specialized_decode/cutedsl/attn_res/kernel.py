# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Seven-CTA cluster AttnRes: one cluster per token, 1024 hidden columns per CTA.

The delta is a tensor (read after the PDL wait) or `slots` Lamport mailbox slots
summed in slot order, rounded to BF16 and re-armed. The final boundary rounds the
mixture to BF16 before its RMSNorm. Prefix and saved blocks must be complete at launch.
With `precomputed` the saved-block statistics (`statistics`: logits, softmax maximum
and exponent sum, FP32 mixture) come in as arguments 9-10, stored by an earlier kernel
with the same arithmetic (`oproj.consumer`), so the kernel starts at the delta.

The statistics take the stock kernel's forms (attn_res_kernel.cu, --use_fast_math):
the logit's and the fused output norm's reciprocal RMS, and at the final boundary
the mixture as acc * rcp(denominator), followed by the backbone norm as vllm_c's
rms_norm.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32, Uint64

from ..common.mailbox import poll, rearm
from ..common.primitives import (
    alloc,
    at,
    attn_res_rstd,
    bf16,
    bf16x2,
    dot,
    fma,
    group_barrier,
    launch,
    load_f32x4,
    load_words,
    push_float,
    rcp_approx,
    rms_norm_rstd,
    scatter_reduce,
    store_words,
    wait,
    zeros,
)


@cute.jit
def round_pair(t, row: cutlass.Constexpr, j: cutlass.Constexpr):
    word = bf16x2(t[row, 2 * j], t[row, 2 * j + 1])
    t[row, 2 * j], t[row, 2 * j + 1] = bf16(word, 0), bf16(word, 1)


@cute.jit
def unpack(words, out, row: cutlass.Constexpr):
    for i in cutlass.range_constexpr(8):
        out[row, i] = bf16(words[i // 2], i % 2)


def block_ptr(blocks, token, block, col):
    """`&blocks[token, block, col]` from the tensor's strides."""
    return blocks.iterator + token * blocks.stride[0] + block * blocks.stride[1] + col


@cute.jit
def dot8(a, ar: cutlass.Constexpr, b, br: cutlass.Constexpr):
    return dot((a[ar, i] for i in range(8)), (b[br, i] for i in range(8)), Float32(0))


@cute.jit
def arm_exchanges(bars, counts):
    """Thread 0 arms barrier `r` for `counts[r]` FP32 pushes from each of 7 CTAs."""
    if cute.arch.thread_idx()[0] == 0:
        for r, count in enumerate(counts):
            if cutlass.const_expr(count > 0):
                cute.arch.mbarrier_init(bars.iterator + r, 1)
                cute.arch.mbarrier_arrive_and_expect_tx(bars.iterator + r, 28 * count)
        cute.arch.mbarrier_init_fence()


class ClusterAttnRes:
    """`final`: the last AttnRes, normalized by the backbone norm. `precomputed`:
    arguments 9-10 are the statistics' FP32 mixture [rows, 7168] and (maximum,
    exponent sum) [rows, 2], read instead of running `pre`."""

    def __init__(
        self, nb, *, slots=1, has_delta=True, write=-1, final=False, precomputed=False
    ):
        self.nb, self.slots, self.has_delta = nb, slots, has_delta
        self.write, self.final = write, final
        self.precomputed = precomputed
        self.npre = 2 * nb
        self.npost = (2 if final else 4) if nb else 1
        self.kpre = 1 << max(self.npre - 1, 0).bit_length()
        self.kpost = 1 << max(self.npost - 1, 0).bit_length()

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (7, args[1].shape[0], 1), 128, 7)

    @cute.jit
    def exchange(
        self,
        values,
        count: cutlass.Constexpr,
        kpad: cutlass.Constexpr,
        bufs,
        r,
        ids,
        group: cutlass.Constexpr[int] = 0,
    ):
        """Cluster-sum `count` per-thread partials into round `r`'s total row;
        `group`: the 128-thread group running it (its named barrier)."""
        wb, peer_buf, total, bars = bufs
        peer_buf, total, bar = peer_buf[r, None], total[r, None], bars.iterator + r
        rank, _, tid = ids
        lane, warp = tid % 32, tid // 32
        reduced, index = scatter_reduce(values, lane, kpad)
        if (lane & (32 // kpad - 1)) == 0 and index < count:
            wb[warp * 16 + index] = reduced
        group_barrier(group)
        if tid < count:
            value = (wb[tid] + wb[16 + tid]) + (wb[32 + tid] + wb[48 + tid])
            for peer in cutlass.range_constexpr(7):
                push_float(peer_buf.iterator + rank * 16 + tid, value, bar, Int32(peer))
        wait(bar)
        if tid < count:
            sums = [Float32(0), Float32(0)]
            for peer in cutlass.range_constexpr(7):
                sums[peer % 2] += peer_buf[peer * 16 + tid]
            total[tid] = sums[0] + sums[1]
        group_barrier(group)

    @cute.kernel
    def kernel(self, args):
        delta, prefix, blocks, norm, query, out_norm, output, eps, out_eps = args[0:9]
        rank, token, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        bufs = self.setup(cutlass.utils.SmemAllocator())
        cute.arch.cluster_arrive_relaxed()
        cute.arch.griddepcontrol_launch_dependents()
        state = self.state()
        rows, ids = (prefix, blocks, norm, query, out_norm), (rank, token, tid)
        if cutlass.const_expr(self.precomputed):
            self.load_statistics(state, args[9], args[10], ids)
            cute.arch.cluster_wait()
            self.post(state, bufs, delta, rows, output, eps, out_eps, ids, args[11:])
        else:
            self.pre(state, bufs, rows, eps, ids)
            self.post(state, bufs, delta, rows, output, eps, out_eps, ids, args[9:])
        cute.arch.griddepcontrol_wait()

    def state(self):
        """Carried from pre() to post(); the rest reloads so a GEMM can run between."""
        return cute.make_rmem_tensor((1, 8), Float32), cute.make_rmem_tensor(2, Float32)

    @cute.jit
    def setup(self, smem):
        warp_buf = alloc(smem, Float32, 64)
        peer_buf = alloc(smem, Float32, (3, 112))
        total = alloc(smem, Float32, (3, 16))
        bars = alloc(smem, Uint64, 3, 8)
        npre = 0 if self.precomputed else self.npre
        arm_exchanges(bars, (npre, self.npost, int(self.final)))
        return warp_buf, peer_buf, total, bars

    @cute.jit
    def pre(self, state, bufs, rows, eps: Float32, ids):
        """Inputs complete at launch, then the saved-block statistics exchange."""
        prefix, blocks, norm, query, out_norm = rows
        rank, token, tid = ids
        # Rows: saved blocks, combined weight, gamma, old prefix, new prefix.
        v = cute.make_rmem_tensor((self.nb + 4, 8), Float32)
        col = (rank * 128 + tid) * 8
        for source in cutlass.range_constexpr(self.nb):
            words = load_words(
                block_ptr(blocks, token, source, col), count=4, mutable=True
            )
            unpack(words, v, source)
        self.weights(v, norm, query, col)
        cute.arch.cluster_wait()
        self.statistics(state, bufs, v, eps, ids)

    @cute.jit
    def statistics(
        self,
        state,
        bufs,
        v,
        eps: Float32,
        ids,
        block=None,
        group: cutlass.Constexpr[int] = 0,
    ):
        """The saved-block statistics (one cluster exchange) into `state`. Block
        `source` is row `source` of `v`, or the (tensor, row) `block(source)`
        returns; row `nb` of `v` is the combined weight. `group`: the 128-thread
        group running it."""
        total = bufs[2]
        mixture, scal = state
        W = self.nb
        maximum, exp_sum = Float32(0), Float32(0)
        mixture.fill(0)
        if cutlass.const_expr(self.nb > 0):
            values = zeros(self.kpre, Float32)
            for source in cutlass.range_constexpr(self.nb):
                if cutlass.const_expr(block is None):
                    t, r = v, source
                else:
                    t, r = block(source)
                values[source] = dot8(t, r, t, r)
                values[self.nb + source] = dot8(t, r, v, W)
            self.exchange(values, self.npre, self.kpre, bufs, 0, ids, group)
            logits = cute.make_rmem_tensor(self.nb, Float32)
            for b in cutlass.range_constexpr(self.nb):
                logits[b] = total[0, self.nb + b] * attn_res_rstd(total[0, b], eps)
            maximum = logits[0]
            for source in cutlass.range_constexpr(1, self.nb):
                maximum = cute.arch.fmax(maximum, logits[source])
            for source in cutlass.range_constexpr(self.nb):
                e = cute.math.exp(logits[source] - maximum, fastmath=True)
                exp_sum += e
                if cutlass.const_expr(block is None):
                    t, r = v, source
                else:
                    t, r = block(source)
                for i in cutlass.range_constexpr(8):
                    mixture[0, i] = fma(e, t[r, i], mixture[0, i])
        scal[0], scal[1] = maximum, exp_sum

    @cute.jit
    def load_statistics(self, state, mix, scal, ids):
        """`statistics` of this thread's columns, stored by an earlier kernel."""
        mixture, sc = state
        rank, token, tid = ids
        col = (rank * 128 + tid) * 8
        for half in cutlass.range_constexpr(2):
            values = load_f32x4(at(mix, token, col + 4 * half), mutable=True)
            for i in cutlass.range_constexpr(4):
                mixture[0, 4 * half + i] = values[i]
        pair = load_words(
            at(scal, token, 0), mutable=True, result_type=Float32, word_type="f32"
        )
        sc[0], sc[1] = pair[0], pair[1]

    @cute.jit
    def weights(self, v, norm, query, col):
        if cutlass.const_expr(self.nb > 0):
            nw = load_words(norm.iterator + col, count=4)
            qw = load_words(query.iterator + col, count=4)
            for i in cutlass.range_constexpr(8):
                v[self.nb, i] = bf16(nw[i // 2], i % 2) * bf16(qw[i // 2], i % 2)

    @cute.jit
    def post(
        self, state, bufs, delta, rows, output, eps: Float32, out_eps: Float32, ids, aux
    ):
        """Add the delta, update prefix (and its `aux` copy) and saved block, write
        the normalized output."""
        prefix, blocks, norm, query, out_norm = rows
        rank, token, tid = ids
        warp_buf, peer_buf, total, bars = bufs
        mixture, scal = state
        maximum, exp_sum = scal[0], scal[1]
        v = cute.make_rmem_tensor((self.nb + 4, 8), Float32)
        col = (rank * 128 + tid) * 8
        W, G, OLD, NEW = self.nb, self.nb + 1, self.nb + 2, self.nb + 3
        self.weights(v, norm, query, col)
        unpack(load_words(out_norm.iterator + col, count=4), v, G)
        if cutlass.const_expr(self.has_delta):
            unpack(load_words(at(prefix, token, col), count=4, mutable=True), v, OLD)
        cur = cute.make_rmem_tensor((1, 8), Float32)
        if cutlass.const_expr(self.slots == 0):
            cute.arch.griddepcontrol_wait()
            unpack(load_words(at(delta, token, col), count=4, mutable=True), cur, 0)
        else:
            cur.fill(0)
            base = delta.iterator + token * 7168 + col
            got = poll(base, delta.shape[1] * 7168, self.slots)
            for slot in cutlass.range_constexpr(self.slots):
                part = cute.make_rmem_tensor((1, 8), Float32)
                unpack(tuple(got[slot, k] for k in range(4)), part, 0)
                for i in cutlass.range_constexpr(8):
                    cur[0, i] = cur[0, i] + part[0, i]
            if cutlass.const_expr(self.slots > 1):
                for j in cutlass.range_constexpr(4):  # all-reduce output boundary
                    round_pair(cur, 0, j)
        words = cute.make_rmem_tensor(4, Uint32)
        for j in cutlass.range_constexpr(4):
            lo, hi = cur[0, 2 * j], cur[0, 2 * j + 1]
            if cutlass.const_expr(self.has_delta):
                lo, hi = v[OLD, 2 * j] + lo, v[OLD, 2 * j + 1] + hi
            words[j] = bf16x2(lo, hi)
        unpack(words, v, NEW)
        values = zeros(self.kpost, Float32)
        values[0] = dot8(v, NEW, v, NEW)
        if cutlass.const_expr(self.nb > 0):
            values[1] = dot8(v, NEW, v, W)
            if cutlass.const_expr(not self.final):
                values[2] = dot8(mixture, 0, v, NEW)
                values[3] = dot8(mixture, 0, mixture, 0)
        self.exchange(values, self.npost, self.kpost, bufs, 1, ids)
        row = (words[0], words[1], words[2], words[3])
        store_words(at(prefix, token, col), row)
        for stream_out in aux:
            store_words(at(stream_out, token, col), row)
        if cutlass.const_expr(self.write >= 0):
            store_words(block_ptr(blocks, token, self.write, col), row)
        result = cute.make_rmem_tensor((1, 8), Float32)
        if cutlass.const_expr(self.nb == 0):
            scale = attn_res_rstd(total[1, 0], out_eps)
            for i in cutlass.range_constexpr(8):
                result[0, i] = v[NEW, i] * scale * v[G, i]
        else:
            ss, dot = total[1, 0], total[1, 1]
            logit = dot * attn_res_rstd(ss, eps)
            top = cute.arch.fmax(maximum, logit)
            alpha = cute.math.exp(maximum - top, fastmath=True)
            ep = cute.math.exp(logit - top, fastmath=True)
            denominator = fma(alpha, exp_sum, ep)
            if cutlass.const_expr(self.final):
                # As the stock final AttnRes (no output norm): the rescaled sum
                # acc = alpha * mixture + ep * new, then bf16(acc * (1.f / s)).
                inverse = rcp_approx(denominator)
                for j in cutlass.range_constexpr(4):
                    for i in (2 * j, 2 * j + 1):
                        acc = fma(ep, v[NEW, i], alpha * mixture[0, i])
                        result[0, i] = acc * inverse
                    round_pair(result, 0, j)
                fin = cute.make_rmem_tensor(1, Float32)
                fin[0] = dot8(result, 0, result, 0)
                self.exchange(fin, 1, 1, bufs, 2, ids)
                scale = rms_norm_rstd(total[2, 0], out_eps)
                for i in cutlass.range_constexpr(8):
                    result[0, i] = result[0, i] * scale * v[G, i]
            else:
                a2, t3 = alpha * alpha, total[1, 3]
                ae2 = Float32(2) * alpha * ep
                quad = fma(a2, t3, fma(ae2, total[1, 2], ep * ep * ss))
                # The stock fused output norm: rsqrt(acc^2 / 7168 + eps * s * s).
                scale = attn_res_rstd(quad, out_eps, denominator)
                ca, cp = alpha * scale, ep * scale
                for i in cutlass.range_constexpr(8):
                    result[0, i] = fma(ca, mixture[0, i], cp * v[NEW, i]) * v[G, i]
        for j in cutlass.range_constexpr(4):
            words[j] = bf16x2(result[0, 2 * j], result[0, 2 * j + 1])
        store_words(at(output, token, col), tuple(words[j] for j in range(4)))
        if cutlass.const_expr(self.slots > 0):
            # Re-arm the mailbox fragments only now: sentinel stores between the
            # Lamport poll and the exchange slowed the CTA's other pollers.
            base = delta.iterator + token * 7168 + col
            for slot in cutlass.range_constexpr(self.slots):
                rearm(base + slot * delta.shape[1] * 7168)
