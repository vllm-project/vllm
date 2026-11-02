# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""O-projection all-reduce consumer (``groups`` 128-thread groups poll disjoint
rank slots, group 0 sums in rank order into the AttnRes) and TP16 fused producer +
consumer. With ``twoshot`` the mailbox is the one-slot gather row of
``common.twoshot``: the threads owning this rank's columns first reduce their
receive slots into it (args 11-13: receive view, gather multicast address, rank)."""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Uint32

from ..attn_res.kernel import arm_exchanges, rstd
from ..common.mailbox import rearm
from ..common.primitives import (
    add_rn,
    alloc,
    asm,
    at,
    bits_float,
    copy_g2s,
    float_bits,
    fma,
    launch,
    load_vector,
    named_barrier,
    pack_pair,
    publish_hint,
    published,
    push_float,
    smem_words,
    store_relaxed,
    store_shared_words,
    store_words,
    unpack_bf16,
    zeros,
)
from ..common.primitives import wait_phase_zero_cluster as wait_phase_zero
from ..common.twoshot import reduce
from .producer import OprojProducer, pack8, tree_reduce


@asm((Float32,) * 2, pure=True)
def pair_op(a, b, c=(), *, op="add"):
    values = (*a, *b, *c)
    code = "{ .reg .b64 a,b,c,d; mov.b64 a,{$2,$3}; mov.b64 b,{$4,$5}; "
    if op == "fma":
        code += "mov.b64 c,{$6,$7}; fma.rn.f32x2 d,a,b,c; "
    else:
        code += f"{op}.rn.f32x2 d,a,b; "
    constraints = ",".join(["=f", "=f"] + ["f"] * len(values))
    return code + "mov.b64 {$0,$1},d; }", constraints, tuple(map(Float32, values))


@asm(Float32, pure=True)
def mul_rn(a, b):
    return "mul.rn.f32 $0,$1,$2;", "=f,f,f", (*(Float32(v) for v in (a, b)),)


def two(values, j):
    return values[2 * j], values[2 * j + 1]


@cute.jit
def dot8(a, b):
    s0 = pair_op(two(a, 0), two(b, 0), op="mul")
    s1 = pair_op(two(a, 1), two(b, 1), op="mul")
    s0 = pair_op(two(a, 2), two(b, 2), s0, op="fma")
    s1 = pair_op(two(a, 3), two(b, 3), s1, op="fma")
    s = pair_op(s0, s1)
    return add_rn(s[0], s[1])


@cute.jit
def reduce_scatter(values, size: cutlass.Constexpr[int], lane):
    """Warp-reduce ``size`` values; returns the index of the one this lane holds."""
    for shift in cutlass.range_constexpr(4, -1, -1):
        live: cutlass.Constexpr = max(1, size >> (4 - shift))
        half: cutlass.Constexpr = max(1, live // 2)
        if cutlass.const_expr(live == 1):
            values[0] = add_rn(
                values[0], cute.arch.shuffle_sync_bfly(values[0], 1 << shift)
            )
        else:
            upper = (lane & (1 << shift)) != 0
            for j in cutlass.range_constexpr(half):
                send, keep = values[j + half], values[j]
                if upper:
                    send, keep = values[j], values[j + half]
                values[j] = add_rn(keep, cute.arch.shuffle_sync_bfly(send, 1 << shift))
    idx = cutlass.Int32(0)
    for shift in cutlass.range_constexpr(4, -1, -1):
        width: cutlass.Constexpr = size >> (4 - shift)
        if cutlass.const_expr(width > 1):  # noqa: SIM102
            if (lane & (1 << shift)) != 0:
                idx += width // 2
    return idx


@cute.jit
def fetch(raw, rank, pointer):
    words = load_vector(pointer, cache="volatile")
    for j in cutlass.range_constexpr(4):
        raw[rank, j] = words[j]


class OprojConsumer:
    def __init__(
        self,
        nb,
        *,
        has_delta=True,
        groups=4,
        twoshot=False,
    ):
        if not 0 <= nb <= 8:
            raise ValueError("Output-projection AttnRes supports zero to eight blocks")
        self.nb, self.has_delta = nb, has_delta
        self.groups, self.twoshot = groups, twoshot
        self.kpre = 1 << max(0, (2 * nb - 1).bit_length())
        self.npost = 4 if nb > 0 else 1

    @cute.jit
    def wait_exchange(self, barrier):
        wait_phase_zero(barrier)  # peers complete it by st.async

    @cute.jit
    def allocate(self):
        smem = cutlass.utils.SmemAllocator()
        wpre = alloc(smem, Float32, (4, 16))
        pre = alloc(smem, Float32, (7, 16))
        blk = alloc(smem, Float32, 16)
        post = alloc(smem, Float32, (7, 4, 4))
        total = alloc(smem, Float32, 4)
        bars = alloc(smem, cutlass.Uint64, 2, 8)
        parts = alloc(smem, Float32, (max(1, self.groups - 1), 2, 128, 4))
        lean = self.groups == 1  # above B8: rows for this nb only (co-residency)
        pf = alloc(smem, Uint32, (4 + self.nb if lean else 12, 128, 4))
        return wpre, pre, blk, post, total, bars, parts, pf

    @cute.jit
    def setup(self, bars):
        arm_exchanges(bars, (2 * self.nb, 4 * self.npost))
        cute.arch.cluster_arrive_relaxed()

    @cute.jit
    def prefetch(self, args, pf, rank, token):
        _, prefix, blocks, norm, query, out_norm = args[:6]
        tid, _, _ = cute.arch.thread_idx()
        h = (rank * 128 + tid) * 8
        if cutlass.const_expr(self.nb > 0):
            copy_g2s(pf.iterator + tid * 4, norm.iterator + h)
            copy_g2s(pf.iterator + 512 + tid * 4, query.iterator + h)
        copy_g2s(pf.iterator + 1024 + tid * 4, out_norm.iterator + h)
        if cutlass.const_expr(self.has_delta):
            copy_g2s(pf.iterator + 1536 + tid * 4, at(prefix, token, h))
        for block in cutlass.range_constexpr(self.nb):
            dst = pf.iterator + (4 + block) * 512 + tid * 4
            copy_g2s(dst, at(blocks, token, block, h))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def fragment(self, pf, row, tid):
        words = smem_words(pf.iterator + (row * 128 + tid) * 4)
        values = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            values[2 * j], values[2 * j + 1] = unpack_bf16(words[j])
        return values

    @cute.jit
    def combined_weight(self, pf, tid):
        a, b = self.fragment(pf, 0, tid), self.fragment(pf, 1, tid)
        result = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            result[2 * j], result[2 * j + 1] = pair_op(two(a, j), two(b, j), op="mul")
        return result

    @cute.jit
    def pre(self, state, rank, eps, mix, wait_groups: cutlass.Constexpr[int] = 0):
        wpre, pre, blk, _, _, bars, _, pf = state
        tid, _, _ = cute.arch.thread_idx()
        warp, lane = tid // 32, tid % 32
        maximum, exponent_sum, qpart = Float32(0), Float32(0), Float32(0)
        cute.arch.cp_async_wait_group(wait_groups)
        if cutlass.const_expr(self.nb > 0):
            weight = self.combined_weight(pf, tid)
            partial = zeros(self.kpre, Float32)
            for block in cutlass.range_constexpr(self.nb):
                value = self.fragment(pf, 4 + block, tid)
                partial[block] = dot8(value, value)
                partial[self.nb + block] = dot8(value, weight)
            idx = reduce_scatter(partial, self.kpre, lane)
            if ((lane & (32 // self.kpre - 1)) == 0) & (idx < 2 * self.nb):
                wpre[warp, idx] = partial[0]
            named_barrier()
            cute.arch.cluster_wait()
            if tid < 2 * self.nb:
                pre_sum = Float32(0)
                for w in cutlass.range_constexpr(4):
                    pre_sum = add_rn(pre_sum, wpre[w, tid])
                for peer in cutlass.range_constexpr(7):
                    push_float(at(pre, rank, tid), pre_sum, bars.iterator, peer)
            self.wait_exchange(bars.iterator)
            if tid < 2 * self.nb:
                sums = [Float32(0), Float32(0)]
                for peer in cutlass.range_constexpr(7):
                    sums[peer % 2] = add_rn(sums[peer % 2], pre[peer, tid])
                blk[tid] = add_rn(*sums)
            named_barrier()
            logits = [
                mul_rn(blk[self.nb + b], rstd(blk[b], eps)) for b in range(self.nb)
            ]
            maximum = tree_reduce(logits, max)
            exp = [cute.math.exp(x - maximum, fastmath=True) for x in logits]
            exponent_sum = tree_reduce(exp, add_rn)
            for block in cutlass.range_constexpr(self.nb):
                value = self.fragment(pf, 4 + block, tid)
                for j in cutlass.range_constexpr(4):
                    mix[2 * j], mix[2 * j + 1] = pair_op(
                        (exp[block], exp[block]), two(value, j), two(mix, j), op="fma"
                    )
            qpart = dot8(mix, mix)
        else:
            cute.arch.cluster_wait()
        return maximum, exponent_sum, qpart

    def slots(self, tp):
        """Rank slots each group polls (at least four)."""
        return max(4, (tp + self.groups - 1) // self.groups)

    @cute.jit
    def poll_sum(self, mailbox, token, h, first, count):
        n: cutlass.Constexpr = self.slots(mailbox.shape[0])
        raw = zeros((n, 4), Uint32)
        pending = Int32(0)
        for rank in cutlass.range_constexpr(n):
            if rank < count:
                fetch(raw, rank, at(mailbox, first + rank, token, h))
        for rank in cutlass.range_constexpr(n):
            if rank < count:  # noqa: SIM102
                if not published(raw[rank, j] for j in range(4)):
                    pending |= 1 << rank
        while pending != 0:
            for rank in cutlass.range_constexpr(n):
                if (pending & (1 << rank)) != 0:
                    fetch(raw, rank, at(mailbox, first + rank, token, h))
            for rank in cutlass.range_constexpr(n):
                if (pending & (1 << rank)) != 0:  # noqa: SIM102
                    if published(raw[rank, j] for j in range(4)):
                        pending &= ~(1 << rank)
        values = cute.make_rmem_tensor(8, Float32)
        for j in cutlass.range_constexpr(4):
            pair = unpack_bf16(raw[0, j])
            for rank in cutlass.range_constexpr(1, n):
                if rank < count:
                    pair = pair_op(pair, unpack_bf16(raw[rank, j]))
            values[2 * j], values[2 * j + 1] = pair[0], pair[1]
        return values

    @cute.jit
    def publish(self, output, xmoe, hints, token, h, values):
        tid, _, _ = cute.arch.thread_idx()
        words = pack8(values)
        store_words(at(output, token, h), words)
        if cutlass.const_expr(xmoe is not None):
            words = pack8(values, "latent")
            for rep in cutlass.range_constexpr(xmoe.shape[0]):
                store_relaxed(at(xmoe, rep, token, h), words, scope="gpu")
            cute.arch.sync_warp()
            if tid % 32 == 0:
                publish_hint(hints.iterator)

    @cute.jit
    def post(self, args, state, block_mix, rank, token):
        mailbox, prefix, _, _, _, _, output, xmoe, hints, eps, out_eps = args[:11]
        _, _, _, post, total, bars, parts, pf = state
        mix, maximum, exponent_sum, qpart = block_mix
        thread, _, _ = cute.arch.thread_idx()
        group, tid = thread // 128, thread % 128
        warp, lane = tid // 32, tid % 32
        h = (rank * 128 + tid) * 8
        S: cutlass.Constexpr = (mailbox.shape[0] + self.groups - 1) // self.groups
        everyone: cutlass.Constexpr = 128 * self.groups
        if group > 0:
            first = group * S
            count = min(S, mailbox.shape[0] - first)
            values = self.poll_sum(mailbox, token, h, first, count)
            for half in cutlass.range_constexpr(2):
                words = tuple(float_bits(values[half * 4 + i]) for i in range(4))
                store_shared_words(at(parts, group - 1, half, tid, 0), words)
            named_barrier(2, everyone, arrive=True)
            for r in cutlass.range_constexpr(self.slots(mailbox.shape[0])):
                if r < count:
                    rearm(at(mailbox, first + r, token, h))
        else:
            if cutlass.const_expr(self.twoshot):
                recv, gather_mc, me = args[11:14]
                if h // recv.shape[2] == me:
                    reduce(recv, mailbox, gather_mc, me, token, h)
            if cutlass.const_expr(self.has_delta):
                old = self.fragment(pf, 3, tid)
            attn = self.poll_sum(mailbox, token, h, 0, min(S, mailbox.shape[0]))
            if cutlass.const_expr(self.groups > 1):
                named_barrier(2, everyone)
            for other in cutlass.range_constexpr(self.groups - 1):
                for half in cutlass.range_constexpr(2):
                    words = smem_words(at(parts, other, half, tid, 0))
                    for j in cutlass.range_constexpr(2):
                        k: cutlass.Constexpr = half * 2 + j
                        part = tuple(map(bits_float, two(words, j)))
                        attn[2 * k], attn[2 * k + 1] = pair_op(two(attn, k), part)
            prefix_value = cute.make_rmem_tensor(8, Float32)
            for j in cutlass.range_constexpr(4):
                rounded = unpack_bf16(pack_pair(*two(attn, j)))
                if cutlass.const_expr(self.has_delta):
                    rounded = pair_op(two(old, j), rounded)
                    rounded = unpack_bf16(pack_pair(rounded[0], rounded[1]))
                prefix_value[2 * j], prefix_value[2 * j + 1] = rounded[0], rounded[1]
            partial = cute.make_rmem_tensor(self.npost, Float32)
            if cutlass.const_expr(self.nb > 0):
                weight = self.combined_weight(pf, tid)
                partial[0], partial[1] = dot8(mix, prefix_value), qpart
                partial[2] = dot8(prefix_value, prefix_value)
                partial[3] = dot8(prefix_value, weight)
            else:
                partial[0] = dot8(prefix_value, prefix_value)
            idx = reduce_scatter(partial, self.npost, lane)
            if idx < self.npost:
                for peer in range(lane & (32 // self.npost - 1), 7, 32 // self.npost):
                    dst = at(post, rank, warp, idx)
                    push_float(dst, partial[0], bars.iterator + 1, peer)
            words = pack8(prefix_value)
            store_words(at(prefix, token, h), words)
            for r in cutlass.range_constexpr(min(S, mailbox.shape[0])):
                rearm(at(mailbox, r, token, h))
            if cutlass.const_expr(xmoe is not None):
                cute.arch.fence_acq_rel_gpu()
            self.wait_exchange(bars.iterator + 1)
            if tid < self.npost:
                sums = [Float32(0)] * 4
                for peer in cutlass.range_constexpr(7):
                    for w in cutlass.range_constexpr(4):
                        sums[w] = add_rn(sums[w], post[peer, w, tid])
                total[tid] = tree_reduce(sums, add_rn)
            named_barrier()
            norm = self.fragment(pf, 2, tid)
            final = cute.make_rmem_tensor(8, Float32)
            if cutlass.const_expr(self.nb > 0):
                lp = mul_rn(total[3], rstd(total[2], eps))
                mx = max(maximum, lp)
                alpha = cute.math.exp(maximum - mx, fastmath=True)
                ep = cute.math.exp(lp - mx, fastmath=True)
                den = fma(alpha, exponent_sum, ep)
                a2, t1 = mul_rn(alpha, alpha), total[1]
                ae2 = mul_rn(mul_rn(2.0, alpha), ep)
                quad = fma(a2, t1, fma(ae2, total[0], mul_rn(mul_rn(ep, ep), total[2])))
                scale = rstd(quad, mul_rn(mul_rn(out_eps, den), den))
                ca, cp = mul_rn(alpha, scale), mul_rn(ep, scale)
            else:
                cp = rstd(total[0], out_eps)
            for j in cutlass.range_constexpr(4):
                value = pair_op((cp, cp), two(prefix_value, j), op="mul")
                if cutlass.const_expr(self.nb > 0):
                    value = pair_op((ca, ca), two(mix, j), value, op="fma")
                value = pair_op(value, two(norm, j), op="mul")
                final[2 * j], final[2 * j + 1] = value[0], value[1]
            self.publish(output, xmoe, hints, token, h, final)

    @cute.jit
    def __call__(self, *args):
        launch(self.kernel, args, (7, args[1].shape[0], 1), 128 * self.groups, 7)

    @cute.kernel
    def kernel(self, args):
        rank, token, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        state = self.allocate()
        self.setup(state[5])
        cute.arch.griddepcontrol_launch_dependents()
        mix = zeros(8, Float32)
        maximum, exponent_sum, qpart = Float32(0), Float32(0), Float32(0)
        if tid < 128:
            self.prefetch(args, state[7], rank, token)
            maximum, exponent_sum, qpart = self.pre(state, rank, args[9], mix)
        self.post(args, state, (mix, maximum, exponent_sum, qpart), rank, token)
        cute.arch.griddepcontrol_wait()


class OprojFused:
    def __init__(self, nb, *, has_delta=True):
        self.producer = OprojProducer(768, staged_blocks=0)
        # TP16 requires prefix/blocks to be ready at launch.
        self.consumer = OprojConsumer(nb, has_delta=has_delta)

    @cute.jit
    def __call__(self, *args):
        if cutlass.const_expr(args[0].shape[0] > 8):
            raise ValueError("The selected fused schedule supports B1..8")
        launch(self.kernel, args, (8, 14, 1), 512, 8, min_blocks_per_mp=1)

    @cute.kernel
    def kernel(self, args):
        x, weight, rows = args[0], args[1], (args[2], *args[5:])
        rank, cluster, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        cta = cluster * 8 + rank
        active = (cluster < x.shape[0]) & (rank < 7)
        state = self.consumer.allocate()
        smem = self.producer.allocate(x.shape[0])
        self.consumer.setup(state[5])
        cute.arch.griddepcontrol_launch_dependents()
        group0 = active & (tid < 128)
        if group0:
            self.consumer.prefetch(rows, state[7], rank, cluster)
        weights = self.producer.load_weight(weight, smem[2], cta)
        mix = zeros(8, Float32)
        maximum, exponent_sum, qpart = Float32(0), Float32(0), Float32(0)
        if group0:
            maximum, exponent_sum, qpart = self.consumer.pre(state, rank, rows[9], mix)
        self.producer.compute(args, smem, weights, cta, False)
        if active:
            block_mix = (mix, maximum, exponent_sum, qpart)
            self.consumer.post(rows, state, block_mix, rank, cluster)
