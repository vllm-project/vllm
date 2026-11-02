# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA RecoverSSM verify with the fused short convolution on the fifth-generation
tensor cores, warp specialized (the dim-first conv state layout).

The math and the per-item MMA / solve are those of ``verify.py`` (read its
docstring first). What differs is how an item's inputs reach the CTA:

* A producer warp (warp 4) walks the CTA's items (the first is the CTA's own, the
  rest come from the work counter) one item ahead of the 4 compute warps. Per item
  it bulk-copies everything the activation phase reads into shared memory: the
  conv state rows of the item's 3 x 128 channels (7 BF16 each, contiguous in the
  dim-first layout), the Q raw q / k / v and gate rows (256 bytes each), the conv
  taps and dt_bias, and loads the item's scalars (slot, tokens, accepted, beta,
  A_log) into a small record. The state tile follows by TMA once the previous
  item's MMAs have read the buffer. The first item's state, conv state and taps
  are requested before the PDL wait.
* The compute warps therefore never wait on a global load: thread c reads channel
  c from shared memory, writes the rolled conv state back in place, and one thread
  bulk-stores it (3 x 1792 bytes) instead of 21 strided 2-byte stores per thread.
* The probes are unnormalized (p_t = e^{L_t} k_t, r_t = e^{L_t} q_t): the q / k
  squared norms join the coupling dot products in one warp reduce-scatter, and the
  norms scale the projections and couplings afterwards (no separate norm pass).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    bf16,
    fma,
    load_words,
    prefetch_bytes,
    shared_address,
    sigmoid,
)

from .commit import bulk_done, bulk_read_done, store_bulk
from .verify import (
    A_KGROUP_BYTES,
    B_KGROUP_BYTES,
    PROBES,
    K,
    bf16_split,
    proxy_fence,
    reset_counter,
    smem_desc,
    state_smem_layout,
    take_ticket,
    tc_fence_after,
    tc_fence_before,
    tmem_alloc,
    tmem_dealloc,
    tmem_load_row,
    tmem_relinquish,
    umma,
    umma_commit,
)

CONSUMERS = 128  # 4 compute warps: thread c owns channel c / state row c
PRODUCER = 4  # the producer warp
THREADS = CONSUMERS + 32
# Compute threads that load and activate the betas: past the Q * Q + 2 Q (at
# most 80) threads that reduce the couplings and norms.
BETA_TID = 96
CTAS_PER_SM = 3  # shared memory bound (~60 KB per CTA)
CONV_W = 4  # conv taps


EPS = 1e-6


def state_sw128_layout():
    """State tile in shared memory in the 128-byte swizzled K-major UMMA layout: two
    16 KB K halves (64 keys each) of 128 rows x 128 bytes, rows 128 bytes apart,
    16-byte chunks XOR-swizzled by row % 8. TMA writes it in 2 boxes of 64 x 128
    (the no-swizzle layout takes 16 boxes of 8 x 128, i.e. 16-byte TMA rows)."""
    from cutlass.cute.nvgpu import tcgen05

    atom = tcgen05.make_smem_layout_atom(
        tcgen05.SmemLayoutAtomKind.K_SW128, cutlass.BFloat16
    )
    return cute.tile_to_shape(atom, (K, K), (0, 1))


@asm()
def bulk_load(dst, src, size, barrier):
    """``size`` bytes global -> shared on ``barrier`` (complete_tx)."""
    code = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
    code += " [$0], [$1], $2, [$3];"
    return code, "r,l,r,r,~{memory}", (dst, Int64(src), Int32(size), barrier)


@asm()
def arrive_expect(barrier, size):
    code = "mbarrier.arrive.expect_tx.shared::cta.b64 _, [$0], $1;"
    return code, "r,r,~{memory}", (barrier, Int32(size))


@asm()
def arrive(barrier):
    return "mbarrier.arrive.shared::cta.b64 _, [$0];", "r,~{memory}", (barrier,)


@asm(Int32)
def try_wait_cta(barrier, phase):
    code = "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p,[$1],$2; "
    code += "selp.u32 $0,1,0,p; }"
    return code, "=r,r,r,~{memory}", (barrier, Int32(phase))


@cute.jit
def wait_bar(barrier, phase):
    done = try_wait_cta(barrier, phase)
    while done == 0:
        done = try_wait_cta(barrier, phase)


@asm(Float32, pure=True)
def div_full(a, b):
    return "div.full.f32 $0, $1, $2;", "=f,f,f", (Float32(a), Float32(b))


@asm()
def consumer_sync():
    """Named barrier 1 over the 4 compute warps."""
    return f"bar.sync 1, {CONSUMERS};", "~{memory}", ()


class KdaTcVerify2:
    """Same arguments as ``KdaTcVerify`` with ``conv=True`` and the dim-first conv
    state [slots, 3HK, Q + 2] (channel rows contiguous). Requires 16-byte aligned
    qkv / raw_g rows, conv state channel blocks, conv weight and dt_bias (checked by
    the host wrapper)."""

    def __init__(
        self, heads: int, window: int, ctas: int, conv_weight_f32: bool, swizzle=True
    ):
        assert 1 <= window <= 8
        self.heads, self.window, self.ctas = heads, window, ctas
        self.conv_weight_f32 = conv_weight_f32
        self.swizzle = swizzle

    def smem_layout(self):
        return state_sw128_layout() if self.swizzle else state_smem_layout()

    @cute.jit
    def __call__(self, *args):
        state = args[5]
        state_rows = cute.make_tensor(
            state.iterator,
            cute.make_layout(
                ((K, state.shape[1], state.shape[0]), K),
                stride=((K, state.stride[1], state.stride[0]), 1),
            ),
        )
        from cutlass.cute.nvgpu import cpasync

        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), state_rows, self.smem_layout(), (K, K)
        )
        items = args[10].shape[0] * self.heads
        self.kernel(args[:-1], tma_atom, tma_tensor).launch(
            grid=(cutlass.min(items, self.ctas), 1, 1),
            block=(THREADS, 1, 1),
            stream=args[-1],
            use_pdl=True,
        )

    @cute.jit
    def load_state(self, tma_atom, tma_tensor, a_tile, barrier, slot, head, lane):
        """Warp-wide: TMA state (slot, head) into ``a_tile`` on ``barrier``."""
        from cutlass.cute.nvgpu import cpasync

        H: cutlass.Constexpr = self.heads
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, K * K * 2)
        cute.arch.sync_warp()
        s_state = cute.make_tensor(a_tile.iterator, self.smem_layout())
        g_state = cute.local_tile(tma_tensor, (K, K), (Int32(slot) * H + head, 0))
        t_smem, t_gmem = cpasync.tma_partition(
            tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(s_state, 0, 2),
            cute.group_modes(g_state, 0, 2),
        )
        cute.copy(tma_atom, t_gmem, t_smem, tma_bar_ptr=barrier)

    @cute.kernel
    def kernel(self, args, tma_atom: cute.CopyAtom, tma_tensor: cute.Tensor):
        qkv, raw_g, raw_beta, a_log, dt_bias, state, corr, kg, out = args[:9]
        query_start, slots, conv_state, conv_w, accepted = args[9:14]
        counter, lower_bound, null_slot = args[-3], args[-2], args[-1]
        Q: cutlass.Constexpr = self.window
        H: cutlass.Constexpr = self.heads
        S: cutlass.Constexpr = Q + 2  # conv state length
        W32: cutlass.Constexpr = self.conv_weight_f32
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        warp, lane = tid // 32, tid % 32
        items = slots.shape[0] * H

        smem = cutlass.utils.SmemAllocator()
        a_tile = alloc(smem, cutlass.BFloat16, (K * K,), align=1024)
        b_hi = alloc(smem, cutlass.BFloat16, (K // 8 * B_KGROUP_BYTES // 2,), align=128)
        b_lo = alloc(smem, cutlass.BFloat16, (K // 8 * B_KGROUP_BYTES // 2,), align=128)
        # The activation stage (one item): raw q / k / v rows, gate rows, conv
        # state rows (dim first: channel c's S values at c * S), taps, dt_bias.
        qkv_s = alloc(smem, cutlass.BFloat16, (3, Q, K), align=128)
        g_s = alloc(smem, cutlass.BFloat16, (Q, K), align=128)
        conv_s = alloc(smem, cutlass.BFloat16, (3, K * S), align=128)
        w_dtype = Float32 if W32 else cutlass.BFloat16
        w_s = alloc(smem, w_dtype, (3, K, CONV_W), align=128)
        bias_s = alloc(smem, Float32, (K,), align=128)
        # Item records (double buffered): item, slot, bos, qlen, first, loaded,
        # valid; and A, raw beta[Q].
        meta = alloc(smem, Int32, (2, 8))
        meta_f = alloc(smem, Float32, (2, 8))
        # Reduced values: coupling c_it (i < t), d_it (i <= t), then |k_t|^2, |q_t|^2.
        specs = [(0, t, i) for t in range(Q) for i in range(t)]
        specs += [(1, t, i) for t in range(Q) for i in range(t + 1)]
        NC = len(specs)
        specs += [(2, t, 0) for t in range(Q)] + [(3, t, 0) for t in range(Q)]
        NV = len(specs)
        ROUNDS: cutlass.Constexpr = (NV + 31) // 32
        partial = alloc(smem, Float32, (ROUNDS, 4, 32))
        coef = alloc(smem, Float32, (2 * Q * Q,))  # c[t * Q + i] then d[...]
        scales = alloc(smem, Float32, (2 * Q,))  # 1/|k_t|, K^-1/2 / |q_t|
        betas = alloc(smem, Float32, (8,))
        table = alloc(smem, Int32, (NV,))  # per coupling: slot | na << 8 | nb << 16
        bars = alloc(smem, cutlass.Int64, (4,), align=8)
        tmem_base = alloc(smem, Uint32, (1,), align=4)
        act_full = bars.iterator
        act_empty = bars.iterator + 1
        state_full = bars.iterator + 2
        mma_done = bars.iterator + 3

        if warp == 0:
            cute.arch.sync_warp()
            tmem_alloc(tmem_base.iterator)
            tmem_relinquish()
        if tid == 0:
            cute.arch.mbarrier_init(act_full, 2)
            cute.arch.mbarrier_init(act_empty, 1)
            cute.arch.mbarrier_init(state_full, 1)
            cute.arch.mbarrier_init(mma_done, 1)
            cute.arch.mbarrier_init_fence()
        if tid < CONSUMERS:
            # The zero probe columns, once.
            b_offset = (tid // 8) * (B_KGROUP_BYTES // 2) + tid % 8
            for n in cutlass.range_constexpr(2 * Q, PROBES):
                b_hi[b_offset + n * 8] = cutlass.BFloat16(0)
                b_lo[b_offset + n * 8] = cutlass.BFloat16(0)
            if tid < NC:
                code = Int32(0)
                for j in cutlass.range_constexpr(NC):
                    kind, t, i = specs[j]
                    na = NC + i  # |k_i|^2
                    nb = NC + t if kind == 0 else NC + Q + t  # |k_t|^2 or |q_t|^2
                    word = (kind * Q * Q + t * Q + i) | (na << 8) | (nb << 16)
                    if tid == j:
                        code = Int32(word)
                table[tid] = code
        proxy_fence()
        tc_fence_before()
        cute.arch.barrier()
        tc_fence_after()
        tmem = tmem_base[0]

        if warp == PRODUCER:
            # ================= producer =================
            # Item scalars are read one item ahead (they are step metadata, written
            # before this layer: the first item's are read before the PDL wait like
            # the slots), so after the stage frees only bulk copies are issued.
            T = qkv.shape[0]
            item = cta
            request = item // H
            c_slot = Int64(slots[request])
            c_bos = query_start[request]
            c_end = query_start[request + 1]
            c_first = accepted[request] - 1
            c_a = a_log[item % H].to(Float32)
            issued = Int32(0)
            loads = Int32(0)  # state loads so far
            p_store = Int32(0)  # the previous item's conv rows to store
            p_slot = Int64(0)
            p_head = Int32(0)
            while item < items:
                ticket = Int32(0)
                if lane == 0:
                    ticket = Int32(take_ticket(Int64(counter.iterator.toint())))
                    if ticket == items - 1:
                        reset_counter(Int64(counter.iterator.toint()))
                head = item % H
                slot, bos, qlen = c_slot, c_bos, c_end - c_bos
                loaded = (slot > Int64(null_slot)) & (T > 0)
                mi = issued % 2
                if issued > 0:
                    wait_bar(act_empty, (issued - 1) % 2)
                    # The previous item's rolled conv rows (written in place).
                    if p_store != 0:
                        if lane < 3:
                            channel = lane * H * K + p_head * K
                            store_bulk(
                                at_address(conv_state, p_slot, channel, 0),
                                conv_s.iterator + lane * (K * S),
                                K * S * 2,
                            )
                            bulk_read_done()
                        cute.arch.sync_warp()
                # Pre-dependency copies: conv state rows, taps, dt_bias.
                if lane == 0:
                    if loaded:
                        w_bytes = 3 * K * CONV_W * (4 if W32 else 2)
                        arrive_expect(act_full, 3 * K * S * 2 + w_bytes + K * 4)
                    else:
                        arrive(act_full)
                cute.arch.sync_warp()
                if loaded:
                    if lane < 3:
                        channel = lane * H * K + head * K
                        bulk_load(
                            shared_address(conv_s.iterator) + lane * (K * S * 2),
                            at_address(conv_state, slot, channel, 0),
                            K * S * 2,
                            act_full,
                        )
                    elif lane < 6:
                        x = lane - 3
                        wb: cutlass.Constexpr = K * CONV_W * (4 if W32 else 2)
                        bulk_load(
                            shared_address(w_s.iterator) + x * wb,
                            at_address(conv_w, (x * H + head) * K, 0),
                            wb,
                            act_full,
                        )
                    elif lane == 6:
                        bulk_load(
                            shared_address(bias_s.iterator),
                            at_address(dt_bias, head * K),
                            K * 4,
                            act_full,
                        )
                cute.arch.sync_warp()
                if issued == 0:
                    if loaded:
                        self.load_state(
                            tma_atom, tma_tensor, a_tile, state_full, slot, head, lane
                        )
                    cute.arch.griddepcontrol_wait()
                if lane == 0:
                    meta[mi, 0] = item
                    meta[mi, 1] = Int32(slot)
                    meta[mi, 2] = bos
                    meta[mi, 3] = qlen
                    meta[mi, 4] = c_first
                    meta[mi, 5] = Int32(loaded)
                    meta[mi, 6] = Int32(slot > Int64(null_slot))
                    meta_f[mi, 0] = cute.math.exp(c_a)
                cute.arch.sync_warp()
                if lane == 0:
                    if loaded:
                        arrive_expect(act_full, 4 * Q * K * 2)
                    else:
                        arrive(act_full)
                cute.arch.sync_warp()
                if loaded:  # noqa: SIM102 (traced: `and` does not combine traced values)
                    if lane < 4 * Q:
                        x, t = lane // Q, lane % Q
                        src = Int64(0)
                        dst = Uint32(0)
                        row = bos + cutlass.min(t, cutlass.max(qlen - 1, 0))
                        row = cutlass.min(row, T - 1)
                        if x < 3:
                            src = at_address(qkv, row, x * H * K + head * K)
                            dst = shared_address(qkv_s.iterator) + (x * Q + t) * (K * 2)
                        else:
                            src = at_address(raw_g, row, head * K)
                            dst = shared_address(g_s.iterator) + t * (K * 2)
                        bulk_load(dst, src, K * 2, act_full)
                # The state, once the previous item's MMAs have read the buffer.
                if loaded:
                    if issued > 0:
                        if loads > 0:
                            wait_bar(mma_done, (loads - 1) % 2)
                        self.load_state(
                            tma_atom, tma_tensor, a_tile, state_full, slot, head, lane
                        )
                    loads += 1
                p_store = Int32(loaded & (qlen > 0))
                p_slot, p_head = slot, head
                # The next item: its scalars, and its rows and state head to L2
                # while this one computes.
                nxt = cute.arch.shuffle_sync(ticket, 0) + ctas
                if nxt < items:
                    n_request, n_head = nxt // H, nxt % H
                    c_slot = Int64(slots[n_request])
                    c_bos = query_start[n_request]
                    c_end = query_start[n_request + 1]
                    c_first = accepted[n_request] - 1
                    c_a = a_log[n_head].to(Float32)
                    if (c_slot > Int64(null_slot)) & (T > 0):
                        n_qlen = c_end - c_bos
                        if lane < 4 * Q:
                            x, t = lane // Q, lane % Q
                            row = c_bos + cutlass.min(t, cutlass.max(n_qlen - 1, 0))
                            row = cutlass.min(row, T - 1)
                            src = Int64(0)
                            if x < 3:
                                src = at_address(qkv, row, x * H * K + n_head * K)
                            else:
                                src = at_address(raw_g, row, n_head * K)
                            prefetch_bytes(src, K * 2)
                        elif lane < 4 * Q + 3:
                            channel = (lane - 4 * Q) * H * K + n_head * K
                            prefetch_bytes(
                                at_address(conv_state, c_slot, channel, 0), K * S * 2
                            )
                        elif lane == 4 * Q + 3:
                            prefetch_bytes(
                                at_address(state, c_slot, n_head, 0, 0), K * K * 2
                            )
                issued += 1
                item = nxt
            # The last conv rows, then the end record.
            if issued > 0:
                wait_bar(act_empty, (issued - 1) % 2)
                if p_store != 0:
                    if lane < 3:
                        channel = lane * H * K + p_head * K
                        store_bulk(
                            at_address(conv_state, p_slot, channel, 0),
                            conv_s.iterator + lane * (K * S),
                            K * S * 2,
                        )
                    cute.arch.sync_warp()
            if lane == 0:
                meta[issued % 2, 0] = items
                arrive(act_full)
                arrive(act_full)
            if lane < 3:
                bulk_done()
        else:
            # ================= compute warps =================
            cute.arch.griddepcontrol_wait()
            c = tid
            k_step = Int32(0)
            state_phase = Int32(0)
            mma_phase = Int32(0)
            wait_bar(act_full, 0)
            cur = meta[0, 0]
            b_offset = (c // 8) * (B_KGROUP_BYTES // 2) + c % 8
            while cur < items:
                mi = k_step % 2
                bos = meta[mi, 2]
                qlen = meta[mi, 3]
                loaded = meta[mi, 5] != 0
                valid = meta[mi, 6]
                request, head = cur // H, cur % H
                live = loaded & (qlen > 0)
                qv = cute.make_rmem_tensor((Q,), Float32)
                kv = cute.make_rmem_tensor((Q,), Float32)
                vv = cute.make_rmem_tensor((Q,), Float32)
                ev = cute.make_rmem_tensor((Q,), Float32)  # e^{L_t}
                wv = cute.make_rmem_tensor((Q,), Float32)  # e^{-L_t} k_t
                beta_raw = Float32(0)
                if live:
                    if tid >= BETA_TID and tid < BETA_TID + Q:  # noqa: SIM102 (traced: `and` does not combine traced values)
                        if tid - BETA_TID < qlen:
                            beta_raw = raw_beta[bos + tid - BETA_TID, head].to(Float32)
                    first = meta[mi, 4]
                    a = meta_f[mi, 0]
                    bias = bias_s[c]
                    hist = cute.make_rmem_tensor((3, 3), Float32)
                    raw = cute.make_rmem_tensor((3, Q), cutlass.BFloat16)
                    g_raw = cute.make_rmem_tensor((Q,), cutlass.BFloat16)
                    for x in cutlass.range_constexpr(3):
                        for i in cutlass.range_constexpr(3):
                            hist[x, i] = conv_s[x, c * S + first + i].to(Float32)
                        for t in cutlass.range_constexpr(Q):
                            raw[x, t] = qkv_s[x, t, c]
                    for t in cutlass.range_constexpr(Q):
                        g_raw[t] = g_s[t, c]
                    # The rolled conv state, in place (thread c owns channel c's
                    # S values); bulk-stored after the barrier.
                    for x in cutlass.range_constexpr(3):
                        for i in cutlass.range_constexpr(2):
                            conv_s[x, c * S + i] = hist[x, i + 1].to(cutlass.BFloat16)
                        for t in cutlass.range_constexpr(Q):
                            if t < qlen:
                                conv_s[x, c * S + 2 + t] = raw[x, t]
                    # Width-4 convolution and SiLU (the stock kernel's order).
                    for x in cutlass.range_constexpr(3):
                        if cutlass.const_expr(W32):
                            w = load_words(
                                at_ptr(w_s, x, c, 0),
                                count=4,
                                shared=True,
                                mutable=True,
                                result_type=Float32,
                                word_type="f32",
                            )
                        else:
                            ww = load_words(
                                at_ptr(w_s, x, c, 0), count=2, shared=True, mutable=True
                            )
                            w = [bf16_word(ww[j // 2], j % 2) for j in range(4)]
                        win = [hist[x, i] for i in range(3)]
                        for t in cutlass.range_constexpr(Q):
                            win.append(raw[x, t].to(Float32))
                            acc = Float32(0)
                            for j in cutlass.range_constexpr(CONV_W):
                                if cutlass.const_expr(W32):
                                    acc = fma(win[t + j], w[j], acc)
                                else:
                                    prod = (win[t + j] * w[j]).to(cutlass.BFloat16)
                                    acc = acc + prod.to(Float32)
                            # The stock kernel's SiLU: ex2.approx and div.full.
                            raw[x, t] = div_full(
                                acc, Float32(1) + cute.math.exp(-acc, fastmath=True)
                            ).to(cutlass.BFloat16)
                    cum = Float32(0)
                    for t in cutlass.range_constexpr(Q):
                        vv[t] = raw[2, t].to(Float32)
                        qv[t], kv[t] = Float32(0), Float32(0)
                        if t < qlen:
                            kg[request, head, t, c] = raw[1, t]
                            kg[request, head, t, K + c] = g_raw[t]
                            qv[t], kv[t] = raw[0, t].to(Float32), raw[1, t].to(Float32)
                            gate = sigmoid(a * (g_raw[t].to(Float32) + bias), True)
                            cum = cum + lower_bound * gate
                        ev[t] = cute.math.exp(cum, fastmath=True)
                        wv[t] = cute.math.exp(-cum, fastmath=True) * kv[t]
                    # Unnormalized probes p_t = e^{L_t} k_t, r_t = e^{L_t} q_t.
                    for t in cutlass.range_constexpr(Q):
                        for n, value in ((t, ev[t] * kv[t]), (Q + t, ev[t] * qv[t])):
                            hi, lo = bf16_split(value)
                            b_hi[b_offset + n * 8] = hi
                            b_lo[b_offset + n * 8] = lo
                    # This channel's products, reduce-scattered over the warp.
                    for rnd in cutlass.range_constexpr(ROUNDS):
                        count = min(32, NV - 32 * rnd)
                        LOG: cutlass.Constexpr = (count - 1).bit_length()
                        width: cutlass.Constexpr = 1 << LOG
                        vals = []
                        for j in cutlass.range_constexpr(width):
                            value = Float32(0)
                            if cutlass.const_expr(j < count):
                                kind, t, i = specs[32 * rnd + j]
                                if cutlass.const_expr(kind == 0):
                                    value = wv[i] * (ev[t] * kv[t])
                                elif cutlass.const_expr(kind == 1):
                                    value = wv[i] * (ev[t] * qv[t])
                                elif cutlass.const_expr(kind == 2):
                                    value = kv[t] * kv[t]
                                else:
                                    value = qv[t] * qv[t]
                            vals.append(value)
                        for st in cutlass.range_constexpr(LOG):
                            shift = 16 >> st
                            half = width >> (st + 1)
                            upper = (lane & shift) != 0
                            kept = []
                            for j in cutlass.range_constexpr(half):
                                lo_v, hi_v = vals[j], vals[j + half]
                                send, keep = hi_v, lo_v
                                if upper:
                                    send, keep = lo_v, hi_v
                                kept.append(
                                    keep + cute.arch.shuffle_sync_bfly(send, shift)
                                )
                            vals = kept
                        total = vals[0]
                        for st in cutlass.range_constexpr(LOG, 5):
                            total = total + cute.arch.shuffle_sync_bfly(total, 16 >> st)
                        stride: cutlass.Constexpr = 32 // width
                        if lane % stride == 0:
                            partial[rnd, warp, lane // stride] = total
                # Probes and conv rows visible to the async proxy; TMEM reads of the
                # previous item ordered before this item's MMAs.
                proxy_fence()
                tc_fence_before()
                consumer_sync()
                if tid == 0:
                    arrive(act_empty)  # the stage is read (conv rows: the producer)
                    if loaded:
                        wait_bar(state_full, state_phase)
                        if live:
                            tc_fence_after()
                            a_base = shared_address(a_tile.iterator)
                            hi_base = shared_address(b_hi.iterator)
                            lo_base = shared_address(b_lo.iterator)
                            for kt in cutlass.range_constexpr(K // 16):
                                if cutlass.const_expr(self.swizzle):
                                    # K step kt: 32 bytes into its 128-byte atom row.
                                    a_desc = sw128_desc(
                                        a_base + (kt // 4) * (K * 128) + (kt % 4) * 32
                                    )
                                else:
                                    a_desc = smem_desc(
                                        a_base + kt * 2 * A_KGROUP_BYTES,
                                        A_KGROUP_BYTES,
                                        128,
                                    )
                                for h, base in enumerate((hi_base, lo_base)):
                                    b_desc = smem_desc(
                                        base + kt * 2 * B_KGROUP_BYTES,
                                        B_KGROUP_BYTES,
                                        128,
                                    )
                                    umma(
                                        tmem,
                                        a_desc,
                                        b_desc,
                                        0 if kt == 0 and h == 0 else 1,
                                    )
                            umma_commit(mma_done)
                        else:
                            arrive(mma_done)  # the buffer is free (no tokens)
                if loaded:
                    state_phase = state_phase ^ 1
                if live:
                    # Coupling scalars and norm scales from the warp partials.
                    if tid < NV:
                        r0, j0 = tid // 32, tid % 32
                        total = (partial[r0, 0, j0] + partial[r0, 1, j0]) + (
                            partial[r0, 2, j0] + partial[r0, 3, j0]
                        )
                        if tid < NC:
                            word = table[tid]
                            na, nb = (word >> 8) & 0xFF, (word >> 16) & 0xFF
                            sa = (
                                partial[na // 32, 0, na % 32]
                                + partial[na // 32, 1, na % 32]
                            ) + (
                                partial[na // 32, 2, na % 32]
                                + partial[na // 32, 3, na % 32]
                            )
                            sb = (
                                partial[nb // 32, 0, nb % 32]
                                + partial[nb // 32, 1, nb % 32]
                            ) + (
                                partial[nb // 32, 2, nb % 32]
                                + partial[nb // 32, 3, nb % 32]
                            )
                            scale = cute.math.rsqrt(
                                sa + Float32(EPS)
                            ) * cute.math.rsqrt(sb + Float32(EPS))
                            if word & 0xFF >= Q * Q:
                                scale = scale * Float32(K**-0.5)
                            coef[word & 0xFF] = total * scale
                        else:
                            scale = cute.math.rsqrt(total + Float32(EPS))
                            if tid >= NC + Q:
                                scale = scale * Float32(K**-0.5)
                            scales[tid - NC] = scale
                    elif tid >= BETA_TID and tid < BETA_TID + Q:
                        beta = Float32(0)
                        if tid - BETA_TID < qlen:
                            beta = sigmoid(beta_raw, True)
                        betas[tid - BETA_TID] = beta
                consumer_sync()
                if live:
                    wait_bar(mma_done, mma_phase)
                    tc_fence_after()
                    cute.arch.sync_warp()
                    proj = list(tmem_load_row(tmem + ((warp * 32) << 16)))
                    r = tid
                    u = cute.make_rmem_tensor((Q,), Float32)
                    for t in cutlass.range_constexpr(Q):
                        acc = vv[t] - proj[t] * scales[t]
                        for i in cutlass.range_constexpr(t):
                            acc = acc - coef[t * Q + i] * u[i]
                        u[t] = betas[t] * acc
                        o = proj[Q + t] * scales[Q + t]
                        for i in cutlass.range_constexpr(t + 1):
                            o = fma(coef[Q * Q + t * Q + i], u[i], o)
                        if t < qlen:
                            out[bos + t, head * K + r] = o.to(out.element_type)
                            corr[request, head, t, r] = u[t]
                if loaded:
                    mma_phase = mma_phase ^ 1
                if valid == 0:
                    for t in cutlass.range_constexpr(Q):
                        if t < qlen:
                            out[bos + t, head * K + tid] = out.element_type(0)
                k_step += 1
                wait_bar(act_full, k_step % 2)
                cur = meta[k_step % 2, 0]

        cute.arch.griddepcontrol_launch_dependents()
        tc_fence_before()
        cute.arch.barrier()
        if warp == 0:
            tc_fence_after()
            cute.arch.sync_warp()
            tmem_dealloc(tmem)


@cute.jit
def sw128_desc(address):
    """128-byte swizzled K-major UMMA descriptor: 8-row groups 1024 bytes apart."""
    static = (1 << 16) | (64 << 32) | (1 << 46) | (2 << 61)
    return cutlass.Uint64((address >> 4) & 0x3FFF) | cutlass.Uint64(static)


def at_ptr(tensor, *coord):
    return tensor.iterator + cute.crd2idx(coord, tensor.layout)


def at_address(tensor, *coord):
    """64-bit global address of ``tensor[coord]``."""
    return Int64(at_ptr(tensor, *coord).toint())


def bf16_word(word, high):
    return bf16(word, high)
