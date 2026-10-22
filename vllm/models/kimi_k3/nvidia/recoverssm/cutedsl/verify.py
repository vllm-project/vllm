# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KDA RecoverSSM verify on the fifth-generation tensor cores: one CTA (4 warps) per
(head, request).

With per-channel log decays ``L_t = g_1 + ... + g_t`` and normalized ``k_t``, ``q_t``
(``q_t`` carries the 1/sqrt(K) scale), the verify window satisfies

    u_t = beta_t * (v_t - S_0 p_t - sum_{i<t} c_it u_i),   p_t = e^{L_t} k_t
    o_t = S_0 r_t + sum_{i<=t} d_it u_i,                    r_t = e^{L_t} q_t

with c_it = <w_i, p_t>, d_it = <w_i, r_t> and w_i = e^{-L_i} k_i.

CTAs are persistent and walk the (head, request) items. Per item:

1. The state S_0 [V, K] arrives by TMA in the no-swizzle K-major UMMA layout (core
   matrices of 8 rows x 16 bytes, K-adjacent ones 2 KB apart). The next item's
   state is loaded into the same buffer as soon as this item's MMAs complete, under
   this item's solve and the next item's probes (the first item's is issued before
   the PDL wait: the state is written by the previous step's commit).
2. Thread c owns channel c: the optional short convolution of q, k, v (the stock
   ``causal_conv1d_update`` speculative step), then p_t, r_t (BF16 high and low
   halves, the B operands) and w_t; two threads per (probe, i) pair reduce the
   coupling scalars.
3. One thread issues the M128 x N16 x K16 BF16 MMAs (high and low halves accumulate
   into the same FP32 accumulator): D[row, column] = S_0 row . probe column.
4. Thread r reads state row r's 16 projections from tensor memory, solves the
   row's triangular system and stores its outputs and corrections (coalesced).

Outputs and records (``u_t``, key and raw gate) match ``kda_recoverssm_verify``.
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cute.nvgpu import cpasync

from vllm.models.kimi_k3.nvidia.specialized_decode.cutedsl.common.primitives import (
    alloc,
    asm,
    at,
    bf16,
    fma,
    load_words,
    shared_address,
    sigmoid,
    warp_sum,
)

K = 128
THREADS = 128  # one per key channel / value row
CTAS_PER_SM = 5  # resident CTAs per SM (shared memory bound)
PROBES = 16  # MMA N: 2 Q <= 16 probe columns, zero padded
TMEM_COLS = 32  # the smallest allocation; the accumulator uses PROBES columns
A_KGROUP_BYTES = K * 16  # one 8-wide K group of all 128 rows
# B K groups 272 bytes apart (not 256): the 4 channel groups of a warp's stores land
# in different banks.
B_KGROUP_BYTES = PROBES * 16 + 16


def state_smem_layout():
    """State tile in shared memory: element (row, k) at (k // 8) * 1024 + row * 8 +
    k % 8 (BF16 elements)."""
    return cute.make_layout((K, (8, K // 8)), stride=(8, (1, K * 8)))


# Instruction descriptor, kind::f16: FP32 accumulator, BF16 A and B, both K-major,
# N = 16, M = 128.
IDESC = (1 << 4) | (1 << 7) | (1 << 10) | ((PROBES >> 3) << 17) | ((K >> 4) << 24)


@asm()
def tmem_alloc(dst):
    code = f"tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [$0], {TMEM_COLS};"
    return code, "r,~{memory}", (dst,)


@asm()
def tmem_relinquish():
    return "tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;", "", ()


@asm()
def tmem_dealloc(address):
    code = f"tcgen05.dealloc.cta_group::1.sync.aligned.b32 $0, {TMEM_COLS};"
    return code, "r", (Uint32(address),)


@asm()
def tc_fence_before():
    return "tcgen05.fence::before_thread_sync;", "~{memory}", ()


@asm()
def tc_fence_after():
    return "tcgen05.fence::after_thread_sync;", "~{memory}", ()


@asm()
def proxy_fence():
    """Generic-proxy shared memory writes (``st.shared``, ``cp.async``) visible to
    the tensor cores' async proxy."""
    return "fence.proxy.async.shared::cta;", "~{memory}", ()


@asm(Uint32)
def take_ticket(counter):
    """Relaxed GPU-scope fetch-and-increment of the work counter."""
    return "atom.relaxed.gpu.global.add.u32 $0, [$1], 1;", "=r,l,~{memory}", (counter,)


@asm()
def reset_counter(counter):
    return "st.relaxed.gpu.global.u32 [$0], 0;", "l,~{memory}", (counter,)


@asm()
def cp_async_wait_all():
    return "cp.async.wait_all;", "~{memory}", ()


@asm()
def umma(d_address, a_desc, b_desc, accumulate):
    code = "{ .reg .pred p; setp.ne.b32 p, $3, 0; "
    code += f"tcgen05.mma.cta_group::1.kind::f16 [$0], $1, $2, {IDESC}, p; }}"
    return code, "r,l,l,r", (Uint32(d_address), a_desc, b_desc, Int32(accumulate))


@asm()
def umma_commit(barrier):
    code = "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [$0];"
    return code, "r,~{memory}", (barrier,)


@asm((Float32,) * PROBES)
def tmem_load_row(address):
    """Lane i of warp w: TMEM lane 32 w + i, columns [0, 16) of ``address``."""
    regs = ",".join(f"${i}" for i in range(PROBES))
    code = f"tcgen05.ld.sync.aligned.32x32b.x16.b32 {{{regs}}}, [${PROBES}]; "
    code += "tcgen05.wait::ld.sync.aligned;"
    return code, ",".join(["=f"] * PROBES + ["r"]), (Uint32(address),)


@cute.jit
def smem_desc(address, lead_bytes: cutlass.Constexpr, stride_bytes: cutlass.Constexpr):
    """No-swizzle K-major UMMA shared memory descriptor (SM100 version bits)."""
    static = ((lead_bytes >> 4) << 16) | ((stride_bytes >> 4) << 32) | (1 << 46)
    return Uint64((address >> 4) & 0x3FFF) | Uint64(static)


@cute.jit
def bf16_split(value: Float32):
    """BF16 (high, low) with high + low ~= value to ~2^-17."""
    high = value.to(cutlass.BFloat16)
    low = (value - high.to(Float32)).to(cutlass.BFloat16)
    return high, low


class KdaTcVerify:
    """Args: qkv [T, 3HK], raw_g [T, HK], raw_beta [T, H], a_log [H], dt_bias [HK],
    state [slots, H, V, K], corr [slots, H, Q, V], kg [slots, H, Q, 2K], out [T, HV],
    query_start [B + 1], request slots [B], (conv state [slots, Q + 2, 3HK] token
    major, or [slots, 3HK, Q + 2] with ``conv_dim_first``, conv weight [3HK, 4]
    (BF16, or FP32 with ``conv_weight_f32``: the stock conv1d parameter),
    accepted tokens [B] when ``conv``; qkv is then the raw projection), a zeroed
    int32 work counter [1] (left zeroed),
    lower_bound, null slot, stream. ``ctas``: the persistent grid bound (CTAs
    resident per SM x SMs)."""

    def __init__(
        self,
        heads: int,
        window: int,
        conv: bool = False,
        ctas: int = 740,
        conv_dim_first: bool = False,
        conv_weight_f32: bool = False,
    ):
        assert 1 <= window <= 8
        self.heads, self.window, self.conv, self.ctas = heads, window, conv, ctas
        self.conv_dim_first = conv_dim_first
        self.conv_weight_f32 = conv_weight_f32

    @cute.jit
    def __call__(self, *args):
        # The state as ((V, H, slots), K) rows with its head and slot strides (vLLM
        # pads the slot stride to the cache page; V = K rows of a head are
        # contiguous); one TMA box per (slot, head): all V rows
        # into the no-swizzle K-major UMMA layout (row 16 bytes after row, K groups
        # of 8 a whole 128-row group apart).
        state = args[5]
        state_rows = cute.make_tensor(
            state.iterator,
            cute.make_layout(
                ((K, state.shape[1], state.shape[0]), K),
                stride=((K, state.stride[1], state.stride[0]), 1),
            ),
        )
        tma_atom, tma_tensor = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), state_rows, state_smem_layout(), (K, K)
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
        H: cutlass.Constexpr = self.heads
        if lane == 0:
            cute.arch.mbarrier_arrive_and_expect_tx(barrier, K * K * 2)
        cute.arch.sync_warp()
        s_state = cute.make_tensor(a_tile.iterator, state_smem_layout())
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
        query_start, slots = args[9:11]
        counter, lower_bound, null_slot = args[-3], args[-2], args[-1]
        Q: cutlass.Constexpr = self.window
        H: cutlass.Constexpr = self.heads
        tid, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        ctas, _, _ = cute.arch.grid_dim()
        warp, lane = tid // 32, tid % 32
        c = tid  # thread c owns channel c of q, k, v and g, and state row c
        items = slots.shape[0] * H
        smem = cutlass.utils.SmemAllocator()
        a_tile = alloc(smem, cutlass.BFloat16, (K * K,), align=1024)
        b_hi = alloc(smem, cutlass.BFloat16, (K // 8 * B_KGROUP_BYTES // 2,), align=128)
        b_lo = alloc(smem, cutlass.BFloat16, (K // 8 * B_KGROUP_BYTES // 2,), align=128)
        coef = alloc(smem, Float32, (2 * Q * Q,))  # c[t * Q + i] then d[...]
        # Coupling pairs: c_it (i < t), then d_it (i <= t), in rounds of 32.
        pairs = [(0, t, i) for t in range(Q) for i in range(t)]
        pairs += [(1, t, i) for t in range(Q) for i in range(t + 1)]
        slot_of = [k * Q * Q + t * Q + i for k, t, i in pairs]
        ROUNDS: cutlass.Constexpr = (len(pairs) + 31) // 32
        partial = alloc(smem, Float32, (ROUNDS, 4, 32))
        betas = alloc(smem, Float32, (8,))
        norms = alloc(smem, Float32, (4, 2 * Q))
        mma_done = alloc(smem, cutlass.Int64, (1,), align=8)
        state_full = alloc(smem, cutlass.Int64, (1,), align=8)
        tmem_base = alloc(smem, Uint32, (1,), align=4)
        next_item = alloc(smem, Int32, (1,))

        if warp == 0:
            cute.arch.sync_warp()
            tmem_alloc(tmem_base.iterator)
            tmem_relinquish()
        if tid == 0:
            cute.arch.mbarrier_init(mma_done.iterator, 1)
            cute.arch.mbarrier_init(state_full.iterator, 1)
            cute.arch.mbarrier_init_fence()
        tc_fence_before()
        cute.arch.barrier()
        tc_fence_after()
        tmem = tmem_base[0]
        if warp == 0:
            first_slot = Int64(slots[cta // H])
            if first_slot > Int64(null_slot):
                self.load_state(
                    tma_atom, tma_tensor, a_tile, state_full.iterator, first_slot,
                    cta % H, lane,
                )  # fmt: skip
        cute.arch.griddepcontrol_wait()

        state_phase = Int32(0)
        mma_phase = Int32(0)
        # The first item is the CTA's own; the rest come from the work counter, so
        # CTAs that start late (beyond the resident ones) or run slow items take
        # fewer. Exactly ``items`` tickets are taken per launch (one per item run,
        # the last of each CTA past the end); the taker of the last one resets the
        # counter for the next launch.
        counter_address = Int64(counter.iterator.toint())
        item = cta
        while item < items:
            if tid == 0:
                ticket = Int32(take_ticket(counter_address))
                if ticket == items - 1:
                    reset_counter(counter_address)
                next_item[0] = ticket + ctas
            request, head = item // H, item % H
            slot = Int64(slots[request])
            valid = slot > Int64(null_slot)
            bos = query_start[request]
            qlen = query_start[request + 1] - bos
            live = valid & (qlen > 0)
            qv = cute.make_rmem_tensor((Q,), Float32)
            kv = cute.make_rmem_tensor((Q,), Float32)
            vv = cute.make_rmem_tensor((Q,), Float32)
            logs = cute.make_rmem_tensor((Q,), Float32)
            if live:
                # ---- 2. Activations (thread c owns channel c of q, k, v and g). ----
                bias = dt_bias[head * K + c].to(Float32)
                a = cute.math.exp(a_log[head].to(Float32))
                if cutlass.const_expr(self.conv):
                    conv_state, conv_w, accepted = args[11:14]
                    hist = cute.make_rmem_tensor((3, 3), cutlass.BFloat16)
                    taps = cute.make_rmem_tensor((3, 2), Uint32)
                    taps32 = cute.make_rmem_tensor((3, 4), Float32)
                    first = accepted[request] - 1
                    for x in cutlass.range_constexpr(3):
                        channel = x * H * K + head * K + c
                        for i in cutlass.range_constexpr(3):
                            if cutlass.const_expr(self.conv_dim_first):
                                hist[x, i] = conv_state[slot, channel, first + i]
                            else:
                                hist[x, i] = conv_state[slot, first + i, channel]
                        if cutlass.const_expr(self.conv_weight_f32):
                            for j in cutlass.range_constexpr(4):
                                taps32[x, j] = conv_w[channel, j]
                        else:
                            w2 = load_words(at(conv_w, channel, 0), count=2)
                            taps[x, 0], taps[x, 1] = w2[0], w2[1]
                raw = cute.make_rmem_tensor((3, Q), cutlass.BFloat16)
                g_raw = cute.make_rmem_tensor((Q,), cutlass.BFloat16)
                for t in cutlass.range_constexpr(Q):
                    row = bos + cutlass.min(t, qlen - 1)
                    for x in cutlass.range_constexpr(3):
                        raw[x, t] = qkv[row, x * H * K + head * K + c]
                    g_raw[t] = raw_g[row, head * K + c]
                if cutlass.const_expr(self.conv):
                    # The rolled state: the last 2 history rows, then the raw tokens.
                    for x in cutlass.range_constexpr(3):
                        channel = x * H * K + head * K + c
                        for i in cutlass.range_constexpr(2):
                            if cutlass.const_expr(self.conv_dim_first):
                                conv_state[slot, channel, i] = hist[x, i + 1]
                            else:
                                conv_state[slot, i, channel] = hist[x, i + 1]
                        for t in cutlass.range_constexpr(Q):
                            if t < qlen:
                                if cutlass.const_expr(self.conv_dim_first):
                                    conv_state[slot, channel, 2 + t] = raw[x, t]
                                else:
                                    conv_state[slot, 2 + t, channel] = raw[x, t]
                    # Width-4 convolution and SiLU in the stock kernel's order: with
                    # BF16 taps the products round to BF16, with FP32 taps they stay
                    # FP32; the output rounds to BF16.
                    for x in cutlass.range_constexpr(3):
                        if cutlass.const_expr(self.conv_weight_f32):
                            w = [taps32[x, j] for j in range(4)]
                        else:
                            w = [bf16(taps[x, j // 2], j % 2) for j in range(4)]
                        win = [hist[x, i].to(Float32) for i in range(3)]
                        for t in cutlass.range_constexpr(Q):
                            win.append(raw[x, t].to(Float32))
                            acc = Float32(0)
                            for j in cutlass.range_constexpr(4):
                                if cutlass.const_expr(self.conv_weight_f32):
                                    acc = fma(win[t + j], w[j], acc)
                                else:
                                    prod = (win[t + j] * w[j]).to(cutlass.BFloat16)
                                    acc = acc + prod.to(Float32)
                            raw[x, t] = (acc / (Float32(1) + cute.math.exp(-acc))).to(
                                cutlass.BFloat16
                            )
                cum = Float32(0)
                for t in cutlass.range_constexpr(Q):
                    vv[t] = raw[2, t].to(Float32)
                    qv[t], kv[t] = Float32(0), Float32(0)
                    if t < qlen:
                        kg[slot, head, t, c] = raw[1, t]
                        kg[slot, head, t, K + c] = g_raw[t]
                        qv[t], kv[t] = raw[0, t].to(Float32), raw[1, t].to(Float32)
                        gate = sigmoid(a * (g_raw[t].to(Float32) + bias), True)
                        cum = cum + lower_bound * gate
                    logs[t] = cum
                for t in cutlass.range_constexpr(Q):
                    sq = warp_sum(qv[t] * qv[t])
                    sk = warp_sum(kv[t] * kv[t])
                    if lane == 0:
                        norms[warp, 2 * t] = sq
                        norms[warp, 2 * t + 1] = sk
                if tid < Q:
                    beta = Float32(0)
                    if tid < qlen:
                        beta = sigmoid(raw_beta[bos + tid, head].to(Float32), True)
                    betas[tid] = beta
            cute.arch.barrier()
            nxt = next_item[0]  # rewritten only after this item's end barrier
            if live:
                # ---- Probes: B operands (K-major, no swizzle) and the coupling inputs.
                b_offset = (c // 8) * (B_KGROUP_BYTES // 2) + c % 8
                for t in cutlass.range_constexpr(Q):
                    sq = (norms[0, 2 * t] + norms[1, 2 * t]) + (
                        norms[2, 2 * t] + norms[3, 2 * t]
                    )
                    sk = (norms[0, 2 * t + 1] + norms[1, 2 * t + 1]) + (
                        norms[2, 2 * t + 1] + norms[3, 2 * t + 1]
                    )
                    q_n = qv[t] * cute.math.rsqrt(sq + Float32(1e-6)) * Float32(K**-0.5)
                    k_n = kv[t] * cute.math.rsqrt(sk + Float32(1e-6))
                    decay = cute.math.exp(logs[t])
                    p_t, r_t = decay * k_n, decay * q_n
                    for n, value in ((t, p_t), (Q + t, r_t)):
                        b_hi[b_offset + n * 8], b_lo[b_offset + n * 8] = bf16_split(
                            value
                        )
                    # From here on qv holds r_t, kv holds p_t and logs holds w_t.
                    qv[t], kv[t] = r_t, p_t
                    logs[t] = cute.math.exp(-logs[t]) * k_n
                for n in cutlass.range_constexpr(2 * Q, PROBES):
                    b_hi[b_offset + n * 8] = cutlass.BFloat16(0)
                    b_lo[b_offset + n * 8] = cutlass.BFloat16(0)
            # The probes visible to the MMA issuer.
            proxy_fence()
            cute.arch.barrier()
            if live:
                # ---- 3. The projections: 8 K steps x (high, low) MMAs. ----
                if tid == 0:
                    cute.arch.mbarrier_wait(state_full.iterator, state_phase)
                    tc_fence_after()
                    a_base = shared_address(a_tile.iterator)
                    hi_base = shared_address(b_hi.iterator)
                    lo_base = shared_address(b_lo.iterator)
                    for kt in cutlass.range_constexpr(K // 16):
                        a_desc = smem_desc(
                            a_base + kt * 2 * A_KGROUP_BYTES, A_KGROUP_BYTES, 128
                        )
                        for h, base in enumerate((hi_base, lo_base)):
                            b_desc = smem_desc(
                                base + kt * 2 * B_KGROUP_BYTES, B_KGROUP_BYTES, 128
                            )
                            umma(tmem, a_desc, b_desc, 0 if kt == 0 and h == 0 else 1)
                    umma_commit(mma_done.iterator)
                # The coupling scalars while the tensor cores run: this channel's
                # products, reduce-scattered over the warp (lane l ends with pair
                # 32 round + l), then summed over warps.
                for rnd in cutlass.range_constexpr(ROUNDS):
                    vals = []
                    for j in cutlass.range_constexpr(32):
                        value = Float32(0)
                        if cutlass.const_expr(32 * rnd + j < len(pairs)):
                            kind, t, i = pairs[32 * rnd + j]
                            value = logs[i] * (kv[t] if kind == 0 else qv[t])
                        vals.append(value)
                    for shift in (16, 8, 4, 2, 1):
                        upper = (lane & shift) != 0
                        kept = []
                        for j in cutlass.range_constexpr(shift):
                            lo_v, hi_v = vals[j], vals[j + shift]
                            send, keep = hi_v, lo_v
                            if upper:
                                send, keep = lo_v, hi_v
                            kept.append(keep + cute.arch.shuffle_sync_bfly(send, shift))
                        vals = kept
                    partial[rnd, warp, lane] = vals[0]
            cute.arch.barrier()
            if live:
                for rnd in cutlass.range_constexpr(ROUNDS):
                    if tid < len(pairs) - 32 * rnd:
                        total = (partial[rnd, 0, tid] + partial[rnd, 1, tid]) + (
                            partial[rnd, 2, tid] + partial[rnd, 3, tid]
                        )
                        target = Int32(0)  # pair index -> coefficient slot
                        for j in cutlass.range_constexpr(
                            min(32, len(pairs) - 32 * rnd)
                        ):
                            if tid == j:
                                target = Int32(slot_of[32 * rnd + j])
                        coef[target] = total
            cute.arch.barrier()  # the coupling scalars
            proj = [Float32(0)] * PROBES
            if live:
                cute.arch.mbarrier_wait(mma_done.iterator, mma_phase)
                tc_fence_after()
                cute.arch.sync_warp()  # tcgen05.ld is warp aligned
                proj = list(tmem_load_row(tmem + ((warp * 32) << 16)))
                mma_phase = mma_phase ^ 1
            elif valid:
                if tid == 0:
                    cute.arch.mbarrier_wait(state_full.iterator, state_phase)  # qlen 0
            if valid:
                state_phase = state_phase ^ 1
            # The state buffer is free: the next item's state streams under the solve.
            if warp == 0:  # noqa: SIM102 (traced: `and` does not combine traced values)
                if nxt < items:
                    next_slot = Int64(slots[nxt // H])
                    if next_slot > Int64(null_slot):
                        self.load_state(
                            tma_atom, tma_tensor, a_tile, state_full.iterator,
                            next_slot, nxt % H, lane,
                        )  # fmt: skip
            if live:
                # ---- 4. Thread r solves state row r. ----
                r = tid
                u = cute.make_rmem_tensor((Q,), Float32)
                for t in cutlass.range_constexpr(Q):
                    acc = vv[t] - proj[t]
                    for i in cutlass.range_constexpr(t):
                        acc = acc - coef[t * Q + i] * u[i]
                    u[t] = betas[t] * acc
                    o = proj[Q + t]
                    for i in cutlass.range_constexpr(t + 1):
                        o = fma(coef[Q * Q + t * Q + i], u[i], o)
                    if t < qlen:
                        out[bos + t, head * K + r] = o.to(out.element_type)
                        corr[slot, head, t, r] = u[t]
            if slot <= Int64(null_slot):
                for t in cutlass.range_constexpr(Q):
                    if t < qlen:
                        out[bos + t, head * K + tid] = out.element_type(0)
            tc_fence_before()
            cute.arch.barrier()  # shared scalars and TMEM reused by the next item
            tc_fence_after()
            item = nxt

        cute.arch.griddepcontrol_launch_dependents()
        if warp == 0:
            cute.arch.sync_warp()
            tmem_dealloc(tmem)
