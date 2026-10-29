# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Small-M dense FP8 GEMM on the Humming W8A16 layout, in CuTe DSL (SM90).

``Y[M, N] = X[M, K] @ dequant(W)^T`` for at most 48 BF16 rows of ``X``, reading
the weight and scales of a stock SM90 Humming W8A16 linear in place:

- ``weight`` int32 ``[Kp / 16, Np * 4]``: one 1 KiB block per (16 k, 64 n),
  laid out as ``mma.sync`` A fragments (lane ``(n % 8) * 4 + (k % 8) // 2``,
  word ``(n % 64) // 8``, byte ``[0, 2, 1, 3][((k % 16) // 8) * 2 + k % 2]``);
- ``weight_scale`` UE8M0 ``[Kp / 32, Np]``, the 8 x 8 (n % 8, n // 8) index of
  each 64-column block transposed.

Swap-AB: the weight is operand A (64 output columns per tile, ``tw`` tiles per
CTA) and the tokens operand B, ``m16n8k16`` MMAs with FP32 accumulation. Two
operand precisions:

- ``prec`` 0 (f16): weights convert exactly with ``cvt.rn.f16x2.e4m3x2``; each
  32-k group's partial is scaled by its UE8M0 scale in FP32. X is staged once
  per CTA and converted to f16 with a per-token power-of-two pre-scale chosen
  from its absolute maximum over the slice, undone exactly on the FP32 sums.
- ``prec`` 1 (bf16): weights are dequantized by bit surgery and multiplied by
  the scale in BF16 (exact for an e4m3 value times a power of two).

Pipeline: two DMA warps stream ``S``-row (16 k each) W/S stages through ``NS``
shared-memory buffers with ``cp.async`` and mbarriers, the weights with an
evict-first L2 policy (each is read once a step); they start before the
programmatic grid wait (the weights do not depend on the previous kernel).
The consumer warps (``4 * tw / wm`` column groups x ``kg`` k groups) wait, load
X with bulk copies, then consume the stages; the k groups reduce through shared
memory in a fixed order. K is split ``sk`` ways over a ``(sk, 1, 1)`` cluster,
whose partials rank 0 sums through distributed shared memory in a fixed order,
so the result is deterministic. Grouped mode (``wo_a``): output columns
``[g * n_group, (g + 1) * n_group)`` read ``X[:, g * x_gstride : ...]``.

Publishing (``pub``): the epilogue also stores the BF16 output, negative zero
cleared, into this rank's slot of generation ``stage_state[0]`` of a fused
all-reduce + mHC Lamport mailbox on every rank (``multimem.st``), and one
thread sets ``stage_state[1]`` to that generation: the work of the all-reduce's
publish kernel (``_SharedOnlyPublishDeviceKernel``). It reads ``stage_state[0]``
after its grid wait, which the previous consumer wrote; the consumer, the next
kernel, reads ``stage_state[1]`` after its own grid wait, after this grid.

Every launch uses programmatic dependent launch and triggers its dependents
right after its prologue.
"""

import cutlass
import cutlass.cute as cute
from cuda.bindings.driver import CUstream
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.arch import inline_ptx

DMA_WARPS = 2


class DenseFp8Gemm:
    """One compiled shape class of the kernel (see the module docstring).

    Args:
        nt: 8-token MMA column tiles (``ceil(M / 8)``).
        tw: 64-column weight tiles per CTA.
        S: 16-k rows per pipeline stage.
        NS: pipeline stages.
        wm: 16-column MMA tiles per consumer warp.
        kg: consumer warps splitting each stage's k rows.
        sk: K split over the CTAs of a cluster.
        xr: X rows (16 k each) staged per CTA, even.
        prec: 0 f16 operands, 1 bf16 operands.
        pub: also publish into a Lamport mailbox.

    """

    def __init__(
        self,
        nt: int,
        tw: int,
        S: int,
        NS: int,
        wm: int,
        kg: int,
        sk: int,
        xr: int,
        prec: int,
        pub: int,
        cn: int = 1,
    ):
        assert (4 * tw) % wm == 0 and S % (2 * kg) == 0
        self.nt, self.mpad, self.tw, self.S, self.NS = nt, 8 * nt, tw, S, NS
        self.wm, self.ng, self.kg, self.sk, self.xr = wm, 4 * tw // wm, kg, sk, xr
        self.cw = self.ng * kg
        self.nthreads = (self.cw + DMA_WARPS) * 32
        assert (S * tw * 64) % (DMA_WARPS * 32) == 0
        self.prec, self.pub = prec, pub
        assert sk * cn <= 8
        self.cn = cn
        self.w_bytes = S * tw * 1024
        self.s_bytes = (S // 2) * tw * 64
        self.stage_bytes = self.w_bytes + self.s_bytes
        self.x_pitch = xr * 32 + 16
        self.x_bytes = self.mpad * self.x_pitch
        self.red_pitch = 64 * tw + 4
        self.red_one = self.mpad * self.red_pitch * 4
        self.red_bytes = kg * self.red_one
        self.slot_bytes = max(sk - 1, 0) * self.mpad * 16 * tw * 16
        self.smem_bytes = (
            NS * self.stage_bytes
            + self.x_bytes
            + self.red_bytes
            + 4 * self.mpad
            + 16 * NS
            + self.slot_bytes
            + 16
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        s: cute.Tensor,
        y: cute.Tensor,
        M: Int32,
        N: Int32,
        K: Int32,
        Np: Int32,
        ldx: Int32,
        ldy: Int32,
        n_group: Int32,
        x_gstride: Int32,
        mc: Int64,
        sst: Int64,
        rank_cap: Int32,
        tp_cap: Int32,
        stream: CUstream,
    ):
        T = N // (64 * self.tw)
        args = [x, w, s, y, M, N, K, Np, ldx, ldy, n_group, x_gstride]
        args += [mc, sst, rank_cap, tp_cap]
        if const_expr(self.sk * self.cn > 1):
            self.kernel(*args).launch(
                grid=[T * self.sk, 1, 1],
                block=[self.nthreads, 1, 1],
                cluster=[self.sk * self.cn, 1, 1],
                stream=stream,
                use_pdl=True,
            )
        else:
            self.kernel(*args).launch(
                grid=[T, 1, 1],
                block=[self.nthreads, 1, 1],
                stream=stream,
                use_pdl=True,
            )

    @cute.jit
    def issue_stage(self, dt, st, t, r, rows, wg, sg, Np, pol):
        """DMA thread ``dt``'s copies of stage rows ``[r, r + rows)``, the
        weights with L2 policy ``pol`` (evict-first: they are read once a
        step)."""
        tw = self.tw
        ndt = DMA_WARPS * 32
        cpr = tw * 64
        wsrc = wg + Int64(r) * Int64(Np * 16) + Int64(t * tw * 1024)
        if rows == self.S:
            for c_ in cutlass.range_constexpr((self.S * cpr) // ndt):
                c = c_ * ndt + dt
                row = c // cpr
                ci = c % cpr
                cp_async16_hint(
                    st + row * (tw * 1024) + (ci ^ ((ci >> 3) & 1)) * 16,
                    wsrc + Int64(row) * Int64(Np * 16) + Int64(ci * 16),
                    pol,
                )
        else:
            for c in cutlass.range(dt, rows * cpr, ndt):
                row = c // cpr
                ci = c % cpr
                cp_async16_hint(
                    st + row * (tw * 1024) + (ci ^ ((ci >> 3) & 1)) * 16,
                    wsrc + Int64(row) * Int64(Np * 16) + Int64(ci * 16),
                    pol,
                )
        spr = tw * 4
        ssrc = sg + Int64(r // 2) * Int64(Np) + Int64(t * tw * 64)
        for c in cutlass.range(dt, (rows // 2) * spr, ndt):
            gr = c // spr
            off = (c % spr) * 16
            cp_async16(
                st + self.w_bytes + gr * (tw * 64) + off,
                ssrc + Int64(gr) * Int64(Np) + Int64(off),
            )

    @cute.jit
    def pair_mma(self, acc, st, pr, xrow0, lane, ngi, sx, lm_tok, lm_half):
        """The MMAs of stage row pair ``pr`` (one 32-k scale group)."""
        tw, nt, wm = self.tw, self.nt, self.wm
        g = lane // 4
        mig0 = ngi * wm
        sc = []
        sbase = st + self.w_bytes + pr * (tw * 64) + (mig0 // 4) * 64 + g * 8
        evs = []
        if const_expr(wm == 4):
            s01 = lds_v2(sbase)
            for jm in cutlass.range_constexpr(4):
                evs.append((s01[jm // 2] >> (16 * (jm % 2))) & 0xFFFF)
        elif const_expr(wm == 2):
            sw = lds_u32(sbase + (mig0 % 4) * 2)
            evs.append(sw & 0xFFFF)
            evs.append((sw >> 16) & 0xFFFF)
        else:
            evs.append(lds_u16(sbase + (mig0 % 4) * 2))
        for jm in cutlass.range_constexpr(wm):
            ev = evs[jm]
            if const_expr(self.prec == 0):
                sc.append((e8m0_f32(ev & 0xFF), e8m0_f32(ev >> 8)))
            else:
                sc.append(scale2_bf16x2(ev))
        tmp = []
        for h in cutlass.range_constexpr(2):
            row = 2 * pr + h
            xrow = xrow0 + row
            bfr = []
            for j in cutlass.range_constexpr(nt // 2):
                b0, b1, b2, b3 = ldmatrix_x4(
                    sx + (j * 16 + lm_tok) * self.x_pitch + xrow * 32 + lm_half
                )
                bfr.append((b0, b1))
                bfr.append((b2, b3))
            if const_expr(nt % 2 == 1):
                b0, b1 = ldmatrix_x2(
                    sx
                    + ((nt - 1) * 8 + lane % 8) * self.x_pitch
                    + xrow * 32
                    + ((lane // 8) % 2) * 16
                )
                bfr.append((b0, b1))
            words = []
            if const_expr(wm >= 2):
                for jc in cutlass.range_constexpr(wm // 2):
                    mig = mig0 + 2 * jc
                    ch = 2 * lane + (mig % 4) // 2
                    ch = ch ^ ((ch >> 3) & 1)
                    w0, w1, w2, w3 = lds_v4(
                        st + row * (tw * 1024) + (mig // 4) * 1024 + ch * 16
                    )
                    words.append((w0, w1))
                    words.append((w2, w3))
            else:
                mig = mig0
                ch = 2 * lane + (mig % 4) // 2
                ch = ch ^ ((ch >> 3) & 1)
                w0, w1 = lds_v2(
                    st + row * (tw * 1024) + (mig // 4) * 1024 + ch * 16 + (mig % 2) * 8
                )
                words.append((w0, w1))
            for jm in cutlass.range_constexpr(wm):
                if const_expr(self.prec == 1):
                    a0, a2 = deq_word(words[jm][0], sc[jm][0])
                    a1, a3 = deq_word(words[jm][1], sc[jm][1])
                    for j in cutlass.range_constexpr(nt):
                        o = (jm * nt + j) * 4
                        d0, d1, d2, d3 = mma_bf16(
                            a0,
                            a1,
                            a2,
                            a3,
                            bfr[j][0],
                            bfr[j][1],
                            acc[o],
                            acc[o + 1],
                            acc[o + 2],
                            acc[o + 3],
                        )
                        acc[o] = d0
                        acc[o + 1] = d1
                        acc[o + 2] = d2
                        acc[o + 3] = d3
                    continue
                a0, a2 = cvt_e4m3_word(words[jm][0])
                a1, a3 = cvt_e4m3_word(words[jm][1])
                for j in cutlass.range_constexpr(nt):
                    if const_expr(h == 0):
                        tmp.append(mma_f16_z(a0, a1, a2, a3, bfr[j][0], bfr[j][1]))
                    else:
                        k_ = jm * nt + j
                        tmp[k_] = mma_f16(
                            a0,
                            a1,
                            a2,
                            a3,
                            bfr[j][0],
                            bfr[j][1],
                            tmp[k_][0],
                            tmp[k_][1],
                            tmp[k_][2],
                            tmp[k_][3],
                        )
        for jm in cutlass.range_constexpr(wm if self.prec == 0 else 0):
            for j in cutlass.range_constexpr(nt):
                o = (jm * nt + j) * 4
                d = tmp[jm * nt + j]
                acc[o] = acc[o] + d[0] * sc[jm][0]
                acc[o + 1] = acc[o + 1] + d[1] * sc[jm][0]
                acc[o + 2] = acc[o + 2] + d[2] * sc[jm][1]
                acc[o + 3] = acc[o + 3] + d[3] * sc[jm][1]

    @cute.jit
    def sum_slots(self, a):
        """The k groups' partials at ``a``, summed in k-group order."""
        v0, v1, v2, v3 = lds_f32x4(a)
        for kk in cutlass.range_constexpr(1, self.kg):
            u0, u1, u2, u3 = lds_f32x4(a + kk * self.red_one)
            v0 = v0 + u0
            v1 = v1 + u1
            v2 = v2 + u2
            v3 = v3 + u3
        return v0, v1, v2, v3

    @cute.jit
    def store_out(
        self, yg, tok, col, v0, v1, v2, v3, ldy, N, mc, stage, rank_cap, tp_cap
    ):
        lo = pack_bf16x2(v0, v1)
        hi = pack_bf16x2(v2, v3)
        stg_b32x2(yg + Int64(tok * ldy + col) * 2, lo, hi)
        if const_expr(self.pub == 1):
            slot = Int64(stage * tp_cap + rank_cap + tok) * Int64(N) + Int64(col)
            stmc_b32x2(
                mc + slot * 2, clear_negzero_bf16x2(lo), clear_negzero_bf16x2(hi)
            )

    @cute.jit
    def release_stage(self, cta, tidx, sst, stage):
        if const_expr(self.pub == 1):  # noqa: SIM102
            if cta == 0:  # noqa: SIM102
                if tidx == 0:
                    st_global_u32(sst + 4, stage)

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        w: cute.Tensor,
        s: cute.Tensor,
        y: cute.Tensor,
        M: Int32,
        N: Int32,
        K: Int32,
        Np: Int32,
        ldx: Int32,
        ldy: Int32,
        n_group: Int32,
        x_gstride: Int32,
        mc: Int64,
        sst: Int64,
        rank_cap: Int32,
        tp_cap: Int32,
    ):
        S, NS, tw, nt, sk = self.S, self.NS, self.tw, self.nt, self.sk
        tidx, _, _ = cute.arch.thread_idx()
        cta, _, _ = cute.arch.block_idx()
        warp = tidx // 32
        lane = tidx % 32

        smem = cutlass.memory.SmemAllocator()
        sbase = smem.allocate(self.smem_bytes, byte_alignment=1024).toint()
        s_stage0 = sbase
        sx = sbase + NS * self.stage_bytes
        s_red = sx + self.x_bytes
        s_amax = s_red + self.red_bytes
        s_full = s_amax + 4 * self.mpad
        s_empty = s_full + 8 * NS
        s_slot = s_empty + 8 * NS
        s_xbar = s_slot + self.slot_bytes

        xg = x.iterator.toint()
        wg = w.iterator.toint()
        sg = s.iterator.toint()
        yg = y.iterator.toint()

        R = K // 16
        cn = self.cn
        crank = cta % (sk * cn)
        kidx = crank % sk
        # The cn CTAs of a K slice (different N tiles) share its X.
        cidx = crank // sk
        t = (cta // (sk * cn)) * cn + cidx
        rb = ((kidx * R) // sk) & ~1
        re = (((kidx + 1) * R) // sk) & ~1
        if kidx == sk - 1:
            re = R
        nch = (re - rb + S - 1) // S

        if tidx == 0:
            for i in cutlass.range_constexpr(NS):
                mbar_init(s_full + 8 * i, DMA_WARPS * 32)
                mbar_init(s_empty + 8 * i, self.cw)
            mbar_init(s_xbar, 1)
            fence_mbar_init()
        for i in cutlass.range(tidx, self.red_bytes // 16, self.nthreads):
            sts_f32x4(
                s_red + i * 16, Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)
            )
        if tidx < self.mpad:
            sts_u32(s_amax + tidx * 4, Int32(0))
        cute.arch.sync_threads()
        if const_expr(sk * cn > 1):
            # s_red is zeroed and the X barrier armed before the peers write.
            cluster_arrive_release()
        cute.arch.griddepcontrol_launch_dependents()

        if warp >= self.cw:
            # DMA warps: the W/S stages.
            dt = tidx - self.cw * 32
            pol = evict_first_policy()
            for c in cutlass.range(nch):
                stage = c % NS
                r = rb + c * S
                rows = re - r
                if rows > S:
                    rows = Int32(S)
                if c >= NS:
                    phase = ((c // NS) - 1) % 2
                    eb = s_empty + 8 * stage
                    done = mbar_try_wait(eb, phase)
                    while done == 0:
                        done = mbar_try_wait(eb, phase)
                self.issue_stage(
                    dt,
                    s_stage0 + stage * self.stage_bytes,
                    t,
                    r,
                    rows,
                    wg,
                    sg,
                    Np,
                    pol,
                )
                cp_async_arrive_noinc(s_full + 8 * stage)
            if const_expr(sk > 1):
                cluster_wait_acquire()
                cluster_arrive_release()
                cluster_wait_acquire()
        else:
            ctid = tidx
            nct = self.cw * 32
            cute.arch.griddepcontrol_wait()
            if const_expr(cn > 1):
                cluster_wait_acquire()  # the peers' X barriers are armed
            # Stage X[:, rb * 16 : re * 16] once, one bulk copy per token; with
            # cn > 1, each CTA multicasts 1/cn of the tokens to its K slice's.
            nrow = re - rb
            cpt = nrow * 2
            xsrc = xg + Int64((((t * tw * 64) // n_group) * x_gstride) * 2 + rb * 32)
            if warp == 0:
                if lane == 0:
                    mbar_arrive_expect_tx(s_xbar, M * nrow * 32)
                cute.arch.sync_warp()
                if const_expr(cn > 1):
                    mask = 0
                    for c_ in range(cn):
                        mask |= 1 << (sk * c_)
                    for tok in cutlass.range(lane, M, 32):
                        if tok % cn == cidx:
                            bulk_g2s_mc(
                                sx + tok * self.x_pitch,
                                xsrc + Int64(tok) * Int64(ldx * 2),
                                nrow * 32,
                                s_xbar,
                                Int32(mask) << kidx,
                            )
                else:
                    for tok in cutlass.range(lane, M, 32):
                        bulk_g2s(
                            sx + tok * self.x_pitch,
                            xsrc + Int64(tok) * Int64(ldx * 2),
                            nrow * 32,
                            s_xbar,
                        )
            xdone = mbar_try_wait(s_xbar, Int32(0))
            while xdone == 0:
                xdone = mbar_try_wait(s_xbar, Int32(0))
            named_bar_sync(14, nct)
            if const_expr(self.prec == 0):
                # Per-token |x| max -> 2^-kx pre-scale; bf16 -> f16 in place.
                for tok in cutlass.range(warp, M, self.cw):
                    mx = Int32(0)
                    for c in cutlass.range(lane, cpt, 32):
                        v0, v1, v2, v3 = lds_v4(sx + tok * self.x_pitch + c * 16)
                        mb = bf16x8_absmax_bits(v0, v1, v2, v3)
                        if mb > mx:
                            mx = mb
                    mx = redux_max_u32(mx)
                    kx = (mx >> 23) - 141
                    if kx < 0:
                        kx = Int32(0)
                    scl = i2f_bits((127 - kx) << 23)
                    for c in cutlass.range(lane, cpt, 32):
                        a_ = sx + tok * self.x_pitch + c * 16
                        v0, v1, v2, v3 = lds_v4(a_)
                        sts_v4_b32(
                            a_,
                            bf16x2_to_f16x2_scaled(v0, scl),
                            bf16x2_to_f16x2_scaled(v1, scl),
                            bf16x2_to_f16x2_scaled(v2, scl),
                            bf16x2_to_f16x2_scaled(v3, scl),
                        )
                    if lane == 0:
                        sts_u32(s_amax + tok * 4, mx)
                named_bar_sync(14, nct)

            ngi = warp // self.kg
            kgi = warp % self.kg
            g = lane // 4
            q = lane % 4
            nacc = self.wm * nt * 4
            acc = cute.make_rmem_tensor(cute.make_layout(nacc), Float32)
            for a in cutlass.range_constexpr(nacc):
                acc[a] = Float32(0.0)
            lm_q = lane // 8
            lm_tok = (lm_q // 2) * 8 + lane % 8
            lm_half = (lm_q % 2) * 16
            for c in cutlass.range(nch):
                stage = c % NS
                phase = (c // NS) % 2
                r = rb + c * S
                rows = re - r
                if rows > S:
                    rows = Int32(S)
                fb = s_full + 8 * stage
                done = mbar_try_wait(fb, phase)
                while done == 0:
                    done = mbar_try_wait(fb, phase)
                st = s_stage0 + stage * self.stage_bytes
                xrow0 = r - rb
                if rows == S:
                    for ii in cutlass.range_constexpr(S // (2 * self.kg)):
                        self.pair_mma(
                            acc,
                            st,
                            ii * self.kg + kgi,
                            xrow0,
                            lane,
                            ngi,
                            sx,
                            lm_tok,
                            lm_half,
                        )
                else:
                    for ii in cutlass.range_constexpr(S // (2 * self.kg)):
                        pr = ii * self.kg + kgi
                        if 2 * pr < rows:
                            self.pair_mma(
                                acc, st, pr, xrow0, lane, ngi, sx, lm_tok, lm_half
                            )
                cute.arch.sync_warp()
                if lane == 0:
                    mbar_arrive(s_empty + 8 * stage)

            # Undo the per-token pre-scale; one shared slot per k group.
            for jm in cutlass.range_constexpr(self.wm):
                nl = (ngi * self.wm + jm) * 16 + g
                for j in cutlass.range_constexpr(nt):
                    o = (jm * nt + j) * 4
                    tok = j * 8 + 2 * q
                    ab0 = lds_u32(s_amax + tok * 4)
                    ab1 = lds_u32(s_amax + tok * 4 + 4)
                    k0 = (ab0 >> 23) - 141
                    if k0 < 0:
                        k0 = Int32(0)
                    k1 = (ab1 >> 23) - 141
                    if k1 < 0:
                        k1 = Int32(0)
                    u0 = i2f_bits((127 + k0 * (1 - self.prec)) << 23)
                    u1 = i2f_bits((127 + k1 * (1 - self.prec)) << 23)
                    a_ = s_red + (tok * self.red_pitch + nl) * 4 + kgi * self.red_one
                    sts_f32(a_, acc[o] * u0)
                    sts_f32(a_ + self.red_pitch * 4, acc[o + 1] * u1)
                    sts_f32(a_ + 32, acc[o + 2] * u0)
                    sts_f32(a_ + self.red_pitch * 4 + 32, acc[o + 3] * u1)
            n0c = t * (64 * tw)
            nv4 = 16 * tw
            stage = Int32(0)
            if const_expr(self.pub == 1):
                # After the grid wait: the previous consumer advanced it.
                stage = ld_volatile_u32(sst)
            total = M * nv4
            if const_expr(sk > 1):
                # Push this CTA's partial into its K group's rank 0 slot kidx.
                named_bar_sync(15, nct)
                if const_expr(cn == 1):
                    cluster_wait_acquire()
                for e in cutlass.range(ctid, total, nct):
                    tok = e // nv4
                    c4 = e % nv4
                    v0, v1, v2, v3 = self.sum_slots(
                        s_red + (tok * self.red_pitch + c4 * 4) * 4
                    )
                    if kidx > 0:
                        st_dsmem_f32x4(
                            mapa(
                                s_slot + ((kidx - 1) * total + e) * 16,
                                cidx * sk,
                            ),
                            v0,
                            v1,
                            v2,
                            v3,
                        )
                cluster_arrive_release()
                if kidx == 0:
                    cluster_wait_acquire()
                    for e in cutlass.range(ctid, total, nct):
                        tok = e // nv4
                        c4 = e % nv4
                        v0, v1, v2, v3 = self.sum_slots(
                            s_red + (tok * self.red_pitch + c4 * 4) * 4
                        )
                        for pp in cutlass.range_constexpr(sk - 1):
                            u0_, u1_, u2_, u3_ = lds_f32x4(
                                s_slot + (pp * total + e) * 16
                            )
                            v0 = v0 + u0_
                            v1 = v1 + u1_
                            v2 = v2 + u2_
                            v3 = v3 + u3_
                        self.store_out(
                            yg,
                            tok,
                            n0c + c4 * 4,
                            v0,
                            v1,
                            v2,
                            v3,
                            ldy,
                            N,
                            mc,
                            stage,
                            rank_cap,
                            tp_cap,
                        )
                    self.release_stage(cta, tidx, sst, stage)
                else:
                    cluster_wait_acquire()
            else:
                named_bar_sync(15, nct)
                for e in cutlass.range(ctid, total, nct):
                    tok = e // nv4
                    c4 = e % nv4
                    v0, v1, v2, v3 = self.sum_slots(
                        s_red + (tok * self.red_pitch + c4 * 4) * 4
                    )
                    self.store_out(
                        yg,
                        tok,
                        n0c + c4 * 4,
                        v0,
                        v1,
                        v2,
                        v3,
                        ldy,
                        N,
                        mc,
                        stage,
                        rank_cap,
                        tp_cap,
                    )
                self.release_stage(cta, tidx, sst, stage)


# PTX building blocks.


def bulk_g2s(dst_smem: Int32, src_gmem: Int64, nbytes: Int32, mbar: Int32):
    inline_ptx(
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes "
        "[{$r0}], [{$r1}], {$r2}, [{$r3}];",
        read_only_args=[dst_smem, src_gmem, nbytes, mbar],
    )


def bulk_g2s_mc(
    dst_smem: Int32, src_gmem: Int64, nbytes: Int32, mbar: Int32, mask: Int32
):
    """``bulk_g2s`` into the same offset of every cluster CTA in ``mask``."""
    inline_ptx(
        "{ .reg .b16 m; cvt.u16.u32 m, {$r4};\n"
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        ".multicast::cluster [{$r0}], [{$r1}], {$r2}, [{$r3}], m; }",
        read_only_args=[dst_smem, src_gmem, nbytes, mbar, mask],
    )


def mbar_init(mbar: Int32, count: int):
    inline_ptx(
        "mbarrier.init.shared::cta.b64 [{$r0}], {$r1};",
        read_only_args=[mbar, Int32(count)],
    )


def mbar_arrive_expect_tx(mbar: Int32, nbytes: Int32):
    inline_ptx(
        "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [{$r0}], {$r1};",
        read_only_args=[mbar, nbytes],
    )


def mbar_arrive(mbar: Int32):
    inline_ptx(
        "mbarrier.arrive.release.cta.shared::cta.b64 _, [{$r0}];",
        read_only_args=[mbar],
    )


def mbar_try_wait(mbar: Int32, parity: Int32) -> Int32:
    return inline_ptx(
        "{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [{$r0}], {$r1}; "
        "selp.b32 {$w0}, 1, 0, p; }",
        write_only_types=[Int32],
        read_only_args=[mbar, parity],
    )


def fence_mbar_init():
    inline_ptx("fence.mbarrier_init.release.cluster;")


def cp_async16(dst_smem: Int32, src_gmem: Int64):
    inline_ptx(
        "cp.async.cg.shared.global [{$r0}], [{$r1}], 16;",
        read_only_args=[dst_smem, src_gmem],
    )


def evict_first_policy() -> Int64:
    return inline_ptx(
        "createpolicy.fractional.L2::evict_first.b64 {$w0}, 1.0;",
        write_only_types=[Int64],
    )


def cp_async16_hint(dst_smem: Int32, src_gmem: Int64, pol: Int64):
    inline_ptx(
        "cp.async.cg.shared.global.L2::cache_hint [{$r0}], [{$r1}], 16, {$r2};",
        read_only_args=[dst_smem, src_gmem, pol],
    )


def cp_async_arrive_noinc(mbar: Int32):
    inline_ptx(
        "cp.async.mbarrier.arrive.noinc.shared::cta.b64 [{$r0}];",
        read_only_args=[mbar],
    )


def lds_v2(addr: Int32):
    return inline_ptx(
        "ld.shared.v2.b32 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32, Int32],
        read_only_args=[addr],
    )


def lds_v4(addr: Int32):
    return inline_ptx(
        "ld.shared.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32] * 4,
        read_only_args=[addr],
    )


def lds_u16(addr: Int32) -> Int32:
    return inline_ptx(
        "{ .reg .b16 h; ld.shared.b16 h, [{$r0}]; cvt.u32.u16 {$w0}, h; }",
        write_only_types=[Int32],
        read_only_args=[addr],
    )


def lds_u32(addr: Int32) -> Int32:
    return inline_ptx(
        "ld.shared.b32 {$w0}, [{$r0}];",
        write_only_types=[Int32],
        read_only_args=[addr],
    )


def lds_f32x4(addr: Int32):
    return inline_ptx(
        "ld.shared.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32] * 4,
        read_only_args=[addr],
    )


def sts_u32(addr: Int32, v: Int32):
    inline_ptx("st.shared.b32 [{$r0}], {$r1};", read_only_args=[addr, v])


def sts_f32(addr: Int32, a):
    inline_ptx("st.shared.f32 [{$r0}], {$r1};", read_only_args=[addr, a])


def sts_f32x4(addr: Int32, a, b, c, d):
    inline_ptx(
        "st.shared.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, a, b, c, d],
    )


def sts_v4_b32(addr: Int32, a, b, c, d):
    inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, a, b, c, d],
    )


def ldmatrix_x4(addr: Int32):
    return inline_ptx(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32] * 4,
        read_only_args=[addr],
    )


def ldmatrix_x2(addr: Int32):
    return inline_ptx(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32] * 2,
        read_only_args=[addr],
    )


def mma_bf16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    return inline_ptx(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, "
        "{{$r4}, {$r5}}, {{$r6}, {$r7}, {$r8}, {$r9}};",
        write_only_types=[Float32] * 4,
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


def mma_f16(a0, a1, a2, a3, b0, b1, c0, c1, c2, c3):
    return inline_ptx(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, "
        "{{$r4}, {$r5}}, {{$r6}, {$r7}, {$r8}, {$r9}};",
        write_only_types=[Float32] * 4,
        read_only_args=[a0, a1, a2, a3, b0, b1, c0, c1, c2, c3],
    )


def mma_f16_z(a0, a1, a2, a3, b0, b1):
    return inline_ptx(
        "{ .reg .f32 z; mov.f32 z, 0f00000000;\n"
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{{$w0}, {$w1}, {$w2}, {$w3}}, {{$r0}, {$r1}, {$r2}, {$r3}}, "
        "{{$r4}, {$r5}}, {z, z, z, z}; }",
        write_only_types=[Float32] * 4,
        read_only_args=[a0, a1, a2, a3, b0, b1],
    )


def scale2_bf16x2(ev: Int32):
    """Two UE8M0 bytes (u16) -> BF16x2 ``2^(b - 7)`` each (the e4m3 -> bf16
    exponent bias folded in); needs ``b <= 134``."""
    return inline_ptx(
        "{ .reg .b32 t, u, z; mov.b32 z, 0;\n"
        "add.u32 t, {$r0}, 0x7878;\n"
        "prmt.b32 u, t, z, 0x4040;\n"
        "shl.b32 {$w0}, u, 7;\n"
        "prmt.b32 u, t, z, 0x4141;\n"
        "shl.b32 {$w1}, u, 7; }",
        write_only_types=[Int32, Int32],
        read_only_args=[ev],
    )


def deq_word(w: Int32, s: Int32):
    """Humming FP8 word ``[k0, k8, k1, k9]`` -> (BF16x2 ``{k0, k1}``, BF16x2
    ``{k8, k9}``), times the BF16x2 scale ``s``."""
    return inline_ptx(
        "{ .reg .b32 t0, t1, t2, z; mov.b32 z, 0;\n"
        "prmt.b32 t0, {$r0}, z, 0x2404;\n"
        "shr.u32 t1, t0, 4;\n"
        "and.b32 t1, t1, 0x07F007F0;\n"
        "and.b32 t0, t0, 0x80008000;\n"
        "or.b32 t0, t0, t1;\n"
        "shr.u32 t1, {$r0}, 4;\n"
        "and.b32 t1, t1, 0x07F007F0;\n"
        "and.b32 t2, {$r0}, 0x80008000;\n"
        "or.b32 t2, t2, t1;\n"
        "mul.rn.bf16x2 {$w0}, t0, {$r1};\n"
        "mul.rn.bf16x2 {$w1}, t2, {$r1}; }",
        write_only_types=[Int32, Int32],
        read_only_args=[w, s],
    )


def cvt_e4m3_word(w: Int32):
    """Humming FP8 word ``[k0, k8, k1, k9]`` -> (f16x2 ``{k0, k1}``, f16x2
    ``{k8, k9}``), exactly."""
    return inline_ptx(
        "{ .reg .b32 p; .reg .b16 l, h;\n"
        "prmt.b32 p, {$r0}, {$r0}, 0x3120;\n"
        "mov.b32 {l, h}, p;\n"
        "cvt.rn.f16x2.e4m3x2 {$w0}, l;\n"
        "cvt.rn.f16x2.e4m3x2 {$w1}, h; }",
        write_only_types=[Int32, Int32],
        read_only_args=[w],
    )


def e8m0_f32(e: Int32) -> Float32:
    return inline_ptx(
        "{ .reg .b32 t; shl.b32 t, {$r0}, 23; mov.b32 {$w0}, t; }",
        write_only_types=[Float32],
        read_only_args=[e],
    )


def bf16x8_absmax_bits(a0, a1, a2, a3) -> Int32:
    """The FP32 bit pattern of max |x| over 8 BF16 values."""
    return inline_ptx(
        "{ .reg .b32 m, t;\n"
        "and.b32 m, {$r0}, 0x7FFF7FFF;\n"
        "and.b32 t, {$r1}, 0x7FFF7FFF; max.u16x2 m, m, t;\n"
        "and.b32 t, {$r2}, 0x7FFF7FFF; max.u16x2 m, m, t;\n"
        "and.b32 t, {$r3}, 0x7FFF7FFF; max.u16x2 m, m, t;\n"
        "shr.u32 t, m, 16; and.b32 m, m, 0xFFFF; max.u32 m, m, t;\n"
        "shl.b32 {$w0}, m, 16; }",
        write_only_types=[Int32],
        read_only_args=[a0, a1, a2, a3],
    )


def bf16x2_to_f16x2_scaled(v: Int32, sc: Float32) -> Int32:
    return inline_ptx(
        "{ .reg .b32 lo, hi; .reg .f32 fl, fh;\n"
        "shl.b32 lo, {$r0}, 16; and.b32 hi, {$r0}, 0xFFFF0000;\n"
        "mov.b32 fl, lo; mov.b32 fh, hi;\n"
        "mul.rn.f32 fl, fl, {$r1}; mul.rn.f32 fh, fh, {$r1};\n"
        "cvt.rn.f16x2.f32 {$w0}, fh, fl; }",
        write_only_types=[Int32],
        read_only_args=[v, sc],
    )


def i2f_bits(v: Int32) -> Float32:
    return inline_ptx(
        "mov.b32 {$w0}, {$r0};", write_only_types=[Float32], read_only_args=[v]
    )


def redux_max_u32(v: Int32) -> Int32:
    return inline_ptx(
        "redux.sync.max.u32 {$w0}, {$r0}, 0xffffffff;",
        write_only_types=[Int32],
        read_only_args=[v],
    )


def named_bar_sync(bar_id: int, nthreads: int):
    inline_ptx(
        "bar.sync {$r0}, {$r1};", read_only_args=[Int32(bar_id), Int32(nthreads)]
    )


def mapa(addr: Int32, rank: Int32) -> Int32:
    return inline_ptx(
        "mapa.shared::cluster.u32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Int32],
        read_only_args=[addr, rank],
    )


def st_dsmem_f32x4(addr: Int32, a, b, c, d):
    inline_ptx(
        "st.shared::cluster.v4.f32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, a, b, c, d],
    )


def cluster_arrive_release():
    inline_ptx("barrier.cluster.arrive.release.aligned;")


def cluster_wait_acquire():
    inline_ptx("barrier.cluster.wait.acquire.aligned;")


def pack_bf16x2(a, b) -> Int32:
    return inline_ptx(
        "cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};",
        write_only_types=[Int32],
        read_only_args=[a, b],
    )


def stg_b32x2(addr: Int64, a, b):
    inline_ptx("st.global.v2.b32 [{$r0}], {{$r1}, {$r2}};", read_only_args=[addr, a, b])


def ld_volatile_u32(addr: Int64) -> Int32:
    return inline_ptx(
        "ld.volatile.global.u32 {$w0}, [{$r0}];",
        write_only_types=[Int32],
        read_only_args=[addr],
    )


def st_global_u32(addr: Int64, v: Int32):
    inline_ptx("st.global.u32 [{$r0}], {$r1};", read_only_args=[addr, v])


def stmc_b32x2(addr: Int64, a, b):
    """``multimem.st`` of 4 BF16 values into every rank's mailbox copy."""
    inline_ptx(
        "multimem.st.relaxed.sys.global.v2.bf16x2 [{$r0}], {{$r1}, {$r2}};",
        read_only_args=[addr, a, b],
    )


def clear_negzero_bf16x2(w: Int32) -> Int32:
    """BF16x2 with negative-zero halves (the Lamport sentinel) made +0."""
    return inline_ptx(
        "{ .reg .pred p; .reg .b32 t, r;\n"
        "mov.b32 r, {$r0};\n"
        "and.b32 t, r, 0xFFFF; setp.eq.u32 p, t, 0x8000; "
        "@p and.b32 r, r, 0xFFFF0000;\n"
        "shr.u32 t, r, 16; setp.eq.u32 p, t, 0x8000; "
        "@p and.b32 r, r, 0x0000FFFF;\n"
        "mov.b32 {$w0}, r; }",
        write_only_types=[Int32],
        read_only_args=[w],
    )
