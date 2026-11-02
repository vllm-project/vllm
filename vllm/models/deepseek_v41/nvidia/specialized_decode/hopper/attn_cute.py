# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PDL-early-trigger CuTe DSL replacements for the small kernels of the DSV4.1 SM90
attention chain (TP4: q_lora 1280, head_dim 512 = 448 NoPE + 64 RoPE, 16 local heads,
2 wo_a groups of 8 heads, V4 584 B fp8_ds_mla SWA record).

Every kernel calls griddepcontrol.launch_dependents at block start, loads only static
data (norm weights) before griddepcontrol.wait, reads activations after it with L2
(.cg) loads, and uses no or tiny smem, so the next kernel (a weight-preloading GEMV)
can become resident and stage its weights while this one runs.

  QkvNorm       fused_q_kv_rmsnorm + the KV half of
                fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert (GPT-J RoPE on dims
                448..511, UE8M0 FP8 quant, V4 record insert) straight from the
                fused_wqa_wkv output [M, 1792].
  QRope         the Q half of that op: GPT-J RoPE of the 16 live heads, copied
                into the padded [M, 64, 512] query buffer.
  CombineInvRope  ``mla_cute.Combine`` with fused_inv_rope_fp8_quant(quantize=False)
                folded in: merges the split partials with the sink, inverse GPT-J
                RoPE, writes the wo_a input [T, 2, 4096] (any token / group stride).

Host calls: ``attn_chain``.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.arch import inline_ptx

LOG2E = 1.4426950408889634
BM, DV = 64, 512
NOPE = 448


# ------------------------------------------------------------------ PTX helpers
def ldg4_cg(addr: Int64):
    return inline_ptx(
        "ld.global.cg.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32] * 4,
        read_only_args=[addr],
    )


def ldg4_nc(addr: Int64):
    return inline_ptx(
        "ld.global.nc.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32] * 4,
        read_only_args=[addr],
    )


def ldg2_cg(addr: Int64):
    return inline_ptx(
        "ld.global.cg.v2.b32 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32] * 2,
        read_only_args=[addr],
    )


def ldf4_nc(addr: Int64):
    return inline_ptx(
        "ld.global.nc.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32] * 4,
        read_only_args=[addr],
    )


def ldf_nc(addr: Int64) -> Float32:
    return inline_ptx(
        "ld.global.nc.f32 {$w0}, [{$r0}];",
        write_only_types=[Float32],
        read_only_args=[addr],
    )


def ldi64_cg(addr: Int64) -> Int64:
    return inline_ptx(
        "ld.global.cg.b64 {$w0}, [{$r0}];",
        write_only_types=[Int64],
        read_only_args=[addr],
    )


def stg4(addr: Int64, a, b, c, d):
    inline_ptx(
        "st.global.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, a, b, c, d],
    )


def stg2(addr: Int64, a, b):
    inline_ptx("st.global.v2.b32 [{$r0}], {{$r1}, {$r2}};", read_only_args=[addr, a, b])


def stg_u8(addr: Int64, v: Int32):
    inline_ptx(
        "{ .reg .b16 h; cvt.u16.u32 h, {$r1}; st.global.b8 [{$r0}], h; }",
        read_only_args=[addr, v],
    )


def bf2(w: Int32):
    """bf16x2 word -> (lo, hi) fp32."""
    return bitsf(w << 16), bitsf(w & Int32(-65536))


def pk(lo, hi) -> Int32:
    """(lo, hi) fp32 -> bf16x2 (RN)."""
    return inline_ptx(
        "cvt.rn.bf16x2.f32 {$w0}, {$r1}, {$r0};",
        write_only_types=[Int32],
        read_only_args=[lo, hi],
    )


def rnd(x) -> Float32:
    """Round fp32 to bf16 and back."""
    return bitsf(
        inline_ptx(
            "{ .reg .b16 h, z; mov.b16 z, 0; cvt.rn.bf16.f32 h, {$r0};"
            " mov.b32 {$w0}, {z, h}; }",
            write_only_types=[Int32],
            read_only_args=[x],
        )
    )


def rsqrt_approx(x) -> Float32:
    return inline_ptx(
        "rsqrt.approx.f32 {$w0}, {$r0};", write_only_types=[Float32], read_only_args=[x]
    )


def div_rn(a, b) -> Float32:
    return inline_ptx(
        "div.rn.f32 {$w0}, {$r0}, {$r1};",
        write_only_types=[Float32],
        read_only_args=[a, b],
    )


def fbits(x) -> Int32:
    return inline_ptx(
        "mov.b32 {$w0}, {$r0};", write_only_types=[Int32], read_only_args=[x]
    )


def bitsf(x) -> Float32:
    return inline_ptx(
        "mov.b32 {$w0}, {$r0};", write_only_types=[Float32], read_only_args=[x]
    )


def e4m3x4(a, b, c, d) -> Int32:
    """4 fp32 -> 4 e4m3 bytes (a at the lowest address), RN satfinite."""
    return inline_ptx(
        "{ .reg .b16 l, h; cvt.rn.satfinite.e4m3x2.f32 l, {$r1}, {$r0};"
        " cvt.rn.satfinite.e4m3x2.f32 h, {$r3}, {$r2}; mov.b32 {$w0}, {l, h}; }",
        write_only_types=[Int32],
        read_only_args=[a, b, c, d],
    )


def warp_sum(v):
    for o in (16, 8, 4, 2, 1):
        v = v + cute.arch.shuffle_sync_bfly(v, offset=o)
    return v


def fmaxf(a, b):
    return cutlass.max(a, b)


def fminf(a, b):
    return cutlass.min(a, b)


# ---------------------------------------------------------------- QkvNorm (K1)
class QkvNorm:
    """grid = M CTAs x 64 threads. Warp 0: qr = RMSNorm(qkv[:, :1280]) * wq -> bf16.
    Warp 1: kv = RMSNorm(qkv[:, 1280:]) * wkv rounded to bf16; then the stock V4 KV
    insert (GPT-J RoPE on 448..511, 64-dim UE8M0 FP8 blocks, bf16 RoPE). Lane L of
    warp 1 owns dims [16L, 16L+16) (stock map). ``kvo`` is unused."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    @cute.jit
    def __call__(
        self,
        qkv: Int64,
        ld_qkv: Int32,
        qr: Int64,
        ld_qr: Int32,
        kvo: Int64,
        wq: Int64,
        wkv: Int64,
        cache: Int64,
        bstride: Int64,
        bs: Int32,
        slots: Int64,
        pos: Int64,
        cs: Int64,
        M: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            qkv, ld_qkv, qr, ld_qr, kvo, wq, wkv, cache, bstride, bs, slots, pos, cs
        ).launch(grid=[M, 1, 1], block=[64, 1, 1], stream=stream, use_pdl=True)

    @cute.kernel
    def kernel(
        self,
        qkv: Int64,
        ld_qkv: Int32,
        qr: Int64,
        ld_qr: Int32,
        kvo: Int64,
        wq: Int64,
        wkv: Int64,
        cache: Int64,
        bstride: Int64,
        bs: Int32,
        slots: Int64,
        pos: Int64,
        cs: Int64,
    ):
        tid, _, _ = cute.arch.thread_idx()
        t, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_launch_dependents()
        warp = tid // 32
        lane = tid % 32
        row = qkv + Int64(t) * Int64(ld_qkv) * 2
        if warp == 0:
            w = cute.make_rmem_tensor(20, Int32)
            x = cute.make_rmem_tensor(20, Int32)
            for i in cutlass.range_constexpr(5):
                w0, w1, w2, w3 = ldg4_nc(wq + Int64((lane + 32 * i) * 16))
                w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3] = w0, w1, w2, w3
            cute.arch.griddepcontrol_wait()
            for i in cutlass.range_constexpr(5):
                x0, x1, x2, x3 = ldg4_cg(row + Int64((lane + 32 * i) * 16))
                x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3] = x0, x1, x2, x3
            ss = Float32(0.0)
            for j in cutlass.range_constexpr(20):
                lo, hi = bf2(x[j])
                ss = ss + lo * lo + hi * hi
            ss = warp_sum(ss)
            rs = rsqrt_approx(ss / Float32(1280.0) + Float32(self.eps))
            o = cute.make_rmem_tensor(20, Int32)
            for j in cutlass.range_constexpr(20):
                lo, hi = bf2(x[j])
                wl, wh = bf2(w[j])
                o[j] = pk(lo * rs * wl, hi * rs * wh)
            orow = qr + Int64(t) * Int64(ld_qr) * 2
            for i in cutlass.range_constexpr(5):
                stg4(
                    orow + Int64((lane + 32 * i) * 16),
                    o[4 * i],
                    o[4 * i + 1],
                    o[4 * i + 2],
                    o[4 * i + 3],
                )
        else:
            w = cute.make_rmem_tensor(8, Int32)
            x = cute.make_rmem_tensor(8, Int32)
            for i in cutlass.range_constexpr(2):
                w0, w1, w2, w3 = ldg4_nc(wkv + Int64(lane * 32 + i * 16))
                w[4 * i], w[4 * i + 1], w[4 * i + 2], w[4 * i + 3] = w0, w1, w2, w3
            cute.arch.griddepcontrol_wait()
            for i in cutlass.range_constexpr(2):
                x0, x1, x2, x3 = ldg4_cg(row + Int64(1280 * 2 + lane * 32 + i * 16))
                x[4 * i], x[4 * i + 1], x[4 * i + 2], x[4 * i + 3] = x0, x1, x2, x3
            ss = Float32(0.0)
            for j in cutlass.range_constexpr(8):
                lo, hi = bf2(x[j])
                ss = ss + lo * lo + hi * hi
            ss = warp_sum(ss)
            rs = rsqrt_approx(ss / Float32(512.0) + Float32(self.eps))
            e = cute.make_rmem_tensor(16, Float32)
            for j in cutlass.range_constexpr(8):
                lo, hi = bf2(x[j])
                wl, wh = bf2(w[j])
                e[2 * j] = rnd(lo * rs * wl)
                e[2 * j + 1] = rnd(hi * rs * wh)
            slot = ldi64_cg(slots + Int64(t) * 8)
            is_rope = lane >= 28
            if is_rope:
                p = ldi64_cg(pos + Int64(t) * 8)
                cb = cs + (p * Int64(64) + Int64((lane - 28) * 8)) * 4
                c0, c1, c2, c3 = ldf4_nc(cb)
                c4, c5, c6, c7 = ldf4_nc(cb + 16)
                s0, s1, s2, s3 = ldf4_nc(cb + 128)
                s4, s5, s6, s7 = ldf4_nc(cb + 144)
                cc = [c0, c1, c2, c3, c4, c5, c6, c7]
                sn = [s0, s1, s2, s3, s4, s5, s6, s7]
                for j in cutlass.range_constexpr(8):
                    xe = e[2 * j]
                    xo = e[2 * j + 1]
                    e[2 * j] = rnd(xe * cc[j] - xo * sn[j])
                    e[2 * j + 1] = rnd(xe * sn[j] + xo * cc[j])
            am = Float32(0.0)
            for j in cutlass.range_constexpr(16):
                am = fmaxf(am, cutlass.max(e[j], -e[j]))
            am = fmaxf(am, cute.arch.shuffle_sync_bfly(am, offset=1))
            am = fmaxf(am, cute.arch.shuffle_sync_bfly(am, offset=2))
            am = fmaxf(am, Float32(1e-4))
            r = div_rn(am, Float32(448.0))
            rb = fbits(r)
            ex = (rb >> 23) - 127
            if (rb & 0x7FFFFF) != 0:
                ex = ex + 1
            inv = bitsf((127 - ex) << 23)
            if slot >= 0:
                blk = slot // Int64(bs)
                pin = slot - blk * Int64(bs)
                base = cache + blk * bstride
                data = base + pin * Int64(576)
                if is_rope:
                    b = data + Int64(NOPE + (lane * 16 - NOPE) * 2)
                    stg4(
                        b,
                        pk(e[0], e[1]),
                        pk(e[2], e[3]),
                        pk(e[4], e[5]),
                        pk(e[6], e[7]),
                    )
                    stg4(
                        b + 16,
                        pk(e[8], e[9]),
                        pk(e[10], e[11]),
                        pk(e[12], e[13]),
                        pk(e[14], e[15]),
                    )
                else:
                    q = cute.make_rmem_tensor(4, Int32)
                    for j in cutlass.range_constexpr(4):
                        q[j] = e4m3x4(
                            fminf(fmaxf(e[4 * j] * inv, -448.0), 448.0),
                            fminf(fmaxf(e[4 * j + 1] * inv, -448.0), 448.0),
                            fminf(fmaxf(e[4 * j + 2] * inv, -448.0), 448.0),
                            fminf(fmaxf(e[4 * j + 3] * inv, -448.0), 448.0),
                        )
                    stg4(data + Int64(lane * 16), q[0], q[1], q[2], q[3])
                    sc = base + Int64(bs) * Int64(576) + pin * Int64(8)
                    if lane % 4 == 0:
                        enc = cutlass.min(cutlass.max(ex + 127, 0), 255)
                        stg_u8(sc + Int64(lane // 4), enc)
                    if lane == 0:
                        stg_u8(sc + Int64(7), Int32(0))


# ------------------------------------------------------------------ QRope (K2)
class QRope:
    """grid = M CTAs x 512 threads (warp = live head, lane L = dims [16L, 16L+16)).
    Copies q_in [M, 16, 512] (row stride ld_in) into q_out, whose heads 16..63
    stay zero (a pre-zeroed, persistent q_out)."""

    @cute.jit
    def __call__(
        self,
        q_in: Int64,
        ld_in: Int32,
        q_out: Int64,
        pos: Int64,
        cs: Int64,
        M: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(q_in, ld_in, q_out, pos, cs).launch(
            grid=[M, 1, 1], block=[512, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(self, q_in: Int64, ld_in: Int32, q_out: Int64, pos: Int64, cs: Int64):
        tid, _, _ = cute.arch.thread_idx()
        t, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_launch_dependents()
        h = tid // 32
        lane = tid % 32
        dst = q_out + (Int64(t) * Int64(64) + Int64(h)) * Int64(1024) + Int64(lane * 32)
        src = q_in + Int64(t) * Int64(ld_in) * 2 + Int64(h * 1024 + lane * 32)
        cute.arch.griddepcontrol_wait()
        x0, x1, x2, x3 = ldg4_cg(src)
        x4, x5, x6, x7 = ldg4_cg(src + 16)
        xs = [x0, x1, x2, x3, x4, x5, x6, x7]
        if lane >= 28:
            p = ldi64_cg(pos + Int64(t) * 8)
            cb = cs + (p * Int64(64) + Int64((lane - 28) * 8)) * 4
            c0, c1, c2, c3 = ldf4_nc(cb)
            c4, c5, c6, c7 = ldf4_nc(cb + 16)
            s0, s1, s2, s3 = ldf4_nc(cb + 128)
            s4, s5, s6, s7 = ldf4_nc(cb + 144)
            cc = [c0, c1, c2, c3, c4, c5, c6, c7]
            sn = [s0, s1, s2, s3, s4, s5, s6, s7]
            ys = []
            for j in cutlass.range_constexpr(8):
                xe, xo = bf2(xs[j])
                ys.append(pk(xe * cc[j] - xo * sn[j], xe * sn[j] + xo * cc[j]))
            stg4(dst, ys[0], ys[1], ys[2], ys[3])
            stg4(dst + 16, ys[4], ys[5], ys[6], ys[7])
        else:
            stg4(dst, x0, x1, x2, x3)
            stg4(dst + 16, x4, x5, x6, x7)


# ------------------------------------------------------- CombineInvRope (K3)
class CombineInvRope:
    """``mla_cute.Combine`` (4 warps per real (row, head), 128 columns each) whose
    output is rounded to bf16 (= the stock `out`), inverse-GPT-J-rotated on dims
    448..511 and stored at rot[t * rot_t + (h // 8) * rot_g + (h % 8) * 512 + col]
    (the wo_a input; stock fused_inv_rope_fp8_quant(quantize=False) is [G, T, 4096])."""

    def __init__(self, H: int = 16, S: int = 40):
        self.H, self.S = H, S
        self.rpg = BM // H

    @cute.jit
    def __call__(
        self,
        work_o: Int64,
        work_ml: Int64,
        sink_ptr: Int64,
        rot: Int64,
        rot_t: Int32,
        rot_g: Int32,
        pos: Int64,
        cs: Int64,
        R: Int32,
        ngr: Int32,
        G: Int32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class Smem:
            w: cute.struct.MemRange[Float32, 8 * 64]

        self.kernel(
            work_o, work_ml, sink_ptr, rot, rot_t, rot_g, pos, cs, R, ngr, Smem
        ).launch(
            grid=[G * BM * 4 // 8, 1, 1], block=[256, 1, 1], stream=stream, use_pdl=True
        )

    @cute.kernel
    def kernel(
        self,
        work_o: Int64,
        work_ml: Int64,
        sink_ptr: Int64,
        rot: Int64,
        rot_t: Int32,
        rot_g: Int32,
        pos: Int64,
        cs: Int64,
        R: Int32,
        ngr: Int32,
        Smem: cutlass.Constexpr,
    ):
        H, SPL, rpg = self.H, self.S, self.rpg
        tid, _, _ = cute.arch.thread_idx()
        bx, _, _ = cute.arch.block_idx()
        lane = tid % 32
        warp = tid // 32
        cute.arch.griddepcontrol_launch_dependents()
        smem = cutlass.utils.SmemAllocator()
        st = smem.allocate(Smem)
        sw = st.w.get_tensor(cute.make_layout(8 * 64))
        gw = bx * 8 + warp
        q4 = gw % 4
        gr = gw // 4
        g = gr // BM
        m = gr % BM
        b = g // ngr
        kg = g - b * ngr
        nrows = cutlass.min(rpg, R - kg * rpg)
        rr = m // H
        hh = m % H
        sk = ldf_nc(sink_ptr + Int64(hh) * 4) * LOG2E
        col = q4 * 128 + lane * 4
        rope = col >= NOPE
        cv = cute.make_rmem_tensor(4, Float32)
        cute.arch.griddepcontrol_wait()
        if rr < nrows:
            t = b * R + kg * rpg + rr
            if rope:
                p = ldi64_cg(pos + Int64(t) * 8)
                cb = cs + (p * Int64(64) + Int64((col - NOPE) // 2)) * 4
                cv[0] = ldf_nc(cb)
                cv[1] = ldf_nc(cb + 4)
                cv[2] = ldf_nc(cb + 128)
                cv[3] = ldf_nc(cb + 132)
            ms = cute.make_rmem_tensor(2, Float32)
            ls = cute.make_rmem_tensor(2, Float32)
            for k in cutlass.range_constexpr(2):
                s = lane + 32 * k
                mv = Float32(-1e30)
                lv = Float32(0.0)
                if s < SPL:
                    x0, x1 = ldg2_cg(work_ml + (Int64((g * SPL + s) * BM + m) * 2) * 4)
                    mv = bitsf(x0)
                    lv = bitsf(x1)
                ms[k] = mv
                ls[k] = lv
            mx = Float32(-1e30)
            for k in cutlass.range_constexpr(2):
                if ls[k] > 0.0:
                    mx = cutlass.max(mx, ms[k])
            for o in cutlass.range_constexpr(5):
                mx = cutlass.max(mx, cute.arch.shuffle_sync_bfly(mx, offset=1 << o))
            den = Float32(0.0)
            for k in cutlass.range_constexpr(2):
                w = Float32(0.0)
                if ls[k] > 0.0:
                    w = cute.arch.exp2(ms[k] - mx) * ls[k]
                ms[k] = w
                den += w
            for o in cutlass.range_constexpr(5):
                den = den + cute.arch.shuffle_sync_bfly(den, offset=1 << o)
            inv = Float32(0.0)
            if den > 0.0:
                inv = 1.0 / (den + cute.arch.exp2(sk - mx))
            for k in cutlass.range_constexpr(2):
                sw[warp * 64 + lane + 32 * k] = ms[k] * inv
            cute.arch.sync_warp()
            off = (m * DV + (((col // 8) ^ (m % 8)) * 8) + col % 8) * 2
            xs = cute.make_rmem_tensor(SPL * 2, Int32)
            for s in cutlass.range_constexpr(SPL):
                a0 = Int32(0)
                a1 = Int32(0)
                if sw[warp * 64 + s] > 0.0:
                    a0, a1 = ldg2_cg(
                        work_o + Int64(g * SPL + s) * Int64(BM * DV * 2) + Int64(off)
                    )
                xs[2 * s] = a0
                xs[2 * s + 1] = a1
            acc = cute.make_rmem_tensor(4, Float32)
            acc.fill(0.0)
            for s in cutlass.range_constexpr(SPL):
                w = sw[warp * 64 + s]
                for j in cutlass.range_constexpr(2):
                    lo, hi = bf2(xs[2 * s + j])
                    acc[2 * j] = acc[2 * j] + w * lo
                    acc[2 * j + 1] = acc[2 * j + 1] + w * hi
            for j in cutlass.range_constexpr(4):
                acc[j] = rnd(acc[j])
            if rope:
                for j in cutlass.range_constexpr(2):
                    xe = acc[2 * j]
                    xo = acc[2 * j + 1]
                    c = cv[j]
                    s_ = cv[2 + j]
                    acc[2 * j] = xe * c + xo * s_
                    acc[2 * j + 1] = xo * c - xe * s_
            orow = (
                rot
                + (
                    Int64(t) * Int64(rot_t)
                    + Int64((hh // 8) * rot_g + (hh % 8) * 512 + col)
                )
                * 2
            )
            stg2(orow, pk(acc[0], acc[1]), pk(acc[2], acc[3]))
