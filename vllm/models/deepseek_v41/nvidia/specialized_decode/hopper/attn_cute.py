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

Host calls: `attn_chain`.
"""

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.arch import inline_ptx

from .mla_ptx import (
    bf16x2_to_f16x2,
    dq_e4m3x4_f16,
    dq_e4m3x4_f16_2,
    st_global_v4,
)

LOG2E = 1.4426950408889634
BM, DV = 64, 512
NOPE = 448
# mla_cute's tile images: blocks of 64 keys x 512 fp16 dims, SW128 K-major;
# up to NSWB SWA blocks and NBC compressed blocks per row group.
IMG, NSWB, NBC, RPG = 64 * 512, 8, 32, 4


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


def ldf4_nc(addr: Int64):
    return inline_ptx(
        "ld.global.nc.v4.f32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Float32] * 4,
        read_only_args=[addr],
    )


def ldg_b32(addr: Int64) -> Int32:
    return inline_ptx(
        "ld.global.cg.b32 {$w0}, [{$r0}];",
        write_only_types=[Int32],
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
    warp 1 owns dims [16L, 16L+16) (stock map). `kvo` is unused.

    Warp 1 also writes the inserted key, dequantized as `mla_cute.SwaImage`
    would, into the `mla_cute.ChainDecode` SWA tile image `img` of each row
    group of its request whose union holds the key's slot (found before the
    wait)."""

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
        img: Int64,
        kidx: Int64,
        klen: Int64,
        cap: Int32,
        R: Int32,
        ngr: Int32,
        M: Int32,
        stream: cuda.CUstream,
    ):
        self.kernel(
            qkv,
            ld_qkv,
            qr,
            ld_qr,
            kvo,
            wq,
            wkv,
            cache,
            bstride,
            bs,
            slots,
            pos,
            cs,
            img,
            kidx,
            klen,
            cap,
            R,
            ngr,
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
        img: Int64,
        kidx: Int64,
        klen: Int64,
        cap: Int32,
        R: Int32,
        ngr: Int32,
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
            slot = ldi64_cg(slots + Int64(t) * 8)
            # The RoPE lanes' cos / sin (static: before the wait).
            is_rope = lane >= 28
            cv = cute.make_rmem_tensor(16, Float32)
            if is_rope:
                p = ldi64_cg(pos + Int64(t) * 8)
                cb = cs + (p * Int64(64) + Int64((lane - 28) * 8)) * 4
                for k in cutlass.range_constexpr(2):
                    c0, c1, c2, c3 = ldf4_nc(cb + 16 * k)
                    s0, s1, s2, s3 = ldf4_nc(cb + 128 + 16 * k)
                    cv[4 * k], cv[4 * k + 1], cv[4 * k + 2], cv[4 * k + 3] = (
                        c0,
                        c1,
                        c2,
                        c3,
                    )
                    cv[8 + 4 * k], cv[9 + 4 * k] = s0, s1
                    cv[10 + 4 * k], cv[11 + 4 * k] = s2, s3
            up = cute.make_rmem_tensor(2, Int32)
            up.fill(-1)
            self._find(up, t, slot, lane, kidx, klen, cap, R, ngr)
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
            if is_rope:
                for j in cutlass.range_constexpr(8):
                    xe = e[2 * j]
                    xo = e[2 * j + 1]
                    e[2 * j] = rnd(xe * cv[j] - xo * cv[8 + j])
                    e[2 * j + 1] = rnd(xe * cv[8 + j] + xo * cv[j])
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
                    f = cute.make_rmem_tensor(8, Int32)
                    for k in cutlass.range_constexpr(8):
                        f[k] = Int32(bf16x2_to_f16x2(pk(e[2 * k], e[2 * k + 1])))
                    self._put(up, f, t, lane, img, R, ngr)
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
                    f = cute.make_rmem_tensor(8, Int32)
                    self._dequant(f, q, ex)
                    self._put(up, f, t, lane, img, R, ngr)
                    sc = base + Int64(bs) * Int64(576) + pin * Int64(8)
                    if lane % 4 == 0:
                        enc = cutlass.min(cutlass.max(ex + 127, 0), 255)
                        stg_u8(sc + Int64(lane // 4), enc)
                    if lane == 0:
                        stg_u8(sc + Int64(7), Int32(0))

    @cute.jit
    def _find(self, up, t, slot, lane, kidx, klen, cap, R, ngr):
        """up[kg]: the position of `slot` in the SWA union of row group kg of
        row t's request, or -1."""
        b = t // R
        for kg in cutlass.range_constexpr(2):
            if kg < ngr:
                g = b * ngr + kg
                kl = ldg_b32(klen + Int64(g) * 4)
                e = cute.make_rmem_tensor(NSWB * 2, Int32)
                for i in cutlass.range_constexpr(NSWB * 2):
                    u = i * 32 + lane
                    e[i] = Int32(-2)
                    if u < kl:
                        e[i] = ldg_b32(kidx + (Int64(g) * Int64(cap) + Int64(u)) * 4)
                hit = Int32(-1)
                for i in cutlass.range_constexpr(NSWB * 2):
                    if slot >= 0 and Int64(e[i]) == slot:
                        hit = i * 32 + lane
                for o in (16, 8, 4, 2, 1):
                    hit = cutlass.max(hit, cute.arch.shuffle_sync_bfly(hit, offset=o))
                up[kg] = hit

    @cute.jit
    def _dequant(self, f, q, ex):
        """`mla_cute._Common._store_token_any` of this lane's 16 e4m3 values
        (scale 2^ex): 8 fp16x2 words."""
        k = cutlass.min(cutlass.min(cutlass.max(ex + 127, 0), 255) - 127, 15)
        s1 = cutlass.Uint32(cutlass.max(k + 15, 1) << 10)
        s1 = s1 | (s1 << 16)
        if k < -14:
            k2 = k + 14
            s2 = cutlass.Uint32(0)
            if k2 >= -14:
                s2 = cutlass.Uint32((k2 + 15) << 10)
            elif k2 >= -24:
                s2 = cutlass.Uint32(1 << (k2 + 24))
            s2 = s2 | (s2 << 16)
            for j in cutlass.range_constexpr(4):
                lo, hi = dq_e4m3x4_f16_2(cutlass.Uint32(q[j]), s1, s2)
                f[2 * j] = Int32(lo)
                f[2 * j + 1] = Int32(hi)
        else:
            for j in cutlass.range_constexpr(4):
                lo, hi = dq_e4m3x4_f16(cutlass.Uint32(q[j]), s1)
                f[2 * j] = Int32(lo)
                f[2 * j + 1] = Int32(hi)

    @cute.jit
    def _put(self, up, f, t, lane, img, R, ngr):
        """This lane's 16 fp16 dims (chunks 2 * lane, 2 * lane + 1) of the key
        into row up[kg] of each group's SWA tile image."""
        b = t // R
        for kg in cutlass.range_constexpr(2):
            u = up[kg]
            if u >= 0:
                base = img + Int64((b * ngr + kg) * NSWB + u // 64) * Int64(IMG * 2)
                m = u % 64
                for h in cutlass.range_constexpr(2):
                    cc = 2 * lane + h
                    off = ((m // 8) + (cc // 8) * 8) * 512 + (m % 8) * 64
                    off = off + ((cc % 8) ^ (m % 8)) * 8
                    st_global_v4(
                        base + Int64(off) * 2,
                        cutlass.Uint32(f[4 * h]),
                        cutlass.Uint32(f[4 * h + 1]),
                        cutlass.Uint32(f[4 * h + 2]),
                        cutlass.Uint32(f[4 * h + 3]),
                    )
