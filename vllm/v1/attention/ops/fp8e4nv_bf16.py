# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Software fp8e4m3 (E4M3FN / fp8e4nv) <-> bf16 conversion for Triton on pre-SM89.

SM80/SM86 have no native fp8e4nv cast; these helpers implement it explicitly via
tl.inline_asm_elementwise (PTX), requiring NO change to triton-lang.
"""

# ---------------------------------------------------------------------------
# Encode: bf16 -> fp8e4m3  (write path). Round-to-nearest-even; bit-exact vs
# torch over all finite bf16.
#
# Decode: fp8e4m3 -> bf16  (read path). Exact expansion, no rounding; prmt-as-LUT
# decode, exact for all 256 bytes (NaN bytes 0x7f/0xff -> large-finite).
# ---------------------------------------------------------------------------

from vllm.triton_utils import tl

# ---- encode: bf16 -> fp8e4m3, RNE (round-to-nearest-even) -------------------
_BF16_TO_FP8E4M3_RNE_ASM = """\
{
    .reg .u16 b<4>;
    .reg .u32 raw<4>;
    .reg .u32 a0, e0, m0, r0, norm0, sub0, sh0, sh20, rnd0, ec0;
    .reg .u32 sgn0, tmp0, o0, ntmp0, ndec0, one0, shp10, mask0, rem0, sdec0;
    .reg .u32 a1, e1, m1, r1, norm1, sub1, sh1, sh21, rnd1, ec1;
    .reg .u32 sgn1, tmp1, o1, ntmp1, ndec1, one1, shp11, mask1, rem1, sdec1;
    .reg .u32 a2, e2, m2, r2, norm2, sub2, sh2, sh22, rnd2, ec2;
    .reg .u32 sgn2, tmp2, o2, ntmp2, ndec2, one2, shp12, mask2, rem2, sdec2;
    .reg .u32 a3, e3, m3, r3, norm3, sub3, sh3, sh23, rnd3, ec3;
    .reg .u32 sgn3, tmp3, o3, ntmp3, ndec3, one3, shp13, mask3, rem3, sdec3;
    .reg .u32 out;
    .reg .pred p_hi0, p_norm0, p_tiny0, p_ntie0, p_stie0;
    .reg .pred p_hi1, p_norm1, p_tiny1, p_ntie1, p_stie1;
    .reg .pred p_hi2, p_norm2, p_tiny2, p_ntie2, p_stie2;
    .reg .pred p_hi3, p_norm3, p_tiny3, p_ntie3, p_stie3;

    mov.b32 {b0, b1}, $1;
    mov.b32 {b2, b3}, $2;
    cvt.u32.u16 raw0, b0;
    cvt.u32.u16 raw1, b1;
    cvt.u32.u16 raw2, b2;
    cvt.u32.u16 raw3, b3;

    and.b32  a0, raw0, 0x7fff;
    and.b32  sgn0, raw0, 0x8000;
    shr.u32  sgn0, sgn0, 8;
    shr.u32  e0, a0, 7;
    and.b32  m0, a0, 0x7f;
    add.u32  r0, m0, 8;
    shr.u32  r0, r0, 4;
    and.b32  ntmp0, m0, 0x1f;
    setp.eq.u32 p_ntie0, ntmp0, 8;
    sub.u32  ndec0, r0, 1;
    selp.u32 r0, ndec0, r0, p_ntie0;
    sub.u32  norm0, e0, 120;
    shl.b32  norm0, norm0, 3;
    add.u32  norm0, norm0, r0;
    min.u32  norm0, norm0, 0x7e;
    max.u32  ec0, e0, 117;
    sub.u32  sh0, 125, ec0;
    sub.u32  sh20, sh0, 1;
    mov.u32  rnd0, 1;
    shl.b32  rnd0, rnd0, sh20;
    or.b32   tmp0, m0, 0x80;
    add.u32  sub0, tmp0, rnd0;
    shr.u32  sub0, sub0, sh0;
    add.u32  shp10, sh0, 1;
    mov.u32  one0, 1;
    shl.b32  mask0, one0, shp10;
    sub.u32  mask0, mask0, 1;
    and.b32  rem0, tmp0, mask0;
    setp.eq.u32 p_stie0, rem0, rnd0;
    sub.u32  sdec0, sub0, 1;
    selp.u32 sub0, sdec0, sub0, p_stie0;
    min.u32  sub0, sub0, 8;
    setp.lt.u32 p_tiny0, e0, 117;
    selp.u32 sub0, 0, sub0, p_tiny0;
    setp.gt.u32 p_norm0, e0, 120;
    selp.u32 o0, norm0, sub0, p_norm0;
    setp.ge.u32 p_hi0, a0, 0x43e0;
    selp.u32 o0, 0x7e, o0, p_hi0;
    or.b32 o0, o0, sgn0;

    and.b32  a1, raw1, 0x7fff;
    and.b32  sgn1, raw1, 0x8000;
    shr.u32  sgn1, sgn1, 8;
    shr.u32  e1, a1, 7;
    and.b32  m1, a1, 0x7f;
    add.u32  r1, m1, 8;
    shr.u32  r1, r1, 4;
    and.b32  ntmp1, m1, 0x1f;
    setp.eq.u32 p_ntie1, ntmp1, 8;
    sub.u32  ndec1, r1, 1;
    selp.u32 r1, ndec1, r1, p_ntie1;
    sub.u32  norm1, e1, 120;
    shl.b32  norm1, norm1, 3;
    add.u32  norm1, norm1, r1;
    min.u32  norm1, norm1, 0x7e;
    max.u32  ec1, e1, 117;
    sub.u32  sh1, 125, ec1;
    sub.u32  sh21, sh1, 1;
    mov.u32  rnd1, 1;
    shl.b32  rnd1, rnd1, sh21;
    or.b32   tmp1, m1, 0x80;
    add.u32  sub1, tmp1, rnd1;
    shr.u32  sub1, sub1, sh1;
    add.u32  shp11, sh1, 1;
    mov.u32  one1, 1;
    shl.b32  mask1, one1, shp11;
    sub.u32  mask1, mask1, 1;
    and.b32  rem1, tmp1, mask1;
    setp.eq.u32 p_stie1, rem1, rnd1;
    sub.u32  sdec1, sub1, 1;
    selp.u32 sub1, sdec1, sub1, p_stie1;
    min.u32  sub1, sub1, 8;
    setp.lt.u32 p_tiny1, e1, 117;
    selp.u32 sub1, 0, sub1, p_tiny1;
    setp.gt.u32 p_norm1, e1, 120;
    selp.u32 o1, norm1, sub1, p_norm1;
    setp.ge.u32 p_hi1, a1, 0x43e0;
    selp.u32 o1, 0x7e, o1, p_hi1;
    or.b32 o1, o1, sgn1;

    and.b32  a2, raw2, 0x7fff;
    and.b32  sgn2, raw2, 0x8000;
    shr.u32  sgn2, sgn2, 8;
    shr.u32  e2, a2, 7;
    and.b32  m2, a2, 0x7f;
    add.u32  r2, m2, 8;
    shr.u32  r2, r2, 4;
    and.b32  ntmp2, m2, 0x1f;
    setp.eq.u32 p_ntie2, ntmp2, 8;
    sub.u32  ndec2, r2, 1;
    selp.u32 r2, ndec2, r2, p_ntie2;
    sub.u32  norm2, e2, 120;
    shl.b32  norm2, norm2, 3;
    add.u32  norm2, norm2, r2;
    min.u32  norm2, norm2, 0x7e;
    max.u32  ec2, e2, 117;
    sub.u32  sh2, 125, ec2;
    sub.u32  sh22, sh2, 1;
    mov.u32  rnd2, 1;
    shl.b32  rnd2, rnd2, sh22;
    or.b32   tmp2, m2, 0x80;
    add.u32  sub2, tmp2, rnd2;
    shr.u32  sub2, sub2, sh2;
    add.u32  shp12, sh2, 1;
    mov.u32  one2, 1;
    shl.b32  mask2, one2, shp12;
    sub.u32  mask2, mask2, 1;
    and.b32  rem2, tmp2, mask2;
    setp.eq.u32 p_stie2, rem2, rnd2;
    sub.u32  sdec2, sub2, 1;
    selp.u32 sub2, sdec2, sub2, p_stie2;
    min.u32  sub2, sub2, 8;
    setp.lt.u32 p_tiny2, e2, 117;
    selp.u32 sub2, 0, sub2, p_tiny2;
    setp.gt.u32 p_norm2, e2, 120;
    selp.u32 o2, norm2, sub2, p_norm2;
    setp.ge.u32 p_hi2, a2, 0x43e0;
    selp.u32 o2, 0x7e, o2, p_hi2;
    or.b32 o2, o2, sgn2;

    and.b32  a3, raw3, 0x7fff;
    and.b32  sgn3, raw3, 0x8000;
    shr.u32  sgn3, sgn3, 8;
    shr.u32  e3, a3, 7;
    and.b32  m3, a3, 0x7f;
    add.u32  r3, m3, 8;
    shr.u32  r3, r3, 4;
    and.b32  ntmp3, m3, 0x1f;
    setp.eq.u32 p_ntie3, ntmp3, 8;
    sub.u32  ndec3, r3, 1;
    selp.u32 r3, ndec3, r3, p_ntie3;
    sub.u32  norm3, e3, 120;
    shl.b32  norm3, norm3, 3;
    add.u32  norm3, norm3, r3;
    min.u32  norm3, norm3, 0x7e;
    max.u32  ec3, e3, 117;
    sub.u32  sh3, 125, ec3;
    sub.u32  sh23, sh3, 1;
    mov.u32  rnd3, 1;
    shl.b32  rnd3, rnd3, sh23;
    or.b32   tmp3, m3, 0x80;
    add.u32  sub3, tmp3, rnd3;
    shr.u32  sub3, sub3, sh3;
    add.u32  shp13, sh3, 1;
    mov.u32  one3, 1;
    shl.b32  mask3, one3, shp13;
    sub.u32  mask3, mask3, 1;
    and.b32  rem3, tmp3, mask3;
    setp.eq.u32 p_stie3, rem3, rnd3;
    sub.u32  sdec3, sub3, 1;
    selp.u32 sub3, sdec3, sub3, p_stie3;
    min.u32  sub3, sub3, 8;
    setp.lt.u32 p_tiny3, e3, 117;
    selp.u32 sub3, 0, sub3, p_tiny3;
    setp.gt.u32 p_norm3, e3, 120;
    selp.u32 o3, norm3, sub3, p_norm3;
    setp.ge.u32 p_hi3, a3, 0x43e0;
    selp.u32 o3, 0x7e, o3, p_hi3;
    or.b32 o3, o3, sgn3;

    shl.b32 o1, o1, 8;
    shl.b32 o2, o2, 16;
    shl.b32 o3, o3, 24;
    or.b32  out, o0, o1;
    or.b32  out, out, o2;
    or.b32  $0, out, o3;
}"""

# ---- decode: fp8e4m3 -> bf16 (prmt byte-LUT) --------------------------------
_FP8E4M3_TO_BF16_ASM = """\
{
    .reg .u32 hilo, hihi, lolo, lohi;
    .reg .u32 raw0, mag0, m0, sign0, norm0, sub0, hi0, lo0, o0;
    .reg .u32 raw1, mag1, m1, sign1, norm1, sub1, hi1, lo1, o1;
    .reg .u32 raw2, mag2, m2, sign2, norm2, sub2, hi2, lo2, o2;
    .reg .u32 raw3, mag3, m3, sign3, norm3, sub3, hi3, lo3, o3;
    .reg .u32 out0, out1;
    .reg .pred p_norm0;
    .reg .pred p_norm1;
    .reg .pred p_norm2;
    .reg .pred p_norm3;
    mov.u32 hilo, 0x3b3b3b00;
    mov.u32 hihi, 0x3c3c3c3c;
    mov.u32 lolo, 0xc0800000;
    mov.u32 lohi, 0x60402000;

    and.b32 raw0, $2, 0xff;
    shr.u32 raw1, $2, 8;
    and.b32 raw1, raw1, 0xff;
    shr.u32 raw2, $2, 16;
    and.b32 raw2, raw2, 0xff;
    shr.u32 raw3, $2, 24;
    and.b32 raw3, raw3, 0xff;


    and.b32 mag0, raw0, 0x7f;
    and.b32 sign0, raw0, 0x80;
    shl.b32 sign0, sign0, 8;
    shl.b32 norm0, mag0, 4;
    add.u32 norm0, norm0, 0x3c00;
    and.b32 m0, raw0, 0x07;
    prmt.b32 hi0, hilo, hihi, m0;
    and.b32 hi0, hi0, 0xff;
    shl.b32 hi0, hi0, 8;
    prmt.b32 lo0, lolo, lohi, m0;
    and.b32 lo0, lo0, 0xff;
    or.b32 sub0, hi0, lo0;
    setp.ge.u32 p_norm0, mag0, 8;
    selp.u32 o0, norm0, sub0, p_norm0;
    or.b32 o0, o0, sign0;


    and.b32 mag1, raw1, 0x7f;
    and.b32 sign1, raw1, 0x80;
    shl.b32 sign1, sign1, 8;
    shl.b32 norm1, mag1, 4;
    add.u32 norm1, norm1, 0x3c00;
    and.b32 m1, raw1, 0x07;
    prmt.b32 hi1, hilo, hihi, m1;
    and.b32 hi1, hi1, 0xff;
    shl.b32 hi1, hi1, 8;
    prmt.b32 lo1, lolo, lohi, m1;
    and.b32 lo1, lo1, 0xff;
    or.b32 sub1, hi1, lo1;
    setp.ge.u32 p_norm1, mag1, 8;
    selp.u32 o1, norm1, sub1, p_norm1;
    or.b32 o1, o1, sign1;


    and.b32 mag2, raw2, 0x7f;
    and.b32 sign2, raw2, 0x80;
    shl.b32 sign2, sign2, 8;
    shl.b32 norm2, mag2, 4;
    add.u32 norm2, norm2, 0x3c00;
    and.b32 m2, raw2, 0x07;
    prmt.b32 hi2, hilo, hihi, m2;
    and.b32 hi2, hi2, 0xff;
    shl.b32 hi2, hi2, 8;
    prmt.b32 lo2, lolo, lohi, m2;
    and.b32 lo2, lo2, 0xff;
    or.b32 sub2, hi2, lo2;
    setp.ge.u32 p_norm2, mag2, 8;
    selp.u32 o2, norm2, sub2, p_norm2;
    or.b32 o2, o2, sign2;


    and.b32 mag3, raw3, 0x7f;
    and.b32 sign3, raw3, 0x80;
    shl.b32 sign3, sign3, 8;
    shl.b32 norm3, mag3, 4;
    add.u32 norm3, norm3, 0x3c00;
    and.b32 m3, raw3, 0x07;
    prmt.b32 hi3, hilo, hihi, m3;
    and.b32 hi3, hi3, 0xff;
    shl.b32 hi3, hi3, 8;
    prmt.b32 lo3, lolo, lohi, m3;
    and.b32 lo3, lo3, 0xff;
    or.b32 sub3, hi3, lo3;
    setp.ge.u32 p_norm3, mag3, 8;
    selp.u32 o3, norm3, sub3, p_norm3;
    or.b32 o3, o3, sign3;

    shl.b32 o1, o1, 16;
    or.b32  out0, o0, o1;
    shl.b32 o3, o3, 16;
    or.b32  out1, o2, o3;
    mov.b32 $0, out0;
    mov.b32 $1, out1;
}"""

# Triton @jit functions may only reference globals instantiated as constexpr.
_BF16_TO_FP8E4M3_RNE_ASM = tl.constexpr(_BF16_TO_FP8E4M3_RNE_ASM)
_FP8E4M3_TO_BF16_ASM = tl.constexpr(_FP8E4M3_TO_BF16_ASM)
