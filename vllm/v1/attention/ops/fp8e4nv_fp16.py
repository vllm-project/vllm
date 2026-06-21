# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Software fp8e4m3 (E4M3FN / fp8e4nv) <-> fp16 conversion for Triton, SM75+.

Companion to fp8e4nv_bf16.py (which converts fp8 <-> *bf16*). bf16 is an SM80+
hardware type, so the bf16 path cannot serve SM75 (Turing, e.g. Tesla T4). On
SM75 the only software dtype for an fp8 KV cache is fp16, so these helpers
convert fp8e4m3 <-> fp16 directly (NOT via bf16) using ONLY SM75-legal integer
PTX (and/or/shl/shr/add/sub/min/max/setp/selp/prmt/cvt.u32.u16) -- no native
fp8 cvt, no bf16, no video-SIMD (vset2/vsub2). Verified to assemble for sm_75
(`ptxas -arch=sm_75`). Runs on SM75-SM88; on SM89+ use the native fp8e4nv cvt.

Validated bit-exact (decode and RNE encode) vs torch.float8_e4m3fn over the
entire finite domain (all 254 finite fp8 bytes; all 63,488 finite fp16
patterns), denormals included.

FP8 E4M3FN byte:  S EEEE MMM        bias 7,  finite-only (0x7f/0xff = NaN), max 448
FP16 (IEEE half): S EEEEE MMMMMMMMMM  bias 15
Normal map: fp16_exp = fp8_exp + 8, fp16_mant = fp8_mant << 7
  decode normal:  fp16 = (mag << 7) + 0x2000      (0x2000 == 8<<10)
  encode normal:  fp8  = ((e5 - 8) << 3) + round(m10 >> 7)

Each conversion is a `pack=4` inline-asm block: the 4 elements are unrolled
explicitly into 4 independent lanes (suffix 0..3), one $-operand-pair in, one
out. The lanes are identical except for the index; they are written out in full
(not codegen'd) so the emitted PTX is exactly what you read here.

# ---------------------------------------------------------------------------
# Encode: fp16 -> fp8e4m3  (write path). Saturating overflow (>448 -> 0x7e,
# never NaN). Round-to-nearest-even; bit-exact vs torch over all finite fp16.
#
# Decode: fp8e4m3 -> fp16  (read path). Exact expansion, no rounding; subnormal
# values come from a single prmt byte-LUT (LUT[m]<<8). Exact incl. denormals.
# ---------------------------------------------------------------------------
"""

from vllm.triton_utils import tl

# ---- encode: fp16 -> fp8e4m3, RNE (round-to-nearest-even), saturating -------
_FP16_TO_FP8E4M3_RNE_ASM = """\
{
    .reg .u16 b<4>;
    .reg .u32 raw<4>;
    .reg .u32 a0, e0, m0, r0, tmp0, ndec0, norm0, ec0, sh0, shm10;
    .reg .u32 shp10, one0, half0, mask0, rem0, sub0, sdec0, sgn0, o0;
    .reg .u32 a1, e1, m1, r1, tmp1, ndec1, norm1, ec1, sh1, shm11;
    .reg .u32 shp11, one1, half1, mask1, rem1, sub1, sdec1, sgn1, o1;
    .reg .u32 a2, e2, m2, r2, tmp2, ndec2, norm2, ec2, sh2, shm12;
    .reg .u32 shp12, one2, half2, mask2, rem2, sub2, sdec2, sgn2, o2;
    .reg .u32 a3, e3, m3, r3, tmp3, ndec3, norm3, ec3, sh3, shm13;
    .reg .u32 shp13, one3, half3, mask3, rem3, sub3, sdec3, sgn3, o3;
    .reg .u32 out;
    .reg .pred p_ntie0, p_stie0, p_tiny0, p_norm0, p_hi0;
    .reg .pred p_ntie1, p_stie1, p_tiny1, p_norm1, p_hi1;
    .reg .pred p_ntie2, p_stie2, p_tiny2, p_norm2, p_hi2;
    .reg .pred p_ntie3, p_stie3, p_tiny3, p_norm3, p_hi3;

    mov.b32 {b0, b1}, $1;
    mov.b32 {b2, b3}, $2;
    cvt.u32.u16 raw0, b0;
    cvt.u32.u16 raw1, b1;
    cvt.u32.u16 raw2, b2;
    cvt.u32.u16 raw3, b3;

    // lane 0
    and.b32  a0, raw0, 0x7fff;
    and.b32  sgn0, raw0, 0x8000;
    shr.u32  sgn0, sgn0, 8;
    shr.u32  e0, a0, 10;
    and.b32  m0, a0, 0x3ff;

    add.u32  r0, m0, 0x40;
    shr.u32  r0, r0, 7;
    and.b32  tmp0, m0, 0xff;
    setp.eq.u32 p_ntie0, tmp0, 0x40;
    sub.u32  ndec0, r0, 1;
    selp.u32 r0, ndec0, r0, p_ntie0;
    sub.u32  norm0, e0, 8;
    shl.b32  norm0, norm0, 3;
    add.u32  norm0, norm0, r0;
    min.u32  norm0, norm0, 0x7f;

    max.u32  ec0, e0, 5;
    sub.u32  sh0, 16, ec0;
    sub.u32  shm10, sh0, 1;
    mov.u32  one0, 1;
    shl.b32  half0, one0, shm10;
    or.b32   tmp0, m0, 0x400;
    add.u32  sub0, tmp0, half0;
    shr.u32  sub0, sub0, sh0;
    add.u32  shp10, sh0, 1;
    shl.b32  mask0, one0, shp10;
    sub.u32  mask0, mask0, 1;
    and.b32  rem0, tmp0, mask0;
    setp.eq.u32 p_stie0, rem0, half0;
    sub.u32  sdec0, sub0, 1;
    selp.u32 sub0, sdec0, sub0, p_stie0;
    min.u32  sub0, sub0, 8;
    setp.lt.u32 p_tiny0, e0, 5;
    selp.u32 sub0, 0, sub0, p_tiny0;

    setp.gt.u32 p_norm0, e0, 8;
    selp.u32 o0, norm0, sub0, p_norm0;
    setp.ge.u32 p_hi0, a0, 0x5f41;
    selp.u32 o0, 0x7e, o0, p_hi0;
    or.b32 o0, o0, sgn0;

    // lane 1
    and.b32  a1, raw1, 0x7fff;
    and.b32  sgn1, raw1, 0x8000;
    shr.u32  sgn1, sgn1, 8;
    shr.u32  e1, a1, 10;
    and.b32  m1, a1, 0x3ff;

    add.u32  r1, m1, 0x40;
    shr.u32  r1, r1, 7;
    and.b32  tmp1, m1, 0xff;
    setp.eq.u32 p_ntie1, tmp1, 0x40;
    sub.u32  ndec1, r1, 1;
    selp.u32 r1, ndec1, r1, p_ntie1;
    sub.u32  norm1, e1, 8;
    shl.b32  norm1, norm1, 3;
    add.u32  norm1, norm1, r1;
    min.u32  norm1, norm1, 0x7f;

    max.u32  ec1, e1, 5;
    sub.u32  sh1, 16, ec1;
    sub.u32  shm11, sh1, 1;
    mov.u32  one1, 1;
    shl.b32  half1, one1, shm11;
    or.b32   tmp1, m1, 0x400;
    add.u32  sub1, tmp1, half1;
    shr.u32  sub1, sub1, sh1;
    add.u32  shp11, sh1, 1;
    shl.b32  mask1, one1, shp11;
    sub.u32  mask1, mask1, 1;
    and.b32  rem1, tmp1, mask1;
    setp.eq.u32 p_stie1, rem1, half1;
    sub.u32  sdec1, sub1, 1;
    selp.u32 sub1, sdec1, sub1, p_stie1;
    min.u32  sub1, sub1, 8;
    setp.lt.u32 p_tiny1, e1, 5;
    selp.u32 sub1, 0, sub1, p_tiny1;

    setp.gt.u32 p_norm1, e1, 8;
    selp.u32 o1, norm1, sub1, p_norm1;
    setp.ge.u32 p_hi1, a1, 0x5f41;
    selp.u32 o1, 0x7e, o1, p_hi1;
    or.b32 o1, o1, sgn1;

    // lane 2
    and.b32  a2, raw2, 0x7fff;
    and.b32  sgn2, raw2, 0x8000;
    shr.u32  sgn2, sgn2, 8;
    shr.u32  e2, a2, 10;
    and.b32  m2, a2, 0x3ff;

    add.u32  r2, m2, 0x40;
    shr.u32  r2, r2, 7;
    and.b32  tmp2, m2, 0xff;
    setp.eq.u32 p_ntie2, tmp2, 0x40;
    sub.u32  ndec2, r2, 1;
    selp.u32 r2, ndec2, r2, p_ntie2;
    sub.u32  norm2, e2, 8;
    shl.b32  norm2, norm2, 3;
    add.u32  norm2, norm2, r2;
    min.u32  norm2, norm2, 0x7f;

    max.u32  ec2, e2, 5;
    sub.u32  sh2, 16, ec2;
    sub.u32  shm12, sh2, 1;
    mov.u32  one2, 1;
    shl.b32  half2, one2, shm12;
    or.b32   tmp2, m2, 0x400;
    add.u32  sub2, tmp2, half2;
    shr.u32  sub2, sub2, sh2;
    add.u32  shp12, sh2, 1;
    shl.b32  mask2, one2, shp12;
    sub.u32  mask2, mask2, 1;
    and.b32  rem2, tmp2, mask2;
    setp.eq.u32 p_stie2, rem2, half2;
    sub.u32  sdec2, sub2, 1;
    selp.u32 sub2, sdec2, sub2, p_stie2;
    min.u32  sub2, sub2, 8;
    setp.lt.u32 p_tiny2, e2, 5;
    selp.u32 sub2, 0, sub2, p_tiny2;

    setp.gt.u32 p_norm2, e2, 8;
    selp.u32 o2, norm2, sub2, p_norm2;
    setp.ge.u32 p_hi2, a2, 0x5f41;
    selp.u32 o2, 0x7e, o2, p_hi2;
    or.b32 o2, o2, sgn2;

    // lane 3
    and.b32  a3, raw3, 0x7fff;
    and.b32  sgn3, raw3, 0x8000;
    shr.u32  sgn3, sgn3, 8;
    shr.u32  e3, a3, 10;
    and.b32  m3, a3, 0x3ff;

    add.u32  r3, m3, 0x40;
    shr.u32  r3, r3, 7;
    and.b32  tmp3, m3, 0xff;
    setp.eq.u32 p_ntie3, tmp3, 0x40;
    sub.u32  ndec3, r3, 1;
    selp.u32 r3, ndec3, r3, p_ntie3;
    sub.u32  norm3, e3, 8;
    shl.b32  norm3, norm3, 3;
    add.u32  norm3, norm3, r3;
    min.u32  norm3, norm3, 0x7f;

    max.u32  ec3, e3, 5;
    sub.u32  sh3, 16, ec3;
    sub.u32  shm13, sh3, 1;
    mov.u32  one3, 1;
    shl.b32  half3, one3, shm13;
    or.b32   tmp3, m3, 0x400;
    add.u32  sub3, tmp3, half3;
    shr.u32  sub3, sub3, sh3;
    add.u32  shp13, sh3, 1;
    shl.b32  mask3, one3, shp13;
    sub.u32  mask3, mask3, 1;
    and.b32  rem3, tmp3, mask3;
    setp.eq.u32 p_stie3, rem3, half3;
    sub.u32  sdec3, sub3, 1;
    selp.u32 sub3, sdec3, sub3, p_stie3;
    min.u32  sub3, sub3, 8;
    setp.lt.u32 p_tiny3, e3, 5;
    selp.u32 sub3, 0, sub3, p_tiny3;

    setp.gt.u32 p_norm3, e3, 8;
    selp.u32 o3, norm3, sub3, p_norm3;
    setp.ge.u32 p_hi3, a3, 0x5f41;
    selp.u32 o3, 0x7e, o3, p_hi3;
    or.b32 o3, o3, sgn3;

    shl.b32 o1, o1, 8;
    shl.b32 o2, o2, 16;
    shl.b32 o3, o3, 24;
    or.b32  out, o0, o1;
    or.b32  out, out, o2;
    or.b32  $0, out, o3;
}"""

# ---- decode: fp8e4m3 -> fp16 (prmt byte-LUT for subnormals; exact) ----------
_FP8E4M3_TO_FP16_ASM = """\
{
    .reg .u32 sublut, subhi;
    .reg .u32 raw0, mag0, m0, sign0, norm0, sub0, o0;
    .reg .u32 raw1, mag1, m1, sign1, norm1, sub1, o1;
    .reg .u32 raw2, mag2, m2, sign2, norm2, sub2, o2;
    .reg .u32 raw3, mag3, m3, sign3, norm3, sub3, o3;
    .reg .u32 out0, out1;
    .reg .pred p_norm0;
    .reg .pred p_norm1;
    .reg .pred p_norm2;
    .reg .pred p_norm3;
    mov.u32 sublut, 0x1e1c1800;
    mov.u32 subhi, 0x23222120;

    and.b32 raw0, $2, 0xff;
    shr.u32 raw1, $2, 8;
    and.b32 raw1, raw1, 0xff;
    shr.u32 raw2, $2, 16;
    and.b32 raw2, raw2, 0xff;
    shr.u32 raw3, $2, 24;
    and.b32 raw3, raw3, 0xff;

    // lane 0
    and.b32 mag0, raw0, 0x7f;
    and.b32 sign0, raw0, 0x80;
    shl.b32 sign0, sign0, 8;
    shl.b32 norm0, mag0, 7;
    add.u32 norm0, norm0, 0x2000;
    and.b32 m0, raw0, 0x07;
    shl.b32 m0, m0, 4;
    prmt.b32 sub0, sublut, subhi, m0;
    setp.ge.u32 p_norm0, mag0, 8;
    selp.u32 o0, norm0, sub0, p_norm0;
    or.b32 o0, o0, sign0;

    // lane 1
    and.b32 mag1, raw1, 0x7f;
    and.b32 sign1, raw1, 0x80;
    shl.b32 sign1, sign1, 8;
    shl.b32 norm1, mag1, 7;
    add.u32 norm1, norm1, 0x2000;
    and.b32 m1, raw1, 0x07;
    shl.b32 m1, m1, 4;
    prmt.b32 sub1, sublut, subhi, m1;
    setp.ge.u32 p_norm1, mag1, 8;
    selp.u32 o1, norm1, sub1, p_norm1;
    or.b32 o1, o1, sign1;

    // lane 2
    and.b32 mag2, raw2, 0x7f;
    and.b32 sign2, raw2, 0x80;
    shl.b32 sign2, sign2, 8;
    shl.b32 norm2, mag2, 7;
    add.u32 norm2, norm2, 0x2000;
    and.b32 m2, raw2, 0x07;
    shl.b32 m2, m2, 4;
    prmt.b32 sub2, sublut, subhi, m2;
    setp.ge.u32 p_norm2, mag2, 8;
    selp.u32 o2, norm2, sub2, p_norm2;
    or.b32 o2, o2, sign2;

    // lane 3
    and.b32 mag3, raw3, 0x7f;
    and.b32 sign3, raw3, 0x80;
    shl.b32 sign3, sign3, 8;
    shl.b32 norm3, mag3, 7;
    add.u32 norm3, norm3, 0x2000;
    and.b32 m3, raw3, 0x07;
    shl.b32 m3, m3, 4;
    prmt.b32 sub3, sublut, subhi, m3;
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

_FP16_TO_FP8E4M3_RNE_ASM = tl.constexpr(_FP16_TO_FP8E4M3_RNE_ASM)
_FP8E4M3_TO_FP16_ASM = tl.constexpr(_FP8E4M3_TO_FP16_ASM)
