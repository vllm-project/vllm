// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// clang++-18 --cuda-device-only -nocudainc -nocudalib \
//   --cuda-gpu-arch=sm_75 -fcuda-flush-denormals-to-zero -O3 -emit-llvm \
//   -c fp8e4nv_helper.cu -o fp8e4nv_helper_sm75.bc

using u32 = unsigned int;
using u64 = unsigned long long;

extern "C" __attribute__((device, always_inline)) u64
fp8e4m3x4_to_fp16x4(u32 input) {
  u32 out01;
  u32 out23;
  asm(R"ptx({
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

    and.b32 raw0, %2, 0xff;
    shr.u32 raw1, %2, 8;
    and.b32 raw1, raw1, 0xff;
    shr.u32 raw2, %2, 16;
    and.b32 raw2, raw2, 0xff;
    shr.u32 raw3, %2, 24;
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
    mov.b32 %0, out0;
    mov.b32 %1, out1;
  })ptx"
      : "=r"(out01), "=r"(out23)
      : "r"(input));
  return static_cast<u64>(out01) | (static_cast<u64>(out23) << 32);
}

extern "C" __attribute__((device, always_inline)) u32
fp16x4_to_fp8e4m3x4(u32 input01, u32 input23) {
  u32 output;
  asm(R"ptx({
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

    mov.b32 {b0, b1}, %1;
    mov.b32 {b2, b3}, %2;
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
    or.b32  %0, out, o3;
  })ptx"
      : "=r"(output)
      : "r"(input01), "r"(input23));
  return output;
}

extern "C" __attribute__((device, always_inline)) u32
bf16x4_to_fp8e4m3x4(u32 input01, u32 input23) {
  u32 output;
  asm(R"ptx({
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

    mov.b32 {b0, b1}, %1;
    mov.b32 {b2, b3}, %2;
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
    or.b32  %0, out, o3;
  })ptx"
      : "=r"(output)
      : "r"(input01), "r"(input23));
  return output;
}

extern "C" __attribute__((device, always_inline)) u64
fp8e4m3x4_to_bf16x4(u32 input) {
  u32 out01;
  u32 out23;
  asm(R"ptx({
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

    and.b32 raw0, %2, 0xff;
    shr.u32 raw1, %2, 8;
    and.b32 raw1, raw1, 0xff;
    shr.u32 raw2, %2, 16;
    and.b32 raw2, raw2, 0xff;
    shr.u32 raw3, %2, 24;
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
    mov.b32 %0, out0;
    mov.b32 %1, out1;
  })ptx"
      : "=r"(out01), "=r"(out23)
      : "r"(input));
  return static_cast<u64>(out01) | (static_cast<u64>(out23) << 32);
}

using u8 = unsigned char;
using u16 = unsigned short;
using u16x1 = u16 __attribute__((ext_vector_type(1)));
using u16x2 = u16 __attribute__((ext_vector_type(2)));
using u32x1 = u32 __attribute__((ext_vector_type(1)));

template <u32 (*Convert)(u32, u32)>
__attribute__((device, always_inline)) u8 convert_lane0_with_pack4(u16 input) {
  // Match Triton's partial pack-4 lowering: lane 0 is real; the rest are unused.
  const u16x2 lanes01 =
      __builtin_shufflevector(u16x1{input}, u16x1{}, 0, -1);
  const u32x1 input23 = __builtin_shufflevector(u32x1{}, u32x1{}, -1);
  return static_cast<u8>(
      Convert(__builtin_bit_cast(u32, lanes01), input23[0]));
}

extern "C" __attribute__((device, always_inline)) u8
fp16x1_to_fp8e4m3(u16 input) {
  return convert_lane0_with_pack4<fp16x4_to_fp8e4m3x4>(input);
}

extern "C" __attribute__((device, always_inline)) u8
bf16x1_to_fp8e4m3(u16 input) {
  return convert_lane0_with_pack4<bf16x4_to_fp8e4m3x4>(input);
}
