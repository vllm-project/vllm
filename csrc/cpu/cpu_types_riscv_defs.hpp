#ifndef CPU_TYPES_RISCV_DEFS_HPP
#define CPU_TYPES_RISCV_DEFS_HPP

// VLEN-to-LMUL mapping for RISC-V Vector extension.
//
// LMUL_<N> expands to the LMUL suffix giving N total bits of vector data:
//   VLEN=128: LMUL_128=m1,  LMUL_256=m2,  LMUL_512=m4,  LMUL_1024=m8
//   VLEN=256: LMUL_128=mf2, LMUL_256=m1,  LMUL_512=m2,  LMUL_1024=m4

#include <riscv_vector.h>

#if __riscv_v_min_vlen == 128
  #define LMUL_128 m1
  #define LMUL_256 m2
  #define LMUL_512 m4
  #define LMUL_1024 m8
  #define BOOL_256 b16
  #define BOOL_512 b8
#elif __riscv_v_min_vlen == 256
  #define LMUL_128 mf2
  #define LMUL_256 m1
  #define LMUL_512 m2
  #define LMUL_1024 m4
  #define BOOL_256 b32
  #define BOOL_512 b16
#else
  #error "cpu_types_riscv_defs.hpp: unsupported __riscv_v_min_vlen"
#endif

// Token-paste helpers.
#define _RVV_P2(a, b) a##b
#define _RVV_P3(a, b, c) a##b##c
#define _RVV_P4(a, b, c, d) a##b##c##d
#define RVVTYPE(base, lmul, suffix) _RVV_P3(base, lmul, suffix)
#define RVVI(base, lmul) _RVV_P2(base, lmul)
#define RVVI3(base, lmul, suffix) _RVV_P3(base, lmul, suffix)
#define RVVI4(a, b, c, d) _RVV_P4(a, b, c, d)
// For mask intrinsics: RVVIB(base, LMUL_256, BOOL_256) → base##m2##_##b16
#define _RVV_PB(base, lmul, btype) base##lmul##_##btype
#define RVVIB(base, lmul, btype) _RVV_PB(base, lmul, btype)

// ---- Semantic fixed-vector typedefs (named by element count) ----

// float16
typedef RVVTYPE(vfloat16, LMUL_128, _t) fixed_fp16x8_t
    __attribute__((riscv_rvv_vector_bits(128)));
typedef RVVTYPE(vfloat16, LMUL_256, _t) fixed_fp16x16_t
    __attribute__((riscv_rvv_vector_bits(256)));

// float32
typedef RVVTYPE(vfloat32, LMUL_128, _t) fixed_fp32x4_t
    __attribute__((riscv_rvv_vector_bits(128)));
typedef RVVTYPE(vfloat32, LMUL_256, _t) fixed_fp32x8_t
    __attribute__((riscv_rvv_vector_bits(256)));
typedef RVVTYPE(vfloat32, LMUL_512, _t) fixed_fp32x16_t
    __attribute__((riscv_rvv_vector_bits(512)));
typedef RVVTYPE(vfloat32, LMUL_1024, _t) fixed_fp32x32_t
    __attribute__((riscv_rvv_vector_bits(1024)));

// int32
typedef RVVTYPE(vint32, LMUL_256, _t) fixed_i32x8_t
    __attribute__((riscv_rvv_vector_bits(256)));
typedef RVVTYPE(vint32, LMUL_512, _t) fixed_i32x16_t
    __attribute__((riscv_rvv_vector_bits(512)));

// uint16
typedef RVVTYPE(vuint16, LMUL_128, _t) fixed_u16x8_t
    __attribute__((riscv_rvv_vector_bits(128)));
typedef RVVTYPE(vuint16, LMUL_256, _t) fixed_u16x16_t
    __attribute__((riscv_rvv_vector_bits(256)));
typedef RVVTYPE(vuint16, LMUL_512, _t) fixed_u16x32_t
    __attribute__((riscv_rvv_vector_bits(512)));

// bfloat16
#ifdef __riscv_zvfbfmin
typedef RVVTYPE(vbfloat16, LMUL_128, _t) fixed_bf16x8_t
    __attribute__((riscv_rvv_vector_bits(128)));
typedef RVVTYPE(vbfloat16, LMUL_256, _t) fixed_bf16x16_t
    __attribute__((riscv_rvv_vector_bits(256)));
typedef RVVTYPE(vbfloat16, LMUL_512, _t) fixed_bf16x32_t
    __attribute__((riscv_rvv_vector_bits(512)));
#endif

// ---- Reduction accumulator type (always m1 = one register of f32) ----
// Used for scalar reductions; only element [0] is meaningful.
typedef vfloat32m1_t rvv_f32_accum_t
    __attribute__((riscv_rvv_vector_bits(__riscv_v_min_vlen)));

// ---- Mask types for f32 elements ----
#if __riscv_v_min_vlen == 128
typedef vbool16_t rvv_mask_f32x8_t;
typedef vbool8_t rvv_mask_f32x16_t;
#elif __riscv_v_min_vlen == 256
typedef vbool32_t rvv_mask_f32x8_t;
typedef vbool16_t rvv_mask_f32x16_t;
#endif

#endif  // CPU_TYPES_RISCV_DEFS_HPP
