// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Shared types and WMMA wrappers for the RDNA3 paged-prefill attention
// kernels (gfx1100 / RX 7900 XTX class).
//
// Conventions follow the W4A16 RDNA3 sibling kernels in
// csrc/quantization/gptq/qdq_4_rdna3.cuh — specifically the wave32
// fragment layout (mode 1: A row-major in M with K in slot, B col-major
// in N with K in slot, output C[m=2*i+lane_hi][n=lane_lo]) and the
// dtype-template approach (half + __hip_bfloat16).

#ifndef _paged_prefill_attn_rdna3_cuh
#define _paged_prefill_attn_rdna3_cuh

#include <cstdint>

#if defined(USE_ROCM)
  #include <hip/hip_runtime.h>
  #include <hip/hip_bf16.h>
  #include <hip/hip_fp16.h>
#else
  #include <cuda_runtime.h>
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
#endif

namespace vllm {
namespace prefill_attn_rdna3 {

#if defined(USE_ROCM)
using bf16_t = __hip_bfloat16;
using bf162_t = __hip_bfloat162;

// Native AMDGPU vector types expected by the WMMA built-ins.
using v16fp16 = _Float16 __attribute__((ext_vector_type(16)));
using v16bf16 = __bf16 __attribute__((ext_vector_type(16)));
using v8fp32 = float __attribute__((ext_vector_type(8)));

// 16x16x16 WMMA, fp32 accumulator. fp16 and bf16 inputs go through their
// dtype-specific built-in. Both have a 16-cycle nominal throughput on
// gfx1100; bf16 is native (no v_pk_fma_bf16 fallback like the W4A16
// scalar path needs because WMMA uses a different functional unit).
__device__ __forceinline__ v8fp32 wmma_mma(v16fp16 a, v16fp16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
__device__ __forceinline__ v8fp32 wmma_mma(v16bf16 a, v16bf16 b, v8fp32 c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}

// 16x16x16 INT8 WMMA: A=signed int8, B=unsigned int8, C=int32 accumulator.
// Used for INT4 Q×K path: Q quantized to int8, K nibbles in [0,15] as uint8.
// Same 16-cycle throughput as bf16 WMMA on gfx1100.
using v16i8 = int8_t __attribute__((ext_vector_type(16)));
using v16u8 = uint8_t __attribute__((ext_vector_type(16)));
using v8i32 = int __attribute__((ext_vector_type(8)));

__device__ __forceinline__ v8i32 wmma_mma_iu8(v16i8 a_signed, v16u8 b_unsigned,
                                              v8i32 c) {
  // signA=true (A is signed), signB=false (B is unsigned), clamp=false
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
      /*signA=*/true, a_signed, /*signB=*/false, b_unsigned, c,
      /*clamp=*/false);
}

// Both A and B signed (for INT4 centered: nibble - zp → range [-15, 15]).
__device__ __forceinline__ v8i32 wmma_mma_ii8(v16i8 a_signed, v16i8 b_signed,
                                              v8i32 c) {
  return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(
      /*signA=*/true, a_signed, /*signB=*/true, b_signed, c, /*clamp=*/false);
}

template <typename T>
struct WmmaNative;
template <>
struct WmmaNative<half> {
  using elem = _Float16;
  using v16 = v16fp16;
};
template <>
struct WmmaNative<bf16_t> {
  using elem = __bf16;
  using v16 = v16bf16;
};

// Bitcast between HIP wrapper types (half, __hip_bfloat16) and native
// compiler types (_Float16, __bf16) used by the WMMA built-ins. Lowers
// to a register-rename in practice.
template <typename FROM, typename TO>
__device__ __forceinline__ TO bitcast_elem(FROM x) {
  static_assert(sizeof(FROM) == sizeof(TO),
                "bitcast_elem requires equal-sized types");
  TO r;
  __builtin_memcpy(&r, &x, sizeof(TO));
  return r;
}

template <typename T>
__device__ __forceinline__ T to_T(float v);
template <>
__device__ __forceinline__ half to_T<half>(float v) {
  return __float2half_rn(v);
}
template <>
__device__ __forceinline__ bf16_t to_T<bf16_t>(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ float to_f(T v);
template <>
__device__ __forceinline__ float to_f<half>(half v) {
  return __half2float(v);
}
template <>
__device__ __forceinline__ float to_f<bf16_t>(bf16_t v) {
  return __bfloat162float(v);
}

#endif  // USE_ROCM

}  // namespace prefill_attn_rdna3
}  // namespace vllm

#endif  // _paged_prefill_attn_rdna3_cuh
