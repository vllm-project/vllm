// Helpers shared by the RDNA3.5 decode and prefill kernels.
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <type_traits>

namespace rdna_singleseq {

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef short s16 __attribute__((ext_vector_type(16)));
typedef short s2v __attribute__((ext_vector_type(2)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned u8v __attribute__((ext_vector_type(8)));
typedef unsigned u4v __attribute__((ext_vector_type(4)));
typedef unsigned u2v __attribute__((ext_vector_type(2)));

// The element type, bf16 or fp16, and what depends on it.
template <bool kBf16>
struct Elem {
  using type = std::conditional_t<kBf16, __bf16, _Float16>;
  typedef type v16 __attribute__((ext_vector_type(16)));
  typedef type v8 __attribute__((ext_vector_type(8)));
  typedef type v2 __attribute__((ext_vector_type(2)));

  // Round to nearest even; a bf16 NaN may become Inf.
  static __device__ __forceinline__ type from_float(float x) {
    if constexpr (kBf16) {
      const unsigned bits = __builtin_bit_cast(unsigned, x);
      return __builtin_bit_cast(
          type, (unsigned short)((bits + 0x7FFFu + ((bits >> 16) & 1u)) >> 16));
    } else {
      return (type)x;
    }
  }

  static __device__ __forceinline__ f8 wmma(v16 a, v16 b, f8 c) {
    if constexpr (kBf16)
      return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
          __builtin_bit_cast(s16, a), __builtin_bit_cast(s16, b), c);
    else
      return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
          __builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b), c);
  }
};

// Barrier that waits for LDS only, not for global loads.
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// Lane l takes lane l ^ 16's value. fetch-inactive avoids a false dependency
// on the source register.
__device__ __forceinline__ unsigned xhalf_u(unsigned value) {
  return (unsigned)__builtin_amdgcn_permlanex16(
      (int)value, (int)value, 0x76543210u, 0xFEDCBA98u, true, false);
}

__device__ __forceinline__ float xhalf(float value) {
  return __builtin_bit_cast(float,
                            xhalf_u(__builtin_bit_cast(unsigned, value)));
}

// Lane i of each 16-lane row takes lane i & 7's value (lower8) or i | 8's.
__device__ __forceinline__ unsigned lower8(unsigned value) {
  return (unsigned)__builtin_amdgcn_permlane16(
      (int)value, (int)value, 0x76543210u, 0x76543210u, true, false);
}

__device__ __forceinline__ float upper8f(float value) {
  const int bits = __builtin_bit_cast(int, value);
  return __builtin_bit_cast(
      float, __builtin_amdgcn_permlane16(bits, bits, 0xFEDCBA98u, 0xFEDCBA98u,
                                         true, false));
}

__device__ __forceinline__ float weight_of(float partMax, float maxAll) {
  return (maxAll == -INFINITY) ? 0.f : __builtin_amdgcn_exp2f(partMax - maxAll);
}

}  // namespace rdna_singleseq
