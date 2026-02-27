// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <cassert>

#ifdef USE_ROCM
  #include <hip/hip_runtime.h>
#else
  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <cuda_runtime.h>
#endif

// Device-side: SM100+ architecture with CUDA 12.9+ toolkit, which
// together enable 256-bit (v8.u32) PTX load/store instructions.
// Use for PTX instruction selection with architecture fallback paths.
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(CUDA_VERSION) && CUDA_VERSION >= 12090
  #define VLLM_256B_PTX_ENABLED 1
#else
  #define VLLM_256B_PTX_ENABLED 0
#endif

namespace vllm {

// ============================================================
// Types and traits
// ============================================================

// 256-bit (32-byte) aligned vector type: 8 x uint32_t
struct alignas(32) u32x8_t {
  uint32_t d[8];
};

// VecTraits — select between 128-bit (int4) and 256-bit
// (u32x8_t) vector types at compile time.
template <bool support_256>
struct VecTraits;

template <>
struct VecTraits<true> {
  static constexpr int ARCH_MAX_VEC_SIZE = 32;
  using vec_t = u32x8_t;
};

template <>
struct VecTraits<false> {
  static constexpr int ARCH_MAX_VEC_SIZE = 16;
  using vec_t = int4;
};

// PackedTypeConverter — map between CUDA scalar and packed types
//   half  <-> half2,  __nv_bfloat16 <-> __nv_bfloat162, etc.
template <typename T>
struct PackedTypeConverter {
  static_assert(sizeof(T) == 0,
                "PackedTypeConverter is not specialized for this type.");
};

template <>
struct PackedTypeConverter<half2> {
  using Type = half;
};

template <>
struct PackedTypeConverter<half> {
  using Type = half2;
};

template <>
struct PackedTypeConverter<__nv_bfloat162> {
  using Type = __nv_bfloat16;
};

template <>
struct PackedTypeConverter<__nv_bfloat16> {
  using Type = __nv_bfloat162;
};

template <>
struct PackedTypeConverter<float> {
  using Type = float2;
};

template <>
struct PackedTypeConverter<float2> {
  using Type = float;
};

template <>
struct PackedTypeConverter<c10::Half> {
  using Type = half2;
};

template <>
struct PackedTypeConverter<c10::BFloat16> {
  using Type = __nv_bfloat162;
};

// CUDATypeConverter — map PyTorch scalar types to CUDA scalar
//   c10::Half -> half,  c10::BFloat16 -> __nv_bfloat16
template <typename T>
struct CUDATypeConverter {
  using Type = T;
};

template <>
struct CUDATypeConverter<c10::Half> {
  using Type = half;
};

template <>
struct CUDATypeConverter<c10::BFloat16> {
  using Type = __nv_bfloat16;
};

// PackedVec — typed vector container for packed element access.
//   Derives alignment and element count from VecTraits.
//   Type is the CUDA scalar type (e.g. half, __nv_bfloat16).
template <class Type, bool use_256b>
struct alignas(VecTraits<use_256b>::ARCH_MAX_VEC_SIZE) PackedVec {
  static constexpr int NUM_ELTS =
      VecTraits<use_256b>::ARCH_MAX_VEC_SIZE /
      sizeof(typename PackedTypeConverter<Type>::Type);
  typename PackedTypeConverter<Type>::Type elts[NUM_ELTS];
};

// ============================================================
// Load / store primitives
// ============================================================

// 256-bit load / store — SM100+ only (PTX v8 instructions).
__device__ __forceinline__ void ld256(u32x8_t& val, const u32x8_t* ptr) {
#if VLLM_256B_PTX_ENABLED
  asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
                 "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
               : "l"(ptr));
#else
  assert(false && "ld256 requires SM100+ with CUDA 12.9+");
#endif
}

__device__ __forceinline__ void st256(u32x8_t& val, u32x8_t* ptr) {
#if VLLM_256B_PTX_ENABLED
  asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
               :
               : "l"(ptr), "r"(val.d[0]), "r"(val.d[1]), "r"(val.d[2]),
                 "r"(val.d[3]), "r"(val.d[4]), "r"(val.d[5]), "r"(val.d[6]),
                 "r"(val.d[7])
               : "memory");
#else
  assert(false && "st256 requires SM100+ with CUDA 12.9+");
#endif
}

// Generic ld256 / st256 for any 32-byte aligned type (e.g. PackedVec).
// Non-template overloads above are preferred for u32x8_t.
template <typename T>
__device__ __forceinline__ void ld256(T& val, const T* ptr) {
  static_assert(sizeof(T) == 32, "ld256 requires a 32-byte type");
  ld256(reinterpret_cast<u32x8_t&>(val), reinterpret_cast<const u32x8_t*>(ptr));
}

template <typename T>
__device__ __forceinline__ void st256(T& val, T* ptr) {
  static_assert(sizeof(T) == 32, "st256 requires a 32-byte type");
  st256(reinterpret_cast<u32x8_t&>(val), reinterpret_cast<u32x8_t*>(ptr));
}

// 128-bit load / store via __ldg (read-only cache hint).
template <typename T>
__device__ __forceinline__ void ld128(T& val, const T* ptr) {
  static_assert(sizeof(T) == 16, "ld128 requires a 16-byte type");
  *reinterpret_cast<int4*>(&val) = __ldg(reinterpret_cast<const int4*>(ptr));
}

template <typename T>
__device__ __forceinline__ void st128(T& val, T* ptr) {
  static_assert(sizeof(T) == 16, "st128 requires a 16-byte type");
  *reinterpret_cast<int4*>(ptr) = *reinterpret_cast<int4*>(&val);
}

// 256-bit cache-streaming (.cs) load / store  — SM100+ only.
__forceinline__ __device__ u32x8_t ld256_cs(const u32x8_t* addr) {
#if VLLM_256B_PTX_ENABLED
  u32x8_t val;
  asm volatile("ld.global.cs.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
                 "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
               : "l"(addr));
  return val;
#else
  assert(false && "ld256_cs requires SM100+ with CUDA 12.9+");
  return {};
#endif
}

__forceinline__ __device__ void st256_cs(u32x8_t* addr, u32x8_t val) {
#if VLLM_256B_PTX_ENABLED
  asm volatile(
      "st.global.cs.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(addr),
      "r"(val.d[0]), "r"(val.d[1]), "r"(val.d[2]), "r"(val.d[3]), "r"(val.d[4]),
      "r"(val.d[5]), "r"(val.d[6]), "r"(val.d[7]));
#else
  assert(false && "st256_cs requires SM100+ with CUDA 12.9+");
#endif
}

// 32-bit cache-streaming (.cs) load / store  — SM100+ only.
__forceinline__ __device__ int ld32_cs(const int* addr) {
#if VLLM_256B_PTX_ENABLED
  int val;
  asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(val) : "l"(addr));
  return val;
#else
  assert(false && "ld32_cs requires SM100+ with CUDA 12.9+");
  return 0;
#endif
}

__forceinline__ __device__ void st32_cs(int* addr, int val) {
#if VLLM_256B_PTX_ENABLED
  asm volatile("st.global.cs.b32 [%0], %1;" ::"l"(addr), "r"(val));
#else
  assert(false && "st32_cs requires SM100+ with CUDA 12.9+");
#endif
}

// Predicated 256-bit / 128-bit cache-global (.cg) loads.
// Returns zero if pred is false.  SM100+ only.
__device__ __forceinline__ void ld256_cg_or_zero(u32x8_t& val, const void* ptr,
                                                 bool pred) {
#if VLLM_256B_PTX_ENABLED
  asm volatile(
      "{\n"
      "  .reg .pred pr;\n"
      "  setp.ne.u32 pr, %8, 0;\n"
      "  mov.u32 %0, 0;\n"
      "  mov.u32 %1, 0;\n"
      "  mov.u32 %2, 0;\n"
      "  mov.u32 %3, 0;\n"
      "  mov.u32 %4, 0;\n"
      "  mov.u32 %5, 0;\n"
      "  mov.u32 %6, 0;\n"
      "  mov.u32 %7, 0;\n"
      "  @pr ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%9];\n"
      "}\n"
      : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
        "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
      : "r"((int)pred), "l"(ptr));
#else
  assert(false && "ld256_cg_or_zero requires SM100+ with CUDA 12.9+");
#endif
}

__device__ __forceinline__ void ld128_cg_or_zero(uint4& val, const void* ptr,
                                                 bool pred) {
#if VLLM_256B_PTX_ENABLED
  uint32_t r0, r1, r2, r3;

  asm volatile(
      "{\n"
      "  .reg .pred pr;\n"
      "  setp.ne.u32 pr, %4, 0;\n"
      "  mov.u32 %0, 0;\n"
      "  mov.u32 %1, 0;\n"
      "  mov.u32 %2, 0;\n"
      "  mov.u32 %3, 0;\n"
      "  @pr ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%5];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"((int)pred), "l"(ptr));

  val = uint4{r0, r1, r2, r3};
#else
  assert(false && "ld128_cg_or_zero requires SM100+ with CUDA 12.9+");
#endif
}

// ============================================================
// Alignment helpers
// ============================================================

__host__ __device__ __forceinline__ bool is_16byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

__host__ __device__ __forceinline__ bool is_32byte_aligned(const void* ptr) {
  return (reinterpret_cast<uintptr_t>(ptr) & 31) == 0;
}

// ============================================================
// Packed type conversion and arithmetic
// ============================================================

template <typename packed_t>
__device__ __forceinline__ float2 cast_to_float2(const packed_t& val) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
    return __bfloat1622float2(val);
  } else if constexpr (std::is_same_v<packed_t, __half2>) {
    return __half22float2(val);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return float2(val);
  }
}

template <typename packed_t>
__device__ __forceinline__ packed_t cast_to_packed(const float2& val) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162>) {
    return __float22bfloat162_rn(val);
  } else if constexpr (std::is_same_v<packed_t, __half2>) {
    return __float22half2_rn(val);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return float2(val);
  }
}

template <typename packed_t>
__device__ __forceinline__ packed_t packed_mul(const packed_t& x,
                                               const packed_t& y) {
  if constexpr (std::is_same_v<packed_t, __nv_bfloat162> ||
                std::is_same_v<packed_t, __half2>) {
    return __hmul2(x, y);
  } else if constexpr (std::is_same_v<packed_t, float2>) {
    return make_float2(x.x * y.x, x.y * y.y);
  }
}

}  // namespace vllm
