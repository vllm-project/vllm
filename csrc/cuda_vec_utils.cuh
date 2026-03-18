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
// ROCm gfx950+ (CDNA4/MI350X/MI355X): 256-bit logical width via 2× dwordx4.
// Use for PTX/ISA instruction selection with architecture fallback paths.
#if !defined(USE_ROCM) && defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && \
    defined(CUDA_VERSION) && CUDA_VERSION >= 12090
  #define VLLM_256B_PTX_ENABLED 1
// gfx950 (CDNA4): 256-bit path disabled for now.  The gfx950 ISA (ROCm 7.2,
// LLVM 22.0) lacks vector dwordx8 load/store instructions, so the compiler
// decomposes 256-bit ops into 2× dwordx4 issued in separate cycles, leading
// to uncoalesced memory accesses within a warp.  Re-enable once a future
// ROCm version adds native global_load/store_dwordx8 support.
// See: https://github.com/vllm-project/vllm/pull/36743#discussion_r2048743213
// #elif defined(USE_ROCM) && defined(__gfx950__)
//   #define VLLM_256B_PTX_ENABLED 1
#else
  #define VLLM_256B_PTX_ENABLED 0
#endif

// ROCm gfx942+ (CDNA3/MI300X): non-temporal hints for cache bypass.
#if defined(USE_ROCM) && (defined(__gfx942__) || defined(__gfx950__))
  #define VLLM_ROCM_USE_NT_HINTS 1
#else
  #define VLLM_ROCM_USE_NT_HINTS 0
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

// 256-bit load / store — SM100+ (PTX v8) or ROCm gfx950+ (2× dwordx4).
__device__ __forceinline__ void ld256(u32x8_t& val, const u32x8_t* ptr) {
#if VLLM_256B_PTX_ENABLED
  #if defined(USE_ROCM) && defined(__gfx950__)
  // gfx950 (CDNA4): 256-bit logical load.  The gfx950 ISA does not have
  // global_load_dwordx8 — the compiler emits 2× global_load_dwordx4 with
  // adjacent offsets (off + off:16).  This still halves loop iterations vs
  // the 128-bit path.  Verified on MI350X, ROCm 7.2 / LLVM 22.0.
  // If a future ROCm adds vector dwordx8, this code benefits automatically.
  const uint32_t* src = reinterpret_cast<const uint32_t*>(ptr);
  val.d[0] = src[0];
  val.d[1] = src[1];
  val.d[2] = src[2];
  val.d[3] = src[3];
  val.d[4] = src[4];
  val.d[5] = src[5];
  val.d[6] = src[6];
  val.d[7] = src[7];
  #else
  asm volatile("ld.global.nc.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];\n"
               : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
                 "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
               : "l"(ptr));
  #endif
#else
  assert(false && "ld256 requires SM100+ with CUDA 12.9+ or ROCm gfx950+");
#endif
}

__device__ __forceinline__ void st256(u32x8_t& val, u32x8_t* ptr) {
#if VLLM_256B_PTX_ENABLED
  #if defined(USE_ROCM) && defined(__gfx950__)
  // gfx950 (CDNA4): 256-bit logical store → 2× global_store_dwordx4.
  // See ld256 comment for ISA details (verified on MI350X, ROCm 7.2).
  uint32_t* dst = reinterpret_cast<uint32_t*>(ptr);
  dst[0] = val.d[0];
  dst[1] = val.d[1];
  dst[2] = val.d[2];
  dst[3] = val.d[3];
  dst[4] = val.d[4];
  dst[5] = val.d[5];
  dst[6] = val.d[6];
  dst[7] = val.d[7];
  #else
  asm volatile("st.global.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};\n"
               :
               : "l"(ptr), "r"(val.d[0]), "r"(val.d[1]), "r"(val.d[2]),
                 "r"(val.d[3]), "r"(val.d[4]), "r"(val.d[5]), "r"(val.d[6]),
                 "r"(val.d[7])
               : "memory");
  #endif
#else
  assert(false && "st256 requires SM100+ with CUDA 12.9+ or ROCm gfx950+");
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

// 256-bit cache-streaming (.cs) load / store.
// SM100+: PTX .cs hint.  ROCm gfx950+: slc (system-level coherent) hint.
__forceinline__ __device__ u32x8_t ld256_cs(const u32x8_t* addr) {
#if VLLM_256B_PTX_ENABLED
  u32x8_t val;
  #if defined(USE_ROCM) && defined(__gfx950__)
  // gfx950 (CDNA4): 256-bit non-temporal load for streaming.
  // The compiler coalesces sequential __builtin_nontemporal_load calls
  // into vectorized global_load instructions at -O3 (verified on gfx942
  // where 4× scalar NT → single global_load_dwordx4 nt).
  // On gfx950: emits 2× global_load_dwordx4 nt (no vector dwordx8 in ISA).
  // Verified on MI350X, ROCm 7.2 / LLVM 22.0.
  // TODO: Check if future ROCm versions add a vector dwordx8 instruction to the
  // gfx950 ISA.
  const uint32_t* src = reinterpret_cast<const uint32_t*>(addr);
  for (int i = 0; i < 8; i++) {
    val.d[i] =
        __builtin_nontemporal_load(reinterpret_cast<const int*>(src) + i);
  }
  #else
  asm volatile("ld.global.cs.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
               : "=r"(val.d[0]), "=r"(val.d[1]), "=r"(val.d[2]), "=r"(val.d[3]),
                 "=r"(val.d[4]), "=r"(val.d[5]), "=r"(val.d[6]), "=r"(val.d[7])
               : "l"(addr));
  #endif
  return val;
#else
  assert(false && "ld256_cs requires SM100+ with CUDA 12.9+ or ROCm gfx950+");
  return u32x8_t{};
#endif
}

__forceinline__ __device__ void st256_cs(u32x8_t* addr, u32x8_t val) {
#if VLLM_256B_PTX_ENABLED
  #if defined(USE_ROCM) && defined(__gfx950__)
  // gfx950 (CDNA4): 256-bit non-temporal store for streaming write.
  // Emits 2× global_store_dwordx4 nt (no vector dwordx8 in gfx950 ISA).
  // Verified on MI350X, ROCm 7.2 / LLVM 22.0.
  // TODO: Check if future ROCm versions add a vector dwordx8 instruction to the
  // gfx950 ISA.
  int* dst = reinterpret_cast<int*>(addr);
  for (int i = 0; i < 8; i++) {
    __builtin_nontemporal_store(static_cast<int>(val.d[i]), dst + i);
  }
  #else
  asm volatile(
      "st.global.cs.v8.u32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};" ::"l"(addr),
      "r"(val.d[0]), "r"(val.d[1]), "r"(val.d[2]), "r"(val.d[3]), "r"(val.d[4]),
      "r"(val.d[5]), "r"(val.d[6]), "r"(val.d[7]));
  #endif
#else
  assert(false && "st256_cs requires SM100+ with CUDA 12.9+ or ROCm gfx950+");
#endif
}

// 32-bit load / store.
__device__ __forceinline__ int ld32(const int* addr) { return __ldg(addr); }

__device__ __forceinline__ void st32(int* addr, int val) { *addr = val; }

// 32-bit cache-streaming (.cs) load / store.
// ROCm gfx942+: uses __builtin_nontemporal_store for write-side bypass.
// Falls back to ld32/st32 on other ROCm targets.
__forceinline__ __device__ int ld32_cs(const int* addr) {
  int val;
#ifndef USE_ROCM
  asm volatile("ld.global.cs.b32 %0, [%1];" : "=r"(val) : "l"(addr));
#else
  val = ld32(addr);
#endif
  return val;
}

__forceinline__ __device__ void st32_cs(int* addr, int val) {
#ifndef USE_ROCM
  asm volatile("st.global.cs.b32 [%0], %1;" ::"l"(addr), "r"(val));
#elif VLLM_ROCM_USE_NT_HINTS
  __builtin_nontemporal_store(val, addr);
#else
  st32(addr, val);
#endif
}

// 128-bit cache-streaming (.cs) load / store.
// ROCm gfx942+: uses __builtin_nontemporal_store for write-side cache
// bypass. Reads use normal loads to benefit from L2 cache on small/medium
// token counts (NT load hurts latency for batches < 2048).
// Falls back to ld128/st128 on other ROCm targets.
__forceinline__ __device__ int4 ld128_cs(const int4* addr) {
  int4 val;
#ifndef USE_ROCM
  asm volatile("ld.global.cs.v4.u32 {%0,%1,%2,%3}, [%4];"
               : "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w)
               : "l"(addr));
#else
  ld128(val, addr);
#endif
  return val;
}

__forceinline__ __device__ void st128_cs(int4* addr, int4 val) {
#ifndef USE_ROCM
  asm volatile("st.global.cs.v4.u32 [%0], {%1,%2,%3,%4};" ::"l"(addr),
               "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#elif VLLM_ROCM_USE_NT_HINTS
  int* dst = reinterpret_cast<int*>(addr);
  __builtin_nontemporal_store(val.x, dst);
  __builtin_nontemporal_store(val.y, dst + 1);
  __builtin_nontemporal_store(val.z, dst + 2);
  __builtin_nontemporal_store(val.w, dst + 3);
#else
  st128(val, addr);
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
#ifndef USE_ROCM
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
  assert(false && "ld128_cg_or_zero is not supported on ROCm");
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
