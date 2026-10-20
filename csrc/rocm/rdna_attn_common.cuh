// Helpers shared by the RDNA decode and prefill kernels.
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <type_traits>

namespace rdna {

typedef _Float16 h16 __attribute__((ext_vector_type(16)));
typedef _Float16 h2 __attribute__((ext_vector_type(2)));
typedef short s16 __attribute__((ext_vector_type(16)));
typedef short s2v __attribute__((ext_vector_type(2)));
typedef float f8 __attribute__((ext_vector_type(8)));
typedef float f4 __attribute__((ext_vector_type(4)));
typedef unsigned u8v __attribute__((ext_vector_type(8)));
typedef unsigned u4v __attribute__((ext_vector_type(4)));
typedef unsigned u2v __attribute__((ext_vector_type(2)));

// Whether this device pass is for gfx12 (RDNA4).
#if defined(__GFX12__)
inline constexpr bool kGfx12 = true;
#else
inline constexpr bool kGfx12 = false;
#endif

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

  // WMMA 16x16x16 operands. gfx11: a lane holds all 16 k of its row (lane &
  // 15). gfx12: each half-wave holds 8 k, any 8 as long as A and B agree.
  // Both keep gfx11's result layout, element e of lane l being row
  // 2e + (l >> 4): gfx12 loads A rows in a_row order or reads them back with
  // acc_row.
#if defined(__GFX12__)
  static constexpr int kOpWords = 4;
  using op = v8;
  using opw = u4v;
#else
  static constexpr int kOpWords = 8;
  using op = v16;
  using opw = u8v;
#endif

  // This lane's operand of the 16 k at `row`, k contiguous.
  static __device__ __forceinline__ op load_op(const type* row, int laneHalf) {
#if defined(__GFX12__)
    return *(const v8*)(row + 8 * laneHalf);
#else
    return *(const v16*)row;
#endif
  }

  // An operand of this half-wave's 8 k (dword i is word(i)) and, on gfx11,
  // the other half-wave's after them (exchange).
  template <typename Word, typename Exchange>
  static __device__ __forceinline__ op own_first_op(Word word,
                                                    Exchange exchange) {
#if defined(__GFX12__)
    u4v own;
  #pragma unroll
    for (int i = 0; i < 4; ++i) own[i] = word(i);
    return __builtin_bit_cast(v8, own);
#else
    u8v packed;
  #pragma unroll
    for (int i = 0; i < 4; ++i) {
      packed[i] = word(i);
      packed[4 + i] = exchange(packed[i]);
    }
    return __builtin_bit_cast(v16, packed);
#endif
  }

  // GLOBAL_LOAD_TR_B128 (gfx12): within each group of 8 lanes, lane 8g + u
  // receives element u of the 16 bytes each of lanes 8g..8g+7 read, in lane
  // order.
  static __device__ __forceinline__ v8 load_tr(const type* p) {
#if defined(__GFX12__)
    typedef short s8v __attribute__((ext_vector_type(8)));
    return __builtin_bit_cast(v8,
                              __builtin_amdgcn_global_load_tr_b128_v8i16(
                                  (__attribute__((address_space(1))) s8v*)p));
#else
    return *(const v8*)p;  // never used: no gfx11 transpose load
#endif
  }

  static __device__ __forceinline__ f8 wmma(op a, op b, f8 c) {
#if defined(__GFX12__)
    typedef short s8v __attribute__((ext_vector_type(8)));
    typedef _Float16 h8 __attribute__((ext_vector_type(8)));
    if constexpr (kBf16)
      return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(
          __builtin_bit_cast(s8v, a), __builtin_bit_cast(s8v, b), c);
    else
      return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
          __builtin_bit_cast(h8, a), __builtin_bit_cast(h8, b), c);
#else
    if constexpr (kBf16)
      return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(
          __builtin_bit_cast(s16, a), __builtin_bit_cast(s16, b), c);
    else
      return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
          __builtin_bit_cast(h16, a), __builtin_bit_cast(h16, b), c);
#endif
  }
};

// The A row a lane loads so that result element e is row 2e + half.
__device__ __forceinline__ int a_row(int laneRow) {
#if defined(__GFX12__)
  return 2 * (laneRow & 7) + (laneRow >> 3);
#else
  return laneRow;
#endif
}

// The row of result element e, for A tiles loaded in plain lane order.
__device__ __forceinline__ int acc_row(int e, int laneHalf) {
#if defined(__GFX12__)
  return e + 8 * laneHalf;
#else
  return 2 * e + laneHalf;
#endif
}

// Barrier that waits for LDS only, not for global loads. gfx11's LGKMcnt
// also waits for scalar loads; gfx12's DScnt is LDS only.
__device__ __forceinline__ void lds_barrier() {
#if defined(__GFX12__)
  asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait -1" ::
                   : "memory");
#else
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#endif
}

// lds_barrier in two halves, so the work in between (issuing the next global
// loads) overlaps the wait for the other waves. gfx12 only.
__device__ __forceinline__ void lds_arrive() {
#if defined(__GFX12__)
  asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1" ::: "memory");
#endif
}

__device__ __forceinline__ void lds_wait() {
#if defined(__GFX12__)
  asm volatile("s_barrier_wait -1" ::: "memory");
#endif
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

}  // namespace rdna
