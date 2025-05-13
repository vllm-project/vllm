#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>

typedef __hip_bfloat16 nv_bfloat16;
typedef __hip_bfloat162 nv_bfloat162;
namespace quickreduce {

#define __device_inline__ __device__ __forceinline__
#define __quickreduce_launch_bounds__ __launch_bounds__(256, 4)

// Setup acquire-release semantics for vector memory reads (mubuf instruction)
// as per architecture.
#if defined(__gfx942__)
// CDNA3: Scope bits sc0, sc1
  #define MUBUF_ACQUIRE 16
  #define MUBUF_RELEASE 16
#elif (defined(__gfx908__) || defined(__gfx90a__))
// CDNA1 and CDNA2 - glc bit
  #define MUBUF_ACQUIRE 1
  #define MUBUF_RELEASE 0
#endif

// Vector types
using int8x8_t = __attribute__((__vector_size__(8 * sizeof(int8_t)))) int8_t;

using int32x2_t = __attribute__((__vector_size__(2 * sizeof(int)))) int;
using int32x4_t = __attribute__((__vector_size__(4 * sizeof(int)))) int;
using int32x8_t = __attribute__((__vector_size__(8 * sizeof(int)))) int;
using int32x16_t = __attribute__((__vector_size__(16 * sizeof(int)))) int;

using fp8_t = uint8_t;
using fp8x8_t = __attribute__((__vector_size__(8 * sizeof(uint8_t)))) uint8_t;

using fp16x4_t = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using fp16x8_t = __attribute__((__vector_size__(8 * sizeof(__fp16)))) __fp16;
using fp16x16_t = __attribute__((__vector_size__(16 * sizeof(__fp16)))) __fp16;

using fp32x2_t = __attribute__((__vector_size__(2 * sizeof(float)))) float;
using fp32x4_t = __attribute__((__vector_size__(4 * sizeof(float)))) float;
using fp32x8_t = __attribute__((__vector_size__(8 * sizeof(float)))) float;
using fp32x16_t = __attribute__((__vector_size__(16 * sizeof(float)))) float;

static int constexpr kNegOne = 0xBC00BC00;  // {-1, -1}, fp16x2_t

// Standard CDNA wavefront size.
static int constexpr kWavefront = 64;

// 256 thread, 4 wavefronts.
static dim3 constexpr kBlock = {64, 4, 1};

// Methods
__device_inline__ __host__ unsigned long divceil(unsigned long x,
                                                 unsigned long y) {
  return ((x + y - 1) / y);
}

union BufferResource {
  __device_inline__ constexpr BufferResource() : config(0x00020000U) {}

  __device_inline__ constexpr BufferResource(void* buffer_address,
                                             uint32_t buffer_size)
      : address(buffer_address), range(buffer_size), config(0x00020000U) {}

  int32x4_t descriptor;
  struct {
    void* address;  // 8B, out of which first 48b is address, and 16b is stride
    // (unused)
    uint32_t range;   // Byte range for the buffer resource
    uint32_t config;  // Constant, DFMT=32b
  };
};

__device_inline__ static int32x4_t buffer_load_dwordx4(
    int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");

__device_inline__ static void buffer_store_dwordx4(
    int32x4_t data, int32x4_t srsrc, int32_t voffset, int32_t soffset,
    int32_t aux) __asm("llvm.amdgcn.raw.buffer.store.v4i32");

__device_inline__ static void set_fp16_ovfl(bool const value) {
  // short size = 0b00001;    // Specifies the bit size to modify
  // const short offset = 0b10111;  // Corrected offset to 23, which is the bit
  // position of FP16_OVFL const short hwRegId = 0b000001; // HW register ID for
  // MODE const short simm16 = (size << 11) | (offset << 6) | hwRegId; simm16 =
  // 0xdc1

#if defined(__gfx942__)
  if (value) {
    asm volatile("s_setreg_imm32_b32 0xdc1, 1;" ::);
  } else {
    asm volatile("s_setreg_imm32_b32 0xdc1, 0;" ::);
  }
#endif
}

template <typename T>
__device_inline__ void packed_assign_add(int32x4_t* A, int32x4_t* B);

template <>
__device_inline__ void packed_assign_add<half>(int32x4_t* A, int32x4_t* B) {
  int32x4_t& tR_fragment = A[0];
  int32x4_t& tA_fragment = B[0];

  asm volatile("v_pk_add_f16 %0, %1, %2"
               : "=v"(tR_fragment[0])
               : "v"(tR_fragment[0]), "v"(tA_fragment[0]));
  asm volatile("v_pk_add_f16 %0, %1, %2"
               : "=v"(tR_fragment[1])
               : "v"(tR_fragment[1]), "v"(tA_fragment[1]));
  asm volatile("v_pk_add_f16 %0, %1, %2"
               : "=v"(tR_fragment[2])
               : "v"(tR_fragment[2]), "v"(tA_fragment[2]));
  asm volatile("v_pk_add_f16 %0, %1, %2"
               : "=v"(tR_fragment[3])
               : "v"(tR_fragment[3]), "v"(tA_fragment[3]));
}

template <>
__device_inline__ void packed_assign_add<nv_bfloat16>(int32x4_t* A,
                                                      int32x4_t* B) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(A);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(B);
#pragma unroll
  for (int i = 0; i < 4; i++) {
    tA[i] = __hadd2(tA[i], tB[i]);
  }
}

template <typename T>
__device_inline__ int packed_max(int a, int b);

template <>
__device_inline__ int packed_max<half>(int a, int b) {
  int result;
  asm volatile("v_pk_max_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <>
__device_inline__ int packed_max<nv_bfloat16>(int a, int b) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(&a);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(&b);
  nv_bfloat162 tR = __hmax2(*tA, *tB);
  return *(reinterpret_cast<int*>(&tR));
}

template <typename T>
__device_inline__ int packed_min(int a, int b);

template <>
__device_inline__ int packed_min<half>(int a, int b) {
  int result;
  asm volatile("v_pk_min_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <>
__device_inline__ int packed_min<nv_bfloat16>(int a, int b) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(&a);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(&b);
  nv_bfloat162 tR = __hmin2(*tA, *tB);
  return *(reinterpret_cast<int*>(&tR));
}

template <typename T>
__device_inline__ int packed_abs_max(int a, int b);

template <>
__device_inline__ int packed_abs_max<half>(int a, int b) {
  half2 wmaxh2 = __builtin_bit_cast(half2, a);
  half2 wminh2 = __builtin_bit_cast(half2, b);
  half2 wblockmaxh2;

  wblockmaxh2.x =
      __hgt(__habs(wmaxh2.x), __habs(wminh2.x)) ? wmaxh2.x : wminh2.x;
  wblockmaxh2.y =
      __hgt(__habs(wmaxh2.y), __habs(wminh2.y)) ? wmaxh2.y : wminh2.y;
  return __builtin_bit_cast(int, wblockmaxh2);
}

template <>
__device_inline__ int packed_abs_max<nv_bfloat16>(int a, int b) {
  nv_bfloat162 wmaxh2 = *(reinterpret_cast<nv_bfloat162*>(&a));
  nv_bfloat162 wminh2 = *(reinterpret_cast<nv_bfloat162*>(&b));
  nv_bfloat162 wblockmaxh2;
  wblockmaxh2.x =
      __hgt(__habs(wmaxh2.x), __habs(wminh2.x)) ? wmaxh2.x : wminh2.x;
  wblockmaxh2.y =
      __hgt(__habs(wmaxh2.y), __habs(wminh2.y)) ? wmaxh2.y : wminh2.y;

  return *(reinterpret_cast<int*>(&wblockmaxh2));
}

template <typename T>
__device_inline__ int packed_add(int a, int b);

template <>
__device_inline__ int packed_add<half>(int a, int b) {
  int result;
  asm volatile("v_pk_add_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <>
__device_inline__ int packed_add<nv_bfloat16>(int a, int b) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(&a);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(&b);
  nv_bfloat162 tR = __hadd2(*tA, *tB);
  return *(reinterpret_cast<int*>(&tR));
}

template <>
__device_inline__ int packed_add<int16_t>(int a, int b) {
  int result;
  asm volatile("v_pk_add_i16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <typename T>
__device_inline__ int packed_sub(int a, int b);

template <>
__device_inline__ int packed_sub<half>(int a, int b) {
  int result;

  // MI300 lacks packed fp16 sub instruction. So we do -1 * min + max
  asm volatile("v_pk_fma_f16 %0, %1, %2 %3"
               : "=v"(result)
               : "v"(kNegOne), "v"(b), "v"(a));
  return result;
}

template <>
__device_inline__ int packed_sub<nv_bfloat16>(int a, int b) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(&a);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(&b);
  nv_bfloat162 tR = __hsub2(*tA, *tB);
  return *(reinterpret_cast<int*>(&tR));
}

template <typename T>
__device_inline__ int packed_mul(int a, int b);

template <>
__device_inline__ int packed_mul<half>(int a, int b) {
  int result;
  asm volatile("v_pk_mul_f16 %0, %1, %2" : "=v"(result) : "v"(a), "v"(b));
  return result;
}

template <>
__device_inline__ int packed_mul<nv_bfloat16>(int a, int b) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(&a);
  nv_bfloat162* tB = reinterpret_cast<nv_bfloat162*>(&b);
  nv_bfloat162 tR = __hmul2(*tA, *tB);
  return *(reinterpret_cast<int*>(&tR));
}

template <typename T>
__device_inline__ int packed_rcp(int a);

template <>
__device_inline__ int packed_rcp<half>(int a) {
  return __builtin_bit_cast(int, h2rcp(__builtin_bit_cast(half2, a)));
}

template <>
__device_inline__ int packed_rcp<nv_bfloat16>(int a) {
  nv_bfloat162* tA = reinterpret_cast<nv_bfloat162*>(&a);
  nv_bfloat162 tR = h2rcp(*tA);
  return *(reinterpret_cast<int*>(&tR));
}

template <typename T>
__device_inline__ T float2T_cast(float a);

template <>
__device_inline__ half float2T_cast<half>(float a) {
  return __float2half(a);
}

template <>
__device_inline__ nv_bfloat16 float2T_cast<nv_bfloat16>(float a) {
  return __float2bfloat16(a);
}

template <typename T>
__device_inline__ float T2float_cast(T a);

template <>
__device_inline__ float T2float_cast<half>(half a) {
  return __half2float(a);
}

template <>
__device_inline__ float T2float_cast<nv_bfloat16>(nv_bfloat16 a) {
  return __bfloat162float(a);
}

}  // namespace quickreduce