#pragma once

#include <cstdint>
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

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
