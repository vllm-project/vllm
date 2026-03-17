#ifndef PERSISTENT_TOPK_COMMON_CUH_
#define PERSISTENT_TOPK_COMMON_CUH_

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace vllm {
namespace persistent {

// ============================================================================
// Constants
// ============================================================================

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr int RADIX = 256;
constexpr size_t kSmemMedium = 8 * 1024 * sizeof(uint32_t);  // 32KB
constexpr int MAX_BUFFERED_ITEMS = kSmemMedium / (2 * sizeof(int));  // 4096
constexpr uint32_t LARGE_THRESHOLD = 65536;  // 64K

// Decode path constants
constexpr int kDecodeBins = 2048;
constexpr uint32_t DECODE_THRESHOLD = 8192;

// Large path: fixed shared memory for histograms + scalars
constexpr size_t kFixedSmemLarge =
    ((RADIX + RADIX + 5) * sizeof(uint32_t) + 15) & ~size_t(15);

// ============================================================================
// Common helpers
// ============================================================================

__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ auto convert_to_uint8(float x) -> uint8_t {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

// ============================================================================
// Large path: inter-CTA coordination state (one per group)
// ============================================================================

struct RadixRowState {
  uint32_t histogram[3][256];  // Triple-buffered histograms
  uint32_t remaining_k;
  uint32_t prefix;
  int arrival_counter;
  int output_counter;
};

// ============================================================================
// Kernel parameters
// ============================================================================

struct PersistentTopKParams {
  const float* __restrict__ input;   // [num_rows, stride]
  int32_t* __restrict__ output;      // [num_rows, TopK]
  int32_t* __restrict__ lengths;     // [num_rows]
  RadixRowState* row_states;         // large path: per-group state
  uint32_t num_rows;
  uint32_t stride;
  uint32_t chunk_size;               // large path: elements per CTA
  uint32_t ctas_per_group;           // 1=medium, >1=large
};

// ============================================================================
// Vectorized load helpers
// ============================================================================

// Unconditional float4 load with cache hint (.cg = cache at global level only).
__device__ __forceinline__ void load_float4(
    const float* ptr, float& v0, float& v1, float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  asm volatile(
      "ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

// Per-element predicated scalar loads with -inf default.
__device__ __forceinline__ void load_float4_predicated(
    const float* ptr, int base, int seq_len,
    float& v0, float& v1, float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  int p0 = (base     < seq_len);
  int p1 = (base + 1 < seq_len);
  int p2 = (base + 2 < seq_len);
  int p3 = (base + 3 < seq_len);
  asm volatile(
      "{\n"
      "  .reg .pred pr0, pr1, pr2, pr3;\n"
      "  setp.ne.u32 pr0, %4, 0;\n"
      "  setp.ne.u32 pr1, %5, 0;\n"
      "  setp.ne.u32 pr2, %6, 0;\n"
      "  setp.ne.u32 pr3, %7, 0;\n"
      "  mov.u32 %0, 0xFF800000;\n"
      "  mov.u32 %1, 0xFF800000;\n"
      "  mov.u32 %2, 0xFF800000;\n"
      "  mov.u32 %3, 0xFF800000;\n"
      "  @pr0 ld.global.cg.u32 %0, [%8];\n"
      "  @pr1 ld.global.cg.u32 %1, [%8+4];\n"
      "  @pr2 ld.global.cg.u32 %2, [%8+8];\n"
      "  @pr3 ld.global.cg.u32 %3, [%8+12];\n"
      "}\n"
      : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
      : "r"(p0), "r"(p1), "r"(p2), "r"(p3), "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

}  // namespace persistent
}  // namespace vllm

#endif  // PERSISTENT_TOPK_COMMON_CUH_
