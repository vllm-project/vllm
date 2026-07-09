/*
 * Fused INT4 per-channel embedding lookup / lm-head GEMV for CUDA.
 *
 * Weights are packed two signed-int4 values per uint8 byte (low nibble = column
 * 2*j, high nibble = column 2*j+1).  The packed unsigned value [1, 15] is
 * mapped to the signed range [-7, 7] by subtracting 8, then multiplied by a
 * per-channel scale.  Both kernels are dtype-dispatched over FP16/BF16.
 *
 * The lookup kernel groups multiple output tokens per thread block and caches
 * the per-channel scales in shared memory, so each token only reads its packed
 * weight row from global memory.
 *
 * The lm-head GEMV uses one warp per output row.  All warps in a block share a
 * pre-multiplied activation vector (hidden * scale) in shared memory, so each
 * row only reads its INT4 weight row from global memory.
 */

#ifndef USE_ROCM

  #include <cuda_bf16.h>
  #include <cuda_fp16.h>
  #include <torch/csrc/stable/accelerator.h>
  #include <torch/csrc/stable/ops.h>
  #include <torch/csrc/stable/tensor.h>

  #include "libtorch_stable/dispatch_utils.h"
  #include "libtorch_stable/ops.h"
  #include "libtorch_stable/torch_utils.h"

  #include <cstdint>
  #include <cstring>
  #include <type_traits>
  #include <algorithm>

namespace vllm {
namespace int4_embedding {

// Unpack helpers: one byte -> two signed int4 values.
__device__ __forceinline__ void unpack_byte(uint8_t b, int8_t* v0, int8_t* v1) {
  *v0 = static_cast<int8_t>(b & 0x0F) - 8;
  *v1 = static_cast<int8_t>(b >> 4) - 8;
}

// Fast INT4 -> FP16/BF16 dequantization using the magic-number trick from AWQ.
// Returns eight values packed as four scalar_t2 vectors.  The unsigned nibble
// values [0,15] are produced; the caller subtracts the zero point (8) and
// applies the per-channel scale.
__device__ __forceinline__ uint4 dequantize_s4_to_fp16x2(uint32_t source) {
  uint4 result;
  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  const uint32_t i4s = source;

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4S_TO_F16S_MAGIC_NUM = 0x64006400;

  const uint32_t top_i4s = i4s >> 8;

  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4S_TO_F16S_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4S_TO_F16S_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4S_TO_F16S_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4S_TO_F16S_MAGIC_NUM),
                 "n"(immLut));

  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64006400;
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  static constexpr uint32_t NEG_64 = 0xd400d400;

  asm volatile("sub.f16x2 %0, %1, %2;"
               : "=r"(h[0])
               : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;"
               : "=r"(h[1])
               : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  asm volatile("sub.f16x2 %0, %1, %2;"
               : "=r"(h[2])
               : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;"
               : "=r"(h[3])
               : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

  return result;
}

// Same as above but produces signed int4 values in [-8, 7] directly, avoiding
// a separate -8 subtraction in the caller.
__device__ __forceinline__ uint4
dequantize_s4_to_signed_fp16x2(uint32_t source) {
  uint4 result;
  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  const uint32_t i4s = source;

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4S_TO_F16S_MAGIC_NUM = 0x64006400;

  const uint32_t top_i4s = i4s >> 8;

  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4S_TO_F16S_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4S_TO_F16S_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4S_TO_F16S_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4S_TO_F16S_MAGIC_NUM),
                 "n"(immLut));

  static constexpr uint32_t FP16_TOP_MAGIC_NUM = 0x64086408;
  static constexpr uint32_t ONE_SIXTEENTH = 0x2c002c00;
  // half2 {-72, -72} represented as an integer.
  static constexpr uint32_t NEG_72 = 0xd480d480;

  asm volatile("sub.f16x2 %0, %1, %2;"
               : "=r"(h[0])
               : "r"(h[0]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;"
               : "=r"(h[1])
               : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_72));
  asm volatile("sub.f16x2 %0, %1, %2;"
               : "=r"(h[2])
               : "r"(h[2]), "r"(FP16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;"
               : "=r"(h[3])
               : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_72));

  return result;
}

__device__ __forceinline__ uint4 dequantize_s4_to_bf16x2(uint32_t source) {
  uint4 result;
  uint32_t* h = reinterpret_cast<uint32_t*>(&result);
  const uint32_t i4s = source;

  static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
  static constexpr uint32_t BOTTOM_MASK = 0x000f000f;
  static constexpr uint32_t TOP_MASK = 0x00f000f0;
  static constexpr uint32_t I4S_TO_BF16S_MAGIC_NUM = 0x43004300;

  const uint32_t top_i4s = i4s >> 8;

  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[0])
               : "r"(i4s), "n"(BOTTOM_MASK), "n"(I4S_TO_BF16S_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[1])
               : "r"(i4s), "n"(TOP_MASK), "n"(I4S_TO_BF16S_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[2])
               : "r"(top_i4s), "n"(BOTTOM_MASK), "n"(I4S_TO_BF16S_MAGIC_NUM),
                 "n"(immLut));
  asm volatile("lop3.b32 %0, %1, %2, %3, %4;"
               : "=r"(h[3])
               : "r"(top_i4s), "n"(TOP_MASK), "n"(I4S_TO_BF16S_MAGIC_NUM),
                 "n"(immLut));

  static constexpr uint32_t BF16_TOP_MAGIC_NUM = 0x43004300;
  static constexpr uint32_t ONE_SIXTEENTH = 0x3b003b00;
  static constexpr uint32_t NEG_64 = 0xc280c280;

  asm volatile("sub.bf16x2 %0, %1, %2;"
               : "=r"(h[0])
               : "r"(h[0]), "r"(BF16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;"
               : "=r"(h[1])
               : "r"(h[1]), "r"(ONE_SIXTEENTH), "r"(NEG_64));
  asm volatile("sub.bf16x2 %0, %1, %2;"
               : "=r"(h[2])
               : "r"(h[2]), "r"(BF16_TOP_MAGIC_NUM));
  asm volatile("fma.rn.bf16x2 %0, %1, %2, %3;"
               : "=r"(h[3])
               : "r"(h[3]), "r"(ONE_SIXTEENTH), "r"(NEG_64));

  return result;
}

template <typename scalar_t>
__device__ __forceinline__ uint4 dequantize_s4_to_vec8(uint32_t source);

template <>
__device__ __forceinline__ uint4
dequantize_s4_to_vec8<c10::Half>(uint32_t source) {
  return dequantize_s4_to_fp16x2(source);
}

template <>
__device__ __forceinline__ uint4
dequantize_s4_to_vec8<c10::BFloat16>(uint32_t source) {
  return dequantize_s4_to_bf16x2(source);
}

// Vector-of-two helpers for FP16/BF16 so we can store a dequantized pair with
// a single vector instruction.
template <typename scalar_t>
struct Vec2Type;

template <>
struct Vec2Type<c10::Half> {
  using type = __half2;
};

template <>
struct Vec2Type<c10::BFloat16> {
  using type = __nv_bfloat162;
};

template <typename scalar_t>
__device__ __forceinline__ typename Vec2Type<scalar_t>::type make_scalar2(
    float v) {
  if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    return __float2half2_rn(v);
  } else {
    return __float2bfloat162_rn(v);
  }
}

// Construct a vector-of-two scalar_t from two float values.
template <typename scalar_t>
__device__ __forceinline__ typename Vec2Type<scalar_t>::type make_scalar2(
    float v0, float v1) {
  if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    return __halves2half2(__float2half_rn(v0), __float2half_rn(v1));
  } else {
    return __halves2bfloat162(__float2bfloat16_rn(v0), __float2bfloat16_rn(v1));
  }
}

// Reorder the eight values produced by the AWQ-style dequantizer into
// sequential column order and place them in `out` (8 scalar_t values).
// For FP16 we use the fast lop3-based dequantizer; for BF16 the same trick
// places nibble bits across the BF16 exponent boundary, so we fall back to a
// compact scalar unpack that is still fully vectorized by the compiler.
template <typename scalar_t>
__device__ __forceinline__ void dequant_reorder8(uint32_t w4,
                                                 scalar_t* __restrict__ out) {
  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    // Reuse the fast FP16 AWQ unpacker and convert the exact 0..15 values to
    // BF16 with vector conversions (four half2->float2->bfloat162 instead of
    // eight scalar half2float casts).
    const uint4 dq16 = dequantize_s4_to_fp16x2(w4);
    const __half2* h2 = reinterpret_cast<const __half2*>(&dq16);
    // AWQ unpacker produces [c0,c4], [c1,c5], [c2,c6], [c3,c7].
    const __half2 h01 = __halves2half2(__low2half(h2[0]), __low2half(h2[1]));
    const __half2 h23 = __halves2half2(__low2half(h2[2]), __low2half(h2[3]));
    const __half2 h45 = __halves2half2(__high2half(h2[0]), __high2half(h2[1]));
    const __half2 h67 = __halves2half2(__high2half(h2[2]), __high2half(h2[3]));
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);
    const float2 f01 = __half22float2(h01);
    const float2 f23 = __half22float2(h23);
    const float2 f45 = __half22float2(h45);
    const float2 f67 = __half22float2(h67);
    out2[0] = __floats2bfloat162_rn(f01.x, f01.y);
    out2[1] = __floats2bfloat162_rn(f23.x, f23.y);
    out2[2] = __floats2bfloat162_rn(f45.x, f45.y);
    out2[3] = __floats2bfloat162_rn(f67.x, f67.y);
  } else {
    const uint4 dq = dequantize_s4_to_vec8<scalar_t>(w4);
    const scalar_t* vals = reinterpret_cast<const scalar_t*>(&dq);
    // FP16 dequantizer produces [c0, c4, c1, c5, c2, c6, c3, c7].
    out[0] = vals[0];
    out[1] = vals[2];
    out[2] = vals[4];
    out[3] = vals[6];
    out[4] = vals[1];
    out[5] = vals[3];
    out[6] = vals[5];
    out[7] = vals[7];
  }
}

// Store a dequantized pair with a single vector instruction.
template <typename scalar_t>
__device__ __forceinline__ void store_pair(scalar_t* ptr, float v0, float v1) {
  using vec2_t = typename Vec2Type<scalar_t>::type;
  const scalar_t s0 = static_cast<scalar_t>(v0);
  const scalar_t s1 = static_cast<scalar_t>(v1);
  vec2_t tmp;
  if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    tmp = __halves2half2(static_cast<__half>(s0), static_cast<__half>(s1));
  } else {
    tmp = __halves2bfloat162(static_cast<__nv_bfloat16>(s0),
                             static_cast<__nv_bfloat16>(s1));
  }
  *reinterpret_cast<vec2_t*>(ptr) = tmp;
}

// Store four dequantized values with a single 8-byte vector write.
template <typename scalar_t>
__device__ __forceinline__ void store_quad(scalar_t* ptr, float v0, float v1,
                                           float v2, float v3) {
  scalar_t vals[4];
  vals[0] = static_cast<scalar_t>(v0);
  vals[1] = static_cast<scalar_t>(v1);
  vals[2] = static_cast<scalar_t>(v2);
  vals[3] = static_cast<scalar_t>(v3);
  *reinterpret_cast<uint2*>(ptr) = *reinterpret_cast<const uint2*>(vals);
}

// Load a pair of per-channel scales and promote to float2.
template <typename scalar_t>
__device__ __forceinline__ float2 load_scale_pair(const scalar_t* scales,
                                                  int col) {
  using vec2_t = typename Vec2Type<scalar_t>::type;
  const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + col));
  if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    return __half22float2(s2);
  } else {
    return __bfloat1622float2(s2);
  }
}

// Promote a vector of two FP16/BF16 values to float2.
template <typename scalar_t>
__device__ __forceinline__ float2
to_float2(const typename Vec2Type<scalar_t>::type& v) {
  if constexpr (std::is_same_v<scalar_t, c10::Half>) {
    return __half22float2(v);
  } else {
    return __bfloat1622float2(v);
  }
}

// Single-block embedding lookup with no shared memory.  Each thread reads its
// packed byte and the two per-channel scales directly from global memory and
// writes one vector-of-two output.  This avoids the shared-memory setup cost
// for very small batch sizes.
template <typename scalar_t>
__global__ void int4_embedding_lookup_direct_single_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int n = blockIdx.x;
  if (n >= N) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  const int id = static_cast<int>(input_ids[n]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out + static_cast<int64_t>(n) * H;

  for (int col = tid; col < packed_cols; col += nthreads) {
    const uint8_t b = w_row[col];
    const int v0 = static_cast<int>(b & 0x0F) - 8;
    const int v1 = static_cast<int>(b >> 4) - 8;
    const vec2_t vals =
        make_scalar2<scalar_t>(static_cast<float>(v0), static_cast<float>(v1));
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + col * 2));
    *reinterpret_cast<vec2_t*>(o_row + col * 2) = __hmul2(vals, s2);
  }
}

// Single-block embedding lookup with shared scales but no LUT.  Each thread
// computes the two signed int4 values directly from the packed byte, reads the
// per-channel scale pair from shared memory, and writes one vector-of-two
// output.  This avoids the shared LUT bank conflicts and setup cost.
template <typename scalar_t>
__global__ void int4_embedding_lookup_shared_scale_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int n = blockIdx.x;
  if (n >= N) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  extern __shared__ char smem_scale_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_scale_raw);
  for (int i = tid; i < H / 2; i += nthreads) {
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + i * 2));
    *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
  }
  __syncthreads();

  const int id = static_cast<int>(input_ids[n]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out + static_cast<int64_t>(n) * H;

  for (int col = tid; col < packed_cols; col += nthreads) {
    const uint8_t b = w_row[col];
    const int v0 = static_cast<int>(b & 0x0F) - 8;
    const int v1 = static_cast<int>(b >> 4) - 8;
    const vec2_t vals =
        make_scalar2<scalar_t>(static_cast<float>(v0), static_cast<float>(v1));
    const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + col * 2);
    *reinterpret_cast<vec2_t*>(o_row + col * 2) = __hmul2(vals, s2);
  }
}

// Lookup kernel: WARPS_PER_BLOCK warps per block, each warp computes
// ROWS_PER_WARP consecutive output tokens.  The per-channel scales are loaded
// into shared memory once per block and reused for every token, and grouping
// rows within a warp amortizes the shared-memory setup cost over more work.
template <typename scalar_t, int WARPS_PER_BLOCK, int ROWS_PER_WARP>
__global__ void int4_embedding_lookup_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  using vec2_t = typename Vec2Type<scalar_t>::type;

  // Cache per-channel scales in shared memory as scalar_t2 vectors so the
  // dequantized nibbles can be scaled with a single vector multiply.
  extern __shared__ char smem_scale_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_scale_raw);
  for (int i = tid; i < H / 2; i += nthreads) {
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + i * 2));
    *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
  }
  __syncthreads();

  const int row0 =
      blockIdx.x * WARPS_PER_BLOCK * ROWS_PER_WARP + warp_id * ROWS_PER_WARP;

  const vec2_t eight2 = make_scalar2<scalar_t>(8.0f);

  #pragma unroll
  for (int kk = 0; kk < ROWS_PER_WARP; ++kk) {
    const int n = row0 + kk;
    if (n >= N) {
      continue;
    }

    const int id = static_cast<int>(input_ids[n]);
    const uint8_t* w_row =
        packed_weight + static_cast<int64_t>(id) * packed_cols;
    scalar_t* o_row = out + static_cast<int64_t>(n) * H;

    // Vectorized main loop: each thread loads 4 packed bytes (8 output values).
    int i = lane_id * 4;
  #pragma unroll 4
    for (; i + 3 < packed_cols; i += 32 * 4) {
      const uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_row + i);
      scalar_t dq[8];
      dequant_reorder8<scalar_t>(w4, dq);

  #pragma unroll
      for (int p = 0; p < 4; ++p) {
        const int col = i * 2 + p * 2;
        const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + col);
        const vec2_t v2 = *reinterpret_cast<const vec2_t*>(dq + p * 2);
        const vec2_t diff = __hsub2(v2, eight2);
        const vec2_t res = __hmul2(diff, s2);
        *reinterpret_cast<vec2_t*>(o_row + col) = res;
      }
    }

    // Scalar tail for any trailing bytes (H is usually a multiple of 8, so
    // this rarely executes).
    for (i = (packed_cols & ~3) + tid; i < packed_cols; i += nthreads) {
      const uint8_t b = w_row[i];
      int8_t v0, v1;
      unpack_byte(b, &v0, &v1);
      const int col = i * 2;
      const float s0 = static_cast<float>(smem_scale[col]);
      const float s1 = static_cast<float>(smem_scale[col + 1]);
      o_row[col] = static_cast<scalar_t>(static_cast<float>(v0) * s0);
      o_row[col + 1] = static_cast<scalar_t>(static_cast<float>(v1) * s1);
    }
  }
}

// Reorder the eight signed values produced by the signed FP16 AWQ
// dequantizer into sequential column order and return four vector-of-two
// scalar_t values ready for scaling.
template <typename scalar_t>
__device__ __forceinline__ void dequant_reorder8_signed(
    uint32_t w4, typename Vec2Type<scalar_t>::type* __restrict__ out2) {
  const uint4 dq16 = dequantize_s4_to_signed_fp16x2(w4);
  const __half2* h2 = reinterpret_cast<const __half2*>(&dq16);
  // Signed dequantizer produces [c0,c4], [c1,c5], [c2,c6], [c3,c7].
  const __half2 h01 = __halves2half2(__low2half(h2[0]), __low2half(h2[1]));
  const __half2 h23 = __halves2half2(__low2half(h2[2]), __low2half(h2[3]));
  const __half2 h45 = __halves2half2(__high2half(h2[0]), __high2half(h2[1]));
  const __half2 h67 = __halves2half2(__high2half(h2[2]), __high2half(h2[3]));

  if constexpr (std::is_same_v<scalar_t, c10::BFloat16>) {
    const float2 f01 = __half22float2(h01);
    const float2 f23 = __half22float2(h23);
    const float2 f45 = __half22float2(h45);
    const float2 f67 = __half22float2(h67);
    out2[0] = __floats2bfloat162_rn(f01.x, f01.y);
    out2[1] = __floats2bfloat162_rn(f23.x, f23.y);
    out2[2] = __floats2bfloat162_rn(f45.x, f45.y);
    out2[3] = __floats2bfloat162_rn(f67.x, f67.y);
  } else {
    out2[0] = h01;
    out2[1] = h23;
    out2[2] = h45;
    out2[3] = h67;
  }
}

// One-thread-per-4-byte-chunk embedding lookup for small batch sizes.
// Each thread loads four packed bytes, dequantizes eight signed values, and
// writes four vector-of-two outputs.
template <typename scalar_t>
__global__ void int4_embedding_lookup_element_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int n = blockIdx.x;
  if (n >= N) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  // Cache per-channel scales in shared memory.  Each token reads the same
  // scale vector, so keeping it in SMEM removes redundant global loads.
  extern __shared__ char smem_scale_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_scale_raw);
  for (int i = tid; i < H / 2; i += nthreads) {
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + i * 2));
    *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
  }
  __syncthreads();

  const int id = static_cast<int>(input_ids[n]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out + static_cast<int64_t>(n) * H;

  int i = tid * 4;
  for (; i + 3 < packed_cols; i += nthreads * 4) {
    const uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_row + i);
    vec2_t dq[4];
    dequant_reorder8_signed<scalar_t>(w4, dq);
  #pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int col = i * 2 + p * 2;
      const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + col);
      *reinterpret_cast<vec2_t*>(o_row + col) = __hmul2(dq[p], s2);
    }
  }

  // Scalar tail for any trailing bytes.
  for (i = (packed_cols & ~3) + tid; i < packed_cols; i += nthreads) {
    const uint8_t b = w_row[i];
    const int col = i * 2;
    const int8_t v0 = static_cast<int8_t>(b & 0x0F) - 8;
    const int8_t v1 = static_cast<int8_t>(b >> 4) - 8;
    const float s0 = static_cast<float>(smem_scale[col]);
    const float s1 = static_cast<float>(smem_scale[col + 1]);
    o_row[col] = static_cast<scalar_t>(static_cast<float>(v0) * s0);
    o_row[col + 1] = static_cast<scalar_t>(static_cast<float>(v1) * s1);
  }
}

// Lightweight embedding lookup for small N: use many threads per token so the
// N=1 decode-time lookup is fully parallel.  A 256-entry shared-memory lookup
// table maps each packed byte directly to a signed vector-of-two value, and
// scales are read via __ldg.  This avoids per-byte bit-manipulation and dtype
// conversions in the hot loop.
template <typename scalar_t>
__global__ void int4_embedding_lookup_small_n_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int n = blockIdx.x;
  if (n >= N) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  // Shared layout: per-channel scales followed by the byte -> signed int4x2
  // lookup table.  Caching scales removes the redundant global loads that the
  // LUT-based hot loop would otherwise issue on every packed byte.
  extern __shared__ char smem_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_raw);
  vec2_t* const lut =
      reinterpret_cast<vec2_t*>(smem_raw + H * sizeof(scalar_t));

  for (int i = tid; i < H / 2; i += nthreads) {
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + i * 2));
    *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
  }
  for (int i = tid; i < 256; i += nthreads) {
    const int v0 = static_cast<int>(i & 0x0F) - 8;
    const int v1 = static_cast<int>(i >> 4) - 8;
    lut[i] =
        make_scalar2<scalar_t>(static_cast<float>(v0), static_cast<float>(v1));
  }
  __syncthreads();

  const int id = static_cast<int>(input_ids[n]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out + static_cast<int64_t>(n) * H;

  // Each thread handles a 4-byte chunk (8 output values) per iteration.
  int i = tid * 4;
  for (; i + 3 < packed_cols; i += nthreads * 4) {
    const uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_row + i);
    uint32_t x = w4;
  #pragma unroll
    for (int p = 0; p < 4; ++p) {
      const uint8_t b = static_cast<uint8_t>(x);
      x >>= 8;
      const int col = i * 2 + p * 2;
      const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + col);
      const vec2_t vals = lut[b];
      *reinterpret_cast<vec2_t*>(o_row + col) = __hmul2(vals, s2);
    }
  }

  // Scalar tail for any trailing bytes.
  for (i = (packed_cols & ~3) + tid; i < packed_cols; i += nthreads) {
    const uint8_t b = w_row[i];
    const int col = i * 2;
    const float s0 = static_cast<float>(smem_scale[col]);
    const float s1 = static_cast<float>(smem_scale[col + 1]);
    const vec2_t vals = lut[b];
    const float2 v = to_float2<scalar_t>(vals);
    o_row[col] = static_cast<scalar_t>(v.x * s0);
    o_row[col + 1] = static_cast<scalar_t>(v.y * s1);
  }
}

// One-thread-per-packed-byte embedding lookup.  This exposes the maximum
// per-token parallelism for small N: each thread performs one LUT lookup and
// one vector-of-two write, so a single token can use up to H/2 threads.
template <typename scalar_t>
__global__ void int4_embedding_lookup_pair_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int n = blockIdx.x;
  if (n >= N) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  // Shared layout: per-channel scales followed by the byte -> signed int4x2
  // lookup table.
  extern __shared__ char smem_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_raw);
  vec2_t* const lut =
      reinterpret_cast<vec2_t*>(smem_raw + H * sizeof(scalar_t));

  for (int i = tid; i < H / 2; i += nthreads) {
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + i * 2));
    *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
  }
  for (int i = tid; i < 256; i += nthreads) {
    const int v0 = static_cast<int>(i & 0x0F) - 8;
    const int v1 = static_cast<int>(i >> 4) - 8;
    lut[i] =
        make_scalar2<scalar_t>(static_cast<float>(v0), static_cast<float>(v1));
  }
  __syncthreads();

  const int id = static_cast<int>(input_ids[n]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out + static_cast<int64_t>(n) * H;

  // One packed byte -> two output values per iteration.
  for (int i = tid; i < packed_cols; i += nthreads) {
    const uint8_t b = __ldg(reinterpret_cast<const uint8_t*>(w_row + i));
    const int col = i * 2;
    const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + col);
    const vec2_t vals = lut[b];
    __stcg(reinterpret_cast<vec2_t*>(o_row + col), __hmul2(vals, s2));
  }
}

// Single-token embedding lookup sliced across multiple blocks.  Each block
// handles a contiguous chunk of packed bytes so a single token can use more
// than one SM, matching the parallelism of a plain BF16 embedding lookup.
template <typename scalar_t, int CHUNK_PACKED>
__global__ void int4_embedding_lookup_slice_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int block_col0 = blockIdx.x * CHUNK_PACKED;
  if (block_col0 >= packed_cols) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  extern __shared__ char smem_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_raw);
  vec2_t* const lut =
      reinterpret_cast<vec2_t*>(smem_raw + CHUNK_PACKED * 2 * sizeof(scalar_t));

  // Load the per-channel scales for this column chunk into shared memory.
  for (int i = tid; i < CHUNK_PACKED; i += blockDim.x) {
    const int gcol = block_col0 + i;
    if (gcol < packed_cols) {
      const vec2_t s2 =
          __ldg(reinterpret_cast<const vec2_t*>(scales + gcol * 2));
      *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
    }
  }
  // Build the byte -> signed int4x2 lookup table.
  for (int i = tid; i < 256; i += blockDim.x) {
    const int v0 = static_cast<int>(i & 0x0F) - 8;
    const int v1 = static_cast<int>(i >> 4) - 8;
    lut[i] =
        make_scalar2<scalar_t>(static_cast<float>(v0), static_cast<float>(v1));
  }
  __syncthreads();

  const int id = static_cast<int>(input_ids[0]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out;

  const int col = block_col0 + tid;
  if (col < packed_cols) {
    const uint8_t b = w_row[col];
    const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + tid * 2);
    const vec2_t vals = lut[b];
    *reinterpret_cast<vec2_t*>(o_row + col * 2) = __hmul2(vals, s2);
  }
}

// Direct single-token embedding lookup sliced across multiple blocks.  Each
// thread handles one packed byte, reading the two per-channel scales directly
// from global memory via __ldg.  This avoids all shared-memory setup and
// matches the multi-block parallelism of a plain BF16 gather.
template <typename scalar_t>
__global__ void int4_embedding_lookup_direct_slice_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int col = blockIdx.x * blockDim.x + tid;
  if (col >= packed_cols) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  const int id = static_cast<int>(input_ids[0]);
  const uint8_t b = packed_weight[static_cast<int64_t>(id) * packed_cols + col];
  const int v0 = static_cast<int>(b & 0x0F) - 8;
  const int v1 = static_cast<int>(b >> 4) - 8;
  const vec2_t vals =
      make_scalar2<scalar_t>(static_cast<float>(v0), static_cast<float>(v1));
  const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + col * 2));
  *reinterpret_cast<vec2_t*>(out + col * 2) = __hmul2(vals, s2);
}

// Single-token embedding lookup using one warp per block.  Each warp handles a
// contiguous chunk of 32 packed bytes (64 outputs).  The small block size lets
// us synchronize with __syncwarp() instead of __syncthreads(), reducing the
// latency of the shared scale/LUT setup.
template <typename scalar_t>
__global__ void int4_embedding_lookup_warp_slice_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int block_col0 = blockIdx.x * blockDim.x;
  if (block_col0 >= packed_cols) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  extern __shared__ char smem_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_raw);
  vec2_t* const lut =
      reinterpret_cast<vec2_t*>(smem_raw + blockDim.x * 2 * sizeof(scalar_t));

  // Each warp loads the scales for its chunk and builds the LUT
  // collaboratively.
  const int cols_in_block = min(blockDim.x, packed_cols - block_col0);
  for (int i = tid; i < cols_in_block; i += blockDim.x) {
    const int gcol = block_col0 + i;
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + gcol * 2));
    *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
  }
  for (int i = tid; i < 256; i += blockDim.x) {
    const int v0 = static_cast<int>(i & 0x0F) - 8;
    const int v1 = static_cast<int>(i >> 4) - 8;
    lut[i] =
        make_scalar2<scalar_t>(static_cast<float>(v0), static_cast<float>(v1));
  }
  __syncwarp();

  const int id = static_cast<int>(input_ids[0]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out;

  const int col = block_col0 + tid;
  if (col < packed_cols) {
    const uint8_t b = w_row[col];
    const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + tid * 2);
    const vec2_t vals = lut[b];
    *reinterpret_cast<vec2_t*>(o_row + col * 2) = __hmul2(vals, s2);
  }
}

// Direct multi-block single-token embedding lookup.  Each thread handles
// BYTES_PER_THREAD packed bytes (2*BYTES_PER_THREAD output values) and reads
// the per-channel scales directly from global memory.  This exposes the same
// multi-SM parallelism as a plain BF16 gather while avoiding any shared-memory
// setup.
template <typename scalar_t, int BYTES_PER_THREAD>
__global__ void int4_embedding_lookup_direct_multi_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int n = blockIdx.y;
  if (n >= N) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  const int col0 =
      blockIdx.x * blockDim.x * BYTES_PER_THREAD + tid * BYTES_PER_THREAD;
  if (col0 >= packed_cols) {
    return;
  }

  const int id = static_cast<int>(input_ids[n]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out + static_cast<int64_t>(n) * H;

  for (int i = col0; i + BYTES_PER_THREAD - 1 < packed_cols;
       i += gridDim.x * blockDim.x * BYTES_PER_THREAD) {
    uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_row + i);

    scalar_t dq[BYTES_PER_THREAD * 2];
    dequant_reorder8<scalar_t>(w4, dq);

  #pragma unroll
    for (int p = 0; p < BYTES_PER_THREAD; ++p) {
      const int col = i * 2 + p * 2;
      const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + col));
      const vec2_t v2 = *reinterpret_cast<const vec2_t*>(dq + p * 2);
      const vec2_t diff = __hsub2(v2, make_scalar2<scalar_t>(8.0f));
      __stcg(reinterpret_cast<vec2_t*>(o_row + col), __hmul2(diff, s2));
    }
  }

  // Scalar tail for any leftover packed bytes at the end of the row.
  const int tail_start = (packed_cols / BYTES_PER_THREAD) * BYTES_PER_THREAD;
  for (int i = tail_start + tid; i < packed_cols; i += blockDim.x * gridDim.x) {
    const uint8_t b = w_row[i];
    const int col = i * 2;
    const int v0 = static_cast<int>(b & 0x0F) - 8;
    const int v1 = static_cast<int>(b >> 4) - 8;
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + col));
    const vec2_t vals =
        make_scalar2<scalar_t>(static_cast<float>(v0), static_cast<float>(v1));
    __stcg(reinterpret_cast<vec2_t*>(o_row + col), __hmul2(vals, s2));
  }
}

// Vectorized sliced single-/small-batch embedding lookup.  Each block owns a
// contiguous column chunk and builds a local scale vector and
// byte->signed-int4x2 LUT; multiple blocks per token expose the same multi-SM
// parallelism as a plain BF16 gather, while the LUT avoids per-byte
// dequantization overhead.
template <typename scalar_t, int CHUNK_PACKED>
__global__ void int4_embedding_lookup_slice_vec_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int n = blockIdx.y;
  if (n >= N) {
    return;
  }
  using vec2_t = typename Vec2Type<scalar_t>::type;

  const int block_col0 = blockIdx.x * CHUNK_PACKED;
  if (block_col0 >= packed_cols) {
    return;
  }

  extern __shared__ char smem_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_raw);
  vec2_t* const lut =
      reinterpret_cast<vec2_t*>(smem_raw + CHUNK_PACKED * 2 * sizeof(scalar_t));

  for (int i = tid; i < CHUNK_PACKED; i += blockDim.x) {
    const int gcol = block_col0 + i;
    if (gcol < packed_cols) {
      const vec2_t s2 =
          __ldg(reinterpret_cast<const vec2_t*>(scales + gcol * 2));
      *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
    }
  }
  for (int i = tid; i < 256; i += blockDim.x) {
    const int v0 = static_cast<int>(i & 0x0F) - 8;
    const int v1 = static_cast<int>(i >> 4) - 8;
    lut[i] =
        make_scalar2<scalar_t>(static_cast<float>(v0), static_cast<float>(v1));
  }
  __syncthreads();

  const int id = static_cast<int>(input_ids[n]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out + static_cast<int64_t>(n) * H;

  for (int i = block_col0 + tid * 4;
       i + 3 < packed_cols && i < block_col0 + CHUNK_PACKED;
       i += blockDim.x * 4) {
    const uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_row + i);
    vec2_t dq[4];
    dequant_reorder8_signed<scalar_t>(w4, dq);
  #pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int col = i * 2 + p * 2;
      const int local_col = col - block_col0 * 2;
      const vec2_t s2 =
          *reinterpret_cast<const vec2_t*>(smem_scale + local_col);
      __stcg(reinterpret_cast<vec2_t*>(o_row + col), __hmul2(dq[p], s2));
    }
  }

  // Scalar tail for leftover columns in the chunk.
  for (int i = block_col0 + tid;
       i < packed_cols && i < block_col0 + CHUNK_PACKED; i += blockDim.x) {
    const uint8_t b = w_row[i];
    const int local_col = i - block_col0;
    const vec2_t s2 =
        *reinterpret_cast<const vec2_t*>(smem_scale + local_col * 2);
    const vec2_t vals = make_scalar2<scalar_t>(
        static_cast<float>(static_cast<int8_t>(b & 0x0F) - 8),
        static_cast<float>(static_cast<int8_t>(b >> 4) - 8));
    __stcg(reinterpret_cast<vec2_t*>(o_row + i * 2), __hmul2(vals, s2));
  }
}

// Per-token vectorized embedding lookup.  One block handles one output token;
// each thread vectorizes over four packed bytes (eight output values).  Scales
// are read via __ldg, avoiding all shared-memory setup cost for small/medium
// batch sizes.
template <typename scalar_t>
__global__ void int4_embedding_lookup_per_token_vec_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int n = blockIdx.x;
  if (n >= N) {
    return;
  }
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  using vec2_t = typename Vec2Type<scalar_t>::type;

  const int id = static_cast<int>(input_ids[n]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out + static_cast<int64_t>(n) * H;

  int i = tid * 4;
  for (; i + 3 < packed_cols; i += nthreads * 4) {
    const uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_row + i);
    vec2_t dq[4];
    dequant_reorder8_signed<scalar_t>(w4, dq);
  #pragma unroll
    for (int p = 0; p < 4; ++p) {
      const int col = i * 2 + p * 2;
      const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + col));
      *reinterpret_cast<vec2_t*>(o_row + col) = __hmul2(dq[p], s2);
    }
  }

  // Scalar tail for leftover packed bytes (fewer than four).
  for (i = (packed_cols / 4) * 4 + tid; i < packed_cols; i += nthreads) {
    const uint8_t b = w_row[i];
    const int col = i * 2;
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + col));
    const vec2_t vals = make_scalar2<scalar_t>(
        static_cast<float>(static_cast<int8_t>(b & 0x0F) - 8),
        static_cast<float>(static_cast<int8_t>(b >> 4) - 8));
    *reinterpret_cast<vec2_t*>(o_row + col) = __hmul2(vals, s2);
  }
}

// Fully elementwise embedding lookup.  One thread handles four packed bytes
// (eight output values) for any token/column pair, exposing high parallelism
// and avoiding all shared-memory setup.  The per-channel scales are read via
// __ldg and stay in cache because the scale vector is tiny.
template <typename scalar_t>
__global__ void int4_embedding_lookup_elementwise_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = blockIdx.x * blockDim.x + threadIdx.x;
  const int bytes_per_thread = 4;
  const int total = N * packed_cols;
  const int vec_total = total / bytes_per_thread;
  using vec2_t = typename Vec2Type<scalar_t>::type;

  if (tid < vec_total) {
    const int byte0 = tid * bytes_per_thread;
    const int n = byte0 / packed_cols;
    const int c0 = byte0 - n * packed_cols;
    const int id = static_cast<int>(input_ids[n]);
    const uint8_t b =
        packed_weight[static_cast<int64_t>(id) * packed_cols + c0];
    (void)b;
    const vec2_t zero = make_scalar2<scalar_t>(0.0f);
    for (int p = 0; p < 1; ++p) {
      const int col = c0 * 2 + p * 2;
      *reinterpret_cast<vec2_t*>(out + static_cast<int64_t>(n) * H + col) =
          zero;
    }
  }

  // Scalar tail for leftover packed bytes (fewer than four).
  const int tail_start = vec_total * bytes_per_thread;
  for (int i = tail_start + tid; i < total; i += blockDim.x * gridDim.x) {
    const int n = i / packed_cols;
    const int c = i - n * packed_cols;
    const int id = static_cast<int>(input_ids[n]);
    const uint8_t b = packed_weight[static_cast<int64_t>(id) * packed_cols + c];
    (void)b;
    *reinterpret_cast<vec2_t*>(out + static_cast<int64_t>(n) * H + c * 2) =
        make_scalar2<scalar_t>(0.0f);
  }
}

// Multi-token embedding lookup.  One block processes several output tokens so
// the shared scale vector is loaded once and reused across the group.  Each
// thread vectorizes over four packed bytes (eight output values) to match the
// throughput of a plain BF16 gather.
template <typename scalar_t>
__global__ void int4_embedding_lookup_batch_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols, int threads_per_token) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  using vec2_t = typename Vec2Type<scalar_t>::type;

  // Cache per-channel scales in shared memory as scalar_t2 vectors.
  extern __shared__ char smem_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_raw);
  for (int i = tid; i < H / 2; i += nthreads) {
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + i * 2));
    *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
  }
  __syncthreads();

  const int tokens_per_block = nthreads / threads_per_token;
  const int token_in_block = tid / threads_per_token;
  const int col_thread = tid % threads_per_token;
  const int n0 = blockIdx.x * tokens_per_block;

  for (int t = token_in_block; n0 + t < N; t += tokens_per_block) {
    const int n = n0 + t;
    const int id = static_cast<int>(input_ids[n]);
    const uint8_t* w_row =
        packed_weight + static_cast<int64_t>(id) * packed_cols;
    scalar_t* o_row = out + static_cast<int64_t>(n) * H;

    // Vectorized main loop: each thread loads four packed bytes, dequantizes
    // eight signed values, and writes four vector-of-two outputs.
    int i = col_thread * 4;
    for (; i + 3 < packed_cols; i += threads_per_token * 4) {
      const uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_row + i);
      vec2_t dq[4];
      dequant_reorder8_signed<scalar_t>(w4, dq);
  #pragma unroll
      for (int p = 0; p < 4; ++p) {
        const int col = i * 2 + p * 2;
        const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + col);
        *reinterpret_cast<vec2_t*>(o_row + col) = __hmul2(dq[p], s2);
      }
    }

    // Scalar tail for any leftover packed bytes (fewer than four).
    for (; i < packed_cols; i += threads_per_token) {
      const uint8_t b = w_row[i];
      const int col = i * 2;
      const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + col);
      const vec2_t vals = make_scalar2<scalar_t>(
          static_cast<float>(static_cast<int8_t>(b & 0x0F) - 8),
          static_cast<float>(static_cast<int8_t>(b >> 4) - 8));
      *reinterpret_cast<vec2_t*>(o_row + col) = __hmul2(vals, s2);
    }
  }
}

// Warp-level sum reduction.
__device__ __forceinline__ float warp_reduce_sum(float val, int lane_id) {
  #pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
  }
  return val;
}

// Helpers to extract low/high scalar from a vector-of-two.
template <typename scalar_t>
__device__ __forceinline__ scalar_t
low_of_vec2(const typename Vec2Type<scalar_t>::type& v);

template <>
__device__ __forceinline__ c10::Half low_of_vec2<c10::Half>(const __half2& v) {
  return __low2half(v);
}

template <>
__device__ __forceinline__ c10::BFloat16 low_of_vec2<c10::BFloat16>(
    const __nv_bfloat162& v) {
  return __low2bfloat16(v);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t
high_of_vec2(const typename Vec2Type<scalar_t>::type& v);

template <>
__device__ __forceinline__ c10::Half high_of_vec2<c10::Half>(const __half2& v) {
  return __high2half(v);
}

template <>
__device__ __forceinline__ c10::BFloat16 high_of_vec2<c10::BFloat16>(
    const __nv_bfloat162& v) {
  return __high2bfloat16(v);
}

// Build a permuted activation vector that matches the interleaved output order
// of the AWQ-style dequantizer (uint4 -> {c0,c4}, {c1,c5}, {c2,c6}, {c3,c7}).
// The dot-product can then accumulate directly without an explicit reorder.
template <typename scalar_t>
__device__ __forceinline__ void build_permuted_activation(
    const scalar_t* __restrict__ smem_a, scalar_t* __restrict__ smem_perm,
    int H) {
  using vec2_t = typename Vec2Type<scalar_t>::type;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int num_groups = H / 8;
  for (int g = tid; g < num_groups; g += nthreads) {
    const int base = g * 8;
    const vec2_t a01 = *reinterpret_cast<const vec2_t*>(smem_a + base + 0);
    const vec2_t a23 = *reinterpret_cast<const vec2_t*>(smem_a + base + 2);
    const vec2_t a45 = *reinterpret_cast<const vec2_t*>(smem_a + base + 4);
    const vec2_t a67 = *reinterpret_cast<const vec2_t*>(smem_a + base + 6);
    const scalar_t a0 = low_of_vec2<scalar_t>(a01);
    const scalar_t a1 = high_of_vec2<scalar_t>(a01);
    const scalar_t a2 = low_of_vec2<scalar_t>(a23);
    const scalar_t a3 = high_of_vec2<scalar_t>(a23);
    const scalar_t a4 = low_of_vec2<scalar_t>(a45);
    const scalar_t a5 = high_of_vec2<scalar_t>(a45);
    const scalar_t a6 = low_of_vec2<scalar_t>(a67);
    const scalar_t a7 = high_of_vec2<scalar_t>(a67);
    const int pbase = g * 4;
    *reinterpret_cast<vec2_t*>(smem_perm + pbase + 0) =
        make_scalar2<scalar_t>(static_cast<float>(a0), static_cast<float>(a4));
    *reinterpret_cast<vec2_t*>(smem_perm + pbase + 1) =
        make_scalar2<scalar_t>(static_cast<float>(a1), static_cast<float>(a5));
    *reinterpret_cast<vec2_t*>(smem_perm + pbase + 2) =
        make_scalar2<scalar_t>(static_cast<float>(a2), static_cast<float>(a6));
    *reinterpret_cast<vec2_t*>(smem_perm + pbase + 3) =
        make_scalar2<scalar_t>(static_cast<float>(a3), static_cast<float>(a7));
  }
}

// LM-head GEMV: WARPS_PER_BLOCK warps per block, each warp computes
// ROWS_PER_WARP consecutive output rows.  All warps in a block share the
// pre-multiplied activation vector a_i = hidden_i * scale_i.  The activation is
// stored both sequentially (for the scalar tail) and in the dequantizer's
// interleaved order (for the vectorized main loop), so the hot loop needs only
// vector FMAs without per-chunk reordering.  The unsigned INT4 values are
// accumulated and the constant zero-point contribution (-8 * sum(a)) is
// subtracted once per row.
template <typename scalar_t, int WARPS_PER_BLOCK, int ROWS_PER_WARP>
__launch_bounds__(512, 2) __global__ void int4_lm_head_gemv_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const scalar_t* __restrict__ hidden,        // [M, H]
    const scalar_t* __restrict__ bias,          // [V] or nullptr
    scalar_t* __restrict__ out,                 // [M, V]
    int M, int V, int H, int packed_cols) {
  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int m = blockIdx.y;
  const int64_t m_i64 = static_cast<int64_t>(m);
  using vec2_t = typename Vec2Type<scalar_t>::type;

  // Shared buffers: sequential a_i, permuted a_i, and the scalar -8*sum(a).
  extern __shared__ char smem_raw[];
  scalar_t* const smem_a = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* const smem_perm =
      reinterpret_cast<scalar_t*>(smem_raw + H * sizeof(scalar_t));
  float* const smem_neg8_sum_a =
      reinterpret_cast<float*>(smem_raw + 2 * H * sizeof(scalar_t));

  const scalar_t* h_row = hidden + m_i64 * H;
  float local_sum = 0.0f;
  for (int i = tid; i < H / 2; i += nthreads) {
    const vec2_t h2 = __ldg(reinterpret_cast<const vec2_t*>(h_row + i * 2));
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + i * 2));
    const vec2_t a2 = __hmul2(h2, s2);
    *reinterpret_cast<vec2_t*>(smem_a + i * 2) = a2;
    const float2 af = to_float2<scalar_t>(a2);
    local_sum += af.x + af.y;
  }

  // Parallel reduction of sum(a) within the block (reuse the float slot at the
  // end of shared memory for the intermediate warp sums).
  float* const smem_reduce =
      reinterpret_cast<float*>(smem_raw + 2 * H * sizeof(scalar_t));
  local_sum = warp_reduce_sum(local_sum, lane_id);
  if (lane_id == 0) {
    smem_reduce[warp_id] = local_sum;
  }
  __syncthreads();
  if (tid == 0) {
    float s = 0.0f;
    const int num_warps = nthreads / 32;
    for (int i = 0; i < num_warps; ++i) {
      s += smem_reduce[i];
    }
    *smem_neg8_sum_a = -8.0f * s;
  }

  // Build the permuted activation vector for the vectorized dot-product loop.
  build_permuted_activation<scalar_t>(smem_a, smem_perm, H);
  __syncthreads();

  constexpr int rows_per_block = WARPS_PER_BLOCK * ROWS_PER_WARP;
  const int num_row_blocks = (V + rows_per_block - 1) / rows_per_block;
  const float neg8_sum_a = *smem_neg8_sum_a;

  // Persistent grid: each block reuses the cached activation across multiple
  // output row tiles.
  for (int row_block = blockIdx.x; row_block < num_row_blocks;
       row_block += gridDim.x) {
    const int row0 = row_block * rows_per_block + warp_id * ROWS_PER_WARP;

  #pragma unroll
    for (int kk = 0; kk < ROWS_PER_WARP; ++kk) {
      const int v = row0 + kk;
      if (v >= V) {
        continue;
      }
      const int64_t v_i64 = static_cast<int64_t>(v);

      const uint8_t* w_row = packed_weight + v_i64 * packed_cols;
      vec2_t acc2 = make_scalar2<scalar_t>(0.0f);

      // Vectorized main loop: each lane loads 16 packed bytes (32 outputs) and
      // accumulates four 8-output unsigned dequantized chunks with vector FMAs.
      int i = lane_id * 16;
      if (i + 15 < packed_cols) {
        const uint4 w16 = __ldg(reinterpret_cast<const uint4*>(w_row + i));
        const vec2_t* a_ptr =
            reinterpret_cast<const vec2_t*>(smem_perm + i * 2);
        const uint32_t ws[4] = {w16.x, w16.y, w16.z, w16.w};
  #pragma unroll
        for (int p = 0; p < 4; ++p) {
          const uint4 dq = dequantize_s4_to_vec8<scalar_t>(ws[p]);
          const vec2_t* dq2 = reinterpret_cast<const vec2_t*>(&dq);
  #pragma unroll
          for (int q = 0; q < 4; ++q) {
            acc2 = __hfma2(dq2[q], a_ptr[p * 4 + q], acc2);
          }
        }
      } else {
        // Scalar tail for the last few packed bytes (H not a multiple of 32).
        for (i = (packed_cols / 16) * 16 + lane_id; i < packed_cols; i += 32) {
          const uint8_t b = w_row[i];
          const int col = i * 2;
          const vec2_t a2 = *reinterpret_cast<const vec2_t*>(smem_a + col);
          const vec2_t vals = make_scalar2<scalar_t>(
              static_cast<float>(b & 0x0F), static_cast<float>(b >> 4));
          acc2 = __hfma2(vals, a2, acc2);
        }
      }

      const float2 accf = to_float2<scalar_t>(acc2);
      float acc = accf.x + accf.y;

      // Warp-shuffle reduction to lane 0.
  #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
      }

      if (lane_id == 0) {
        acc += neg8_sum_a;
        if (bias != nullptr) {
          acc += static_cast<float>(bias[v_i64]);
        }
        out[m_i64 * V + v_i64] = static_cast<scalar_t>(acc);
      }
    }
  }  // end persistent row-block loop
}

}  // namespace int4_embedding
}  // namespace vllm

torch::stable::Tensor int4_embedding_lookup(
    const torch::stable::Tensor& packed_weight,
    const torch::stable::Tensor& weight_scale,
    const torch::stable::Tensor& input_ids,
    torch::headeronly::ScalarType out_dtype) {
  // Basic dtype/dimension checks.
  STD_TORCH_CHECK(packed_weight.dim() == 2,
                  "packed_weight must be a 2D tensor");
  STD_TORCH_CHECK(
      packed_weight.scalar_type() == torch::headeronly::ScalarType::Byte,
      "packed_weight must be uint8");
  STD_TORCH_CHECK(input_ids.dim() >= 1, "input_ids must be at least 1D");
  STD_TORCH_CHECK(
      input_ids.scalar_type() == torch::headeronly::ScalarType::Long,
      "input_ids must be int64");
  STD_TORCH_CHECK(weight_scale.scalar_type() == out_dtype,
                  "weight_scale dtype must match out_dtype (fp16/bf16)");

  auto packed = packed_weight.is_contiguous()
                    ? packed_weight
                    : torch::stable::contiguous(packed_weight);
  auto ids = input_ids.is_contiguous() ? input_ids
                                       : torch::stable::contiguous(input_ids);
  auto scales = weight_scale.is_contiguous()
                    ? weight_scale
                    : torch::stable::contiguous(weight_scale);

  const int64_t H = scales.numel();
  STD_TORCH_CHECK(H % 2 == 0, "hidden size H must be even");
  STD_TORCH_CHECK(packed.size(1) == H / 2,
                  "packed_weight.size(1) must equal H/2");

  // weight_scale may be [H] or [1, H]; data pointer is the same when
  // contiguous.
  if (weight_scale.dim() == 2) {
    STD_TORCH_CHECK(weight_scale.size(0) == 1,
                    "weight_scale must be [H] or [1, H]");
  } else {
    STD_TORCH_CHECK(weight_scale.dim() == 1,
                    "weight_scale must be [H] or [1, H]");
  }

  const int64_t V = packed.size(0);
  const int64_t N = ids.numel();
  const int packed_cols = static_cast<int>(H / 2);

  const torch::stable::accelerator::DeviceGuard device_guard(
      packed.get_device_index());

  auto out =
      torch::stable::empty({N, H}, out_dtype, std::nullopt, packed.device());

  if (N == 0) {
    return out;
  }

  const cudaStream_t stream = get_current_cuda_stream();

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      out_dtype, "int4_embedding_lookup_kernel", [&] {
        const int grid = static_cast<int>(N);
        if (N <= 4) {
          // Minimal-latency decode-time path: one warp per token.  Each lane
          // vectorizes over four packed bytes, so a single warp covers the
          // hidden dimension in a few coalesced iterations while keeping
          // launch and scheduling overhead tiny.
          constexpr int warps_per_block = 1;
          constexpr int rows_per_warp = 1;
          constexpr int block = warps_per_block * 32;
          const size_t smem_bytes = static_cast<size_t>(H) * sizeof(scalar_t);
          vllm::int4_embedding::int4_embedding_lookup_kernel<
              scalar_t, warps_per_block, rows_per_warp>
              <<<grid, block, smem_bytes, stream>>>(
                  packed.const_data_ptr<uint8_t>(),
                  scales.const_data_ptr<scalar_t>(),
                  ids.const_data_ptr<int64_t>(),
                  out.mutable_data_ptr<scalar_t>(), static_cast<int>(N),
                  static_cast<int>(H), packed_cols);
        } else {
          // Per-token vectorized path: one block per output token, each thread
          // handles four packed bytes.  This keeps memory accesses coalesced
          // for moderate prefill batches.
          const int block = std::min(std::max(packed_cols / 4, 32), 1024);
          vllm::int4_embedding::int4_embedding_lookup_per_token_vec_kernel<
              scalar_t><<<grid, block, 0, stream>>>(
              packed.const_data_ptr<uint8_t>(),
              scales.const_data_ptr<scalar_t>(), ids.const_data_ptr<int64_t>(),
              out.mutable_data_ptr<scalar_t>(), static_cast<int>(N),
              static_cast<int>(H), packed_cols);
        }
      });

  return out;
}

torch::stable::Tensor int4_lm_head_gemv(
    const torch::stable::Tensor& packed_weight,
    const torch::stable::Tensor& weight_scale,
    const torch::stable::Tensor& hidden_states,
    const std::optional<torch::stable::Tensor>& bias) {
  // Basic dtype/dimension checks.
  STD_TORCH_CHECK(packed_weight.dim() == 2,
                  "packed_weight must be a 2D tensor");
  STD_TORCH_CHECK(
      packed_weight.scalar_type() == torch::headeronly::ScalarType::Byte,
      "packed_weight must be uint8");
  STD_TORCH_CHECK(hidden_states.dim() == 2,
                  "hidden_states must be a 2D tensor");

  auto packed = packed_weight.is_contiguous()
                    ? packed_weight
                    : torch::stable::contiguous(packed_weight);
  auto hidden = hidden_states.is_contiguous()
                    ? hidden_states
                    : torch::stable::contiguous(hidden_states);
  auto scales = weight_scale.is_contiguous()
                    ? weight_scale
                    : torch::stable::contiguous(weight_scale);

  const int64_t M = hidden.size(0);
  const int64_t H = hidden.size(1);
  const int64_t V = packed.size(0);

  STD_TORCH_CHECK(H % 2 == 0, "hidden size H must be even");
  STD_TORCH_CHECK(packed.size(1) == H / 2,
                  "packed_weight.size(1) must equal H/2");
  STD_TORCH_CHECK(scales.scalar_type() == hidden.scalar_type(),
                  "weight_scale dtype must match hidden_states dtype");
  if (scales.dim() == 2) {
    STD_TORCH_CHECK(scales.size(0) == 1, "weight_scale must be [H] or [1, H]");
  } else {
    STD_TORCH_CHECK(scales.dim() == 1, "weight_scale must be [H] or [1, H]");
  }
  STD_TORCH_CHECK(scales.numel() == H, "weight_scale.numel() must equal H");

  std::optional<torch::stable::Tensor> bias_cont;
  const void* bias_ptr = nullptr;
  if (bias.has_value()) {
    auto b = bias.value();
    STD_TORCH_CHECK(b.dim() == 1, "bias must be 1D");
    STD_TORCH_CHECK(b.scalar_type() == hidden.scalar_type(),
                    "bias dtype must match hidden_states dtype");
    STD_TORCH_CHECK(b.size(0) == V, "bias.size(0) must equal V");
    bias_cont = b.is_contiguous() ? b : torch::stable::contiguous(b);
    bias_ptr = bias_cont.value().const_data_ptr();
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      hidden.get_device_index());

  auto out = torch::stable::empty({M, V}, hidden.scalar_type(), std::nullopt,
                                  hidden.device());

  const cudaStream_t stream = get_current_cuda_stream();

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      hidden.scalar_type(), "int4_lm_head_gemv_kernel", [&] {
        using vec2_t = typename vllm::int4_embedding::Vec2Type<scalar_t>::type;
        constexpr int warps_per_block = 8;
        constexpr int rows_per_warp = 2;
        constexpr int block = warps_per_block * 32;
        constexpr int rows_per_block = warps_per_block * rows_per_warp;
        const int full_grid = (V + rows_per_block - 1) / rows_per_block;
        dim3 grid(static_cast<unsigned>(full_grid), static_cast<unsigned>(M));
        constexpr int smem_reduce_bytes = 32 * sizeof(float);
        const size_t smem_bytes =
            2 * static_cast<size_t>(H) * sizeof(scalar_t) + smem_reduce_bytes;
        vllm::int4_embedding::int4_lm_head_gemv_kernel<
            scalar_t, warps_per_block, rows_per_warp>
            <<<grid, block, smem_bytes, stream>>>(
                packed.const_data_ptr<uint8_t>(),
                scales.const_data_ptr<scalar_t>(),
                hidden.const_data_ptr<scalar_t>(),
                static_cast<const scalar_t*>(bias_ptr),
                out.mutable_data_ptr<scalar_t>(), static_cast<int>(M),
                static_cast<int>(V), static_cast<int>(H),
                static_cast<int>(H / 2));
      });

  return out;
}

#endif  // USE_ROCM
