/*
 * Fused INT4 per-channel embedding lookup / lm-head GEMV for CUDA.
 *
 * Weights are packed two signed-int4 values per uint8 byte (low nibble = column
 * 2*j, high nibble = column 2*j+1).  The packed unsigned value [0, 15] is
 * mapped to the signed range [-8, 7] by subtracting 8, then multiplied by a
 * per-channel scale.  Both kernels are dtype-dispatched over FP16/BF16.
 *
 * The lookup kernel uses a warp-per-token path for small batches (N <= 4) and
 * a vectorized per-token path for larger prefill batches.  The lm-head GEMV
 * uses one warp per output row.
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

  #include <algorithm>
  #include <cstdint>
  #include <cstring>
  #include <type_traits>

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

// Lookup kernel: one warp per block, each warp computes one output token.
// The per-channel scales are loaded into shared memory once per block.
template <typename scalar_t>
__global__ void int4_embedding_lookup_kernel(
    const uint8_t* __restrict__ packed_weight,  // [V, H/2]
    const scalar_t* __restrict__ scales,        // [H]
    const int64_t* __restrict__ input_ids,      // [N]
    scalar_t* __restrict__ out,                 // [N, H]
    int N, int H, int packed_cols) {
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  using vec2_t = typename Vec2Type<scalar_t>::type;

  // Cache per-channel scales in shared memory as scalar_t2 vectors.
  extern __shared__ char smem_scale_raw[];
  scalar_t* const smem_scale = reinterpret_cast<scalar_t*>(smem_scale_raw);
  for (int i = tid; i < H / 2; i += nthreads) {
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + i * 2));
    *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
  }
  __syncthreads();

  const int n = blockIdx.x;
  if (n >= N) {
    return;
  }

  const int id = static_cast<int>(input_ids[n]);
  const uint8_t* w_row = packed_weight + static_cast<int64_t>(id) * packed_cols;
  scalar_t* o_row = out + static_cast<int64_t>(n) * H;

  const vec2_t eight2 = make_scalar2<scalar_t>(8.0f);

  // Vectorized main loop: each thread loads four packed bytes (8 output
  // values).
  int i = tid * 4;
  #pragma unroll 4
  for (; i + 3 < packed_cols; i += nthreads * 4) {
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
  for (i = (packed_cols / 4) * 4 + tid; i < packed_cols; i += nthreads) {
    const uint8_t b = w_row[i];
    int8_t v0, v1;
    unpack_byte(b, &v0, &v1);
    const int col = i * 2;
    o_row[col] = static_cast<scalar_t>(static_cast<float>(v0) *
                                       static_cast<float>(smem_scale[col]));
    o_row[col + 1] = static_cast<scalar_t>(
        static_cast<float>(v1) * static_cast<float>(smem_scale[col + 1]));
  }
}

// Per-token vectorized embedding lookup.  One block handles one output token;
// each thread vectorizes over four packed bytes (eight output values).  Scales
// are read via __ldg, avoiding all shared-memory setup cost for moderate
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

// LM-head GEMV: WARPS_PER_BLOCK warps per block, each warp computes
// ROWS_PER_WARP consecutive output rows.  All warps in a block share the
// per-channel scales and the current hidden row in shared memory, so each row
// only reads its INT4 weight row from global memory.  The signed INT4 weight is
// built as (nibble - 8) * scale in the target dtype, then the dot product with
// the hidden row is accumulated in FP32 to match PyTorch matmul precision.
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

  // Shared buffers: the current hidden row and the per-channel scales.
  extern __shared__ char smem_raw[];
  scalar_t* const smem_hidden = reinterpret_cast<scalar_t*>(smem_raw);
  scalar_t* const smem_scale =
      reinterpret_cast<scalar_t*>(smem_raw + H * sizeof(scalar_t));

  const scalar_t* h_row = hidden + m_i64 * H;
  for (int i = tid; i < H / 2; i += nthreads) {
    const vec2_t h2 = __ldg(reinterpret_cast<const vec2_t*>(h_row + i * 2));
    const vec2_t s2 = __ldg(reinterpret_cast<const vec2_t*>(scales + i * 2));
    *reinterpret_cast<vec2_t*>(smem_hidden + i * 2) = h2;
    *reinterpret_cast<vec2_t*>(smem_scale + i * 2) = s2;
  }
  __syncthreads();

  constexpr int rows_per_block = WARPS_PER_BLOCK * ROWS_PER_WARP;
  const int num_row_blocks = (V + rows_per_block - 1) / rows_per_block;

  // Persistent grid: each block reuses the cached hidden row and scales across
  // multiple output row tiles.
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
      float2 acc2 = make_float2(0.0f, 0.0f);

      // Vectorized main loop: each lane loads four packed bytes (eight output
      // values), dequantizes them into sequential signed values, builds the
      // weight as signed_nibble * scale, and accumulates hidden * weight in
      // FP32.
      int i = lane_id * 4;
      for (; i + 3 < packed_cols; i += 32 * 4) {
        const uint32_t w4 = *reinterpret_cast<const uint32_t*>(w_row + i);
        vec2_t dq[4];
        dequant_reorder8_signed<scalar_t>(w4, dq);
  #pragma unroll
        for (int p = 0; p < 4; ++p) {
          const int col = i * 2 + p * 2;
          const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + col);
          const vec2_t h2 = *reinterpret_cast<const vec2_t*>(smem_hidden + col);
          const vec2_t w2 = __hmul2(dq[p], s2);
          const float2 hf = to_float2<scalar_t>(h2);
          const float2 wf = to_float2<scalar_t>(w2);
          acc2.x += hf.x * wf.x;
          acc2.y += hf.y * wf.y;
        }
      }

      // Scalar tail for any leftover packed bytes (fewer than four).
      for (i = (packed_cols & ~3) + lane_id; i < packed_cols; i += 32) {
        const uint8_t b = w_row[i];
        const int col = i * 2;
        const vec2_t s2 = *reinterpret_cast<const vec2_t*>(smem_scale + col);
        const vec2_t h2 = *reinterpret_cast<const vec2_t*>(smem_hidden + col);
        const vec2_t vals = make_scalar2<scalar_t>(
            static_cast<float>(static_cast<int8_t>(b & 0x0F) - 8),
            static_cast<float>(static_cast<int8_t>(b >> 4) - 8));
        const vec2_t w2 = __hmul2(vals, s2);
        const float2 hf = to_float2<scalar_t>(h2);
        const float2 wf = to_float2<scalar_t>(w2);
        acc2.x += hf.x * wf.x;
        acc2.y += hf.y * wf.y;
      }

      float acc = acc2.x + acc2.y;

      // Warp-shuffle reduction to lane 0.
  #pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        acc += __shfl_down_sync(0xFFFFFFFF, acc, offset);
      }

      if (lane_id == 0) {
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
          // Minimal-latency decode-time path: one warp per token.
          constexpr int warps_per_block = 1;
          constexpr int rows_per_warp = 1;
          constexpr int block = warps_per_block * 32;
          const size_t smem_bytes = static_cast<size_t>(H) * sizeof(scalar_t);
          int max_smem = 0;
          cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock,
                                 packed.get_device_index());
          STD_TORCH_CHECK(
              smem_bytes <= static_cast<size_t>(max_smem),
              "int4_embedding_lookup requires too much shared memory for H=",
              H);
          vllm::int4_embedding::int4_embedding_lookup_kernel<scalar_t>
              <<<grid, block, smem_bytes, stream>>>(
                  packed.const_data_ptr<uint8_t>(),
                  scales.const_data_ptr<scalar_t>(),
                  ids.const_data_ptr<int64_t>(),
                  out.mutable_data_ptr<scalar_t>(), static_cast<int>(N),
                  static_cast<int>(H), packed_cols);
        } else {
          // Per-token vectorized path: one block per output token.
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
        const size_t smem_bytes = static_cast<size_t>(H) * sizeof(scalar_t) * 2;

        int max_smem = 0;
        cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlock,
                               hidden.get_device_index());
        STD_TORCH_CHECK(
            smem_bytes <= static_cast<size_t>(max_smem),
            "int4_lm_head_gemv requires too much shared memory for H=", H,
            " (need ", smem_bytes, " bytes, device limit ", max_smem,
            " bytes)");

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
