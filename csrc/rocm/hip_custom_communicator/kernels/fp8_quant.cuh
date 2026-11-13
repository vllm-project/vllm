#pragma once

// Per-row dynamic FP8 (OCP e4m3fn) quantization of the rows a block holds, as
// the epilogue of AttnRes. The arithmetic is AITER's
// dynamic_per_token_scaled_quant (data_to_per_row_scale, scaled_quant_vgpr_impl
// and fp32_to_fp8_scaled_x2): scale = amax * (1 / 448), code =
// cvt_pk_fp8(med3(x * rcp(scale), -448, 448)), so the codes and scales equal
// the separate quant of the bf16 output. The amax is floored at 1e-10 as vLLM's
// Triton attn_res(quant_dtype=...) does: an all-zero row gives zero codes, not
// 0 * rcp(0).

#include "common.cuh"

namespace hip_custom_communicator {

constexpr float kFp8E4m3Max = 448.0f;
constexpr float kFp8AmaxFloor = 1e-10f;

// values[m][k][j] is row offs_m + m, column (offs_n / P + t + k * THREADS) * P
// + j of thread t (the tile layout of load and rms_normalize). Codes go to
// quant_ptr (one byte each, row stride quant_stride_m), the scale of each row
// to scale_ptr[row * scale_stride_m]. Every thread of the block calls it.
template <int THREADS_PER_BLOCK, int WAVE_SIZE, Layout LAYOUT, typename DTYPE,
          int ROWS, int K, int P>
DINLINE void quantize_rows_fp8(const DTYPE (&values)[ROWS][K][P],
                               uint8_t* quant_ptr, int64_t quant_stride_m,
                               float* scale_ptr, int64_t scale_stride_m,
                               int size_m, int size_n, int offs_m, int offs_n) {
  static_assert(LAYOUT == Layout::tile, "the FP8 copy is written from tiles");
  static_assert(P == 2 || P % 4 == 0, "a pack is a code pair or 32-bit words");
  static_assert(WAVE_SIZE == 64, "the cross-lane max is for 64-lane waves");
  constexpr int kWaves = (THREADS_PER_BLOCK + WAVE_SIZE - 1) / WAVE_SIZE;
  const int t = static_cast<int>(threadIdx.x) % THREADS_PER_BLOCK;
  const auto pack_at = [&](int k) {
    return offs_n / P + t + k * THREADS_PER_BLOCK;
  };

  float amax[ROWS];
#pragma unroll
  for (int m = 0; m < ROWS; ++m) {
    amax[m] = 0.0f;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      // Packs past the row are loaded clamped: select, never reduce them.
      const bool in = pack_at(k) < size_n / P;
#pragma unroll
      for (int j = 0; j < P; ++j)
        amax[m] = fmaxf(amax[m],
                        in ? fabsf(static_cast<float>(values[m][k][j])) : 0.0f);
    }
#pragma unroll
    for (int offset = WAVE_SIZE / 2; offset > 0; offset /= 2)
      amax[m] = fmaxf(amax[m], __shfl_xor(amax[m], offset, WAVE_SIZE));
  }

  __shared__ float wave_amax[ROWS][kWaves];
  const int lane = static_cast<int>(threadIdx.x) % WAVE_SIZE;
  const int wave = static_cast<int>(threadIdx.x) / WAVE_SIZE;
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < ROWS; ++m) wave_amax[m][wave] = amax[m];
  }
  __syncthreads();
  float inv_scale[ROWS];
#pragma unroll
  for (int m = 0; m < ROWS; ++m) {
    float row_amax = wave_amax[m][0];
#pragma unroll
    for (int w = 1; w < kWaves; ++w) row_amax = fmaxf(row_amax, wave_amax[m][w]);
    const float scale = fmaxf(row_amax, kFp8AmaxFloor) * (1.0f / kFp8E4m3Max);
    inv_scale[m] = __builtin_amdgcn_rcpf(scale);
    const int row = offs_m + m;
    if (t == 0 && row < size_m) scale_ptr[row * scale_stride_m] = scale;
  }
  // wave_amax is read: the next call may write it.
  __syncthreads();

#pragma unroll
  for (int m = 0; m < ROWS; ++m) {
    const int row = offs_m + m;
    if (row >= size_m) continue;
#pragma unroll
    for (int k = 0; k < K; ++k) {
      const int n = pack_at(k);
      if (n >= size_n / P) continue;
      float x[P];
#pragma unroll
      for (int j = 0; j < P; ++j)
        x[j] = __builtin_amdgcn_fmed3f(
            static_cast<float>(values[m][k][j]) * inv_scale[m], -kFp8E4m3Max,
            kFp8E4m3Max);
      uint8_t* const at = quant_ptr + row * quant_stride_m + int64_t{n} * P;
      if constexpr (P == 2) {
        const int w = __builtin_amdgcn_cvt_pk_fp8_f32(x[0], x[1], 0, false);
        *reinterpret_cast<uint16_t*>(at) = static_cast<uint16_t>(w);
      } else {
        uint32_t words[P / 4];
#pragma unroll
        for (int q = 0; q < P / 4; ++q) {
          int w = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * q], x[4 * q + 1], 0,
                                                  false);
          w = __builtin_amdgcn_cvt_pk_fp8_f32(x[4 * q + 2], x[4 * q + 3], w,
                                              true);
          words[q] = static_cast<uint32_t>(w);
        }
        if constexpr (P == 4)
          *reinterpret_cast<uint32_t*>(at) = words[0];
        else if constexpr (P == 8)
          *reinterpret_cast<uint2*>(at) = make_uint2(words[0], words[1]);
        else
#pragma unroll
          for (int q = 0; q < P / 4; ++q)
            reinterpret_cast<uint32_t*>(at)[q] = words[q];
      }
    }
  }
}

}  // namespace hip_custom_communicator
