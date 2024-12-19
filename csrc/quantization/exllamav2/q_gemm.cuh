/*
 * Adapted from https://github.com/turboderp/exllamav2
 * Copyright (c) 2024 turboderp
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */
#include "compat.cuh"

namespace vllm {
namespace exl2 {

#define MAX_Q_GEMM_WEIGHTS 4
#define EXL2_BLOCK_KN_SIZE 64
#define EXL2_BLOCK_M_SIZE_MAX 8
#define EXL2_MAX_GROUPS_IN_BLOCK (EXL2_BLOCK_KN_SIZE / 32)

__forceinline__ __device__ half2 dot22_8(half2 (&dq)[4], const half* a_ptr,
                                         const half2 g_result,
                                         const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_16(half2 (&dq)[8], const half* a_ptr,
                                          const half2 g_result,
                                          const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ half2 dot22_32(half2 (&dq)[16], const half* a_ptr,
                                          const half2 g_result,
                                          const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hfma2(result, __halves2half2(qs_h, qs_h), g_result);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr,
                                           const float g_result,
                                           const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_16_f(half2 (&dq)[8], const half* a_ptr,
                                            const float g_result,
                                            const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ float dot22_32_f(half2 (&dq)[16], const half* a_ptr,
                                            const float g_result,
                                            const float qs_f) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
  float result_f =
      __half2float(__low2half(result)) + __half2float(__high2half(result));
  return fma(result_f, qs_f, g_result);
}

__forceinline__ __device__ half dot22_8_h(half2 (&dq)[4], const half* a_ptr,
                                          const half g_result,
                                          const half qs_h) {
  // Use FP32 accumulator to avoid potential overflow since unscaled weights are
  // in the range -128..127

  float result = {};
#pragma unroll
  for (int i = 0; i < 4; i++) {
    half2 w01 = dq[i];
    float w0 = __low2float(w01);
    float w1 = __high2float(w01);
    float x0 = __half2float(*a_ptr++);
    float x1 = __half2float(*a_ptr++);
    result = fma(w0, x0, result);
    result = fma(w1, x1, result);
  }
  float qs = __half2float(qs_h);
  result *= qs;
  half result_h = __float2half_rn(result);
  return __hadd(result_h, g_result);
}

__forceinline__ __device__ half dot22_16_h(half2 (&dq)[8], const half* a_ptr,
                                           const half g_result,
                                           const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 8; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  half result_h = __hadd(__low2half(result), __high2half(result));
  return __hfma(result_h, qs_h, g_result);
}

__forceinline__ __device__ half dot22_32_h(half2 (&dq)[16], const half* a_ptr,
                                           const half g_result,
                                           const half qs_h) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 16; i += 1) result = __hfma2(dq[i], *a2_ptr++, result);
  half result_h = __hadd(__low2half(result), __high2half(result));
  return __hfma(result_h, qs_h, g_result);
}

typedef void (*fp_gemm_half_q_half_kernel)(
    const half*, const uint32_t*, const uint32_t*, const half*, half*,
    const int, const int, const int, const int, const int, const uint16_t*,
    const uint16_t*, const int, const int, const int, const int, const int,
    const int, const bool);

template <int m_count>
__global__ void gemm_half_q_half_kernel(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_q_scale,
    const half* __restrict__ b_q_scale_max, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int height,
    const int groups, const uint16_t* __restrict__ b_q_group_map,
    const uint16_t* __restrict__ b_q_perm, const int rows_8, const int rows_6,
    const int rows_5, const int rows_4, const int rows_3, const int rows_2,
    const bool clear) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q4_row b_q_scale_(b_q_scale, groups, size_n);

  int t = threadIdx.x;

  // Block

  int offset_n = blockIdx.x * EXL2_BLOCK_KN_SIZE * 4;
  int offset_m = blockIdx.y * m_count;
  int offset_k = blockIdx.z * EXL2_BLOCK_KN_SIZE;

  int end_n = min(offset_n + EXL2_BLOCK_KN_SIZE * 4, size_n);
  int end_m = min(offset_m + m_count, size_m);
  int end_k = min(offset_k + EXL2_BLOCK_KN_SIZE, height);
  int n = offset_n + t * 4;

  // Read weights

  half_uint16 weights[MAX_Q_GEMM_WEIGHTS];

  // Preload block_a

  __shared__ half block_a[m_count][EXL2_BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];
      half a0 = a_ptr[b_q_perm[offset_k + t]];
      //            half a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Clear

  if (n >= size_n) return;

  if (clear && blockIdx.z == 0)  // && (threadIdx.x & 1) == 0)
  {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group

  // int group = offset_k / groupsize;
  int group = b_q_group_map[offset_k * 2];

  //    if (offset_m == 0 && t == 0)
  //        DBGI2(offset_k, group);

  // Preload scales

  half scales[EXL2_MAX_GROUPS_IN_BLOCK][4];

  // int groups_in_block = DIVIDE((end_k - offset_k), groupsize);
  int temp_k = offset_k;
  for (int g = 0; temp_k < end_k; g++) {
    int qscales[4];
    b_q_scale_.item4(qscales, group + g, n);
    qscales[0]++;
    qscales[1]++;
    qscales[2]++;
    qscales[3]++;
    half maxscale = b_q_scale_max[group + g];
    scales[g][0] = __hmul(__int2half_rn(qscales[0] * qscales[0]), maxscale);
    scales[g][1] = __hmul(__int2half_rn(qscales[1] * qscales[1]), maxscale);
    scales[g][2] = __hmul(__int2half_rn(qscales[2] * qscales[2]), maxscale);
    scales[g][3] = __hmul(__int2half_rn(qscales[3] * qscales[3]), maxscale);
    temp_k += b_q_group_map[temp_k * 2 + 1];
  }

  // a, b offset

  int pre_rows_8 = min(rows_8, offset_k);
  int pre_rows_6 = offset_k > rows_8 ? min(rows_6, offset_k) - rows_8 : 0;
  int pre_rows_5 = offset_k > rows_6 ? min(rows_5, offset_k) - rows_6 : 0;
  int pre_rows_4 = offset_k > rows_5 ? min(rows_4, offset_k) - rows_5 : 0;
  int pre_rows_3 = offset_k > rows_4 ? min(rows_3, offset_k) - rows_4 : 0;
  int pre_rows_2 = offset_k > rows_3 ? min(rows_2, offset_k) - rows_3 : 0;
  int qk = 0;
  qk += pre_rows_8 / 32 * 8;
  qk += pre_rows_6 / 32 * 6;
  qk += pre_rows_5 / 32 * 5;
  qk += pre_rows_4 / 32 * 4;
  qk += pre_rows_3 / 32 * 3;
  qk += pre_rows_2 / 32 * 2;

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = EXL2_BLOCK_KN_SIZE;

  // Initial group

  int scales_idx = 0;
  half qs_h0 = scales[scales_idx][0];
  half qs_h1 = scales[scales_idx][1];
  half qs_h2 = scales[scales_idx][2];
  half qs_h3 = scales[scales_idx][3];
  int nextgroup = offset_k + b_q_group_map[offset_k * 2 + 1];

  // Column result

  half block_c[m_count][4] = {};

  // Dequantize groups

  int k = offset_k;

  while (k < rows_8 && k < end_k) {
    if (k == nextgroup) {
      group++;
      scales_idx++;
      qs_h0 = scales[scales_idx][0];
      qs_h1 = scales[scales_idx][1];
      qs_h2 = scales[scales_idx][2];
      qs_h3 = scales[scales_idx][3];
      nextgroup += b_q_group_map[k * 2 + 1];
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      int4 load_int4[2];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][4];
      dequant_8bit_8(load_int4[0].x, load_int4[1].x, dq[0], size_n);
      dequant_8bit_8(load_int4[0].y, load_int4[1].y, dq[1], size_n);
      dequant_8bit_8(load_int4[0].z, load_int4[1].z, dq[2], size_n);
      dequant_8bit_8(load_int4[0].w, load_int4[1].w, dq[3], size_n);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_8_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0);
        block_c[m][1] =
            dot22_8_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1);
        block_c[m][2] =
            dot22_8_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2);
        block_c[m][3] =
            dot22_8_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3);
      }
      a_ptr += 8;
    }
    k += 32;
  }

  while (k < rows_6 && k < end_k) {
    if (k == nextgroup) {
      group++;
      scales_idx++;
      qs_h0 = scales[scales_idx][0];
      qs_h1 = scales[scales_idx][1];
      qs_h2 = scales[scales_idx][2];
      qs_h3 = scales[scales_idx][3];
      nextgroup += b_q_group_map[k * 2 + 1];
    }

#pragma unroll
    for (int j = 0; j < 2; j++) {
      int4 load_int4[3];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][8];
      dequant_6bit_16(load_int4[0].x, load_int4[1].x, load_int4[2].x, dq[0],
                      size_n);
      dequant_6bit_16(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1],
                      size_n);
      dequant_6bit_16(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2],
                      size_n);
      dequant_6bit_16(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3],
                      size_n);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_16_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0);
        block_c[m][1] =
            dot22_16_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1);
        block_c[m][2] =
            dot22_16_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2);
        block_c[m][3] =
            dot22_16_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3);
      }
      a_ptr += 16;
    }
    k += 32;
  }

  while (k < rows_5 && k < end_k) {
    if (k == nextgroup) {
      group++;
      scales_idx++;
      qs_h0 = scales[scales_idx][0];
      qs_h1 = scales[scales_idx][1];
      qs_h2 = scales[scales_idx][2];
      qs_h3 = scales[scales_idx][3];
      nextgroup += b_q_group_map[k * 2 + 1];
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      int4 load_int4[5];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[3] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[4] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][16];
      dequant_5bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x,
                      load_int4[3].x, load_int4[4].x, dq[0], size_n);
      dequant_5bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y,
                      load_int4[3].y, load_int4[4].y, dq[1], size_n);
      dequant_5bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z,
                      load_int4[3].z, load_int4[4].z, dq[2], size_n);
      dequant_5bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w,
                      load_int4[3].w, load_int4[4].w, dq[3], size_n);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_32_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0);
        block_c[m][1] =
            dot22_32_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1);
        block_c[m][2] =
            dot22_32_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2);
        block_c[m][3] =
            dot22_32_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3);
      }
      a_ptr += 32;
    }

    k += 32;
  }

  while (k < rows_4 && k < end_k) {
    if (k == nextgroup) {
      group++;
      scales_idx++;
      qs_h0 = scales[scales_idx][0];
      qs_h1 = scales[scales_idx][1];
      qs_h2 = scales[scales_idx][2];
      qs_h3 = scales[scales_idx][3];
      nextgroup += b_q_group_map[k * 2 + 1];
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      int4 load_int4[1];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][4];
      dequant_4bit_8(load_int4[0].x, dq[0], size_n);
      dequant_4bit_8(load_int4[0].y, dq[1], size_n);
      dequant_4bit_8(load_int4[0].z, dq[2], size_n);
      dequant_4bit_8(load_int4[0].w, dq[3], size_n);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_8_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0);
        block_c[m][1] =
            dot22_8_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1);
        block_c[m][2] =
            dot22_8_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2);
        block_c[m][3] =
            dot22_8_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3);
      }
      a_ptr += 8;
    }
    k += 32;
  }

  while (k < rows_3 && k < end_k) {
    if (k == nextgroup) {
      group++;
      scales_idx++;
      qs_h0 = scales[scales_idx][0];
      qs_h1 = scales[scales_idx][1];
      qs_h2 = scales[scales_idx][2];
      qs_h3 = scales[scales_idx][3];
      nextgroup += b_q_group_map[k * 2 + 1];
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      int4 load_int4[3];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][16];
      dequant_3bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x, dq[0],
                      size_n);
      dequant_3bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1],
                      size_n);
      dequant_3bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2],
                      size_n);
      dequant_3bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3],
                      size_n);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_32_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0);
        block_c[m][1] =
            dot22_32_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1);
        block_c[m][2] =
            dot22_32_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2);
        block_c[m][3] =
            dot22_32_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3);
      }
      a_ptr += 32;
    }
    k += 32;
  }

  while (k < rows_2 && k < end_k) {
    if (k == nextgroup) {
      group++;
      scales_idx++;
      qs_h0 = scales[scales_idx][0];
      qs_h1 = scales[scales_idx][1];
      qs_h2 = scales[scales_idx][2];
      qs_h3 = scales[scales_idx][3];
      nextgroup += b_q_group_map[k * 2 + 1];
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      int4 load_int4[1];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][8];
      dequant_2bit_16(load_int4[0].x, dq[0], size_n);
      dequant_2bit_16(load_int4[0].y, dq[1], size_n);
      dequant_2bit_16(load_int4[0].z, dq[2], size_n);
      dequant_2bit_16(load_int4[0].w, dq[3], size_n);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_16_h(dq[0], a_ptr + m * a_stride, block_c[m][0], qs_h0);
        block_c[m][1] =
            dot22_16_h(dq[1], a_ptr + m * a_stride, block_c[m][1], qs_h1);
        block_c[m][2] =
            dot22_16_h(dq[2], a_ptr + m * a_stride, block_c[m][2], qs_h2);
        block_c[m][3] =
            dot22_16_h(dq[3], a_ptr + m * a_stride, block_c[m][3], qs_h3);
      }

      a_ptr += 16;
    }
    k += 16;
  }

  // Accumulate column sums in c

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);

    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
    //        *out = result01;
    //        *(out + 1) = result23;
  }
}

struct map_m_count_exl2 {
  static constexpr fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(
      const int m_count) {
#if EXL2_BLOCK_M_SIZE_MAX >= 1
    if (m_count == 1) return gemm_half_q_half_kernel<1>;
#endif
#if EXL2_BLOCK_M_SIZE_MAX >= 2
    if (m_count == 2) return gemm_half_q_half_kernel<2>;
#endif
#if EXL2_BLOCK_M_SIZE_MAX >= 3
    if (m_count == 3) return gemm_half_q_half_kernel<3>;
#endif
#if EXL2_BLOCK_M_SIZE_MAX >= 4
    if (m_count == 4) return gemm_half_q_half_kernel<4>;
#endif
#if EXL2_BLOCK_M_SIZE_MAX >= 5
    if (m_count == 5) return gemm_half_q_half_kernel<5>;
#endif
#if EXL2_BLOCK_M_SIZE_MAX >= 6
    if (m_count == 6) return gemm_half_q_half_kernel<6>;
#endif
#if EXL2_BLOCK_M_SIZE_MAX >= 7
    if (m_count == 7) return gemm_half_q_half_kernel<7>;
#endif
#if EXL2_BLOCK_M_SIZE_MAX >= 8
    if (m_count == 8) return gemm_half_q_half_kernel<8>;
#endif
    return NULL;
  }
};

fp_gemm_half_q_half_kernel pick_gemm_half_q_half_kernel(const int m_count) {
  return map_m_count_exl2::pick_gemm_half_q_half_kernel(m_count);
}

}  // namespace exl2
}  // namespace vllm