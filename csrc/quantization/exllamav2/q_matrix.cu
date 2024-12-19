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
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

#include "q_matrix.cuh"
#include "matrix_view.cuh"

#include "quant/qdq_2.cuh"
#include "quant/qdq_3.cuh"
#include "quant/qdq_4.cuh"
#include "quant/qdq_5.cuh"
#include "quant/qdq_6.cuh"
#include "quant/qdq_8.cuh"

namespace vllm {
namespace exl2 {

#define BLOCK_KN_SIZE 128

#define THREADS_X 32
#define THREADS_Y 32

#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

// Shuffle quantized data on load

__global__ void shuffle_kernel(uint32_t* __restrict__ b_q_weight,
                               const int size_k, const int size_n,
                               const int rows_8, const int rows_6,
                               const int rows_5, const int rows_4,
                               const int rows_3, const int rows_2) {
  int n = blockIdx.x * THREADS_X + threadIdx.x;
  if (n >= size_n) return;
  int k = 0;
  uint32_t* b_ptr = b_q_weight + n;
  while (k < rows_8) {
    shuffle_8bit_4(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 4;
  }
  while (k < rows_6) {
    shuffle_6bit_16(b_ptr, size_n);
    b_ptr += 3 * size_n;
    k += 16;
  }
  while (k < rows_5) {
    shuffle_5bit_32(b_ptr, size_n);
    b_ptr += 5 * size_n;
    k += 32;
  }
  while (k < rows_4) {
    shuffle_4bit_8(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 8;
  }
  while (k < rows_3) {
    shuffle_3bit_32(b_ptr, size_n);
    b_ptr += 3 * size_n;
    k += 32;
  }
  while (k < rows_2) {
    shuffle_2bit_16(b_ptr, size_n);
    b_ptr += 1 * size_n;
    k += 16;
  }
}

// QMatrix constructor

QMatrix::QMatrix(const int _device, const int _height, const int _width,
                 const int _groups,

                 uint32_t* _q_weight, uint16_t* _q_perm, uint16_t* _q_invperm,
                 uint32_t* _q_scale, half* _q_scale_max, uint16_t* _q_groups,
                 uint16_t* _q_group_map)
    : device(_device), height(_height), width(_width), groups(_groups) {
  cudaSetDevice(device);

  failed = false;

  cuda_q_weight = _q_weight;
  cuda_q_perm = _q_perm;
  cuda_q_invperm = _q_invperm;
  cuda_q_scale = _q_scale;
  cuda_q_scale_max = _q_scale_max;
  cuda_q_groups = _q_groups;
  cuda_q_group_map = _q_group_map;

  // Create group map

  rows_8 = 0;
  rows_6 = 0;
  rows_5 = 0;
  rows_4 = 0;
  rows_3 = 0;
  rows_2 = 0;

  {
    uint16_t* cpu_q_groups = (uint16_t*)calloc(groups * 2, sizeof(uint16_t));
    cudaMemcpy(cpu_q_groups, cuda_q_groups, groups * 2 * sizeof(uint16_t),
               cudaMemcpyDeviceToHost);

    int row = 0;
    for (int i = 0; i < groups; i++) {
      int bits = cpu_q_groups[i * 2];

      int rows;
      if (i < groups - 1) {
        int qrows = cpu_q_groups[i * 2 + 3] - cpu_q_groups[i * 2 + 1];
        rows = qrows * 32 / bits;
      } else
        rows = height - row;

      if (bits == 8) rows_8 += rows;
      if (bits == 6) rows_6 += rows;
      if (bits == 5) rows_5 += rows;
      if (bits == 4) rows_4 += rows;
      if (bits == 3) rows_3 += rows;
      if (bits == 2) rows_2 += rows;
      row += rows;
    }

    free(cpu_q_groups);

    rows_6 += rows_8;
    rows_5 += rows_6;
    rows_4 += rows_5;
    rows_3 += rows_4;
    rows_2 += rows_3;
  }

  // Shuffle quantized data

  dim3 blockDim, gridDim;
  blockDim.x = THREADS_X;
  blockDim.y = 1;
  gridDim.x = DIVIDE(width, THREADS_X);
  gridDim.y = 1;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  shuffle_kernel<<<gridDim, blockDim, 0, stream>>>(cuda_q_weight, height, width,
                                                   rows_8, rows_6, rows_5,
                                                   rows_4, rows_3, rows_2);
}

QMatrix::~QMatrix() {}

// Reconstruct b[k,n]

__global__ void reconstruct_kernel(const uint32_t* __restrict__ b_q_weight,
                                   const uint16_t* __restrict__ b_q_perm,
                                   const uint32_t* __restrict__ b_q_scale,
                                   const half* __restrict__ b_q_scale_max,
                                   const uint16_t* __restrict__ b_q_group_map,
                                   const int size_k, const int size_n,
                                   // const int groupsize,
                                   const int groups, half* __restrict__ b,
                                   const int rows_8, const int rows_6,
                                   const int rows_5, const int rows_4,
                                   const int rows_3, const int rows_2) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q4_row b_q_scale_(b_q_scale, groups, size_n);

  int offset_k = BLOCK_KN_SIZE * blockIdx.y;
  int offset_n = BLOCK_KN_SIZE * blockIdx.x;

  // Preload remapping table

  int t = threadIdx.x;
  __shared__ uint16_t perm[BLOCK_KN_SIZE];
  if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];

  // Column

  int n = offset_n + t;
  if (n >= size_n) return;

  // Find initial group

  // int group = offset_k / groupsize;
  int group = b_q_group_map[offset_k * 2];

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

  half qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
  half2 qs_h2 = __halves2half2(qs_h, qs_h);
  int nextgroup = offset_k + b_q_group_map[offset_k * 2 + 1];

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);
  int k = offset_k;
  int lk = 0;

  __syncthreads();

  while (k < rows_8 && k < end_k) {
    if (k == nextgroup) {
      group++;
      qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
      nextgroup += b_q_group_map[k * 2 + 1];
      qs_h2 = __halves2half2(qs_h, qs_h);
    }
    for (int p = 0; p < 4; p++) {
      half2 dq[4];
      uint32_t q_0 = *b_ptr;
      b_ptr += size_n;
      uint32_t q_1 = *b_ptr;
      b_ptr += size_n;
      dequant_8bit_8(q_0, q_1, dq, size_n);
      for (int j = 0; j < 4; j++) dq[j] = __hmul2(dq[j], qs_h2);
      half* dqh = (half*)dq;
      for (int j = 0; j < 8; j++) b_.set(perm[lk++], n, dqh[j]);
    }
    k += 32;
  }

  while (k < rows_6 && k < end_k) {
    if (k == nextgroup) {
      group++;
      qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
      nextgroup += b_q_group_map[k * 2 + 1];
      qs_h2 = __halves2half2(qs_h, qs_h);
    }
    for (int p = 0; p < 2; p++) {
      half2 dq[8];
      uint32_t q_0 = *b_ptr;
      b_ptr += size_n;
      uint32_t q_1 = *b_ptr;
      b_ptr += size_n;
      uint32_t q_2 = *b_ptr;
      b_ptr += size_n;
      dequant_6bit_16(q_0, q_1, q_2, dq, size_n);
      for (int j = 0; j < 8; j++) dq[j] = __hmul2(dq[j], qs_h2);
      half* dqh = (half*)dq;
      for (int j = 0; j < 16; j++) b_.set(perm[lk++], n, dqh[j]);
    }
    k += 32;
  }

  while (k < rows_5 && k < end_k) {
    if (k == nextgroup) {
      group++;
      qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
      nextgroup += b_q_group_map[k * 2 + 1];
      qs_h2 = __halves2half2(qs_h, qs_h);
    }
    for (int p = 0; p < 1; p++) {
      half2 dq[16];
      uint32_t q_0 = *b_ptr;
      b_ptr += size_n;
      uint32_t q_1 = *b_ptr;
      b_ptr += size_n;
      uint32_t q_2 = *b_ptr;
      b_ptr += size_n;
      uint32_t q_3 = *b_ptr;
      b_ptr += size_n;
      uint32_t q_4 = *b_ptr;
      b_ptr += size_n;
      dequant_5bit_32(q_0, q_1, q_2, q_3, q_4, dq, size_n);
      for (int j = 0; j < 16; j++) dq[j] = __hmul2(dq[j], qs_h2);
      half* dqh = (half*)dq;
      for (int j = 0; j < 32; j++) b_.set(perm[lk++], n, dqh[j]);
    }
    k += 32;
  }

  while (k < rows_4 && k < end_k) {
    if (k == nextgroup) {
      group++;
      qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
      nextgroup += b_q_group_map[k * 2 + 1];
      qs_h2 = __halves2half2(qs_h, qs_h);
    }
    for (int p = 0; p < 4; p++) {
      half2 dq[4];
      uint32_t q_0 = *b_ptr;
      b_ptr += size_n;
      dequant_4bit_8(q_0, dq, size_n);
      for (int j = 0; j < 4; j++) dq[j] = __hmul2(dq[j], qs_h2);
      half* dqh = (half*)dq;
      for (int j = 0; j < 8; j++) b_.set(perm[lk++], n, dqh[j]);
    }
    k += 32;
  }

  while (k < rows_3 && k < end_k) {
    if (k == nextgroup) {
      group++;
      qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
      nextgroup += b_q_group_map[k * 2 + 1];
      qs_h2 = __halves2half2(qs_h, qs_h);
    }
    for (int p = 0; p < 1; p++) {
      half2 dq[16];
      uint32_t q_0 = *b_ptr;
      b_ptr += size_n;
      uint32_t q_1 = *b_ptr;
      b_ptr += size_n;
      uint32_t q_2 = *b_ptr;
      b_ptr += size_n;
      dequant_3bit_32(q_0, q_1, q_2, dq, size_n);
      for (int j = 0; j < 16; j++) dq[j] = __hmul2(dq[j], qs_h2);
      half* dqh = (half*)dq;
      for (int j = 0; j < 32; j++) b_.set(perm[lk++], n, dqh[j]);
    }
    k += 32;
  }

  while (k < rows_2 && k < end_k) {
    if (k == nextgroup) {
      group++;
      qs_h = dq_scale(b_q_scale_.item(group, n), b_q_scale_max[group]);
      nextgroup += b_q_group_map[k * 2 + 1];
      qs_h2 = __halves2half2(qs_h, qs_h);
    }
    for (int p = 0; p < 1; p++) {
      half2 dq[8];
      uint32_t q_0 = *b_ptr;
      b_ptr += size_n;
      dequant_2bit_16(q_0, dq, size_n);
      for (int j = 0; j < 8; j++) dq[j] = __hmul2(dq[j], qs_h2);
      half* dqh = (half*)dq;
      for (int j = 0; j < 16; j++) b_.set(perm[lk++], n, dqh[j]);
    }
    k += 16;
  }
}

void QMatrix::reconstruct(half* out) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);

  {
    gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    reconstruct_kernel<<<gridDim, blockDim, 0, stream>>>(
        cuda_q_weight, cuda_q_perm, cuda_q_scale, cuda_q_scale_max,
        cuda_q_group_map, height, width,
        // groupsize,
        groups, out, rows_8, rows_6, rows_5, rows_4, rows_3, rows_2);
  }
}

}  // namespace exl2
}  // namespace vllm