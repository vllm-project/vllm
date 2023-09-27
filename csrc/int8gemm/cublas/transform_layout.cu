/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// #include "src/fastertransformer/kernels/layout_transformer_int8_kernels.h"
#include "transform_layout.h"
#include <cuda_runtime.h>

// transform row-major to COL32
// input matrix is (m, n) row-major
// output matrix is (m, n) COL32
// n should be a multiple of 32
// grid((n+31)/32, (m+31)/32)
// block(8, 32)
__global__ void rowMajorToCOL32_kernel(char4 *dst, const char4 *src, const int m, const int n) {

  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {

    // COL32_col = n_id >> 5 ; COL32_row = (m_id << 5) + (n_id & 31);
    // COL32_idx = (COL32_col << 5) * m + COL32_row = (n_id & 0xffffffe0)*m +
    // (m_id << 5) + (n_id & 31)
    dst[((n_id & 0xffffffe0) * m + (m_id << 5) + (n_id & 31)) >> 2] =
        __ldg(src + ((m_id * n + n_id) >> 2));
  }
}

__global__ void col32ToRowMajor_kernel(char4 *dst, const char4 *src,
                                       const int m, const int n) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    int idx = m_id * n + n_id;
    dst[(((idx >> 5) % m) * n + (((idx >> 5) / m) << 5) + (idx & 31)) >> 2] =
        __ldg(src + (idx >> 2));
  }
}

void invokeRowMajorToCOL32(int8_t *dst, const int8_t *src, const int m,
                           const int n, cudaStream_t stream) {
  assert(n % 32 == 0);
  rowMajorToCOL32_kernel<<<dim3((n + 31) / 32, (m + 31) / 32), dim3(8, 32), 0,
                           stream>>>((char4 *)dst, (const char4 *)src, m, n);
}

void invokeCOL32ToRowMajor(int8_t *dst, const int8_t *src, const int m,
                           const int n, cudaStream_t stream) {
  assert(n % 32 == 0);
  col32ToRowMajor_kernel<<<dim3((n + 31) / 32, (m + 31) / 32), dim3(8, 32), 0,
                           stream>>>((char4 *)dst, (const char4 *)src, m, n);
}

__global__ void rowMajorToAmpere_kernel(char4 *dst, const char4 *src,
                                        const int m, const int n) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    int new_col = n_id >> 5;
    int row_in_tile = m_id & 31;
    int col_in_tile = n_id & 31;
    int new_row = // CUBLASLT_ORDER_COL32_2R_4R4
        (((m_id >> 5) << 10) +
         //(((row%8)/2*4+row/8)*2+row%2)*32+col
         (((((((row_in_tile & 7) >> 1) << 2) + (row_in_tile >> 3)) << 1) +
           (row_in_tile & 1))
          << 5) +
         col_in_tile);
    int idx = m_id * n + n_id;
    dst[(new_col * (m << 5) + new_row) >> 2] = __ldg(src + (idx >> 2));
  }
}

void invokeRowMajorToAmpere(int8_t *dst, const int8_t *src, const int m,
                            const int n, cudaStream_t stream) {
  assert(n % 32 == 0);
  rowMajorToAmpere_kernel<<<dim3((n + 31) / 32, (m + 31) / 32), dim3(8, 32), 0,
                            stream>>>((char4 *)dst, (const char4 *)src, m, n);
}

__global__ void rowMajorToTuring_kernel(char4 *dst, const char4 *src,
                                        const int m, const int n) {
  int n_id = (blockIdx.x * blockDim.x + threadIdx.x) << 2;
  int m_id = blockIdx.y * blockDim.y + threadIdx.y;

  bool check = ((m_id < m) && (n_id < n));
  if (check) {
    int new_col = n_id >> 5;
    int new_row = // CUBLASLT_ORDER_COL4_4R2_8C
                  ////m_id/8 is the number of tile of (8 rows 32 columns) --
                  /// column-major /m_id%2 is even row, otherwise odd row
                  ////n_id%COL32_/8 is the number tile of (8 rows 8 columns)
        (((((m_id >> 3) << 3) + ((m_id & 1) << 2) + ((n_id & 31) >> 3)) << 5) +
         ////n_id%8 >= 4 is the right half of (8 rows 8 columns) tile
         ////(m_id%8/2) is (the row id of alternating 4 rows) - 1
         (((((n_id & 7) >= 4) ? 4 : 0) + ((m_id & 7) >> 1)) << 2) +
         ////n_id%4 is the id of 4 cols
         (n_id & 3));
    int idx = m_id * n + n_id;
    dst[(new_col * (m << 5) + new_row) >> 2] = __ldg(src + (idx >> 2));
  }
}

void invokeRowMajorToTuring(int8_t *dst, const int8_t *src, const int m,
                            const int n, cudaStream_t stream) {
  assert(n % 32 == 0);
  rowMajorToTuring_kernel<<<dim3((n + 31) / 32, (m + 31) / 32), dim3(8, 32), 0,
                            stream>>>((char4 *)dst, (const char4 *)src, m, n);
}
