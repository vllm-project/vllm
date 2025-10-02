/*
Adapted from `q_gemm.cu`, which is adapted from
https://github.com/turboderp/exllamav2 and
https://github.com/qwopqwop200/GPTQ-for-LLaMa.

This supports GPTQ v2 format checkpoints (checkpoint_format: 'gptq_v2'),
by removing the v1-specific "zero + 1" logic during dequantization.
Specifically, GPTQ v1 format checkpoints store (zero - 1), and need to + 1 at
runtime during dequantization. GPTQ v2 format checkpoints store the zero point
as is, and doesn't require + 1 at runtime. For more details, please refer to
ModelCloud/GPTQModel:
https://github.com/ModelCloud/GPTQModel/blob/020ac04b74f6263f22491e6a6a034cb4fa5bf181/gptqmodel/utils/model.py#L625
*/

#include <cstdint>
#include <cstdio>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "compat.cuh"
#include "matrix_view.cuh"
#include "qdq_2.cuh"
#include "qdq_3.cuh"
#include "qdq_4.cuh"
#include "qdq_8.cuh"

namespace vllm {
namespace gptq {

#define BLOCK_KN_SIZE 128
#define BLOCK_M_SIZE_MAX 8
#define MAX_GROUPS_IN_BLOCK (BLOCK_KN_SIZE / 32)
#define MAX_Q_GEMM_ROWS 50
#define MAX_Q_GEMM_ROWS_8BIT 24
#define MAX_ALT_GEMM_ROWS 8
#define THREADS_X 32
#define THREADS_Y 32
#define DIVIDE(x, size) (((x) + (size) - 1) / (size))

#if defined(USE_ROCM)
  #include <hipblas/hipblas.h>
__host__ __forceinline__ hipblasStatus_t __compat_hipblasHgemm(
    hipblasHandle_t handle, hipblasOperation_t transA,
    hipblasOperation_t transB, int m, int n, int k, const half* alpha,
    const half* AP, int lda, const half* BP, int ldb, const half* beta,
    half* CP, int ldc) {
  return hipblasHgemm(handle, transA, transB, m, n, k,
                      reinterpret_cast<const hipblasHalf*>(alpha),
                      reinterpret_cast<const hipblasHalf*>(AP), lda,
                      reinterpret_cast<const hipblasHalf*>(BP), ldb,
                      reinterpret_cast<const hipblasHalf*>(beta),
                      reinterpret_cast<hipblasHalf*>(CP), ldc);
}
  #define hipblasHgemm __compat_hipblasHgemm

  // Previous version of PyTorch were converting to rocBLAS instead of hipBLAS.
  #define rocblas_operation_none HIPBLAS_OP_N
  #define rocblas_hgemm __compat_hipblasHgemm
#endif

__forceinline__ __device__ half2 dot22_8(half2 (&dq)[4], const half* a_ptr,
                                         const half2 g_result) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __hadd2(result, g_result);
}

__forceinline__ __device__ float dot22_8_f(half2 (&dq)[4], const half* a_ptr) {
  half2 result = {};
  const half2* a2_ptr = (const half2*)a_ptr;
#pragma unroll
  for (int i = 0; i < 4; i++) result = __hfma2(dq[i], *a2_ptr++, result);
  return __half2float(__low2half(result)) + __half2float(__high2half(result));
}

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

typedef void (*fp_gemm_half_q_half_gptq_kernel)(const half*, const uint32_t*,
                                                const uint32_t*, const half*,
                                                half*, const int, const int,
                                                const int, const int,
                                                const int*);

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_4bit_kernel_v2(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 4);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  float scales[4];
  half2 z1z16[4][2];
  half2 y1y16[4][2];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_f(scales, group, n);
  dequant_4bit_8_prep_zero(zeros[0], z1z16[0], y1y16[0]);
  dequant_4bit_8_prep_zero(zeros[1], z1z16[1], y1y16[1]);
  dequant_4bit_8_prep_zero(zeros[2], z1z16[2], y1y16[2]);
  dequant_4bit_8_prep_zero(zeros[3], z1z16[3], y1y16[3]);

  // Column result
  float block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_f(scales, group, n);
      dequant_4bit_8_prep_zero(zeros[0], z1z16[0], y1y16[0]);
      dequant_4bit_8_prep_zero(zeros[1], z1z16[1], y1y16[1]);
      dequant_4bit_8_prep_zero(zeros[2], z1z16[2], y1y16[2]);
      dequant_4bit_8_prep_zero(zeros[3], z1z16[3], y1y16[3]);
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][4];
      dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n,
                          false);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] = fma(dot22_8_f(dq[0], a_ptr + m * a_stride), scales[0],
                            block_c[m][0]);
        block_c[m][1] = fma(dot22_8_f(dq[1], a_ptr + m * a_stride), scales[1],
                            block_c[m][1]);
        block_c[m][2] = fma(dot22_8_f(dq[2], a_ptr + m * a_stride), scales[2],
                            block_c[m][2]);
        block_c[m][3] = fma(dot22_8_f(dq[3], a_ptr + m * a_stride), scales[3],
                            block_c[m][3]);
      }

      b_ptr += size_n;
      a_ptr += 8;
    }

    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(__float2half_rn(block_c[m][0]),
                                    __float2half_rn(block_c[m][1]));
    half2 result23 = __halves2half2(__float2half_rn(block_c[m][2]),
                                    __float2half_rn(block_c[m][3]));
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_2bit_kernel_v2(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q2_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 2);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 1; j++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][8];
      dequant_2bit_16(load_int4.x, dq[0], size_n, zeros[0]);
      dequant_2bit_16(load_int4.y, dq[1], size_n, zeros[1]);
      dequant_2bit_16(load_int4.z, dq[2], size_n, zeros[2]);
      dequant_2bit_16(load_int4.w, dq[3], size_n, zeros[3]);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_16_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_16_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_16_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_16_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }

      b_ptr += size_n;
      a_ptr += 16;
    }

    k += 16;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_3bit_kernel_v2(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q3_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / 32 * 3;

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
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
                      size_n, zeros[0]);
      dequant_3bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1],
                      size_n, zeros[1]);
      dequant_3bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2],
                      size_n, zeros[2]);
      dequant_3bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3],
                      size_n, zeros[3]);

#pragma unroll
      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_32_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_32_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_32_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_32_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }
      a_ptr += 32;
    }

    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

template <bool first_block, int m_count>
__global__ void gemm_half_q_half_gptq_8bit_kernel_v2(
    const half* __restrict__ a, const uint32_t* __restrict__ b_q_weight,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, half* __restrict__ c,
    const int size_m, const int size_n, const int size_k, const int groups,
    const int* __restrict__ b_q_perm) {
  MatrixView_half a_(a, size_m, size_k);
  MatrixView_half_rw c_(c, size_m, size_n);
  MatrixView_q8_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto t = threadIdx.x;

  // Block
  auto offset_n = blockIdx.x * BLOCK_KN_SIZE * 4;
  auto offset_m = blockIdx.y * m_count;
  auto offset_k = blockIdx.z * BLOCK_KN_SIZE;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  int n = offset_n + t * 4;

  // Preload block_a
  __shared__ half block_a[m_count][BLOCK_KN_SIZE];

  if (offset_k + t < end_k) {
    for (int m = 0; m < m_count; ++m) {
      const half* a_ptr = a_.item_ptr(offset_m + m, 0);
      half* block_a_ptr = block_a[m];

      half a0;
      if (b_q_perm)
        a0 = a_ptr[b_q_perm[offset_k + t]];
      else
        a0 = a_ptr[offset_k + t];
      block_a_ptr[t] = a0;
    }
  }

  // Zero output
  if (n >= size_n) return;

  if (blockIdx.z == 0) {
    for (int m = 0; m < m_count; m++)
      *((uint64_t*)c_.item_ptr(offset_m + m, n)) = 0;
  }

  __syncthreads();

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // a, b offset
  int qk = offset_k / (32 / 8);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;
  const half* a_ptr = &block_a[0][0];
  int a_stride = BLOCK_KN_SIZE;

  // Initial group
  int zeros[4];
  half scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4(scales, group, n);
  // Column result
  half block_c[m_count][4] = {};

  // Dequantize and multiply
  int k = offset_k;
  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4(scales, group, n);
    }

#pragma unroll
    for (int j = 0; j < 4; j++) {
      int4 load_int4[2];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][4];
      dequant_8bit_8(load_int4[0].x, load_int4[1].x, dq[0], size_n, zeros[0]);
      dequant_8bit_8(load_int4[0].y, load_int4[1].y, dq[1], size_n, zeros[1]);
      dequant_8bit_8(load_int4[0].z, load_int4[1].z, dq[2], size_n, zeros[2]);
      dequant_8bit_8(load_int4[0].w, load_int4[1].w, dq[3], size_n, zeros[3]);

      for (int m = 0; m < m_count; m++) {
        block_c[m][0] =
            dot22_8_h(dq[0], a_ptr + m * a_stride, block_c[m][0], scales[0]);
        block_c[m][1] =
            dot22_8_h(dq[1], a_ptr + m * a_stride, block_c[m][1], scales[1]);
        block_c[m][2] =
            dot22_8_h(dq[2], a_ptr + m * a_stride, block_c[m][2], scales[2]);
        block_c[m][3] =
            dot22_8_h(dq[3], a_ptr + m * a_stride, block_c[m][3], scales[3]);
      }
      a_ptr += 8;
    }
    k += 32;
  }

  for (int m = 0; m < m_count; m++) {
    half2* out = (half2*)c_.item_ptr(offset_m + m, n);
    half2 result01 = __halves2half2(block_c[m][0], block_c[m][1]);
    half2 result23 = __halves2half2(block_c[m][2], block_c[m][3]);
    atomicAdd(out, result01);
    atomicAdd(out + 1, result23);
  }
}

fp_gemm_half_q_half_gptq_kernel pick_gemm_half_q_half_gptq_kernel_v2(
    bool first_block, const int m_count, const int bit) {
#define SELECT_KERNEL(M_COUNT)                                                \
  if (m_count == M_COUNT) {                                                   \
    if (bit == 2) return gemm_half_q_half_gptq_2bit_kernel_v2<true, M_COUNT>; \
    if (bit == 3) return gemm_half_q_half_gptq_3bit_kernel_v2<true, M_COUNT>; \
    if (bit == 4) return gemm_half_q_half_gptq_4bit_kernel_v2<true, M_COUNT>; \
    if (bit == 8) return gemm_half_q_half_gptq_8bit_kernel_v2<true, M_COUNT>; \
  }
#if BLOCK_M_SIZE_MAX >= 1
  SELECT_KERNEL(1);
#endif
#if BLOCK_M_SIZE_MAX >= 2
  SELECT_KERNEL(2);
#endif
#if BLOCK_M_SIZE_MAX >= 3
  SELECT_KERNEL(3);
#endif
#if BLOCK_M_SIZE_MAX >= 4
  SELECT_KERNEL(4);
#endif
#if BLOCK_M_SIZE_MAX >= 5
  SELECT_KERNEL(5);
#endif
#if BLOCK_M_SIZE_MAX >= 6
  SELECT_KERNEL(6);
#endif
#if BLOCK_M_SIZE_MAX >= 7
  SELECT_KERNEL(7);
#endif
#if BLOCK_M_SIZE_MAX >= 8
  SELECT_KERNEL(8);
#endif
  return NULL;
}

void gemm_half_q_half_cuda_part_v2(const half* a, const uint32_t* b_q_weight,
                                   const uint32_t* b_gptq_qzeros,
                                   const half* b_gptq_scales,
                                   const int* b_q_perm, half* c, int size_m,
                                   int size_n, int size_k, int m_count,
                                   int groups, int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE * 4);
  gridDim.y = DIVIDE(size_m, m_count);
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

  fp_gemm_half_q_half_gptq_kernel kernel =
      pick_gemm_half_q_half_gptq_kernel_v2(true, m_count, bit);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<gridDim, blockDim, 0, stream>>>(a, b_q_weight, b_gptq_qzeros,
                                           b_gptq_scales, c, size_m, size_n,
                                           size_k, groups, b_q_perm);
}

__global__ void reconstruct_exllama_8bit_kernel_v2(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q8_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 8);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 4; p++) {
      int4 load_int4[2];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][4];
      dequant_8bit_8(load_int4[0].x, load_int4[1].x, dq[0], size_n, zeros[0]);
      dequant_8bit_8(load_int4[0].y, load_int4[1].y, dq[1], size_n, zeros[1]);
      dequant_8bit_8(load_int4[0].z, load_int4[1].z, dq[2], size_n, zeros[2]);
      dequant_8bit_8(load_int4[0].w, load_int4[1].w, dq[3], size_n, zeros[3]);

      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

__global__ void reconstruct_exllama_4bit_kernel_v2(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q4_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 4);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  half2 z1z16[4][2];
  half2 y1y16[4][2];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);
  dequant_4bit_8_prep_zero(zeros[0], z1z16[0], y1y16[0]);
  dequant_4bit_8_prep_zero(zeros[1], z1z16[1], y1y16[1]);
  dequant_4bit_8_prep_zero(zeros[2], z1z16[2], y1y16[2]);
  dequant_4bit_8_prep_zero(zeros[3], z1z16[3], y1y16[3]);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
      dequant_4bit_8_prep_zero(zeros[0], z1z16[0], y1y16[0]);
      dequant_4bit_8_prep_zero(zeros[1], z1z16[1], y1y16[1]);
      dequant_4bit_8_prep_zero(zeros[2], z1z16[2], y1y16[2]);
      dequant_4bit_8_prep_zero(zeros[3], z1z16[3], y1y16[3]);
    }

    for (int p = 0; p < 4; p++) {
      half2 dq[4][4];
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      dequant_4bit_8_gptq(load_int4.x, dq[0], z1z16[0], y1y16[0], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.y, dq[1], z1z16[1], y1y16[1], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.z, dq[2], z1z16[2], y1y16[2], size_n,
                          false);
      dequant_4bit_8_gptq(load_int4.w, dq[3], z1z16[3], y1y16[3], size_n,
                          false);

      b_ptr += size_n;
      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 4; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

__global__ void reconstruct_exllama_3bit_kernel_v2(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q3_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / 32 * 3;

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 1; p++) {
      int4 load_int4[3];
      load_int4[0] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[1] = *((int4*)b_ptr);
      b_ptr += size_n;
      load_int4[2] = *((int4*)b_ptr);
      b_ptr += size_n;

      half2 dq[4][16];
      dequant_3bit_32(load_int4[0].x, load_int4[1].x, load_int4[2].x, dq[0],
                      size_n, zeros[0]);
      dequant_3bit_32(load_int4[0].y, load_int4[1].y, load_int4[2].y, dq[1],
                      size_n, zeros[1]);
      dequant_3bit_32(load_int4[0].z, load_int4[1].z, load_int4[2].z, dq[2],
                      size_n, zeros[2]);
      dequant_3bit_32(load_int4[0].w, load_int4[1].w, load_int4[2].w, dq[3],
                      size_n, zeros[3]);

      if (b_q_perm) {
        for (int j = 0; j < 16; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 16; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

__global__ void reconstruct_exllama_2bit_kernel_v2(
    const uint32_t* __restrict__ b_q_weight, const int* __restrict__ b_q_perm,
    const uint32_t* __restrict__ b_gptq_qzeros,
    const half* __restrict__ b_gptq_scales, const int size_k, const int size_n,
    const int groups, half* __restrict__ b) {
  MatrixView_half_rw b_(b, size_k, size_n);
  MatrixView_q2_row b_gptq_qzeros_(b_gptq_qzeros, groups, size_n);
  MatrixView_half b_gptq_scales_(b_gptq_scales, groups, size_n);

  auto offset_k = BLOCK_KN_SIZE * blockIdx.y;
  auto offset_n = BLOCK_KN_SIZE * blockIdx.x * 4;

  int end_k = min(offset_k + BLOCK_KN_SIZE, size_k);

  // Preload remapping table
  __shared__ int perm[BLOCK_KN_SIZE];
  auto t = threadIdx.x;

  if (b_q_perm) {
    if (offset_k + t < size_k) perm[t] = b_q_perm[offset_k + t];
  }

  // Column
  int n = offset_n + t * 4;
  if (n >= size_n) return;

  // Find initial group
  int groupsize = size_k / groups;
  int group = offset_k / groupsize;
  int nextgroup = offset_k + groupsize;

  // b offset
  int qk = offset_k / (32 / 2);

  const uint32_t* b_ptr = b_q_weight + qk * size_n + n;

  // Initial zeros/scale
  int zeros[4];
  half2 scales[4];
  b_gptq_qzeros_.item4(zeros, group, n);
  b_gptq_scales_.item4_h2(scales, group, n);

  __syncthreads();

  int k = offset_k;
  int lk = 0;

  while (k < end_k) {
    if (k == nextgroup) {
      group++;
      nextgroup += groupsize;
      b_gptq_qzeros_.item4(zeros, group, n);
      b_gptq_scales_.item4_h2(scales, group, n);
    }

    for (int p = 0; p < 2; p++) {
      const int4* b_ptr4 = (int4*)b_ptr;
      int4 load_int4 = *b_ptr4;

      half2 dq[4][8];
      dequant_2bit_16(load_int4.x, dq[0], size_n, zeros[0]);
      dequant_2bit_16(load_int4.y, dq[1], size_n, zeros[1]);
      dequant_2bit_16(load_int4.z, dq[2], size_n, zeros[2]);
      dequant_2bit_16(load_int4.w, dq[3], size_n, zeros[3]);

      b_ptr += size_n;
      // half* dqh = (half*)dq;
      if (b_q_perm) {
        for (int j = 0; j < 8; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(perm[lk++], n, __low2half(dq[0][j]), __low2half(dq[1][j]),
                  __low2half(dq[2][j]), __low2half(dq[3][j]));
          b_.set4(perm[lk++], n, __high2half(dq[0][j]), __high2half(dq[1][j]),
                  __high2half(dq[2][j]), __high2half(dq[3][j]));
        }
      } else {
        for (int j = 0; j < 8; j++) {
          for (int v = 0; v < 4; v++) dq[v][j] = __hmul2(scales[v], dq[v][j]);
          b_.set4(offset_k + lk++, n, __low2half(dq[0][j]),
                  __low2half(dq[1][j]), __low2half(dq[2][j]),
                  __low2half(dq[3][j]));
          b_.set4(offset_k + lk++, n, __high2half(dq[0][j]),
                  __high2half(dq[1][j]), __high2half(dq[2][j]),
                  __high2half(dq[3][j]));
        }
      }
    }
    k += 32;
  }
}

void reconstruct_exllama_v2(const uint32_t* b_q_weight,
                            const uint32_t* b_gptq_qzeros,
                            const half* b_gptq_scales, const int* b_q_perm,
                            half* out, int height, int width, int groups,
                            int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  gridDim.y = DIVIDE(height, BLOCK_KN_SIZE);
  gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);

  auto reconstruct_exllama_kernel = reconstruct_exllama_4bit_kernel_v2;
  if (bit == 2) {
    reconstruct_exllama_kernel = reconstruct_exllama_2bit_kernel_v2;
  } else if (bit == 3) {
    reconstruct_exllama_kernel = reconstruct_exllama_3bit_kernel_v2;
  } else if (bit == 8) {
    reconstruct_exllama_kernel = reconstruct_exllama_8bit_kernel_v2;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  reconstruct_exllama_kernel<<<gridDim, blockDim, 0, stream>>>(
      b_q_weight, b_q_perm, b_gptq_qzeros, b_gptq_scales, height, width, groups,
      out);
}

__global__ void gemm_half_q_half_alt_4bit_kernel_v2(
    const half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    half* __restrict__ mul, const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width) {
  int zero_width = width / 8;
  int vec_height = height * 4;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  auto b = blockIdx.y * BLOCK_M_SIZE_MAX;
  int b_end = min(BLOCK_M_SIZE_MAX, batch - b);
  auto h = BLOCK_KN_SIZE * blockIdx.z / 8;
  int h_end = min(BLOCK_KN_SIZE / 8, height - h) * 4;
  auto w = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;

  __shared__ half2 blockvec[BLOCK_M_SIZE_MAX][blockwidth2];
  if (threadIdx.x < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][threadIdx.x] =
          vec[(m + b) * vec_height + blockIdx.z * BLOCK_KN_SIZE / 2 +
              threadIdx.x];
    }
  }

  __shared__ half2 deq2[256][8];
  auto val = threadIdx.x / 8;
  auto off = threadIdx.x % 8;
  for (; val < 256; val += BLOCK_KN_SIZE / 8) {
    deq2[val][off] =
        __halves2half2(__int2half_rn(val & 0xF), __int2half_rn(val >> 4));
  }

  if (blockIdx.z == 0) {
    for (int m = 0; m < b_end; m++) mul[(b + m) * width + w] = __int2half_rn(0);
  }
  __syncthreads();

  int i = width * h + w;
  int g_h = h * 8;
  int k = 0;
  int z_w = w / 8;
  int z_mod = (w % 8) * 4;
  half2 res2;
  half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    half2 scales_tmp[4];
    half2 zeros_tmp[4];
    for (int tmp_k = 0; tmp_k < 4; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      half scale_f = scales[g * width + w];
      half scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
          __hmul(scale_f, __int2half_rn(
                              -((zeros[g * zero_width + z_w] >> z_mod) & 0xF))),
          __hmul(
              scale_f2,
              __int2half_rn(-((zeros[g2 * zero_width + z_w] >> z_mod) & 0xF))));
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
#ifndef USE_ROCM
      res2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
#endif
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 0) & 0xff][off], scales_tmp[0], zeros_tmp[0]),
          blockvec[m][k + 0], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 8) & 0xff][off], scales_tmp[1], zeros_tmp[1]),
          blockvec[m][k + 1], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 16) & 0xff][off], scales_tmp[2], zeros_tmp[2]),
          blockvec[m][k + 2], res2);
      res2 = __hfma2(
          __hfma2(deq2[(tmp >> 24) & 0xff][off], scales_tmp[3], zeros_tmp[3]),
          blockvec[m][k + 3], res2);
#ifndef USE_ROCM
      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));
#else
      res[m] = __hadd(
          res[m], __hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)));
#endif
    }
    i += width;
    k += 4;
  }
  for (int m = 0; m < b_end; m++) {
    atomicAdd(&mul[(b + m) * width + w], res[m]);
  }
}

__global__ void gemm_half_q_half_alt_8bit_kernel_v2(
    const half2* __restrict__ vec, const uint32_t* __restrict__ mat,
    half* __restrict__ mul, const half* __restrict__ scales,
    const uint32_t* __restrict__ zeros, const int* __restrict__ g_idx,
    int batch, int height, int width) {
  int zero_width = width / 4;
  int vec_height = height * 2;
  const int blockwidth2 = BLOCK_KN_SIZE / 2;
  auto b = blockIdx.y * BLOCK_M_SIZE_MAX;
  int b_end = min(BLOCK_M_SIZE_MAX, batch - b);
  auto h = BLOCK_KN_SIZE * blockIdx.z / 4;
  int h_end = min(BLOCK_KN_SIZE / 4, height - h) * 2;
  auto w = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;

  __shared__ half2 blockvec[BLOCK_M_SIZE_MAX][blockwidth2];
  if (threadIdx.x < h_end) {
    for (int m = 0; m < b_end; ++m) {
      blockvec[m][threadIdx.x] =
          vec[(m + b) * vec_height + blockIdx.z * BLOCK_KN_SIZE / 2 +
              threadIdx.x];
    }
  }

  if (blockIdx.z == 0) {
    for (int m = 0; m < b_end; m++) mul[(b + m) * width + w] = __int2half_rn(0);
  }
  __syncthreads();

  int i = width * h + w;
  int g_h = h * 4;
  int k = 0;
  int z_w = w / 4;
  int z_mod = (w % 4) * 8;
  half2 res2;
  half res[BLOCK_M_SIZE_MAX] = {};

  unsigned int tmp;
  while (k < h_end) {
    tmp = mat[i];
    half2 scales_tmp[2];
    half2 zeros_tmp[2];
    for (int tmp_k = 0; tmp_k < 2; tmp_k++) {
      int g = g_idx[g_h + (k + tmp_k) * 2];
      int g2 = g_idx[g_h + (k + tmp_k) * 2 + 1];
      half scale_f = scales[g * width + w];
      half scale_f2 = scales[g2 * width + w];
      half2 scale = __halves2half2(scale_f, scale_f2);
      half2 zero = __halves2half2(
          __hmul(scale_f, __int2half_rn(-(
                              (zeros[g * zero_width + z_w] >> z_mod) & 0xff))),
          __hmul(scale_f2,
                 __int2half_rn(
                     -((zeros[g2 * zero_width + z_w] >> z_mod) & 0xff))));
      scales_tmp[tmp_k] = scale;
      zeros_tmp[tmp_k] = zero;
    }
    for (int m = 0; m < b_end; m++) {
#ifndef USE_ROCM
      res2 = {};
#else
      res2.x = __half_as_ushort(__float2half(0));
      res2.y = __half_as_ushort(__float2half(0));
#endif
      half2 v12 = __halves2half2(__int2half_rn(tmp & 0xFF),
                                 __int2half_rn((tmp >> 8) & 0xFF));
      res2 = __hfma2(__hfma2(v12, scales_tmp[0], zeros_tmp[0]),
                     blockvec[m][k + 0], res2);
      half2 v34 = __halves2half2(__int2half_rn((tmp >> 16) & 0xFF),
                                 __int2half_rn((tmp >> 24) & 0xFF));
      res2 = __hfma2(__hfma2(v34, scales_tmp[1], zeros_tmp[1]),
                     blockvec[m][k + 1], res2);
#ifndef USE_ROCM
      res[m] = __hadd(res[m], __hadd(res2.x, res2.y));
#else
      res[m] = __hadd(
          res[m], __hadd(__ushort_as_half(res2.x), __ushort_as_half(res2.y)));
#endif
    }
    i += width;
    k += 2;
  }
  for (int m = 0; m < b_end; m++) {
    atomicAdd(&mul[(b + m) * width + w], res[m]);
  }
}

void gemm_half_q_half_alt_v2(const half* a, const uint32_t* b_q_weight,
                             const uint32_t* b_gptq_qzeros,
                             const half* b_gptq_scales, const int* b_g_idx,
                             half* c, int size_m, int size_n, int size_k,
                             int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  blockDim.z = 1;
  gridDim.x = DIVIDE(size_n, BLOCK_KN_SIZE);
  gridDim.y = DIVIDE(size_m, BLOCK_M_SIZE_MAX);
  gridDim.z = DIVIDE(size_k, BLOCK_KN_SIZE);

  auto kernel = gemm_half_q_half_alt_4bit_kernel_v2;
  if (bit == 8) {
    kernel = gemm_half_q_half_alt_8bit_kernel_v2;
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<gridDim, blockDim, 0, stream>>>(
      (const half2*)a, b_q_weight, c, b_gptq_scales, b_gptq_qzeros, b_g_idx,
      size_m, size_k / 32 * bit, size_n);
}

template <class T, int bit>
__global__ void reconstruct_gptq_kernel_v2(const uint32_t* __restrict__ w,
                                           const half* __restrict__ w_scales,
                                           const uint32_t* __restrict__ w_zeros,
                                           const int* __restrict__ g_idx,
                                           const int height, const int width,
                                           const int group,
                                           half* __restrict__ out) {
  // Start of block

  auto column = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  auto row = blockIdx.y * 32 / bit;
  if (column >= width) return;

  // Views

  MatrixView_half_rw out_(out, height, width);
  MatrixView_half w_scales_(w_scales, group, width);
  T w_zeros_(w_zeros, group, width);

  uint32_t w_read = w[blockIdx.y * width + column];
  half* out_ptr = out_.item_ptr(row, column);

#pragma unroll
  for (int s = 0; s < 32; s += bit) {
    int group = g_idx[row + s / bit];
    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column);
    half w_item =
        __hmul(__int2half_rn((int)((w_read >> s) & ((1 << bit) - 1)) - w_zero),
               w_scale);
    *out_ptr = w_item;
    out_ptr += out_.width;
  }
}

__global__ void reconstruct_gptq_3bit_kernel_v2(
    const uint32_t* __restrict__ w, const half* __restrict__ w_scales,
    const uint32_t* __restrict__ w_zeros, const int* __restrict__ g_idx,
    const int height, const int width, const int group,
    half* __restrict__ out) {
  // Start of block
  auto column = BLOCK_KN_SIZE * blockIdx.x + threadIdx.x;
  auto row = blockIdx.y * 32;
  if (column >= width) return;

  // Views

  MatrixView_half_rw out_(out, height, width);
  MatrixView_half w_scales_(w_scales, group, width);
  MatrixView_q3_row w_zeros_(w_zeros, group, width);

  uint32_t w1 = w[(blockIdx.y * 3) * width + column];
  uint32_t w2 = w[(blockIdx.y * 3 + 1) * width + column];
  uint32_t w3 = w[(blockIdx.y * 3 + 2) * width + column];
  half* out_ptr = out_.item_ptr(row, column);

#pragma unroll
  for (int i = 0; i < 32; i += 1) {
    int group = g_idx[row + i];
    half w_scale = w_scales_.item(group, column);
    uint32_t w_zero = w_zeros_.item(group, column);
    int w_item;
    if (i == 10) {
      w_item = (w1 >> 30) | ((w2 << 2) & 0x4);
    } else if (i == 21) {
      w_item = (w2 >> 31) | ((w3 << 1) & 0x6);
    } else if (i < 10) {
      w_item = ((w1 >> (i * 3)) & 0x7);
    } else if (i < 21) {
      w_item = ((w2 >> (i * 3 - 32)) & 0x7);
    } else {
      w_item = ((w3 >> (i * 3 - 64)) & 0x7);
    }
    *out_ptr = __hmul(__int2half_rn(w_item - w_zero), w_scale);
    out_ptr += out_.width;
  }
}

void reconstruct_gptq_v2(const uint32_t* b_q_weight,
                         const uint32_t* b_gptq_qzeros,
                         const half* b_gptq_scales, const int* b_g_idx,
                         half* out, int height, int width, int groups,
                         int bit) {
  dim3 blockDim, gridDim;
  blockDim.x = BLOCK_KN_SIZE;
  blockDim.y = 1;
  gridDim.y = DIVIDE(height, 32 / bit);
  gridDim.x = DIVIDE(width, BLOCK_KN_SIZE);

  auto kernel = reconstruct_gptq_kernel_v2<MatrixView_q4_row, 4>;
  if (bit == 2) {
    kernel = reconstruct_gptq_kernel_v2<MatrixView_q2_row, 2>;
  } else if (bit == 8) {
    kernel = reconstruct_gptq_kernel_v2<MatrixView_q8_row, 8>;
  } else if (bit == 3) {
    kernel = reconstruct_gptq_3bit_kernel_v2;
    gridDim.y = DIVIDE(height, 32);
  }

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  kernel<<<gridDim, blockDim, 0, stream>>>(b_q_weight, b_gptq_scales,
                                           b_gptq_qzeros, b_g_idx, height,
                                           width, groups, out);
}

void gemm_half_q_half_cuda_v2(cublasHandle_t cublas_handle, const half* a,
                              const uint32_t* b_q_weight,
                              const uint32_t* b_gptq_qzeros,
                              const half* b_gptq_scales, const int* b_g_idx,
                              half* c, half* temp_dq, int size_m, int size_n,
                              int size_k, int groups, bool use_exllama,
                              int bit) {
  bool use_reconstruct;
  if (use_exllama) {
    use_reconstruct = ((bit == 8 && size_m > MAX_Q_GEMM_ROWS_8BIT) ||
                       (bit != 8 && size_m > MAX_Q_GEMM_ROWS));
  } else {
    // The 2/3-bit kernels are somehow slower than dequant + gemm baseline, so
    // we disabled them for now.
    use_reconstruct = (bit < 4 || size_m > MAX_ALT_GEMM_ROWS);
  }
  if (use_reconstruct) {
    // Reconstruct FP16 matrix, then cuBLAS
    if (use_exllama) {
      reconstruct_exllama_v2(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                             temp_dq, size_k, size_n, groups, bit);
    } else {
      reconstruct_gptq_v2(b_q_weight, b_gptq_qzeros, b_gptq_scales, b_g_idx,
                          temp_dq, size_k, size_n, groups, bit);
    }

    const half alpha = __float2half(1.0f);
    const half beta = __float2half(0.0f);
    cublasHgemm(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, size_n, size_m, size_k,
                &alpha, temp_dq, size_n, a, size_k, &beta, c, size_n);
  } else if (use_exllama) {
    // Quantized matmul
    int max_chunks = size_m / BLOCK_M_SIZE_MAX;
    int last_chunk = max_chunks * BLOCK_M_SIZE_MAX;
    int last_chunk_size = size_m - last_chunk;

    if (max_chunks) {
      gemm_half_q_half_cuda_part_v2(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                                    b_g_idx, c, last_chunk, size_n, size_k,
                                    BLOCK_M_SIZE_MAX, groups, bit);
    }

    if (last_chunk_size) {
      gemm_half_q_half_cuda_part_v2(
          a + last_chunk * size_k, b_q_weight, b_gptq_qzeros, b_gptq_scales,
          b_g_idx, c + last_chunk * size_n, last_chunk_size, size_n, size_k,
          last_chunk_size, groups, bit);
    }
  } else {
    gemm_half_q_half_alt_v2(a, b_q_weight, b_gptq_qzeros, b_gptq_scales,
                            b_g_idx, c, size_m, size_n, size_k, bit);
  }
}

}  // namespace gptq
}  // namespace vllm

torch::Tensor gptq_gemm_v2(torch::Tensor a, torch::Tensor b_q_weight,
                           torch::Tensor b_gptq_qzeros,
                           torch::Tensor b_gptq_scales, torch::Tensor b_g_idx,
                           bool use_exllama, int64_t bit) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(a));
  auto options = torch::TensorOptions().dtype(a.dtype()).device(a.device());
  at::Tensor c = torch::empty({a.size(0), b_q_weight.size(1)}, options);
  at::Tensor temp_dq = torch::empty(
      {b_q_weight.size(0) * 32 / bit, b_q_weight.size(1)}, options);

  vllm::gptq::gemm_half_q_half_cuda_v2(
      at::cuda::getCurrentCUDABlasHandle(), (const half*)a.data_ptr(),
      (const uint32_t*)b_q_weight.data_ptr(),
      (const uint32_t*)b_gptq_qzeros.data_ptr(),
      (const half*)b_gptq_scales.data_ptr(),
      b_g_idx.device().is_meta() ? NULL : (const int*)b_g_idx.data_ptr(),
      (half*)c.data_ptr(), (half*)temp_dq.data_ptr(),
      c.size(0),              // m
      c.size(1),              // n
      a.size(1),              // k
      b_gptq_qzeros.size(0),  // group number
      use_exllama, bit);
  return c;
}