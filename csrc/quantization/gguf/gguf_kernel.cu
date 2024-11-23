#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "cuda_compat.h"

#include "ggml-common.h"
#include "vecdotq.cuh"
#include "dequantize.cuh"
#include "mmvq.cuh"
#include "mmq.cuh"

// Q8 gemv
static __global__ void quantize_q8_1(const half* __restrict__ x,
                                     void* __restrict__ vy, const int kx,
                                     const int kx_padded) {
  const int ix = blockDim.x * blockIdx.x + threadIdx.x;
  if (ix >= kx_padded) {
    return;
  }
  const int iy = blockDim.y * blockIdx.y + threadIdx.y;
  const int i_padded = iy * kx_padded + ix;

  block_q8_1* y = (block_q8_1*)vy;

  const int ib = i_padded / QK8_1;   // block index
  const int iqs = i_padded % QK8_1;  // quant index

  const float xi = ix < kx ? __half2float(x[iy * kx + ix]) : 0.0f;
  float amax = fabsf(xi);
  float sum = xi;

#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    amax = fmaxf(amax, VLLM_SHFL_XOR_SYNC_WIDTH(amax, mask, 32));
    sum += VLLM_SHFL_XOR_SYNC_WIDTH(sum, mask, 32);
  }

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  y[ib].ds.x = __float2half(d);
  y[ib].ds.y = __float2half(sum);
}

static void quantize_row_q8_1_cuda(const half* x, void* vy, const int kx,
                                   const int ky, cudaStream_t stream) {
  const int64_t kx_padded = (kx + 512 - 1) / 512 * 512;
  const int block_num_x =
      (kx_padded + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
  const dim3 num_blocks(block_num_x, ky, 1);
  const dim3 block_size(CUDA_DEQUANTIZE_BLOCK_SIZE, 1, 1);
  quantize_q8_1<<<num_blocks, block_size, 0, stream>>>(x, vy, kx, kx_padded);
}

torch::Tensor ggml_dequantize(torch::Tensor W,  // quant weight
                              int64_t type, int64_t m, int64_t n) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(W));
  auto options =
      torch::TensorOptions().dtype(torch::kFloat16).device(W.device());
  at::Tensor DW = torch::empty({m, n}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  const to_fp16_cuda_t to_fp16_cuda = ggml_get_to_fp16_cuda(type);
  to_fp16_cuda((void*)W.data_ptr(), (half*)DW.data_ptr(), m * n, stream);
  return DW;
}

torch::Tensor ggml_mul_mat_vec_a8(torch::Tensor W,  // quant weight
                                  torch::Tensor X,  // input
                                  int64_t type, int64_t row) {
  int col = X.sizes()[1];
  const int padded = (col + 512 - 1) / 512 * 512;
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options =
      torch::TensorOptions().dtype(torch::kFloat16).device(W.device());
  at::Tensor Y = torch::empty({1, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({1, padded / 32 * 9}, options);
  quantize_row_q8_1_cuda((half*)X.data_ptr(), (void*)quant_X.data_ptr(), col, 1,
                         stream);
  switch (type) {
    case 2:
      mul_mat_vec_q4_0_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 3:
      mul_mat_vec_q4_1_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 6:
      mul_mat_vec_q5_0_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 7:
      mul_mat_vec_q5_1_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 8:
      mul_mat_vec_q8_0_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 10:
      mul_mat_vec_q2_K_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 11:
      mul_mat_vec_q3_K_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 12:
      mul_mat_vec_q4_K_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 13:
      mul_mat_vec_q5_K_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 14:
      mul_mat_vec_q6_K_q8_1_cuda((void*)W.data_ptr(), (void*)quant_X.data_ptr(),
                                 (half*)Y.data_ptr(), col, row, stream);
      break;
    case 16:
      mul_mat_vec_iq2_xxs_q8_1_cuda((void*)W.data_ptr(),
                                    (void*)quant_X.data_ptr(),
                                    (half*)Y.data_ptr(), col, row, stream);
      break;
    case 17:
      mul_mat_vec_iq2_xs_q8_1_cuda((void*)W.data_ptr(),
                                   (void*)quant_X.data_ptr(),
                                   (half*)Y.data_ptr(), col, row, stream);
      break;
    case 18:
      mul_mat_vec_iq3_xxs_q8_1_cuda((void*)W.data_ptr(),
                                    (void*)quant_X.data_ptr(),
                                    (half*)Y.data_ptr(), col, row, stream);
      break;
    case 19:
      mul_mat_vec_iq1_s_q8_1_cuda((void*)W.data_ptr(),
                                  (void*)quant_X.data_ptr(),
                                  (half*)Y.data_ptr(), col, row, stream);
      break;
    case 20:
      mul_mat_vec_iq4_nl_q8_1_cuda((void*)W.data_ptr(),
                                   (void*)quant_X.data_ptr(),
                                   (half*)Y.data_ptr(), col, row, stream);
      break;
    case 21:
      mul_mat_vec_iq3_s_q8_1_cuda((void*)W.data_ptr(),
                                  (void*)quant_X.data_ptr(),
                                  (half*)Y.data_ptr(), col, row, stream);
      break;
    case 22:
      mul_mat_vec_iq2_s_q8_1_cuda((void*)W.data_ptr(),
                                  (void*)quant_X.data_ptr(),
                                  (half*)Y.data_ptr(), col, row, stream);
      break;
    case 23:
      mul_mat_vec_iq4_xs_q8_1_cuda((void*)W.data_ptr(),
                                   (void*)quant_X.data_ptr(),
                                   (half*)Y.data_ptr(), col, row, stream);
      break;
    case 29:
      mul_mat_vec_iq1_m_q8_1_cuda((void*)W.data_ptr(),
                                  (void*)quant_X.data_ptr(),
                                  (half*)Y.data_ptr(), col, row, stream);
      break;
  }
  return Y;
}

torch::Tensor ggml_mul_mat_a8(torch::Tensor W,  // quant weight
                              torch::Tensor X,  // input
                              int64_t type, int64_t row) {
  int col = X.sizes()[1];
  int padded = (col + 512 - 1) / 512 * 512;
  int batch = X.sizes()[0];
  const at::cuda::OptionalCUDAGuard device_guard(device_of(X));
  auto options =
      torch::TensorOptions().dtype(torch::kFloat16).device(W.device());
  at::Tensor Y = torch::empty({batch, row}, options);
  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  options = torch::TensorOptions().dtype(torch::kInt32).device(W.device());
  at::Tensor quant_X = torch::empty({batch, padded / 32 * 9}, options);
  quantize_row_q8_1_cuda((half*)X.data_ptr(), (void*)quant_X.data_ptr(), col,
                         batch, stream);

  switch (type) {
    case 2:
      ggml_mul_mat_q4_0_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
    case 3:
      ggml_mul_mat_q4_1_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
    case 6:
      ggml_mul_mat_q5_0_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
    case 7:
      ggml_mul_mat_q5_1_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
    case 8:
      ggml_mul_mat_q8_0_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
    case 10:
      ggml_mul_mat_q2_K_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
    case 11:
      ggml_mul_mat_q3_K_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
    case 12:
      ggml_mul_mat_q4_K_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
    case 13:
      ggml_mul_mat_q5_K_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
    case 14:
      ggml_mul_mat_q6_K_q8_1_cuda(
          (void*)W.data_ptr(), (void*)quant_X.data_ptr(), (half*)Y.data_ptr(),
          col, row, batch, padded, row, stream);
      break;
  }
  return Y;
}
