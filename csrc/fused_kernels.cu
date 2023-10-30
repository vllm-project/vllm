#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dispatch_utils.h"
#include "utils.cuh"
#include <cassert>

namespace vllm {
template <typename T>
__global__ void dequant_add_residual_kernel(const int32_t *__restrict__ input,
                                            const T *__restrict__ residual,
                                            T *__restrict__ output,
                                            const float scale, int m, int n) {
  const int tid = threadIdx.x;
  for (int i = tid; i < n; i += blockDim.x) {
    output[blockIdx.x * n + i] =
        (T)((((float)input[blockIdx.x * n + i]) * scale) +
            (float)residual[blockIdx.x * n + i]);
  }
}

template <typename T>
__global__ void dequant_kernel(const int32_t *__restrict__ input,
                              T *__restrict__ output,
                              const float scale, int m, int n, int input_stride, int out_stride) {
  const int tid = threadIdx.x;
  for (int i = tid; i < n; i += blockDim.x) {
    output[blockIdx.x * out_stride + i] =
        (T)(((float)input[blockIdx.x * input_stride + i]) * scale);
  }
}

template <typename T>
__global__ void quant_kernel(const T *__restrict__ input,
                              int8_t *__restrict__ output,
                              const float scale, int m, int n) {
  const int tid = threadIdx.x;
  for (int i = tid; i < n; i += blockDim.x) {
    output[blockIdx.x * n + i] =
        float_to_int8_rn(((float)input[blockIdx.x * n + i]) / scale);
  }
}
} // namespace vllm

void invoke_dequant_add_residual(
    torch::Tensor &out,      // [num_tokens, hidden_size]
    torch::Tensor &input,    // [num_tokens, hidden_size]
    torch::Tensor &residual, // [num_tokens, hidden_size]
    float scale) {
  int m = input.size(0);
  int n = input.size(1);
  dim3 grid(m);
  dim3 block(min(n, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_kernel", [&] {
        vllm::dequant_add_residual_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.data_ptr<int32_t>(), residual.data_ptr<scalar_t>(),
            out.data_ptr<scalar_t>(), scale, m, n);
      });
}

void invoke_dequant(
    torch::Tensor &out,      // [num_tokens, hidden_size]
    torch::Tensor &input,    // [num_tokens, hidden_size]
    float scale) {
  int m = input.size(0);
  int n = input.size(1);
  int input_stride = input.stride(0);
  int out_stride = out.stride(0);
  dim3 grid(m);
  dim3 block(min(n, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      out.scalar_type(), "dequant_kernel", [&] {
        vllm::dequant_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), scale, m, n, input_stride, out_stride);
      });
}

void invoke_quant(
    torch::Tensor &out,      // [num_tokens, hidden_size]
    torch::Tensor &input,    // [num_tokens, hidden_size]
    float scale) {
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int m = input.size(0);
  int n = input.size(1);
  dim3 grid(m);
  dim3 block(min(n, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "quant_kernel", [&] {
        vllm::quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
            input.data_ptr<scalar_t>(), out.data_ptr<int8_t>(), scale, m, n);
      });
}