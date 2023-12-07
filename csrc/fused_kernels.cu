#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dispatch_utils.h"
#include "utils.cuh"
#include "reduction_utils.cuh"
#include <cassert>

namespace vllm {
template <typename T, typename scale_type, bool use_per_token_dequant>
__global__ void dequant_add_residual_kernel(
  const int32_t* __restrict__ input,
  const T* __restrict__ residual,
  T* __restrict__ output,
  const scale_type scale,
  int num_tokens,
  int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    if constexpr (use_per_token_dequant) {
      output[token_idx * hidden_size + i] =
          (T)((((float)input[token_idx * hidden_size + i]) * scale[token_idx]) +
              (float)residual[token_idx * hidden_size + i]);
    } else {
      output[token_idx * hidden_size + i] =
          (T)((((float)input[token_idx * hidden_size + i]) * scale) +
              (float)residual[token_idx * hidden_size + i]);
    }
  }
}

template <typename T, typename scale_type, bool use_per_token_dequant>
__global__ void dequant_kernel(
  const int32_t* __restrict__ input,
  T* __restrict__ output,
  const scale_type scale,
  int m,
  int hidden_size,
  int input_stride,
  int out_stride) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    if constexpr (use_per_token_dequant) {
      output[token_idx * out_stride + i] =
          (T)(((float)input[token_idx * input_stride + i]) * scale[token_idx]);
    } else {
      output[token_idx * out_stride + i] =
        (T)(((float)input[token_idx * input_stride + i]) * scale);
    }
  }
}

template <typename T, typename scale_type, bool use_per_token_quant>
__global__ void quant_kernel(
  const T* __restrict__ input,
  int8_t* __restrict__ output,
  scale_type scale,
  int num_tokens,
  int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    const float zero = 0.0f;

    for (int i = tid; i < hidden_size; i += blockDim.x) {
      float val = (float)input[token_idx * hidden_size + i];
      val = val > zero ? val : -val;
      if (val > amax_val)
        amax_val = val;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    if (tid == 0) {
      s_amax = block_amax_val;
      scale[token_idx] = block_amax_val / 127.0f;
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) * tmp_scale);
    }
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      output[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) / scale);
    }
  }
}
} // namespace vllm

void dequant_add_residual(
  torch::Tensor& out,      // [..., hidden_size]
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& residual, // [..., hidden_size]
  float scale) {
  int m = input.size(0);
  int n = input.size(1);
  dim3 grid(m);
  dim3 block(min(n, 1024));

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_kernel", [&] {
        vllm::dequant_add_residual_kernel<scalar_t, float, false><<<grid, block, 0, stream>>>(
          input.data_ptr<int32_t>(),
          residual.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          scale,
          m,
          n);
      });
}

void dequant_add_residual(
  torch::Tensor& out,      // [..., hidden_size]
  torch::Tensor& input,    // [..., hidden_size]
  torch::Tensor& residual, // [..., hidden_size]
  torch::Tensor& scale) {  // [num_tokens]
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_kernel", [&] {
        vllm::dequant_add_residual_kernel<scalar_t, float*, true><<<grid, block, 0, stream>>>(
          input.data_ptr<int32_t>(),
          residual.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          scale.data_ptr<float>(),
          num_tokens,
          hidden_size);
      });
}

void dequant(
  torch::Tensor& out,   // [..., hidden_size]
  torch::Tensor& input, // [..., hidden_size]
  float scale) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  int input_stride = input.stride(-2);
  int out_stride = out.stride(-2);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(out.scalar_type(), "dequant_kernel", [&] {
    vllm::dequant_kernel<scalar_t, float, false><<<grid, block, 0, stream>>>(
        input.data_ptr<int32_t>(),
        out.data_ptr<scalar_t>(),
        scale,
        num_tokens,
        hidden_size,
        input_stride,
        out_stride);
  });
}

void dequant(
  torch::Tensor& out,   // [..., hidden_size]
  torch::Tensor& input, // [..., hidden_size]
  torch::Tensor& scale) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  int input_stride = input.stride(-2);
  int out_stride = out.stride(-2);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(out.scalar_type(), "dequant_kernel", [&] {
    vllm::dequant_kernel<scalar_t, float*, true><<<grid, block, 0, stream>>>(
        input.data_ptr<int32_t>(),
        out.data_ptr<scalar_t>(),
        scale.data_ptr<float>(),
        num_tokens,
        hidden_size,
        input_stride,
        out_stride);
  });
}

void quant(
  torch::Tensor& out,   // [..., hidden_size]
  torch::Tensor& input, // [..., hidden_size]
  float scale) {
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel", [&] {
    vllm::quant_kernel<scalar_t, float, false><<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        out.data_ptr<int8_t>(),
        scale,
        num_tokens,
        hidden_size);
  });
}

void quant(
  torch::Tensor& out,   // [..., hidden_size]
  torch::Tensor& input, // [..., hidden_size]
  torch::Tensor& scale) { // [num_tokens]
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel", [&] {
    vllm::quant_kernel<scalar_t, float*, true><<<grid, block, 0, stream>>>(
        input.data_ptr<scalar_t>(),
        out.data_ptr<int8_t>(),
        scale.data_ptr<float>(),
        num_tokens,
        hidden_size);
  });
}