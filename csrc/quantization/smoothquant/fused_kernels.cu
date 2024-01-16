#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <assert.h>

#include "../../dispatch_utils.h"
#include "../../reduction_utils.cuh"
#include "quant_utils.cuh"

namespace vllm {
template <typename scalar_t, bool use_per_token_dequant>
__global__ void dequant_add_residual_kernel(
  const int32_t* __restrict__ input,
  const scalar_t* __restrict__ residual,
  scalar_t* __restrict__ out,
  const float scale,
  const int num_tokens,
  const int hidden_size,
  const float* __restrict__ act_scale = nullptr) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  float scale_ = scale;
  if constexpr (use_per_token_dequant) {
    scale_ = scale * act_scale[token_idx];
  }
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * hidden_size + i] =
      (scalar_t)((((float)input[token_idx * hidden_size + i]) * scale_) +
          (float)residual[token_idx * hidden_size + i]);
  }
}

template <typename scalar_t, bool use_per_token_dequant>
__global__ void dequant_kernel(
  const int32_t* __restrict__ input,
  scalar_t* __restrict__ out,
  const float scale,
  const int m,
  const int hidden_size,
  const int input_stride,
  const int out_stride,
  const float* __restrict__ act_scale = nullptr) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  float scale_ = scale;
  if constexpr (use_per_token_dequant) {
    scale_ = scale * act_scale[token_idx];
  }
  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * out_stride + i] =
    (scalar_t)(((float)input[token_idx * input_stride + i]) * scale_);
  }
}

template <typename scalar_t, typename scale_type, bool use_per_token_quant>
__global__ void quant_kernel(
  const scalar_t* __restrict__ input,
  int8_t* __restrict__ out,
  scale_type scale,
  const int num_tokens,
  const int hidden_size) {
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
      out[token_idx * hidden_size + i] =
          float_to_int8_rn(((float)input[token_idx * hidden_size + i]) * tmp_scale);
    }
  } else {
    for (int i = tid; i < hidden_size; i += blockDim.x) {
      out[token_idx * hidden_size + i] =
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
        vllm::dequant_add_residual_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
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
  torch::Tensor& scale,
  float weight_dequant_scale) {  // [num_tokens]
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      residual.scalar_type(), "dequant_add_residual_kernel", [&] {
        vllm::dequant_add_residual_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
          input.data_ptr<int32_t>(),
          residual.data_ptr<scalar_t>(),
          out.data_ptr<scalar_t>(),
          weight_dequant_scale,
          num_tokens,
          hidden_size,
          scale.data_ptr<float>());
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
    vllm::dequant_kernel<scalar_t, false><<<grid, block, 0, stream>>>(
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
  torch::Tensor& scale,
  float weight_dequant_scale) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  int input_stride = input.stride(-2);
  int out_stride = out.stride(-2);

  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(out.scalar_type(), "dequant_kernel", [&] {
    vllm::dequant_kernel<scalar_t, true><<<grid, block, 0, stream>>>(
      input.data_ptr<int32_t>(),
      out.data_ptr<scalar_t>(),
      weight_dequant_scale,
      num_tokens,
      hidden_size,
      input_stride,
      out_stride,
      scale.data_ptr<float>());
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