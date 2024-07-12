
#include <ATen/cuda/CUDAContext.h>
// #include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../dispatch_utils.h"
#include "../../reduction_utils.cuh"
#include "common.cuh"

namespace vllm {


// int8 dynamic per token helper. Norm the input and compute per-token int8
// scales
template <typename scalar_t>
__device__ void norm_and_compute_dynamic_per_token_int8_scales(
    float* __restrict__ out,  // TODO(varun) : Ignore tmp if you can!
    float* __restrict__ token_scale, float* __restrict__ all_token_scales,
    scalar_t const* __restrict__ input, scalar_t const* __restrict__ weight,
    __shared__ float* s_rms, int const hidden_size) {
  __shared__ float s_amax;
  float amax_val = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    x = x * (*s_rms) * (float)(weight[idx]);
    // TODO (varun) : Try to avoid intermediate storage.
    out[blockIdx.x * hidden_size + idx] = x;
    amax_val = fmaxf(amax_val, fabsf(x));
  }
  amax_val = blockReduceMax(amax_val);
  if (threadIdx.x == 0) {
    s_amax = amax_val;
    all_token_scales[blockIdx.x] = amax_val / 127.0f;
  }
  __syncthreads();

  *token_scale = 127.0f / s_amax;
}

// RMS norm + int8 quant kernel

template <typename scalar_t>
__global__ void rms_norm_dynamic_per_token_int8_quant_kernel(
    int8_t* __restrict__ out,             // [..., hidden_size]
    float* __restrict__ scales,           // [num_tokens]
    float* __restrict__ tmp,              // [..., hidden_size]
    scalar_t const* __restrict__ input,   // [..., hidden_size]
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const epsilon, int const hidden_size) {
  __shared__ float s_rms;
  compute_rms<scalar_t>(&s_rms, input, hidden_size, epsilon);

  float token_scale = 0.0f;
  norm_and_compute_dynamic_per_token_int8_scales<scalar_t>(
      tmp, &token_scale, scales, input, weight, &s_rms, hidden_size);

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    out[blockIdx.x * hidden_size + idx] = ScaledQuant<int8_t>::quant_fn(
        tmp[blockIdx.x * hidden_size + idx], token_scale);
  }
}

// Residual add + RMS norm + int8 quant kernel

template <typename scalar_t>
__global__ void residual_add_rms_norm_dynamic_per_token_int8_quant_kernel(
    int8_t* __restrict__ out,      // [..., hidden_size]
    float* __restrict__ scales,    // [num_tokens]
    float* __restrict__ tmp,       // [..., hidden_size]
    scalar_t* __restrict__ input,  // [..., hidden_size]
    scalar_t* __restrict__ residual,
    scalar_t const* __restrict__ weight,  // [hidden_size]
    float const epsilon, int const hidden_size) {
  __shared__ float s_rms;
  compute_rms<scalar_t>(&s_rms, input, residual, hidden_size, epsilon);

  float token_scale = 0.0f;
  norm_and_compute_dynamic_per_token_int8_scales<scalar_t>(
      tmp, &token_scale, scales, residual, weight, &s_rms, hidden_size);

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    out[blockIdx.x * hidden_size + idx] = ScaledQuant<int8_t>::quant_fn(
        tmp[blockIdx.x * hidden_size + idx], token_scale);
  }
}
}  // namespace vllm

// RMS norm + dynamic per token int8

void rms_norm_dynamic_per_token_int8_quant(
    torch::Tensor& out,     // [..., hidden_size]
    torch::Tensor& tmp,     // [..., hidden_size]
    torch::Tensor& input,   // [..., hidden_size]
    torch::Tensor& weight,  // [hidden_size]
    torch::Tensor& scales,  // [num_tokens]
    double const epsilon) {
  TORCH_CHECK(out.dtype() == torch::kInt8);

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_dynamic_per_token_int8_quant_kernel", [&] {
        vllm::rms_norm_dynamic_per_token_int8_quant_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<int8_t>(), scales.data_ptr<float>(),
                tmp.data_ptr<float>(), input.data_ptr<scalar_t>(),
                weight.data_ptr<scalar_t>(), epsilon, hidden_size);
      });
}

// Residual add + RMS norm + dynamic per token int8

void residual_add_rms_norm_dynamic_per_token_int8_quant(
    torch::Tensor& out,       // [..., hidden_size]
    torch::Tensor& tmp,       // [..., hidden_size]
    torch::Tensor& input,     // [..., hidden_size]
    torch::Tensor& residual,  // [..., hidden_size]
    torch::Tensor& weight,    // [hidden_size]
    torch::Tensor& scales,    // [num_tokens]
    double const epsilon) {
  TORCH_CHECK(out.dtype() == torch::kInt8);

  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(),
      "residual_add_rms_norm_dynamic_per_token_int8_quant_kernel", [&] {
        vllm::residual_add_rms_norm_dynamic_per_token_int8_quant_kernel<
            scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<int8_t>(), scales.data_ptr<float>(),
            tmp.data_ptr<float>(), input.data_ptr<scalar_t>(),
            residual.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), epsilon,
            hidden_size);
      });
}
