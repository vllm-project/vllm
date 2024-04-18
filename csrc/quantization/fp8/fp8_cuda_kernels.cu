#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

__device__ __forceinline__ float atomicMaxFloat(float* addr, float value) {
    float old;
    old = (value >= 0) ? __int_as_float(atomicMax((int*)addr, __float_as_int(value))) :
         __uint_as_float(atomicMin((unsigned int*)addr, __float_as_uint(value)));

    return old;
}

template<typename scalar_t>
__global__ void segmented_max_reduction(
  float* __restrict__ scale,
  const scalar_t* __restrict__ input,
  int64_t num_elems) {
  __shared__ float cache[1024];
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int cacheIndex = threadIdx.x;

  scalar_t tmp = 0.0;
  while (i < num_elems) {
    float x = static_cast<float>(input[i]);
    tmp = max(tmp, fabs(x));
    i += blockDim.x * gridDim.x;
  }

  cache[cacheIndex] = tmp;

  __syncthreads();

  // perform parallel reduction
  int ib = blockDim.x / 2;
  while (ib != 0) {
    if (cacheIndex < ib && cache[cacheIndex + ib] > cache[cacheIndex]) {
        cache[cacheIndex] = cache[cacheIndex + ib];
    }
    __syncthreads();
    ib /= 2;
  }
  // now cache[0] contains the maximum for this thread block,
  // atomically write the max to the target location
  if (cacheIndex == 0) {
    atomicMaxFloat(scale, cache[0] / std::numeric_limits<c10::Float8_e4m3fn>::max());
  }
}

template<typename scalar_t>
__global__ void scaled_fp8_quant_kernel(
  c10::Float8_e4m3fn* __restrict__ out,
  const scalar_t* __restrict__ input,
  const float* __restrict__ scale,
  int64_t num_elems) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < num_elems) {
    out[i] = static_cast<c10::Float8_e4m3fn>(input[i] / *scale);
    i += blockDim.x * gridDim.x;
  }
}

template<typename scalar_t>
__global__ void fp8_silu_and_mul_kernel(
  c10::Float8_e4m3fn* __restrict__ out,
  const scalar_t* __restrict__ input,
  const float* __restrict__ scale,
  const int d) {
  const int64_t token_idx = blockIdx.x;
  for (int64_t idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const float x = (float) input[token_idx * 2 * d + idx];
    const float y = (float) input[token_idx * 2 * d + d + idx];
    float r = silu_kernel(x) * y;
    out[token_idx * d + idx] = static_cast<c10::Float8_e4m3fn>(r / *scale);
  }
}

} // namespace vllm

void scaled_fp8_quant(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input,    // [..., d]
  torch::Tensor& scale)   // [1]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int64_t num_elems = input.numel();
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "scaled_fp8_quant_kernel",
    [&] {
      vllm::segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
        scale.data_ptr<float>(),
        input.data_ptr<scalar_t>(),
        num_elems);
      vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(),
        input.data_ptr<scalar_t>(),
        scale.data_ptr<float>(),
        num_elems);
      });
}

void fp8_silu_and_mul_kernel(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input,    // [..., 2 * d]
  torch::Tensor& scale)   // [1]
{
  int d = input.size(-1) / 2;
  int64_t num_tokens = input.numel() / input.size(-1);
  dim3 grid(num_tokens);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    out.scalar_type(),
    "scaled_silu_and_mul_kernel",
    [&] {
      vllm::segmented_max_reduction<scalar_t><<<grid, block, 0, stream>>>(
        scale.data_ptr<float>(),
        input.data_ptr<scalar_t>(),
        input.numel());
      vllm::scaled_silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(),
        input.data_ptr<scalar_t>(),
        scale.data_ptr<float>(),
        d);
      });
}
