#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include "dispatch_utils.h"

namespace vllm {

template<typename T>
__device__ __forceinline__ T silu(const T& x) {
  // x * sigmoid(x)
  return (T) (((float) x) / (1.0f + expf((float) -x)));
}

template<typename scalar_t>
__global__ void silu_and_mul_kernel(
  scalar_t* __restrict__ out,               // [num_tokens, d]
  const scalar_t* __restrict__ input,       // [num_tokens, 2, d]
  const int d) {
  const int token_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx * 2 * d + idx]);
    const scalar_t y = __ldg(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = silu(x) * y;
  }
}

} // namespace vllm

void silu_and_mul(
  torch::Tensor& out,      // [num_tokens, d]
  torch::Tensor& input)    // [num_tokens, 2 * d]
{
  int num_tokens = input.size(0);
  int d = input.size(1) / 2;

  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "silu_and_mul_kernel",
    [&] {
      vllm::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(),
        input.data_ptr<scalar_t>(),
        d);
    });
}

namespace vllm {

// Element-wise activation kernel template.
template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
  scalar_t* __restrict__ out,               // [num_tokens, d]
  const scalar_t* __restrict__ input,       // [num_tokens, d]
  const int d) {
  const int token_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx * d + idx]);
    out[token_idx * d + idx] = ACT_FN(x);
  }
}

} // namespace vllm

// Launch element-wise activation kernel.
#define LAUNCH_ACTIVATION_KERNEL(KERNEL)                                                  \
  int num_tokens = input.size(0);                                                         \
  int d = input.size(1);                                                                  \
  dim3 grid(num_tokens);                                                                  \
  dim3 block(std::min(d, 1024));                                                          \
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();                           \
  VLLM_DISPATCH_FLOATING_TYPES(                                                           \
    input.scalar_type(),                                                                  \
    "activation_kernel",                                                                  \
    [&] {                                                                                 \
      vllm::activation_kernel<scalar_t, KERNEL<scalar_t>><<<grid, block, 0, stream>>>(    \
        out.data_ptr<scalar_t>(),                                                         \
        input.data_ptr<scalar_t>(),                                                       \
        d);                                                                               \
    });

namespace vllm {

template<typename T>
__device__ __forceinline__ T gelu_new_kernel(const T& x) {
  const float x3 = (float) (x * x * x);
  const T t = (T) tanhf((T) (0.79788456f * (float) (x + (T) (0.044715f * x3))));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

template<typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T& x) {
  const float f = (float) x;
  const T t = (T) tanhf(((T) (f * 0.79788456f)) * (((T) 1.0) + (T) (0.044715f * f) * x));
  return ((T) 0.5) * x * (((T) 1.0) + t);
}

} // namespace vllm

void gelu_new(
  torch::Tensor& out,     // [num_tokens, d]
  torch::Tensor& input)   // [num_tokens, d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}

void gelu_fast(
  torch::Tensor& out,     // [num_tokens, d]
  torch::Tensor& input)   // [num_tokens, d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}
