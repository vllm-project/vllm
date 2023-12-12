#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include "dispatch_utils.h"
#include "utils.cuh"
#include "reduction_utils.cuh"

namespace vllm {

template <typename T> __device__ __forceinline__ T silu(const T &x) {
  // x * sigmoid(x)
  return (T)(((float)x) / (1.0f + expf((float)-x)));
}

  
template<typename scalar_t>
__global__ void silu_and_mul_kernel(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., 2 * d]
  const int d) {

  const int token_idx = blockIdx.x;
  for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
    const scalar_t x = __ldg(&input[token_idx * 2 * d + idx]);
    const scalar_t y = __ldg(&input[token_idx * 2 * d + d + idx]);
    out[token_idx * d + idx] = silu(x) * y;
  }
}

// dequant int32 input, apply silu and mul, then per token quant to int8
template <typename scale_type, bool use_per_token_quant>
__global__ void dequant_silu_and_mul_quant_kernel(
    int8_t *__restrict__ out,          // [..., d]
    const int32_t *__restrict__ input, // [..., 2 * d]
    const int d, const float gate_scale, const float up_scale,
    scale_type out_scale,                  // [num_tokens]
    float *__restrict__ tmp = nullptr // [num_tokens, d]
) {
  const int token_idx = blockIdx.x;
  if constexpr (use_per_token_quant) {
    float amax_val = 0.0f;
    const float zero = 0.0f;

    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const float x =
          (float)__ldg(&input[token_idx * 2 * d + idx]) * gate_scale;
      const float y =
          (float)__ldg(&input[token_idx * 2 * d + d + idx]) * up_scale;
      float t = silu(x) * y;
      tmp[token_idx * d + idx] = t;
      t = t > zero ? t : -t;
      if (t > amax_val)
        amax_val = t;
    }

    __shared__ float s_amax;
    const float block_amax_val = blockReduceMax(amax_val);
    if (threadIdx.x == 0) {
      s_amax = block_amax_val;
      out_scale[token_idx] = block_amax_val / 127.0f;
    }
    __syncthreads();

    float tmp_scale = 127.0f / s_amax;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      out[token_idx * d + idx] =
          float_to_int8_rn(tmp_scale * tmp[token_idx * d + idx]);
    }
  } else {
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x) {
      const float x =
          (float)__ldg(&input[token_idx * 2 * d + idx]) * gate_scale;
      const float y =
          (float)__ldg(&input[token_idx * 2 * d + d + idx]) * up_scale;
      out[token_idx * d + idx] = float_to_int8_rn(silu(x) * y / out_scale);
    }
  }
}
} // namespace vllm


void silu_and_mul(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input)    // [..., 2 * d]
{
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "silu_and_mul_kernel", [&] {
    vllm::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(), d);
  });
}

void invoke_dequant_silu_and_mul_quant(
    torch::Tensor &out,   // [..., d]
    torch::Tensor &input, // [..., 2 * d]
    float gate_scale,
    float up_scale,
    float out_scale) {
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::dequant_silu_and_mul_quant_kernel<float, false><<<grid, block, 0, stream>>>(
      out.data_ptr<int8_t>(), input.data_ptr<int32_t>(), d, gate_scale,
      up_scale, out_scale);
}


void invoke_dequant_silu_and_mul_quant(
    torch::Tensor &out,   // [..., d]
    torch::Tensor &input, // [..., 2 * d]
    float gate_scale,
    float up_scale,
    torch::Tensor &out_scale, // [num_tokens]
    torch::Tensor &tmp // [..., d]
) {
  int64_t num_tokens = input.numel() / input.size(-1);
  int d = input.size(-1) / 2;
  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  vllm::dequant_silu_and_mul_quant_kernel<float*, true><<<grid, block, 0, stream>>>(
      out.data_ptr<int8_t>(), input.data_ptr<int32_t>(),
       d, gate_scale, up_scale, out_scale.data_ptr<float>(), tmp.data_ptr<float>());
}

namespace vllm {

// Element-wise activation kernel template.
template<typename scalar_t, scalar_t (*ACT_FN)(const scalar_t&)>
__global__ void activation_kernel(
  scalar_t* __restrict__ out,               // [..., d]
  const scalar_t* __restrict__ input,       // [..., d]
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
  int d = input.size(-1);                                                                 \
  int num_tokens = input.numel() / d;                                                     \
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

template <typename T> __device__ __forceinline__ T gelu_new_kernel(const T &x) {
  const float x3 = (float)(x * x * x);
  const T t = (T)tanhf((T)(0.79788456f * (float)(x + (T)(0.044715f * x3))));
  return ((T)0.5) * x * (((T)1.0) + t);
}

template <typename T>
__device__ __forceinline__ T gelu_fast_kernel(const T &x) {
  const float f = (float)x;
  const T t =
      (T)tanhf(((T)(f * 0.79788456f)) * (((T)1.0) + (T)(0.044715f * f) * x));
  return ((T)0.5) * x * (((T)1.0) + t);
}

} // namespace vllm

void gelu_new(
  torch::Tensor& out,     // [..., d]
  torch::Tensor& input)   // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_new_kernel);
}


void gelu_fast(
  torch::Tensor& out,     // [..., d]
  torch::Tensor& input)   // [..., d]
{
  LAUNCH_ACTIVATION_KERNEL(vllm::gelu_fast_kernel);
}
