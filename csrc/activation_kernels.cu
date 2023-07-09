#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

namespace vllm
{

  template <typename T>
  __device__ __forceinline__ T silu(const T &x)
  {
    // x * sigmoid(x)
    return (T)(((float)x) / (1.0f + expf((float)-x)));
  }

  template <typename T>
  __device__ __forceinline__ T fast_tanh(cosnt T &x)
  {
    // 1 - (2 * (1 / (1 + exp(x*2))))
    return (T)(1.0f - (2.0f * (1.0f / (1.0f + expf((float)x * 2.0f)))));
  }

  template <typename scalar_t>
  __global__ void silu_and_mul_kernel(
      scalar_t *__restrict__ out,         // [num_tokens, d]
      const scalar_t *__restrict__ input, // [num_tokens, 2, d]
      const int d)
  {
    const int token_idx = blockIdx.x;
    for (int idx = threadIdx.x; idx < d; idx += blockDim.x)
    {
      const scalar_t x = __ldg(&input[token_idx * 2 * d + idx]);
      const scalar_t y = __ldg(&input[token_idx * 2 * d + d + idx]);
      out[token_idx * d + idx] = silu(x) * y;
    }
  }

  template <typename scalar_t>
  __global__ void fast_gelu(
      scalar_t *__restrict__ out,
      const scalar_t *__restrict__ input,
      const int d)
  {
    const int token_idx = blockIdx.x;
    const int chunk = blockDim.x;

    for (int idx = threadIdx.x; idx < d; i += chunk)
    {
      const scalar_t tensor = __ldg(&input[token_idx * 2 * d + idx]);

      // scale = sqrt(2/pi)
      //  6 decimal precision for scale
      scalar_t cdf = 0.5f * (1.0f + fast_tanh(0.797885f * (tensor + 0.044715f * (tensor * tensor * tensor))));
      out[token_idx * d + idx] = tensor * cdf;
    }
  }

} // namespace vllm

void silu_and_mul(
    torch::Tensor &out,   // [num_tokens, d]
    torch::Tensor &input) // [num_tokens, 2 * d]
{
  int num_tokens = input.size(0);
  int d = input.size(1) / 2;

  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "silu_and_mul_kernel",
      [&]
      {
        vllm::silu_and_mul_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            d);
      });
}

void fast_gelu(
    torch::Tensor &out,
    torch::Tensor &input)
{
  int num_tokens = input.size(0);
  int d = input.size(1) / 2;

  dim3 grid(num_tokens);
  dim3 block(std::min(d, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  AT_DISPATCH_FLOATING_TYPES_AND2(
      at::ScalarType::Half,
      at::ScalarType::BFloat16,
      input.scalar_type(),
      "fast_gelu_kernel",
      [&]
      {
        vllm::fast_gelu<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<scalar_t>(),
            input.data_ptr<scalar_t>(),
            d);
      });
}