#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>

#include "cuda_compat.h"
#include "dispatch_utils.h"

namespace vllm {

template<typename scalar_t>
__global__ void scaled_fp8_quant_kernel(
  c10::Float8_e4m3fn* __restrict__ out,
  const scalar_t* __restrict__ input,
  const float* __restrict__ scale,
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
  // now cache[0] contains the maximum, rescale the numbers
  i = blockDim.x * blockIdx.x + threadIdx.x;
  while (i < num_elems) {
    out[i] = static_cast<c10::Float8_e4m3fn>(input[i] / cache[0]);
    i += blockDim.x * gridDim.x;
  }
}

} // namespace vllm

void scaled_fp8_quant(
  torch::Tensor& out,      // [..., d]
  torch::Tensor& input,    // [..., d]
  torch::Tensor& scales)   // [d]
{
  int64_t num_elems = input.numel();
  dim3 grid(1);
  dim3 block(1024);
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
    input.scalar_type(),
    "scaled_fp8_quant_kernel",
    [&] {
      vllm::scaled_fp8_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
        out.data_ptr<c10::Float8_e4m3fn>(),
        input.data_ptr<scalar_t>(),
        scales.data_ptr<float>(),
        num_elems);
      });
}