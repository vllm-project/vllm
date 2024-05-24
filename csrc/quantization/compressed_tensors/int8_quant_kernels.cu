#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cmath>

#include "../../dispatch_utils.h"
#include "../../reduction_utils.cuh"


static inline __device__ int8_t float_to_int8_rn(float x) {
#ifdef USE_ROCM
  static const float i8_min =
      static_cast<float>(std::numeric_limits<int8_t>::min());
  static const float i8_max =
      static_cast<float>(std::numeric_limits<int8_t>::max());
  // round
  float dst = std::nearbyint(x);
  // saturate
  dst = std::clamp(dst, i8_min, i8_max);
  return static_cast<int8_t>(dst);
#else
  // CUDA path
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
#endif
}

namespace vllm {

template <typename scalar_t, typename scale_type>
__global__ void static_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ out,
    scale_type scale, const int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * hidden_size + i] =
        float_to_int8_rn(((float)input[token_idx * hidden_size + i]) / scale);
  }
}


template <typename scalar_t, typename scale_type>
__global__ void dynamic_scaled_int8_quant_kernel(
  const scalar_t* __restrict__ input,
  int8_t* __restrict__ out,
  scale_type scale,
  const int hidden_size) {

  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

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
}



void static_scaled_int8_quant(torch::Tensor& out,    // [..., hidden_size]
                              torch::Tensor& input,  // [..., hidden_size]
                              float scale) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_kernel", [&] {
        vllm::static_scaled_int8_quant_kernel<scalar_t, float>
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),
                                         out.data_ptr<int8_t>(), scale,
                                         hidden_size);
      });
}


void dynamic_scaled_int8_quant(
  torch::Tensor& out,   // [..., hidden_size]
  torch::Tensor& input, // [..., hidden_size]
  torch::Tensor& scales) {
  assert(input.is_contiguous());
  assert(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "dynamic_scaled_int8_quant_kernel", [&] {
    vllm::dynamic_scaled_int8_quant_kernel<scalar_t, float*><<<grid, block, 0, stream>>>(
      input.data_ptr<scalar_t>(),
      out.data_ptr<int8_t>(),
      scales.data_ptr<float>(),
      hidden_size);
  });
}

