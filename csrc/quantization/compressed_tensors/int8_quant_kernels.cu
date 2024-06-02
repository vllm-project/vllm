#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>
#include <cmath>

#include "../../dispatch_utils.h"

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

static inline __device__ int8_t half_to_int8_rn(__half x) {
  uint32_t dst;
  unsigned short conv = *(reinterpret_cast<unsigned short*>(&(x)));
  asm("cvt.rni.sat.s8.f16 %0, %1;" : "=r"(dst) : "h"(conv));
  return static_cast<int8_t>dst;
}

namespace vllm {

typedef struct __align__(8) {
  half x;
  half y;
  half z;
  half w;
}
half4;

template <typename scalar_t, typename scale_type>
__global__ void static_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ out,
    scale_type inverted_scale, const int hidden_size);

template <>
__global__ void static_scaled_int8_quant_kernel<at::Half, float>(
    const at::Half* __restrict__ input, int8_t* __restrict__ out,
    float inverted_scale, const int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;
  const half4* vectorized = reinterpret_cast<const half4*>(input);
  const int traverse_space = hidden_size >> 2;
  // float2half is inexpensive plus multiplying two halves leads to a larger
  // loss of precision
  const __half h_inverted_scale = __float2half(inverted_scale);

#pragma unroll 4
  for (int i = tid; i < traverse_space; i += blockDim.x) {
    int index = token_idx * traverse_space + i;

    half4 data_half4 = vectorized[index];

    float4 data_float4;
    data_float4.x = __half2float(data_half4.x) * inverted_scale;
    data_float4.y = __half2float(data_half4.y) * inverted_scale;
    data_float4.z = __half2float(data_half4.z) * inverted_scale;
    data_float4.w = __half2float(data_half4.w) * inverted_scale;

    int8_t quantized0 = half_to_int8_rn(__float2half(data_float4.x));
    int8_t quantized1 = half_to_int8_rn(__float2half(data_float4.y));
    int8_t quantized2 = half_to_int8_rn(__float2half(data_float4.z));
    int8_t quantized3 = half_to_int8_rn(__float2half(data_float4.w));

    char4 store_value;
    store_value.x = static_cast<char>(quantized0);
    store_value.y = static_cast<char>(quantized1);
    store_value.z = static_cast<char>(quantized2);
    store_value.w = static_cast<char>(quantized3);

    reinterpret_cast<char4*>(out)[index] = store_value;
  }
}

template <typename scalar_t, typename scale_type>
__global__ void static_scaled_int8_quant_kernel(
    const scalar_t* __restrict__ input, int8_t* __restrict__ out,
    scale_type scale, const int hidden_size) {
  const int tid = threadIdx.x;
  const int token_idx = blockIdx.x;

  for (int i = tid; i < hidden_size; i += blockDim.x) {
    out[token_idx * hidden_size + i] =
        float_to_int8_rn(((float)input[token_idx * hidden_size + i]) * scale);
  }
}
}  // namespace vllm

void static_scaled_int8_quant(torch::Tensor& out,    // [..., hidden_size]
                              torch::Tensor& input,  // [..., hidden_size]
                              float scale) {
  TORCH_CHECK(input.is_contiguous());
  TORCH_CHECK(out.is_contiguous());
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;
  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const float inverted_scale = 1.0f / scale;
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "static_scaled_int8_quant_kernel", [&] {
        vllm::static_scaled_int8_quant_kernel<scalar_t, float>
            <<<grid, block, 0, stream>>>(input.data_ptr<scalar_t>(),
                                         out.data_ptr<int8_t>(), inverted_scale,
                                         hidden_size);
      });
}
