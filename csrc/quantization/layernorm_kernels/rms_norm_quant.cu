#include <ATen/cuda/CUDAContext.h>
#include <torch/all.h>
#include <c10/cuda/CUDAGuard.h>

#include "../../dispatch_utils.h"
#include "../../reduction_utils.cuh"
// #include "quant_utils.cuh"

#ifndef USE_ROCM
using FP8_TYPE = c10::Float8_e4m3fn;
C10_HOST_DEVICE constexpr auto FP8_E4M3_MAX =
    std::numeric_limits<FP8_TYPE>::max();
#else
  #include "amd/hip_float8.h"
using FP8_TYPE = c10::Float8_e4m3fnuz;
// Using the default max value from pytorch (240.0) will cause accuracy
// issue when running dynamic quantization. Here use 224.0f for rocm.
constexpr auto FP8_E4M3_MAX = 224.0f;
#endif
namespace vllm {

template <bool is_scale_inverted>
__device__ __forceinline__ FP8_TYPE scaled_fp8_conversion(float const val,
                                                          float const scale) {
  float x = 0.0f;
  if constexpr (is_scale_inverted) {
    x = val * scale;
  } else {
    x = val / scale;
  }

  float r = fmax(-FP8_E4M3_MAX, fmin(x, FP8_E4M3_MAX));
#ifndef USE_ROCM
  return static_cast<c10::Float8_e4m3fn>(r);
#else
  // Use hardware cvt instruction for fp8 on rocm
  return c10::Float8_e4m3fnuz(hip_fp8(r).data,
                              c10::Float8_e4m3fnuz::from_bits());
#endif
}

static inline __device__ int8_t float_to_int8_rn(float x) {
  uint32_t dst;
  asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(dst) : "f"(x));
  return reinterpret_cast<const int8_t&>(dst);
}

template <typename scalar_t>
__global__ void rms_norm_quant_kernel(
    FP8_TYPE* __restrict__ out,             // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    float* __restrict__ tmp,              // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    float* __restrict__ scale,            // [num_tokens]
    const double epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * hidden_size + idx];
    variance += x * x;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // __shared__ float s_amax;
  // float amax_val = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * hidden_size + idx];
    x = x * s_variance * (float)(weight[idx]);
    // input[blockIdx.x * hidden_size + idx] = (scalar_t) x;
    tmp[blockIdx.x * hidden_size + idx] = x;
    // amax_val = fmaxf(amax_val, fabsf(x));
  }
  // amax_val = blockReduceMax(amax_val);
  // if (threadIdx.x == 0) {
  //   s_amax = amax_val;
  //   scale[blockIdx.x] = amax_val / 127.0f;
  // }
  // __syncthreads();

  // float tmp_scale = 127.0f / s_amax;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    // out[blockIdx.x * hidden_size + idx] =
    //     float_to_int8_rn(((float) input[blockIdx.x * hidden_size + idx]) *
    //     tmp_scale);
    out[blockIdx.x * hidden_size + idx] =
        scaled_fp8_conversion<false>(tmp[blockIdx.x * hidden_size + idx], *scale);
  }
}

template <typename scalar_t>
__global__ void add_residual_rms_norm_quant_kernel(
    FP8_TYPE* __restrict__ out,             // [..., hidden_size]
    const scalar_t* __restrict__ input,   // [..., hidden_size]
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    float* __restrict__ tmp,              // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    float* __restrict__ scale,            // [num_tokens]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * hidden_size + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }
  variance = blockReduceSum<float>(variance);
  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // __shared__ float s_amax;
  // float amax_val = 0.0f;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    x = x * s_variance * (float)(weight[idx]);
    // [blockIdx.x * hidden_size + idx] = (scalar_t) x;
    tmp[blockIdx.x * hidden_size + idx] = x;
    // amax_val = fmaxf(amax_val, fabsf(x));
  }
  // amax_val = blockReduceMax(amax_val);
  // if (threadIdx.x == 0) {
  //   s_amax = amax_val;
  //   scale[blockIdx.x] = amax_val / 127.0f;
  // }
  // __syncthreads();

  // float tmp_scale = 127.0f / s_amax;
  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    // out[blockIdx.x * hidden_size + idx] =
    //     float_to_int8_rn(((float) input[blockIdx.x * hidden_size + idx]) *
    //     tmp_scale);
    out[blockIdx.x * hidden_size + idx] =
        scaled_fp8_conversion<false>(tmp[blockIdx.x * hidden_size + idx], *scale);
  }
}

// template <typename scalar_t>
// __global__ void quant_kernel(
//   const scalar_t* __restrict__ input,
//   int8_t* __restrict__ out,
//   float* __restrict__ scale,
//   const int num_tokens,
//   const int hidden_size) {
//   __shared__ float s_amax;
//   float amax_val = 0.0f;

//   for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
//     float x = (float) input[blockIdx.x * hidden_size + idx];
//     amax_val = fmaxf(amax_val, fabsf(x));
//   }
//   amax_val = blockReduceMax(amax_val);
//   if (threadIdx.x == 0) {
//     s_amax = amax_val;
//     scale[blockIdx.x] = amax_val / 127.0f;
//   }
//   __syncthreads();

//   float tmp_scale = 127.0f / s_amax;
//   for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
//     out[blockIdx.x * hidden_size + idx] =
//         float_to_int8_rn(((float) input[blockIdx.x * hidden_size + idx]) *
//         tmp_scale);
//   }
// }

}  // namespace vllm

void rms_norm_quant(torch::Tensor& out,           // [..., hidden_size]
                    torch::Tensor const& input,   // [..., hidden_size]
                    torch::Tensor& tmp,           // [..., hidden_size]
                    torch::Tensor const& weight,  // [hidden_size]
                    torch::Tensor& scale,         // [num_tokens]
                    double const epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_quant_kernel", [&] {
        vllm::rms_norm_quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
            out.data_ptr<FP8_TYPE>(), input.data_ptr<scalar_t>(),
            tmp.data_ptr<float>(), weight.data_ptr<scalar_t>(),
            scale.data_ptr<float>(), epsilon, num_tokens, hidden_size);
      });
}

void add_residual_rms_norm_quant(
    torch::Tensor& out,           // [..., hidden_size]
    torch::Tensor const& input,   // [..., hidden_size]
    torch::Tensor& residual,      // [..., hidden_size]
    torch::Tensor& tmp,           // [..., hidden_size]
    torch::Tensor const& weight,  // [hidden_size]
    torch::Tensor& scale,         // [num_tokens]
    double const epsilon) {
  int hidden_size = input.size(-1);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "add_residual_rms_norm_quant_kernel", [&] {
        vllm::add_residual_rms_norm_quant_kernel<scalar_t>
            <<<grid, block, 0, stream>>>(
                out.data_ptr<FP8_TYPE>(), input.data_ptr<scalar_t>(),
                residual.data_ptr<scalar_t>(), tmp.data_ptr<float>(),
                weight.data_ptr<scalar_t>(), scale.data_ptr<float>(), epsilon,
                num_tokens, hidden_size);
      });
}

// void quant(
//   torch::Tensor& out,   // [..., hidden_size]
//   torch::Tensor& input, // [..., hidden_size]
//   torch::Tensor& scale) { // [num_tokens]
//   assert(input.is_contiguous());
//   assert(out.is_contiguous());
//   int hidden_size = input.size(-1);
//   int num_tokens = input.numel() / hidden_size;

//   dim3 grid(num_tokens);
//   dim3 block(std::min(hidden_size, 1024));
//   const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
//   const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
//   VLLM_DISPATCH_FLOATING_TYPES(input.scalar_type(), "quant_kernel", [&] {
//     vllm::quant_kernel<scalar_t><<<grid, block, 0, stream>>>(
//       input.data_ptr<scalar_t>(),
//       out.data_ptr<int8_t>(),
//       scale.data_ptr<float>(),
//       num_tokens,
//       hidden_size);
//   });
// }