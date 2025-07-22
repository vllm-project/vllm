/*
 * This file contains the CUDA kernels for the fused quantized layernorm.
 * The kernels correspond to the kernels in layernorm_kernels.cu, except they
 * also produce quantized output directly.
 * Currently, only static fp8 quantization is supported.
 */

#include "type_convert.cuh"
#include "quantization/fp8/common.cuh"
#include "dispatch_utils.h"

#include <torch/cuda.h>
#include <c10/cuda/CUDAGuard.h>

#ifndef USE_ROCM
  #include <cub/cub.cuh>
#else
  #include <hipcub/hipcub.hpp>
#endif

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, typename fp8_type>
__global__ void rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,          // [..., hidden_size]
    const scalar_t* __restrict__ input,  // [..., hidden_size]
    const int input_stride,
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float* __restrict__ scale,      // [1]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    const float x = (float)input[blockIdx.x * input_stride + idx];
    variance += x * x;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // invert scale to avoid division
  float const scale_inv = 1.0f / *scale;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)input[blockIdx.x * input_stride + idx];
    float const out_norm = ((scalar_t)(x * s_variance)) * weight[idx];
    out[blockIdx.x * hidden_size + idx] =
        scaled_fp8_conversion<true, fp8_type>(out_norm, scale_inv);
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width, typename fp8_type>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,    // [..., hidden_size]
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float* __restrict__ scale,      // [1]
    const float epsilon, const int num_tokens, const int hidden_size) {
  // Sanity checks on our vector struct and type-punned pointer arithmetic
  static_assert(std::is_pod_v<_f16Vec<scalar_t, width>>);
  static_assert(sizeof(_f16Vec<scalar_t, width>) == sizeof(scalar_t) * width);

  const int vec_hidden_size = hidden_size / width;
  const int vec_input_stride = input_stride / width;
  __shared__ float s_variance;
  float variance = 0.0f;
  /* These and the argument pointers are all declared `restrict` as they are
     not aliased in practice. Argument pointers should not be dereferenced
     in this kernel as that would be undefined behavior */
  auto* __restrict__ input_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(input);
  auto* __restrict__ residual_v =
      reinterpret_cast<_f16Vec<scalar_t, width>*>(residual);
  auto* __restrict__ weight_v =
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(weight);

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int stride_id = blockIdx.x * vec_input_stride + idx;
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = input_v[stride_id];
    temp += residual_v[id];
    variance += temp.sum_squares();
    residual_v[id] = temp;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // invert scale to avoid division
  float const scale_inv = 1.0f / *scale;

  for (int idx = threadIdx.x; idx < vec_hidden_size; idx += blockDim.x) {
    int id = blockIdx.x * vec_hidden_size + idx;
    _f16Vec<scalar_t, width> temp = residual_v[id];
    temp *= s_variance;
    temp *= weight_v[idx];
#pragma unroll
    for (int i = 0; i < width; ++i) {
      out[id * width + i] =
          scaled_fp8_conversion<true, fp8_type>(float(temp.data[i]), scale_inv);
    }
  }
}

/* Generic fused_add_rms_norm_kernel
   The width field is not used here but necessary for other specializations.
 */
template <typename scalar_t, int width, typename fp8_type>
__global__ std::enable_if_t<(width == 0) || !_typeConvert<scalar_t>::exists>
fused_add_rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,    // [..., hidden_size]
    scalar_t* __restrict__ input,  // [..., hidden_size]
    const int input_stride,
    scalar_t* __restrict__ residual,      // [..., hidden_size]
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float* __restrict__ scale,      // [1]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    scalar_t z = input[blockIdx.x * input_stride + idx];
    z += residual[blockIdx.x * hidden_size + idx];
    float x = (float)z;
    variance += x * x;
    residual[blockIdx.x * hidden_size + idx] = z;
  }

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, cub::Sum{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // invert scale to avoid division
  float const scale_inv = 1.0f / *scale;

  for (int idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float x = (float)residual[blockIdx.x * hidden_size + idx];
    float const out_norm = ((scalar_t)(x * s_variance)) * weight[idx];
    out[blockIdx.x * hidden_size + idx] =
        scaled_fp8_conversion<true, fp8_type>(out_norm, scale_inv);
  }
}

}  // namespace vllm

void rms_norm_static_fp8_quant(torch::Tensor& out,     // [..., hidden_size]
                               torch::Tensor& input,   // [..., hidden_size]
                               torch::Tensor& weight,  // [hidden_size]
                               torch::Tensor& scale,   // [1]
                               double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  int hidden_size = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  dim3 block(std::min(hidden_size, 1024));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_kernel_scalar_type", [&] {
        VLLM_DISPATCH_FP8_TYPES(
            out.scalar_type(), "rms_norm_kernel_fp8_type", [&] {
              vllm::rms_norm_static_fp8_quant_kernel<scalar_t, fp8_t>
                  <<<grid, block, 0, stream>>>(
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),
                      input_stride, weight.data_ptr<scalar_t>(),
                      scale.data_ptr<float>(), epsilon, num_tokens,
                      hidden_size);
            });
      });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                     \
  VLLM_DISPATCH_FLOATING_TYPES(                                              \
      input.scalar_type(), "fused_add_rms_norm_kernel_scalar_type", [&] {    \
        VLLM_DISPATCH_FP8_TYPES(                                             \
            out.scalar_type(), "fused_add_rms_norm_kernel_fp8_type", [&] {   \
              vllm::fused_add_rms_norm_static_fp8_quant_kernel<scalar_t,     \
                                                               width, fp8_t> \
                  <<<grid, block, 0, stream>>>(                              \
                      out.data_ptr<fp8_t>(), input.data_ptr<scalar_t>(),     \
                      input_stride, residual.data_ptr<scalar_t>(),           \
                      weight.data_ptr<scalar_t>(), scale.data_ptr<float>(),  \
                      epsilon, num_tokens, hidden_size);                     \
            });                                                              \
      });
void fused_add_rms_norm_static_fp8_quant(
    torch::Tensor& out,       // [..., hidden_size],
    torch::Tensor& input,     // [..., hidden_size]
    torch::Tensor& residual,  // [..., hidden_size]
    torch::Tensor& weight,    // [hidden_size]
    torch::Tensor& scale,     // [1]
    double epsilon) {
  TORCH_CHECK(out.is_contiguous());
  TORCH_CHECK(residual.is_contiguous());
  int hidden_size = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  dim3 grid(num_tokens);
  /* This kernel is memory-latency bound in many scenarios.
     When num_tokens is large, a smaller block size allows
     for increased block occupancy on CUs and better latency
     hiding on global mem ops. */
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 block(std::min(hidden_size, max_block_size));
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  if (ptrs_are_aligned && hidden_size % 8 == 0 && input_stride % 8 == 0) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}
