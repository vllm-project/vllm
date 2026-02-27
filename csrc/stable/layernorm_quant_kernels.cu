/*
 * This file contains the CUDA kernels for the fused quantized layernorm.
 * The kernels correspond to the kernels in layernorm_kernels.cu, except they
 * also produce quantized output directly.
 * Currently, only static fp8 quantization is supported.
 */

#include <torch/csrc/stable/tensor.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/macros/Macros.h>

#include <numeric>

#include "type_convert.cuh"
#include "quantization/w8a8/fp8/common.cuh"
#include "cub_helpers.h"
#include "core/batch_invariant.hpp"
#include "quantization/vectorization_utils.cuh"
#include "dispatch_utils.h"
#include "torch_utils.h"

namespace vllm {

// TODO(woosuk): Further optimize this kernel.
template <typename scalar_t, typename fp8_type, int VEC_SIZE>
__global__ void rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,          // [..., hidden_size]
    const scalar_t* __restrict__ input,  // [..., hidden_size]
    const int input_stride,
    const scalar_t* __restrict__ weight,  // [hidden_size]
    const float* __restrict__ scale,      // [1]
    const float epsilon, const int num_tokens, const int hidden_size) {
  __shared__ float s_variance;
  float variance = 0.0f;

  const scalar_t* input_row = input + blockIdx.x * input_stride;

  auto vec_op = [&variance](const vec_n_t<scalar_t, VEC_SIZE>& vec) {
#pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
      float x = static_cast<float>(vec.val[i]);
      variance += x * x;
    }
  };
  auto scalar_op = [&variance](const scalar_t& val) {
    float x = static_cast<float>(val);
    variance += x * x;
  };
  vllm::vectorize_read_with_alignment<VEC_SIZE>(
      input_row, hidden_size, threadIdx.x, blockDim.x, vec_op, scalar_op);

  using BlockReduce = cub::BlockReduce<float, 1024>;
  __shared__ typename BlockReduce::TempStorage reduceStore;
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

  if (threadIdx.x == 0) {
    s_variance = rsqrtf(variance / hidden_size + epsilon);
  }
  __syncthreads();

  // invert scale to avoid division
  float const scale_inv = 1.0f / *scale;

  auto* v_in = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(input_row);
  auto* v_w = reinterpret_cast<const vec_n_t<scalar_t, VEC_SIZE>*>(weight);
  for (int idx = threadIdx.x; idx < hidden_size / VEC_SIZE; idx += blockDim.x) {
    vec_n_t<scalar_t, VEC_SIZE> src1 = v_in[idx];
    vec_n_t<scalar_t, VEC_SIZE> src2 = v_w[idx];
#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float x = static_cast<float>(src1.val[j]);
      float const out_norm = ((scalar_t)(x * s_variance)) * src2.val[j];
      out[blockIdx.x * hidden_size + idx * VEC_SIZE + j] =
          scaled_fp8_conversion<true, fp8_type>(out_norm, scale_inv);
    }
  }
}

/* Function specialization in the case of FP16/BF16 tensors.
   Additional optimizations we can make in this case are
   packed and vectorized operations, which help with the
   memory latency bottleneck. */
template <typename scalar_t, int width, typename fp8_type>
__global__ std::enable_if_t<(width > 0) && _typeConvert<scalar_t>::exists>
fused_add_rms_norm_static_fp8_quant_kernel(
    fp8_type* __restrict__ out,          // [..., hidden_size]
    const scalar_t* __restrict__ input,  // [..., hidden_size]
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
      reinterpret_cast<const _f16Vec<scalar_t, width>*>(input);
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
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

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
    fp8_type* __restrict__ out,          // [..., hidden_size]
    const scalar_t* __restrict__ input,  // [..., hidden_size]
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
  variance = BlockReduce(reduceStore).Reduce(variance, CubAddOp{}, blockDim.x);

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

void rms_norm_static_fp8_quant(
    torch::stable::Tensor& out,           // [..., hidden_size]
    const torch::stable::Tensor& input,   // [..., hidden_size]
    const torch::stable::Tensor& weight,  // [hidden_size]
    const torch::stable::Tensor& scale,   // [1]
    double epsilon) {
  STD_TORCH_CHECK(out.is_contiguous());
  int hidden_size = input.size(-1);
  int input_stride = input.stride(-2);
  int num_tokens = input.numel() / hidden_size;

  // For large num_tokens, use smaller blocks to increase SM concurrency.
  const int max_block_size = (num_tokens < 256) ? 1024 : 256;
  dim3 grid(num_tokens);
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "rms_norm_kernel_scalar_type", [&] {
        VLLM_STABLE_DISPATCH_FP8_TYPES(
            out.scalar_type(), "rms_norm_kernel_fp8_type", [&] {
              const int calculated_vec_size =
                  std::gcd(16 / sizeof(scalar_t), hidden_size);
              const int block_size =
                  std::min(hidden_size / calculated_vec_size, max_block_size);
              dim3 block(block_size);
              VLLM_STABLE_DISPATCH_VEC_SIZE(calculated_vec_size, [&] {
                vllm::rms_norm_static_fp8_quant_kernel<scalar_t, fp8_t,
                                                       vec_size>
                    <<<grid, block, 0, stream>>>(
                        out.mutable_data_ptr<fp8_t>(),
                        input.const_data_ptr<scalar_t>(), input_stride,
                        weight.const_data_ptr<scalar_t>(),
                        scale.const_data_ptr<float>(), epsilon, num_tokens,
                        hidden_size);
              });
            });
      });
}

#define LAUNCH_FUSED_ADD_RMS_NORM(width)                                     \
  VLLM_STABLE_DISPATCH_FLOATING_TYPES(                                       \
      input.scalar_type(), "fused_add_rms_norm_kernel_scalar_type", [&] {    \
        VLLM_STABLE_DISPATCH_FP8_TYPES(                                      \
            out.scalar_type(), "fused_add_rms_norm_kernel_fp8_type", [&] {   \
              vllm::fused_add_rms_norm_static_fp8_quant_kernel<scalar_t,     \
                                                               width, fp8_t> \
                  <<<grid, block, 0, stream>>>(                              \
                      out.mutable_data_ptr<fp8_t>(),                         \
                      input.const_data_ptr<scalar_t>(), input_stride,        \
                      residual.mutable_data_ptr<scalar_t>(),                 \
                      weight.const_data_ptr<scalar_t>(),                     \
                      scale.const_data_ptr<float>(),                         \
                      static_cast<float>(epsilon), num_tokens, hidden_size); \
            });                                                              \
      });

void fused_add_rms_norm_static_fp8_quant(
    torch::stable::Tensor& out,           // [..., hidden_size],
    const torch::stable::Tensor& input,   // [..., hidden_size]
    torch::stable::Tensor& residual,      // [..., hidden_size]
    const torch::stable::Tensor& weight,  // [hidden_size]
    const torch::stable::Tensor& scale,   // [1]
    double epsilon) {
  STD_TORCH_CHECK(out.is_contiguous());
  STD_TORCH_CHECK(residual.is_contiguous());
  STD_TORCH_CHECK(residual.scalar_type() == input.scalar_type());
  STD_TORCH_CHECK(weight.scalar_type() == input.scalar_type());
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
  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device());
  /*If the tensor types are FP16/BF16, try to use the optimized kernel
    with packed + vectorized ops.
    Max optimization is achieved with a width-8 vector of FP16/BF16s
    since we can load at most 128 bits at once in a global memory op.
    However, this requires each tensor's data to be aligned to 16
    bytes.
   */
  auto inp_ptr = reinterpret_cast<std::uintptr_t>(input.const_data_ptr());
  auto res_ptr = reinterpret_cast<std::uintptr_t>(residual.mutable_data_ptr());
  auto wt_ptr = reinterpret_cast<std::uintptr_t>(weight.const_data_ptr());
  bool ptrs_are_aligned =
      inp_ptr % 16 == 0 && res_ptr % 16 == 0 && wt_ptr % 16 == 0;
  bool batch_invariant_launch = vllm::vllm_is_batch_invariant();
  if (ptrs_are_aligned && hidden_size % 8 == 0 && input_stride % 8 == 0 &&
      !batch_invariant_launch) {
    LAUNCH_FUSED_ADD_RMS_NORM(8);
  } else {
    LAUNCH_FUSED_ADD_RMS_NORM(0);
  }
}
