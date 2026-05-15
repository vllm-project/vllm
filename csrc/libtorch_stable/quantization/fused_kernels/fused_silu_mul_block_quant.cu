// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cuda_runtime.h>
#include <optional>

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>

#include "libtorch_stable/dispatch_utils.h"
#include "libtorch_stable/torch_utils.h"
#include "quantization/fused_kernels/quant_conversions.cuh"
#include "quantization/w8a8/fp8/common.cuh"

namespace vllm {

template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ input, float const* scale_ub,
    int32_t const hidden_size) {
  static_assert((group_size & (group_size - 1)) == 0,
                "group_size must be a power of 2 for correct reduction");

  int const token_idx = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const tid = threadIdx.x;
  int const num_tokens = gridDim.x;

  int const input_stride = hidden_size * 2;
  int const group_start = group_idx * group_size;

  scalar_t const* token_input_gate =
      input + token_idx * input_stride + group_start;
  scalar_t const* token_input_up = token_input_gate + hidden_size;
  scalar_out_t* token_output = out + token_idx * hidden_size + group_start;

  int const num_groups = gridDim.y;
  float* group_scale_ptr = is_scale_transposed
                               ? scales + group_idx * num_tokens + token_idx
                               : scales + token_idx * num_groups + group_idx;

  __shared__ float shared_max[group_size];

  float gate = static_cast<float>(token_input_gate[tid]);
  float up = static_cast<float>(token_input_up[tid]);

  float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
  float silu_gate = gate * sigmoid_gate;
  float result = silu_gate * up;

  shared_max[tid] = fabsf(result);
  __syncthreads();

#pragma unroll
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  if (tid == 0) {
    float group_max = shared_max[0];

    float const quant_range = quant_type_max_v<scalar_out_t>;
    float group_scale = group_max / quant_range;

    if (scale_ub != nullptr) {
      group_scale = fminf(group_scale, *scale_ub);
    }

    group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());

    *group_scale_ptr = group_scale;

    shared_max[0] = group_scale;
  }
  __syncthreads();

  float group_scale = shared_max[0];

  token_output[tid] =
      vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale);
}

}  // namespace vllm

namespace {

template <typename scalar_in_t, typename scalar_out_t, bool transpose_scale,
          int gs>
void launch_silu_and_mul_per_block_quant(
    int64_t num_tokens, int32_t num_groups, int64_t group_size,
    int32_t hidden_size, cudaStream_t stream, torch::stable::Tensor& out,
    torch::stable::Tensor const& input, torch::stable::Tensor& scales,
    std::optional<torch::stable::Tensor> const& scale_ub) {
  dim3 const grid(static_cast<unsigned int>(num_tokens),
                  static_cast<unsigned int>(num_groups));
  dim3 const block(static_cast<unsigned int>(group_size));
  vllm::silu_and_mul_per_block_quant_kernel<scalar_in_t, scalar_out_t,
                                            transpose_scale, gs>
      <<<grid, block, 0, stream>>>(
          out.mutable_data_ptr<scalar_out_t>(),
          scales.mutable_data_ptr<float>(), input.const_data_ptr<scalar_in_t>(),
          scale_ub.has_value() ? scale_ub->const_data_ptr<float>() : nullptr,
          hidden_size);
}

template <typename scalar_in_t>
void silu_and_mul_per_block_quant_dispatch(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor& scales, int64_t group_size,
    std::optional<torch::stable::Tensor> const& scale_ub,
    bool is_scale_transposed, int64_t num_tokens, int32_t num_groups,
    int32_t hidden_size, cudaStream_t stream) {
  auto const od = out.scalar_type();
  VLLM_STABLE_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
    VLLM_STABLE_DISPATCH_BOOL(is_scale_transposed, transpose_scale, [&] {
      if (od == torch::headeronly::ScalarType::Float8_e4m3fn) {
        launch_silu_and_mul_per_block_quant<scalar_in_t, c10::Float8_e4m3fn,
                                            transpose_scale, gs>(
            num_tokens, num_groups, group_size, hidden_size, stream, out, input,
            scales, scale_ub);
      } else if (od == torch::headeronly::ScalarType::Float8_e4m3fnuz) {
        launch_silu_and_mul_per_block_quant<scalar_in_t, c10::Float8_e4m3fnuz,
                                            transpose_scale, gs>(
            num_tokens, num_groups, group_size, hidden_size, stream, out, input,
            scales, scale_ub);
      } else if (od == torch::headeronly::ScalarType::Char) {
        launch_silu_and_mul_per_block_quant<scalar_in_t, int8_t,
                                            transpose_scale, gs>(
            num_tokens, num_groups, group_size, hidden_size, stream, out, input,
            scales, scale_ub);
      } else {
        STD_TORCH_CHECK(false, "silu_and_mul_per_block_quant: bad out dtype");
      }
    });
  });
}

}  // namespace

void silu_and_mul_per_block_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor& scales, int64_t group_size,
    std::optional<torch::stable::Tensor> const& scale_ub,
    bool is_scale_transposed) {
  torch::headeronly::ScalarType const kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;

  STD_TORCH_CHECK(out.scalar_type() == kFp8Type ||
                  out.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Input must be FP16 or BF16");
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float,
                  "Scales must be FP32");
  STD_TORCH_CHECK(group_size == 128 || group_size == 64,
                  "Unsupported group size: ", group_size);

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  }

  int32_t const hidden_size = static_cast<int32_t>(out.size(-1));
  auto const num_tokens = input.size(0);
  int32_t const num_groups =
      hidden_size / static_cast<int32_t>(group_size);

  STD_TORCH_CHECK(input.size(-1) == hidden_size * 2,
                  "input last dim must be 2x output hidden_size");
  STD_TORCH_CHECK(hidden_size % group_size == 0,
                  "hidden_size must be divisible by group_size");

  torch::stable::accelerator::DeviceGuard const device_guard(
      input.get_device_index());
  cudaStream_t const stream = get_current_cuda_stream();

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        silu_and_mul_per_block_quant_dispatch<scalar_t>(
            out, input, scales, group_size, scale_ub, is_scale_transposed,
            num_tokens, num_groups, hidden_size, stream);
      });
}
