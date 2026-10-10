// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include "../../dispatch_utils.h"
#include "quant_conversions.cuh"

// TODO look at other shapes
constexpr int MAX_BLOCK_SIZE = 1024;

namespace vllm {

// Logic: one thread block per (token, group) pair

template <typename scalar_t, typename scalar_out_t>
__global__ void silu_and_mul_per_token_quant_kernel(
    scalar_out_t* __restrict__ out,      // Output: [num_tokens, hidden_size] in
                                         // FP8/INT8
    float* __restrict__ scales,          // Output: [num_tokens]
    scalar_t const* __restrict__ input,  // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,               // Optional scale upper bound
    int32_t const hidden_size,   // Output hidden size (input is 2x this)
    float* intermediate_results  // Temp storage: [num_tokens, hidden_size]
) {
  // Grid: (num_tokens)
  int64_t const token_idx = blockIdx.x;
  int const tid = threadIdx.x;
  int const block_size = blockDim.x;

  // Input layout: [gate || up] concatenated along last dimension
  int const input_stride = hidden_size * 2;

  // Pointers to this token's data
  scalar_t const* token_input_gate = input + token_idx * input_stride;
  scalar_t const* token_input_up = token_input_gate + hidden_size;
  scalar_out_t* token_output = out + token_idx * hidden_size;
  float* token_intermediate_results =
      intermediate_results + token_idx * hidden_size;

  float* block_scale_ptr = scales + token_idx;

  // Shared memory for reduction (compile-time sized)
  __shared__ float shared_max[MAX_BLOCK_SIZE];

  // Step 1: Each thread loads one element, computes SiLU, stores in register

  float local_max = 0.0f;

  for (int idx = tid; idx < hidden_size; idx += block_size) {
    float gate = static_cast<float>(token_input_gate[idx]);
    float up = static_cast<float>(token_input_up[idx]);

    // Compute SiLU(gate) * up
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    float silu_gate = gate * sigmoid_gate;
    float result = silu_gate * up;
    local_max = fmaxf(local_max, fabsf(result));
    token_intermediate_results[idx] = result;  // Keep in memory
  }

  // Step 2: Reduce to find max per token
  shared_max[tid] = local_max;
  __syncthreads();

// Power-of-2 reduction (group_size guaranteed to be power of 2)
#pragma unroll
  for (int stride = block_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  // Step 3: Compute scale (thread 0), broadcast via shared memory
  if (tid == 0) {
    float block_max = shared_max[0];

    float const quant_range = quant_type_max_v<scalar_out_t>;
    float block_scale = block_max / quant_range;

    // Apply scale upper bound if provided
    if (scale_ub != nullptr) {
      block_scale = fminf(block_scale, *scale_ub);
    }

    // Use minimum safe scaling factor
    block_scale = fmaxf(block_scale, min_scaling_factor<scalar_out_t>::val());

    // Store scale to global memory
    *block_scale_ptr = block_scale;

    // Reuse shared_max[0] to broadcast scale
    shared_max[0] = block_scale;
  }
  __syncthreads();

  float block_scale = shared_max[0];

  for (int idx = tid; idx < hidden_size; idx += block_size) {
    // Step 4: Quantize and write output
    token_output[idx] = vllm::ScaledQuant<scalar_out_t, false>::quant_fn(
        token_intermediate_results[idx], block_scale);
  }
}

}  // namespace vllm

void silu_and_mul_per_token_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor& scales,
    std::optional<torch::stable::Tensor> scale_ub) {
  static torch::headeronly::ScalarType kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;

  STD_TORCH_CHECK(out.scalar_type() == kFp8Type ||
                  out.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Input must be FP16 or BF16");
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float);

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  }

  int32_t hidden_size = out.size(-1);
  auto num_tokens = input.size(0);
  auto intermediate_results =
      torch::stable::empty(input.sizes(), torch::headeronly::ScalarType::Float,
                           input.layout(), input.device());

  STD_TORCH_CHECK(input.size(-1) == hidden_size * 2,
                  "input last dim must be 2x output hidden_size");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  int block_size_candidate = min(hidden_size, MAX_BLOCK_SIZE);
  if (block_size_candidate != MAX_BLOCK_SIZE) {
    // Make sure block_size_candidate is a power of 2
    int power_of_2 = 1;
    while (power_of_2 <= block_size_candidate) {
      power_of_2 <<= 1;
    }
    block_size_candidate = (power_of_2 >> 1);
  }
  dim3 grid(num_tokens);
  dim3 block(block_size_candidate);

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_STABLE_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant", [&] {
              using scalar_out_t = scalar_t;

              vllm::silu_and_mul_per_token_quant_kernel<scalar_in_t,
                                                        scalar_out_t>
                  <<<grid, block, 0, stream>>>(
                      out.mutable_data_ptr<scalar_out_t>(),
                      scales.mutable_data_ptr<float>(),
                      input.const_data_ptr<scalar_in_t>(),
                      scale_ub.has_value() ? scale_ub->const_data_ptr<float>()
                                           : nullptr,
                      hidden_size,
                      intermediate_results.mutable_data_ptr<float>());
            });
      });
}
