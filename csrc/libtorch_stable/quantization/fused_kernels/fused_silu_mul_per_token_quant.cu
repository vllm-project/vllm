// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include "../../../cub_helpers.h"
#include "../../dispatch_utils.h"
#include "../vectorization.cuh"
#include "quant_conversions.cuh"

#include <cmath>
#include <limits>

namespace vllm {

static constexpr int BLOCK_SIZE = 256;
static constexpr int VEC_SIZE = 16;

template <typename scalar_t, bool has_clamp, bool round_activation>
__device__ __forceinline__ float silu_and_mul_value(scalar_t gate, scalar_t up,
                                                    float swiglu_limit) {
  if constexpr (has_clamp) {
    const float gate_float = static_cast<float>(gate);
    const float up_float = static_cast<float>(up);
    gate = static_cast<scalar_t>(gate_float > swiglu_limit ? swiglu_limit
                                                           : gate_float);
    up = static_cast<scalar_t>(
        up_float > swiglu_limit
            ? swiglu_limit
            : (up_float < -swiglu_limit ? -swiglu_limit : up_float));
  }

  const float gate_float = static_cast<float>(gate);
  const float up_float = static_cast<float>(up);
  if constexpr (round_activation) {
    const scalar_t activated =
        static_cast<scalar_t>(gate_float / (1.0f + expf(-gate_float)));
    return static_cast<float>(
        static_cast<scalar_t>(static_cast<float>(activated) * up_float));
  }
  return gate_float / (1.0f + expf(-gate_float)) * up_float;
}

template <typename scalar_t, typename scalar_out_t, bool has_clamp,
          bool round_activation>
__global__ void silu_and_mul_per_token_quant_kernel(
    scalar_out_t* __restrict__ out,      // [num_tokens, d]
    float* __restrict__ scale,           // [num_tokens, 1]
    scalar_t const* __restrict__ input,  // [num_tokens, 2 * d]
    float const* __restrict__ scale_ub,  // optional
    int32_t const d, float swiglu_limit, float absmax_floor) {
  using in_vec_t = vec_n_t<scalar_t, VEC_SIZE>;
  using out_vec_t = vec_n_t<scalar_out_t, VEC_SIZE>;

  int64_t const token_idx = blockIdx.x;
  int const tid = threadIdx.x;

  scalar_t const* gate_ptr = input + token_idx * 2 * d;
  scalar_t const* up_ptr = gate_ptr + d;
  scalar_out_t* out_ptr = out + token_idx * d;

  int const num_vecs = d / VEC_SIZE;
  in_vec_t const* gate_vecs = reinterpret_cast<in_vec_t const*>(gate_ptr);
  in_vec_t const* up_vecs = reinterpret_cast<in_vec_t const*>(up_ptr);

  // Pass 1: vectorized silu(gate)*up, accumulate absmax
  float thread_max = 0.0f;
  for (int vi = tid; vi < num_vecs; vi += BLOCK_SIZE) {
    in_vec_t gv = gate_vecs[vi];
    in_vec_t uv = up_vecs[vi];

#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float val = silu_and_mul_value<scalar_t, has_clamp, round_activation>(
          gv.val[j], uv.val[j], swiglu_limit);
      thread_max = fmaxf(thread_max, fabsf(val));
    }
  }
  // Scalar tail
  for (int i = num_vecs * VEC_SIZE + tid; i < d; i += BLOCK_SIZE) {
    float val = silu_and_mul_value<scalar_t, has_clamp, round_activation>(
        gate_ptr[i], up_ptr[i], swiglu_limit);
    thread_max = fmaxf(thread_max, fabsf(val));
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float const block_max =
      BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

  __shared__ float token_scale;
  if (tid == 0) {
    float max_val = scale_ub ? fminf(block_max, *scale_ub) : block_max;
    if (absmax_floor > 0.0f) {
      max_val = fmaxf(max_val, absmax_floor) / quant_type_max_v<scalar_out_t>;
    } else {
      max_val = fmaxf(max_val / quant_type_max_v<scalar_out_t>,
                      min_scaling_factor<scalar_out_t>::val());
    }
    scale[token_idx] = max_val;
    token_scale = max_val;
  }
  __syncthreads();

  // Pass 2: vectorized recompute + quantize
  out_vec_t* out_vecs = reinterpret_cast<out_vec_t*>(out_ptr);

  for (int vi = tid; vi < num_vecs; vi += BLOCK_SIZE) {
    in_vec_t gv = gate_vecs[vi];
    in_vec_t uv = up_vecs[vi];
    out_vec_t ov;

#pragma unroll
    for (int j = 0; j < VEC_SIZE; j++) {
      float val = silu_and_mul_value<scalar_t, has_clamp, round_activation>(
          gv.val[j], uv.val[j], swiglu_limit);
      ov.val[j] = ScaledQuant<scalar_out_t, false>::quant_fn(val, token_scale);
    }
    out_vecs[vi] = ov;
  }
  // Scalar tail
  for (int i = num_vecs * VEC_SIZE + tid; i < d; i += BLOCK_SIZE) {
    float val = silu_and_mul_value<scalar_t, has_clamp, round_activation>(
        gate_ptr[i], up_ptr[i], swiglu_limit);
    out_ptr[i] = ScaledQuant<scalar_out_t, false>::quant_fn(val, token_scale);
  }
}

template <typename scalar_t, typename scalar_out_t>
void launch_silu_and_mul_per_token_quant(scalar_out_t* out, float* scale,
                                         scalar_t const* input,
                                         float const* scale_ub, int32_t d,
                                         float swiglu_limit, float absmax_floor,
                                         bool has_clamp, bool round_activation,
                                         dim3 grid, dim3 block,
                                         cudaStream_t stream) {
  if (has_clamp && round_activation) {
    silu_and_mul_per_token_quant_kernel<scalar_t, scalar_out_t, true, true>
        <<<grid, block, 0, stream>>>(out, scale, input, scale_ub, d,
                                     swiglu_limit, absmax_floor);
  } else if (has_clamp) {
    silu_and_mul_per_token_quant_kernel<scalar_t, scalar_out_t, true, false>
        <<<grid, block, 0, stream>>>(out, scale, input, scale_ub, d,
                                     swiglu_limit, absmax_floor);
  } else if (round_activation) {
    silu_and_mul_per_token_quant_kernel<scalar_t, scalar_out_t, false, true>
        <<<grid, block, 0, stream>>>(out, scale, input, scale_ub, d,
                                     swiglu_limit, absmax_floor);
  } else {
    silu_and_mul_per_token_quant_kernel<scalar_t, scalar_out_t, false, false>
        <<<grid, block, 0, stream>>>(out, scale, input, scale_ub, d,
                                     swiglu_limit, absmax_floor);
  }
}

}  // namespace vllm

void silu_and_mul_per_token_quant(torch::stable::Tensor& out,
                                  torch::stable::Tensor const& input,
                                  torch::stable::Tensor& scale,
                                  std::optional<torch::stable::Tensor> scale_ub,
                                  std::optional<double> swiglu_limit,
                                  std::optional<double> absmax_floor,
                                  bool round_activation_to_input_dtype) {
  static torch::headeronly::ScalarType kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;

  STD_TORCH_CHECK(out.scalar_type() == kFp8Type, "output must be FP8 E4M3");
  STD_TORCH_CHECK(input.dim() == 2 && out.dim() == 2,
                  "input and output must be 2D");
  STD_TORCH_CHECK(
      out.is_contiguous() && input.is_contiguous() && scale.is_contiguous(),
      "input, output, and scale must be contiguous");
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Input must be FP16 or BF16");
  STD_TORCH_CHECK(scale.scalar_type() == torch::headeronly::ScalarType::Float,
                  "scale must be FP32");
  STD_TORCH_CHECK(out.get_device_index() == input.get_device_index() &&
                      scale.get_device_index() == input.get_device_index(),
                  "input, output, and scale must be on the same device");
  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(
        scale_ub->scalar_type() == torch::headeronly::ScalarType::Float,
        "scale_ub must be FP32");
    STD_TORCH_CHECK(scale_ub->is_contiguous(), "scale_ub must be contiguous");
    STD_TORCH_CHECK(scale_ub->numel() == 1, "scale_ub must contain one value");
    STD_TORCH_CHECK(scale_ub->get_device_index() == input.get_device_index(),
                    "scale_ub and input must be on the same device");
  }
  if (swiglu_limit.has_value()) {
    STD_TORCH_CHECK(std::isfinite(*swiglu_limit),
                    "swiglu_limit must be finite");
  }
  if (absmax_floor.has_value()) {
    STD_TORCH_CHECK(std::isfinite(*absmax_floor) && *absmax_floor > 0.0,
                    "absmax_floor must be finite and positive");
  }

  const int64_t d64 = out.size(1);
  const int64_t num_tokens = input.size(0);

  STD_TORCH_CHECK(input.size(1) == d64 * 2,
                  "input last dim must be 2x output hidden_size");
  STD_TORCH_CHECK(out.size(0) == num_tokens,
                  "input and output token dimensions must match");
  STD_TORCH_CHECK(
      scale.dim() == 2 && scale.size(0) == num_tokens && scale.size(1) == 1,
      "scale must have shape [num_tokens, 1]");
  STD_TORCH_CHECK(d64 <= std::numeric_limits<int32_t>::max(),
                  "hidden size exceeds int32 range");
  STD_TORCH_CHECK(num_tokens <= std::numeric_limits<int32_t>::max(),
                  "token count exceeds CUDA grid range");

  if (num_tokens == 0) {
    return;
  }

  const int32_t d = static_cast<int32_t>(d64);
  const bool has_clamp = swiglu_limit.has_value() && *swiglu_limit > 0.0;
  const float swiglu_limit_value =
      has_clamp ? static_cast<float>(*swiglu_limit) : 0.0f;
  const float absmax_floor_value =
      absmax_floor.has_value() ? static_cast<float>(*absmax_floor) : 0.0f;

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  dim3 grid(static_cast<unsigned int>(num_tokens));
  dim3 block(vllm::BLOCK_SIZE);

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_token_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_STABLE_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_token_quant", [&] {
              using scalar_out_t = scalar_t;

              vllm::launch_silu_and_mul_per_token_quant<scalar_in_t,
                                                        scalar_out_t>(
                  out.mutable_data_ptr<scalar_out_t>(),
                  scale.mutable_data_ptr<float>(),
                  input.const_data_ptr<scalar_in_t>(),
                  scale_ub.has_value() ? scale_ub->const_data_ptr<float>()
                                       : nullptr,
                  d, swiglu_limit_value, absmax_floor_value, has_clamp,
                  round_activation_to_input_dtype, grid, block, stream);
            });
      });
}
