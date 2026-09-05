// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include "../../../cub_helpers.h"
#include "../../dispatch_utils.h"
#include "../vectorization.cuh"
#include "quant_conversions.cuh"

namespace vllm {

static constexpr int BLOCK_SIZE = 256;
static constexpr int VEC_SIZE = 16;

template <typename scalar_t, typename scalar_out_t>
__global__ void silu_and_mul_per_token_quant_kernel(
    scalar_out_t* __restrict__ out,      // [num_tokens, d]
    float* __restrict__ scale,           // [num_tokens, 1]
    scalar_t const* __restrict__ input,  // [num_tokens, 2 * d]
    float const* __restrict__ scale_ub,  // optional
    int32_t const d) {
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
      float gf = static_cast<float>(gv.val[j]);
      float uf = static_cast<float>(uv.val[j]);
      float val = gf / (1.0f + expf(-gf)) * uf;
      thread_max = fmaxf(thread_max, fabsf(val));
    }
  }
  // Scalar tail
  for (int i = num_vecs * VEC_SIZE + tid; i < d; i += BLOCK_SIZE) {
    float gf = static_cast<float>(gate_ptr[i]);
    float uf = static_cast<float>(up_ptr[i]);
    float val = gf / (1.0f + expf(-gf)) * uf;
    thread_max = fmaxf(thread_max, fabsf(val));
  }

  using BlockReduce = cub::BlockReduce<float, BLOCK_SIZE>;
  __shared__ typename BlockReduce::TempStorage tmp;
  float const block_max =
      BlockReduce(tmp).Reduce(thread_max, CubMaxOp{}, blockDim.x);

  __shared__ float token_scale;
  if (tid == 0) {
    float max_val = scale_ub ? fminf(block_max, *scale_ub) : block_max;
    max_val = fmaxf(max_val / quant_type_max_v<scalar_out_t>,
                    min_scaling_factor<scalar_out_t>::val());
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
      float gf = static_cast<float>(gv.val[j]);
      float uf = static_cast<float>(uv.val[j]);
      float val = gf / (1.0f + expf(-gf)) * uf;
      ov.val[j] = ScaledQuant<scalar_out_t, false>::quant_fn(val, token_scale);
    }
    out_vecs[vi] = ov;
  }
  // Scalar tail
  for (int i = num_vecs * VEC_SIZE + tid; i < d; i += BLOCK_SIZE) {
    float gf = static_cast<float>(gate_ptr[i]);
    float uf = static_cast<float>(up_ptr[i]);
    float val = gf / (1.0f + expf(-gf)) * uf;
    out_ptr[i] = ScaledQuant<scalar_out_t, false>::quant_fn(val, token_scale);
  }
}

}  // namespace vllm

void silu_and_mul_per_token_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor& scale,
    std::optional<torch::stable::Tensor> scale_ub) {
  static torch::headeronly::ScalarType kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;

  STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  STD_TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Input must be FP16 or BF16");
  STD_TORCH_CHECK(scale.scalar_type() == torch::headeronly::ScalarType::Float);

  int32_t d = out.size(-1);
  auto num_tokens = input.numel() / input.size(-1);

  STD_TORCH_CHECK(input.size(-1) == d * 2,
                  "input last dim must be 2x output hidden_size");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  dim3 grid(num_tokens);
  dim3 block(vllm::BLOCK_SIZE);

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_token_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_STABLE_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_token_quant", [&] {
              using scalar_out_t = scalar_t;

              vllm::silu_and_mul_per_token_quant_kernel<scalar_in_t,
                                                        scalar_out_t>
                  <<<grid, block, 0, stream>>>(
                      out.mutable_data_ptr<scalar_out_t>(),
                      scale.mutable_data_ptr<float>(),
                      input.const_data_ptr<scalar_in_t>(),
                      scale_ub.has_value() ? scale_ub->const_data_ptr<float>()
                                           : nullptr,
                      d);
            });
      });
}
