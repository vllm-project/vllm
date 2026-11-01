// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include <cmath>

#include "../../cub_helpers.h"
#include "../../cuda_vec_utils.cuh"
#include "../../dispatch_utils.h"
#include "../vectorization.cuh"
#include "../w8a8/fp8/common.cuh"

namespace vllm {

namespace {

// A row is kept in shared memory between the two passes when it fits in the
// default 48 KB budget (less room for the reduction storage); wider rows are
// recomputed from the input instead.
constexpr size_t kMaxCachedRowBytes = 47 * 1024;

// gelu_tanh(gate) * up, computed and rounded exactly like gelu_tanh_and_mul:
// the GELU expression is the one in gelu_tanh_kernel
// (csrc/activation_kernels.cu), its result is rounded to the input dtype,
// multiplied by (up + beta) in fp32 with beta = 0 as in act_and_mul_kernel, and
// rounded again.
template <typename scalar_t>
__device__ __forceinline__ scalar_t gelu_tanh_and_mul(scalar_t const gate,
                                                      scalar_t const up) {
  constexpr float BETA = M_SQRT2 * M_2_SQRTPI * 0.5f;
  constexpr float KAPPA = 0.044715;
  float const f = static_cast<float>(gate);
  float x_cube = f * f * f;
  float inner = BETA * (f + KAPPA * x_cube);
  scalar_t const act =
      static_cast<scalar_t>(0.5f * f * (1.0f + ::tanhf(inner)));
  return static_cast<scalar_t>(static_cast<float>(act) *
                               (static_cast<float>(up) + 0.0f));
}

// One thread block per token. Pass 1 computes gelu_tanh(gate) * up, keeps it in
// shared memory and reduces its absmax; pass 2 quantizes it with the token's
// scale. The input is read once and the FP8 output written once.
template <typename scalar_t, typename fp8_type, int kBlockSize, int kVecSize,
          bool kCacheRow>
__global__ void gelu_tanh_and_mul_per_token_quant_kernel(
    fp8_type* __restrict__ out,          // [num_tokens, d]
    float* __restrict__ scales,          // [num_tokens]
    scalar_t const* __restrict__ input,  // [num_tokens, 2 * d]
    float const* __restrict__ scale_ub, int32_t const d) {
  using vec_t = vec_n_t<scalar_t, kVecSize>;
  using out_vec_t = q8_n_t<fp8_type, kVecSize>;

  extern __shared__ __align__(16) unsigned char smem[];
  vec_t* row = reinterpret_cast<vec_t*>(smem);

  int64_t const token_idx = blockIdx.x;
  vec_t const* gate = reinterpret_cast<vec_t const*>(input + token_idx * 2 * d);
  vec_t const* up =
      reinterpret_cast<vec_t const*>(input + token_idx * 2 * d + d);
  out_vec_t* token_out = reinterpret_cast<out_vec_t*>(out + token_idx * d);
  int32_t const num_vecs = d / kVecSize;

  float absmax = 0.0f;
  for (int32_t i = threadIdx.x; i < num_vecs; i += kBlockSize) {
    vec_t const g = gate[i];
    vec_t const u = up[i];
    vec_t a;
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      a.val[j] = gelu_tanh_and_mul(g.val[j], u.val[j]);
      absmax = fmaxf(absmax, fabsf(static_cast<float>(a.val[j])));
    }
    if constexpr (kCacheRow) {
      row[i] = a;
    }
  }

  using BlockReduce = cub::BlockReduce<float, kBlockSize>;
  __shared__ typename BlockReduce::TempStorage reduce_storage;
  float const row_max = BlockReduce(reduce_storage).Reduce(absmax, CubMaxOp{});

  // Same scale as dynamic_per_token_scaled_fp8_quant.
  __shared__ float token_scale;
  if (threadIdx.x == 0) {
    float const bounded = scale_ub ? fminf(row_max, *scale_ub) : row_max;
    token_scale = fmaxf(bounded / quant_type_max_v<fp8_type>,
                        min_scaling_factor<fp8_type>::val());
    scales[token_idx] = token_scale;
  }
  __syncthreads();
  float const scale = token_scale;

  for (int32_t i = threadIdx.x; i < num_vecs; i += kBlockSize) {
    vec_t a;
    if constexpr (kCacheRow) {
      a = row[i];
    } else {
      vec_t const g = gate[i];
      vec_t const u = up[i];
#pragma unroll
      for (int j = 0; j < kVecSize; ++j) {
        a.val[j] = gelu_tanh_and_mul(g.val[j], u.val[j]);
      }
    }
    out_vec_t q;
#pragma unroll
    for (int j = 0; j < kVecSize; ++j) {
      q.val[j] = scaled_fp8_conversion<false, fp8_type>(
          static_cast<float>(a.val[j]), scale);
    }
    token_out[i] = q;
  }
}

template <typename scalar_t, typename fp8_type, int kBlockSize, int kVecSize>
void launch(fp8_type* out, float* scales, scalar_t const* input,
            float const* scale_ub, int32_t d, int64_t num_tokens,
            cudaStream_t stream) {
  size_t const row_bytes = static_cast<size_t>(d) * sizeof(scalar_t);
  VLLM_STABLE_DISPATCH_BOOL(row_bytes <= kMaxCachedRowBytes, cache_row, [&] {
    gelu_tanh_and_mul_per_token_quant_kernel<scalar_t, fp8_type, kBlockSize,
                                             kVecSize, cache_row>
        <<<dim3(num_tokens), dim3(kBlockSize), cache_row ? row_bytes : 0,
           stream>>>(out, scales, input, scale_ub, d);
  });
}

}  // namespace

}  // namespace vllm

void gelu_tanh_and_mul_per_token_quant(
    torch::stable::Tensor& result,       // [..., d]
    torch::stable::Tensor const& input,  // [..., 2 * d]
    torch::stable::Tensor& scale,        // [num_tokens, 1]
    std::optional<torch::stable::Tensor> const& scale_ub) {
  STD_TORCH_CHECK(
      result.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn ||
      result.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fnuz);
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Input must be FP16 or BF16");
  STD_TORCH_CHECK(result.is_contiguous() && input.is_contiguous());
  STD_TORCH_CHECK(scale.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(input.size(-1) % 2 == 0);

  int32_t const d = input.size(-1) / 2;
  int64_t const num_tokens = input.numel() / input.size(-1);
  STD_TORCH_CHECK(result.numel() == num_tokens * d);
  STD_TORCH_CHECK(scale.numel() == num_tokens);
  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(scale_ub->scalar_type() ==
                        torch::headeronly::ScalarType::Float &&
                    scale_ub->numel() == 1);
  }
  if (num_tokens == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  // With fewer tokens than SMs each SM runs at most one block, so use the most
  // threads per block to pull bandwidth. Otherwise smaller blocks let more
  // rows (each cached in shared memory) reside on an SM. Measured on SM89.
  int const block_size = num_tokens < get_device_prop()->multiProcessorCount
                             ? 1024
                             : (d <= 12288 ? 256 : 512);

  VLLM_STABLE_DISPATCH_HALF_TYPES(
      input.scalar_type(), "gelu_tanh_and_mul_per_token_quant", [&] {
        VLLM_STABLE_DISPATCH_FP8_TYPES(
            result.scalar_type(), "gelu_tanh_and_mul_per_token_quant", [&] {
              scalar_t const* in_ptr = input.const_data_ptr<scalar_t>();
              fp8_t* out_ptr = result.mutable_data_ptr<fp8_t>();
              float* scale_ptr = scale.mutable_data_ptr<float>();
              float const* ub_ptr = scale_ub.has_value()
                                        ? scale_ub->const_data_ptr<float>()
                                        : nullptr;
              // 16-byte loads of gate and up when every row allows them.
              constexpr int kVecSize = 16 / sizeof(scalar_t);
              bool const vectorize = d % kVecSize == 0 &&
                                     vllm::is_16byte_aligned(in_ptr) &&
                                     vllm::is_16byte_aligned(out_ptr);
              auto dispatch = [&](auto block_const) {
                constexpr int kBlockSize = decltype(block_const)::value;
                if (vectorize) {
                  vllm::launch<scalar_t, fp8_t, kBlockSize, kVecSize>(
                      out_ptr, scale_ptr, in_ptr, ub_ptr, d, num_tokens,
                      stream);
                } else {
                  vllm::launch<scalar_t, fp8_t, kBlockSize, 1>(
                      out_ptr, scale_ptr, in_ptr, ub_ptr, d, num_tokens,
                      stream);
                }
              };
              if (block_size == 1024) {
                dispatch(std::integral_constant<int, 1024>{});
              } else if (block_size == 512) {
                dispatch(std::integral_constant<int, 512>{});
              } else {
                dispatch(std::integral_constant<int, 256>{});
              }
            });
      });
}
