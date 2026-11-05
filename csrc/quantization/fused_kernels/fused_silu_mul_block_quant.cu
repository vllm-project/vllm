// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include "../../dispatch_utils.h"
#include "../../core/cuda_compat.h"
#include "quant_conversions.cuh"

namespace vllm {

// Logic: one thread block per (token, group) pair

template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size, bool use_ue8m0, bool has_clamp>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,  // Output: [num_tokens, hidden_size] in
                                     // FP8/INT8
    float* __restrict__ scales,      // Output: [num_tokens, hidden_size /
                                 // group_size] or [hidden_size / group_size,
                                 // num_tokens]
    scalar_t const* __restrict__ input,  // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,               // Optional scale upper bound
    int32_t const hidden_size,  // Output hidden size (input is 2x this)
    float const clamp_limit) {
  static_assert((group_size & (group_size - 1)) == 0,
                "group_size must be a power of 2 for correct reduction");

  // Grid: (num_tokens, num_groups)
  int64_t const token_idx = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const tid = threadIdx.x;  // tid in [0, group_size)
  int const num_tokens = gridDim.x;

  // Input layout: [gate || up] concatenated along last dimension
  int const input_stride = hidden_size * 2;
  int const group_start = group_idx * group_size;

  // Pointers to this token's data
  scalar_t const* token_input_gate =
      input + token_idx * input_stride + group_start;
  scalar_t const* token_input_up = token_input_gate + hidden_size;
  scalar_out_t* token_output = out + token_idx * hidden_size + group_start;

  // Scale pointer for this group
  int const num_groups = gridDim.y;
  float* group_scale_ptr = is_scale_transposed
                               ? scales + group_idx * num_tokens + token_idx
                               : scales + token_idx * num_groups + group_idx;

  // Shared memory for reduction (compile-time sized)
  __shared__ float shared_max[group_size];

  // Step 1: Each thread loads one element, computes SiLU, stores in register
  float gate = static_cast<float>(token_input_gate[tid]);
  float up = static_cast<float>(token_input_up[tid]);
  if constexpr (has_clamp) {
    gate = fminf(gate, clamp_limit);
    up = fminf(fmaxf(up, -clamp_limit), clamp_limit);
  }

  // Compute SiLU(gate) * up
  float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
  float silu_gate = gate * sigmoid_gate;
  float result = silu_gate * up;  // Keep in register

  // Step 2: Reduce to find group max
  shared_max[tid] = fabsf(result);
  __syncthreads();

// Power-of-2 reduction (group_size guaranteed to be power of 2)
#pragma unroll
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  // Step 3: Compute scale (thread 0), broadcast via shared memory
  if (tid == 0) {
    float group_max = shared_max[0];

    float const quant_range = quant_type_max_v<scalar_out_t>;
    float group_scale = group_max / quant_range;

    // Apply scale upper bound if provided
    if (scale_ub != nullptr) {
      group_scale = fminf(group_scale, *scale_ub);
    }

    // Use minimum safe scaling factor
    group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());

    if constexpr (use_ue8m0) {
      group_scale = exp2f(ceilf(log2f(group_scale)));
    }

    // Store scale to global memory
    *group_scale_ptr = group_scale;

    // Reuse shared_max[0] to broadcast scale
    shared_max[0] = group_scale;
  }
  __syncthreads();

  float group_scale = shared_max[0];

  // Step 4: Quantize and write output
  token_output[tid] =
      vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale);
}

// Vectorized variant: each group is handled by group_size / 8 lanes of a
// warp, each loading 8 contiguous 16-bit elements of gate and up (16B loads)
// and storing 8 quantized bytes. The group max is reduced with warp shuffles,
// so there is no shared memory or __syncthreads. Requires 16B-aligned input
// and 8B-aligned output rows.
template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size, bool use_ue8m0, bool has_clamp>
__global__ void silu_and_mul_per_block_quant_vec_kernel(
    scalar_out_t* __restrict__ out, float* __restrict__ scales,
    scalar_t const* __restrict__ input, float const* scale_ub,
    int32_t const hidden_size, int32_t const num_groups,
    int64_t const num_tokens, float const clamp_limit) {
  constexpr int kVec = 16 / sizeof(scalar_t);
  constexpr int kLanes = group_size / kVec;
  static_assert(kVec == 8, "expects 16-bit input");
  static_assert(kLanes > 0 && kLanes <= 32 && (32 % kLanes) == 0);

  int64_t const gid =
      (static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) / kLanes;
  int const lane = threadIdx.x % kLanes;
  bool const valid = gid < num_tokens * num_groups;
  int64_t const token_idx = valid ? gid / num_groups : 0;
  int const group_idx = valid ? static_cast<int>(gid % num_groups) : 0;

  int64_t const in_off = token_idx * (2 * static_cast<int64_t>(hidden_size)) +
                         group_idx * group_size + lane * kVec;
  alignas(16) scalar_t g[kVec];
  alignas(16) scalar_t u[kVec];
  if (valid) {
    *reinterpret_cast<uint4*>(g) =
        *reinterpret_cast<uint4 const*>(input + in_off);
    *reinterpret_cast<uint4*>(u) =
        *reinterpret_cast<uint4 const*>(input + in_off + hidden_size);
  } else {
    *reinterpret_cast<uint4*>(g) = make_uint4(0, 0, 0, 0);
    *reinterpret_cast<uint4*>(u) = make_uint4(0, 0, 0, 0);
  }

  float r[kVec];
  float local_max = 0.f;
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    float gate = static_cast<float>(g[i]);
    float up = static_cast<float>(u[i]);
    if constexpr (has_clamp) {
      gate = fminf(gate, clamp_limit);
      up = fminf(fmaxf(up, -clamp_limit), clamp_limit);
    }
    float const sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    r[i] = gate * sigmoid_gate * up;
    local_max = fmaxf(local_max, fabsf(r[i]));
  }
#pragma unroll
  for (int off = kLanes / 2; off > 0; off >>= 1) {
    local_max =
        fmaxf(local_max, VLLM_SHFL_XOR_SYNC_WIDTH(local_max, off, kLanes));
  }

  float group_scale = local_max / quant_type_max_v<scalar_out_t>;
  if (scale_ub != nullptr) {
    group_scale = fminf(group_scale, *scale_ub);
  }
  group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());
  if constexpr (use_ue8m0) {
    group_scale = exp2f(ceilf(log2f(group_scale)));
  }
  if (!valid) {
    return;
  }

  alignas(8) scalar_out_t q[kVec];
#pragma unroll
  for (int i = 0; i < kVec; ++i) {
    q[i] = vllm::ScaledQuant<scalar_out_t, false>::quant_fn(r[i], group_scale);
  }
  *reinterpret_cast<uint2*>(out + token_idx * hidden_size +
                            group_idx * group_size + lane * kVec) =
      *reinterpret_cast<uint2 const*>(q);
  if (lane == 0) {
    float* scale_ptr = is_scale_transposed
                           ? scales + group_idx * num_tokens + token_idx
                           : scales + token_idx * num_groups + group_idx;
    *scale_ptr = group_scale;
  }
}

}  // namespace vllm

void silu_and_mul_per_block_quant(torch::stable::Tensor& out,
                                  torch::stable::Tensor const& input,
                                  torch::stable::Tensor& scales,
                                  int64_t group_size,
                                  std::optional<torch::stable::Tensor> scale_ub,
                                  bool is_scale_transposed, bool use_ue8m0,
                                  std::optional<double> clamp_limit) {
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
  STD_TORCH_CHECK(group_size == 128 || group_size == 64,
                  "Unsupported group size: ", group_size);

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  }

  int32_t hidden_size = out.size(-1);
  auto num_tokens = input.size(0);
  int32_t num_groups = hidden_size / group_size;

  STD_TORCH_CHECK(input.size(-1) == hidden_size * 2,
                  "input last dim must be 2x output hidden_size");
  STD_TORCH_CHECK(hidden_size % group_size == 0,
                  "hidden_size must be divisible by group_size");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  dim3 grid(num_tokens, num_groups);
  dim3 block(group_size);
  // Below ~3k groups the one-block-per-group kernel has lower latency.
  bool const use_vec =
      static_cast<int64_t>(num_tokens) * num_groups >= 3072 &&
      input.element_size() == 2 &&
      reinterpret_cast<uintptr_t>(input.const_data_ptr()) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(out.const_data_ptr()) % 8 == 0;
  constexpr int kVecBlock = 256;

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_STABLE_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant", [&] {
              using scalar_out_t = scalar_t;

              VLLM_STABLE_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                VLLM_STABLE_DISPATCH_BOOL(
                    is_scale_transposed, transpose_scale, [&] {
                      VLLM_STABLE_DISPATCH_BOOL(use_ue8m0, ue8m0, [&] {
                        VLLM_STABLE_DISPATCH_BOOL(
                            clamp_limit.has_value(), clamp, [&] {
                              if constexpr (sizeof(scalar_in_t) == 2) {
                                if (use_vec) {
                                  constexpr int kLanes = gs / 8;
                                  int64_t const total =
                                      static_cast<int64_t>(num_tokens) *
                                      num_groups * kLanes;
                                  dim3 vgrid(static_cast<unsigned>(
                                      (total + kVecBlock - 1) / kVecBlock));
                                  vllm::silu_and_mul_per_block_quant_vec_kernel<
                                      scalar_in_t, scalar_out_t,
                                      transpose_scale, gs, ue8m0, clamp>
                                      <<<vgrid, kVecBlock, 0, stream>>>(
                                          out.mutable_data_ptr<scalar_out_t>(),
                                          scales.mutable_data_ptr<float>(),
                                          input.const_data_ptr<scalar_in_t>(),
                                          scale_ub.has_value()
                                              ? scale_ub
                                                    ->const_data_ptr<float>()
                                              : nullptr,
                                          hidden_size, num_groups, num_tokens,
                                          static_cast<float>(
                                              clamp_limit.value_or(0.0)));
                                  return;
                                }
                              }
                              vllm::silu_and_mul_per_block_quant_kernel<
                                  scalar_in_t, scalar_out_t, transpose_scale,
                                  gs, ue8m0, clamp><<<grid, block, 0, stream>>>(
                                  out.mutable_data_ptr<scalar_out_t>(),
                                  scales.mutable_data_ptr<float>(),
                                  input.const_data_ptr<scalar_in_t>(),
                                  scale_ub.has_value()
                                      ? scale_ub->const_data_ptr<float>()
                                      : nullptr,
                                  hidden_size,
                                  static_cast<float>(
                                      clamp_limit.value_or(0.0)));
                            });
                      });
                    });
              });
            });
      });
}
