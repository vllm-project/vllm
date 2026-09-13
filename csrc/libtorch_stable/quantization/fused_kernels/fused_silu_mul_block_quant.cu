// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include "../../dispatch_utils.h"
#include "quant_conversions.cuh"

namespace vllm {

// One logical 32-lane warp owns each (token, group) pair. Four groups share a
// 128-thread block. Keeping the logical warp width at 32 preserves the CUDA
// mapping on ROCm wave64 while the explicit mask below isolates each half-wave.
constexpr int kWarpsPerBlock = 4;
constexpr int kLogicalWarpSize = 32;
constexpr int kThreadsPerBlock = kWarpsPerBlock * kLogicalWarpSize;

#ifdef USE_ROCM
__device__ __forceinline__ unsigned long long logical_warp_mask_32(int tid) {
  return warpSize == kLogicalWarpSize
             ? 0xffffffffULL
             : (0xffffffffULL << (tid & kLogicalWarpSize));
}
#else
__device__ __forceinline__ unsigned logical_warp_mask_32(int) {
  return 0xffffffffu;
}
#endif

template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size>
__global__ void
__launch_bounds__(kThreadsPerBlock) silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,  // Output: [num_tokens, hidden_size] in
                                     // FP8/INT8
    float* __restrict__ scales,      // Output: [num_tokens, hidden_size /
                                 // group_size] or [hidden_size / group_size,
                                 // num_tokens]
    scalar_t const* __restrict__ input,  // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,               // Optional scale upper bound
    int32_t const hidden_size  // Output hidden size (input is 2x this)
) {
  static_assert((group_size & (group_size - 1)) == 0,
                "group_size must be a power of 2 for correct reduction");

  static_assert(group_size % kLogicalWarpSize == 0,
                "group_size must be a multiple of the logical warp size");
  constexpr int EPT = group_size / kLogicalWarpSize;

  int const tid = threadIdx.x;
  int const warp_id = tid / kLogicalWarpSize;
  int const lane_id = tid & (kLogicalWarpSize - 1);
  int64_t const token_idx = blockIdx.x;
  int const num_tokens = gridDim.x;
  int const num_groups = hidden_size / group_size;
  int const group_idx = blockIdx.y * kWarpsPerBlock + warp_id;
  if (group_idx >= num_groups) return;

  // Input layout: [gate || up] concatenated along the last dimension. Each
  // lane owns EPT contiguous values, enabling one wide load/store per lane.
  int const input_stride = hidden_size * 2;
  int const group_start = group_idx * group_size;
  int const lane_base = group_start + lane_id * EPT;

  scalar_t const* token_input_gate =
      input + token_idx * input_stride + lane_base;
  scalar_t const* token_input_up = token_input_gate + hidden_size;
  scalar_out_t* token_output = out + token_idx * hidden_size + lane_base;

  // Scale pointer for this group
  float* group_scale_ptr = is_scale_transposed
                               ? scales + group_idx * num_tokens + token_idx
                               : scales + token_idx * num_groups + group_idx;

  struct alignas(sizeof(scalar_t) * EPT) InVec {
    scalar_t v[EPT];
  };
  InVec const gate_v = *reinterpret_cast<InVec const*>(token_input_gate);
  InVec const up_v = *reinterpret_cast<InVec const*>(token_input_up);

  float result[EPT];
  float thread_max = 0.0f;
#pragma unroll
  for (int k = 0; k < EPT; ++k) {
    float gate = static_cast<float>(gate_v.v[k]);
    float up = static_cast<float>(up_v.v[k]);
    float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
    float silu_gate = gate * sigmoid_gate;
    result[k] = silu_gate * up;
    thread_max = fmaxf(thread_max, fabsf(result[k]));
  }

  auto const warp_mask = logical_warp_mask_32(tid);
#pragma unroll
  for (int offset = kLogicalWarpSize / 2; offset > 0; offset >>= 1) {
    thread_max = fmaxf(thread_max, __shfl_xor_sync(warp_mask, thread_max,
                                                   offset, kLogicalWarpSize));
  }

  float const quant_range = quant_type_max_v<scalar_out_t>;
  float group_scale = thread_max / quant_range;
  if (scale_ub != nullptr) {
    group_scale = fminf(group_scale, *scale_ub);
  }
  group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());
  if (lane_id == 0) {
    *group_scale_ptr = group_scale;
  }

  struct alignas(sizeof(scalar_out_t) * EPT) OutVec {
    scalar_out_t q[EPT];
  };
  OutVec out_v;
#pragma unroll
  for (int k = 0; k < EPT; ++k) {
    out_v.q[k] = vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result[k],
                                                                  group_scale);
  }
  *reinterpret_cast<OutVec*>(token_output) = out_v;
}

}  // namespace vllm

void silu_and_mul_per_block_quant(torch::stable::Tensor& out,
                                  torch::stable::Tensor const& input,
                                  torch::stable::Tensor& scales,
                                  int64_t group_size,
                                  std::optional<torch::stable::Tensor> scale_ub,
                                  bool is_scale_transposed) {
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

  dim3 grid(num_tokens,
            (num_groups + vllm::kWarpsPerBlock - 1) / vllm::kWarpsPerBlock);
  dim3 block(vllm::kThreadsPerBlock);

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_STABLE_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant", [&] {
              using scalar_out_t = scalar_t;

              VLLM_STABLE_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                VLLM_STABLE_DISPATCH_BOOL(
                    is_scale_transposed, transpose_scale, [&] {
                      vllm::silu_and_mul_per_block_quant_kernel<
                          scalar_in_t, scalar_out_t, transpose_scale, gs>
                          <<<grid, block, 0, stream>>>(
                              out.mutable_data_ptr<scalar_out_t>(),
                              scales.mutable_data_ptr<float>(),
                              input.const_data_ptr<scalar_in_t>(),
                              scale_ub.has_value()
                                  ? scale_ub->const_data_ptr<float>()
                                  : nullptr,
                              hidden_size);
                    });
              });
            });
      });
}
