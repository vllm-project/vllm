// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include "../../dispatch_utils.h"
#include "quant_conversions.cuh"

#ifndef USE_ROCM
  #include <cuda_bf16.h>
  #include <cuda_fp8.h>
#endif

namespace vllm {

// Logic: one thread block per (token, group) pair

template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          bool use_ue8m0, int32_t group_size>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,  // Output: [num_tokens, hidden_size] in
                                     // FP8/INT8
    float* __restrict__ scales,      // Output: [num_tokens, hidden_size /
                                 // group_size] or [hidden_size / group_size,
                                 // num_tokens]
    scalar_t const* __restrict__ input,  // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,               // Optional scale upper bound
    float const clamp_limit,
    int32_t const hidden_size  // Output hidden size (input is 2x this)
) {
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

  if (clamp_limit > 0.0f) {
    gate = fminf(gate, clamp_limit);
    up = fmaxf(fminf(up, clamp_limit), -clamp_limit);
  }

  // Compute SiLU(gate) * up
  float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
  float silu_gate = gate * sigmoid_gate;
  float result = silu_gate * up;  // Keep in register
  if (clamp_limit > 0.0f) {
    // Match the low-precision SiLU and multiply in the unfused activation.
    scalar_t const silu_narrowed = static_cast<scalar_t>(silu_gate);
    scalar_t const up_narrowed = static_cast<scalar_t>(up);
    result = static_cast<float>(silu_narrowed * up_narrowed);
  }

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

    if constexpr (use_ue8m0) {
      // Match per_token_group_quant_8bit_kernel: DeepGEMM consumes FP32
      // scales rounded up to an exact power of two on Hopper.
      group_scale = exp2f(ceilf(log2f(fmaxf(fabsf(group_scale), 1e-10f))));
    } else {
      // Use minimum safe scaling factor
      group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());
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

#ifndef USE_ROCM
constexpr int32_t kDsv4GroupSize = 128;
constexpr int32_t kDsv4ElementsPerThread = 8;
constexpr float kDsv4QuantEps = 1e-10f;
// Below this size, one block per quantization group has lower launch cost than
// filling the GPU with persistent blocks. The crossover is stable when scaled
// by the number of output elements across the 512 and 2048 hidden-size paths.
constexpr int64_t kDsv4PersistentMinElements = 1 << 20;

__device__ __forceinline__ float dsv4_half_warp_reduce_max(float value) {
  unsigned const mask = threadIdx.x % 32 >= 16 ? 0xffff0000u : 0x0000ffffu;
  #pragma unroll
  for (int32_t offset = 8; offset >= 1; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(mask, value, offset, 16));
  }
  return value;
}

__device__ __forceinline__ void load_bf16x8(
    __nv_bfloat16 const* ptr, float (&values)[kDsv4ElementsPerThread]) {
  uint4 const packed = *reinterpret_cast<uint4 const*>(ptr);
  auto const* pairs = reinterpret_cast<__nv_bfloat162 const*>(&packed);
  #pragma unroll
  for (int32_t i = 0; i < 4; ++i) {
    float2 const pair = __bfloat1622float2(pairs[i]);
    values[2 * i] = pair.x;
    values[2 * i + 1] = pair.y;
  }
}

__device__ __forceinline__ void store_fp8x8(
    __nv_fp8_e4m3* ptr, float const (&values)[kDsv4ElementsPerThread]) {
  __nv_fp8x4_e4m3 const packed0(
      make_float4(values[0], values[1], values[2], values[3]));
  __nv_fp8x4_e4m3 const packed1(
      make_float4(values[4], values[5], values[6], values[7]));
  uint2 packed;
  packed.x = *reinterpret_cast<uint32_t const*>(&packed0);
  packed.y = *reinterpret_cast<uint32_t const*>(&packed1);
  *reinterpret_cast<uint2*>(ptr) = packed;
}

// Persistent DeepSeek-V4 specialization adapted from SGLang PR #32058.
// It fuses clamp, SiLU, gated multiplication, and FP8 group quantization,
// while skipping token assignments routed to non-local experts.
template <bool has_clamp, bool has_expert_ids, bool has_expert_map,
          bool use_ue8m0>
__global__ void dsv4_silu_and_mul_per_block_quant_kernel(
    __nv_fp8_e4m3* __restrict__ output, float* __restrict__ output_scale,
    __nv_bfloat16 const* __restrict__ input,
    int32_t const* __restrict__ expert_ids,
    int32_t const* __restrict__ expert_map, int32_t expert_step,
    int32_t num_tokens, int32_t hidden_size, int32_t num_groups,
    int32_t num_block_columns, int32_t num_block_rows, float clamp_limit) {
  int32_t const block_linear_id = blockIdx.x;
  int32_t const block_column = block_linear_id % num_block_columns;
  int32_t const block_row = block_linear_id / num_block_columns;
  int32_t const lane_id = threadIdx.x % 32;
  int32_t const thread_column = threadIdx.x + block_column * blockDim.x;

  constexpr int32_t rows_per_iteration = 4;
  for (int32_t row_start = block_row * rows_per_iteration;
       row_start < num_tokens;
       row_start += num_block_rows * rows_per_iteration) {
  #pragma unroll
    for (int32_t row_offset = 0; row_offset < rows_per_iteration;
         ++row_offset) {
      int32_t const row = row_start + row_offset;
      if (row >= num_tokens) {
        break;
      }

      if constexpr (has_expert_ids) {
        int32_t const expert_id = expert_ids[row / expert_step];
        if (expert_id < 0) {
          continue;
        }
        if constexpr (has_expert_map) {
          if (expert_map[expert_id] < 0) {
            continue;
          }
        }
      }

      int32_t const column = thread_column * kDsv4ElementsPerThread;
      if (column >= hidden_size) {
        continue;
      }

      auto const* gate_ptr =
          input + static_cast<int64_t>(row) * hidden_size * 2 + column;
      auto const* up_ptr = gate_ptr + hidden_size;
      auto* output_ptr =
          output + static_cast<int64_t>(row) * hidden_size + column;
      auto* scale_ptr = output_scale + static_cast<int64_t>(row) * num_groups;

      float gate[kDsv4ElementsPerThread];
      float up[kDsv4ElementsPerThread];
      float result[kDsv4ElementsPerThread];
      load_bf16x8(gate_ptr, gate);
      load_bf16x8(up_ptr, up);

      float thread_max = kDsv4QuantEps;
  #pragma unroll
      for (int32_t i = 0; i < kDsv4ElementsPerThread; ++i) {
        if constexpr (has_clamp) {
          gate[i] = fminf(gate[i], clamp_limit);
          up[i] = fmaxf(fminf(up[i], clamp_limit), -clamp_limit);
        }
        float const sigmoid_gate = 1.0f / (1.0f + expf(-gate[i]));
        __nv_bfloat16 const silu_narrowed =
            __float2bfloat16_rn(gate[i] * sigmoid_gate);
        __nv_bfloat16 const up_narrowed = __float2bfloat16_rn(up[i]);
        __nv_bfloat16 const narrowed = __hmul(silu_narrowed, up_narrowed);
        result[i] = __bfloat162float(narrowed);
        thread_max = fmaxf(thread_max, fabsf(result[i]));
      }

      float const group_max = dsv4_half_warp_reduce_max(thread_max);
      float scale = group_max / 448.0f;
      if constexpr (use_ue8m0) {
        scale = exp2f(ceilf(log2f(fmaxf(fabsf(scale), 1e-10f))));
      }
      float const inverted_scale = 1.0f / scale;
  #pragma unroll
      for (int32_t i = 0; i < kDsv4ElementsPerThread; ++i) {
        result[i] *= inverted_scale;
      }
      store_fp8x8(output_ptr, result);

      if (lane_id == 0 || lane_id == 16) {
        scale_ptr[column / kDsv4GroupSize] = scale;
      }
    }
  }
}
#endif

}  // namespace vllm

void silu_and_mul_per_block_quant(
    torch::stable::Tensor& out, torch::stable::Tensor const& input,
    torch::stable::Tensor& scales, int64_t group_size,
    std::optional<torch::stable::Tensor> scale_ub, bool is_scale_transposed,
    std::optional<double> clamp_limit,
    std::optional<torch::stable::Tensor> expert_ids,
    std::optional<torch::stable::Tensor> expert_map, int64_t expert_step,
    bool use_ue8m0) {
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
  STD_TORCH_CHECK(expert_step > 0, "expert_step must be positive");
  STD_TORCH_CHECK(!expert_map.has_value() || expert_ids.has_value(),
                  "expert_map requires expert_ids");
#ifdef USE_ROCM
  STD_TORCH_CHECK(!expert_ids.has_value(),
                  "expert filtering is only supported on CUDA");
#endif
  if (expert_ids.has_value()) {
    STD_TORCH_CHECK(expert_ids->scalar_type() ==
                    torch::headeronly::ScalarType::Int);
    STD_TORCH_CHECK(expert_ids->is_contiguous());
    STD_TORCH_CHECK(expert_ids->get_device_index() == input.get_device_index());
    STD_TORCH_CHECK(expert_ids->numel() * expert_step >= num_tokens,
                    "expert_ids does not cover all input rows");
  }
  if (expert_map.has_value()) {
    STD_TORCH_CHECK(expert_map->scalar_type() ==
                    torch::headeronly::ScalarType::Int);
    STD_TORCH_CHECK(expert_map->is_contiguous());
    STD_TORCH_CHECK(expert_map->get_device_index() == input.get_device_index());
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

#ifndef USE_ROCM
  bool const use_dsv4_kernel =
      input.scalar_type() == torch::headeronly::ScalarType::BFloat16 &&
      out.scalar_type() == torch::headeronly::ScalarType::Float8_e4m3fn &&
      group_size == vllm::kDsv4GroupSize && !is_scale_transposed &&
      !scale_ub.has_value() &&
      (clamp_limit.has_value() || expert_ids.has_value()) &&
      (expert_ids.has_value() ||
       static_cast<int64_t>(num_tokens) * hidden_size >=
           vllm::kDsv4PersistentMinElements);
  if (use_dsv4_kernel) {
    if (num_tokens == 0) {
      return;
    }
    constexpr int32_t block_size = 256;
    int32_t const num_block_columns =
        (hidden_size / vllm::kDsv4ElementsPerThread + block_size - 1) /
        block_size;
    int32_t num_sms;
    cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount,
                           input.get_device_index());
    int32_t const num_block_rows = std::max(num_sms * 8 / num_block_columns, 1);
    dim3 const grid(num_block_rows * num_block_columns);
    dim3 const block(block_size);

    float const clamp = static_cast<float>(clamp_limit.value_or(0.0));
    auto const* ids = expert_ids.has_value()
                          ? expert_ids->const_data_ptr<int32_t>()
                          : nullptr;
    auto const* map = expert_map.has_value()
                          ? expert_map->const_data_ptr<int32_t>()
                          : nullptr;

  #define LAUNCH_DSV4(HAS_CLAMP, HAS_IDS, HAS_MAP, USE_UE8M0)               \
    vllm::dsv4_silu_and_mul_per_block_quant_kernel<HAS_CLAMP, HAS_IDS,      \
                                                   HAS_MAP, USE_UE8M0>      \
        <<<grid, block, 0, stream>>>(                                       \
            reinterpret_cast<__nv_fp8_e4m3*>(out.mutable_data_ptr()),       \
            scales.mutable_data_ptr<float>(),                               \
            reinterpret_cast<__nv_bfloat16 const*>(input.const_data_ptr()), \
            ids, map, static_cast<int32_t>(expert_step), num_tokens,        \
            hidden_size, num_groups, num_block_columns, num_block_rows, clamp)

    if (clamp > 0.0f) {
      if (ids != nullptr) {
        if (map != nullptr) {
          if (use_ue8m0) {
            LAUNCH_DSV4(true, true, true, true);
          } else {
            LAUNCH_DSV4(true, true, true, false);
          }
        } else {
          if (use_ue8m0) {
            LAUNCH_DSV4(true, true, false, true);
          } else {
            LAUNCH_DSV4(true, true, false, false);
          }
        }
      } else {
        if (use_ue8m0) {
          LAUNCH_DSV4(true, false, false, true);
        } else {
          LAUNCH_DSV4(true, false, false, false);
        }
      }
    } else if (ids != nullptr) {
      if (map != nullptr) {
        if (use_ue8m0) {
          LAUNCH_DSV4(false, true, true, true);
        } else {
          LAUNCH_DSV4(false, true, true, false);
        }
      } else {
        if (use_ue8m0) {
          LAUNCH_DSV4(false, true, false, true);
        } else {
          LAUNCH_DSV4(false, true, false, false);
        }
      }
    } else {
      if (use_ue8m0) {
        LAUNCH_DSV4(false, false, false, true);
      } else {
        LAUNCH_DSV4(false, false, false, false);
      }
    }
  #undef LAUNCH_DSV4
    return;
  }
#endif

  dim3 grid(num_tokens, num_groups);
  dim3 block(group_size);

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_STABLE_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant", [&] {
              using scalar_out_t = scalar_t;

              VLLM_STABLE_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                VLLM_STABLE_DISPATCH_BOOL(
                    is_scale_transposed, transpose_scale, [&] {
                      VLLM_STABLE_DISPATCH_BOOL(use_ue8m0, use_ue8m0_, [&] {
                        vllm::silu_and_mul_per_block_quant_kernel<
                            scalar_in_t, scalar_out_t, transpose_scale,
                            use_ue8m0_, gs><<<grid, block, 0, stream>>>(
                            out.mutable_data_ptr<scalar_out_t>(),
                            scales.mutable_data_ptr<float>(),
                            input.const_data_ptr<scalar_in_t>(),
                            scale_ub.has_value()
                                ? scale_ub->const_data_ptr<float>()
                                : nullptr,
                            static_cast<float>(clamp_limit.value_or(0.0)),
                            hidden_size);
                      });
                    });
              });
            });
      });
}
