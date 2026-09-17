// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "cpu/cpu_arch_macros.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>

#include "cpu/cpu_fused_moe_activations.hpp"
#include "cpu/cpu_types.hpp"
#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"
#include "cpu/utils.hpp"

#if defined(ARM_I8MM_SUPPORT) && defined(ARM_BF16_SUPPORT)
  #include "cpu/micro_gemm/cpu_micro_gemm_int8_neon.hpp"
  #define NEON_DISPATCH(SCALAR_TYPE, ...)                                   \
    case cpu_utils::ISA::NEON: {                                            \
      using gemm_t =                                                        \
          cpu_micro_gemm::MicroGemmINT8<cpu_utils::ISA::NEON, SCALAR_TYPE>; \
      return __VA_ARGS__();                                                 \
    }
#else
  #define NEON_DISPATCH(SCALAR_TYPE, ...) case cpu_utils::ISA::NEON:
#endif

#define CPU_INT8_ISA_DISPATCH_IMPL(ISA_TYPE, SCALAR_TYPE, ...) \
  [&] {                                                        \
    switch (ISA_TYPE) {                                        \
      NEON_DISPATCH(SCALAR_TYPE, __VA_ARGS__)                  \
      default: {                                               \
        TORCH_CHECK(false, "Invalid CPU ISA type.");           \
      }                                                        \
    }                                                          \
  }()

namespace {
using cpu_fused_moe_utils::apply_gated_act;
using cpu_fused_moe_utils::FusedMOEAct;

template <typename gemm_t>
void prepack_moe_weight_int8_impl(const int8_t* __restrict__ weight_ptr,
                                  int8_t* __restrict__ packed_weight_ptr,
                                  const int32_t expert_num,
                                  const int32_t output_size,
                                  const int32_t input_size,
                                  const int64_t expert_stride) {
#pragma omp parallel for
  for (int32_t e_idx = 0; e_idx < expert_num; ++e_idx) {
    gemm_t::pack_weight(weight_ptr + expert_stride * e_idx,
                        packed_weight_ptr + expert_stride * e_idx, output_size,
                        input_size);
  }
}

// INT8 MoE kernel, based on the original BF16 kernel in cpu_fused_moe.cpp
template <typename scalar_t, typename gemm_t>
void fused_moe_int8_impl(
    scalar_t* __restrict__ output, const scalar_t* __restrict__ input,
    const int8_t* __restrict__ w13, const int8_t* __restrict__ w2,
    const float* __restrict__ w13_scales, const float* __restrict__ w2_scales,
    scalar_t* __restrict__ w13_bias, scalar_t* __restrict__ w2_bias,
    const float* __restrict__ topk_weights, const int32_t* __restrict__ topk_id,
    const FusedMOEAct act_type, const int32_t token_num,
    const int32_t expert_num, const int32_t topk_num,
    const int32_t input_size_13, const int32_t output_size_13,
    const int32_t input_size_2, const int32_t output_size_2,
    const bool skip_weighted) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  constexpr int32_t gemm_n_tile_size = gemm_t::NSize;
  constexpr int32_t gemm_m_tile_size = gemm_t::MaxMSize;
  constexpr int32_t min_w13_n_tile_size = 2 * gemm_n_tile_size;

  TORCH_CHECK_EQ(input_size_13 % gemm_t::K, 0);
  TORCH_CHECK_EQ(input_size_2 % gemm_t::K, 0);
  TORCH_CHECK_EQ(output_size_13 % min_w13_n_tile_size, 0);
  TORCH_CHECK_EQ(output_size_2 % gemm_n_tile_size, 0);
  TORCH_CHECK_EQ(output_size_13 / 2, input_size_2);

  const int32_t thread_num = cpu_utils::get_max_threads();
  const int32_t w13_input_buffer_size = cpu_utils::round_up<64>(
      gemm_m_tile_size * input_size_13 * sizeof(int8_t));
  const int32_t w2_input_buffer_size =
      cpu_utils::round_up<64>(gemm_m_tile_size * input_size_2 * sizeof(int8_t));

  const int32_t w13_n_tile_size = [&]() {
    const int64_t cache_size = cpu_utils::get_available_l2_size();
    const int32_t n_size_cache_limit =
        (cache_size - w13_input_buffer_size) /
        (gemm_m_tile_size * sizeof(float) + input_size_13 * sizeof(int8_t));
    const int32_t n_size_thread_limit =
        output_size_13 / std::max(1, thread_num / topk_num);
    const int32_t n_size = cpu_utils::round_down<min_w13_n_tile_size>(
        std::min(n_size_cache_limit, n_size_thread_limit));
    return std::max(n_size, min_w13_n_tile_size);
  }();

  const int32_t w2_n_tile_size = [&]() {
    const int64_t cache_size = cpu_utils::get_available_l2_size();
    const int32_t n_size_cache_limit =
        (cache_size - w2_input_buffer_size) / (input_size_2 * sizeof(int8_t));
    const int32_t n_size_thread_limit =
        output_size_2 / std::max(1, thread_num / topk_num);
    const int32_t n_size = cpu_utils::round_down<gemm_n_tile_size>(
        std::min(n_size_cache_limit, n_size_thread_limit));
    return std::max(n_size, gemm_n_tile_size);
  }();

  int32_t common_buffer_offset = 0;
  const int32_t token_num_per_group_buffer_offset = common_buffer_offset;
  common_buffer_offset += cpu_utils::round_up<64>(expert_num * sizeof(int32_t));
  const int32_t cu_token_num_per_group_buffer_offset = common_buffer_offset;
  common_buffer_offset +=
      cpu_utils::round_up<64>((expert_num + 1) * sizeof(int32_t));
  const int32_t expanded_token_num = token_num * topk_num;
  const int32_t expand_token_id_buffer_offset = common_buffer_offset;
  common_buffer_offset +=
      cpu_utils::round_up<64>(expanded_token_num * sizeof(int32_t));
  const int32_t expand_token_id_index_buffer_offset = common_buffer_offset;
  common_buffer_offset +=
      cpu_utils::round_up<64>(expanded_token_num * sizeof(int32_t));
  const int32_t input_quant_buffer_offset = common_buffer_offset;
  common_buffer_offset +=
      cpu_utils::round_up<64>(token_num * input_size_13 * sizeof(int8_t));
  const int32_t input_scale_buffer_offset = common_buffer_offset;
  common_buffer_offset += cpu_utils::round_up<64>(token_num * sizeof(float));
  const int32_t w13_gemm_output_buffer_offset = common_buffer_offset;
  common_buffer_offset += cpu_utils::round_up<64>(
      expanded_token_num * input_size_2 * sizeof(scalar_t));
  const int32_t w13_output_scale_buffer_offset = common_buffer_offset;
  common_buffer_offset +=
      cpu_utils::round_up<64>(expanded_token_num * sizeof(float));
  const int32_t w2_gemm_output_buffer_offset = common_buffer_offset;
  common_buffer_offset += cpu_utils::round_up<64>(
      expanded_token_num * output_size_2 * sizeof(float));

  int32_t gemm_thread_buffer_offset = 0;
  const int32_t gemm_input_buffer_offset = gemm_thread_buffer_offset;
  gemm_thread_buffer_offset +=
      std::max(w13_input_buffer_size, w2_input_buffer_size);
  const int32_t gemm_output_buffer_offset = gemm_thread_buffer_offset;
  gemm_thread_buffer_offset += cpu_utils::round_up<64>(
      gemm_m_tile_size * std::max(w13_n_tile_size, w2_n_tile_size) *
      sizeof(int32_t));

  const int32_t ws_output_buffer_offset = 0;
  const int32_t ws_thread_buffer_size =
      cpu_utils::round_up<64>(output_size_2 * sizeof(float));
  const int32_t thread_buffer_size =
      std::max(gemm_thread_buffer_offset, ws_thread_buffer_size);
  const int32_t buffer_size =
      common_buffer_offset + thread_buffer_size * thread_num;
  cpu_utils::ScratchPadManager::get_scratchpad_manager()->realloc(buffer_size);
  uint8_t* common_buffer_start =
      cpu_utils::ScratchPadManager::get_scratchpad_manager()
          ->get_data<uint8_t>();
  uint8_t* thread_buffer_start = common_buffer_start + common_buffer_offset;

  int32_t* __restrict__ token_num_per_group_buffer = reinterpret_cast<int32_t*>(
      common_buffer_start + token_num_per_group_buffer_offset);
  int32_t* __restrict__ cu_token_num_per_group_buffer =
      reinterpret_cast<int32_t*>(common_buffer_start +
                                 cu_token_num_per_group_buffer_offset);
  int32_t* __restrict__ expand_token_id_buffer = reinterpret_cast<int32_t*>(
      common_buffer_start + expand_token_id_buffer_offset);
  int32_t* __restrict__ expand_token_id_index_buffer =
      reinterpret_cast<int32_t*>(common_buffer_start +
                                 expand_token_id_index_buffer_offset);
  int8_t* __restrict__ input_quant_buffer = reinterpret_cast<int8_t*>(
      common_buffer_start + input_quant_buffer_offset);
  float* __restrict__ input_scale_buffer =
      reinterpret_cast<float*>(common_buffer_start + input_scale_buffer_offset);

  std::memset(token_num_per_group_buffer, 0, expert_num * sizeof(int32_t));
  for (int32_t i = 0; i < expanded_token_num; ++i) {
    ++token_num_per_group_buffer[topk_id[i]];
  }

  int32_t token_num_sum = 0;
  cu_token_num_per_group_buffer[0] = 0;
  int32_t* token_index_buffer = cu_token_num_per_group_buffer + 1;
  for (int32_t i = 0; i < expert_num; ++i) {
    token_index_buffer[i] = token_num_sum;
    token_num_sum += token_num_per_group_buffer[i];
  }

  for (int32_t i = 0; i < token_num; ++i) {
    const int32_t* curr_topk_id = topk_id + i * topk_num;
    int32_t* curr_index_buffer = expand_token_id_index_buffer + i * topk_num;
    for (int32_t j = 0; j < topk_num; ++j) {
      const int32_t curr_expert_id = curr_topk_id[j];
      const int32_t curr_index = token_index_buffer[curr_expert_id]++;
      expand_token_id_buffer[curr_index] = i;
      curr_index_buffer[j] = curr_index;
    }
  }

// quantize inputs
#pragma omp parallel for
  for (int32_t token_idx = 0; token_idx < token_num; ++token_idx) {
    gemm_t::quantize_row(input + token_idx * input_size_13,
                         input_quant_buffer + token_idx * input_size_13,
                         input_scale_buffer[token_idx], input_size_13);
  }

  {
    alignas(64) cpu_utils::Counter counter;
    cpu_utils::Counter* counter_ptr = &counter;

// w13 GEMM + act
#pragma omp parallel for schedule(static, 1)
    for (int32_t thread_id = 0; thread_id < thread_num; ++thread_id) {
      const int32_t task_num_per_expert =
          (output_size_13 + w13_n_tile_size - 1) / w13_n_tile_size;
      const int32_t task_num = task_num_per_expert * expert_num;
      uint8_t* __restrict__ thread_buffer =
          thread_buffer_start + thread_id * thread_buffer_size;
      int8_t* __restrict__ gemm_input_buffer =
          reinterpret_cast<int8_t*>(thread_buffer + gemm_input_buffer_offset);
      float* __restrict__ gemm_output_buffer =
          reinterpret_cast<float*>(thread_buffer + gemm_output_buffer_offset);
      auto* __restrict__ w13_gemm_output_buffer = reinterpret_cast<scalar_t*>(
          common_buffer_start + w13_gemm_output_buffer_offset);
      gemm_t gemm;

      const int32_t w13_n_group_stride =
          gemm_t::WeightOCGroupSize * input_size_13;
      const int32_t w13_n_tile_stride = gemm_n_tile_size * input_size_13;

      for (;;) {
        const int32_t task_id = counter_ptr->acquire_counter();
        if (task_id >= task_num) {
          break;
        }
        const int32_t curr_expert_id = task_id / task_num_per_expert;
        const int32_t curr_output_group_id = task_id % task_num_per_expert;
        const int32_t curr_token_num =
            token_num_per_group_buffer[curr_expert_id];
        if (curr_token_num == 0) {
          continue;
        }

        const int32_t actual_n_tile_size =
            std::min(w13_n_tile_size,
                     output_size_13 - curr_output_group_id * w13_n_tile_size);
        const int32_t* __restrict__ curr_expand_token_id_buffer =
            expand_token_id_buffer +
            cu_token_num_per_group_buffer[curr_expert_id];
        scalar_t* __restrict__ curr_w13_gemm_output_buffer =
            w13_gemm_output_buffer +
            cu_token_num_per_group_buffer[curr_expert_id] * input_size_2 +
            curr_output_group_id * w13_n_tile_size / 2;

        const int8_t* w13_weight_ptr_0 = nullptr;
        const int8_t* w13_weight_ptr_1 = nullptr;
        const float* w13_scale_ptr_0 = nullptr;
        const float* w13_scale_ptr_1 = nullptr;
        scalar_t* w13_bias_ptr_0 = nullptr;
        scalar_t* w13_bias_ptr_1 = nullptr;
        if (act_type == FusedMOEAct::SwigluOAIAndMul) {
          const int32_t output_offset = curr_output_group_id * w13_n_tile_size;
          w13_weight_ptr_0 = w13 +
                             curr_expert_id * input_size_13 * output_size_13 +
                             output_offset * input_size_13;
          w13_weight_ptr_1 =
              w13_weight_ptr_0 + actual_n_tile_size / 2 * input_size_13;
          w13_scale_ptr_0 =
              w13_scales + curr_expert_id * output_size_13 + output_offset;
          w13_scale_ptr_1 = w13_scale_ptr_0 + actual_n_tile_size / 2;
          if (w13_bias != nullptr) {
            w13_bias_ptr_0 =
                w13_bias + curr_expert_id * output_size_13 + output_offset;
            w13_bias_ptr_1 = w13_bias_ptr_0 + actual_n_tile_size / 2;
          }
        } else {
          const int32_t output_offset =
              curr_output_group_id * (w13_n_tile_size / 2);
          w13_weight_ptr_0 = w13 +
                             curr_expert_id * input_size_13 * output_size_13 +
                             output_offset * input_size_13;
          w13_weight_ptr_1 =
              w13_weight_ptr_0 + output_size_13 / 2 * input_size_13;
          w13_scale_ptr_0 =
              w13_scales + curr_expert_id * output_size_13 + output_offset;
          w13_scale_ptr_1 = w13_scale_ptr_0 + output_size_13 / 2;
          if (w13_bias != nullptr) {
            w13_bias_ptr_0 =
                w13_bias + curr_expert_id * output_size_13 + output_offset;
            w13_bias_ptr_1 = w13_bias_ptr_0 + output_size_13 / 2;
          }
        }

        for (int32_t token_idx = 0; token_idx < curr_token_num;
             token_idx += gemm_m_tile_size) {
          const int32_t actual_token_num =
              std::min(gemm_m_tile_size, curr_token_num - token_idx);
          const int8_t* input_rows[gemm_m_tile_size];
          alignas(64) float input_scales[gemm_m_tile_size];
          // gather and pack
          for (int32_t i = 0; i < actual_token_num; ++i) {
            const int32_t curr_token_id = curr_expand_token_id_buffer[i];
            input_rows[i] = input_quant_buffer + curr_token_id * input_size_13;
            input_scales[i] = input_scale_buffer[curr_token_id];
          }
          gemm_t::pack_input_from_rows(input_rows, gemm_input_buffer,
                                       actual_token_num, input_size_13);
          curr_expand_token_id_buffer += actual_token_num;

          const int8_t* w13_weight_ptr_0_iter = w13_weight_ptr_0;
          const int8_t* w13_weight_ptr_1_iter = w13_weight_ptr_1;
          const float* w13_scale_ptr_0_iter = w13_scale_ptr_0;
          const float* w13_scale_ptr_1_iter = w13_scale_ptr_1;
          scalar_t* w13_bias_ptr_0_iter = w13_bias_ptr_0;
          scalar_t* w13_bias_ptr_1_iter = w13_bias_ptr_1;
          float* w13_output_buffer_0_iter = gemm_output_buffer;
          float* w13_output_buffer_1_iter =
              gemm_output_buffer + actual_n_tile_size / 2;

          for (int32_t i = 0; i < actual_n_tile_size;
               i += min_w13_n_tile_size) {
            auto* output_0_int32 =
                reinterpret_cast<int32_t*>(w13_output_buffer_0_iter);
            gemm.gemm(gemm_input_buffer, w13_weight_ptr_0_iter, output_0_int32,
                      actual_token_num, input_size_13, w13_n_group_stride,
                      actual_n_tile_size);
            gemm_t::dequantize_tile(output_0_int32, w13_output_buffer_0_iter,
                                    input_scales, w13_scale_ptr_0_iter,
                                    actual_token_num, gemm_n_tile_size,
                                    actual_n_tile_size);
            if (w13_bias != nullptr) {
              cpu_micro_gemm::add_bias_epilogue<gemm_n_tile_size>(
                  w13_output_buffer_0_iter, w13_output_buffer_0_iter,
                  w13_bias_ptr_0_iter, actual_token_num, actual_n_tile_size,
                  actual_n_tile_size);
              w13_bias_ptr_0_iter += gemm_n_tile_size;
            }

            auto* output_1_int32 =
                reinterpret_cast<int32_t*>(w13_output_buffer_1_iter);
            gemm.gemm(gemm_input_buffer, w13_weight_ptr_1_iter, output_1_int32,
                      actual_token_num, input_size_13, w13_n_group_stride,
                      actual_n_tile_size);
            gemm_t::dequantize_tile(output_1_int32, w13_output_buffer_1_iter,
                                    input_scales, w13_scale_ptr_1_iter,
                                    actual_token_num, gemm_n_tile_size,
                                    actual_n_tile_size);
            if (w13_bias != nullptr) {
              cpu_micro_gemm::add_bias_epilogue<gemm_n_tile_size>(
                  w13_output_buffer_1_iter, w13_output_buffer_1_iter,
                  w13_bias_ptr_1_iter, actual_token_num, actual_n_tile_size,
                  actual_n_tile_size);
              w13_bias_ptr_1_iter += gemm_n_tile_size;
            }

            w13_weight_ptr_0_iter += w13_n_tile_stride;
            w13_weight_ptr_1_iter += w13_n_tile_stride;
            w13_scale_ptr_0_iter += gemm_n_tile_size;
            w13_scale_ptr_1_iter += gemm_n_tile_size;
            w13_output_buffer_0_iter += gemm_n_tile_size;
            w13_output_buffer_1_iter += gemm_n_tile_size;
          }

          apply_gated_act(act_type, gemm_output_buffer,
                          curr_w13_gemm_output_buffer, actual_token_num,
                          actual_n_tile_size, actual_n_tile_size, input_size_2);
          curr_w13_gemm_output_buffer += gemm_m_tile_size * input_size_2;
        }
      }
    }
  }

  auto* __restrict__ w13_gemm_output_buffer = reinterpret_cast<scalar_t*>(
      common_buffer_start + w13_gemm_output_buffer_offset);
  float* __restrict__ w13_output_scale_buffer = reinterpret_cast<float*>(
      common_buffer_start + w13_output_scale_buffer_offset);

// quantize w2 inputs - in place
#pragma omp parallel for
  for (int32_t token_idx = 0; token_idx < expanded_token_num; ++token_idx) {
    scalar_t* input_row = w13_gemm_output_buffer + token_idx * input_size_2;
    int8_t* output_row = reinterpret_cast<int8_t*>(input_row);
    gemm_t::quantize_row(input_row, output_row,
                         w13_output_scale_buffer[token_idx], input_size_2);
  }

  {
    alignas(64) cpu_utils::Counter counter;
    cpu_utils::Counter* counter_ptr = &counter;

// w2 gemm
#pragma omp parallel for schedule(static, 1)
    for (int32_t thread_id = 0; thread_id < thread_num; ++thread_id) {
      const int32_t task_num_per_expert =
          (output_size_2 + w2_n_tile_size - 1) / w2_n_tile_size;
      const int32_t task_num = task_num_per_expert * expert_num;
      uint8_t* __restrict__ thread_buffer =
          thread_buffer_start + thread_id * thread_buffer_size;
      int8_t* __restrict__ gemm_input_buffer =
          reinterpret_cast<int8_t*>(thread_buffer + gemm_input_buffer_offset);
      float* __restrict__ gemm_output_buffer =
          reinterpret_cast<float*>(thread_buffer + gemm_output_buffer_offset);
      float* __restrict__ w2_gemm_output_buffer = reinterpret_cast<float*>(
          common_buffer_start + w2_gemm_output_buffer_offset);
      gemm_t gemm;

      const int32_t w2_n_group_stride =
          gemm_t::WeightOCGroupSize * input_size_2;
      const int32_t w2_n_tile_stride = gemm_n_tile_size * input_size_2;

      for (;;) {
        const int32_t task_id = counter_ptr->acquire_counter();
        if (task_id >= task_num) {
          break;
        }
        const int32_t curr_expert_id = task_id / task_num_per_expert;
        const int32_t curr_output_group_id = task_id % task_num_per_expert;
        const int32_t curr_token_num =
            token_num_per_group_buffer[curr_expert_id];
        if (curr_token_num == 0) {
          continue;
        }

        const int32_t actual_n_tile_size =
            std::min(w2_n_tile_size,
                     output_size_2 - curr_output_group_id * w2_n_tile_size);
        scalar_t* __restrict__ curr_w13_gemm_output_buffer =
            w13_gemm_output_buffer +
            cu_token_num_per_group_buffer[curr_expert_id] * input_size_2;
        float* __restrict__ curr_w13_output_scale_buffer =
            w13_output_scale_buffer +
            cu_token_num_per_group_buffer[curr_expert_id];
        float* __restrict__ curr_w2_gemm_output_buffer =
            w2_gemm_output_buffer +
            cu_token_num_per_group_buffer[curr_expert_id] * output_size_2 +
            curr_output_group_id * w2_n_tile_size;
        const int8_t* __restrict__ w2_weight_ptr =
            w2 + curr_expert_id * output_size_2 * input_size_2 +
            curr_output_group_id * w2_n_tile_size * input_size_2;
        const float* __restrict__ w2_scale_ptr =
            w2_scales + curr_expert_id * output_size_2 +
            curr_output_group_id * w2_n_tile_size;
        scalar_t* w2_bias_ptr = nullptr;
        if (w2_bias != nullptr) {
          w2_bias_ptr = w2_bias + curr_expert_id * output_size_2 +
                        curr_output_group_id * w2_n_tile_size;
        }

        for (int32_t token_idx = 0; token_idx < curr_token_num;
             token_idx += gemm_m_tile_size) {
          const int32_t actual_token_num =
              std::min(gemm_m_tile_size, curr_token_num - token_idx);
          const int8_t* input_rows[gemm_m_tile_size];
          alignas(64) float input_scales[gemm_m_tile_size];
          for (int32_t i = 0; i < actual_token_num; ++i) {
            input_rows[i] = reinterpret_cast<const int8_t*>(
                curr_w13_gemm_output_buffer + i * input_size_2);
            input_scales[i] = curr_w13_output_scale_buffer[i];
          }
          gemm_t::pack_input_from_rows(input_rows, gemm_input_buffer,
                                       actual_token_num, input_size_2);

          const int8_t* w2_weight_ptr_iter = w2_weight_ptr;
          const float* w2_scale_ptr_iter = w2_scale_ptr;
          scalar_t* w2_bias_ptr_iter = w2_bias_ptr;
          float* curr_w2_gemm_output_buffer_iter = curr_w2_gemm_output_buffer;
          for (int32_t i = 0; i < actual_n_tile_size; i += gemm_n_tile_size) {
            auto* output_int32 = reinterpret_cast<int32_t*>(gemm_output_buffer);
            gemm.gemm(gemm_input_buffer, w2_weight_ptr_iter, output_int32,
                      actual_token_num, input_size_2, w2_n_group_stride,
                      gemm_n_tile_size);
            gemm_t::dequantize_tile(output_int32, gemm_output_buffer,
                                    input_scales, w2_scale_ptr_iter,
                                    actual_token_num, gemm_n_tile_size,
                                    gemm_n_tile_size);
            if (w2_bias != nullptr) {
              cpu_micro_gemm::add_bias_epilogue<gemm_n_tile_size>(
                  gemm_output_buffer, gemm_output_buffer, w2_bias_ptr_iter,
                  actual_token_num, gemm_n_tile_size, gemm_n_tile_size);
              w2_bias_ptr_iter += gemm_n_tile_size;
            }
            for (int32_t m_idx = 0; m_idx < actual_token_num; ++m_idx) {
              std::memcpy(
                  curr_w2_gemm_output_buffer_iter + m_idx * output_size_2,
                  gemm_output_buffer + m_idx * gemm_n_tile_size,
                  gemm_n_tile_size * sizeof(float));
            }

            w2_weight_ptr_iter += w2_n_tile_stride;
            w2_scale_ptr_iter += gemm_n_tile_size;
            curr_w2_gemm_output_buffer_iter += gemm_n_tile_size;
          }

          curr_w13_gemm_output_buffer += gemm_m_tile_size * input_size_2;
          curr_w13_output_scale_buffer += gemm_m_tile_size;
          curr_w2_gemm_output_buffer += gemm_m_tile_size * output_size_2;
        }
      }
    }
  }

  {
    alignas(64) cpu_utils::Counter counter;
    cpu_utils::Counter* counter_ptr = &counter;

#pragma omp parallel for schedule(static, 1)
    for (int32_t thread_id = 0; thread_id < thread_num; ++thread_id) {
      uint8_t* __restrict__ thread_buffer =
          thread_buffer_start + thread_id * thread_buffer_size;
      float* __restrict__ ws_output_buffer =
          reinterpret_cast<float*>(thread_buffer + ws_output_buffer_offset);
      float* __restrict__ w2_gemm_output_buffer = reinterpret_cast<float*>(
          common_buffer_start + w2_gemm_output_buffer_offset);

      for (;;) {
        const int32_t token_id = counter_ptr->acquire_counter();
        if (token_id >= token_num) {
          break;
        }
        int32_t* __restrict__ curr_expand_token_id_index_buffer =
            expand_token_id_index_buffer + token_id * topk_num;
        const float* __restrict__ curr_weight =
            topk_weights + token_id * topk_num;
        const float first_weight = skip_weighted ? 1.0f : curr_weight[0];
        scalar_t* __restrict__ curr_output_buffer =
            output + token_id * output_size_2;

        if (topk_num > 1) {
          int32_t w2_output_idx = curr_expand_token_id_index_buffer[0];
          float* w2_output_iter =
              w2_gemm_output_buffer + w2_output_idx * output_size_2;
          float* ws_output_buffer_iter = ws_output_buffer;
          vec_op::FP32Vec16 weight_vec(first_weight);
          for (int32_t i = 0; i < output_size_2; i += 16) {
            vec_op::FP32Vec16 vec(w2_output_iter);
            (vec * weight_vec).save(ws_output_buffer_iter);
            w2_output_iter += 16;
            ws_output_buffer_iter += 16;
          }

          for (int32_t idx = 1; idx < topk_num - 1; ++idx) {
            w2_output_idx = curr_expand_token_id_index_buffer[idx];
            w2_output_iter =
                w2_gemm_output_buffer + w2_output_idx * output_size_2;
            ws_output_buffer_iter = ws_output_buffer;
            weight_vec = vec_op::FP32Vec16(curr_weight[idx]);
            for (int32_t i = 0; i < output_size_2; i += 16) {
              vec_op::FP32Vec16 vec(w2_output_iter);
              vec_op::FP32Vec16 sum(ws_output_buffer_iter);
              (sum + vec * weight_vec).save(ws_output_buffer_iter);
              w2_output_iter += 16;
              ws_output_buffer_iter += 16;
            }
          }

          const int32_t last_idx = topk_num - 1;
          w2_output_idx = curr_expand_token_id_index_buffer[last_idx];
          w2_output_iter =
              w2_gemm_output_buffer + w2_output_idx * output_size_2;
          ws_output_buffer_iter = ws_output_buffer;
          scalar_t* curr_output_buffer_iter = curr_output_buffer;
          weight_vec = vec_op::FP32Vec16(curr_weight[last_idx]);
          for (int32_t i = 0; i < output_size_2; i += 16) {
            vec_op::FP32Vec16 vec(w2_output_iter);
            vec_op::FP32Vec16 sum(ws_output_buffer_iter);
            scalar_vec_t(sum + vec * weight_vec).save(curr_output_buffer_iter);
            w2_output_iter += 16;
            ws_output_buffer_iter += 16;
            curr_output_buffer_iter += 16;
          }
        } else {
          const int32_t w2_output_idx = curr_expand_token_id_index_buffer[0];
          float* w2_output_iter =
              w2_gemm_output_buffer + w2_output_idx * output_size_2;
          scalar_t* curr_output_buffer_iter = curr_output_buffer;
          vec_op::FP32Vec16 weight_vec(first_weight);
          for (int32_t i = 0; i < output_size_2; i += 16) {
            vec_op::FP32Vec16 vec(w2_output_iter);
            scalar_vec_t(vec * weight_vec).save(curr_output_buffer_iter);
            w2_output_iter += 16;
            curr_output_buffer_iter += 16;
          }
        }
      }
    }
  }
}
}  // namespace

void prepack_moe_weight_int8(
    const torch::Tensor& weight,  // [expert_num, output_size, input_size]
    torch::Tensor& packed_weight, const std::string& isa) {
  TORCH_CHECK(weight.is_contiguous());
  const int32_t expert_num = weight.size(0);
  const int32_t output_size = weight.size(1);
  const int32_t input_size = weight.size(2);
  const int64_t expert_stride = weight.stride(0);
  const cpu_utils::ISA isa_type = cpu_utils::get_isa(isa);
  TORCH_CHECK_EQ(output_size % 32, 0);

  CPU_INT8_ISA_DISPATCH_IMPL(isa_type, c10::BFloat16, [&]() {
    TORCH_CHECK_EQ(input_size % gemm_t::K, 0);
    prepack_moe_weight_int8_impl<gemm_t>(
        weight.data_ptr<int8_t>(), packed_weight.data_ptr<int8_t>(), expert_num,
        output_size, input_size, expert_stride);
  });
}

void cpu_fused_moe_int8(torch::Tensor& output, const torch::Tensor& input,
                        const torch::Tensor& w13, const torch::Tensor& w2,
                        const torch::Tensor& w13_scale,
                        const torch::Tensor& w2_scale,
                        const std::optional<torch::Tensor>& w13_bias,
                        const std::optional<torch::Tensor>& w2_bias,
                        const torch::Tensor& topk_weights,
                        const torch::Tensor& topk_id, const bool skip_weighted,
                        const std::string& act, const std::string& isa) {
  const int32_t token_num = input.size(0);
  const int32_t input_size_13 = input.size(1);
  const int64_t input_stride = input.stride(0);
  TORCH_CHECK_EQ(input_stride, input_size_13);
  const int32_t expert_num = w13.size(0);
  const int32_t output_size_13 = w13.size(1);
  const int32_t input_size_2 = w2.size(2);
  const int32_t output_size_2 = w2.size(1);
  const int32_t topk_num = topk_id.size(1);
  const FusedMOEAct act_type = cpu_fused_moe_utils::get_act_type(act);
  const cpu_utils::ISA isa_type = cpu_utils::get_isa(isa);
  TORCH_CHECK(!skip_weighted || topk_num == 1,
              "skip_weighted is only supported for topk=1 on CPU");

  VLLM_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "cpu_fused_moe_int8", [&]() {
        CPU_INT8_ISA_DISPATCH_IMPL(isa_type, scalar_t, [&]() {
          fused_moe_int8_impl<scalar_t, gemm_t>(
              output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
              w13.data_ptr<int8_t>(), w2.data_ptr<int8_t>(),
              w13_scale.data_ptr<float>(), w2_scale.data_ptr<float>(),
              w13_bias.has_value() ? w13_bias->data_ptr<scalar_t>() : nullptr,
              w2_bias.has_value() ? w2_bias->data_ptr<scalar_t>() : nullptr,
              topk_weights.data_ptr<float>(), topk_id.data_ptr<int32_t>(),
              act_type, token_num, expert_num, topk_num, input_size_13,
              output_size_13, input_size_2, output_size_2, skip_weighted);
        });
      });
}
