#include "cpu/cpu_types.hpp"
#include "cpu/utils.hpp"
#include "cpu/micro_gemm/cpu_micro_gemm_vec.hpp"
#include "cpu/cpu_arch_macros.h"

#ifdef CPU_CAPABILITY_AMXBF16
  #include "cpu/micro_gemm/cpu_micro_gemm_amx.hpp"
  #define AMX_DISPATCH(...)                                                    \
    case cpu_utils::ISA::AMX: {                                                \
      using gemm_t = cpu_micro_gemm::MicroGemm<cpu_utils::ISA::AMX, scalar_t>; \
      return __VA_ARGS__();                                                    \
    }
#else
  #define AMX_DISPATCH(...) case cpu_utils::ISA::AMX:
#endif

#define CPU_ISA_DISPATCH_IMPL(ISA_TYPE, ...)                          \
  [&] {                                                               \
    switch (ISA_TYPE) {                                               \
      AMX_DISPATCH(__VA_ARGS__)                                       \
      case cpu_utils::ISA::VEC: {                                     \
        using gemm_t =                                                \
            cpu_micro_gemm::MicroGemm<cpu_utils::ISA::VEC, scalar_t>; \
        return __VA_ARGS__();                                         \
      }                                                               \
      default: {                                                      \
        TORCH_CHECK(false, "Invalid CPU ISA type.");                  \
      }                                                               \
    }                                                                 \
  }()

namespace {
enum class FusedMOEAct { SiluAndMul, SwigluOAIAndMul };

FusedMOEAct get_act_type(const std::string& act) {
  if (act == "silu") {
    return FusedMOEAct::SiluAndMul;
  } else if (act == "swigluoai") {
    return FusedMOEAct::SwigluOAIAndMul;
  } else {
    TORCH_CHECK(false, "Invalid act type: " + act);
  }
}

template <typename scalar_t>
void swigluoai_and_mul(float* __restrict__ input, scalar_t* __restrict__ output,
                       const int32_t m_size, const int32_t n_size,
                       const int32_t input_stride,
                       const int32_t output_stride) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  // For GPT-OSS interleaved gate-up weights
  alignas(64) static int32_t index[16] = {0,  2,  4,  6,  8,  10, 12, 14,
                                          16, 18, 20, 22, 24, 26, 28, 30};
  vec_op::INT32Vec16 index_vec(index);
  vec_op::FP32Vec16 gate_up_max_vec(7.0);
  vec_op::FP32Vec16 up_min_vec(-7.0);
  vec_op::FP32Vec16 alpha_vec(1.702);
  vec_op::FP32Vec16 one_vec(1.0);

  DEFINE_FAST_EXP

  for (int32_t m = 0; m < m_size; ++m) {
    for (int32_t n = 0; n < n_size; n += 32) {
      vec_op::FP32Vec16 gate_vec(input + n, index_vec);
      vec_op::FP32Vec16 up_vec(input + n + 1, index_vec);
      gate_vec = gate_vec.min(gate_up_max_vec);
      up_vec = up_vec.clamp(up_min_vec, gate_up_max_vec);
      auto sigmoid_vec = one_vec / (one_vec + fast_exp(-gate_vec * alpha_vec));
      auto glu = gate_vec * sigmoid_vec;
      auto gated_output_fp32 = (one_vec + up_vec) * glu;
      scalar_vec_t gated_output = scalar_vec_t(gated_output_fp32);
      gated_output.save(output + n / 2);
    }
    input += input_stride;
    output += output_stride;
  }
}

template <typename scalar_t>
void silu_and_mul(float* __restrict__ input, scalar_t* __restrict__ output,
                  const int32_t m_size, const int32_t n_size,
                  const int32_t input_stride, const int32_t output_stride) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  const int32_t dim = n_size / 2;
  float* __restrict__ gate = input;
  float* __restrict__ up = input + dim;
  vec_op::FP32Vec16 one_vec(1.0);

  DEFINE_FAST_EXP

  for (int32_t m = 0; m < m_size; ++m) {
    for (int32_t n = 0; n < dim; n += 16) {
      vec_op::FP32Vec16 gate_vec(gate + n);
      vec_op::FP32Vec16 up_vec(up + n);
      auto sigmoid_vec = one_vec / (one_vec + fast_exp(-gate_vec));
      auto silu = gate_vec * sigmoid_vec;
      auto gated_output_fp32 = up_vec * silu;
      scalar_vec_t gated_output = scalar_vec_t(gated_output_fp32);
      gated_output.save(output + n);
    }
    gate += input_stride;
    up += input_stride;
    output += output_stride;
  }
}

template <typename scalar_t>
FORCE_INLINE void apply_gated_act(const FusedMOEAct act,
                                  float* __restrict__ input,
                                  scalar_t* __restrict__ output,
                                  const int32_t m, const int32_t n,
                                  const int32_t input_stride,
                                  const int32_t output_stride) {
  switch (act) {
    case FusedMOEAct::SwigluOAIAndMul:
      swigluoai_and_mul(input, output, m, n, input_stride, output_stride);
      return;
    case FusedMOEAct::SiluAndMul:
      silu_and_mul(input, output, m, n, input_stride, output_stride);
      return;
    default:
      TORCH_CHECK(false, "Unsupported act type.");
  }
}

template <typename scalar_t, typename gemm_t>
void prepack_moe_weight_impl(scalar_t* __restrict__ weight_ptr,
                             scalar_t* __restrict__ packed_weight_ptr,
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

template <typename scalar_t, typename w_t, typename gemm_t>
void fused_moe_impl(scalar_t* __restrict__ output, scalar_t* __restrict__ input,
                    w_t* __restrict__ w13, w_t* __restrict__ w2,
                    w_t* __restrict__ w13_bias, w_t* __restrict__ w2_bias,
                    float* __restrict__ topk_weights,
                    int32_t* __restrict__ topk_id, FusedMOEAct act_type,
                    const int32_t token_num, const int32_t expert_num,
                    const int32_t topk_num, const int32_t input_size_13,
                    const int32_t output_size_13, const int32_t input_size_2,
                    const int32_t output_size_2) {
  using scalar_vec_t = typename cpu_utils::VecTypeTrait<scalar_t>::vec_t;
  constexpr int32_t gemm_n_tile_size = gemm_t::NSize;
  constexpr int32_t gemm_m_tile_size = gemm_t::MaxMSize;
  constexpr int32_t min_w13_n_tile_size = 2 * gemm_n_tile_size;
  static_assert(gemm_n_tile_size % 16 == 0);

  TORCH_CHECK_EQ(output_size_13 % min_w13_n_tile_size, 0);
  TORCH_CHECK_EQ(output_size_2 % gemm_n_tile_size, 0);
  TORCH_CHECK_EQ(output_size_13 / 2, input_size_2);

  const int32_t thread_num = omp_get_max_threads();

  const int32_t w13_input_buffer_size = cpu_utils::round_up<64>(
      gemm_m_tile_size * input_size_13 * sizeof(scalar_t));

  const int32_t w13_n_tile_size = [&]() {
    const int64_t cache_size = cpu_utils::get_available_l2_size();
    // input buffer + output buffer + weight
    const int32_t n_size_cache_limit =
        (cache_size - w13_input_buffer_size) /
        (gemm_m_tile_size * sizeof(float) + input_size_13 * sizeof(scalar_t));
    const int32_t n_size_thread_limit =
        output_size_13 / std::max(1, thread_num / topk_num);
    const int32_t n_size = cpu_utils::round_down<min_w13_n_tile_size>(
        std::min(n_size_cache_limit, n_size_thread_limit));
    return std::max(n_size, min_w13_n_tile_size);
  }();

  const int32_t w2_input_tile_size = cpu_utils::round_up<64>(
      gemm_m_tile_size * input_size_2 * sizeof(scalar_t));

  const int32_t w2_n_tile_size = [&]() {
    const int64_t cache_size = cpu_utils::get_available_l2_size();
    // input tile + weight
    const int32_t n_size_cache_limit =
        (cache_size - w2_input_tile_size) / (input_size_2 * sizeof(scalar_t));
    const int32_t n_size_thread_limit =
        output_size_2 / std::max(1, thread_num / topk_num);
    const int32_t n_size = cpu_utils::round_down<gemm_n_tile_size>(
        std::min(n_size_cache_limit, n_size_thread_limit));
    return std::max(n_size, gemm_n_tile_size);
  }();

  // allocate buffers
  int32_t common_buffer_offset = 0;
  int32_t w13_thread_buffer_offset = 0;
  int32_t ws_thread_buffer_offset = 0;

  // common buffers
  const int32_t token_num_per_group_buffer_size =
      cpu_utils::round_up<64>(expert_num * sizeof(int32_t));
  const int32_t token_num_per_group_buffer_offset = common_buffer_offset;
  common_buffer_offset += token_num_per_group_buffer_size;

  const int32_t cu_token_num_per_group_buffer_size =
      cpu_utils::round_up<64>((expert_num + 1) * sizeof(int32_t));
  const int32_t cu_token_num_per_group_buffer_offset = common_buffer_offset;
  common_buffer_offset += cu_token_num_per_group_buffer_size;

  const int32_t expand_token_id_buffer_size =
      cpu_utils::round_up<64>(token_num * topk_num * sizeof(int32_t));
  const int32_t expand_token_id_buffer_offset = common_buffer_offset;
  common_buffer_offset += expand_token_id_buffer_size;

  const int32_t expand_token_id_index_buffer_size =
      cpu_utils::round_up<64>(token_num * topk_num * sizeof(int32_t));
  const int32_t expand_token_id_index_buffer_offset = common_buffer_offset;
  common_buffer_offset += expand_token_id_index_buffer_size;

  const int32_t w13_gemm_output_buffer_size = cpu_utils::round_up<64>(
      token_num * topk_num * (output_size_13 / 2) * sizeof(scalar_t));
  const int32_t w13_gemm_output_buffer_offset = common_buffer_offset;
  common_buffer_offset += w13_gemm_output_buffer_size;

  const int32_t w2_gemm_output_buffer_size = cpu_utils::round_up<64>(
      token_num * topk_num * output_size_2 * sizeof(float));
  const int32_t w2_gemm_output_buffer_offset = common_buffer_offset;
  common_buffer_offset += w2_gemm_output_buffer_size;

  // w13 GEMM thread buffers
  const int32_t w13_input_buffer_offset = w13_thread_buffer_offset;
  w13_thread_buffer_offset += w13_input_buffer_size;

  const int32_t w13_output_buffer_size = cpu_utils::round_up<64>(
      gemm_m_tile_size * w13_n_tile_size * sizeof(float));
  const int32_t w13_output_buffer_offset = w13_thread_buffer_offset;
  w13_thread_buffer_offset += w13_output_buffer_size;

  // Weighted sum thread buffer
  const int32_t ws_output_buffer_size =
      cpu_utils::round_up<64>(output_size_2 * sizeof(float));
  const int32_t ws_output_buffer_offset = ws_thread_buffer_offset;
  ws_thread_buffer_offset += ws_output_buffer_size;

  const int32_t buffer_size =
      common_buffer_offset +
      std::max(w13_thread_buffer_offset, ws_thread_buffer_offset) * thread_num;
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

  // prepare token-expert mappings
  {
    std::memset(token_num_per_group_buffer, 0, expert_num * sizeof(int32_t));
    for (int32_t i = 0; i < token_num * topk_num; ++i) {
      int32_t curr_expert_id = topk_id[i];
      ++token_num_per_group_buffer[curr_expert_id];
    }

    int32_t token_num_sum = 0;
    cu_token_num_per_group_buffer[0] = 0;
    int32_t* token_index_buffer = cu_token_num_per_group_buffer + 1;
    for (int32_t i = 0; i < expert_num; ++i) {
      token_index_buffer[i] = token_num_sum;
      token_num_sum += token_num_per_group_buffer[i];
    }

    for (int32_t i = 0; i < token_num; ++i) {
      int32_t* curr_topk_id = topk_id + i * topk_num;
      int32_t* curr_index_buffer = expand_token_id_index_buffer + i * topk_num;
      for (int32_t j = 0; j < topk_num; ++j) {
        int32_t curr_expert_id = curr_topk_id[j];
        int32_t curr_index = token_index_buffer[curr_expert_id];
        ++token_index_buffer[curr_expert_id];
        expand_token_id_buffer[curr_index] = i;
        curr_index_buffer[j] = curr_index;
      }
    }
  }

  // w13 GEMM + act
  {
    alignas(64) cpu_utils::Counter counter;
    cpu_utils::Counter* counter_ptr = &counter;

#pragma omp parallel for schedule(static, 1)
    for (int32_t thread_id = 0; thread_id < thread_num; ++thread_id) {
      const int32_t task_num_per_expert =
          (output_size_13 + w13_n_tile_size - 1) / w13_n_tile_size;
      const int32_t task_num = task_num_per_expert * expert_num;

      uint8_t* __restrict__ thread_buffer =
          thread_buffer_start + thread_id * w13_thread_buffer_offset;
      scalar_t* __restrict__ w13_input_buffer =
          reinterpret_cast<scalar_t*>(thread_buffer + w13_input_buffer_offset);
      float* __restrict__ w13_output_buffer =
          reinterpret_cast<float*>(thread_buffer + w13_output_buffer_offset);
      scalar_t* __restrict__ w13_gemm_output_buffer =
          reinterpret_cast<scalar_t*>(common_buffer_start +
                                      w13_gemm_output_buffer_offset);

      gemm_t gemm;

      const int32_t input_size_13_bytes = input_size_13 * sizeof(scalar_t);
      const int32_t w13_n_group_stride = 16 * input_size_13;
      const int32_t w13_n_tile_stride = gemm_n_tile_size * input_size_13;

      for (;;) {
        int32_t task_id = counter_ptr->acquire_counter();
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
            cu_token_num_per_group_buffer[curr_expert_id] *
                (output_size_13 / 2) +
            curr_output_group_id * w13_n_tile_size / 2;

        w_t* __restrict__ w13_weight_ptr_0 = nullptr;
        w_t* __restrict__ w13_weight_ptr_1 = nullptr;
        w_t* __restrict__ w13_bias_ptr_0 = nullptr;
        w_t* __restrict__ w13_bias_ptr_1 = nullptr;
        if (act_type == FusedMOEAct::SwigluOAIAndMul) {
          // For SwigluOAIAndMul, up and down weights are interleaved
          w13_weight_ptr_0 =
              w13 + curr_expert_id * input_size_13 * output_size_13 +
              curr_output_group_id * w13_n_tile_size * input_size_13;
          w13_weight_ptr_1 =
              w13_weight_ptr_0 + actual_n_tile_size / 2 * input_size_13;
          if (w13_bias != nullptr) {
            w13_bias_ptr_0 = w13_bias + curr_expert_id * output_size_13 +
                             curr_output_group_id * w13_n_tile_size;
            w13_bias_ptr_1 = w13_bias_ptr_0 + actual_n_tile_size / 2;
          }
        } else {
          w13_weight_ptr_0 =
              w13 + curr_expert_id * input_size_13 * output_size_13 +
              curr_output_group_id * (w13_n_tile_size / 2) * input_size_13;
          w13_weight_ptr_1 =
              w13_weight_ptr_0 + output_size_13 / 2 * input_size_13;
          if (w13_bias != nullptr) {
            w13_bias_ptr_0 = w13_bias + curr_expert_id * output_size_13 +
                             curr_output_group_id * (w13_n_tile_size / 2);
            w13_bias_ptr_1 = w13_bias_ptr_0 + output_size_13 / 2;
          }
        }

        scalar_t* __restrict__ curr_w13_input_buffer = w13_input_buffer;
        for (int32_t token_idx = 0; token_idx < curr_token_num;
             token_idx += gemm_m_tile_size) {
          const int32_t actual_token_num =
              std::min(gemm_m_tile_size, curr_token_num - token_idx);
          // copy inputs
          {
            scalar_t* __restrict__ curr_w13_input_buffer_iter =
                curr_w13_input_buffer;
            for (int32_t i = 0; i < actual_token_num; ++i) {
              const int32_t curr_token_id = curr_expand_token_id_buffer[i];
              int8_t* __restrict__ curr_input_iter = reinterpret_cast<int8_t*>(
                  input + curr_token_id * input_size_13);
              int8_t* __restrict__ curr_output_iter =
                  reinterpret_cast<int8_t*>(curr_w13_input_buffer_iter);
              int32_t j = 0;
              for (; j < input_size_13_bytes - 64; j += 64) {
                vec_op::INT8Vec64 vec(curr_input_iter);
                vec.save(curr_output_iter);
                curr_input_iter += 64;
                curr_output_iter += 64;
              }
              vec_op::INT8Vec64 vec(curr_input_iter);
              vec.save(curr_output_iter, input_size_13_bytes - j);

              // update
              curr_w13_input_buffer_iter += input_size_13;
            }
            // update
            curr_expand_token_id_buffer += actual_token_num;
          }

          // gemm + act
          {
            scalar_t* __restrict__ w13_weight_ptr_0_iter = w13_weight_ptr_0;
            scalar_t* __restrict__ w13_weight_ptr_1_iter = w13_weight_ptr_1;
            scalar_t* __restrict__ w13_bias_ptr_0_iter = w13_bias_ptr_0;
            scalar_t* __restrict__ w13_bias_ptr_1_iter = w13_bias_ptr_1;
            scalar_t* __restrict__ curr_w13_input_buffer_iter =
                curr_w13_input_buffer;
            float* __restrict__ w13_output_buffer_0_iter = w13_output_buffer;
            float* __restrict__ w13_output_buffer_1_iter =
                w13_output_buffer + actual_n_tile_size / 2;
            for (int32_t i = 0; i < actual_n_tile_size;
                 i += min_w13_n_tile_size) {
              gemm.gemm(curr_w13_input_buffer_iter, w13_weight_ptr_0_iter,
                        w13_output_buffer_0_iter, actual_token_num,
                        input_size_13, input_size_13, w13_n_group_stride,
                        actual_n_tile_size, false);

              if (w13_bias != nullptr) {
                cpu_micro_gemm::add_bias_epilogue<gemm_n_tile_size>(
                    w13_output_buffer_0_iter, w13_output_buffer_0_iter,
                    w13_bias_ptr_0_iter, actual_token_num, actual_n_tile_size,
                    actual_n_tile_size);
                w13_bias_ptr_0_iter += gemm_n_tile_size;
              }

              gemm.gemm(curr_w13_input_buffer_iter, w13_weight_ptr_1_iter,
                        w13_output_buffer_1_iter, actual_token_num,
                        input_size_13, input_size_13, w13_n_group_stride,
                        actual_n_tile_size, false);

              if (w13_bias != nullptr) {
                cpu_micro_gemm::add_bias_epilogue<gemm_n_tile_size>(
                    w13_output_buffer_1_iter, w13_output_buffer_1_iter,
                    w13_bias_ptr_1_iter, actual_token_num, actual_n_tile_size,
                    actual_n_tile_size);
                w13_bias_ptr_1_iter += gemm_n_tile_size;
              }

              // update
              w13_weight_ptr_0_iter += w13_n_tile_stride;
              w13_weight_ptr_1_iter += w13_n_tile_stride;
              w13_output_buffer_0_iter += gemm_n_tile_size;
              w13_output_buffer_1_iter += gemm_n_tile_size;
            }

            apply_gated_act(act_type, w13_output_buffer,
                            curr_w13_gemm_output_buffer, actual_token_num,
                            actual_n_tile_size, actual_n_tile_size,
                            output_size_13 / 2);

            // update
            curr_w13_gemm_output_buffer +=
                gemm_m_tile_size * (output_size_13 / 2);
          }
        }
      }
    }
  }

  // w2 GEMM
  {
    alignas(64) cpu_utils::Counter counter;
    cpu_utils::Counter* counter_ptr = &counter;

#pragma omp parallel for schedule(static, 1)
    for (int32_t thread_id = 0; thread_id < thread_num; ++thread_id) {
      const int32_t task_num_per_expert =
          (output_size_2 + w2_n_tile_size - 1) / w2_n_tile_size;
      const int32_t task_num = task_num_per_expert * expert_num;
      scalar_t* __restrict__ w13_gemm_output_buffer =
          reinterpret_cast<scalar_t*>(common_buffer_start +
                                      w13_gemm_output_buffer_offset);
      float* __restrict__ w2_gemm_output_buffer = reinterpret_cast<float*>(
          common_buffer_start + w2_gemm_output_buffer_offset);

      gemm_t gemm;

      const int32_t w2_n_tile_stride = gemm_n_tile_size * input_size_2;
      const int32_t w2_n_group_stride = 16 * input_size_2;

      for (;;) {
        int32_t task_id = counter_ptr->acquire_counter();
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
        float* __restrict__ curr_w2_gemm_output_buffer =
            w2_gemm_output_buffer +
            cu_token_num_per_group_buffer[curr_expert_id] * output_size_2 +
            curr_output_group_id * w2_n_tile_size;
        scalar_t* __restrict__ w2_weight_ptr =
            w2 + curr_expert_id * output_size_2 * input_size_2 +
            curr_output_group_id * w2_n_tile_size * input_size_2;
        scalar_t* __restrict__ w2_bias_ptr = nullptr;
        if (w2_bias != nullptr) {
          w2_bias_ptr = w2_bias + curr_expert_id * output_size_2 +
                        curr_output_group_id * w2_n_tile_size;
        }

        for (int32_t token_idx = 0; token_idx < curr_token_num;
             token_idx += gemm_m_tile_size) {
          const int32_t actual_token_num =
              std::min(gemm_m_tile_size, curr_token_num - token_idx);

          scalar_t* __restrict__ w2_weight_ptr_iter = w2_weight_ptr;
          scalar_t* __restrict__ w2_bias_ptr_iter = w2_bias_ptr;
          float* __restrict__ curr_w2_gemm_output_buffer_iter =
              curr_w2_gemm_output_buffer;
          for (int32_t i = 0; i < actual_n_tile_size; i += gemm_n_tile_size) {
            gemm.gemm(curr_w13_gemm_output_buffer, w2_weight_ptr_iter,
                      curr_w2_gemm_output_buffer_iter, actual_token_num,
                      input_size_2, input_size_2, w2_n_group_stride,
                      output_size_2, false);

            if (w2_bias != nullptr) {
              cpu_micro_gemm::add_bias_epilogue<gemm_n_tile_size>(
                  curr_w2_gemm_output_buffer_iter,
                  curr_w2_gemm_output_buffer_iter, w2_bias_ptr_iter,
                  actual_token_num, output_size_2, output_size_2);
              w2_bias_ptr_iter += gemm_n_tile_size;
            }

            w2_weight_ptr_iter += w2_n_tile_stride;
            curr_w2_gemm_output_buffer_iter += gemm_n_tile_size;
          }

          // update
          curr_w13_gemm_output_buffer += gemm_m_tile_size * input_size_2;
          curr_w2_gemm_output_buffer += gemm_m_tile_size * output_size_2;
        }
      }
    }
  }

  // weighted sum
  {
    alignas(64) cpu_utils::Counter counter;
    cpu_utils::Counter* counter_ptr = &counter;

#pragma omp parallel for schedule(static, 1)
    for (int32_t thread_id = 0; thread_id < thread_num; ++thread_id) {
      const int32_t task_num = token_num;
      uint8_t* __restrict__ thread_buffer =
          thread_buffer_start + thread_id * ws_thread_buffer_offset;
      float* __restrict__ ws_output_buffer =
          reinterpret_cast<float*>(thread_buffer + ws_output_buffer_offset);
      float* __restrict__ w2_gemm_output_buffer = reinterpret_cast<float*>(
          common_buffer_start + w2_gemm_output_buffer_offset);

      for (;;) {
        int32_t task_id = counter_ptr->acquire_counter();
        if (task_id >= task_num) {
          break;
        }

        int32_t token_id = task_id;
        int32_t* __restrict__ curr_expand_token_id_index_buffer =
            expand_token_id_index_buffer + token_id * topk_num;
        float* __restrict__ curr_weight = topk_weights + token_id * topk_num;
        scalar_t* __restrict__ curr_output_buffer =
            output + token_id * output_size_2;

        if (topk_num > 1) {
          {
            int32_t w2_output_idx = curr_expand_token_id_index_buffer[0];
            float* __restrict__ w2_output_iter =
                w2_gemm_output_buffer + w2_output_idx * output_size_2;
            float* __restrict__ ws_output_buffer_iter = ws_output_buffer;
            vec_op::FP32Vec16 weight_vec(curr_weight[0]);
            for (int32_t i = 0; i < output_size_2; i += 16) {
              vec_op::FP32Vec16 vec(w2_output_iter);
              vec = vec * weight_vec;
              vec.save(ws_output_buffer_iter);

              // update
              w2_output_iter += 16;
              ws_output_buffer_iter += 16;
            }
          }

          {
            for (int32_t idx = 1; idx < topk_num - 1; ++idx) {
              int32_t w2_output_idx = curr_expand_token_id_index_buffer[idx];
              float* __restrict__ w2_output_iter =
                  w2_gemm_output_buffer + w2_output_idx * output_size_2;
              float* __restrict__ ws_output_buffer_iter = ws_output_buffer;
              vec_op::FP32Vec16 weight_vec(curr_weight[idx]);
              for (int32_t i = 0; i < output_size_2; i += 16) {
                vec_op::FP32Vec16 vec(w2_output_iter);
                vec_op::FP32Vec16 sum(ws_output_buffer_iter);
                sum = sum + vec * weight_vec;
                sum.save(ws_output_buffer_iter);

                // update
                w2_output_iter += 16;
                ws_output_buffer_iter += 16;
              }
            }
          }

          {
            int32_t idx = topk_num - 1;
            int32_t w2_output_idx = curr_expand_token_id_index_buffer[idx];
            float* __restrict__ w2_output_iter =
                w2_gemm_output_buffer + w2_output_idx * output_size_2;
            float* __restrict__ ws_output_buffer_iter = ws_output_buffer;
            scalar_t* __restrict__ curr_output_buffer_iter = curr_output_buffer;
            vec_op::FP32Vec16 weight_vec(curr_weight[idx]);
            for (int32_t i = 0; i < output_size_2; i += 16) {
              vec_op::FP32Vec16 vec(w2_output_iter);
              vec_op::FP32Vec16 sum(ws_output_buffer_iter);
              sum = sum + vec * weight_vec;
              scalar_vec_t out_vec(sum);
              out_vec.save(curr_output_buffer_iter);

              // update
              w2_output_iter += 16;
              ws_output_buffer_iter += 16;
              curr_output_buffer_iter += 16;
            }
          }
        } else {
          int32_t w2_output_idx = curr_expand_token_id_index_buffer[0];
          float* __restrict__ w2_output_iter =
              w2_gemm_output_buffer + w2_output_idx * output_size_2;
          scalar_t* __restrict__ curr_output_buffer_iter = curr_output_buffer;
          vec_op::FP32Vec16 weight_vec(curr_weight[0]);
          for (int32_t i = 0; i < output_size_2; i += 16) {
            vec_op::FP32Vec16 vec(w2_output_iter);
            vec = vec * weight_vec;
            scalar_vec_t out_vec(vec);
            out_vec.save(curr_output_buffer_iter);

            // update
            w2_output_iter += 16;
            curr_output_buffer_iter += 16;
          }
        }
      }
    }
  }
}
}  // namespace

void prepack_moe_weight(
    const torch::Tensor& weight,  // [expert_num, output_size, input_size]
    torch::Tensor& packed_weight, const std::string& isa) {
  TORCH_CHECK(weight.is_contiguous());
  const int32_t expert_num = weight.size(0);
  const int32_t output_size = weight.size(1);
  const int32_t input_size = weight.size(2);
  TORCH_CHECK_EQ(output_size % 32, 0);
  const int64_t expert_stride = weight.stride(0);
  cpu_utils::ISA isa_type = cpu_utils::get_isa(isa);

  VLLM_DISPATCH_FLOATING_TYPES(
      weight.scalar_type(), "prepack_moe_weight", [&]() {
        CPU_ISA_DISPATCH_IMPL(isa_type, [&]() {
          scalar_t* weight_ptr = weight.data_ptr<scalar_t>();
          scalar_t* packed_weight_ptr = packed_weight.data_ptr<scalar_t>();
          prepack_moe_weight_impl<scalar_t, gemm_t>(
              weight_ptr, packed_weight_ptr, expert_num, output_size,
              input_size, expert_stride);
        });
      });
}

void cpu_fused_moe(
    torch::Tensor& output,       // [token_num, output_size_2]
    const torch::Tensor& input,  // [token_num, input_size_13]
    const torch::Tensor&
        w13,  // [expert_num, output_size_13, input_size_13], packed
    const torch::Tensor&
        w2,  // [expert_num, output_size_2, input_size_2], packed
    const std::optional<torch::Tensor>&
        w13_bias,  // [expert_num, output_size_13]
    const std::optional<torch::Tensor>& w2_bias,  // [expert_num, output_size_2]
    const torch::Tensor& topk_weights,            // [token_num, k], float32
    const torch::Tensor& topk_id,                 // [token_num, k], int32
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
  const FusedMOEAct act_type = get_act_type(act);
  cpu_utils::ISA isa_type = cpu_utils::get_isa(isa);

  VLLM_DISPATCH_FLOATING_TYPES(w13.scalar_type(), "cpu_fused_moe", [&]() {
    CPU_ISA_DISPATCH_IMPL(isa_type, [&]() {
      fused_moe_impl<scalar_t, scalar_t, gemm_t>(
          output.data_ptr<scalar_t>(), input.data_ptr<scalar_t>(),
          w13.data_ptr<scalar_t>(), w2.data_ptr<scalar_t>(),
          w13_bias.has_value() ? w13_bias->data_ptr<scalar_t>() : nullptr,
          w2_bias.has_value() ? w2_bias->data_ptr<scalar_t>() : nullptr,
          topk_weights.data_ptr<float>(), topk_id.data_ptr<int32_t>(), act_type,
          token_num, expert_num, topk_num, input_size_13, output_size_13,
          input_size_2, output_size_2);
    });
  });
}
