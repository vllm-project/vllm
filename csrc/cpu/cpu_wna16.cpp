#include "cpu/cpu_types.hpp"
#include "cpu/utils.hpp"

#ifdef CPU_CAPABILITY_AMXBF16
  #include "cpu/micro_gemm/cpu_micro_gemm_amx.hpp"
#endif
#include "cpu/micro_gemm/cpu_micro_gemm_vec.hpp"

#define VLLM_DISPATCH_CASE_16B_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)

#define VLLM_DISPATCH_16B_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, VLLM_DISPATCH_CASE_16B_TYPES(__VA_ARGS__))

template <typename T>
void print_logits(const char* name, T* ptr, int32_t row, int32_t col,
                  int32_t stride) {
  std::stringstream ss;
  ss << std::fixed << std::setprecision(5) << name << ": [\n";
  auto* curr_logits_buffer = ptr;
  for (int32_t m = 0; m < row; ++m) {
    for (int32_t n = 0; n < col; ++n) {
      ss << curr_logits_buffer[n] << ", ";
    }
    ss << "\n";
    curr_logits_buffer += stride;
  }
  ss << "]\n";
  std::printf("%s", ss.str().c_str());
}

namespace {
using cpu_utils::ISA;
using cpu_utils::VecTypeTrait;

template <typename scalar_t, ISA isa, bool has_zp, bool use_desc_act>
class Dequantizer4b {
 public:
  constexpr static int32_t pack_num = 32 / 4;
  using scalar_vec_t = typename VecTypeTrait<scalar_t>::vec_t;

 public:
  static void dequant(int32_t* __restrict__ q_weight,
                      scalar_t* __restrict__ weight,
                      scalar_t* __restrict__ scales,
                      int32_t* __restrict__ zeros, int32_t* __restrict__ g_idx,
                      const int64_t scales_stride, const int64_t zeros_stride,
                      const int32_t k_size, const int32_t group_size) {
    vec_op::FP32Vec16 lut;
    if constexpr (has_zp) {
      // AWQ
      alignas(64) static const float LUT[16] = {
          0.0f, 1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,
          8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f};
      lut = vec_op::FP32Vec16(LUT);
    } else {
      // GPTQ
      alignas(64) static const float LUT[16] = {
          -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
          0.0f,  1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f};
      lut = vec_op::FP32Vec16(LUT);
    }

    // per 64-bits elem contains 16 output channels
    int64_t* __restrict__ curr_q_weight = reinterpret_cast<int64_t*>(q_weight);
    int64_t* __restrict__ curr_zeros = reinterpret_cast<int64_t*>(zeros);
    scalar_t* __restrict__ curr_weight = weight;
    scalar_t* __restrict__ curr_scale = scales;
    vec_op::FP32Vec16 scale_0;
    vec_op::FP32Vec16 scale_1;
    vec_op::FP32Vec16 zero_0;
    vec_op::FP32Vec16 zero_1;
    int32_t group_counter = 0;
    for (int32_t k_idx = 0; k_idx < k_size; k_idx += 2) {
      int64_t qwb_0 = *curr_q_weight;
      int64_t qwb_1 = *(curr_q_weight + 1);
      vec_op::FP32Vec16 wb_0(qwb_0, lut);
      vec_op::FP32Vec16 wb_1(qwb_1, lut);

      if constexpr (!use_desc_act) {
        if (group_counter == 0) {
          scale_0 = vec_op::FP32Vec16(scalar_vec_t(curr_scale));
          scale_1 = vec_op::FP32Vec16(scale_0);
          curr_scale += scales_stride;

          if constexpr (has_zp) {
            zero_0 = vec_op::FP32Vec16(*curr_zeros, lut);
            zero_1 = vec_op::FP32Vec16(zero_0);
            curr_zeros += zeros_stride / 2;
          }
        }
      } else {
        int32_t g_idx_0 = g_idx[k_idx];
        int32_t g_idx_1 = g_idx[k_idx + 1];
        scale_0 = vec_op::FP32Vec16(
            scalar_vec_t(curr_scale + g_idx_0 * scales_stride));
        scale_1 = vec_op::FP32Vec16(
            scalar_vec_t(curr_scale + g_idx_1 * scales_stride));
        if constexpr (has_zp) {
          zero_0 = vec_op::FP32Vec16(*(curr_zeros + g_idx_0 * zeros_stride / 2),
                                     lut);
          zero_1 = vec_op::FP32Vec16(*(curr_zeros + g_idx_1 * zeros_stride / 2),
                                     lut);
        }
      }

      if constexpr (has_zp) {
        wb_0 = wb_0 - zero_0;
        wb_1 = wb_1 - zero_1;
      }

      wb_0 = wb_0 * scale_0;
      wb_1 = wb_1 * scale_1;

      scalar_vec_t output_vec_0(wb_0);
      scalar_vec_t output_vec_1(wb_1);

      // AMX needs to interlave K elements to pack as 32 bits
      if constexpr (isa == ISA::AMX) {
        vec_op::interleave_save(output_vec_0, output_vec_1, curr_weight);
      } else {
        output_vec_0.save(curr_weight);
        output_vec_1.save(curr_weight + 16);
      }

      // update
      curr_q_weight += 2;
      curr_weight += 32;
      if constexpr (!use_desc_act) {
        group_counter += 2;
        if (group_counter == group_size) {
          group_counter = 0;
        }
      }
    }
  }
};
};  // namespace

template <typename scalar_t, typename dequantizer_t, typename gemm_t>
void cpu_gemm_wna16_impl(
    scalar_t* __restrict__ input, int32_t* __restrict__ q_weight,
    scalar_t* __restrict__ output, scalar_t* __restrict__ scales,
    int32_t* __restrict__ zeros, int32_t* __restrict__ g_idx,
    scalar_t* __restrict__ bias, const int32_t m_size, const int32_t n_size,
    const int32_t k_size, const int64_t input_stride,
    const int64_t output_stride, const int64_t scales_group_stride,
    const int64_t zeros_group_stride, const int32_t group_num,
    const int32_t group_size, const int64_t pack_factor) {
  constexpr int32_t gemm_n_tile_size = gemm_t::NSize;
  constexpr int32_t gemm_m_tile_size = gemm_t::MaxMSize;
  constexpr int32_t n_block_size = 16;
  static_assert(gemm_n_tile_size % n_block_size == 0);
  const int32_t thread_num = omp_get_max_threads();

  // a simple schedule policy, just to hold more B tiles in L2 and make sure
  // each thread has tasks
  const int32_t n_partition_size = [&]() {
    const int64_t cache_size = cpu_utils::get_available_l2_size();
    int64_t ps_cache_limit = cache_size / (k_size * sizeof(scalar_t));
    int64_t ps_thread_limit = n_size / thread_num;
    ps_cache_limit =
        std::max((ps_cache_limit / gemm_n_tile_size) * gemm_n_tile_size,
                 (int64_t)gemm_n_tile_size);
    ps_thread_limit =
        std::max((ps_thread_limit / gemm_n_tile_size) * gemm_n_tile_size,
                 (int64_t)gemm_n_tile_size);
    return std::min(ps_cache_limit, ps_thread_limit);
  }();
  const int32_t task_num = (n_size + n_partition_size - 1) / n_partition_size;

  // get buffer size
  const int64_t b_buffer_size =
      (((n_partition_size * k_size * sizeof(scalar_t) + 63) / 64) * 64);
  const int64_t c_buffer_size =
      (((gemm_m_tile_size * gemm_n_tile_size * sizeof(float) + 63) / 64) * 64);
  const int64_t b_buffer_offset = 0;
  const int64_t c_buffer_offset = b_buffer_size;
  const int64_t buffer_size = b_buffer_size + c_buffer_size;
  cpu_utils::ScratchPadManager::get_scratchpad_manager()->realloc(buffer_size *
                                                                  thread_num);

  alignas(64) cpu_utils::Counter counter;
  cpu_utils::Counter* counter_ptr = &counter;

#pragma omp parallel for schedule(static, 1)
  for (int32_t thread_id = 0; thread_id < thread_num; ++thread_id) {
    scalar_t* __restrict__ b_buffer = nullptr;
    float* __restrict__ c_buffer = nullptr;
    {
      uint8_t* buffer_ptr =
          cpu_utils::ScratchPadManager::get_scratchpad_manager()
              ->get_data<uint8_t>() +
          thread_id * buffer_size;
      b_buffer = reinterpret_cast<scalar_t*>(buffer_ptr + b_buffer_offset);
      c_buffer = reinterpret_cast<float*>(buffer_ptr + c_buffer_offset);
    }

    const int64_t q_weight_block_stride = n_block_size / pack_factor * k_size;
    const int64_t b_buffer_block_stride = n_block_size * k_size;
    const int32_t zeros_block_stride = n_block_size / pack_factor;

    gemm_t gemm;

    for (;;) {
      int32_t task_id = counter_ptr->acquire_counter();

      if (task_id >= task_num) {
        break;
      }

      const int32_t n_start_idx = task_id * n_partition_size;
      const int32_t n_block_start_idx = n_start_idx / n_block_size;
      const int32_t n_num = std::min(n_partition_size, n_size - n_start_idx);
      const int32_t n_block_num = n_num / n_block_size;
      // std::printf("thread_id: %d, task_id: %d, n_start_idx: %d, n_num: %d\n",
      // thread_id, task_id, n_start_idx, n_num);

      // dequant weight
      {
        int32_t* __restrict__ curr_q_weight =
            q_weight + n_block_start_idx * q_weight_block_stride;
        scalar_t* __restrict__ curr_b_buffer = b_buffer;
        scalar_t* __restrict__ curr_scales = scales + n_start_idx;
        int32_t* __restrict__ curr_zeros = zeros + n_start_idx / pack_factor;
        for (int32_t block_idx = 0; block_idx < n_block_num; ++block_idx) {
          dequantizer_t::dequant(curr_q_weight, curr_b_buffer, curr_scales,
                                 curr_zeros, g_idx, scales_group_stride,
                                 zeros_group_stride, k_size, group_size);

          // if (block_idx == 0 && n_start_idx == 0) {
          //     print_logits("depacked weight", curr_b_buffer, k_size,
          //     n_block_size, n_block_size);
          // }

          // update
          curr_q_weight += q_weight_block_stride;
          curr_b_buffer += b_buffer_block_stride;
          curr_scales += n_block_size;
          curr_zeros += zeros_block_stride;
        }
      }

      // compute loop
      {
        const int32_t n_tile_num = n_num / gemm_n_tile_size;
        scalar_t* __restrict__ curr_input = input;
        scalar_t* __restrict__ init_bias = bias;
        if (bias != nullptr) {
          init_bias += n_start_idx;
        }
        scalar_t* __restrict__ init_output = output + n_start_idx;
        for (int32_t m_idx = 0; m_idx < m_size; m_idx += gemm_m_tile_size) {
          const int32_t curr_m_size =
              std::min(gemm_m_tile_size, m_size - m_idx);
          scalar_t* __restrict__ curr_b_buffer = b_buffer;
          scalar_t* __restrict__ curr_bias = init_bias;
          scalar_t* __restrict__ curr_output = init_output;
          for (int32_t n_tile_idx = 0; n_tile_idx < n_tile_num; ++n_tile_idx) {
            gemm.gemm(curr_input, curr_b_buffer, c_buffer, curr_m_size, k_size,
                      input_stride, b_buffer_block_stride, gemm_n_tile_size,
                      false);

            if (bias != nullptr) {
              cpu_micro_gemm::bias_epilogue<gemm_n_tile_size>(
                  c_buffer, curr_output, curr_bias, curr_m_size,
                  gemm_n_tile_size, output_stride);
              curr_bias += gemm_n_tile_size;
            } else {
              cpu_micro_gemm::default_epilogue<gemm_n_tile_size>(
                  c_buffer, curr_output, curr_m_size, gemm_n_tile_size,
                  output_stride);
            }

            curr_b_buffer +=
                b_buffer_block_stride * (gemm_n_tile_size / n_block_size);
            curr_output += gemm_n_tile_size;
          }
          curr_input += gemm_m_tile_size * input_stride;
          init_output += gemm_m_tile_size * output_stride;
        }
      }
    }
  }
}

void cpu_gemm_wna16(
    const torch::Tensor& input,  // [M, K]
    const torch::Tensor&
        q_weight,           // [N / 16, K * 16 / pack_factor], packed as int32
    torch::Tensor& output,  // [M, N]
    const torch::Tensor& scales,  // [group_num, N]
    const std::optional<torch::Tensor>&
        zeros,  // [group_num, N / pack_factor], packed as int32
    const std::optional<torch::Tensor>& g_idx,  // [K]
    const std::optional<torch::Tensor>& bias,   // [N]
    const int64_t pack_factor, const std::string& isa_hint) {
  using cpu_utils::ISA;
  TORCH_CHECK_EQ(pack_factor, 8);  // only supports 4bits
  const int32_t a_m_size = input.size(0);
  const int32_t a_k_size = input.size(1);
  const int64_t a_m_stride = input.stride(0);
  const int32_t b_n_size = q_weight.size(0) * 16;
  TORCH_CHECK_EQ(a_k_size % 32, 0);
  TORCH_CHECK_EQ(b_n_size % 32, 0);
  const int32_t group_num = scales.size(0);
  const int32_t group_size = a_k_size / group_num;
  TORCH_CHECK_EQ(group_size % 2, 0);
  const int64_t scales_group_stride = scales.stride(0);
  const int64_t output_m_stride = output.stride(0);

  bool has_zp = zeros.has_value();
  bool use_desc_act = g_idx.has_value();
  TORCH_CHECK(!(has_zp && use_desc_act));

  ISA isa = [&]() {
    if (isa_hint == "amx") {
      return ISA::AMX;
    } else if (isa_hint == "vec") {
      return ISA::VEC;
    } else {
      TORCH_CHECK(false, "unsupported isa hint: " + isa_hint);
    }
  }();

  int32_t* zeros_ptr = has_zp ? zeros->data_ptr<int32_t>() : nullptr;
  const int64_t zeros_group_stride = has_zp ? zeros->stride(0) : 0;
  int32_t* g_idx_ptr = use_desc_act ? g_idx->data_ptr<int32_t>() : nullptr;

  VLLM_DISPATCH_16B_TYPES(input.scalar_type(), "cpu_gemm_wna16", [&]() {
    if (isa == ISA::AMX) {
      using gemm_t = cpu_micro_gemm::MicroGemm<ISA::AMX, scalar_t>;
      if (has_zp) {
        using dequantizer_t = Dequantizer4b<scalar_t, ISA::AMX, true, false>;
        cpu_gemm_wna16_impl<scalar_t, dequantizer_t, gemm_t>(
            input.data_ptr<scalar_t>(), q_weight.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(), zeros_ptr,
            g_idx_ptr, bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
            a_m_size, b_n_size, a_k_size, a_m_stride, output_m_stride,
            scales_group_stride, zeros_group_stride, group_num, group_size,
            pack_factor);
        return;
      }
      if (use_desc_act) {
        using dequantizer_t = Dequantizer4b<scalar_t, ISA::AMX, false, true>;
        cpu_gemm_wna16_impl<scalar_t, dequantizer_t, gemm_t>(
            input.data_ptr<scalar_t>(), q_weight.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(), zeros_ptr,
            g_idx_ptr, bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
            a_m_size, b_n_size, a_k_size, a_m_stride, output_m_stride,
            scales_group_stride, zeros_group_stride, group_num, group_size,
            pack_factor);
        return;
      } else {
        using dequantizer_t = Dequantizer4b<scalar_t, ISA::AMX, false, false>;
        cpu_gemm_wna16_impl<scalar_t, dequantizer_t, gemm_t>(
            input.data_ptr<scalar_t>(), q_weight.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(), zeros_ptr,
            g_idx_ptr, bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
            a_m_size, b_n_size, a_k_size, a_m_stride, output_m_stride,
            scales_group_stride, zeros_group_stride, group_num, group_size,
            pack_factor);
        return;
      }
    } else if (isa == ISA::VEC) {
      using gemm_t = cpu_micro_gemm::MicroGemm<ISA::VEC, scalar_t>;
      if (has_zp) {
        using dequantizer_t = Dequantizer4b<scalar_t, ISA::VEC, true, false>;
        cpu_gemm_wna16_impl<scalar_t, dequantizer_t, gemm_t>(
            input.data_ptr<scalar_t>(), q_weight.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(), zeros_ptr,
            g_idx_ptr, bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
            a_m_size, b_n_size, a_k_size, a_m_stride, output_m_stride,
            scales_group_stride, zeros_group_stride, group_num, group_size,
            pack_factor);
        return;
      }
      if (use_desc_act) {
        using dequantizer_t = Dequantizer4b<scalar_t, ISA::VEC, false, true>;
        cpu_gemm_wna16_impl<scalar_t, dequantizer_t, gemm_t>(
            input.data_ptr<scalar_t>(), q_weight.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(), zeros_ptr,
            g_idx_ptr, bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
            a_m_size, b_n_size, a_k_size, a_m_stride, output_m_stride,
            scales_group_stride, zeros_group_stride, group_num, group_size,
            pack_factor);
        return;
      } else {
        using dequantizer_t = Dequantizer4b<scalar_t, ISA::VEC, false, false>;
        cpu_gemm_wna16_impl<scalar_t, dequantizer_t, gemm_t>(
            input.data_ptr<scalar_t>(), q_weight.data_ptr<int32_t>(),
            output.data_ptr<scalar_t>(), scales.data_ptr<scalar_t>(), zeros_ptr,
            g_idx_ptr, bias.has_value() ? bias->data_ptr<scalar_t>() : nullptr,
            a_m_size, b_n_size, a_k_size, a_m_stride, output_m_stride,
            scales_group_stride, zeros_group_stride, group_num, group_size,
            pack_factor);
        return;
      }
    }
  });
}
