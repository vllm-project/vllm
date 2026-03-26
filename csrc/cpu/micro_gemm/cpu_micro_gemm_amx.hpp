#ifndef CPU_MICRO_GEMM_AMX_HPP
#define CPU_MICRO_GEMM_AMX_HPP
#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"

namespace cpu_micro_gemm {
namespace {
// AMX specific
constexpr static int64_t AMX_TILE_ROW_BYTES = 64;
constexpr static int64_t AMX_TILE_ROW_NUM = 16;
constexpr static int64_t AMX_TILE_BYTES = AMX_TILE_ROW_BYTES * AMX_TILE_ROW_NUM;

typedef struct __tile_config {
  uint8_t palette_id = 1;
  uint8_t start_row = 0;
  uint8_t reserved_0[14] = {0};
  uint16_t colsb[16] = {0};
  uint8_t rows[16] = {0};
} __tilecfg;

// 2-2-4 pattern, for 16 < m <= 32
// TILE 0, 1: load A matrix, row num should be 16, m - 16
// TILE 2, 3: load B matrix, row num should be 16
// TILE 4, 5, 6, 7: store results C matrix, row num should be 16, 16, m - 16, m
// - 16
template <typename scalar_t>
class TileGemm224 {
 public:
  FORCE_INLINE static void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    TORCH_CHECK(false, "Unsupported data type for TileGemm224");
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    TORCH_CHECK(false, "Unsupported data type for TileGemm224");
  }
};

template <>
class TileGemm224<c10::BFloat16> {
 public:
  using scalar_t = c10::BFloat16;
  FORCE_INLINE static void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    const int32_t k_times = k / (AMX_TILE_ROW_NUM * 4 / sizeof(c10::BFloat16));
    c10::BFloat16* __restrict__ a_tile_0 = a_ptr;
    c10::BFloat16* __restrict__ a_tile_1 = a_ptr + lda * AMX_TILE_ROW_NUM;
    const int64_t a_tile_stride = lda * sizeof(c10::BFloat16);

    // B is always packed as 16 output channels block
    c10::BFloat16* __restrict__ b_tile_2 = b_ptr;
    c10::BFloat16* __restrict__ b_tile_3 = b_ptr + b_n_group_stride;
    const int32_t b_tile_stride = AMX_TILE_ROW_BYTES;

    float* __restrict__ c_tile_4 = c_ptr;
    float* __restrict__ c_tile_5 =
        c_tile_4 + AMX_TILE_ROW_BYTES / sizeof(float);
    float* __restrict__ c_tile_6 = c_ptr + AMX_TILE_ROW_NUM * ldc;
    float* __restrict__ c_tile_7 =
        c_tile_6 + AMX_TILE_ROW_BYTES / sizeof(float);
    const int32_t c_tile_stride = ldc * sizeof(float);

    if (accum_c) {
      _tile_loadd(4, c_tile_4, c_tile_stride);
      _tile_loadd(5, c_tile_5, c_tile_stride);
      _tile_loadd(6, c_tile_6, c_tile_stride);
      _tile_loadd(7, c_tile_7, c_tile_stride);
    } else {
      _tile_zero(4);
      _tile_zero(5);
      _tile_zero(6);
      _tile_zero(7);
    }

    for (int32_t k = 0; k < k_times; ++k) {
      _tile_loadd(0, a_tile_0, a_tile_stride);
      _tile_stream_loadd(2, b_tile_2, b_tile_stride);
      _tile_dpbf16ps(4, 0, 2);
      _tile_stream_loadd(3, b_tile_3, b_tile_stride);
      _tile_dpbf16ps(5, 0, 3);
      _tile_loadd(1, a_tile_1, a_tile_stride);
      _tile_dpbf16ps(6, 1, 2);
      _tile_dpbf16ps(7, 1, 3);

      // update ptrs
      a_tile_0 += AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      a_tile_1 += AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      b_tile_2 += AMX_TILE_BYTES / sizeof(c10::BFloat16);
      b_tile_3 += AMX_TILE_BYTES / sizeof(c10::BFloat16);
    }

    _tile_stored(4, c_tile_4, c_tile_stride);
    _tile_stored(5, c_tile_5, c_tile_stride);
    _tile_stored(6, c_tile_6, c_tile_stride);
    _tile_stored(7, c_tile_7, c_tile_stride);
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    const int32_t m_0 = AMX_TILE_ROW_NUM;
    const int32_t m_1 = m - AMX_TILE_ROW_NUM;
    config.rows[0] = m_0;
    config.rows[1] = m_1;
    config.rows[2] = AMX_TILE_ROW_NUM;
    config.rows[3] = AMX_TILE_ROW_NUM;
    config.rows[4] = m_0;
    config.rows[5] = m_0;
    config.rows[6] = m_1;
    config.rows[7] = m_1;
    _tile_loadconfig(&config);
  }
};

// 1-2-2 pattern, for 0 < m <= 16
// TILE 0, (1): load A matrix, use extra 1 tile for prefetch, row num should be
// m, m
// TILE 2, 3, (4, 5): load B matrix, use extra 2 tiles for prefetch, row
// num should be 16
// TILE 6, 7, (6, 7): store results C matrix, row num should be
// m
template <typename scalar_t>
class TileGemm122 {
 public:
  FORCE_INLINE static void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    TORCH_CHECK(false, "Unsupported data type for TileGemm122");
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    TORCH_CHECK(false, "Unsupported data type for TileGemm122");
  }
};

template <>
class TileGemm122<c10::BFloat16> {
 public:
  using scalar_t = c10::BFloat16;
  FORCE_INLINE static void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    c10::BFloat16* __restrict__ a_tile_0 = a_ptr;
    c10::BFloat16* __restrict__ a_tile_1 =
        a_ptr + AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
    const int64_t a_tile_stride = lda * sizeof(c10::BFloat16);

    c10::BFloat16* __restrict__ b_tile_2 = b_ptr;
    c10::BFloat16* __restrict__ b_tile_3 = b_ptr + b_n_group_stride;
    c10::BFloat16* __restrict__ b_tile_4 =
        b_tile_2 + AMX_TILE_BYTES / sizeof(c10::BFloat16);
    c10::BFloat16* __restrict__ b_tile_5 =
        b_tile_3 + AMX_TILE_BYTES / sizeof(c10::BFloat16);
    int64_t b_stride = AMX_TILE_ROW_BYTES;

    float* __restrict__ c_tile_6 = c_ptr;
    float* __restrict__ c_tile_7 = c_ptr + AMX_TILE_ROW_BYTES / sizeof(float);
    int64_t c_stride = ldc * sizeof(float);

    const int32_t k_times = k / (AMX_TILE_ROW_NUM * 4 / sizeof(c10::BFloat16));
    const int32_t k_group_times = k_times / 2;
    const bool has_tail = (k_times % 2 == 1);

    if (accum_c) {
      _tile_loadd(6, c_tile_6, c_stride);
      _tile_loadd(7, c_tile_7, c_stride);
    } else {
      _tile_zero(6);
      _tile_zero(7);
    }

    for (int32_t k = 0; k < k_group_times; ++k) {
      _tile_loadd(0, a_tile_0, a_tile_stride);
      _tile_stream_loadd(2, b_tile_2, b_stride);
      _tile_dpbf16ps(6, 0, 2);
      _tile_stream_loadd(3, b_tile_3, b_stride);
      _tile_dpbf16ps(7, 0, 3);
      _tile_loadd(1, a_tile_1, a_tile_stride);
      _tile_stream_loadd(4, b_tile_4, b_stride);
      _tile_dpbf16ps(6, 1, 4);
      _tile_stream_loadd(5, b_tile_5, b_stride);
      _tile_dpbf16ps(7, 1, 5);

      // update ptrs
      a_tile_0 += 2 * AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      a_tile_1 += 2 * AMX_TILE_ROW_BYTES / sizeof(c10::BFloat16);
      b_tile_2 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
      b_tile_3 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
      b_tile_4 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
      b_tile_5 += 2 * AMX_TILE_BYTES / sizeof(c10::BFloat16);
    }

    if (has_tail) {
      _tile_loadd(0, a_tile_0, a_tile_stride);
      _tile_stream_loadd(2, b_tile_2, b_stride);
      _tile_dpbf16ps(6, 0, 2);
      _tile_stream_loadd(3, b_tile_3, b_stride);
      _tile_dpbf16ps(7, 0, 3);
    }

    _tile_stored(6, c_tile_6, c_stride);
    _tile_stored(7, c_tile_7, c_stride);
  }

  FORCE_INLINE static void init_tile_config(int32_t m, __tilecfg& config) {
    config.rows[0] = m;
    config.rows[1] = m;
    config.rows[2] = AMX_TILE_ROW_NUM;
    config.rows[3] = AMX_TILE_ROW_NUM;
    config.rows[4] = AMX_TILE_ROW_NUM;
    config.rows[5] = AMX_TILE_ROW_NUM;
    config.rows[6] = m;
    config.rows[7] = m;
    _tile_loadconfig(&config);
  }
};
}  // namespace

// Gemm kernel uses AMX, requires B matrix to be packed
template <typename scalar_t>
class MicroGemm<cpu_utils::ISA::AMX, scalar_t> {
 public:
  static constexpr int32_t MaxMSize = 32;
  static constexpr int32_t NSize = 32;

 public:
  MicroGemm() : curr_m_(-1) {
    vec_op::unroll_loop<int, 8>([&](int i) { amx_tile_config_.colsb[i] = 64; });
  }

  void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    if (m > AMX_TILE_ROW_NUM) {
      if (m != curr_m_) {
        curr_m_ = m;
        TileGemm224<scalar_t>::init_tile_config(m, amx_tile_config_);
      }
      TileGemm224<scalar_t>::gemm(CPU_MICRO_GEMM_PARAMS);
    } else {
      if (m != curr_m_) {
        curr_m_ = m;
        TileGemm122<scalar_t>::init_tile_config(m, amx_tile_config_);
      }
      TileGemm122<scalar_t>::gemm(CPU_MICRO_GEMM_PARAMS);
    }
  }

  static void pack_weight(const scalar_t* __restrict__ weight,
                          scalar_t* __restrict__ packed_weight,
                          const int32_t output_size, const int32_t input_size) {
    constexpr int32_t elem_num_per_group = 4 / sizeof(scalar_t);
    TORCH_CHECK_EQ(output_size % 16, 0);
    TORCH_CHECK_EQ(input_size % (16 * elem_num_per_group), 0);

    const int32_t output_group_num = output_size / 16;
    const int32_t input_32b_num = input_size / elem_num_per_group;
    for (int32_t output_group_idx = 0; output_group_idx < output_group_num;
         ++output_group_idx) {
      const int32_t* __restrict__ weight_32b =
          reinterpret_cast<const int32_t*>(weight);
      int32_t* __restrict__ packed_weight_32b =
          reinterpret_cast<int32_t*>(packed_weight);
      for (int32_t output_idx = 0; output_idx < 16; ++output_idx) {
        for (int32_t weight_offset = 0, packed_offset = 0;
             weight_offset < input_32b_num;
             ++weight_offset, packed_offset += 16) {
          packed_weight_32b[packed_offset] = weight_32b[weight_offset];
        }

        // update
        weight_32b += input_32b_num;
        packed_weight_32b += 1;
      }

      // update
      weight += 16 * input_size;
      packed_weight += 16 * input_size;
    }
  }

 private:
  alignas(64) __tilecfg amx_tile_config_;
  int32_t curr_m_;
};

}  // namespace cpu_micro_gemm

#endif
