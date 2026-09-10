#ifndef CPU_MICRO_GEMM_VSX_HPP
#define CPU_MICRO_GEMM_VSX_HPP
#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"
#include <altivec.h>

namespace cpu_micro_gemm {
namespace {

// 8-16-16 pattern, 2 regs for A, 4 regs for B, 8 regs for C, [8, K] @ [K, 16]
template <typename scalar_t>
class TileGemmVSX {
 public:
  FORCE_INLINE static void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    switch (m) {
      case 1:
        gemm_micro<1>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 2:
        gemm_micro<2>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 3:
        gemm_micro<3>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 4:
        gemm_micro<4>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 5:
        gemm_micro<5>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 6:
        gemm_micro<6>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 7:
        gemm_micro<7>(CPU_MICRO_GEMM_PARAMS);
        break;
      case 8:
        gemm_micro<8>(CPU_MICRO_GEMM_PARAMS);
        break;
    }
  }

  template <int32_t M>
  static void gemm_micro_vsx_fallback(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    __vector float c_regs[M][4];

    if (accum_c) {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < 4; j++) {
          c_regs[i][j] = (__vector float)vec_xl(0, &c_ptr[i * ldc + j * 4]);
        }
      }
    } else {
      for (int i = 0; i < M; i++) {
        for (int j = 0; j < 4; j++) {
          c_regs[i][j] = (__vector float){0.0f, 0.0f, 0.0f, 0.0f};
        }
      }
    }

    const __vector unsigned short vzero = {0};
    const __vector unsigned char mask_k0 = {0,  1,  2,  3,  8,  9,  10, 11,
                                            16, 17, 18, 19, 24, 25, 26, 27};
    const __vector unsigned char mask_k1 = {4,  5,  6,  7,  12, 13, 14, 15,
                                            20, 21, 22, 23, 28, 29, 30, 31};

    for (int32_t k_idx = 0; k_idx < k; k_idx += 2) {
      __vector float B_k0[4];
      __vector float B_k1[4];

      for (int j = 0; j < 4; j++) {
        __vector unsigned short vB_short = (__vector unsigned short)vec_xl(
            0, (const unsigned char*)&b_ptr[k_idx * 16 + j * 8]);
        __vector unsigned int vB_f32_01 =
            (__vector unsigned int)vec_mergeh(vzero, vB_short);
        __vector unsigned int vB_f32_23 =
            (__vector unsigned int)vec_mergel(vzero, vB_short);

        B_k0[j] = vec_perm((__vector float)vB_f32_01, (__vector float)vB_f32_23,
                           mask_k0);
        B_k1[j] = vec_perm((__vector float)vB_f32_01, (__vector float)vB_f32_23,
                           mask_k1);
      }

      for (int i = 0; i < M; i++) {
        uint32_t a_val;
        std::memcpy(&a_val, &a_ptr[i * lda + k_idx], 4);
        uint32_t a0_u = (a_val & 0xFFFF) << 16;
        uint32_t a1_u = a_val & 0xFFFF0000;
        float a0_f, a1_f;
        std::memcpy(&a0_f, &a0_u, 4);
        std::memcpy(&a1_f, &a1_u, 4);

        __vector float vA0 = vec_splats(a0_f);
        __vector float vA1 = vec_splats(a1_f);

#pragma GCC unroll 4
        for (int j = 0; j < 4; j++) {
          c_regs[i][j] = vec_madd(vA0, B_k0[j], c_regs[i][j]);
          c_regs[i][j] = vec_madd(vA1, B_k1[j], c_regs[i][j]);
        }
      }
    }

    for (int i = 0; i < M; i++) {
      for (int j = 0; j < 4; j++) {
        vec_xst(c_regs[i][j], 0, &c_ptr[i * ldc + j * 4]);
      }
    }
  }

  template <int32_t M>
  static void gemm_micro(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    static_assert(0 < M && M <= 8);

    if constexpr (M <= 4) {
      gemm_micro_vsx_fallback<M>(CPU_MICRO_GEMM_PARAMS);
      return;
    }

    // Calculate how many 4x4 tiles we need in M dimension
    constexpr int tiles_m = (M + 3) / 4;
    constexpr int tiles_n =
        4;  // Since NSize=16, we always process 4 tiles in N dimension

    __vector_quad acc[tiles_m][tiles_n];
    for (int i = 0; i < tiles_m; i++) {
      for (int j = 0; j < tiles_n; j++) {
        __builtin_mma_xxsetaccz(&acc[i][j]);
      }
    }

    // Load initial accumulators if accum_c is true
    if (accum_c) {
      typedef float v4sf __attribute__((vector_size(16)));
      for (int i = 0; i < tiles_m; i++) {
        for (int j = 0; j < tiles_n; j++) {
          v4sf tmp[4];
          for (int r = 0; r < 4; r++) {
            int r_idx = i * 4 + r;
            if (r_idx < M) {
              tmp[r] = (v4sf)vec_xl(0, &c_ptr[r_idx * ldc + j * 4]);
            } else {
              tmp[r] = (v4sf){0.0f, 0.0f, 0.0f, 0.0f};
            }
          }
          __builtin_mma_build_acc(&acc[i][j], (__vector unsigned char)tmp[0],
                                  (__vector unsigned char)tmp[1],
                                  (__vector unsigned char)tmp[2],
                                  (__vector unsigned char)tmp[3]);
        }
      }
    }

    int32_t k_idx = 0;
    const __vector unsigned short vzero = {0};

    // Main unrolled loop processing 8 columns (4 k-iterations) at a time
    for (; k_idx <= k - 8; k_idx += 8) {
      __vector unsigned int vA_0[tiles_m];
      __vector unsigned int vA_2[tiles_m];
      __vector unsigned int vA_4[tiles_m];
      __vector unsigned int vA_6[tiles_m];

      if constexpr (M >= 1) {
        __vector unsigned short v0;
        if constexpr (M >= 1)
          v0 = (__vector unsigned short)vec_xl(
              0, (const unsigned char*)&a_ptr[0 * lda + k_idx]);
        else
          v0 = vzero;
        __vector unsigned short v1;
        if constexpr (M >= 2)
          v1 = (__vector unsigned short)vec_xl(
              0, (const unsigned char*)&a_ptr[1 * lda + k_idx]);
        else
          v1 = vzero;
        __vector unsigned short v2;
        if constexpr (M >= 3)
          v2 = (__vector unsigned short)vec_xl(
              0, (const unsigned char*)&a_ptr[2 * lda + k_idx]);
        else
          v2 = vzero;
        __vector unsigned short v3;
        if constexpr (M >= 4)
          v3 = (__vector unsigned short)vec_xl(
              0, (const unsigned char*)&a_ptr[3 * lda + k_idx]);
        else
          v3 = vzero;

        __vector unsigned int u0 = (__vector unsigned int)v0;
        __vector unsigned int u1 = (__vector unsigned int)v1;
        __vector unsigned int u2 = (__vector unsigned int)v2;
        __vector unsigned int u3 = (__vector unsigned int)v3;

        __vector unsigned int h01 = vec_mergeh(u0, u1);
        __vector unsigned int h23 = vec_mergeh(u2, u3);
        __vector unsigned int l01 = vec_mergel(u0, u1);
        __vector unsigned int l23 = vec_mergel(u2, u3);

        vA_0[0] = (__vector unsigned int)vec_mergeh(
            (__vector unsigned long long)h01, (__vector unsigned long long)h23);
        vA_2[0] = (__vector unsigned int)vec_mergel(
            (__vector unsigned long long)h01, (__vector unsigned long long)h23);
        vA_4[0] = (__vector unsigned int)vec_mergeh(
            (__vector unsigned long long)l01, (__vector unsigned long long)l23);
        vA_6[0] = (__vector unsigned int)vec_mergel(
            (__vector unsigned long long)l01, (__vector unsigned long long)l23);
      }

      if constexpr (M >= 5) {
        __vector unsigned short v4;
        if constexpr (M >= 5)
          v4 = (__vector unsigned short)vec_xl(
              0, (const unsigned char*)&a_ptr[4 * lda + k_idx]);
        else
          v4 = vzero;
        __vector unsigned short v5;
        if constexpr (M >= 6)
          v5 = (__vector unsigned short)vec_xl(
              0, (const unsigned char*)&a_ptr[5 * lda + k_idx]);
        else
          v5 = vzero;
        __vector unsigned short v6;
        if constexpr (M >= 7)
          v6 = (__vector unsigned short)vec_xl(
              0, (const unsigned char*)&a_ptr[6 * lda + k_idx]);
        else
          v6 = vzero;
        __vector unsigned short v7;
        if constexpr (M >= 8)
          v7 = (__vector unsigned short)vec_xl(
              0, (const unsigned char*)&a_ptr[7 * lda + k_idx]);
        else
          v7 = vzero;

        __vector unsigned int u4 = (__vector unsigned int)v4;
        __vector unsigned int u5 = (__vector unsigned int)v5;
        __vector unsigned int u6 = (__vector unsigned int)v6;
        __vector unsigned int u7 = (__vector unsigned int)v7;

        __vector unsigned int h45 = vec_mergeh(u4, u5);
        __vector unsigned int h67 = vec_mergeh(u6, u7);
        __vector unsigned int l45 = vec_mergel(u4, u5);
        __vector unsigned int l67 = vec_mergel(u6, u7);

        vA_0[1] = (__vector unsigned int)vec_mergeh(
            (__vector unsigned long long)h45, (__vector unsigned long long)h67);
        vA_2[1] = (__vector unsigned int)vec_mergel(
            (__vector unsigned long long)h45, (__vector unsigned long long)h67);
        vA_4[1] = (__vector unsigned int)vec_mergeh(
            (__vector unsigned long long)l45, (__vector unsigned long long)l67);
        vA_6[1] = (__vector unsigned int)vec_mergel(
            (__vector unsigned long long)l45, (__vector unsigned long long)l67);
      }

      // Load B and GER for k_idx + 0
      __vector unsigned char vB_vec_0[tiles_n];
      for (int j = 0; j < tiles_n; j++)
        vB_vec_0[j] =
            vec_xl(0, (const unsigned char*)&b_ptr[(k_idx + 0) * 16 + j * 8]);
      for (int i = 0; i < tiles_m; i++)
        for (int j = 0; j < tiles_n; j++)
          __builtin_mma_xvbf16ger2pp(
              &acc[i][j], (__vector unsigned char)vA_0[i], vB_vec_0[j]);

      // Load B and GER for k_idx + 2
      __vector unsigned char vB_vec_2[tiles_n];
      for (int j = 0; j < tiles_n; j++)
        vB_vec_2[j] =
            vec_xl(0, (const unsigned char*)&b_ptr[(k_idx + 2) * 16 + j * 8]);
      for (int i = 0; i < tiles_m; i++)
        for (int j = 0; j < tiles_n; j++)
          __builtin_mma_xvbf16ger2pp(
              &acc[i][j], (__vector unsigned char)vA_2[i], vB_vec_2[j]);

      // Load B and GER for k_idx + 4
      __vector unsigned char vB_vec_4[tiles_n];
      for (int j = 0; j < tiles_n; j++)
        vB_vec_4[j] =
            vec_xl(0, (const unsigned char*)&b_ptr[(k_idx + 4) * 16 + j * 8]);
      for (int i = 0; i < tiles_m; i++)
        for (int j = 0; j < tiles_n; j++)
          __builtin_mma_xvbf16ger2pp(
              &acc[i][j], (__vector unsigned char)vA_4[i], vB_vec_4[j]);

      // Load B and GER for k_idx + 6
      __vector unsigned char vB_vec_6[tiles_n];
      for (int j = 0; j < tiles_n; j++)
        vB_vec_6[j] =
            vec_xl(0, (const unsigned char*)&b_ptr[(k_idx + 6) * 16 + j * 8]);
      for (int i = 0; i < tiles_m; i++)
        for (int j = 0; j < tiles_n; j++)
          __builtin_mma_xvbf16ger2pp(
              &acc[i][j], (__vector unsigned char)vA_6[i], vB_vec_6[j]);
    }

    // Remainder loop processing 2 columns at a time
    for (; k_idx < k; k_idx += 2) {
      __vector unsigned int vA_uint[tiles_m];
      if constexpr (M == 1) {
        uint32_t val0;
        std::memcpy(&val0, &a_ptr[0 * lda + k_idx], 4);
        vA_uint[0] = (__vector unsigned int){val0, 0, 0, 0};
      } else if constexpr (M == 2) {
        uint32_t val0, val1;
        std::memcpy(&val0, &a_ptr[0 * lda + k_idx], 4);
        std::memcpy(&val1, &a_ptr[1 * lda + k_idx], 4);
        vA_uint[0] = (__vector unsigned int){val0, val1, 0, 0};
      } else if constexpr (M == 3) {
        uint32_t val0, val1, val2;
        std::memcpy(&val0, &a_ptr[0 * lda + k_idx], 4);
        std::memcpy(&val1, &a_ptr[1 * lda + k_idx], 4);
        std::memcpy(&val2, &a_ptr[2 * lda + k_idx], 4);
        vA_uint[0] = (__vector unsigned int){val0, val1, val2, 0};
      } else if constexpr (M == 4) {
        uint32_t val0, val1, val2, val3;
        std::memcpy(&val0, &a_ptr[0 * lda + k_idx], 4);
        std::memcpy(&val1, &a_ptr[1 * lda + k_idx], 4);
        std::memcpy(&val2, &a_ptr[2 * lda + k_idx], 4);
        std::memcpy(&val3, &a_ptr[3 * lda + k_idx], 4);
        vA_uint[0] = (__vector unsigned int){val0, val1, val2, val3};
      } else if constexpr (M == 5) {
        uint32_t val0, val1, val2, val3, val4;
        std::memcpy(&val0, &a_ptr[0 * lda + k_idx], 4);
        std::memcpy(&val1, &a_ptr[1 * lda + k_idx], 4);
        std::memcpy(&val2, &a_ptr[2 * lda + k_idx], 4);
        std::memcpy(&val3, &a_ptr[3 * lda + k_idx], 4);
        vA_uint[0] = (__vector unsigned int){val0, val1, val2, val3};
        std::memcpy(&val4, &a_ptr[4 * lda + k_idx], 4);
        vA_uint[1] = (__vector unsigned int){val4, 0, 0, 0};
      } else if constexpr (M == 6) {
        uint32_t val0, val1, val2, val3, val4, val5;
        std::memcpy(&val0, &a_ptr[0 * lda + k_idx], 4);
        std::memcpy(&val1, &a_ptr[1 * lda + k_idx], 4);
        std::memcpy(&val2, &a_ptr[2 * lda + k_idx], 4);
        std::memcpy(&val3, &a_ptr[3 * lda + k_idx], 4);
        vA_uint[0] = (__vector unsigned int){val0, val1, val2, val3};
        std::memcpy(&val4, &a_ptr[4 * lda + k_idx], 4);
        std::memcpy(&val5, &a_ptr[5 * lda + k_idx], 4);
        vA_uint[1] = (__vector unsigned int){val4, val5, 0, 0};
      } else if constexpr (M == 7) {
        uint32_t val0, val1, val2, val3, val4, val5, val6;
        std::memcpy(&val0, &a_ptr[0 * lda + k_idx], 4);
        std::memcpy(&val1, &a_ptr[1 * lda + k_idx], 4);
        std::memcpy(&val2, &a_ptr[2 * lda + k_idx], 4);
        std::memcpy(&val3, &a_ptr[3 * lda + k_idx], 4);
        vA_uint[0] = (__vector unsigned int){val0, val1, val2, val3};
        std::memcpy(&val4, &a_ptr[4 * lda + k_idx], 4);
        std::memcpy(&val5, &a_ptr[5 * lda + k_idx], 4);
        std::memcpy(&val6, &a_ptr[6 * lda + k_idx], 4);
        vA_uint[1] = (__vector unsigned int){val4, val5, val6, 0};
      } else if constexpr (M == 8) {
        uint32_t val0, val1, val2, val3, val4, val5, val6, val7;
        std::memcpy(&val0, &a_ptr[0 * lda + k_idx], 4);
        std::memcpy(&val1, &a_ptr[1 * lda + k_idx], 4);
        std::memcpy(&val2, &a_ptr[2 * lda + k_idx], 4);
        std::memcpy(&val3, &a_ptr[3 * lda + k_idx], 4);
        vA_uint[0] = (__vector unsigned int){val0, val1, val2, val3};
        std::memcpy(&val4, &a_ptr[4 * lda + k_idx], 4);
        std::memcpy(&val5, &a_ptr[5 * lda + k_idx], 4);
        std::memcpy(&val6, &a_ptr[6 * lda + k_idx], 4);
        std::memcpy(&val7, &a_ptr[7 * lda + k_idx], 4);
        vA_uint[1] = (__vector unsigned int){val4, val5, val6, val7};
      }

      // Load packed B
      __vector unsigned char vB_vec[tiles_n];
      for (int j = 0; j < tiles_n; j++) {
        vB_vec[j] = vec_xl(0, (const unsigned char*)&b_ptr[k_idx * 16 + j * 8]);
      }

      for (int i = 0; i < tiles_m; i++) {
        for (int j = 0; j < tiles_n; j++) {
          __builtin_mma_xvbf16ger2pp(
              &acc[i][j], (__vector unsigned char)vA_uint[i], vB_vec[j]);
        }
      }
    }

    // Disassemble and store
    typedef float v4sf __attribute__((vector_size(16)));
    for (int i = 0; i < tiles_m; i++) {
      for (int j = 0; j < tiles_n; j++) {
        v4sf tmp[4];
        __builtin_mma_disassemble_acc(tmp, &acc[i][j]);
        for (int r = 0; r < 4; r++) {
          int r_idx = i * 4 + r;
          if (r_idx < M) {
            vec_xst((__vector float)tmp[r], 0, &c_ptr[r_idx * ldc + j * 4]);
          }
        }
      }
    }
  }
};
}  // namespace

// Gemm kernel uses MMA instructions, requires B matrix to be packed
template <typename scalar_t>
class MicroGemm<cpu_utils::ISA::VSX, scalar_t> {
 public:
  static constexpr int32_t MaxMSize = 8;
  static constexpr int32_t NSize = 16;
  static constexpr int32_t WeightOCGroupSize = 16;
  static constexpr bool PackA = false;

 public:
  void gemm(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    TileGemmVSX<scalar_t>::gemm(CPU_MICRO_GEMM_PARAMS);
  }

  // Pack weight:
  // Original B is (output_size, input_size).
  // We need to pack it for MMA. MMA `xvbf16ger2pp` requires 4 output columns
  // and 2 input columns in a single VecBF16. We pack into shape: [output_size /
  // 4, input_size / 2, 8]
  static void pack_weight(const scalar_t* __restrict__ weight,
                          scalar_t* __restrict__ packed_weight,
                          const int32_t output_size, const int32_t input_size) {
    TORCH_CHECK_EQ(output_size % NSize, 0);
    TORCH_CHECK_EQ(input_size % 2, 0);

    scalar_t* pw = packed_weight;
    for (int32_t o_block = 0; o_block < output_size; o_block += NSize) {
      for (int32_t i_idx = 0; i_idx < input_size; i_idx += 2) {
        for (int32_t o_idx = o_block; o_idx < o_block + NSize; o_idx += 4) {
          *pw++ = weight[(o_idx + 0) * input_size + i_idx];
          *pw++ = weight[(o_idx + 0) * input_size + i_idx + 1];

          *pw++ = weight[(o_idx + 1) * input_size + i_idx];
          *pw++ = weight[(o_idx + 1) * input_size + i_idx + 1];

          *pw++ = weight[(o_idx + 2) * input_size + i_idx];
          *pw++ = weight[(o_idx + 2) * input_size + i_idx + 1];

          *pw++ = weight[(o_idx + 3) * input_size + i_idx];
          *pw++ = weight[(o_idx + 3) * input_size + i_idx + 1];
        }
      }
    }
  }
};
}  // namespace cpu_micro_gemm

#endif
