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
      case 1: gemm_micro<1>(CPU_MICRO_GEMM_PARAMS); break;
      case 2: gemm_micro<2>(CPU_MICRO_GEMM_PARAMS); break;
      case 3: gemm_micro<3>(CPU_MICRO_GEMM_PARAMS); break;
      case 4: gemm_micro<4>(CPU_MICRO_GEMM_PARAMS); break;
      case 5: gemm_micro<5>(CPU_MICRO_GEMM_PARAMS); break;
      case 6: gemm_micro<6>(CPU_MICRO_GEMM_PARAMS); break;
      case 7: gemm_micro<7>(CPU_MICRO_GEMM_PARAMS); break;
      case 8: gemm_micro<8>(CPU_MICRO_GEMM_PARAMS); break;
    }
  }

  template <int32_t M>
  static void gemm_micro(DEFINE_CPU_MICRO_GEMM_PARAMS) {
    static_assert(0 < M && M <= 8);

    // Calculate how many 4x4 tiles we need in M dimension
    constexpr int tiles_m = (M + 3) / 4; 
    constexpr int tiles_n = 4; // Since NSize=16, we always process 4 tiles in N dimension

    __vector_quad acc[tiles_m][tiles_n];
    for (int i = 0; i < tiles_m; i++) {
        for (int j = 0; j < tiles_n; j++) {
            __builtin_mma_xxsetaccz(&acc[i][j]);
        }
    }

    // Load initial accumulators if accum_c is true
    if (accum_c) {
        for (int i = 0; i < tiles_m; i++) {
            for (int j = 0; j < tiles_n; j++) {
                __vector float tmp[4];
                #pragma GCC unroll 4
                for(int r=0; r<4; r++) {
                    int r_idx = i*4 + r;
                    if (r_idx < M) {
                        tmp[r] = (__vector float)vec_xl(0, &c_ptr[r_idx * ldc + j*4]);
                    } else {
                        tmp[r] = (__vector float){0.0f, 0.0f, 0.0f, 0.0f};
                    }
                }
                __builtin_mma_build_acc(&acc[i][j], 
                    (__vector unsigned char)tmp[0], 
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
            uint32_t val0_0 = 0, val1_0 = 0, val2_0 = 0, val3_0 = 0;
            uint32_t val0_2 = 0, val1_2 = 0, val2_2 = 0, val3_2 = 0;
            uint32_t val0_4 = 0, val1_4 = 0, val2_4 = 0, val3_4 = 0;
            uint32_t val0_6 = 0, val1_6 = 0, val2_6 = 0, val3_6 = 0;
            
            if constexpr (M >= 1) {
                std::memcpy(&val0_0, &a_ptr[0 * lda + k_idx + 0], 4);
                std::memcpy(&val0_2, &a_ptr[0 * lda + k_idx + 2], 4);
                std::memcpy(&val0_4, &a_ptr[0 * lda + k_idx + 4], 4);
                std::memcpy(&val0_6, &a_ptr[0 * lda + k_idx + 6], 4);
            }
            if constexpr (M >= 2) {
                std::memcpy(&val1_0, &a_ptr[1 * lda + k_idx + 0], 4);
                std::memcpy(&val1_2, &a_ptr[1 * lda + k_idx + 2], 4);
                std::memcpy(&val1_4, &a_ptr[1 * lda + k_idx + 4], 4);
                std::memcpy(&val1_6, &a_ptr[1 * lda + k_idx + 6], 4);
            }
            if constexpr (M >= 3) {
                std::memcpy(&val2_0, &a_ptr[2 * lda + k_idx + 0], 4);
                std::memcpy(&val2_2, &a_ptr[2 * lda + k_idx + 2], 4);
                std::memcpy(&val2_4, &a_ptr[2 * lda + k_idx + 4], 4);
                std::memcpy(&val2_6, &a_ptr[2 * lda + k_idx + 6], 4);
            }
            if constexpr (M >= 4) {
                std::memcpy(&val3_0, &a_ptr[3 * lda + k_idx + 0], 4);
                std::memcpy(&val3_2, &a_ptr[3 * lda + k_idx + 2], 4);
                std::memcpy(&val3_4, &a_ptr[3 * lda + k_idx + 4], 4);
                std::memcpy(&val3_6, &a_ptr[3 * lda + k_idx + 6], 4);
            }
            
            vA_0[0] = (__vector unsigned int){val0_0, val1_0, val2_0, val3_0};
            vA_2[0] = (__vector unsigned int){val0_2, val1_2, val2_2, val3_2};
            vA_4[0] = (__vector unsigned int){val0_4, val1_4, val2_4, val3_4};
            vA_6[0] = (__vector unsigned int){val0_6, val1_6, val2_6, val3_6};
        }
        
        if constexpr (M >= 5) {
            uint32_t val4_0 = 0, val5_0 = 0, val6_0 = 0, val7_0 = 0;
            uint32_t val4_2 = 0, val5_2 = 0, val6_2 = 0, val7_2 = 0;
            uint32_t val4_4 = 0, val5_4 = 0, val6_4 = 0, val7_4 = 0;
            uint32_t val4_6 = 0, val5_6 = 0, val6_6 = 0, val7_6 = 0;
            
            if constexpr (M >= 5) {
                std::memcpy(&val4_0, &a_ptr[4 * lda + k_idx + 0], 4);
                std::memcpy(&val4_2, &a_ptr[4 * lda + k_idx + 2], 4);
                std::memcpy(&val4_4, &a_ptr[4 * lda + k_idx + 4], 4);
                std::memcpy(&val4_6, &a_ptr[4 * lda + k_idx + 6], 4);
            }
            if constexpr (M >= 6) {
                std::memcpy(&val5_0, &a_ptr[5 * lda + k_idx + 0], 4);
                std::memcpy(&val5_2, &a_ptr[5 * lda + k_idx + 2], 4);
                std::memcpy(&val5_4, &a_ptr[5 * lda + k_idx + 4], 4);
                std::memcpy(&val5_6, &a_ptr[5 * lda + k_idx + 6], 4);
            }
            if constexpr (M >= 7) {
                std::memcpy(&val6_0, &a_ptr[6 * lda + k_idx + 0], 4);
                std::memcpy(&val6_2, &a_ptr[6 * lda + k_idx + 2], 4);
                std::memcpy(&val6_4, &a_ptr[6 * lda + k_idx + 4], 4);
                std::memcpy(&val6_6, &a_ptr[6 * lda + k_idx + 6], 4);
            }
            if constexpr (M >= 8) {
                std::memcpy(&val7_0, &a_ptr[7 * lda + k_idx + 0], 4);
                std::memcpy(&val7_2, &a_ptr[7 * lda + k_idx + 2], 4);
                std::memcpy(&val7_4, &a_ptr[7 * lda + k_idx + 4], 4);
                std::memcpy(&val7_6, &a_ptr[7 * lda + k_idx + 6], 4);
            }
            
            vA_0[1] = (__vector unsigned int){val4_0, val5_0, val6_0, val7_0};
            vA_2[1] = (__vector unsigned int){val4_2, val5_2, val6_2, val7_2};
            vA_4[1] = (__vector unsigned int){val4_4, val5_4, val6_4, val7_4};
            vA_6[1] = (__vector unsigned int){val4_6, val5_6, val6_6, val7_6};
        }
        
        // Load B and GER for k_idx + 0
        __vector unsigned char vB_vec_0[tiles_n];
        for (int j = 0; j < tiles_n; j++) vB_vec_0[j] = vec_xl(0, (const unsigned char*)&b_ptr[j * k * 4 + (k_idx + 0) * 4]);
        for (int i = 0; i < tiles_m; i++) for (int j = 0; j < tiles_n; j++) __builtin_mma_xvbf16ger2pp(&acc[i][j], (__vector unsigned char)vA_0[i], vB_vec_0[j]);
        
        // Load B and GER for k_idx + 2
        __vector unsigned char vB_vec_2[tiles_n];
        for (int j = 0; j < tiles_n; j++) vB_vec_2[j] = vec_xl(0, (const unsigned char*)&b_ptr[j * k * 4 + (k_idx + 2) * 4]);
        for (int i = 0; i < tiles_m; i++) for (int j = 0; j < tiles_n; j++) __builtin_mma_xvbf16ger2pp(&acc[i][j], (__vector unsigned char)vA_2[i], vB_vec_2[j]);
        
        // Load B and GER for k_idx + 4
        __vector unsigned char vB_vec_4[tiles_n];
        for (int j = 0; j < tiles_n; j++) vB_vec_4[j] = vec_xl(0, (const unsigned char*)&b_ptr[j * k * 4 + (k_idx + 4) * 4]);
        for (int i = 0; i < tiles_m; i++) for (int j = 0; j < tiles_n; j++) __builtin_mma_xvbf16ger2pp(&acc[i][j], (__vector unsigned char)vA_4[i], vB_vec_4[j]);
        
        // Load B and GER for k_idx + 6
        __vector unsigned char vB_vec_6[tiles_n];
        for (int j = 0; j < tiles_n; j++) vB_vec_6[j] = vec_xl(0, (const unsigned char*)&b_ptr[j * k * 4 + (k_idx + 6) * 4]);
        for (int i = 0; i < tiles_m; i++) for (int j = 0; j < tiles_n; j++) __builtin_mma_xvbf16ger2pp(&acc[i][j], (__vector unsigned char)vA_6[i], vB_vec_6[j]);
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
            vB_vec[j] = vec_xl(0, (const unsigned char*)&b_ptr[j * k * 4 + k_idx * 4]);
        }

        for (int i = 0; i < tiles_m; i++) {
            for (int j = 0; j < tiles_n; j++) {
                __builtin_mma_xvbf16ger2pp(&acc[i][j], (__vector unsigned char)vA_uint[i], vB_vec[j]);
            }
        }
    }

    // Disassemble and store
    for (int i = 0; i < tiles_m; i++) {
        for (int j = 0; j < tiles_n; j++) {
            __vector float tmp[4];
            __builtin_mma_disassemble_acc(tmp, &acc[i][j]);
            #pragma GCC unroll 4
            for(int r=0; r<4; r++) {
                int r_idx = i*4 + r;
                if (r_idx < M) {
                    vec_xst((__vector float)tmp[r], 0, &c_ptr[r_idx * ldc + j*4]);
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
  // We need to pack it for MMA. MMA `xvbf16ger2pp` requires 4 output columns and 2 input columns in a single VecBF16.
  // We pack into shape: [output_size / 4, input_size / 2, 8]
  static void pack_weight(const scalar_t* __restrict__ weight,
                          scalar_t* __restrict__ packed_weight,
                          const int32_t output_size, const int32_t input_size) {
    TORCH_CHECK(output_size % 16 == 0);
    TORCH_CHECK(input_size % 2 == 0);
    
    const uint16_t* w = reinterpret_cast<const uint16_t*>(weight);
    uint16_t* pw = reinterpret_cast<uint16_t*>(packed_weight);

    for (int32_t o_idx = 0; o_idx < output_size; o_idx += 4) {
      int32_t i_idx = 0;
      for (; i_idx <= input_size - 8; i_idx += 8) {
          __vector unsigned short row0 = (__vector unsigned short)vec_xl(0, (const unsigned char*)&w[(o_idx+0)*input_size + i_idx]);
          __vector unsigned short row1 = (__vector unsigned short)vec_xl(0, (const unsigned char*)&w[(o_idx+1)*input_size + i_idx]);
          __vector unsigned short row2 = (__vector unsigned short)vec_xl(0, (const unsigned char*)&w[(o_idx+2)*input_size + i_idx]);
          __vector unsigned short row3 = (__vector unsigned short)vec_xl(0, (const unsigned char*)&w[(o_idx+3)*input_size + i_idx]);
          
          __vector unsigned int w0 = (__vector unsigned int)row0;
          __vector unsigned int w1 = (__vector unsigned int)row1;
          __vector unsigned int w2 = (__vector unsigned int)row2;
          __vector unsigned int w3 = (__vector unsigned int)row3;
          
          __vector unsigned int w01_h = vec_mergeh(w0, w1);
          __vector unsigned int w23_h = vec_mergeh(w2, w3);
          __vector unsigned int w01_l = vec_mergel(w0, w1);
          __vector unsigned int w23_l = vec_mergel(w2, w3);
          
          typedef __vector unsigned long long v_ull_t;
          __vector unsigned short out0 = (__vector unsigned short)vec_mergeh((v_ull_t)w01_h, (v_ull_t)w23_h);
          __vector unsigned short out1 = (__vector unsigned short)vec_mergel((v_ull_t)w01_h, (v_ull_t)w23_h);
          __vector unsigned short out2 = (__vector unsigned short)vec_mergeh((v_ull_t)w01_l, (v_ull_t)w23_l);
          __vector unsigned short out3 = (__vector unsigned short)vec_mergel((v_ull_t)w01_l, (v_ull_t)w23_l);
          
          vec_xst(out0, 0, pw); pw += 8;
          vec_xst(out1, 0, pw); pw += 8;
          vec_xst(out2, 0, pw); pw += 8;
          vec_xst(out3, 0, pw); pw += 8;
      }
      for (; i_idx < input_size; i_idx += 2) {
        *pw++ = w[(o_idx+0)*input_size + i_idx];
        *pw++ = w[(o_idx+0)*input_size + i_idx+1];
        
        *pw++ = w[(o_idx+1)*input_size + i_idx];
        *pw++ = w[(o_idx+1)*input_size + i_idx+1];
        
        *pw++ = w[(o_idx+2)*input_size + i_idx];
        *pw++ = w[(o_idx+2)*input_size + i_idx+1];
        
        *pw++ = w[(o_idx+3)*input_size + i_idx];
        *pw++ = w[(o_idx+3)*input_size + i_idx+1];
      }
    }
  }
};
}  // namespace cpu_micro_gemm

#endif
