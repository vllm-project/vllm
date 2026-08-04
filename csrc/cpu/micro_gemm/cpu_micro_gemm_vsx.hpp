#ifndef CPU_MICRO_GEMM_VSX_HPP
#define CPU_MICRO_GEMM_VSX_HPP
#include "cpu/micro_gemm/cpu_micro_gemm_impl.hpp"
#include <altivec.h>

namespace cpu_micro_gemm {
namespace {

// Float32 to BF16 (RNE) for packing weights
static inline uint16_t f32_to_bf16_rne(float f) {
    uint32_t u; std::memcpy(&u, &f, 4);
    if ((u & 0x7fffffff) > 0x7f800000) return (uint16_t)((u >> 16) | 0x0040);
    uint32_t lsb = (u >> 16) & 1;
    u += 0x7FFF + lsb;
    return (uint16_t)(u >> 16);
}

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
        typedef float v4sf __attribute__((vector_size(16)));
        for (int i = 0; i < tiles_m; i++) {
            for (int j = 0; j < tiles_n; j++) {
                v4sf tmp[4];
                for(int r=0; r<4; r++) {
                    int r_idx = i*4 + r;
                    if (r_idx < M) {
                        tmp[r] = (v4sf)vec_xl(0, &c_ptr[r_idx * ldc + j*4]);
                    } else {
                        tmp[r] = (v4sf){0.0f, 0.0f, 0.0f, 0.0f};
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

    for (int32_t k_idx = 0; k_idx < k; k_idx += 2) {
        // Load A directly from contiguous memory since PackA = false.
        __vector unsigned int vA_uint[tiles_m];
        if constexpr (M == 1) {
            uint32_t val0;
            std::memcpy(&val0, &a_ptr[0 + k_idx], 4);
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

        // Load packed B. B is packed such that we can load exactly 4 cols x 2 elements into one VecBF16.
        // We packed B as [N/4, K/2, 8] elements = 128 bit vectors.
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
    typedef float v4sf __attribute__((vector_size(16)));
    for (int i = 0; i < tiles_m; i++) {
        for (int j = 0; j < tiles_n; j++) {
            v4sf tmp[4];
            __builtin_mma_disassemble_acc(tmp, &acc[i][j]);
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
      for (int32_t i_idx = 0; i_idx < input_size; i_idx += 2) {
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
