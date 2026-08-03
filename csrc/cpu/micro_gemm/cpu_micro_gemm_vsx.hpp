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

    union VecBF16 {
        uint16_t u16[8];
        __vector unsigned char vec;
    };

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
                        for(int c=0; c<4; c++) {
                            ((float*)&tmp[r])[c] = c_ptr[r_idx * ldc + (j*4 + c)];
                        }
                    } else {
                        for(int c=0; c<4; c++) ((float*)&tmp[r])[c] = 0.0f;
                    }
                }
                __builtin_mma_build_acc(&acc[i][j], (__vector unsigned char*)&tmp);
            }
        }
    }

    const uint16_t* __restrict__ curr_a = reinterpret_cast<const uint16_t*>(a_ptr);
    const uint16_t* __restrict__ curr_b = reinterpret_cast<const uint16_t*>(b_ptr);

    // K must be even for xvbf16ger2pp
    for (int32_t k_idx = 0; k_idx < k; k_idx += 2) {
        // Load A
        VecBF16 vA[tiles_m];
        for (int i = 0; i < tiles_m; i++) {
            for(int r=0; r<4; r++) {
                int r_idx = i*4 + r;
                if (r_idx < M) {
                    vA[i].u16[r*2 + 0] = curr_a[r_idx * lda + k_idx];
                    vA[i].u16[r*2 + 1] = curr_a[r_idx * lda + k_idx + 1];
                } else {
                    vA[i].u16[r*2 + 0] = 0;
                    vA[i].u16[r*2 + 1] = 0;
                }
            }
        }

        // Load packed B. B is packed such that we can load exactly 4 cols x 2 elements into one VecBF16.
        // We packed B as [N/4, K/2, 8] elements = 128 bit vectors.
        VecBF16 vB[tiles_n];
        for (int j = 0; j < tiles_n; j++) {
            for(int e=0; e<8; e++) {
                vB[j].u16[e] = curr_b[j * k * 4 + k_idx * 4 + e];
            }
        }

        for (int i = 0; i < tiles_m; i++) {
            for (int j = 0; j < tiles_n; j++) {
                __builtin_mma_xvbf16ger2pp(&acc[i][j], vA[i].vec, vB[j].vec);
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
                    for(int c=0; c<4; c++) {
                        c_ptr[r_idx * ldc + (j*4 + c)] = ((float*)&tmp[r])[c];
                    }
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
        // We pack 4 cols from i_idx, and 4 cols from i_idx+1
        // VSR layout for vB: u16[0,1]=col0, u16[2,3]=col1, u16[4,5]=col2, u16[6,7]=col3
        // So for e in [0, 1, 2, 3]:
        //   u16[2*e] = w[(o_idx+e)*input_size + i_idx]
        //   u16[2*e+1] = w[(o_idx+e)*input_size + i_idx+1]
        
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
