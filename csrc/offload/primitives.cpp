#include "primitives.h"
#include <cstdlib>
#include <sys/syscall.h>
#include <cstring>
#include <cmath>
#include <iostream>

#include <unistd.h>
#include <stdbool.h>
#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iomanip>
// ==================== 数据类型实现 ====================

bfloat16_t bfloat16_t::from_float(float f) {
    uint32_t fp32 = *reinterpret_cast<uint32_t*>(&f);
    uint32_t round_bias = (fp32 >> 16) & 0x1;
    return bfloat16_t(static_cast<uint16_t>((fp32 + round_bias) >> 16));
}

float bfloat16_t::to_float() const {
    uint32_t fp32 = static_cast<uint32_t>(bits) << 16;
    return *reinterpret_cast<float*>(&fp32);
}

float8_e4m3_t float8_e4m3_t::from_float(float f) {
    if (f == 0.0f) return float8_e4m3_t(0);
    if (std::isnan(f) || std::isinf(f)) return float8_e4m3_t(0x7F);

    const bool sign = f < 0;
    const float abs_f = sign ? -f : f;

    if (abs_f > 448.0f) return float8_e4m3_t((sign << 7) | 0x7F);
    if (abs_f < 0.0078125f) return float8_e4m3_t(sign << 7);

    int exponent;
    float normalized = std::frexp(abs_f, &exponent);
    exponent--;

    uint8_t e4m3_exp, e4m3_mant;
    if (exponent < -6) {
        e4m3_exp = 0;
        const float subnormal_val = abs_f * 512.0f;
        e4m3_mant = static_cast<uint8_t>(subnormal_val + 0.5f);
        if (e4m3_mant > 7) e4m3_mant = 7;
    } else {
        e4m3_exp = static_cast<uint8_t>(exponent + 7);
        const float mantissa_val = (normalized - 0.5f) * 16.0f;
        e4m3_mant = static_cast<uint8_t>(mantissa_val + 0.5f);

        const float fraction = mantissa_val - e4m3_mant;
        const bool round_up = (fraction > 0.5f) ||
                             (fraction == 0.5f && (e4m3_mant & 1));
        if (round_up) {
            e4m3_mant++;
            if (e4m3_mant == 8) {
                e4m3_mant = 0;
                e4m3_exp++;
            }
        }
    }
    return float8_e4m3_t((sign << 7) | (e4m3_exp << 3) | e4m3_mant);
}

float float8_e4m3_t::to_float() const {
    const uint8_t sign = bits >> 7;
    const uint8_t exponent_bits = (bits >> 3) & 0x0F;
    const uint8_t mantissa_bits = bits & 0x07;

    if (exponent_bits == 0 && mantissa_bits == 0) {
        union { uint32_t u; float f; } u = { sign ? 0x80000000u : 0x00000000u };
        return u.f;
    }
    if (exponent_bits == 0x0F) {
        union { uint32_t u; float f; } u = { 0x7FC00000u };
        return u.f;
    }

    const float sign_f = sign ? -1.0f : 1.0f;
    if (exponent_bits == 0) {
        const float mantissa_f = static_cast<float>(mantissa_bits) / 8.0f;
        return sign_f * mantissa_f * 0.015625f;
    }

    const float mantissa_f = 1.0f + static_cast<float>(mantissa_bits) / 8.0f;
    return sign_f * mantissa_f * std::exp2f(static_cast<float>(exponent_bits) - 7.0f);
}

void fp32_to_bf16(const float* __restrict f32_in,
                      bfloat16_t*  __restrict bf16_out, int len)
{
    const int step = 16;                 // 每次处理 16 个 float
    for (int i = 0; i < len; i += step)
    {
        __m512 v = _mm512_loadu_ps(f32_in + i);          // 16×float
        _mm256_storeu_si256((__m256i*)(bf16_out + i), (__m256i)_mm512_cvtneps_pbh(v));
    }
}


// ==================== AMX配置实现 ====================

bool set_tiledata_use()
{
    if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA))
    {
       printf("\n Failed to enable XFEATURE_XTILEDATA \n\n");
       return false;
    }
    else
    {
       //printf("\n TILE DATA USE SET - OK \n\n");
       return true;
    }
    return true;
}




// 初始化tile配置
void init_tile_config(__tile_config *cfg, int rows, int colsb) {
    memset(cfg, 0, sizeof(__tile_config));
    cfg->palette_id = 1;
    cfg->start_row = 0;
    for (int i = 0; i < 8; i++) {
        cfg->colsb[i] = colsb; // 每行字节数：BF16为64(32个元素*2字节)
        cfg->rows[i] = rows;   // 行数：最大16
    }
}



// ==================== AMX核心计算实现 ====================

inline __m512i fp8x32_to_bf16(__m256i in8) {
    __m512i fp8_ext = _mm512_cvtepu8_epi16(in8);
    fp8_ext = _mm512_add_epi16(fp8_ext, mm780);
    fp8_ext = _mm512_slli_epi16(fp8_ext, 4);
    fp8_ext = _mm512_and_si512(fp8_ext, mm87f0);
    fp8_ext = _mm512_add_epi16(fp8_ext, mm3c00);
    return fp8_ext;
}


void amx_gemm_block_32_K_32(
    const bfloat16_t *A, const float8_e4m3_t *B, float *scale,
    float *C, int K, int ldc
) {
    // 根据layout计算4个子块地址
    __tile_config cfg;
    init_tile_config(&cfg, 16, 64);
    _tile_loadconfig(&cfg);

    float *C00, *C01, *C10, *C11;

    C00 = C;
    C01 = C + 16;
    C10 = C + ldc * 16;
    C11 = C + ldc * 16 + 16;

    alignas(64) bfloat16_t B_bf16[32 * 32];

    for (int kk = 0; kk < K; kk += 128) {
        __m512 scale_vec = _mm512_set1_ps(scale[kk / 128]);
        alignas(64) float scale_buf[16 * 64];
        _tile_zero(0); _tile_zero(1); _tile_zero(2); _tile_zero(3);

        for (int k = kk; k < kk + 128 && k < K; k += 32) {
            const float8_e4m3_t *B_fp8 = B + k * tile_block_size * 2 / 32;
            const bfloat16_t *A_ptr = A + k / 32 * tile_block_size * 2;

            #pragma GCC unroll 32
            for (int i = 0; i < 32; i++) {
                __m256i fp8_row = _mm256_loadu_si256((const __m256i *)(B_fp8 + i * 32));
                __m512i bf16_row = fp8x32_to_bf16(fp8_row);
                _mm512_store_epi32(B_bf16 + i * 32, bf16_row);
            }
            asm volatile("" ::: "memory");
            _tile_loadd(4, A_ptr, 64);
            _tile_loadd(5, A_ptr + tile_block_size, 64);
            _tile_loadd(6, B_bf16, 128);
            _tile_loadd(7, B_bf16 + 32, 128);

            _tile_dpbf16ps(0, 4, 6);
            _tile_dpbf16ps(1, 4, 7);
            _tile_dpbf16ps(2, 5, 6);
            _tile_dpbf16ps(3, 5, 7);
        }
        _tile_stored(0, scale_buf, 16 * sizeof(float));
        _tile_stored(1, scale_buf + 16 * 16, 16 * sizeof(float));
        _tile_stored(2, scale_buf + 32 * 16, 16 * sizeof(float));
        _tile_stored(3, scale_buf + 48 * 16, 16 * sizeof(float));

        #pragma GCC unroll 16
        for (int i = 0; i < 16; i++) {
            __m512 vec00 = _mm512_loadu_ps(scale_buf + i * 16);
            __m512 vec01 = _mm512_loadu_ps(scale_buf + (i + 16) * 16);
            __m512 vec10 = _mm512_loadu_ps(scale_buf + (i + 32) * 16);
            __m512 vec11 = _mm512_loadu_ps(scale_buf + (i + 48) * 16);

            __m512 c00_vec = _mm512_loadu_ps(C00 + i * ldc);
            __m512 c01_vec = _mm512_loadu_ps(C01 + i * ldc);
            __m512 c10_vec = _mm512_loadu_ps(C10 + i * ldc);
            __m512 c11_vec = _mm512_loadu_ps(C11 + i * ldc);

            c00_vec = _mm512_fmadd_ps(vec00, scale_vec, c00_vec);
            c01_vec = _mm512_fmadd_ps(vec01, scale_vec, c01_vec);
            c10_vec = _mm512_fmadd_ps(vec10, scale_vec, c10_vec);
            c11_vec = _mm512_fmadd_ps(vec11, scale_vec, c11_vec);

            _mm512_storeu_ps(C00 + i * ldc, c00_vec);
            _mm512_storeu_ps(C01 + i * ldc, c01_vec);
            _mm512_storeu_ps(C10 + i * ldc, c10_vec);
            _mm512_storeu_ps(C11 + i * ldc, c11_vec);
        }
    }
}

const static int BLOCK_K = 32;
const static int BLOCK_M = 32;
const static int BLOCK_SIZE = BLOCK_K * BLOCK_M;
void gemv_anni_grouped(const bfloat16_t* B, const uint8_t* A, const float* AS,
                       float* C, int M, int K, int block_size) {
    const int m_blocks = M / BLOCK_M;      // M // 32
    const int k_blocks = K / BLOCK_K;      // K // 16
    const int AS_col_stride = K / 128; // K // 128

    for(int m = 0; m < m_blocks; m+=4) {
        __m512 Cv[8] = {};
        const float* AS_row = AS + m * AS_col_stride;

        for(int kg = 0; kg < k_blocks; kg += 4) {
            __m512 sum[8] = {};
            const uint8_t* A_base = A + (kg + m * k_blocks) * BLOCK_SIZE;

            #pragma GCC unroll 4
            for(int kk=0; kk<4; kk++)
            {
                __m512i b_block = _mm512_loadu_si512((const __m512i*)(B + (kg + kk) * BLOCK_K));

                #pragma GCC unroll 16
                for(int ch = 0; ch < 16; ++ch){
                    __m512i b_vec = _mm512_permutexvar_epi32(_mm512_set1_epi32(ch), b_block);
                    _mm_prefetch((const char*)(A_base + (ch + kk * 16) * 64 + BLOCK_SIZE), _MM_HINT_T0);
                    _mm_prefetch((const char*)(A_base + (ch + kk * 16) * 64 + (1 * k_blocks) * BLOCK_SIZE + BLOCK_SIZE/2), _MM_HINT_T0);
                    _mm_prefetch((const char*)(A_base + (ch + kk * 16) * 64 + (2 * k_blocks) * BLOCK_SIZE + BLOCK_SIZE/2), _MM_HINT_T0);
                    _mm_prefetch((const char*)(A_base + (ch + kk * 16) * 64 + (3 * k_blocks) * BLOCK_SIZE + BLOCK_SIZE/2), _MM_HINT_T0);

                    __m512i block12 = _mm512_loadu_si512((const __m512i*)(A_base + (ch + kk * 16) * 64));
                    __m512i block34 = _mm512_loadu_si512((const __m512i*)(A_base + (ch + kk * 16) * 64 + k_blocks * BLOCK_SIZE));
                    __m512i block56 = _mm512_loadu_si512((const __m512i*)(A_base + (ch + kk * 16) * 64 + 2 * k_blocks * BLOCK_SIZE));
                    __m512i block78 = _mm512_loadu_si512((const __m512i*)(A_base + (ch + kk * 16) * 64 + 3 * k_blocks * BLOCK_SIZE));

                    __m256i v1 = _mm512_extracti64x4_epi64(block12, 0);
                    __m256i v2 = _mm512_extracti64x4_epi64(block12, 1);
                    __m256i v3 = _mm512_extracti64x4_epi64(block34, 0);
                    __m256i v4 = _mm512_extracti64x4_epi64(block34, 1);
                    __m256i v5 = _mm512_extracti64x4_epi64(block56, 0);
                    __m256i v6 = _mm512_extracti64x4_epi64(block56, 1);
                    __m256i v7 = _mm512_extracti64x4_epi64(block78, 0);
                    __m256i v8 = _mm512_extracti64x4_epi64(block78, 1);

                    __m512bh e1 = (__m512bh)fp8x32_to_bf16(v1);
                    __m512bh e2 = (__m512bh)fp8x32_to_bf16(v2);
                    __m512bh e3 = (__m512bh)fp8x32_to_bf16(v3);
                    __m512bh e4 = (__m512bh)fp8x32_to_bf16(v4);
                    __m512bh e5 = (__m512bh)fp8x32_to_bf16(v5);
                    __m512bh e6 = (__m512bh)fp8x32_to_bf16(v6);
                    __m512bh e7 = (__m512bh)fp8x32_to_bf16(v7);
                    __m512bh e8 = (__m512bh)fp8x32_to_bf16(v8);

                    sum[0] = _mm512_dpbf16_ps(sum[0], e1, (__m512bh)b_vec);
                    sum[1] = _mm512_dpbf16_ps(sum[1], e2, (__m512bh)b_vec);
                    sum[2] = _mm512_dpbf16_ps(sum[2], e3, (__m512bh)b_vec);
                    sum[3] = _mm512_dpbf16_ps(sum[3], e4, (__m512bh)b_vec);
                    sum[4] = _mm512_dpbf16_ps(sum[4], e5, (__m512bh)b_vec);
                    sum[5] = _mm512_dpbf16_ps(sum[5], e6, (__m512bh)b_vec);
                    sum[6] = _mm512_dpbf16_ps(sum[6], e7, (__m512bh)b_vec);
                    sum[7] = _mm512_dpbf16_ps(sum[7], e8, (__m512bh)b_vec);
                }
            }

            float wscale = *AS_row;
            __m512 scale_vec = _mm512_set1_ps(wscale);
            Cv[0] = _mm512_fmadd_ps(sum[0], scale_vec, Cv[0]);
            Cv[1] = _mm512_fmadd_ps(sum[1], scale_vec, Cv[1]);
            Cv[2] = _mm512_fmadd_ps(sum[2], scale_vec, Cv[2]);
            Cv[3] = _mm512_fmadd_ps(sum[3], scale_vec, Cv[3]);
            Cv[4] = _mm512_fmadd_ps(sum[4], scale_vec, Cv[4]);
            Cv[5] = _mm512_fmadd_ps(sum[5], scale_vec, Cv[5]);
            Cv[6] = _mm512_fmadd_ps(sum[6], scale_vec, Cv[6]);
            Cv[7] = _mm512_fmadd_ps(sum[7], scale_vec, Cv[7]);
            AS_row += 1;
        }

        for(int i=0; i<8; ++i)
        {
            _mm512_stream_ps(C + m * BLOCK_M + BLOCK_M/2 * i, Cv[i]);
        }
    }
    _mm_sfence();

}