#pragma once

#ifndef __NVCC__
#include <immintrin.h>
#endif
#include <cstdint>
#include <cstddef>
#include <iostream>
#include <type_traits>

// ==================== 数据类型定义 ====================

struct bfloat16_t {
    uint16_t bits;

    bfloat16_t() : bits(0) {}
    explicit bfloat16_t(uint16_t b) : bits(b) {}

    static bfloat16_t from_float(float f);
    float to_float() const;
};

struct float8_e4m3_t {
    uint8_t bits;

    float8_e4m3_t() : bits(0) {}
    explicit float8_e4m3_t(uint8_t b) : bits(b) {}

    static float8_e4m3_t from_float(float f);
    float to_float() const;
};

// ==================== AMX常量与配置 ====================

#if !defined(__CUDACC__) && !defined(__CUDA_ARCH__)
static const __m512i mm780  = _mm512_set1_epi16(0x780);
static const __m512i mm87f0 = _mm512_set1_epi16(0x87f0);
static const __m512i mm3c00 = _mm512_set1_epi16(0x3c00);
#endif

#define ARCH_GET_XCOMP_PERM     0x1022
#define ARCH_REQ_XCOMP_PERM     0x1023
#define XFEATURE_XTILECFG       17
#define XFEATURE_XTILEDATA      18

bool set_tiledata_use();

typedef struct {
    uint8_t palette_id;
    uint8_t start_row;
    uint8_t reserved[14];
    uint16_t colsb[16];
    uint8_t rows[16];
} __tile_config __attribute__((aligned(64)));

void init_tile_config(__tile_config *cfg, int rows, int colsb);

// ==================== 调试辅助 ====================

template<typename T, typename = void>
struct has_to_float : std::false_type {};

template<typename T>
struct has_to_float<T, std::void_t<decltype(std::declval<T>().to_float())>> : std::true_type {};

template<typename T>
void dump_martix(const T *A, int M, int N, const char *prefix, int ldc) {
    std::cout << prefix << ":" << std::endl;
    int stride = ldc == 0 ? N : ldc;
    for (int i = 0; i < M; i++) {
        std::cout << i << ": ";
        for (int j = 0; j < N; j++) {
            if constexpr (has_to_float<T>::value)
                std::cout << A[i * stride + j].to_float() << ", ";
            else
                std::cout << A[i * stride + j] << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << "=====================================" << std::endl;
}

// ==================== AMX核心计算 ====================

enum class CLayout { RowMajor, Contiguous };

static const int tile_block_size = 32 * 16;
static const int tile_c_block_size = 16 * 16;



void amx_gemm_block_32_K_32(
    const bfloat16_t *A,
    const float8_e4m3_t *B,
    float *scale,
    float *C,           // C矩阵起始地址
    int K,
    int ldc
);
void gemv_anni_grouped(const bfloat16_t* B,const uint8_t* A, const float* AS,
                       float* C, int M, int K, int block_size);
// FP8->BF16批量转换 (32个元素)
#if !defined(__CUDACC__) && !defined(__CUDA_ARCH__) && !defined(__NVCC__)
inline __m512i fp8x32_to_bf16(__m256i in8);
#endif
void fp32_to_bf16(const float* __restrict f32_in,
                      bfloat16_t*  __restrict bf16_out, int len);