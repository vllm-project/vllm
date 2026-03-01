#pragma once

#include <ATen/native/CPUBlas.h>

// clang-format off

// amx-bf16
#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

// block size for AMX gemm
constexpr int block_size_m() { return 2 * TILE_M; }
constexpr int block_size_n() { return 2 * TILE_N; }

// define threshold using brgemm (intel AMX)
template <typename T> inline bool can_use_brgemm(int M);
template <> inline bool can_use_brgemm<at::BFloat16>(int M) { return M > 4; }
template <> inline bool can_use_brgemm<at::Half>(int M) { return true; }
// TODO: add u8s8 brgemm, this requires PyTorch 2.7
template <> inline bool can_use_brgemm<int8_t>(int M) { return false; }
template <> inline bool can_use_brgemm<at::Float8_e4m3fn>(int M) { return M > 4; }
template <> inline bool can_use_brgemm<at::quint4x2>(int M) { return M > 4; }

// work around compiler internal error
#define BLOCK_K 128 // 4 * TILE_K

// adjust leading dimension size for K
template <typename T>
inline int64_t get_row_size(int64_t K) {
  return K;
}

template <>
inline int64_t get_row_size<int8_t>(int64_t K) {
  return K + sizeof(int32_t);
}

inline int64_t get_row_size(int64_t K, bool use_int8_w8a8) {
  return use_int8_w8a8 ? K + sizeof(int32_t) : K;
}

// pack weight to vnni format
at::Tensor convert_weight_packed(at::Tensor& weight);

// moe implementations for int8 w8a8
template <typename scalar_t>
void fused_experts_int8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    uint8_t* __restrict__ A_tmp,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad);

// moe implementations for fp8 w8a16
template <typename scalar_t>
void fused_experts_fp8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const at::Float8_e4m3fn* __restrict__ packed_w1,
    const at::Float8_e4m3fn* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad);

// moe implementations for int4 w4a16
template <typename scalar_t>
void fused_experts_int4_w4a16_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ ic2,
    scalar_t* __restrict__ A_tmp,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const at::quint4x2* __restrict__ packed_w1,
    const at::quint4x2* __restrict__ packed_w2,
    const uint8_t* __restrict__ w1z,
    const uint8_t* __restrict__ w2z,
    const scalar_t* __restrict__ w1s,
    const scalar_t* __restrict__ w2s,
    int group_size,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ sorted_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ offsets,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t E,
    int64_t topk,
    int64_t num_tokens_post_pad);

// shared expert implementation for int8 w8a8
template <typename scalar_t>
void shared_expert_int8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic1,
    float* __restrict__ C_tmp,
    uint8_t* __restrict__ Aq_tmp,
    float* __restrict__ As_tmp,
    const scalar_t* __restrict__ input,
    const int8_t* __restrict__ packed_w1,
    const int8_t* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K);

template <typename scalar_t>
void shared_expert_fp8_kernel_impl(
    scalar_t* __restrict__ output,
    scalar_t* __restrict__ ic0,
    scalar_t* __restrict__ ic1,
    scalar_t* __restrict__ B_tmp,
    float* __restrict__ C_tmp,
    const scalar_t* __restrict__ input,
    const at::Float8_e4m3fn* __restrict__ packed_w1,
    const at::Float8_e4m3fn* __restrict__ packed_w2,
    const float* __restrict__ w1s,
    const float* __restrict__ w2s,
    int64_t block_size_N,
    int64_t block_size_K,
    const scalar_t* __restrict__ fused_experts_out,
    float routed_scaling_factor,
    int64_t M,
    int64_t N,
    int64_t K);

// tinygemm interface
template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const scalar_t* __restrict__ B,
    scalar_t* __restrict__ C,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

template <typename scalar_t>
void tinygemm_kernel(
    const uint8_t* __restrict__ A,
    const int8_t* __restrict__ B,
    scalar_t* __restrict__ C,
    int32_t* __restrict__ Ctmp,
    const float* __restrict__ As,
    const float* __restrict__ Bs,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg);

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::Float8_e4m3fn* __restrict__ B,
    scalar_t* __restrict__ C,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    const float* __restrict__ scale,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    bool brg,
    int64_t block_size_K);

template <typename scalar_t>
void tinygemm_kernel(
    const scalar_t* __restrict__ A,
    const at::quint4x2* __restrict__ B,
    scalar_t* __restrict__ C,
    const uint8_t* __restrict__ Bz,
    const scalar_t* __restrict__ Bs,
    scalar_t* __restrict__ Btmp,
    float* __restrict__ Ctmp,
    int64_t M,
    int64_t N,
    int64_t K,
    int group_size,
    int64_t lda,
    int64_t ldb,
    int64_t ldc,
    int64_t strideBz,
    int64_t strideBs,
    bool brg);

// TODO: debug print, remove me later
inline void print_16x32i(const __m512i x) {
  int32_t a[16];
  _mm512_storeu_si512((__m512i *)a, x);

  for (int i = 0; i < 16; i++){
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}

inline void print_16x32(const __m512 x) {
  float a[16];
  _mm512_storeu_ps((__m512 *)a, x);

  for (int i = 0; i < 16; i++){
    std::cout << a[i] << " ";
  }
  std::cout << std::endl;
}


inline void print_32x8u(const __m256i x) {
  uint8_t a[32];
  _mm256_storeu_si256((__m256i *)a, x);

  for (int i = 0; i < 32; ++i) {
    std::cout << int32_t(a[i]) << " ";
  }
  std::cout << std::endl;
}
