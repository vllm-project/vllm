#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>
#include <c10/cuda/CUDAGuard.h>

#include <cmath>
#include <type_traits>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// // #if defined(__CUDACC__) || defined(_NVHPC_CUDA)
// #define HOST_DEVICE_INLINE inline __host__ __device__
// #define DEVICE_INLINE inline __device__
// #define HOST_INLINE inline __host__
// // #else
// // #define HOST_DEVICE_INLINE inline
// // #define DEVICE_INLINE inline
// // #define HOST_INLINE inline
// // #endif // CUTE_HOST_DEVICE, CUTE_DEVICE

// // Use a define to get around clang-formats weird formating for pragmas
// // #define OMP_FOR _Pragma("omp parallel for")
// // #define OMP_FOR_COLLAPSE_4 _Pragma("omp parallel for collapse(4)")

// #include <iostream>

// template <typename T>
// inline std::string str(T x) {
//   return std::to_string(x);
// }

// template <typename T1, typename T2>
// HOST_DEVICE_INLINE constexpr T1 div_ceil(T1 a, T2 b) {
//   return (a + b - 1) / b;
// }

// template <typename T1, typename T2>
// HOST_DEVICE_INLINE constexpr T1 round_up(T1 a, T2 b) {
//   return div_ceil(a, b) * b;
// }

// template <typename T1, typename T2>
// HOST_DEVICE_INLINE constexpr T1 round_down(T1 a, T2 b) {
//   return (a / b) * b;
// }

// template <typename T>
// inline std::enable_if_t<std::is_integral_v<T>, bool> not_zero(T value) {
//   return value != 0;
// }

// template <typename T>
// inline std::enable_if_t<std::is_floating_point_v<T> || std::is_same_v<T, c10::Half> ||
//                             std::is_same_v<T, c10::BFloat16>,
//                         bool>
// not_zero(T value) {
//   using std::fpclassify;
//   return fpclassify(value) != FP_ZERO;
// }

// template <typename T>
// bool is_zero(T value) {
//   return !not_zero(value);
// }

// // 8 warps are a good choice since every SM has 4 schedulers and having more than 1 warp per
// // schedule allows some more latency hiding. At the same time, we want relatively few warps to have
// // many registers per warp and small tiles.
// static int constexpr default_threads = 256;

// static int constexpr pipe_stages = 4; // 4 pipeline stages fit into shared memory
// static int constexpr max_shared_mem =
//     96 * 1024; // max shared memory on compute capability 8.6 (< 8.0)

// static int constexpr min_thread_n = 64;
// static int constexpr min_thread_k = 64;

// static int constexpr tile_size = 16;
// static int constexpr max_par   = 16;

// static int constexpr pack_factor_4bit = 8; // We have 8 4-bit vals inside a 32 bit

// template <typename T, int n>
// struct Vec {
//   T             elems[n];
//   __device__ T& operator[](int i) { return elems[i]; }
// };

// using I4 = Vec<int, 4>;

// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
// // No support for async
// #else
// __device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
//   const int BYTES = 16;
//   uint32_t  smem  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
//   asm volatile("{\n"
//                "   .reg .pred p;\n"
//                "   setp.ne.b32 p, %0, 0;\n"
//                "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
//                "}\n" ::"r"((int)pred),
//                "r"(smem), "l"(glob_ptr), "n"(BYTES));
// }

// __device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
//   const int BYTES = 16;
//   uint32_t  smem  = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
//   asm volatile("{\n"
//                "   .reg .b64 p;\n"
//                "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
//                "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
//                "}\n" ::"r"(smem),
//                "l"(glob_ptr), "n"(BYTES));
// }

// __device__ inline void cp_async_fence() { asm volatile("cp.async.commit_group;\n" ::); }

// template <int n>
// __device__ inline void cp_async_wait() {
//   asm volatile("cp.async.wait_group %0;\n" ::"n"(n));
// }

// #endif

// namespace marlin {

// #if !defined(__CUDA_ARCH__) || !(__CUDA_ARCH__ < 800)
// // // currently only supports f16i4,
// void marlin_mm_moe_f16i4(const void* A, const void* B, void* C, void* sorted_ids, void* s, void* g_idx, void* perm,
//                      void* a_tmp, void* red_tmp, int prob_m, int prob_n, int prob_k, void* workspace,
//                      bool has_act_order, bool is_k_full, int num_groups, int group_size,
//                      int num_tokens_post_padded, int num_experts, int moe_block_size,
//                      int dev = 0, cudaStream_t stream = 0, int thread_k = -1, int thread_n = -1,
//                      int sms = -1, int max_par = 16);

// #endif

// torch::Tensor marlin_gemm_moe(torch::Tensor& a, torch::Tensor& b_q_weights, torch::Tensor& sorted_ids,
//                         torch::Tensor& b_scales, torch::Tensor& workspace,
//                         int64_t size_m, int64_t size_n, int64_t size_k/*,
//                         int64_t num_tokens_post_padded, int64_t num_experts, int64_t moe_block_size*/);

} // namespace marlin
