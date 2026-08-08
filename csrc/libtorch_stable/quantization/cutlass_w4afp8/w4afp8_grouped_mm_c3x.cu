/*
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from SGLang's W4A8 grouped-MoE implementation introduced in
 * https://github.com/sgl-project/sglang/pull/7772. Modified by the vLLM
 * project for stable libtorch, GLM W4AFP8 dispatch, and vLLM bindings.
 */

#include <cudaTypedefs.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include "libtorch_stable/torch_utils.h"

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "cutlass/cutlass.h"
#include "w4afp8_grouped_mm_c3x.cuh"

using namespace cute;

namespace {

enum class Sched { PP, CO };

template <int M, int N, int K, int A, int B, int C, Sched S>
struct SM90W4AFP8Config {
  using KernelSchedule = std::conditional_t<
      S == Sched::PP, cutlass::gemm::KernelPtrArrayTmaWarpSpecializedPingpong,
      cutlass::gemm::KernelPtrArrayTmaWarpSpecializedCooperative>;

  using EpilogueSchedule = std::conditional_t<
      S == Sched::PP, cutlass::epilogue::PtrArrayTmaWarpSpecializedPingpong,
      cutlass::epilogue::PtrArrayTmaWarpSpecializedCooperative>;

  using TileShape = cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>;
  using ClusterShape = cute::Shape<cute::Int<A>, cute::Int<B>, cute::Int<C>>;
  using Cutlass3xW4AFP8Gemm =
      cutlass_3x_w4afp8_group_gemm<TileShape, ClusterShape, KernelSchedule,
                                   EpilogueSchedule>;
};

template <int M, int N, int K, int A, int B, int C>
using SM90_PP = SM90W4AFP8Config<M, N, K, A, B, C, Sched::PP>;

template <int M, int N, int K, int A, int B, int C>
using SM90_CO = SM90W4AFP8Config<M, N, K, A, B, C, Sched::CO>;

template <typename Config>
inline void invoke_gemm(torch::stable::Tensor& d_tensors,
                        torch::stable::Tensor const& a_tensors,
                        torch::stable::Tensor const& b_tensors,
                        torch::stable::Tensor const& a_scales,
                        torch::stable::Tensor const& b_scales,
                        torch::stable::Tensor const& expert_offsets,
                        torch::stable::Tensor const& problem_sizes,
                        torch::stable::Tensor const& a_strides,
                        torch::stable::Tensor const& b_strides,
                        torch::stable::Tensor const& d_strides,
                        torch::stable::Tensor const& s_strides,
                        int64_t chunk_size) {
  using GemmT = typename Config::Cutlass3xW4AFP8Gemm;
  cutlass_w4afp8_group_gemm_caller<GemmT>(
      d_tensors, a_tensors, b_tensors, a_scales, b_scales, expert_offsets,
      problem_sizes, a_strides, b_strides, d_strides, s_strides, chunk_size);
}

// Helper macro to reduce code duplication
// Note: Config must be wrapped in parentheses when it contains commas (e.g.,
// template parameters) This uses a helper macro to strip the parentheses from
// the template parameter
#define INVOKE_GEMM_WITH_CONFIG_HELPER(...)                                    \
  invoke_gemm<__VA_ARGS__>(d_tensors, a_tensors, b_tensors, a_scales,          \
                           b_scales, expert_offsets, problem_sizes, a_strides, \
                           b_strides, d_strides, s_strides, chunk_size)
#define INVOKE_GEMM_WITH_CONFIG(Config) INVOKE_GEMM_WITH_CONFIG_HELPER Config

#define W4AFP8_CONFIG_EQ(value, literal) \
  ((value) != nullptr && std::strcmp((value), (literal)) == 0)

#define TRY_INVOKE_GEMM1_CONFIG(value)                           \
  do {                                                           \
    if (W4AFP8_CONFIG_EQ((value), "pp_64x32x512_c1")) {          \
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 512, 1, 1, 1>));  \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "pp_64x32x512_c2")) {   \
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 512, 2, 1, 1>));  \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "co_128x16x512_c1")) {  \
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>)); \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "co_128x16x512_c2")) {  \
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 2, 1, 1>)); \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "co_128x32x512_c1")) {  \
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 1, 1, 1>)); \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "co_128x32x512_c2")) {  \
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 2, 1, 1>)); \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "co_128x64x512_c1")) {  \
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>)); \
      return;                                                    \
    }                                                            \
  } while (false)

#define TRY_INVOKE_GEMM2_CONFIG(value)                           \
  do {                                                           \
    if (W4AFP8_CONFIG_EQ((value), "pp_64x16x128_c1")) {          \
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 16, 128, 1, 1, 1>));  \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "pp_64x32x128_c1")) {   \
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 128, 1, 1, 1>));  \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "pp_128x16x128_c1")) {  \
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 16, 128, 1, 1, 1>)); \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "pp_128x32x128_c1")) {  \
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 1, 1, 1>)); \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "pp_128x32x128_c2")) {  \
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 2, 1, 1>)); \
      return;                                                    \
    } else if (W4AFP8_CONFIG_EQ((value), "pp_128x64x128_c1")) {  \
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 64, 128, 1, 1, 1>)); \
      return;                                                    \
    }                                                            \
  } while (false)

void dispatch_w4afp8_moe_mm_sm90(torch::stable::Tensor& d_tensors,
                                 torch::stable::Tensor const& a_tensors,
                                 torch::stable::Tensor const& b_tensors,
                                 torch::stable::Tensor const& a_scales,
                                 torch::stable::Tensor const& b_scales,
                                 torch::stable::Tensor const& expert_offsets,
                                 torch::stable::Tensor const& problem_sizes,
                                 torch::stable::Tensor const& a_strides,
                                 torch::stable::Tensor const& b_strides,
                                 torch::stable::Tensor const& d_strides,
                                 torch::stable::Tensor const& s_strides,
                                 int64_t chunk_size, int64_t topk) {
  STD_TORCH_CHECK(topk > 0, "topk must be greater than zero");
  STD_TORCH_CHECK(a_tensors.size(0) % topk == 0,
                  "A tensor rows must be divisible by topk");
  uint32_t const m = a_tensors.size(0) / topk;
  uint32_t const n = d_tensors.size(1);
  uint32_t const k = a_tensors.size(1);
  char const* const force_gemm1_config =
      std::getenv("VLLM_W4AFP8_FORCE_GEMM1_CONFIG");
  char const* const force_gemm2_config =
      std::getenv("VLLM_W4AFP8_FORCE_GEMM2_CONFIG");

  if (n == 512 && k == 6144) {
    // GLM/SGLang W4AFP8 TP8 GEMM1: [M*topk, 6144] x [E, 512, 6144].
    // Optional tuning override values:
    //   pp_64x32x512_c1, pp_64x32x512_c2,
    //   co_128x16x512_c1, co_128x16x512_c2,
    //   co_128x32x512_c1, co_128x32x512_c2, co_128x64x512_c1.
    TRY_INVOKE_GEMM1_CONFIG(force_gemm1_config);
    if (m <= 1) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 1, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 6144 && k == 256) {
    // GLM/SGLang W4AFP8 TP8 GEMM2: [M*topk, 256] x [E, 6144, 256].
    // Optional tuning override values:
    //   pp_64x16x128_c1, pp_64x32x128_c1,
    //   pp_128x16x128_c1, pp_128x32x128_c1,
    //   pp_128x32x128_c2, pp_128x64x128_c1.
    TRY_INVOKE_GEMM2_CONFIG(force_gemm2_config);
    if (m <= 1) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 16, 128, 1, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 1, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 64, 128, 1, 1, 1>));
    }
  } else if (n == 4096 && k == 7168) {
    // group gemm 1
    if (m <= 4) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 2, 1, 1>));
    } else if (m <= 256) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 2, 1, 1>));
    } else if (m <= 4096) {
      // Optimized for prefill: seq_len up to 4096 (m=4096 with topk=1)
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 2, 1, 1>));
    } else {
      // Optimized for prefill: seq_len up to 8192 (m=8192 with topk=1)
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 7168 && k == 2048) {
    // group gemm 2
    if (m <= 8) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 16, 512, 1, 1, 1>));
    } else if (m <= 512) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 1, 1, 1>));
    } else if (m <= 4096) {
      // Optimized for prefill: larger cluster for better throughput
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 512 && k == 7168) {
    // group gemm 1 for tp
    if (m <= 4) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 32, 512, 2, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 2, 1, 1>));
    } else if (m <= 256) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
    } else if (m <= 1024) {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
    }
  } else if (n == 7168 && k == 256) {
    // group gemm 2 for tp
    if (m <= 8) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<64, 16, 128, 1, 1, 1>));
    } else if (m <= 32) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 1, 1, 1>));
    } else if (m <= 512) {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 2, 1, 1>));
    } else {
      INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 64, 128, 1, 1, 1>));
    }
  } else {
    if (k % 512 == 0) {
      // For large m (prefill), prefer larger cluster
      if (m <= 32) {
        // Decode: target batch size (16-32) - use cluster size 1 for better
        // latency
        INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 16, 512, 1, 1, 1>));
      } else if (m <= 1024) {
        // Decode: large batch or small prefill
        INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 32, 512, 1, 1, 1>));
      } else {
        // Prefill: large sequence length - prefer larger cluster
        INVOKE_GEMM_WITH_CONFIG((SM90_CO<128, 64, 512, 1, 1, 1>));
      }
    } else {
      if (m <= 32) {
        // Decode: target batch size (16-32) - use larger tile for better
        // throughput
        INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 32, 128, 1, 1, 1>));
      } else {
        // Prefill: larger sequence length
        INVOKE_GEMM_WITH_CONFIG((SM90_PP<128, 64, 128, 1, 1, 1>));
      }
    }
  }
}

}  // namespace

void cutlass_w4afp8_moe_mm_sm90(torch::stable::Tensor& d_tensors,
                                torch::stable::Tensor const& a_tensors,
                                torch::stable::Tensor const& b_tensors,
                                torch::stable::Tensor const& a_scales,
                                torch::stable::Tensor const& b_scales,
                                torch::stable::Tensor const& expert_offsets,
                                torch::stable::Tensor const& problem_sizes,
                                torch::stable::Tensor const& a_strides,
                                torch::stable::Tensor const& b_strides,
                                torch::stable::Tensor const& d_strides,
                                torch::stable::Tensor const& s_strides,
                                int64_t chunk_size, int64_t topk) {
  dispatch_w4afp8_moe_mm_sm90(d_tensors, a_tensors, b_tensors, a_scales,
                              b_scales, expert_offsets, problem_sizes,
                              a_strides, b_strides, d_strides, s_strides,
                              chunk_size, topk);
}

void cutlass_w4afp8_moe_mm(torch::stable::Tensor& d_tensors,
                           torch::stable::Tensor const& a_tensors,
                           torch::stable::Tensor const& b_tensors,
                           torch::stable::Tensor const& a_scales,
                           torch::stable::Tensor const& b_scales,
                           torch::stable::Tensor const& expert_offsets,
                           torch::stable::Tensor const& problem_sizes,
                           torch::stable::Tensor const& a_strides,
                           torch::stable::Tensor const& b_strides,
                           torch::stable::Tensor const& d_strides,
                           torch::stable::Tensor const& s_strides,
                           int64_t chunk_size, int64_t topk) {
  cutlass_w4afp8_moe_mm_sm90(d_tensors, a_tensors, b_tensors, a_scales,
                             b_scales, expert_offsets, problem_sizes, a_strides,
                             b_strides, d_strides, s_strides, chunk_size, topk);
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("cutlass_w4afp8_moe_mm", TORCH_BOX(&cutlass_w4afp8_moe_mm));
}
