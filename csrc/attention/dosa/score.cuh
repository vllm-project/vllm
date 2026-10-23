#pragma once
#include <cuda_bf16.h>

#include "common.cuh"
#include "geometry.cuh"

namespace DOSA_NAMESPACE {
// Copyright (c) 2025 DeepSeek
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Pipeline portions adapted from sgl-deep-gemm v0.1.5.post3 (SM100 H32).
// Raw FP8 dot transport is retained; the BF16 epilogue follows DG727 below.

namespace dosa_score_emit {
template <uint32_t M, uint32_t N, typename Fp8>
__device__ __forceinline__ void issue_h32_mma(Fp8* keys, Fp8* queries,
                                              uint32_t key_row, uint32_t tmem,
                                              uint32_t completion_barrier) {
  auto instruction =
      cute::UMMA::make_instr_desc<cutlass::float_e4m3_t, cutlass::float_e4m3_t,
                                  float, M, N, cute::UMMA::Major::K,
                                  cute::UMMA::Major::K>();
  auto runtime = cute::UMMA::make_runtime_instr_desc(instruction);
#pragma unroll
  for (uint32_t k = 0; k < 128 / 32; ++k) {
    auto ad =
        deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128,
                                              128>(keys, key_row, k * 32);
    auto bd =
        deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0, 128,
                                              128>(queries, 0, k * 32);
    deep_gemm::ptx::SM100_MMA_F8F6F4_SS::fma(ad, bd, tmem, k, runtime);
  }
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one."
      "shared::cluster.b64 [%0];" ::"r"(completion_barrier));
}
}  // namespace dosa_score_emit

// Helper bodies copied verbatim from the validated N45 source.
// Only avoid compiling unrelated N45 entry points during research iteration.
namespace certicascade_streaming_exact {
constexpr uint32_t kHeads = dosa_geometry::Heads, kHeadDim = 128;
__device__ __forceinline__ void named_sync(uint32_t threads, uint32_t id) {
  asm volatile("" ::: "memory");
  cutlass::arch::NamedBarrier(threads, id).arrive_and_wait_unaligned();
  asm volatile("" ::: "memory");
}
__device__ __forceinline__ void tmem_load(const uint32_t address,
                                          float* destination) {
  [&]<size_t... Is>(cute::index_sequence<Is...>) {
    cute::SM100_TMEM_LOAD_32dp32b16x::copy(
        address, reinterpret_cast<uint32_t*>(destination)[Is]...);
  }(cute::make_index_sequence<16>{});
}
__device__ __forceinline__ void tmem_load_heads(uint32_t address,
                                                float* destination) {
  tmem_load(address, destination);
  cutlass::arch::fence_view_async_tmem_load();
  tmem_load(address + 16, destination + 16);
  cutlass::arch::fence_view_async_tmem_load();
  if constexpr (kHeads == 64) {
    tmem_load(address + 32, destination + 32);
    cutlass::arch::fence_view_async_tmem_load();
    tmem_load(address + 48, destination + 48);
    cutlass::arch::fence_view_async_tmem_load();
  }
}
// Matches DG727's epilogue if raw-dot * scale equals its MX-scaled dot.
// Weights are exact F32 promotions of BF16 values; scale is a positive repeated
// UE8M0 power2 with exponent in [-108,100]. Raw/MX MMA equivalence still needs
// target qualification. Other scales require true native dense recovery.
__device__ __forceinline__ float native_bf16_reduce(const float* accum,
                                                    const float* weights,
                                                    float scale) {
  auto sum0 = __floats2bfloat162_rn(0.0f, 0.0f);
  auto sum1 = __floats2bfloat162_rn(0.0f, 0.0f);
#pragma unroll
  for (uint32_t h = 0; h < kHeads; h += 4) {
    const float2 dot0 =
        make_float2(__fmul_rn(accum[h], scale), __fmul_rn(accum[h + 1], scale));
    const float2 dot1 = make_float2(__fmul_rn(accum[h + 2], scale),
                                    __fmul_rn(accum[h + 3], scale));
    const auto w0 = __floats2bfloat162_rn(weights[h], weights[h + 1]);
    const auto w1 = __floats2bfloat162_rn(weights[h + 2], weights[h + 3]);
    sum0 = __hfma2(deep_gemm::ptx::cvt_relu_bf16x2_f32(dot0), w0, sum0);
    sum1 = __hfma2(deep_gemm::ptx::cvt_relu_bf16x2_f32(dot1), w1, sum1);
  }
  const auto pair = __hadd2_rn(sum0, sum1);
  const auto score = __hadd2_rn(__low2bfloat162(pair), __high2bfloat162(pair));
  return __bfloat162float(__low2bfloat16(score));
}

using dosa_score_emit::issue_h32_mma;
inline CUtensorMap make_map(void* pointer, CUtensorMapDataType dtype,
                            std::initializer_list<cuuint64_t> dimensions,
                            std::initializer_list<cuuint64_t> strides,
                            std::initializer_list<cuuint32_t> box,
                            CUtensorMapSwizzle swizzle) {
  CUtensorMap map;
  const cuuint32_t element_strides[3] = {1, 1, 1};
  const CUresult result = cuTensorMapEncodeTiled(
      &map, dtype, dimensions.size(), pointer, dimensions.begin(),
      strides.begin(), box.begin(), element_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  STD_TORCH_CHECK(result == CUDA_SUCCESS,
                  "cuTensorMapEncodeTiled failed: ", result);
  return map;
}
// A {NK,1} F32 tensor map still requires a 16-byte outer stride. The second
// coordinate is always zero, so aligning this stride does not read padding.
inline cuuint64_t scale_row_stride(int64_t nk) {
  return (static_cast<cuuint64_t>(nk) * sizeof(float) + 15) & ~cuuint64_t(15);
}
}  // namespace certicascade_streaming_exact

}  // namespace DOSA_NAMESPACE
