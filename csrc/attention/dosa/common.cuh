#pragma once

#include "torch_utils.h"
#include <torch/csrc/stable/library.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/float8.h>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace DOSA_NAMESPACE {
using Tensor = torch::stable::Tensor;
using DeviceGuard = torch::stable::accelerator::DeviceGuard;
struct CandidateChunk {
  int row, begin, count;
};

namespace dosa_geometry {
constexpr int Heads = DOSA_SCORE_HEADS;
static_assert(Heads == 32 || Heads == 64, "DoSA supports H32 or H64");
__host__ __device__ constexpr bool valid_certificate_budget(int budget) {
  return budget == 14 || budget == 16 || budget == 20 ||
         (Heads == 64 && budget == 28);
}
using HeadMask = std::conditional_t<Heads == 32, int32_t, int64_t>;
using UnsignedHeadMask = std::make_unsigned_t<HeadMask>;
}  // namespace dosa_geometry
}  // namespace DOSA_NAMESPACE

#define DOSA_OP(name) DOSA_OP_PREFIX name
#define DOSA_REGISTER(name, schema, key, ...)                               \
  STABLE_TORCH_LIBRARY_FRAGMENT(_C, ops) { ops.def(DOSA_OP(name) schema); } \
  STABLE_TORCH_LIBRARY_IMPL(_C, key, ops) {                                 \
    using namespace DOSA_NAMESPACE;                                         \
    ops.impl(DOSA_OP(name), TORCH_BOX((&__VA_ARGS__)));                     \
  }
#define DOSA_CUDA_CHECK(expression)                       \
  do {                                                    \
    cudaError_t dosa_cuda_error = (expression);           \
    STD_TORCH_CHECK(dosa_cuda_error == cudaSuccess,       \
                    cudaGetErrorString(dosa_cuda_error)); \
  } while (false)

namespace DOSA_NAMESPACE {
template <auto Kernel, int Threads, size_t Shared = 0, bool Carveout = false,
          class... Args>
inline void launch(dim3 grid, Args... args) {
  if constexpr (Shared > 0)
    DOSA_CUDA_CHECK(cudaFuncSetAttribute(
        Kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, Shared));
  if constexpr (Carveout)
    DOSA_CUDA_CHECK(cudaFuncSetAttribute(
        Kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));
  Kernel<<<grid, Threads, Shared, get_current_cuda_stream()>>>(args...);
  DOSA_CUDA_CHECK(cudaGetLastError());
}

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

namespace certicascade_streaming_exact {
constexpr uint32_t kHeads = dosa_geometry::Heads, kHeadDim = 128;
__device__ __forceinline__ void named_sync(uint32_t threads, uint32_t id) {
  asm volatile("" ::: "memory");
  cutlass::arch::NamedBarrier(threads, id).arrive_and_wait_unaligned();
  asm volatile("" ::: "memory");
}
template <int Count>
__device__ __forceinline__ void load_tmem(const uint32_t address, float* out) {
  using Loader = std::conditional_t<
      Count == 64, cute::SM100_TMEM_LOAD_32dp32b64x,
      std::conditional_t<
          Count == 32, cute::SM100_TMEM_LOAD_32dp32b32x,
          std::conditional_t<
              Count == 16, cute::SM100_TMEM_LOAD_32dp32b16x,
              std::conditional_t<
                  Count == 8, cute::SM100_TMEM_LOAD_32dp32b8x,
                  std::conditional_t<Count == 4,
                                     cute::SM100_TMEM_LOAD_32dp32b4x,
                                     cute::SM100_TMEM_LOAD_32dp32b2x>>>>>;
  [&]<size_t... I>(cute::index_sequence<I...>) {
    Loader::copy(address, reinterpret_cast<uint32_t*>(out)[I]...);
  }(cute::make_index_sequence<Count>{});
}
// DG727 BF16 epilogue: promoted BF16 weights, repeated UE8M0 scales[-108,100].
// Raw/MX dot equivalence is qualified; other domains recover natively.
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
// TMA requires a16-byte outer stride even for the single-row {NK,1} map.
inline cuuint64_t scale_row_stride(int64_t nk) {
  return (static_cast<cuuint64_t>(nk) * sizeof(float) + 15) & ~cuuint64_t(15);
}

inline CUtensorMap fp8_map(const Tensor& tensor, cuuint32_t tile) {
  return make_map(tensor.data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT8,
                  {128, cuuint64_t(tensor.numel() / 128)}, {128}, {128, tile},
                  CU_TENSOR_MAP_SWIZZLE_128B);
}
}  // namespace certicascade_streaming_exact

namespace record5 {
inline constexpr uint32_t Invalid = 0xFFFFFu;
// Record5 stores low8/high12 score bits and exact20-bit key IDs.
__device__ __forceinline__ uint32_t pack_word(uint32_t code, uint32_t id) {
  return ((code >> 8) << 20) | (id & 0xFFFFFu);
}
}  // namespace record5

namespace h32_record5 {
inline constexpr uint32_t Min = 0x00800u, Max = 0xFF7FFu;
inline constexpr uint32_t Invalid = record5::Invalid;

__device__ __forceinline__ uint32_t encode(float score) {
  uint32_t bits = __float_as_uint(score);
  if ((bits & 0x7FFFFFFFu) == 0) bits = 0;
  const uint32_t ordered = (bits & 0x80000000u) ? ~bits : (bits ^ 0x80000000u);
  return (~ordered) >> 12;
}

using record5::pack_word;
}  // namespace h32_record5

namespace dosa_screen {
using certicascade_streaming_exact::fp8_map;
using certicascade_streaming_exact::make_map;
using certicascade_streaming_exact::scale_row_stride;
using Barrier = cutlass::arch::ClusterTransactionBarrier;
struct BitmapArgs {
  unsigned* words;
  int word_count;
};
struct Param {
  float guard, lo0, lo1, raw, out, tau;
  int start, end, hint;
};
struct Args {
  const float *fine_scale, *meta, *raw, *out, *tau;
  const int *hint, *starts, *ends, *budget;
  int *ids, *counts, *error, *members;
  int nq, nk, seed;
};

template <int Width, bool Trusted>
__device__ __forceinline__ Param row_parameters(const Args& a, int row) {
  return {
      a.meta[row * 64 + Width],
      a.fine_scale[row * Width],
      a.fine_scale[row * Width + 1],
      a.raw[row],
      a.out[row],
      Trusted && a.hint[row] != 2 ? __uint_as_float(0xff800000u) : a.tau[row],
      a.starts[row],
      a.ends[row],
      Trusted ? 2 : a.hint[row]};
}

template <int Readers, size_t Active, size_t Slots>
__device__ __forceinline__ void initialize_slots(Barrier (&full)[Slots],
                                                 Barrier (&empty)[Slots]) {
#pragma unroll
  for (int i = 0; i < Active; ++i) {
    full[i].init(1);
    empty[i].init(Readers);
  }
}

template <int CopyWarp, int Tmem, int Readers, int MmaReaders, int MmaSlots = 2,
          class Storage, class... Maps>
__device__ __forceinline__ void initialize_screen(Storage& s, int warp,
                                                  const Maps&... maps) {
  if (warp == CopyWarp) (cute::prefetch_tma_descriptor(&maps), ...);
  if (warp == CopyWarp && cute::elect_one_sync()) {
    s.full_q.init(1);
    if constexpr (requires { s.full_coeff; }) s.full_coeff.init(1);
    initialize_slots<Readers, std::extent_v<decltype(Storage::raw_k)>>(
        s.full_kv, s.empty_kv);
    cutlass::arch::fence_barrier_init();
  }
  if (warp == CopyWarp + 1) {
    if (cute::elect_one_sync()) {
      initialize_slots<MmaReaders, MmaSlots>(s.full_umma, s.empty_umma);
      cutlass::arch::fence_barrier_init();
    }
    cute::TMEM::Allocator1Sm().allocate(Tmem, &s.tmem);
    cute::TMEM::Allocator1Sm().release_allocation_lock();
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
}

template <int Tile, class Storage>
__device__ __forceinline__ void produce_keys(Storage& s,
                                             const cute::TmaDescriptor& keys,
                                             const cute::TmaDescriptor& scales,
                                             const cute::TmaDescriptor& rho,
                                             int seed, int first, int blocks) {
  constexpr int Slots = std::extent_v<decltype(Storage::raw_k)>;
  for (int block = 0; block < blocks; ++block) {
    const int slot = block % Slots, phase = (block / Slots) & 1;
    s.empty_kv[slot].wait(phase ^ 1);
    const int offset = seed + (first + block) * Tile;
    deep_gemm::tma::copy<128, Tile, 128>(&keys, &s.full_kv[slot], s.raw_k[slot],
                                         0, offset);
    deep_gemm::tma::copy<Tile, 1, 0>(&scales, &s.full_kv[slot],
                                     s.raw_scale[slot], offset, 0);
    deep_gemm::tma::copy<Tile, 1, 0>(&rho, &s.full_kv[slot], s.raw_rho[slot],
                                     offset, 0);
    s.full_kv[slot].arrive_and_expect_tx(Tile * 128 + 2 * Tile * 4);
  }
}

template <class Descriptor>
__device__ __forceinline__ void issue_screen_mma(uint8_t* keys,
                                                 uint8_t* queries,
                                                 uint32_t key_row,
                                                 uint32_t tmem,
                                                 Descriptor runtime) {
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    auto ad = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                    128, 128>(
        reinterpret_cast<cutlass::float_e4m3_t*>(keys), key_row, k * 32);
    auto bd = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                    128, 128>(
        reinterpret_cast<cutlass::float_e4m3_t*>(queries), 0, k * 32);
    cute::SM100_MMA_F8F6F4_SS::fma(ad, bd, tmem, k, runtime);
  }
}

__device__ __forceinline__ uint32_t trusted_keep_mask(float value, float tau,
                                                      uint32_t eligible,
                                                      uint32_t bit) {
  uint32_t mask;
  asm volatile(
      "{.reg .pred finite,below,active,keep; .reg .f32 magnitude; "
      "abs.f32 magnitude,%1; setp.lt.f32 finite,magnitude,0f7f800000; "
      "setp.lt.f32 below,%1,%2; and.pred keep,finite,below; not.pred "
      "keep,keep; "
      "setp.ne.u32 active,%3,0; and.pred keep,keep,active; selp.u32 "
      "%0,%4,0,keep;}"
      : "=r"(mask)
      : "f"(value), "f"(tau), "r"(eligible), "r"(bit));
  return mask;
}
using certicascade_streaming_exact::load_tmem;
}  // namespace dosa_screen

namespace dosa_emit_host {

struct KeyMaps {
  CUtensorMap keys, scales, rho;
};

inline KeyMaps key_maps(const std::vector<Tensor>& x, int tile) {
  using certicascade_streaming_exact::make_map;
  using certicascade_streaming_exact::scale_row_stride;
  const cuuint64_t nk = x[1].size(0);
  return {dosa_screen::fp8_map(x[1], tile),
          make_map(x[2].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, {nk, 1},
                   {scale_row_stride(nk)}, {cuuint32_t(tile), 1},
                   CU_TENSOR_MAP_SWIZZLE_NONE),
          make_map(x[3].data_ptr(), CU_TENSOR_MAP_DATA_TYPE_UINT32, {nk, 1},
                   {scale_row_stride(nk)}, {cuuint32_t(tile), 1},
                   CU_TENSOR_MAP_SWIZZLE_NONE)};
}

inline std::pair<dosa_screen::Args, dosa_screen::BitmapArgs> bind(
    const std::vector<Tensor>& x, const Tensor& error, int64_t seed,
    const Tensor& words) {
  // x15: Q,K,KS,rho,W,fine,fine_scale,meta,raw,out,hint,start,end,budget4,tau.
  const int nq = x[0].size(0), nk = x[1].size(0);
  dosa_screen::Args a{};
  a.fine_scale = x[6].mutable_data_ptr<float>();
  a.meta = x[7].mutable_data_ptr<float>();
  a.raw = x[8].mutable_data_ptr<float>();
  a.out = x[9].mutable_data_ptr<float>();
  a.hint = x[10].mutable_data_ptr<int>();
  a.starts = x[11].mutable_data_ptr<int>();
  a.ends = x[12].mutable_data_ptr<int>();
  a.budget = x[13].mutable_data_ptr<int>();
  a.tau = x[14].mutable_data_ptr<float>();
  a.nq = nq;
  a.nk = nk;
  a.seed = seed;
  a.error = error.mutable_data_ptr<int>();

  dosa_screen::BitmapArgs pg{};
  pg.words = reinterpret_cast<unsigned*>(words.mutable_data_ptr<int>());
  pg.word_count = words.size(1);
  return {a, pg};
}

}  // namespace dosa_emit_host

}  // namespace DOSA_NAMESPACE
