#pragma once
#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include "native_fp32_pages.cuh"
#include "native_fp32_reduce.cuh"

namespace mx_tile {
// All 32 lanes call. Exact pinned 82ca617e shared transpose and UTCCP atom.
__device__ __forceinline__ void upload128(uint32_t* sf, uint32_t tmem) {
  const int lane = threadIdx.x & 31;
  deep_gemm::ptx::tcgen05_after_thread_sync();
  uint32_t value[4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
    value[i] = deep_gemm::ptx::ld_shared(sf + 32 * i + lane);
  __syncwarp();
  deep_gemm::ptx::st_shared(sf + 4 * lane, value[0], value[1], value[2],
                            value[3]);
  cutlass::arch::fence_view_async_shared();
  __syncwarp();
  if (lane == 0) {
    auto desc = deep_gemm::mma::sm100::make_sf_desc(sf);
    cute::SM100_UTCCP_4x32dp128bit_1cta::copy(desc, tmem);
    deep_gemm::ptx::tcgen05_before_thread_sync();
  }
  __syncwarp();
}

template <int N>
__device__ __forceinline__ void issue(uint8_t* k, uint8_t* q, uint32_t c,
                                      uint32_t sfa, uint32_t sfb,
                                      uint32_t done) {
  using F8 = cutlass::float_e4m3_t;
  const auto instr = cute::UMMA::make_instr_desc_block_scaled<
      F8, F8, float, cutlass::float_ue8m0_t, 128, N, cute::UMMA::Major::K,
      cute::UMMA::Major::K>();
#pragma unroll
  for (int step = 0; step < 4; ++step) {
    auto ad = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                    128, 128>(
        reinterpret_cast<F8*>(k), 0, step * 32);
    auto bd = deep_gemm::mma::sm100::make_umma_desc<cute::UMMA::Major::K, 0,
                                                    128, 128>(
        reinterpret_cast<F8*>(q), 0, step * 32);
    const auto runtime =
        deep_gemm::mma::sm100::make_runtime_instr_desc_with_sf_id(instr, step,
                                                                  step);
    deep_gemm::ptx::SM100_MMA_MXF8F6F4_SS::fma(ad, bd, c, step, runtime, sfa,
                                               sfb);
  }
  asm volatile(
      "tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 "
      "[%0];" ::"r"(done));
}
}  // namespace mx_tile

// Count ownership is composed with the production mapper. No standalone
// Arena ABI or explicit seed table entries are introduced.
namespace DOSA_NAMESPACE::refine_mx_pages {
__device__ __forceinline__ int warp_reserve(native_fp32_pages::Arena a, int row,
                                            unsigned mask) {
  const int lane = threadIdx.x & 31, leader = __ffs(mask) - 1;
  int begin = -1;
  if (lane == leader) begin = native_fp32_pages::reserve(a, row, __popc(mask));
  begin = __shfl_sync(mask, begin, leader);
  if (begin < 0) return -1;
  const unsigned preceding = lane ? mask & ((1u << lane) - 1u) : 0u;
  return begin + __popc(preceding);
}
}  // namespace DOSA_NAMESPACE::refine_mx_pages
