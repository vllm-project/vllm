#pragma once

#include <cute/tensor.hpp>

#include "kerutils/device/common.h"

namespace kerutils {

template <typename TMA, typename Tensor0, typename Tensor1>
CUTE_DEVICE void launch_tma_copy(const TMA& tma_copy, Tensor0 src, Tensor1 dst,
                                 transac_bar_t& bar,
                                 const cute::TMA::CacheHintSm90& cache_hint =
                                     cute::TMA::CacheHintSm90::EVICT_NORMAL) {
  auto thr_tma = tma_copy.get_slice(cute::_0{});
  cute::copy(
      tma_copy.with(reinterpret_cast<typename transac_bar_t::ValueType&>(bar),
                    0, cache_hint),
      thr_tma.partition_S(src), thr_tma.partition_D(dst));
}

// In the layout of fragment A and fragment C during WGMMA, data each thread
// holds resides in two particular rows. This function converts the
// local_row_idx (0~2) to the actual row_idx You may refer to this link for the
// detailed layout:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
CUTE_DEVICE
int get_AorC_row_idx(int local_row_idx, int idx_in_warpgroup) {
  int row_idx = (idx_in_warpgroup / 32) * 16 + local_row_idx * 8 +
                (idx_in_warpgroup % 32 / 4);
  return row_idx;
}

// In the layout of fragment A and fragment C during WGMMA, data each thread
// holds resides in some rows. This function converts the local_elem_idx to the
// actual col_idx You may refer to this link for the detailed layout:
// https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
CUTE_DEVICE
int get_AorC_col_idx(int local_elem_idx, int idx_in_warpgroup) {
  int col_idx = 8 * (local_elem_idx / 4) + (idx_in_warpgroup % 4) * 2 +
                (local_elem_idx & 1);
  return col_idx;
}

template <bool commit = true, typename Tensor0, typename Tensor1,
          typename Tensor2, typename TiledMma>
CUTE_DEVICE void wgmma(TiledMma& tiled_mma, Tensor0 const& tCrA,
                       Tensor1 const& tCrB, Tensor2& tCrC, bool zero_init) {
  using namespace cute;
  constexpr bool Is_RS = !cute::is_base_of<cute::GMMA::DescriptorIterator,
                                           typename TiledMma::FrgTypeA>::value;
  // Need to cast away const on tCrA since warpgroup_fence_operand doesn't take
  // const
  if constexpr (Is_RS) {
    cute::warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
  }
  warpgroup_fence_operand(tCrC);
  warpgroup_arrive();
  tiled_mma.accumulate_ =
      zero_init ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  // Unroll the K mode manually to set scale D to 1
  CUTLASS_PRAGMA_UNROLL
  for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
    cute::gemm(tiled_mma, tCrA(_, _, k_block), tCrB(_, _, k_block), tCrC);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  if constexpr (commit) {
    warpgroup_commit_batch();
  }
  warpgroup_fence_operand(tCrC);
  if constexpr (Is_RS) {
    warpgroup_fence_operand(const_cast<Tensor0&>(tCrA));
  }
}

template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename TiledMma>
CUTE_DEVICE void wgmma_ss(bool clear_accum, TiledMma tiled_mma,
                          Tensor0 const& sA, Tensor1 const& sB,
                          Tensor2& rC_frag, int idx_in_warpgroup) {
  using namespace cute;
  ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
  Tensor sA_frag = thr_mma.partition_fragment_A(sA);
  Tensor sB_frag = thr_mma.partition_fragment_B(sB);
  static_assert(size<2>(sA_frag) == size<2>(sB_frag));

  warpgroup_fence_operand(rC_frag);
  warpgroup_arrive();
  tiled_mma.accumulate_ =
      clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < size<2>(sA_frag); ++k) {
    cute::gemm(tiled_mma, sA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_fence_operand(rC_frag);
}

template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename TiledMma>
CUTE_DEVICE void wgmma_rs(bool clear_accum, TiledMma tiled_mma, Tensor0 rA_frag,
                          Tensor1 const& sB, Tensor2& rC_frag,
                          int idx_in_warpgroup) {
  using namespace cute;
  ThrMMA thr_mma = tiled_mma.get_slice(idx_in_warpgroup);
  Tensor sB_frag = thr_mma.partition_fragment_B(sB);
  static_assert(size<2>(rA_frag) == size<2>(sB_frag));

  warpgroup_fence_operand(const_cast<Tensor0&>(rA_frag));
  warpgroup_fence_operand(rC_frag);
  warpgroup_arrive();
  tiled_mma.accumulate_ =
      clear_accum ? GMMA::ScaleOut::Zero : GMMA::ScaleOut::One;
  CUTLASS_PRAGMA_UNROLL
  for (int k = 0; k < size<2>(rA_frag); ++k) {
    cute::gemm(tiled_mma, rA_frag(_, _, k), sB_frag(_, _, k), rC_frag);
    tiled_mma.accumulate_ = GMMA::ScaleOut::One;
  }
  warpgroup_fence_operand(rC_frag);
  warpgroup_fence_operand(const_cast<Tensor0&>(rA_frag));
}

}  // namespace kerutils
