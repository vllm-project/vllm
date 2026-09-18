/* Copyright 2025 SGLang Team. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

namespace sm90 {

__forceinline__ __device__ void cp_async_cacheglobal_l2_prefetch_256B(
    const void* src, void* dst, bool pred, int64_t cache_policy) {
  uint32_t dst_addr = cute::cast_smem_ptr_to_uint(dst);
  asm volatile(
      "cp.async.cg.shared.global.L2::cache_hint.L2::256B [%0], [%1], 16, %2, "
      "%3;\n" ::"r"(dst_addr),
      "l"(src), "r"(pred ? 16 : 0), "l"(cache_policy));
}

__forceinline__ __device__ int64_t createpolicy_evict_last() {
  int64_t res;
  asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0; \n\t"
               : "=l"(res)
               :);
  return res;
}

__forceinline__ __device__ int64_t createpolicy_evict_first() {
  int64_t res;
  asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0; \n\t"
               : "=l"(res)
               :);
  return res;
}

__forceinline__ __device__ int get_AorC_row_idx(int local_row_idx,
                                                int idx_in_warpgroup) {
  // In the layout of fragment A and fragment C during WGMMA, the data each
  // thread holds resides in two particular rows. This function converts the
  // local_row_idx (0~2) to the actual row_idx You may refer to this link for
  // the detailed layout:
  // https://docs.nvidia.com/cuda/parallel-thread-execution/#wgmma-64n16-a
  int row_idx = (idx_in_warpgroup / 32) * 16 + local_row_idx * 8 +
                (idx_in_warpgroup % 32 / 4);
  return row_idx;
}

// A simpler version of gemm
template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename TiledMma>
__forceinline__ __device__ void gemm_ss(bool clear_accum, TiledMma tiled_mma,
                                        Tensor0 const& sA, Tensor1 const& sB,
                                        Tensor2& rC_frag,
                                        int idx_in_warpgroup) {
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
__forceinline__ __device__ void gemm_rs(bool clear_accum, TiledMma tiled_mma,
                                        Tensor0 rA_frag, Tensor1 const& sB,
                                        Tensor2& rC_frag,
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

}  // namespace sm90
