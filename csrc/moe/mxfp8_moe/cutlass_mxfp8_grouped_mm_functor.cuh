// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
// Adapted from SGLang:
// https://github.com/sgl-project/sglang/blob/ded068a76e00878881d52d5bfb791e0f60d7311b/sgl-kernel/csrc/expert_specialization/es_sm100_mxfp8_blockscaled_functor.cuh

#pragma once
#include <cuda.h>

#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass_mxfp8_grouped_mm_traits.cuh"

namespace expert_specialization {

using namespace cute;

template <typename GemmTraits>
struct CutlassMxfp8GroupedMmOffsetFunctor {
  using Gemm = typename GemmTraits::Gemm;
  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementSF = typename GemmTraits::ElementSF;
  using ElementD = typename GemmTraits::ElementOutput;
  // Input
  int* expert_offsets{nullptr};
  int* blockscale_offsets{nullptr};
  // Output
  ElementA* a_base{nullptr};
  ElementB* b_base{nullptr};
  ElementSF* sfa_base{nullptr};
  ElementSF* sfb_base{nullptr};
  ElementD* d_base{nullptr};
  ElementA** a_offsets{nullptr};
  ElementB** b_offsets{nullptr};
  ElementSF** sfa_offsets{nullptr};
  ElementSF** sfb_offsets{nullptr};
  ElementD** d_offsets{nullptr};

  CutlassMxfp8GroupedMmOffsetFunctor() = default;
  CutlassMxfp8GroupedMmOffsetFunctor(
      int* _expert_offsets, int* _blockscale_offsets, ElementA* _a_base,
      ElementB* _b_base, ElementSF* _sfa_base, ElementSF* _sfb_base,
      ElementD* _d_base, ElementA** _a_offsets, ElementB** _b_offsets,
      ElementSF** _sfa_offsets, ElementSF** _sfb_offsets, ElementD** _d_offsets)
      : expert_offsets{_expert_offsets},
        blockscale_offsets{_blockscale_offsets},
        a_base(_a_base),
        b_base(_b_base),
        sfa_base(_sfa_base),
        sfb_base(_sfb_base),
        d_base(_d_base),
        a_offsets(_a_offsets),
        b_offsets(_b_offsets),
        sfa_offsets(_sfa_offsets),
        sfb_offsets(_sfb_offsets),
        d_offsets(_d_offsets) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    int64_t expert_offset = static_cast<int64_t>(expert_offsets[expert_id]);
    int64_t blockscale_offset =
        static_cast<int64_t>(blockscale_offsets[expert_id]);
    int64_t a_stride = expert_offset * k;
    int64_t b_stride = expert_id * k * n;
    int64_t d_stride = expert_offset * n;
    int64_t sfa_stride = blockscale_offset * (k / 32);
    int64_t sfb_stride = expert_id * n * (k / 32);

    a_offsets[expert_id] = a_base + a_stride;
    b_offsets[expert_id] = b_base + b_stride;
    sfa_offsets[expert_id] = sfa_base + sfa_stride;
    sfb_offsets[expert_id] = sfb_base + sfb_stride;
    d_offsets[expert_id] = d_base + d_stride;
  }
};

template <typename GemmTraits>
struct CutlassMxfp8GroupedMmLayoutFunctor {
  using Sm1xxBlkScaledConfig = typename GemmTraits::Sm1xxBlkScaledConfig;
  using LayoutSFA = typename GemmTraits::LayoutSFA;
  using LayoutSFB = typename GemmTraits::LayoutSFB;
  LayoutSFA* layout_sfa_base{nullptr};
  LayoutSFB* layout_sfb_base{nullptr};

  CutlassMxfp8GroupedMmLayoutFunctor() = default;
  CutlassMxfp8GroupedMmLayoutFunctor(LayoutSFA* _layout_sfa_base,
                                     LayoutSFB* _layout_sfb_base)
      : layout_sfa_base(_layout_sfa_base), layout_sfb_base(_layout_sfb_base) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    LayoutSFA* layout_sfa_ptr = layout_sfa_base + expert_id;
    LayoutSFB* layout_sfb_ptr = layout_sfb_base + expert_id;
    *layout_sfa_ptr = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(
        cute::make_shape(m, n, k, 1));
    *layout_sfb_ptr = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(
        cute::make_shape(m, n, k, 1));
  }
};

template <typename GemmTraits>
struct CutlassMxfp8GroupedMmStrideFunctor {
  using StrideA = typename GemmTraits::StrideA;
  using StrideB = typename GemmTraits::StrideB;
  using StrideD = typename GemmTraits::StrideD;
  StrideA* stride_A_base{nullptr};
  StrideB* stride_B_base{nullptr};
  StrideD* stride_D_base{nullptr};

  CutlassMxfp8GroupedMmStrideFunctor() = default;
  CutlassMxfp8GroupedMmStrideFunctor(StrideA* _stride_A_base,
                                     StrideB* _stride_B_base,
                                     StrideD* _stride_D_base)
      : stride_A_base(_stride_A_base),
        stride_B_base(_stride_B_base),
        stride_D_base(_stride_D_base) {}

  void CUTE_DEVICE operator()(int64_t expert_id, int m, int n, int k) {
    StrideA* stride_A = stride_A_base + expert_id;
    StrideB* stride_B = stride_B_base + expert_id;
    StrideD* stride_D = stride_D_base + expert_id;
    *stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    *stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    *stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
  }
};

template <typename OffsetFunctor, typename LayoutFunctor,
          typename StrideFunctor>
__global__ void cutlassMxfp8GroupedMmPreComputeKernel(
    int* problem_sizes, OffsetFunctor offset_functor,
    LayoutFunctor layout_functor, StrideFunctor stride_functor) {
  int64_t expert_id = static_cast<int64_t>(threadIdx.x);
  int m = problem_sizes[expert_id * 3 + 0];
  int n = problem_sizes[expert_id * 3 + 1];
  int k = problem_sizes[expert_id * 3 + 2];

  offset_functor(expert_id, m, n, k);
  layout_functor(expert_id, m, n, k);
  stride_functor(expert_id, m, n, k);
}

}  // namespace expert_specialization