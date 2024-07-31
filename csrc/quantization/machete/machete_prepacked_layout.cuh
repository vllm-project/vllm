#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// clang-format off
// The cutlass inlcude order 
#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
// clang-format on

#include "cutlass/util/device_memory.h"

#include "cutlass_extensions/gemm/kernel/vllm_tile_schedulers.cuh"
#include "cutlass_extensions/cute_utils.cuh"
#include "machete_collective_builder.cuh"

namespace machete {

using namespace cute;

template <typename ElementA_, typename ElementB_, typename ElementD_,
          typename AccumulatorT, class LayoutB, class KernelSchedule>
// clang-format on
struct PrepackedLayoutBBTemplate {
  using MmaType = ElementA_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementD = ElementD_;
  using ElementAccumulator =
      AccumulatorT;  // Element type for internal accumulation
  using ElementMma = MmaType;

  // TODO (Lucas): compare the performance for other sizes
  using PPBlockShape_MK = Shape<_128, _64>;

  // The N here doesnt actually impact the shape of the stored tile directly but
  // may impact the op selected by rs_op_selector
  using PPBlockShape_MNK = decltype(make_shape(
      size<0>(PPBlockShape_MK{}), _128{}, size<1>(PPBlockShape_MK{})));

  static constexpr cute::GMMA::Major GmmaMajorB =
      gmma_rs_tag_to_major_B<LayoutB>();
  using AtomLayoutMNK = cute::conditional_t<
      cute::is_same_v<KernelSchedule,
                      KernelTmaWarpSpecializedCooperativeMixedInput>,
      Layout<Shape<_2, _1, _1>>, Layout<Shape<_1, _1, _1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<ElementMma, ElementMma, ElementAccumulator,
                                 PPBlockShape_MNK, GMMA::Major::K,
                                 GmmaMajorB>(),
      AtomLayoutMNK{}));

  // Prepacked block, (athrid, val) -> (M,K)
  // i.e. ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...))) -> (M,K)
  CUTE_HOST_DEVICE static constexpr auto ppblock_TV_to_MK() {
    return TiledMma{}.thrfrg_A(make_layout(PPBlockShape_MK{}));
  }

  // Prepacked block, (M,K) -> (athrid, val)
  // i.e. (M,K) -> ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...)))
  CUTE_HOST_DEVICE static constexpr auto ppblock_MK_to_TV() {
    return right_inverse(ppblock_TV_to_MK()).with_shape(PPBlockShape_MK{});
  }

  // Prepacked block, (athrid, val) -> (storage_offset)
  // i.e. ((ThrV,(ThrM,ThrK)),(FrgV,(RestM,RestK,...))) -> (storage_idx)
  CUTE_HOST_DEVICE static constexpr auto ppblock_TV_to_offset() {
    return make_ordered_layout(shape(ppblock_TV_to_MK()), Step<_1, _0>{});
  }

  // Prepacked block, (M,K) -> (storage_offset)
  CUTE_HOST_DEVICE static constexpr auto ppblock_MK_to_offset() {
    // do (M,K) -> (athrid, val) -> (storage_idx)
    return ppblock_TV_to_offset().compose(ppblock_MK_to_TV());
  }

  // ((athrid, val), (BlocksM, BlocksK), L) -> (storage_idx)
  template <typename Shape_MKL>
  CUTE_HOST_DEVICE static constexpr auto TVbMbKL_to_offset(
      Shape_MKL shape_mkl) {
    constexpr auto block_layout = ppblock_TV_to_offset();

    // (BlocksM, BlocksK, L)
    auto blocks_shape =
        cute::transform(shape_mkl, append(PPBlockShape_MK{}, _1{}),
                        [](auto x, auto y) { return x / y; });

    // ((athrid, val), (BlocksM, BlocksK, L)) -> (storage_idx)
    auto result = make_layout(
        block_layout,
        make_layout(blocks_shape,
                    compact_col_major(blocks_shape, size(block_layout))));

    // ((athrid, val), (BlocksM, BlocksK, L)) => ((athrid, val), (BlocksM,
    // BlocksK), L)
    return group<1, 3>(result(_, repeat<rank<1>(result)>(_)));
  }

  // ((BlockM, BlockK), (BlocksM, BlocksK), L) -> (storage_idx)
  template <typename Shape_MKL>
  CUTE_HOST_DEVICE static constexpr auto MKbMbKL_to_offset(
      Shape_MKL shape_mkl) {
    constexpr auto block_layout = ppblock_MK_to_offset();

    // (BlocksM, BlocksK, L)
    auto blocks_shape =
        cute::transform(shape_mkl, append(PPBlockShape_MK{}, _1{}),
                        [](auto x, auto y) { return x / y; });

    // ((athrid, val), (BlocksM, BlocksK, L)) -> (storage_idx)
    auto result = make_layout(
        block_layout,
        make_layout(blocks_shape,
                    compact_col_major(blocks_shape, size(block_layout))));

    // ((athrid, val), (BlocksM, BlocksK, L)) => ((athrid, val), (BlocksM,
    // BlocksK), L)
    return group<1, 3>(result(_, repeat<rank<1>(result)>(_)));
  }

  // ((athrid, val), (BlocksM, BlocksK, L)) -> (M, K, L)
  template <class Shape_MKL>
  CUTE_HOST_DEVICE static auto TVbMbK_to_MKL(Shape_MKL shape_mkl) {
    auto tile = make_tile(make_layout(size<0>(PPBlockShape_MK{})),
                          make_layout(size<1>(PPBlockShape_MK{})));

    // ((BlockM, BlockK), (BlocksM, BlocksK, L)) -> (M, K, L)
    auto tiled_A = zipped_divide(make_layout(shape_mkl), tile);
    return tiled_A.compose(ppblock_TV_to_MK(), _);
  }

  // (M, K, L) -> ((athrid, val), (BlocksM, BlocksK), L)
  template <class Shape_MKL>
  CUTE_HOST_DEVICE static auto MKL_to_TVbMbK(Shape_MKL shape_mkl) {
    auto TVbMbK_to_MKL_layout = TVbMbK_to_MKL(shape_mkl);
    return blocked_product(ppblock_MK_to_TV(),
                           make_layout(shape<1>(TVbMbK_to_MKL_layout)));
  }
};

};  // namespace machete