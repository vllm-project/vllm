#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// clang-format off
// The cutlass include order 
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

#include "cutlass_extensions/cute_utils.cuh"
#include "machete_collective_builder.cuh"

namespace machete {

using namespace cute;

template <typename ElementA_, typename ElementB_, typename ElementD_,
          typename AccumulatorT, class LayoutB, class KernelSchedule>
// clang-format on
struct PrepackedLayoutBTemplate {
  using MmaType = ElementA_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementD = ElementD_;
  using ElementAccumulator =
      AccumulatorT;  // Element type for internal accumulation
  using ElementMma = MmaType;

  // TODO (Lucas): compare the performance for other sizes
  // Prepacked block shape, smallest layout atom for loading into registers
  //   (can contain multiple wgmma instructions worth of data in one block)
  using PPBlockShape_NK = Shape<_128, _64>;

  // Create the shape of the tile anticipated to be used by the GEMM kernel,
  //  when the kernel executes we will compute `Ct = Bt * At` since the
  //  quantized weights (B), must be the lhs operand so the flow through
  //  registers.
  // The _128 here doesn't actually impact the shape of the stored tile directly
  //  but may impact the op selected by rs_op_selector
  using GemmTileShape = decltype(make_shape(size<0>(PPBlockShape_NK{}), _128{},
                                            size<1>(PPBlockShape_NK{})));

  static constexpr cute::GMMA::Major GmmaMajorB =
      gmma_rs_tag_to_major_B<LayoutB>();
  using AtomLayoutMNK = cute::conditional_t<
      cute::is_same_v<KernelSchedule,
                      KernelTmaWarpSpecializedCooperativeMixedInput>,
      Layout<Shape<_2, _1, _1>>, Layout<Shape<_1, _1, _1>>>;

  using TiledMma = decltype(cute::make_tiled_mma(
      cute::GMMA::rs_op_selector<ElementMma, ElementMma, ElementAccumulator,
                                 GemmTileShape, GMMA::Major::K, GmmaMajorB>(),
      AtomLayoutMNK{}));

  // Prepacked block, (athrid, val) -> (N,K)
  // i.e. ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK,...))) -> (N,K)
  CUTE_HOST_DEVICE static constexpr auto ppblock_TV_to_NK() {
    return TiledMma{}.thrfrg_A(make_layout(PPBlockShape_NK{}));
  }

  // Prepacked block, (N,K) -> (athrid, val)
  // i.e. (N,K) -> ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK,...)))
  CUTE_HOST_DEVICE static constexpr auto ppblock_NK_to_TV() {
    return right_inverse(ppblock_TV_to_NK()).with_shape(PPBlockShape_NK{});
  }

  // Prepacked block, (athrid, val) -> (storage_offset)
  // i.e. ((ThrV,(ThrN,ThrK)),(FrgV,(RestN,RestK,...))) -> (storage_idx)
  CUTE_HOST_DEVICE static constexpr auto ppblock_TV_to_offset() {
    return make_ordered_layout(shape(ppblock_TV_to_NK()), Step<_1, _0>{});
  }

  // Prepacked block, (N,K) -> (storage_offset)
  CUTE_HOST_DEVICE static constexpr auto ppblock_NK_to_offset() {
    // do (N,K) -> (athrid, val) -> (storage_idx)
    return ppblock_TV_to_offset().compose(ppblock_NK_to_TV());
  }

  // ((athrid, val), (BlocksN, BlocksK), L) -> (storage_idx)
  template <typename Shape_NKL>
  CUTE_HOST_DEVICE static constexpr auto TVbNbKL_to_offset(
      Shape_NKL shape_mkl) {
    constexpr auto block_layout = ppblock_TV_to_offset();

    // (BlocksN, BlocksK, L)
    auto blocks_shape =
        cute::transform(shape_mkl, append(PPBlockShape_NK{}, _1{}),
                        [](auto x, auto y) { return x / y; });

    // ((athrid, val), (BlocksN, BlocksK, L)) -> (storage_idx)
    auto result = make_layout(
        block_layout,
        make_layout(blocks_shape,
                    compact_col_major(blocks_shape, size(block_layout))));

    // ((athrid, val), (BlocksN, BlocksK, L))
    //   => ((athrid, val), (BlocksN, BlocksK), L)
    return group<1, 3>(result(_, repeat<rank<1>(result)>(_)));
  }

  // ((BlockN, BlockK), (BlocksN, BlocksK), L) -> (storage_idx)
  template <typename Shape_NKL>
  CUTE_HOST_DEVICE static constexpr auto NKbNbKL_to_offset(
      Shape_NKL shape_mkl) {
    constexpr auto block_layout = ppblock_NK_to_offset();

    // (BlocksN, BlocksK, L)
    auto blocks_shape =
        cute::transform(shape_mkl, append(PPBlockShape_NK{}, _1{}),
                        [](auto x, auto y) { return x / y; });

    // ((athrid, val), (BlocksN, BlocksK, L)) -> (storage_idx)
    auto result = make_layout(
        block_layout,
        make_layout(blocks_shape,
                    compact_col_major(blocks_shape, size(block_layout))));

    // ((athrid, val), (BlocksN, BlocksK, L)) => ((athrid, val), (BlocksN,
    // BlocksK), L)
    return group<1, 3>(result(_, repeat<rank<1>(result)>(_)));
  }

  // ((athrid, val), (BlocksN, BlocksK, L)) -> (N, K, L)
  template <class Shape_NKL>
  CUTE_HOST_DEVICE static auto TVbNbK_to_NKL(Shape_NKL shape_mkl) {
    auto tile = make_tile(make_layout(size<0>(PPBlockShape_NK{})),
                          make_layout(size<1>(PPBlockShape_NK{})));

    // ((BlockN, BlockK), (BlocksN, BlocksK, L)) -> (N, K, L)
    auto tiled_A = zipped_divide(make_layout(shape_mkl), tile);
    return tiled_A.compose(ppblock_TV_to_NK(), _);
  }

  // (N, K, L) -> ((athrid, val), (BlocksN, BlocksK), L)
  template <class Shape_NKL>
  CUTE_HOST_DEVICE static auto NKL_to_TVbNbK(Shape_NKL shape_mkl) {
    auto TVbNbK_to_NKL_layout = TVbNbK_to_NKL(shape_mkl);
    return blocked_product(ppblock_NK_to_TV(),
                           make_layout(shape<1>(TVbNbK_to_NKL_layout)));
  }
};

};  // namespace machete