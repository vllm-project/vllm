#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/all.h>

// clang-format off
// The cutlass include order matters (annoyingly)

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
#include "machete_interleaving_utils.cuh"

namespace machete {

using namespace cute;

struct IlvBlkLayoutAuto {};

// This defines a prepacked layout for the B matrix, where the matrix is broken
// up into PPBlockShape_NK blocks. The data within each block is then compactly
// stored in memory such that when performing a TiledMMA operation with the same
// shape as prepacked block, all the data for a given thread is contiguous in
// memory. This allows us to use wider shared memory loads when loading B from
// shared memory. The values within a thread are also potentially interlaeved
// inorder to allow for more efficient upconverting.
//
// The contract here is that the `TiledMma` determined below matches the one
// ultimately used in the kernel. (this is also why the other element types are
// required along with the kernel schedule)
template <typename ElementA_, typename ElementB_, typename ElementConvert_,
          typename AccumulatorT, class LayoutB, class KernelSchedule,
          typename IlvBlkLayout_ = IlvBlkLayoutAuto>
// clang-format on
struct PrepackedLayoutBTemplate {
  using MmaType = ElementA_;
  using ElementA = ElementA_;
  using ElementB = ElementB_;
  using ElementAccumulator = AccumulatorT;
  using ElementMma = MmaType;

  // Interleave for 4bit bit types when we are not upconverting to fp8 or int8,
  // in those cases case we use a LUT using prmt instructions to upconvert and
  // is more efficient if the data is not interleaved For 8bit+ prmt
  // instructions makes non-interleaved layouts efficient enough we don't need
  // iterleaved layouts (and can reuse more of the existing cutlass converts)
  static constexpr bool should_interleave =
      sizeof_bits_v<ElementB> <= 4 &&
      !std::is_same_v<ElementConvert_, cutlass::float_e4m3_t> &&
      !std::is_same_v<ElementConvert_, int8_t>;

  // Only use interleaved layouts for subbyte weights,
  using IlvdBlkLayout = std::conditional_t<
      std::is_same_v<IlvBlkLayout_, IlvBlkLayoutAuto>,
      std::conditional_t<
          should_interleave,
          decltype(get_interleaved_blk_layout<
                   ElementB, sizeof_bits_v<ElementConvert_>, 32>()),
          void>,
      IlvBlkLayout_>;

  // TODO (LucasWilkinson): compare the performance for other sizes
  // Prepacked block shape, smallest layout atom for loading into registers
  //   (can contain multiple wgmma instructions worth of data in one block)
  // We ideally want this to be configured such that a thread can perform 128bit
  // loads, i.e. we amount of data associated with each thread within a
  // prepacked block is a multiple of 128bits, when using a cooperative sechdule
  // we have 256 threads working a single block at a time, this means each
  // thread works on `sizeof_bits_v<ElementB> * (128*64) / 256` bits of data,
  // for a 4bit type this would be 128bits
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

  // For coop schedules we have two warp groups cooperatively issuing wgmma
  // instructions so we use 2 atoms along the M dim (one for each warpgroup)
  using AtomLayoutMNK = cute::conditional_t<
      cute::is_same_v<KernelSchedule, KernelTmaWarpSpecializedCooperative>,
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
    // Return iterleaved layout
    return make_ordered_layout(shape(ppblock_TV_to_NK()), Step<_1, _0>{});
  }

  // Prepacked block, (athrid, val) -> (storage_offset)
  // i.e. ((ThrV,(ThrM,ThrK)),(IlvdFrgV,(RestM,RestK,...))) -> (storage_idx)
  CUTE_HOST_DEVICE static constexpr auto ppblock_ilvd_TV_to_offset() {
    auto layout_no_interleave =
        make_ordered_layout(shape(ppblock_TV_to_NK()), Step<_1, _0>{});

    if constexpr (std::is_same_v<IlvdBlkLayout, void>) {
      return layout_no_interleave;
    } else {
      // interleave by transforming FrgV into interleaved blocks where each
      // block has the layout IlvdBlkLayout, for example if IlvdBlkLayout is
      // (2, 2) : (2, 1) then we get: ((2, 2), size(FrgV) / 4) : ((2, 1), 4)
      //   if FrgV is {A, B, C, D, E, F, G, H}
      //   then ((IlvBlk), FrgB) is {A, C, B, D, C, G, D, H}
      auto frgV = get<1, 0>(layout_no_interleave);
      auto ilvdBlk = IlvdBlkLayout{};
      static_assert(size(frgV) % size(ilvdBlk) == 0,
                    "FrgV must be divisible by size(ilvdBlk)");
      auto ilvd_FrgV = make_layout(
          make_shape(shape(ilvdBlk), Int<size(frgV) / size(ilvdBlk)>{}),
          make_stride(stride(ilvdBlk), size(ilvdBlk)));

      // Return iterleaved layout
      return make_layout(
          get<0>(layout_no_interleave),
          make_layout(ilvd_FrgV, get<1, 1>(layout_no_interleave)));
    }
  }

  // Prepacked block, (M,K) -> (storage_offset)
  CUTE_HOST_DEVICE static constexpr auto ppblock_ilvd_NK_to_offset() {
    // do (M,K) -> (athrid, val) -> (storage_idx)
    return ppblock_ilvd_TV_to_offset().compose(ppblock_NK_to_TV());
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

  // ((athrid_val), (BlocksN, BlocksK, L)) -> (N, K, L)
  template <typename Shape_NKL>
  CUTE_HOST_DEVICE static constexpr auto TVbNbKL_to_offset_copy(
      Shape_NKL shape_mkl) {
    auto layout = TVbNbKL_to_offset(shape_mkl);
    // for 4-bit elements, having >= 64 values per column
    // allows TMA to load full 32-byte sectors
    auto inner_layout =
        make_layout(make_shape(_256{}, size<0>(layout) / _256{}));

    return make_layout(inner_layout, get<1>(layout), get<2>(layout));
  }

  // ((BlockN, BlockK), (BlocksN, BlocksK), L) -> (storage_idx)
  template <typename Shape_NKL>
  CUTE_HOST_DEVICE static constexpr auto ilvd_NKbNbKL_to_offset(
      Shape_NKL shape_mkl) {
    constexpr auto block_layout = ppblock_ilvd_NK_to_offset();

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

  // (BlocksN, BlocksK, L) -> (storage_idx)
  template <typename Shape_NKL>
  CUTE_HOST_DEVICE static constexpr auto bNbKL_to_offset(Shape_NKL shape_mkl) {
    // (BlocksN, BlocksK, L)
    auto blocks_shape =
        cute::transform(shape_mkl, append(PPBlockShape_NK{}, _1{}),
                        [](auto x, auto y) { return x / y; });
    auto stride = size(PPBlockShape_NK{});

    // (BlocksN, BlocksK, L) -> (storage_idx)
    return make_layout(blocks_shape, compact_col_major(blocks_shape, stride));
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
