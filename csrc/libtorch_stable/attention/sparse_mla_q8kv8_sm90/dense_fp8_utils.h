/*
 * Taken from FlashMLA PR https://github.com/deepseek-ai/FlashMLA/pull/54
 * originally authored by @endurehero
 */

// Adapted from
// https://github.com/Dao-AILab/flash-attention/blob/main/hopper/utils.h

#pragma once

#include <cute/tensor.hpp>
#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

// For SM80, convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M,
// MMA_N / 2) if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8. For
// SM90, FP16/BF16, convert acc_layout from ((2, 2, N / 8), MMA_M, MMA_N) to
// ((2, 2, 2), MMA_M, (N / 16, MMA_N)) For SM90, FP8, convert acc_layout from
// ((2, 2, N / 8), MMA_M, MMA_N) to ((4, 2, 2), MMA_M, (N / 32, MMA_N))
template <typename MMA_Traits, typename Layout0>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout0 acc_layout) {
  using X = Underscore;
  if constexpr (decltype(rank<0>(acc_layout))::value == 3) {  // SM90
    static_assert(decltype(size<0, 0>(acc_layout))::value == 2);
    static_assert(decltype(size<0, 1>(acc_layout))::value == 2);
    static_assert(decltype(rank(acc_layout))::value == 3);
    static_assert(decltype(rank(get<0>(acc_layout)))::value == 3);
    if constexpr (sizeof(typename MMA_Traits::ValTypeA) == 2) {
      auto l =
          logical_divide(get<0, 2>(acc_layout), Tile<_2>{});  // ((2, N / 16))
      return make_layout(
          make_layout(get<0, 0>(acc_layout), get<0, 1>(acc_layout),
                      get<0, 0>(l)),
          get<1>(acc_layout),
          coalesce(make_layout(get<0, 1>(l), get<2>(acc_layout))));
    } else {
      static_assert(sizeof(typename MMA_Traits::ValTypeA) == 1);
      static_assert(decltype(stride<0, 0>(acc_layout))::value == 1);
      static_assert(decltype(stride<0, 1>(acc_layout))::value == 2);
      auto l =
          logical_divide(get<0, 2>(acc_layout),
                         Tile<Layout<Shape<_2, _2>>>{});  // (((2, 2), N / 32))
      // This combines the first two modes (<0, 0> and <0, 1>) into one mode.
      // Will require register shuffling later to be correct.
      return make_layout(
          make_layout(Layout<_4>{}, get<0, 0, 0>(l), get<0, 0, 1>(l)),
          get<1>(acc_layout),
          coalesce(make_layout(
              get<0, 1>(l),
              get<2>(acc_layout))));  // ((4, 2, 2), MMA_M, N / 32 * MMA_N)
      // This combination is right but doesn't work with register shuffling.
      // return make_layout(make_layout(coalesce(make_layout(get<0,
      // 0>(acc_layout), get<0, 0, 0>(l))), get<0, 1>(acc_layout), get<0, 0,
      // 1>(l)),
      //                    get<1>(acc_layout),
      //                    coalesce(make_layout(get<0, 1>(l),
      //                    get<2>(acc_layout))));
    }
  } else {  // SM80
    static_assert(decltype(size<0>(acc_layout))::value == 4);
    static_assert(decltype(rank(acc_layout))::value == 3);
    constexpr int mma_shape_K = get<2>(typename MMA_Traits::Shape_MNK{});
    static_assert(mma_shape_K == 8 || mma_shape_K == 16);
    if constexpr (mma_shape_K == 8) {
      return acc_layout;
    } else {
      auto l = logical_divide(
          acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
      return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l),
                         get<2, 1>(l));
    }
  }
};

template <typename Fragment>
CUTLASS_DEVICE void permute_Cregs_fp8(Fragment& frag) {
  // frag has shape ((2, 2, N / 8), MMA_M, MMA_N), each element is 32 bits
  static_assert(decltype(size<0, 0>(frag))::value == 2);
  static_assert(decltype(size<0, 1>(frag))::value == 2);
  static_assert(decltype(size<0, 2>(frag))::value % 2 == 0);
  static_assert(decltype(stride<0, 0>(frag))::value == 1);
  static_assert(sizeof(typename Fragment::value_type) == 4);
  Tensor frag_64b = group_modes<1, 3>(
      recast<uint2>(frag));  // ((1, 2, N / 8), (MMA_M, MMA_N))
#pragma unroll
  for (int mi = 0; mi < size<1>(frag_64b); ++mi) {
#pragma unroll
    for (int i = 0; i < size<0, 2>(frag_64b) / 2; ++i) {
      cutlass::swap(frag_64b(make_coord(_0{}, _1{}, 2 * i), mi),
                    frag_64b(make_coord(_0{}, _0{}, 2 * i + 1), mi));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout, typename EngineOut>
CUTLASS_DEVICE void convert_type_out(Tensor<Engine, Layout> const& tensor,
                                     Tensor<EngineOut, Layout>& out) {
  // Somehow if we allocate out inside this function and return it, e2e is
  // slower and the output can be wrong.
  using From_type = typename Engine::value_type;
  using To_type = typename EngineOut::value_type;
  static constexpr int FragmentSize = std::max(
      sizeof(From_type) / sizeof(To_type), sizeof(To_type) / sizeof(From_type));
  static_assert(CUTE_STATIC_V(size(tensor)) % FragmentSize == 0,
                "Fragment size does not vectorize properly");
  Tensor frag = recast<cutlass::Array<From_type, FragmentSize> const>(tensor);
  Tensor out_frg = recast<cutlass::Array<To_type, FragmentSize>>(out);
  static_assert(size(frag) == size(out_frg));
  cutlass::NumericArrayConverter<To_type, From_type, FragmentSize> convert_op;
#pragma unroll
  for (int i = 0; i < size(frag); ++i) {
    out_frg[i] = convert_op(frag[i]);
  }
}

}  // namespace flash
