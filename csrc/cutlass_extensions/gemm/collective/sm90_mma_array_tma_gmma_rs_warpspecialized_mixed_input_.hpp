/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/arch/cluster_sm90.hpp"
#include "cute/arch/copy_sm90.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cutlass/cuda_host_adapter.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/pipeline/pipeline.hpp"
#include "cutlass/trace.h"
#include "cutlass_extensions/detail/collective/mixed_input_utils.hpp"

#define GROUP_SIZE 128

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective {
using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

// WarpSpecialized Mainloop
template <int Stages, class ClusterShape, class KernelSchedule_,
          class TileShape_, class ElementAOptionalTuple, class StrideA_,
          class ElementBOptionalTuple, class StrideB_, class TiledMma_,
          class GmemTiledCopyA_, class SmemLayoutAtomA_, class SmemCopyAtomA_,
          class TransformA_, class GmemTiledCopyB_, class SmemLayoutAtomB_,
          class SmemCopyAtomB_, class TransformB_>
struct CollectiveMmaArrayMixedInput<
    MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape,
                                                      KernelSchedule_>,
    TileShape_, ElementAOptionalTuple, StrideA_, ElementBOptionalTuple,
    StrideB_, TiledMma_, GmemTiledCopyA_, SmemLayoutAtomA_, SmemCopyAtomA_,
    TransformA_, GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_,
    TransformB_> {
 public:
  enum class ConversionMode {
    DirectConvert,
    ConvertAndScale,
    ConvertAndScaleWithZero
  };

  //
  // Type Aliases
  //
  using DispatchPolicy =
      MainloopSm90ArrayTmaGmmaWarpSpecializedMixedInput<Stages, ClusterShape,
                                                        KernelSchedule_>;
  using TileShape = TileShape_;
  using KernelSchedule = KernelSchedule_;

 private:
  template <class T>
  friend struct detail::MixedGroupedGemmInputUtils;
  using CollectiveType =
      CollectiveMma<DispatchPolicy, TileShape_, ElementAOptionalTuple, StrideA_,
                    ElementBOptionalTuple, StrideB_, TiledMma_, GmemTiledCopyA_,
                    SmemLayoutAtomA_, SmemCopyAtomA_, TransformA_,
                    GmemTiledCopyB_, SmemLayoutAtomB_, SmemCopyAtomB_,
                    TransformB_>;
  using Utils = detail::MixedGroupedGemmInputUtils<CollectiveType>;

  //
  // Type Aliases
  //
  using ScaleA = detail::deduce_mixed_width_dtype_t<1, ElementAOptionalTuple>;
  using ScaleB = detail::deduce_mixed_width_dtype_t<1, ElementBOptionalTuple>;
  using ZeroA = detail::deduce_mixed_width_dtype_t<2, ElementAOptionalTuple>;
  using ZeroB = detail::deduce_mixed_width_dtype_t<2, ElementBOptionalTuple>;

 public:
  static_assert(cute::is_tuple<ElementAOptionalTuple>::value ^
                    cute::is_tuple<ElementBOptionalTuple>::value,
                "Either A OR B must be a tuple. It must take the from "
                "{ElementOperand, [ElementScale], [ElementZero]}. Inputs in "
                "[] are optional.");

  using ElementA = detail::deduce_mixed_width_dtype_t<0, ElementAOptionalTuple>;
  using ElementB = detail::deduce_mixed_width_dtype_t<0, ElementBOptionalTuple>;
  static constexpr bool IsATransformed =
      cute::is_tuple<ElementAOptionalTuple>::value;
  using ElementScale = cute::conditional_t<IsATransformed, ScaleA, ScaleB>;
  using ElementZero = cute::conditional_t<IsATransformed, ZeroA, ZeroB>;
  // For cases where we can't have a void type, we can use this to allow the
  // code to compile when the scale / zero is void.
  using NonVoidElementScale =
      cute::conditional_t<cute::is_void_v<ElementScale>, float, ElementScale>;
  using NonVoidElementZero =
      cute::conditional_t<cute::is_void_v<ElementZero>, float, ElementZero>;

  using StrideA = StrideA_;
  using InternalStrideA = cute::remove_pointer_t<StrideA>;
  using StrideB = StrideB_;
  using InternalStrideB = cute::remove_pointer_t<StrideB>;

  using StrideScale = cute::Stride<cute::Int<1>, int64_t, int64_t>;
  using NonVoidStrideScale =
      cute::conditional_t<cute::is_void_v<StrideScale>,
                          cute::Stride<_1, int64_t, int64_t>, StrideScale>;

  static_assert(
      (IsATransformed &&
       (cutlass::gemm::detail::is_k_major<StrideA>() ||
        is_layout<StrideA>::value || is_layout<InternalStrideA>::value)) ||
          (!IsATransformed &&
           (cutlass::gemm::detail::is_k_major<StrideB>() ||
            is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),
      "The transformed type must be K-major.");

  static_assert(
      (IsATransformed && (sizeof(ElementB) == 2)) ||
          (!IsATransformed && (sizeof(ElementA) == 2)) ||
          ((cutlass::gemm::detail::is_k_major<StrideA>() ||
            is_layout<StrideA>::value || is_layout<InternalStrideA>::value) &&
           (cutlass::gemm::detail::is_k_major<StrideB>() ||
            is_layout<StrideB>::value || is_layout<InternalStrideB>::value)),
      "The unscaled element must be 2 bytes OR both inputs must be K-major");

  static_assert(cutlass::gemm::detail::is_mn_major<NonVoidStrideScale>(),
                "Scale must be MN major [Col Major if A is scaled, Row Major "
                "if B is scaled].");

  using CtaShape_MNK = decltype(shape_div(TileShape{}, ClusterShape{}));
  using TiledMma = TiledMma_;
  using ElementAccumulator = typename TiledMma::ValTypeC;
  using GmemTiledCopyA = GmemTiledCopyA_;
  using GmemTiledCopyB = GmemTiledCopyB_;
  using GmemTiledCopyScale = cute::SM90_TMA_LOAD;
  using SmemLayoutAtomA = SmemLayoutAtomA_;
  using SmemLayoutAtomB = SmemLayoutAtomB_;
  using SmemCopyAtomA = SmemCopyAtomA_;
  using SmemCopyAtomB = SmemCopyAtomB_;
  using SmemCopyAtomScale =
      Copy_Atom<cute::AutoVectorizingCopy, NonVoidElementScale>;

  // We must ensure the type to be scaled goes to RF
  static constexpr bool SwapAB = !IsATransformed;
  using SwappedStrideA = cute::conditional_t<!SwapAB, StrideA, StrideB>;
  using SwappedStrideB = cute::conditional_t<!SwapAB, StrideB, StrideA>;
  using InternalSwappedStrideA =
      cute::conditional_t<!SwapAB, InternalStrideA, InternalStrideB>;
  using InternalSwappedStrideB =
      cute::conditional_t<!SwapAB, InternalStrideB, InternalStrideA>;
  using SwappedSmemLayoutAtomA =
      cute::conditional_t<!SwapAB, SmemLayoutAtomA, SmemLayoutAtomB>;
  using SwappedSmemLayoutAtomB =
      cute::conditional_t<!SwapAB, SmemLayoutAtomB, SmemLayoutAtomA>;
  using SwappedSmemCopyAtomA =
      cute::conditional_t<!SwapAB, SmemCopyAtomA, SmemCopyAtomB>;
  using SwappedSmemCopyAtomB =
      cute::conditional_t<!SwapAB, SmemCopyAtomB, SmemCopyAtomA>;
  // TMA converts f32 input to tf32 when copying from GMEM to SMEM
  // For all other types, cast to size equivalent uint type to avoid any
  // rounding by TMA.
  static constexpr bool ConvertF32toTF32A = cute::is_same_v<float, ElementA>;
  static constexpr bool ConvertF32toTF32B = cute::is_same_v<float, ElementB>;
  using ConvertedElementA =
      cute::conditional_t<ConvertF32toTF32A, tfloat32_t,
                          uint_bit_t<sizeof_bits_v<ElementA>>>;
  using ConvertedElementB =
      cute::conditional_t<ConvertF32toTF32B, tfloat32_t,
                          uint_bit_t<sizeof_bits_v<ElementB>>>;
  using RealSwappedElementA = cute::conditional_t<!SwapAB, ElementA, ElementB>;
  using RealSwappedElementB = cute::conditional_t<!SwapAB, ElementB, ElementA>;
  using SwappedElementA =
      cute::conditional_t<!SwapAB, ConvertedElementA, ConvertedElementB>;
  using SwappedElementB =
      cute::conditional_t<!SwapAB, ConvertedElementB, ConvertedElementA>;

  using TransformA = TransformA_;
  using TransformB = TransformB_;
  using SwappedTransformA =
      cute::conditional_t<!SwapAB, TransformA, TransformB>;
  using SwappedTransformB =
      cute::conditional_t<!SwapAB, TransformB, TransformA>;
  using ArchTag = typename DispatchPolicy::ArchTag;

  static constexpr int IsSubbyteA = cute::sizeof_bits_v<SwappedElementA> < 8;
  using TmaElementA = cute::conditional_t<IsSubbyteA, uint8_t, SwappedElementA>;
  using TmaElementScale = uint_bit_t<sizeof_bits_v<
      NonVoidElementScale>>;  // in case we have array. translating to uint
                              // to satisfy tma descriptor's specialization

  using MainloopPipeline = cutlass::PipelineTmaAsync<DispatchPolicy::Stages>;
  using PipelineState = cutlass::PipelineState<DispatchPolicy::Stages>;
  using PipelineParams = typename MainloopPipeline::Params;

  static constexpr int NumProducerThreadEvents = 1;

  using SmemLayoutAtomScale = Layout<
      Shape<decltype(cute::shape<0>(SwappedSmemLayoutAtomA{})), cute::Int<1>>>;
  using ScaleTileShape = decltype(make_shape(shape<0>(TileShape{}),
                                             shape<1>(SmemLayoutAtomScale{})));

  static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2,
                "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<0>(TileShape{}) % size<0>(SwappedSmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomA{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2,
                "SmemLayoutAtom must be rank 2 (M/N, K)");
  static_assert((size<1>(TileShape{}) % size<0>(SwappedSmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SwappedSmemLayoutAtomB{})) == 0,
                "SmemLayoutAtom must evenly divide tile shape.");

  static_assert(rank(SmemLayoutAtomScale{}) == 2,
                "SmemLayoutAtomScale must be rank 2");
  static_assert((size<0>(TileShape{}) % size<0>(SmemLayoutAtomScale{})) == 0,
                "SmemLayoutAtomScale must equal the tile shape.");
  static_assert((size<2>(TileShape{}) % size<1>(SmemLayoutAtomScale{})) == 0,
                "SmemLayoutAtomScale must evenly divide tile k shape.");

  /// Tile along modes in a way that maximizes the TMA box size.
  using SmemLayoutA = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(
      SwappedSmemLayoutAtomA{}, select<0, 2>(TileShape{}),
      InternalSwappedStrideA{}));
  using SmemLayoutB = decltype(detail::get_smem_layout<DispatchPolicy::Stages>(
      SwappedSmemLayoutAtomB{}, select<1, 2>(TileShape{}),
      InternalSwappedStrideB{}));

  // It is assumed that the scales and zero-points share the same smem layout
  using SmemLayoutScale = decltype(tile_to_shape(
      SmemLayoutAtomScale{},
      make_shape(shape<0>(ScaleTileShape{}), shape<1>(ScaleTileShape{}),
                 Int<Stages>{}),
      cute::conditional_t<
          ::cutlass::gemm::detail::is_major<0, NonVoidStrideScale>(),
          Step<_2, _1, _3>, Step<_1, _2, _3>>{}));

  static_assert(DispatchPolicy::Stages >= 2,
                "Specialization requires Stages set to value 2 or more.");
  static_assert(not cute::is_base_of<cute::GMMA::DescriptorIterator,
                                     typename TiledMma::FrgTypeA>::value &&
                    cute::is_base_of<cute::GMMA::DescriptorIterator,
                                     typename TiledMma::FrgTypeB>::value,
                "MMA atom must source A from rmem and B operand from smem_desc "
                "for this mainloop.");
  static_assert(cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD> ||
                    cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>,
                "GmemTiledCopy - invalid SM90 TMA copy atom specified.");
  static_assert(cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD> ||
                    cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>,
                "GmemTiledCopy - invalid SM90 TMA copy atom specified.");

  // To relax them, we need to handle loading more than 1 row of scales for
  // every main loop iteration. We must also handle updating the pipeline
  // transaction bytes on the fly.
  static_assert(size<1>(SmemLayoutAtomScale{}) == 1,
                "size<1>(SmemLayoutAtomScale) must be 1.");

 private:
  static constexpr ConversionMode get_conversion_mode() {
    if constexpr (cute::is_void_v<ElementScale>) {
      return ConversionMode::DirectConvert;
    } else if constexpr (cute::is_void_v<ElementZero>) {
      return ConversionMode::ConvertAndScale;
    } else {
      return ConversionMode::ConvertAndScaleWithZero;
    }
  }

 public:
  static constexpr ConversionMode KernelConversionMode = get_conversion_mode();
  static constexpr bool ModeHasScales =
      KernelConversionMode == ConversionMode::ConvertAndScale ||
      KernelConversionMode == ConversionMode::ConvertAndScaleWithZero;
  static constexpr bool UseScaleLookupTable =
      KernelConversionMode == ConversionMode::ConvertAndScale &&
      cutlass::detail::is_Array_v<ElementScale>;
  static constexpr size_t SmemAlignmentA =
      cutlass::detail::alignment_for_swizzle(SmemLayoutA{});
  static constexpr size_t SmemAlignmentB =
      cutlass::detail::alignment_for_swizzle(SmemLayoutB{});
  static constexpr size_t SmemAlignmentScale =
      cute::max(SmemAlignmentA, SmemAlignmentB);

  static_assert(SmemAlignmentA >= 128 and SmemAlignmentB >= 128,
                "Require at least 128B alignment");

  struct SharedStorage {
    static constexpr int scale_elements = Utils::elements_per_smem_scale();
    static constexpr int zero_elements = Utils::elements_per_smem_zero();
    struct TensorStorage {
      CUTE_ALIGNAS(SmemAlignmentA)
      cute::ArrayEngine<RealSwappedElementA, cute::cosize_v<SmemLayoutA>>
          smem_A;
      CUTE_ALIGNAS(SmemAlignmentB)
      cute::ArrayEngine<typename TiledMma::ValTypeB,
                        cute::cosize_v<SmemLayoutB>>
          smem_B;
      cute::ArrayEngine<NonVoidElementScale, scale_elements> smem_scale;
      cute::ArrayEngine<NonVoidElementZero, zero_elements> smem_zero;
    } tensors;

    struct TensorMapStorage {
      cute::TmaDescriptor smem_tensormap_A;
      cute::TmaDescriptor smem_tensormap_B;
      cute::TmaDescriptor smem_tensormap_scale;
      cute::TmaDescriptor smem_tensormap_zero;
    };

    using PipelineStorage = typename MainloopPipeline::SharedStorage;
    PipelineStorage pipeline;
  };
  using TensorStorage = typename SharedStorage::TensorStorage;
  using TensorMapStorage = typename SharedStorage::TensorMapStorage;
  using PipelineStorage = typename SharedStorage::PipelineStorage;

  static constexpr bool IsGroupedGemmKernel =
      !cute::is_same_v<InternalStrideA, StrideA>;

  // kernel Arguments
  // Host side kernel arguments
  struct Arguments {
    ElementA const** ptr_A;
    StrideA dA;
    ElementB const** ptr_B;
    StrideB dB;
    ElementScale const** ptr_S = nullptr;
    NonVoidStrideScale const* dS{};
    int chunk_size = 0;
    ElementZero const** ptr_Z = nullptr;
  };

  // Device side kernel params
  struct Params {
    // Assumption: StrideA is congruent with Problem_MK
    using LayoutA = decltype(detail::get_gmem_layout(
        repeat_like(InternalSwappedStrideA{}, int32_t(0)),
        InternalSwappedStrideA{}));
    using LayoutB = decltype(detail::get_gmem_layout(
        repeat_like(InternalSwappedStrideB{}, int32_t(0)),
        InternalSwappedStrideB{}));

    using TMA_A = decltype(make_tma_copy<TmaElementA>(
        GmemTiledCopyA{},
        make_tensor(detail::get_logical_ptr(
                        static_cast<SwappedElementA const*>(nullptr)),
                    LayoutA{}),
        SmemLayoutA{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(
            ClusterShape{})));  // mcast along N mode for this M load, if any
    // Assumption: StrideB is congruent with Problem_NK
    using TMA_B = decltype(make_tma_copy(
        GmemTiledCopyB{},
        make_tensor(detail::get_logical_ptr(
                        static_cast<SwappedElementB const*>(nullptr)),
                    LayoutB{}),
        SmemLayoutB{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(
            ClusterShape{})));  // mcast along M mode for this N load, if any

    using TMA_Scale = decltype(make_tma_copy<TmaElementScale>(
        GmemTiledCopyScale{},
        make_tensor(detail::get_logical_ptr(
                        static_cast<NonVoidElementScale const*>(nullptr)),
                    repeat_like(NonVoidStrideScale{}, int32_t(0)),
                    NonVoidStrideScale{}),
        SmemLayoutScale{}(_, _, cute::Int<0>{}), ScaleTileShape{},
        _1{}));  // mcast along N mode for this M load, if any. Scale is ALWAYS
                 // loaded with A for RF kernel

    using TMA_Zero = decltype(make_tma_copy(
        GmemTiledCopyScale{},
        make_tensor(detail::get_logical_ptr(
                        static_cast<NonVoidElementZero const*>(nullptr)),
                    repeat_like(NonVoidStrideScale{}, int32_t(0)),
                    NonVoidStrideScale{}),
        SmemLayoutScale{}(_, _, cute::Int<0>{}), ScaleTileShape{},
        _1{}));  // mcast along N mode for this M load, if any. Scale is ALWAYS
                 // loaded with A for RF kernel

    TMA_A tma_load_a;
    TMA_B tma_load_b;
    uint32_t tma_transaction_bytes = TmaTransactionBytes;
    TMA_Scale tma_load_scale;
    TMA_Zero tma_load_zero;
    void* tensormaps;
    SwappedElementA const** ptr_A;
    SwappedStrideA ptr_dA;
    SwappedElementB const** ptr_B;
    SwappedStrideB ptr_dB;
    NonVoidElementScale const** ptr_S;
    NonVoidStrideScale const* dS;
    NonVoidElementZero const** ptr_Z;
    int64_t scale_k;
    int chunk_size;
    int reload_factor =
        (chunk_size + size<2>(TileShape{}) - 1) / size<2>(TileShape{});
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;
  };

  //
  // Methods
  //

  template <class ProblemShape>
  static constexpr Params to_underlying_arguments(ProblemShape problem_shapes,
                                                  Arguments const& args,
                                                  void* workspace) {
    // These tensor shapes (only applicable for grouped gemm) and pointers are
    // only used to create tensormap/tma desc. These will be replaced with
    // correct values before the initial tma load.
    auto init_shape = repeat_like(
        typename ProblemShape::UnderlyingProblemShape{}, int32_t(1));
    auto init_M = get<0>(init_shape);
    auto init_N = get<1>(init_shape);
    auto init_K = get<2>(init_shape);

    if constexpr (SwapAB) {
      init_M = get<1>(init_shape);
      init_N = get<0>(init_shape);
    }
    // Batches/Groups are managed by using appropriate pointers to input
    // matrices
    const uint32_t mock_L = 1;
    SwappedElementA const* ptr_A_first_batch;
    SwappedElementB const* ptr_B_first_batch;
    SwappedStrideA ptr_dA;
    SwappedStrideB ptr_dB;
    InternalSwappedStrideA dA;
    InternalSwappedStrideB dB;

    if constexpr (not SwapAB) {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_A);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_B);
    } else {
      ptr_A_first_batch = reinterpret_cast<SwappedElementA const*>(args.ptr_B);
      ptr_B_first_batch = reinterpret_cast<SwappedElementB const*>(args.ptr_A);
    }

    if constexpr (IsGroupedGemmKernel) {
      // Strides for Grouped Gemm will be replaced prior to the first access
      // regardless.
      if constexpr (not SwapAB) {
        ptr_dA = args.dA;
        ptr_dB = args.dB;
      } else {
        ptr_dA = args.dB;
        ptr_dB = args.dA;
      }
      dA = InternalSwappedStrideA{};
      if constexpr (is_layout<InternalSwappedStrideA>::value) {
        dA = make_layout(
            transform_leaf(dA.shape(),
                           [](auto x) {
                             if constexpr (not is_static_v<decltype(x)>) {
                               return static_cast<decltype(x)>(1);
                             } else {
                               return x;
                             }
                           }),
            dA.stride());
      }
      dB = InternalSwappedStrideB{};
    } else {
      // Tensor shapes for Ptr-Array are initialized correctly only here.
      auto problem_shape_MNK = problem_shapes.get_host_problem_shape(0);
      init_M = get<0>(problem_shape_MNK);
      init_N = get<1>(problem_shape_MNK);
      init_K = get<2>(problem_shape_MNK);

      if constexpr (not SwapAB) {
        dA = args.dA;
        dB = args.dB;
      } else {
        dA = args.dB;
        dB = args.dA;
      }
      ptr_dA = SwappedStrideA{};
      ptr_dB = SwappedStrideB{};
    }
    Tensor tensor_a = make_tensor(
        ptr_A_first_batch,
        detail::get_gmem_layout(make_shape(init_M, init_K, mock_L), dA));
    Tensor tensor_b = make_tensor(
        ptr_B_first_batch,
        detail::get_gmem_layout(make_shape(init_N, init_K, mock_L), dB));

    typename Params::TMA_A tma_load_a = make_tma_copy<TmaElementA>(
        GmemTiledCopyA{}, tensor_a, SmemLayoutA{}(_, _, cute::Int<0>{}),
        make_shape(shape<0>(TileShape{}), shape<2>(TileShape{})),
        size<1>(ClusterShape{}));  // mcast along N mode for this M load, if any
    typename Params::TMA_B tma_load_b = make_tma_copy(
        GmemTiledCopyB{}, tensor_b, SmemLayoutB{}(_, _, cute::Int<0>{}),
        make_shape(shape<1>(TileShape{}), shape<2>(TileShape{})),
        size<0>(ClusterShape{}));  // mcast along M mode for this N load, if any
    typename Params::TMA_Scale tma_load_scale{};
    typename Params::TMA_Zero tma_load_zero{};

    void* tensormaps = workspace;
    auto args_setup = [&](auto ptr_A, auto ptr_B, int64_t scale_k = 0,
                          int chunk_size = 0, int reload_factor = 1) -> Params {
      return {tma_load_a,
              tma_load_b,
              TmaTransactionBytes,
              tma_load_scale,
              tma_load_zero,
              tensormaps,
              reinterpret_cast<SwappedElementA const**>(ptr_A),
              ptr_dA,
              reinterpret_cast<SwappedElementB const**>(ptr_B),
              ptr_dB,
              reinterpret_cast<NonVoidElementScale const**>(args.ptr_S),
              args.dS,
              reinterpret_cast<NonVoidElementZero const**>(args.ptr_Z),
              scale_k,
              chunk_size,
              reload_factor,
              dA,
              dB};
    };

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return SwapAB ? args_setup(args.ptr_B, args.ptr_A)
                    : args_setup(args.ptr_A, args.ptr_B);
    } else if constexpr (ModeHasScales) {
      auto fake_scale_k = 1;
      ElementScale const* ptr_S =
          reinterpret_cast<ElementScale const*>(args.ptr_S);
      StrideScale dS{};
      Tensor tensor_scale = make_tensor(
          detail::get_logical_ptr(ptr_S),
          make_layout(make_shape(init_M, fake_scale_k, mock_L), dS));
      tma_load_scale = make_tma_copy<TmaElementScale>(
          GmemTiledCopyScale{}, tensor_scale,
          SmemLayoutScale{}(_, _, cute::Int<0>{}), ScaleTileShape{},
          _1{});  // mcast along N mode for this M load, if any

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return SwapAB
                   ? args_setup(args.ptr_B, args.ptr_A, fake_scale_k,
                                args.chunk_size,
                                (args.chunk_size + size<2>(TileShape{}) - 1) /
                                    size<2>(TileShape{}))
                   : args_setup(args.ptr_A, args.ptr_B, fake_scale_k,
                                args.chunk_size,
                                (args.chunk_size + size<2>(TileShape{}) - 1) /
                                    size<2>(TileShape{}));
      } else if constexpr (KernelConversionMode ==
                           ConversionMode::ConvertAndScaleWithZero) {
        ElementZero const* ptr_Z =
            reinterpret_cast<ElementZero const*>(args.ptr_Z);
        Tensor tensor_zero = make_tensor(
            detail::get_logical_ptr(ptr_Z),
            make_layout(make_shape(init_M, fake_scale_k, mock_L), dS));
        tma_load_zero = make_tma_copy(
            GmemTiledCopyScale{}, tensor_zero,
            SmemLayoutScale{}(_, _, cute::Int<0>{}), ScaleTileShape{},
            _1{});  // mcast along N mode for this M load, if any
        return SwapAB
                   ? args_setup(args.ptr_B, args.ptr_A, fake_scale_k,
                                args.chunk_size,
                                (args.chunk_size + size<2>(TileShape{}) - 1) /
                                    size<2>(TileShape{}))
                   : args_setup(args.ptr_A, args.ptr_B, fake_scale_k,
                                args.chunk_size,
                                (args.chunk_size + size<2>(TileShape{}) - 1) /
                                    size<2>(TileShape{}));

      } else {
        static_assert(
            cutlass::detail::dependent_false<KernelSchedule>,
            "Conversion mode not handled in to_underlying_arguments.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in to_underlying_arguments.");
    }
  }

  template <class ProblemShape>
  static size_t get_workspace_size(ProblemShape const& problem_shape,
                                   Arguments const& args, int sm_count) {
    constexpr size_t SizeOfCuTensorMap = sizeof(cute::TmaDescriptor);

    // Calculating workspace size
    auto calculate_workspace_size = [SizeOfCuTensorMap,
                                     sm_count](uint32_t num_input_tensors) {
      return num_input_tensors * SizeOfCuTensorMap * sm_count;
    };

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // Allocate gmem space for input tensormaps per each SM, A tensormap
      // copies followed by B tensormap copies
      return calculate_workspace_size(2);
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScale) {
      // Allocate gmem space for input tensormaps per each SM, A tensormap
      // copies followed by B tensormap copies, followed by scale tensormap
      // copies
      return calculate_workspace_size(3);
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      // Allocate gmem space for input tensormaps per each SM, A tensormap
      // copies followed by B tensormap copies, followed by scale and zeros
      // tensormap copies
      return calculate_workspace_size(4);
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in get_workspace_size.");
    }
  }

  template <class ProblemShape>
  static cutlass::Status initialize_workspace(
      ProblemShape const& problem_shape, Arguments const& args, void* workspace,
      cudaStream_t stream, CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  template <class ProblemShape>
  CUTLASS_HOST_DEVICE static bool can_implement(ProblemShape problem_shapes,
                                                Arguments const& args) {
    constexpr int tma_alignment_bits = 128;
    constexpr int min_tma_aligned_elements_A =
        tma_alignment_bits / cutlass::sizeof_bits<ElementA>::value;
    constexpr int min_tma_aligned_elements_B =
        tma_alignment_bits / cutlass::sizeof_bits<ElementB>::value;

    bool implementable = true;
    if (problem_shapes.is_host_problem_shape_available()) {
      // Check alignment for all problem sizes
      for (int i = 0; i < problem_shapes.groups(); i++) {
        auto problem_shape_MNKL =
            append<4>(problem_shapes.get_host_problem_shape(i), 1);
        auto [M, N, K, L] = problem_shape_MNKL;
        auto get_stride = [](auto stride) {
          if constexpr (cute::is_pointer_v<cute::decay_t<decltype(stride)>>) {
            return *stride;
          } else {
            return stride;
          }
        };
        auto dA = get_stride(args.dA);
        auto dB = get_stride(args.dB);
        implementable =
            implementable &&
            cutlass::detail::check_alignment<min_tma_aligned_elements_A>(
                detail::get_gmem_layout(cute::make_shape(M, K, L), dA));
        implementable =
            implementable &&
            cutlass::detail::check_alignment<min_tma_aligned_elements_B>(
                detail::get_gmem_layout(cute::make_shape(N, K, L), dB));
        if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
          implementable = implementable && (args.ptr_S == nullptr);
          implementable = implementable && (args.ptr_Z == nullptr);
        } else if constexpr (ModeHasScales) {
          const int scale_mn = SwapAB ? N : M;
          const int scale_k = (K + args.chunk_size - 1) / args.chunk_size;
          constexpr int min_tma_aligned_elements_scale =
              tma_alignment_bits / cutlass::sizeof_bits<ElementScale>::value;
          implementable =
              implementable &&
              cutlass::detail::check_alignment<min_tma_aligned_elements_scale>(
                  cute::make_shape(scale_mn, scale_k, L), StrideScale{});
          implementable = implementable &&
                          (args.chunk_size == K ||
                           ((args.chunk_size % size<2>(TileShape{})) == 0));
          implementable = implementable && args.chunk_size != 0;
          implementable = implementable && (args.ptr_S != nullptr);
          if constexpr (KernelConversionMode ==
                        ConversionMode::ConvertAndScale) {
            implementable = implementable && (args.ptr_Z == nullptr);
          } else if constexpr (KernelConversionMode ==
                               ConversionMode::ConvertAndScaleWithZero) {
            constexpr int min_tma_aligned_elements_zero =
                tma_alignment_bits / cutlass::sizeof_bits<ElementZero>::value;
            implementable =
                implementable &&
                cutlass::detail::check_alignment<min_tma_aligned_elements_zero>(
                    cute::make_shape(scale_mn, scale_k, L), StrideScale{});
            implementable = implementable && (args.ptr_Z != nullptr);
          } else {
            static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                          "Conversion mode not handled in can_implement.");
          }
        } else {
          static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                        "Conversion mode not handled in can_implement.");
        }
      }
    }

    if (!implementable) {
      CUTLASS_TRACE_HOST(
          "  CAN IMPLEMENT: Problem Size doesn't meet the minimum alignment "
          "requirements for TMA.\n");
    }
    return implementable;
  }

  static constexpr int K_PIPE_MAX = DispatchPolicy::Stages;
  static constexpr int K_PIPE_MMAS = 1;
  static constexpr uint32_t TmaTransactionBytesMK =
      Utils::compute_tma_transaction_bytes_mk();
  static constexpr uint32_t TmaTransactionBytesNK =
      Utils::compute_tma_transaction_bytes_nk();
  static constexpr uint32_t TmaTransactionBytesExtra =
      Utils::compute_tma_transaction_bytes_extra();
  static constexpr uint32_t TmaTransactionBytes =
      TmaTransactionBytesMK + TmaTransactionBytesNK + TmaTransactionBytesExtra;

  // Set up the data needed by this collective for load and mma.
  // Returns a tuple of tensors. The collective and the kernel layer have the
  // contract that the returned tuple must contain at least two elements, with
  // the first two elements being: gA_mkl - The tma tensor, A after a local tile
  // so it has shape  (BLK_M,BLK_K,m,k,l) gB_nkl - The tma tensor, B after a
  // local tile so it has shape  (BLK_N,BLK_K,n,k,l) The rest of the tensors can
  // be specified as needed by this collective.
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE auto load_init(ProblemShape_MNKL const& problem_shape_MNKL,
                                Params const& mainloop_params) const {
    using X = Underscore;
    // Separate out problem shape for convenience
    auto [M, N, K, L] = problem_shape_MNKL;
    const int32_t mock_L = 1;

    // TMA requires special handling of strides to deal with coord codomain
    // mapping Represent the full tensors -- get these from TMA
    Tensor mA_mkl =
        mainloop_params.tma_load_a.get_tma_tensor(shape(detail::get_gmem_layout(
            make_shape(M, K, mock_L), mainloop_params.dA)));  // (m,k,l)
    Tensor mB_nkl =
        mainloop_params.tma_load_b.get_tma_tensor(shape(detail::get_gmem_layout(
            make_shape(N, K, mock_L), mainloop_params.dB)));  // (n,k,l)

    // Make tiled views, defer the slice
    Tensor gA_mkl = local_tile(mA_mkl, TileShape{}, make_coord(_, _, _),
                               Step<_1, X, _1>{});  // (BLK_M,BLK_K,m,k,l)
    Tensor gB_nkl = local_tile(mB_nkl, TileShape{}, make_coord(_, _, _),
                               Step<X, _1, _1>{});  // (BLK_N,BLK_K,n,k,l)

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(gA_mkl, gB_nkl);
    } else if constexpr (ModeHasScales) {
      // The real scale_k that actually works
      // auto scale_k = K / mainloop_params.chunk_size;
      auto scale_k = K / GROUP_SIZE;

      Tensor mS_mkl = mainloop_params.tma_load_scale.get_tma_tensor(
          make_shape(M, scale_k, L));  // (m,scale_k,l)
      Tensor gS_mkl =
          local_tile(mS_mkl, ScaleTileShape{},
                     make_coord(_, _));  // (BLK_M,BLK_Scale_K,m,scale_k,l)
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(gA_mkl, gB_nkl, gS_mkl);
      } else if constexpr (KernelConversionMode ==
                           ConversionMode::ConvertAndScaleWithZero) {
        Tensor mZ_mkl = mainloop_params.tma_load_zero.get_tma_tensor(
            make_shape(M, scale_k, L));  // (m,scale_k,l)
        Tensor gZ_mkl =
            local_tile(mZ_mkl, ScaleTileShape{},
                       make_coord(_, _));  // (BLK_M,BLK_Scale_K,m,scale_k,l)
        return cute::make_tuple(gA_mkl, gB_nkl, gS_mkl, gZ_mkl);
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled in load_init.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in load_init.");
    }
  }

  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a collective-scoped matrix multiply-accumulate
  // Producer Perspective
  template <class... Ts, class... TMs, class KTileIterator, class BlockCoord>
  CUTLASS_DEVICE void load(
      Params const& mainloop_params, MainloopPipeline pipeline,
      PipelineState smem_pipe_write, cute::tuple<Ts...> const& load_inputs,
      cute::tuple<TMs...> const& input_tensormaps, BlockCoord const& blk_coord,
      KTileIterator k_tile_iter, int k_tile_count, int thread_idx,
      uint32_t block_rank_in_cluster, TensorStorage& shared_tensors) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      static_assert(sizeof...(Ts) == 2, "Direct convert needs two inputs");
      static_assert(sizeof...(TMs) == 2, "Direct convert needs two tensormaps");
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScale) {
      static_assert(sizeof...(Ts) == 3, "Scaled convert needs three inputs");
      static_assert(sizeof...(TMs) == 3,
                    "Scaled convert needs three tensormaps");
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      static_assert(sizeof...(Ts) == 4,
                    "Scaled and zero convert needs four inputs");
      static_assert(sizeof...(TMs) == 4,
                    "Scaled and zero convert needs four tensormaps");
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in TMA load.");
    }

    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                             SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sB_ = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                             SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)
    Tensor sA =
        as_position_independent_swizzle_tensor(sA_);  // (BLK_M,BLK_K,PIPE)
    Tensor sB =
        as_position_independent_swizzle_tensor(sB_);  // (BLK_N,BLK_K,PIPE)

    //
    // Prepare the TMA loads for A and B
    //

    constexpr uint32_t cluster_shape_x =
        get<0>(typename DispatchPolicy::ClusterShape());
    uint2 cluster_local_block_id = {block_rank_in_cluster % cluster_shape_x,
                                    block_rank_in_cluster / cluster_shape_x};

    Tensor gA_mkl = get<0>(load_inputs);
    Tensor gB_nkl = get<1>(load_inputs);

    auto block_tma_a =
        mainloop_params.tma_load_a.get_slice(cluster_local_block_id.y);
    auto block_tma_b =
        mainloop_params.tma_load_b.get_slice(cluster_local_block_id.x);

    // Partition the inputs based on the current block coordinates.
    auto [m_coord, n_coord, k_coord, l_coord] = blk_coord;
    Tensor gA = gA_mkl(_, _, m_coord, _, l_coord);  // (BLK_M,BLK_K,k)
    Tensor gB = gB_nkl(_, _, n_coord, _, l_coord);  // (BLK_N,BLK_K,k)

    // Applies the mapping from block_tma_a
    Tensor tAgA = block_tma_a.partition_S(gA);  // (TMA,TMA_M,TMA_K,k)
    Tensor tAsA = block_tma_a.partition_D(sA);  // (TMA,TMA_M,TMA_K,PIPE)

    Tensor tBgB = block_tma_b.partition_S(gB);  // (TMA,TMA_N,TMA_K,k)
    Tensor tBsB = block_tma_b.partition_D(sB);  // (TMA,TMA_N,TMA_K,PIPE)

    uint16_t mcast_mask_a = 0;
    uint16_t mcast_mask_b = 0;
    uint16_t mcast_mask_s = 0;

    // Issue TmaLoads
    // Maps the tile -> block, value
    if constexpr (cute::is_same_v<GmemTiledCopyA, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout =
          Layout<typename DispatchPolicy::ClusterShape>{};  // (m,n) -> block_id
      for (int n = 0; n < size<1>(block_layout); ++n) {
        mcast_mask_a |= (uint16_t(1) << block_layout(cluster_local_block_id.x,
                                                     n, Int<0>{}));
      }
    }

    if constexpr (cute::is_same_v<GmemTiledCopyB, SM90_TMA_LOAD_MULTICAST>) {
      auto block_layout =
          Layout<typename DispatchPolicy::ClusterShape>{};  // (m,n) -> block_id
      for (int m = 0; m < size<0>(block_layout); ++m) {
        mcast_mask_b |= (uint16_t(1) << block_layout(
                             m, cluster_local_block_id.y, Int<0>{}));
      }
    }

    auto extra_input_partitions = Utils::partition_extra_tma_inputs(
        mainloop_params, load_inputs, shared_tensors, cluster_local_block_id,
        m_coord, l_coord);

    // Mainloop
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 0; --k_tile_count) {
      // LOCK smem_pipe_write for _writing_
      pipeline.producer_acquire(smem_pipe_write);

      //
      // Copy gmem to smem for *k_tile_iter
      //

      using BarrierType = typename MainloopPipeline::ProducerBarrierType;
      BarrierType* tma_barrier = pipeline.producer_get_barrier(smem_pipe_write);

      int write_stage = smem_pipe_write.index();
      if (cute::elect_one_sync()) {
        copy(mainloop_params.tma_load_a.with(get<0>(input_tensormaps),
                                             *tma_barrier, mcast_mask_a),
             tAgA(_, _, _, *k_tile_iter), tAsA(_, _, _, write_stage));
        copy(mainloop_params.tma_load_b.with(get<1>(input_tensormaps),
                                             *tma_barrier, mcast_mask_b),
             tBgB(_, _, _, *k_tile_iter), tBsB(_, _, _, write_stage));
      }
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // Nothing extra to do.
      } else if constexpr (ModeHasScales) {
        // scale copy
        auto tSgS = get<0>(extra_input_partitions);
        auto tSsS = get<1>(extra_input_partitions);

        // Temporary factor which will determine which k tile to reload from
        // gmem. Needed so we don't modify tma transaction bytes on the fly. We
        // must do a ceiling divide here to correctly handle with chunk_size ==
        // K. In that case, we don't require that K is a multiple of the
        // threadblock tile K
        const int scale_load_k = *k_tile_iter / 1;
        // const int scale_load_k = *k_tile_iter /
        // mainloop_params.reload_factor; // This will always be 0 when
        // chunk_size == K.
        if (cute::elect_one_sync()) {
          copy(mainloop_params.tma_load_scale.with(get<2>(input_tensormaps),
                                                   *tma_barrier, mcast_mask_s),
               tSgS(_, _, _, scale_load_k), tSsS(_, _, _, write_stage));
        }

        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
          // Nothing extra to do
        } else if constexpr (KernelConversionMode ==
                             ConversionMode::ConvertAndScaleWithZero) {
          // zero copy
          auto tZgZ = get<2>(extra_input_partitions);
          auto tZsZ = get<3>(extra_input_partitions);
          if (cute::elect_one_sync()) {
            copy(mainloop_params.tma_load_zero.with(get<3>(input_tensormaps),
                                                    *tma_barrier, mcast_mask_s),
                 tZgZ(_, _, _, scale_load_k), tZsZ(_, _, _, write_stage));
          }
        } else {
          static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                        "Conversion mode not handled for TMA copy op.");
        }
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled for TMA copy op.");
      }
      ++k_tile_iter;

      // Advance smem_pipe_write
      ++smem_pipe_write;
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // Perform a Producer Epilogue to prevent early exit of blocks in a Cluster
  CUTLASS_DEVICE void load_tail(MainloopPipeline pipeline,
                                PipelineState smem_pipe_write) {
    int lane_predicate = cute::elect_one_sync();

    // Issue the epilogue waits
    if (lane_predicate) {
      // This helps avoid early exit of blocks in Cluster.
      // Waits for all stages to either be released (all
      // Consumer UNLOCKs), or if the stage was never used
      // then it would just be acquired since the phase was
      // still inverted from make_producer_start_state.
      pipeline.producer_tail(smem_pipe_write);
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Perform a collective-scoped matrix multiply-accumulate
  /// Consumer Perspective
  template <class FrgTensorC>
  CUTLASS_DEVICE void mma(MainloopPipeline pipeline,
                          PipelineState smem_pipe_read, FrgTensorC& accum,
                          int k_tile_count, int thread_idx,
                          TensorStorage& shared_tensors,
                          Params const& mainloop_params) {
    static_assert(is_rmem<FrgTensorC>::value,
                  "C tensor must be rmem resident.");
    static_assert(cute::rank(SmemLayoutA{}) == 3,
                  "Smem layout must be rank 3.");
    static_assert(cute::rank(SmemLayoutB{}) == 3,
                  "Smem layout must be rank 3.");
    static_assert(cute::rank(SwappedSmemLayoutAtomA{}) == 2,
                  "SwappedSmemLayoutAtomA must be rank 2.");
    static_assert(cute::rank(SwappedSmemLayoutAtomB{}) == 2,
                  "SwappedSmemLayoutAtomB must be rank 2.");
    static_assert(!cute::is_void_v<SwappedSmemCopyAtomA>,
                  "SM90 GMMA mainloops must specify a non-void copy atom for "
                  "smem sourced instructions.");
    static_assert(cute::is_void_v<SwappedSmemCopyAtomB>,
                  "SM90 GMMA mainloops cannot have a non-void copy atom for "
                  "smem sourced instructions.");

    // Obtain warp index
    int warp_idx = canonical_warp_idx_sync();
    [[maybe_unused]] int warp_group_thread_idx = thread_idx % 128;

    Tensor sA_ = make_tensor(make_smem_ptr(shared_tensors.smem_A.begin()),
                             SmemLayoutA{});  // (BLK_M,BLK_K,PIPE)
    Tensor sA =
        as_position_independent_swizzle_tensor(sA_);  // (BLK_M,BLK_K,PIPE)

    Tensor sB = make_tensor(make_smem_ptr(shared_tensors.smem_B.begin()),
                            SmemLayoutB{});  // (BLK_N,BLK_K,PIPE)

    //
    // Define C accumulators and A/B partitioning
    //

    // Layout of warp group to thread mapping

    static_assert(
        stride<0>(typename TiledMma::BLayout{}) == 0 and
            size<0>(typename TiledMma::BLayout{}) == NumThreadsPerWarpGroup,
        "Stride of the first mode must be 0 and the size of the mode must be "
        "NumThreadsPerWarpGroup");

    constexpr int MmaWarpGroups = size(TiledMma{}) / NumThreadsPerWarpGroup;
    Layout warp_group_thread_layout =
        make_layout(Int<MmaWarpGroups>{}, Int<NumThreadsPerWarpGroup>{});

    int warp_group_idx =
        __shfl_sync(0xFFFFFFFF, thread_idx / NumThreadsPerWarpGroup, 0);

    TiledMma tiled_mma;
    auto mma_thread_slice = tiled_mma.get_thread_slice(thread_idx);
    Tensor tCsA = mma_thread_slice.partition_A(sA);
    auto mma_warpgroup_slice =
        tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx));

    // Allocate fragments and descriptors
    Tensor tCrA_mma = mma_thread_slice.partition_fragment_A(
        sA(_, _, Int<0>{}));  // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCrA_load = [&] {
      if constexpr (not is_layout<InternalSwappedStrideA>::value) {
        // Make register tensor with MMA layout
        return make_fragment_like<RealSwappedElementA>(tCrA_mma);
      } else {
        // Make register tensor matching smem layout, converter will take care
        // of de-swizzling
        return make_tensor_like<RealSwappedElementA>(tCsA(_, _, _, Int<0>{}));
      }
    }();
    Tensor tCsB =
        mma_warpgroup_slice.partition_B(sB);  // (MMA,MMA_N,MMA_K,PIPE)
    Tensor tCrB =
        mma_warpgroup_slice.make_fragment_B(tCsB);  // (MMA,MMA_N,MMA_K,PIPE)

    //
    // Copy Atom A retiling
    //
    auto smem_tiled_copy_A =
        make_tiled_copy_A(SwappedSmemCopyAtomA{}, tiled_mma);
    auto smem_thr_copy_A =
        smem_tiled_copy_A.get_thread_slice(warp_group_thread_idx);

    Tensor tCrA_copy_view =
        smem_thr_copy_A.retile_D(tCrA_load);  // (CPY,CPY_M,CPY_K)

    // Partition of thread -> shared and thread -> RF
    auto partitioned_extra_info =
        Utils::partition_extra_mma_info(mma_thread_slice, shared_tensors);
    auto copy_partitions_extra_info = Utils::retile_extra_mma_info(
        tiled_mma, partitioned_extra_info, warp_group_thread_idx);

    CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));  // CPY_M
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCrA_copy_view));  // CPY_K
    CUTE_STATIC_ASSERT_V(size<1>(tCrA_mma) == size<1>(accum));       // MMA_M
    CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<2>(accum));           // N
    CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));            // K
    CUTE_STATIC_ASSERT_V(size<3>(tCsA) == size<3>(tCsB));            // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sA));  // PIPE
    CUTE_STATIC_ASSERT_V(Int<DispatchPolicy::Stages>{} == size<2>(sB));  // PIPE

    //
    // PIPELINED MAIN LOOP
    //

    // We release buffers to producer warps(dma load) with some mmas in flight
    PipelineState smem_pipe_release = smem_pipe_read;

    multiply_add<ElementAccumulator> fma;

    constexpr int NumMMAsPerChunk =
        GROUP_SIZE / cute::get<0, 1>(tCsB.shape())();
    constexpr int NumChunksPerTileK = cute::size<1>(sA.shape())() / GROUP_SIZE;
    cute::array<decltype(make_fragment_like(accum)), NumChunksPerTileK>
        intermediate_array;

    constexpr int K_BLOCK_MAX = size<2>(tCrA_load);
    constexpr int K_WAIT_MAX = cute::min(K_BLOCK_MAX - 1, 7);
    static_assert(K_BLOCK_MAX >= 4, "Consider increasing TileShapeK");

    ConsumerToken barrier_token = {BarrierStatus::WaitAgain};
    // First k tile
    {
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
      pipeline.consumer_wait(smem_pipe_read, barrier_token);

      int read_stage = smem_pipe_read.index();

      ++smem_pipe_read;
      barrier_token = pipeline.consumer_try_wait(smem_pipe_read);

      // copy smem->rmem for A operand

      Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,
                             partitioned_extra_info, copy_partitions_extra_info,
                             0, read_stage);
      if (K_BLOCK_MAX > 1) {
        Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,
                               partitioned_extra_info,
                               copy_partitions_extra_info, 1, read_stage);
      }

      // src: tCrA_load, dst: tCrA_mma
      Utils::convert_A_kblock(tCrA_load, tCrA_mma, 0);

      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int chunk_id = 0; chunk_id < NumChunksPerTileK; ++chunk_id) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        CUTLASS_PRAGMA_UNROLL
        for (int mma_id = 0; mma_id < NumMMAsPerChunk; ++mma_id) {
          int k_block = chunk_id * NumMMAsPerChunk + mma_id;

          warpgroup_arrive();

          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma, tCrA_mma(_, _, k_block),
                     tCrB(_, _, k_block, read_stage),
                     intermediate_array[chunk_id]);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;

          warpgroup_commit_batch();

          if (k_block < K_BLOCK_MAX - 2) {
            Utils::copy_tensors_MK(
                smem_tiled_copy_A, tCsA, tCrA_copy_view, partitioned_extra_info,
                copy_partitions_extra_info, k_block + 2, read_stage);
          }
          if (k_block < K_BLOCK_MAX - 1) {
            Utils::convert_A_kblock(tCrA_load, tCrA_mma, k_block + 1);
          }
        }
      }

      warpgroup_wait<0>();

      CUTLASS_PRAGMA_UNROLL
      for (int chunk_id_ = 0; chunk_id_ < NumChunksPerTileK; ++chunk_id_) {
        warpgroup_fence_operand(intermediate_array[chunk_id_]);

        // Apply the group-wise scaling
        // tCrS  ((4, _2, _2), MMA_M, _1)
        // accum ((2, _2, _2), MMA_M, _1)
        auto tCrS = cute::get<1>(partitioned_extra_info);
        for (int mma_m = 0; mma_m < size<1>(accum); mma_m++) {
          for (int m = 0; m < size<0, 1>(accum); m++) {
            for (int n = 0; n < size<0, 2>(accum); n++) {
              for (int e = 0; e < size<0, 0>(accum); e++) {
                auto accum_coord = make_coord(make_tuple(e, m, n), mma_m, 0);
                auto scale_coord = make_coord(make_tuple(0, m, 0), mma_m, 0);

                if (chunk_id_ == 0) {
                  accum(accum_coord) =
                      intermediate_array[chunk_id_](accum_coord) *
                      static_cast<float>(tCrS(scale_coord)[0]);
                } else {
                  accum(accum_coord) =
                      fma(intermediate_array[chunk_id_](accum_coord),
                          static_cast<float>(tCrS(scale_coord)[chunk_id_]),
                          accum(accum_coord));
                }
              }
            }
          }
        }
      }

      --k_tile_count;
      if (k_tile_count > 0) {
        // Wait for K_BLOCK_MAX - 1 to be in flight to ensure that it is safe to
        // overwrite the A registers for the first mma.
        pipeline.consumer_wait(smem_pipe_read, barrier_token);

        Utils::copy_tensors_MK(
            smem_tiled_copy_A, tCsA, tCrA_copy_view, partitioned_extra_info,
            copy_partitions_extra_info, 0, smem_pipe_read.index());

        Utils::copy_tensors_MK(
            smem_tiled_copy_A, tCsA, tCrA_copy_view, partitioned_extra_info,
            copy_partitions_extra_info, 1, smem_pipe_read.index());

        Utils::convert_A_kblock(tCrA_load, tCrA_mma, 0);
      }
    }

    if (k_tile_count == 0) {
      return;
    }

    // Mainloop GMMAs
    CUTLASS_PRAGMA_NO_UNROLL
    for (; k_tile_count > 1; --k_tile_count) {
      //
      // Compute on k_tile
      //

      int read_stage = smem_pipe_read.index();
      ++smem_pipe_read;

      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int chunk_id = 0; chunk_id < NumChunksPerTileK; ++chunk_id) {
        tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

        CUTLASS_PRAGMA_UNROLL
        for (int mma_id = 0; mma_id < NumMMAsPerChunk; ++mma_id) {
          int k_block = chunk_id * NumMMAsPerChunk + mma_id;

          warpgroup_arrive();
          // (V,M) x (V,N) => (V,M,N)
          cute::gemm(tiled_mma, tCrA_mma(_, _, k_block),
                     tCrB(_, _, k_block, read_stage),
                     intermediate_array[chunk_id]);
          tiled_mma.accumulate_ = GMMA::ScaleOut::One;
          warpgroup_commit_batch();

          if (k_block == K_BLOCK_MAX - 1) {
            pipeline.consumer_release(
                smem_pipe_release);  // UNLOCK smem_pipe_release, done
                                     // _computing_ on it
            ++smem_pipe_release;
          }

          if (k_block == 0) {
            barrier_token = pipeline.consumer_try_wait(smem_pipe_read);
          }

          if (k_block == K_BLOCK_MAX - 1) {
            // The last k_block

            warpgroup_wait<0>();

            CUTLASS_PRAGMA_UNROLL
            for (int chunk_id_ = 0; chunk_id_ < NumChunksPerTileK;
                 ++chunk_id_) {
              warpgroup_fence_operand(intermediate_array[chunk_id_]);

              // Apply the group-wise scaling
              auto tCrS = cute::get<1>(partitioned_extra_info);
              for (int mma_m = 0; mma_m < size<1>(accum); mma_m++) {
                for (int m = 0; m < size<0, 1>(accum); m++) {
                  for (int n = 0; n < size<0, 2>(accum); n++) {
                    for (int e = 0; e < size<0, 0>(accum); e++) {
                      auto accum_coord =
                          make_coord(make_tuple(e, m, n), mma_m, 0);
                      auto scale_coord =
                          make_coord(make_tuple(0, m, 0), mma_m, 0);

                      accum(accum_coord) =
                          fma(intermediate_array[chunk_id_](accum_coord),
                              static_cast<float>(tCrS(scale_coord)[chunk_id_]),
                              accum(accum_coord));
                    }
                  }
                }
              }
            }

            pipeline.consumer_wait(smem_pipe_read, barrier_token);

            // copy scales when passing k_block=0
            Utils::copy_tensors_MK(
                smem_tiled_copy_A, tCsA, tCrA_copy_view, partitioned_extra_info,
                copy_partitions_extra_info, 0, smem_pipe_read.index());
            Utils::copy_tensors_MK(
                smem_tiled_copy_A, tCsA, tCrA_copy_view, partitioned_extra_info,
                copy_partitions_extra_info, 1, smem_pipe_read.index());
            Utils::convert_A_kblock(tCrA_load, tCrA_mma, 0);
          } else {
            if (k_block < K_BLOCK_MAX - 2) {
              Utils::copy_tensors_MK(smem_tiled_copy_A, tCsA, tCrA_copy_view,
                                     partitioned_extra_info,
                                     copy_partitions_extra_info, k_block + 2,
                                     read_stage);
            }
            Utils::convert_A_kblock(tCrA_load, tCrA_mma, k_block + 1);
          }
        }
      }
    }

    {
      //
      // Last k tile
      //
      Tensor intermediate = make_fragment_like(accum);

      int read_stage = smem_pipe_read.index();

      tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

      // Unroll the K mode manually to set scale D to 1
      CUTLASS_PRAGMA_UNROLL
      for (int k_block = 0; k_block < K_BLOCK_MAX; ++k_block) {
        warpgroup_arrive();
        // (V,M) x (V,N) => (V,M,N)
        cute::gemm(tiled_mma, tCrA_mma(_, _, k_block),
                   tCrB(_, _, k_block, read_stage), intermediate);
        tiled_mma.accumulate_ = GMMA::ScaleOut::One;
        warpgroup_commit_batch();

        if (k_block == K_BLOCK_MAX - 1) {
          // release prior barrier
          pipeline.consumer_release(
              smem_pipe_release);  // UNLOCK smem_pipe_release, done _computing_
                                   // on it
          ++smem_pipe_release;
        }

        if (k_block < K_BLOCK_MAX - 2) {
          Utils::copy_tensors_MK(
              smem_tiled_copy_A, tCsA, tCrA_copy_view, partitioned_extra_info,
              copy_partitions_extra_info, k_block + 2, read_stage);
        }
        if (k_block < K_BLOCK_MAX - 1) {
          Utils::convert_A_kblock(tCrA_load, tCrA_mma, k_block + 1);
        }

        if ((k_block + 1) % NumMMAsPerChunk == 0) {
          tiled_mma.accumulate_ = GMMA::ScaleOut::Zero;

          warpgroup_wait<0>();
          warpgroup_fence_operand(intermediate);

          // Apply the group-wise scaling
          auto tCrS = cute::get<1>(partitioned_extra_info);
          for (int mma_m = 0; mma_m < size<1>(accum); mma_m++) {
            for (int m = 0; m < size<0, 1>(accum); m++) {
              for (int n = 0; n < size<0, 2>(accum); n++) {
                for (int e = 0; e < size<0, 0>(accum); e++) {
                  auto accum_coord = make_coord(make_tuple(e, m, n), mma_m, 0);
                  auto scale_coord = make_coord(make_tuple(0, m, 0), mma_m, 0);
                  int scale_idx = k_block / NumMMAsPerChunk;

                  accum(accum_coord) =
                      fma(intermediate(accum_coord),
                          static_cast<float>(tCrS(scale_coord)[scale_idx]),
                          accum(accum_coord));
                }
              }
            }
          }
        }
      }
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  /// Perform a Consumer Epilogue to release all buffers
  CUTLASS_DEVICE void mma_tail(MainloopPipeline pipeline,
                               PipelineState smem_pipe_release,
                               int k_tile_count) {
    // Prologue GMMAs
    int prologue_mma_count = 1;
    k_tile_count -= prologue_mma_count;

    smem_pipe_release.advance(k_tile_count);

    // Wait on all GMMAs to complete
    // warpgroup_wait<0>();

    for (int count = 0; count < prologue_mma_count; ++count) {
      pipeline.consumer_release(
          smem_pipe_release);  // UNLOCK smem_pipe_release, done _computing_ on
                               // it
      ++smem_pipe_release;
    }
  }
  /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  //
  // Methods to perform different parts of TMA/Tensormap modifications
  //
  CUTLASS_DEVICE auto tensormaps_init(Params const& mainloop_params,
                                      TensorMapStorage& shared_tensormaps,
                                      int32_t sm_count, int32_t sm_idx) {
    cute::TmaDescriptor* gmem_tensormap =
        reinterpret_cast<cute::TmaDescriptor*>(mainloop_params.tensormaps);

    cute::TmaDescriptor* tma_desc_a = &gmem_tensormap[sm_idx];
    cute::TmaDescriptor* tma_desc_b = &gmem_tensormap[sm_idx + sm_count];
    cute::TmaDescriptor* tma_desc_scale =
        &gmem_tensormap[sm_idx + 2 * sm_count];
    cute::TmaDescriptor* tma_desc_zero = &gmem_tensormap[sm_idx + 3 * sm_count];

    // Bringing tensormaps from params to smem for modification later
    Tensor pA_tensormap = make_tensor(
        mainloop_params.tma_load_a.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sA_tensormap = make_tensor(
        make_smem_ptr(&shared_tensormaps.smem_tensormap_A), Int<1>{}, Int<1>{});
    Tensor pB_tensormap = make_tensor(
        mainloop_params.tma_load_b.get_tma_descriptor(), Int<1>{}, Int<1>{});
    Tensor sB_tensormap = make_tensor(
        make_smem_ptr(&shared_tensormaps.smem_tensormap_B), Int<1>{}, Int<1>{});

    if (cute::elect_one_sync()) {
      copy(recast<uint128_t>(pA_tensormap), recast<uint128_t>(sA_tensormap));
      copy(recast<uint128_t>(pB_tensormap), recast<uint128_t>(sB_tensormap));
    }

    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      Tensor pS_tensormap =
          make_tensor(mainloop_params.tma_load_scale.get_tma_descriptor(),
                      Int<1>{}, Int<1>{});
      Tensor sS_tensormap =
          make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_scale),
                      Int<1>{}, Int<1>{});
      if (cute::elect_one_sync()) {
        copy(recast<uint128_t>(pS_tensormap), recast<uint128_t>(sS_tensormap));
      }
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      Tensor pZ_tensormap =
          make_tensor(mainloop_params.tma_load_zero.get_tma_descriptor(),
                      Int<1>{}, Int<1>{});
      Tensor sZ_tensormap =
          make_tensor(make_smem_ptr(&shared_tensormaps.smem_tensormap_zero),
                      Int<1>{}, Int<1>{});
      if (cute::elect_one_sync()) {
        copy(recast<uint128_t>(pZ_tensormap), recast<uint128_t>(sZ_tensormap));
      }
    } else if constexpr (KernelConversionMode !=
                         ConversionMode::DirectConvert) {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in tensormaps_init.");
    }

    __syncwarp();

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple(tma_desc_a, tma_desc_b);
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScale) {
      return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_scale);
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      return cute::make_tuple(tma_desc_a, tma_desc_b, tma_desc_scale,
                              tma_desc_zero);
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in tensormaps_init.");
    }
  }

  // Replace address for the global tensor (to be done by single thread)
  CUTLASS_DEVICE
  void tensormaps_replace_global_address(TensorMapStorage& shared_tensormaps,
                                         Params const& mainloop_params,
                                         int32_t next_batch) {
    // Replacing global_address for the next batch
    cute::tma_descriptor_replace_addr_in_shared_mem(
        shared_tensormaps.smem_tensormap_A, mainloop_params.ptr_A[next_batch]);
    cute::tma_descriptor_replace_addr_in_shared_mem(
        shared_tensormaps.smem_tensormap_B, mainloop_params.ptr_B[next_batch]);
    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      cute::tma_descriptor_replace_addr_in_shared_mem(
          shared_tensormaps.smem_tensormap_scale,
          mainloop_params.ptr_S[next_batch]);
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      cute::tma_descriptor_replace_addr_in_shared_mem(
          shared_tensormaps.smem_tensormap_zero,
          mainloop_params.ptr_Z[next_batch]);
    } else if constexpr (KernelConversionMode !=
                         ConversionMode::DirectConvert) {
      static_assert(
          cutlass::detail::dependent_false<KernelSchedule>,
          "Conversion mode not handled in tensormaps_replace_global_address.");
    }
  }

  // Replace dim and strides for the global tensor - used only for Grouped GEMM
  // (to be done by single thread)
  template <class ProblemShape_MNKL>
  CUTLASS_DEVICE void tensormaps_replace_global_tensor_properties(
      TensorMapStorage& shared_tensormaps, Params const& mainloop_params,
      int32_t next_group, ProblemShape_MNKL problem_shape_mnkl) {
    const uint32_t M = get<0>(problem_shape_mnkl);
    const uint32_t N = get<1>(problem_shape_mnkl);
    const uint32_t K = get<2>(problem_shape_mnkl);

    // Replace all dims for consistency
    constexpr int MaxTensorRank = 5;
    cute::array<uint32_t, MaxTensorRank> prob_shape_A = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_A = {0, 0, 0, 0, 0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_B = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_B = {0, 0, 0, 0, 0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_scale = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_scale = {0, 0, 0, 0, 0};
    cute::array<uint32_t, MaxTensorRank> prob_shape_zero = {1, 1, 1, 1, 1};
    cute::array<uint64_t, MaxTensorRank> prob_stride_zero = {0, 0, 0, 0, 0};

    SwappedElementA const* ptr_A = nullptr;
    Tensor tensor_a = make_tensor(
        ptr_A, detail::get_gmem_layout(make_shape(M, K, Int<1>{}),
                                       mainloop_params.ptr_dA[next_group]));

    SwappedElementB const* ptr_B = nullptr;
    Tensor tensor_b = make_tensor(
        ptr_B, detail::get_gmem_layout(make_shape(N, K, Int<1>{}),
                                       mainloop_params.ptr_dB[next_group]));

    cute::detail::fill_tma_gmem_shape_stride(
        mainloop_params.tma_load_a, tensor_a, prob_shape_A, prob_stride_A);
    cute::detail::fill_tma_gmem_shape_stride(
        mainloop_params.tma_load_b, tensor_b, prob_shape_B, prob_stride_B);

    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      NonVoidElementScale const* ptr_S = nullptr;
      // auto scale_k = K / mainloop_params.chunk_size;
      auto scale_k = K / GROUP_SIZE;
      Tensor tensor_scale = make_tensor(detail::get_logical_ptr(ptr_S),
                                        make_shape(M, scale_k, Int<1>{}),
                                        mainloop_params.dS[next_group]);
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_scale,
                                               tensor_scale, prob_shape_scale,
                                               prob_stride_scale);
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      ElementZero const* ptr_Z = nullptr;
      // auto scale_k = K / mainloop_params.chunk_size;
      auto scale_k = K / GROUP_SIZE;
      Tensor tensor_zero = make_tensor(detail::get_logical_ptr(ptr_Z),
                                       make_shape(M, scale_k, Int<1>{}),
                                       mainloop_params.dS[next_group]);
      cute::detail::fill_tma_gmem_shape_stride(mainloop_params.tma_load_zero,
                                               tensor_zero, prob_shape_zero,
                                               prob_stride_zero);
    } else if constexpr (KernelConversionMode !=
                         ConversionMode::DirectConvert) {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in "
                    "tensormaps_replace_global_tensor_properties.");
    }

    // Convert strides to byte strides
    for (uint64_t& stride : prob_stride_A) {
      stride = (stride * sizeof_bits_v<SwappedElementA>) / 8;
    }
    for (uint64_t& stride : prob_stride_B) {
      stride = (stride * sizeof_bits_v<SwappedElementB>) / 8;
    }
    for (uint64_t& stride : prob_stride_scale) {
      stride = (stride * sizeof_bits_v<NonVoidElementScale>) / 8;
    }
    for (uint64_t& stride : prob_stride_zero) {
      stride = (stride * sizeof_bits_v<NonVoidElementScale>) / 8;
    }

    cute::tma_descriptor_replace_dims_strides_in_shared_mem(
        shared_tensormaps.smem_tensormap_A, prob_shape_A, prob_stride_A);
    cute::tma_descriptor_replace_dims_strides_in_shared_mem(
        shared_tensormaps.smem_tensormap_B, prob_shape_B, prob_stride_B);

    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(
          shared_tensormaps.smem_tensormap_scale, prob_shape_scale,
          prob_stride_scale);
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      cute::tma_descriptor_replace_dims_strides_in_shared_mem(
          shared_tensormaps.smem_tensormap_zero, prob_shape_zero,
          prob_stride_zero);
    } else if constexpr (KernelConversionMode !=
                         ConversionMode::DirectConvert) {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in "
                    "tensormaps_replace_global_tensor_properties.");
    }
  }

  template <class... TMs, class ProblemShape_MNKL>
  CUTLASS_DEVICE void tensormaps_perform_update(
      TensorMapStorage& shared_tensormaps, Params const& mainloop_params,
      cute::tuple<TMs...> const& input_tensormaps,
      ProblemShape_MNKL problem_shape_mnkl, int32_t next_batch) {
    if (cute::elect_one_sync()) {
      // Replacing global_address for the next batch
      tensormaps_replace_global_address(shared_tensormaps, mainloop_params,
                                        next_batch);

      if constexpr (IsGroupedGemmKernel) {
        // Replacing global dims and strides for the next batch
        tensormaps_replace_global_tensor_properties(
            shared_tensormaps, mainloop_params, next_batch, problem_shape_mnkl);
      }
    }
  }

  template <class... TMs>
  CUTLASS_DEVICE void tensormaps_cp_fence_release(
      TensorMapStorage& shared_tensormaps,
      cute::tuple<TMs...> const& input_tensormaps) {
    if (cute::elect_one_sync()) {
      cute::tma_desc_commit_group();
      cute::tma_desc_wait_group();
    }
    // Entire warp must do this (i.e. it's aligned)
    tma_descriptor_cp_fence_release(get<0>(input_tensormaps),
                                    shared_tensormaps.smem_tensormap_A);
    tma_descriptor_cp_fence_release(get<1>(input_tensormaps),
                                    shared_tensormaps.smem_tensormap_B);
    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      tma_descriptor_cp_fence_release(get<2>(input_tensormaps),
                                      shared_tensormaps.smem_tensormap_scale);
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      tma_descriptor_cp_fence_release(get<3>(input_tensormaps),
                                      shared_tensormaps.smem_tensormap_zero);
    } else if constexpr (KernelConversionMode !=
                         ConversionMode::DirectConvert) {
      static_assert(
          cutlass::detail::dependent_false<KernelSchedule>,
          "Conversion mode not handled in tensormaps_cp_fence_release.");
    }
  }

  // The entire warp must call this function collectively (that is, the
  // instructions are aligned)
  template <class... TMs>
  CUTLASS_DEVICE void tensormaps_fence_acquire(
      cute::tuple<TMs...> const& input_tensormaps) {
    cute::tma_descriptor_fence_acquire(get<0>(input_tensormaps));
    cute::tma_descriptor_fence_acquire(get<1>(input_tensormaps));
    if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
      cute::tma_descriptor_fence_acquire(get<2>(input_tensormaps));
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      cute::tma_descriptor_fence_acquire(get<3>(input_tensormaps));
    } else if constexpr (KernelConversionMode !=
                         ConversionMode::DirectConvert) {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in tensormaps_fence_acquire.");
    }
  }

  template <class InputTensors, class ProblemShape_MNKL>
  CUTLASS_DEVICE InputTensors
  tensors_perform_update(InputTensors const& input_tensors,
                         [[maybe_unused]] Params const& mainloop_params,
                         [[maybe_unused]] ProblemShape_MNKL problem_shape_mnkl,
                         [[maybe_unused]] int32_t next_batch) {
    return input_tensors;
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace cutlass::gemm::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
