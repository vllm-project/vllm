/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include "cute/arch/copy_sm90.hpp"
#include "cute/numeric/arithmetic_tuple.hpp"
#include "cute/util/type_traits.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"

/////////////////////////////////////////////////////////////////////////////////////////////////

namespace cutlass::gemm::collective::detail {

template <class Collective>
struct MixedGroupedGemmInputUtils {
 private:
  using KernelSchedule = typename Collective::KernelSchedule;
  using ConversionMode = typename Collective::ConversionMode;
  using SmemLayoutA = typename Collective::SmemLayoutA;
  using SmemLayoutB = typename Collective::SmemLayoutB;
  using SmemLayoutScale = typename Collective::SmemLayoutScale;
  using SwappedElementA = typename Collective::SwappedElementA;
  using SwappedElementB = typename Collective::SwappedElementB;
  using RealSwappedElementA = typename Collective::RealSwappedElementA;
  using RealSwappedElementB = typename Collective::RealSwappedElementB;
  using ElementScale = typename Collective::ElementScale;
  using ElementZero = typename Collective::ElementZero;
  using SmemCopyAtomScale = typename Collective::SmemCopyAtomScale;
  static constexpr auto KernelConversionMode = Collective::KernelConversionMode;
  static constexpr auto ModeHasScales = Collective::ModeHasScales;
  static constexpr auto UseScaleLookupTable = Collective::UseScaleLookupTable;

 public:
  static constexpr auto elements_per_smem_scale() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    } else if constexpr (ModeHasScales) {
      return cute::cosize_v<SmemLayoutScale>;
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Type not handled in scale smem allocation.");
    }
  }

  static constexpr auto elements_per_smem_zero() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert ||
                  KernelConversionMode == ConversionMode::ConvertAndScale) {
      return 0;
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      return cute::cosize_v<SmemLayoutScale>;
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Type not handled in scale smem allocation.");
    }
  }

  // These methods use some the public members of the class. For that reason, we
  // define them after the public section.
  static constexpr uint32_t compute_tma_transaction_bytes_mk() {
    return cutlass::bits_to_bytes(
        size<0>(SmemLayoutA{}) * size<1>(SmemLayoutA{}) *
        static_cast<uint32_t>(cute::sizeof_bits_v<SwappedElementA>));
  }

  static constexpr uint32_t compute_tma_transaction_bytes_nk() {
    return cutlass::bits_to_bytes(
        size<0>(SmemLayoutB{}) * size<1>(SmemLayoutB{}) *
        static_cast<uint32_t>(cute::sizeof_bits_v<SwappedElementB>));
  }

  static constexpr uint32_t compute_tma_transaction_bytes_extra() {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return 0;
    } else if constexpr (ModeHasScales) {
      constexpr uint32_t scale_tx_bytes = cutlass::bits_to_bytes(
          size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) *
          static_cast<uint32_t>(cute::sizeof_bits_v<ElementScale>));
      static_assert(
          scale_tx_bytes % 128 == 0,
          "Each scale stage must be 128B aligned.");  // required by TMA
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return scale_tx_bytes;
      } else if constexpr (KernelConversionMode ==
                           ConversionMode::ConvertAndScaleWithZero) {
        // Scale and zero share smem layout
        constexpr uint32_t zero_tx_bytes = cutlass::bits_to_bytes(
            size<0>(SmemLayoutScale{}) * size<1>(SmemLayoutScale{}) *
            static_cast<uint32_t>(cute::sizeof_bits_v<ElementZero>));
        static_assert(
            zero_tx_bytes % 128 == 0,
            "Each zero stage must be 128B aligned.");  // required by TMA
        return scale_tx_bytes + zero_tx_bytes;
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Type not handled in tma transaction bytes computation.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Type not handled in tma transaction bytes computation.");
    }
  }

  /// Utilities to copy A and extra inputs from smem to RF
  template <class SmemTiledCopyA, class TensorASmemView, class TensorACopyView,
            class... Ts, class... Us>
  CUTLASS_DEVICE static void copy_tensors_MK(
      SmemTiledCopyA const& smem_tiled_copy_A, TensorASmemView const& tCsA,
      TensorACopyView& tCrA_copy_view,
      cute::tuple<Ts...> const& partitioned_mma_extra_info,
      cute::tuple<Us...> const& tiled_copy_and_views, int k_block,
      int read_stage) {
    copy(smem_tiled_copy_A, tCsA(_, _, k_block, read_stage),
         tCrA_copy_view(_, _, k_block));

    if (k_block == 0) {
      // We are starting a new k-tile so copy the scale
      if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
        // nothing to do
      } else if constexpr (ModeHasScales) {
        auto smem_tiled_copy_S = cute::get<0>(tiled_copy_and_views);
        auto tCrS_copy_view = cute::get<1>(tiled_copy_and_views);
        auto tCsS = cute::get<0>(partitioned_mma_extra_info);
        copy(smem_tiled_copy_S, tCsS(_, _, k_block, read_stage),
             tCrS_copy_view(_, _, k_block));
        if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
          // Nothing extra to do
        } else if constexpr (KernelConversionMode ==
                             ConversionMode::ConvertAndScaleWithZero) {
          auto tCsZ = cute::get<2>(partitioned_mma_extra_info);
          auto tCrZ_copy_view = cute::get<2>(tiled_copy_and_views);
          copy(smem_tiled_copy_S, tCsZ(_, _, k_block, read_stage),
               tCrZ_copy_view(_, _, k_block));
        } else {
          static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                        "Conversion mode not handled in A -> RF path.");
        }
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled in A -> RF path.");
      }
    }
  }

  // The core converter uses a lookup table to converts i4 -> 8 bit value.
  template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut,
            class EngineScale,
            class LayoutScale>
  CUTLASS_DEVICE static void
  lookup_table_convert(  // Accept mutable temporaries
      Tensor<EngineIn, LayoutIn> const& src, Tensor<EngineOut, LayoutOut>&& dst,
      Tensor<EngineScale, LayoutScale> const& scales_neg,
      Tensor<EngineScale, LayoutScale> const& scales_pos) {
    lookup_table_convert(src, dst, scales_neg, scales_pos);
  }
  template <class EngineIn, class LayoutIn, class EngineOut, class LayoutOut,
            class EngineScale, class LayoutScale>
  CUTLASS_DEVICE static void lookup_table_convert(
      Tensor<EngineIn, LayoutIn> const& src, Tensor<EngineOut, LayoutOut>& dst,
      Tensor<EngineScale, LayoutScale> const& scales_neg,
      Tensor<EngineScale, LayoutScale> const& scales_pos) {
    constexpr int N = cute::cosize(LayoutIn{});
    static_assert(N == 4 || N == 8);
    static_assert(cosize(LayoutScale{}) <= N / 4,
                  "at least 4 consecutive weights must share the same scale.");
    using SrcArray = cutlass::Array<cutlass::int4b_t, 8>;
    using DstArray = cutlass::Array<RealSwappedElementB, 8>;
    using RegArray = cutlass::AlignedArray<uint32_t, N / 4, sizeof(DstArray)>;

    // View the input as reg
    auto&& src_reg = cute::recast<uint32_t>(src)(0);
    auto&& r = cute::recast<RegArray>(dst)(0);

    // Determines if to get from the signed or unsigned candidates
    static constexpr uint32_t immLut = (0xf0 & 0xcc) | 0xaa;
    uint32_t sign;  // ((reg & 0x88888888) | 0x64206420) >> 1
    asm volatile(
        "{\n"
        "  lop3.b32 %0, %1, %2, %3, %4;\n"
        "}\n"
        : "=r"(sign)
        : "r"(src_reg), "n"(0x88888888), "n"(0x64206420), "n"(immLut));
    sign = sign >> 1;

    // Ignore sign bit when indexing into LUT
    uint32_t lut_idx = src_reg & 0x77777777;
    Tensor scales_neg_ = cute::filter(scales_neg);
    Tensor scales_pos_ = cute::filter(scales_pos);
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < N / 4; ++i, lut_idx >>= 16, sign >>= 16) {
      auto&& scale_neg_ =
          reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_neg_(i));
      auto&& scale_pos_ =
          reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(scales_pos_(i));
      asm volatile(
          "{\n"
          "  .reg .b32 pos, neg                    ;\n"
          "  prmt .b32 neg, %3, %4, %1             ;\n"
          "  prmt .b32 pos, %5, %6, %1             ;\n"
          "  prmt .b32 %0, pos, neg, %2            ;\n"
          "}\n"
          : "=r"(r[i])
          : "r"(lut_idx), "r"(sign), "r"(scale_neg_[0]), "r"(scale_neg_[1]),
            "r"(scale_pos_[0]), "r"(scale_pos_[1]));
    }
  }

  /// Utilities to dequantize A.
  template <class Layout>
  CUTLASS_DEVICE static void static_check_scale(Layout const& tensor) {
    static_assert(
        shape<0>(Layout{}) >= 4 && stride<0>(Layout{}) == 0,
        "At least 4 adjacent weights in a thread must share the same scale.");
  }
  template <class Engine, class Layout>
  CUTLASS_DEVICE static void static_check_scale(
      Tensor<Engine, Layout> const& tensor) {
    static_check_scale(flatten(Layout{}));
  }

  template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut,
            class... Ts>
  CUTLASS_DEVICE static void dequantize_A_kblock(
      Tensor<EngineIn, LayoutIn> const& tCrA_load,
      Tensor<EngineOut, LayoutOut>& tCrA_mma,
      cute::tuple<Ts...>& partitioned_extra_info, int const k_block) {
    static_assert(is_rmem<EngineIn>::value,
                  "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value,
                  "Output tensor for A conversion must come from registers");
    static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);
    using SrcType = typename EngineIn::value_type;
    using DstType = typename EngineOut::value_type;

    Tensor src = tCrA_load(_, _, k_block);
    Tensor dst = tCrA_mma(_, _, k_block);

    CUTE_STATIC_ASSERT_V(
        size(src(_, 0)) == cosize(src(_, 0).layout()),
        "The first mode of tensor src must be contiguous in memory");
    // try to make the size of the first mode equal to 32bit
    int constexpr NumValPerSrcReg = cute::min(
        decltype(size(src(_, 0)))::value, ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1, -1>(
        cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1, -1>(
        cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));

    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(dst_vm); ++i) {
        LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
      }
    } else if constexpr (UseScaleLookupTable) {
      constexpr int num_elements = decltype(size(src))::value;
      static_assert(
          is_same_v<RealSwappedElementA, cutlass::int4b_t>,
          "Lookup table only supports int4 being the quant type now.");
      static_assert(sizeof_bits_v<ElementScale> == 64,
                    "Lookup table only supports 8 8bit scale values now.");
      static_assert(
          num_elements % 4 == 0 && num_elements >= 4,
          "Lookup table requires a vector size of 4x when converting.");

      Tensor tCrS_neg = cute::get<1>(partitioned_extra_info);
      auto&& tCrS_pos = cute::get<2>(
          partitioned_extra_info);  // modification to its value is needed
      Tensor scales_neg = tCrS_neg(_, _, k_block);
      Tensor scales_pos = tCrS_pos(_, _, k_block);
      CUTE_STATIC_ASSERT_V(cute::size(src) == cute::size(scales_neg));

      static_check_scale(scales_neg);
      static_check_scale(scales_pos);
      Tensor scales_neg_vm = cute::group_modes<1, -1>(
          cute::zipped_divide(scales_neg, Int<NumValPerSrcReg>{}));
      Tensor scales_pos_vm = cute::group_modes<1, -1>(
          cute::zipped_divide(scales_pos, Int<NumValPerSrcReg>{}));

      if (k_block == 0) {
        Tensor scales_neg_vm_ = filter(scales_neg_vm);
        Tensor scales_pos_vm_ = filter(scales_pos_vm);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(scales_neg_vm_.layout()); ++i) {
          auto&& scale_neg_ =
              reinterpret_cast<cutlass::Array<uint32_t, 2> const&>(
                  scales_neg_vm_(i));
          auto&& scale_pos_ =
              reinterpret_cast<cutlass::Array<uint32_t, 2>&>(scales_pos_vm_(i));
          constexpr uint32_t immLut = (0xf0 & 0xcc) ^ 0xaa;
          asm volatile(
              "{\n"
              "  lop3 .b32 %0, %2, %4, %5, %6;\n"
              "  xor  .b32 %1, %3, %5;        \n"
              "}\n"
              : "=r"(scale_pos_[0]), "=r"(scale_pos_[1])
              : "r"(scale_neg_[0]), "r"(scale_neg_[1]), "n"(0xFFFFFF00),
                "n"(0x80808080), "n"(immLut));
        }
      }
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size<1>(dst_vm); ++i) {
        lookup_table_convert(src_vm(_, i), dst_vm(_, i), scales_neg_vm(_, i),
                             scales_pos_vm(_, i));
      }
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScale) {
      Tensor scales = cute::get<1>(partitioned_extra_info)(_, _, k_block);
      CUTE_STATIC_ASSERT_V(size(src) == size(scales));
      Tensor scales_vm = cute::group_modes<1, -1>(
          cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));

      if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            dst_vm(j, i) *= scales_vm(j, i);
          }
        }
      } else {
        auto stage = make_tensor_like<ElementScale>(src_vm(_, 0));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), stage);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            stage(j) *= scales_vm(j, i);
          }
          LayoutAwareConvert(stage, dst_vm(_, i));
        }
      }
    } else if constexpr (KernelConversionMode ==
                         ConversionMode::ConvertAndScaleWithZero) {
      static_assert(is_same_v<ElementScale, ElementZero>,
                    "ElementScale and ElementZero must be the same.");
      Tensor scales = cute::get<1>(partitioned_extra_info)(_, _, k_block);
      Tensor zeros = cute::get<3>(partitioned_extra_info)(_, _, k_block);
      CUTE_STATIC_ASSERT_V(size(src) == size(scales));
      CUTE_STATIC_ASSERT_V(size(src) == size(zeros));
      Tensor scales_vm = cute::group_modes<1, -1>(
          cute::zipped_divide(scales, Int<NumValPerSrcReg>{}));
      Tensor zeros_vm = cute::group_modes<1, -1>(
          cute::zipped_divide(zeros, Int<NumValPerSrcReg>{}));

      if constexpr (is_same_v<DstType, ElementScale>) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            dst_vm(j, i) = dst_vm(j, i) * scales_vm(j, i) + zeros_vm(j, i);
          }
        }
      } else {
        auto stage = make_tensor_like<ElementScale>(src_vm(_, 0));
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size<1>(dst_vm); ++i) {
          LayoutAwareConvert(src_vm(_, i), stage);
          CUTLASS_PRAGMA_UNROLL
          for (int j = 0; j < size<0>(dst_vm); ++j) {
            stage(j) = stage(j) * scales_vm(j, i) + zeros_vm(j, i);
          }
          LayoutAwareConvert(stage, dst_vm(_, i));
        }
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "No A data is loaded.");
    }
  }

  template <class EngineIn, class EngineOut, class LayoutIn, class LayoutOut,
            class... Ts>
  CUTLASS_DEVICE static void convert_A_kblock(
      Tensor<EngineIn, LayoutIn> const& tCrA_load,
      Tensor<EngineOut, LayoutOut>& tCrA_mma, int const k_block) {
    static_assert(is_rmem<EngineIn>::value,
                  "Input tensor for A conversion must come from registers");
    static_assert(is_rmem<EngineOut>::value,
                  "Output tensor for A conversion must come from registers");
    static_assert(cosize_v<LayoutIn> == cosize_v<LayoutOut>);
    static_assert(size_v<LayoutIn> == cosize_v<LayoutIn>);
    static_assert(size_v<LayoutOut> == cosize_v<LayoutOut>);
    using SrcType = typename EngineIn::value_type;

    Tensor src = tCrA_load(_, _, k_block);
    Tensor dst = tCrA_mma(_, _, k_block);

    CUTE_STATIC_ASSERT_V(
        size(src(_, 0)) == cosize(src(_, 0).layout()),
        "The first mode of tensor src must be contiguous in memory");
    // try to make the size of the first mode equal to 32bit
    int constexpr NumValPerSrcReg = cute::min(
        decltype(size(src(_, 0)))::value, ceil_div(32, sizeof_bits_v<SrcType>));
    Tensor src_vm = cute::group_modes<1, -1>(
        cute::zipped_divide(src, Int<NumValPerSrcReg>{}));
    Tensor dst_vm = cute::group_modes<1, -1>(
        cute::zipped_divide(dst, Int<NumValPerSrcReg>{}));

    // KernelConversionMode == ConversionMode::DirectConvert
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < size<1>(dst_vm); ++i) {
      LayoutAwareConvert(src_vm(_, i), dst_vm(_, i));
    }
  }

  /// Utilities for any additional inputs inside of the TMA load
  template <class Params, class TensorStorage, class... Ts>
  CUTLASS_DEVICE static auto partition_extra_tma_inputs(
      Params const& mainloop_params, cute::tuple<Ts...> const& load_inputs,
      TensorStorage& shared_tensors, uint2 const& cluster_local_block_id,
      int const m_coord, int const l_coord) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      return cute::make_tuple();
    } else if constexpr (ModeHasScales) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()),
                              SmemLayoutScale{});  // (BLK_M,BLK_K,PIPE)
      Tensor gS_mkl = get<2>(load_inputs);
      auto block_tma_s =
          mainloop_params.tma_load_scale.get_slice(cluster_local_block_id.y);
      Tensor gS = gS_mkl(_, _, m_coord, _, l_coord);  // (BLK_M,BLK_K,k)

      Tensor tSgS = block_tma_s.partition_S(gS);  // (TMA,TMA_M,TMA_K,k)
      Tensor tSsS = block_tma_s.partition_D(sS);  // (TMA,TMA_M,TMA_K,PIPE)
      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tSgS, tSsS);
      } else if constexpr (KernelConversionMode ==
                           ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()),
                                SmemLayoutScale{});  // (BLK_M,BLK_K,PIPE)
        Tensor gZ_mkl = get<3>(load_inputs);
        auto block_tma_z =
            mainloop_params.tma_load_zero.get_slice(cluster_local_block_id.y);
        Tensor gZ = gZ_mkl(_, _, m_coord, _, l_coord);  // (BLK_M,BLK_K,k)

        Tensor tZgZ = block_tma_z.partition_S(gZ);  // (TMA,TMA_M,TMA_K,k)
        Tensor tZsZ = block_tma_z.partition_D(sZ);  // (TMA,TMA_M,TMA_K,PIPE)
        return cute::make_tuple(tSgS, tSsS, tZgZ, tZsZ);
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled for input partitioning.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled for input partitioning.");
    }
  }

  /// Utilities for partitioning extra inputs for loading from smem in the
  /// mainloop.
  template <class ThreadMma, class TensorStorage>
  CUTLASS_DEVICE static auto partition_extra_mma_info(
      ThreadMma const& mma_thread_slice, TensorStorage& shared_tensors) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // nothing to do
      return cute::make_tuple();
    } else if constexpr (UseScaleLookupTable) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()),
                              SmemLayoutScale{});  // (BLK_M,BLK_SCALE_K,PIPE)
      Tensor tCsS = mma_thread_slice.partition_A(sS);
      Tensor tCrS = make_tensor<ElementScale>(
          mma_thread_slice.partition_fragment_A(sS(_, _, Int<0>{})).layout());

      return cute::make_tuple(tCsS, tCrS);
    } else if constexpr (ModeHasScales) {
      Tensor sS = make_tensor(make_smem_ptr(shared_tensors.smem_scale.begin()),
                              SmemLayoutScale{});  // (BLK_M,BLK_SCALE_K,PIPE)
      Tensor tCsS = mma_thread_slice.partition_A(sS);
      Tensor tCrS = make_tensor<ElementScale>(
          mma_thread_slice.partition_fragment_A(sS(_, _, Int<0>{})).layout());

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(tCsS, tCrS);
      } else if constexpr (KernelConversionMode ==
                           ConversionMode::ConvertAndScaleWithZero) {
        Tensor sZ = make_tensor(make_smem_ptr(shared_tensors.smem_zero.begin()),
                                SmemLayoutScale{});  // (BLK_M,BLK_SCALE_K,PIPE)
        Tensor tCsZ = mma_thread_slice.partition_A(sZ);
        Tensor tCrZ = make_tensor<ElementZero>(
            mma_thread_slice.partition_fragment_A(sZ(_, _, Int<0>{})).layout());
        return cute::make_tuple(tCsS, tCrS, tCsZ, tCrZ);
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled in A -> RF path.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in A -> RF path.");
    }
  }

  /// Returns the tiled copy and copy views for the extra inputs.
  template <class TiledMma, class... Ts>
  CUTLASS_DEVICE static auto retile_extra_mma_info(
      TiledMma const& tiled_mma, cute::tuple<Ts...>& partitioned_extra_info,
      int const warp_group_thread_idx) {
    if constexpr (KernelConversionMode == ConversionMode::DirectConvert) {
      // nothing to do
      return cute::make_tuple();
    } else if constexpr (ModeHasScales) {
      auto smem_tiled_copy_S =
          make_tiled_copy_A(SmemCopyAtomScale{}, tiled_mma);
      auto smem_thr_copy_S =
          smem_tiled_copy_S.get_thread_slice(warp_group_thread_idx);
      Tensor tCrS_copy_view = smem_thr_copy_S.retile_D(
          cute::get<1>(partitioned_extra_info));  // (CPY,CPY_M,CPY_K)

      if constexpr (KernelConversionMode == ConversionMode::ConvertAndScale) {
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view);
      } else if constexpr (KernelConversionMode ==
                           ConversionMode::ConvertAndScaleWithZero) {
        Tensor tCrZ_copy_view = smem_thr_copy_S.retile_D(
            cute::get<3>(partitioned_extra_info));  // (CPY,CPY_M,CPY_K)
        return cute::make_tuple(smem_tiled_copy_S, tCrS_copy_view,
                                tCrZ_copy_view);
      } else {
        static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                      "Conversion mode not handled in A -> RF path.");
      }
    } else {
      static_assert(cutlass::detail::dependent_false<KernelSchedule>,
                    "Conversion mode not handled in A -> RF path.");
    }
  }
};

}  // namespace cutlass::gemm::collective::detail
