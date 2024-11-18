/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights
 *reserved. SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
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
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

//
// This file is a modified excerpt of
// include/cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp
// from https://github.com/NVIDIA/cutlass v3.5.0
// It has been modified to support either row/column or scalar broadcasting
// where the tensor being loaded from is always passed in via a device pointer.
// This lets one compiled kernel handle all cases of per-tensor or
// per-channel/per-token quantization.
//
// This interface also allows the scales to be passed in as tensors that
// consistently reside on the device, which avoids an issue with a previous
// implementation where scalars needed to be on the CPU since they
// were passed in via float values. This created a potential performance hazard
// if scales were initially on the device, and caused torch.compile graphs
// breaks when moving scales to the CPU.
//
#pragma once

// Turn off clang-format for the entire file to keep it close to upstream
// clang-format off

#include "cutlass/cutlass.h"
#include "cutlass/arch/barrier.h"

#include "cute/tensor.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"

namespace cutlass::epilogue::fusion {

using namespace cute;
using namespace detail;

// Row vector broadcast
template<
  int Stages,
  class CtaTileShapeMNK,
  class Element,
  class StrideMNL = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<Element>
>
struct Sm90RowOrScalarBroadcast {
  static_assert(Stages == 0, "Row broadcast doesn't support smem usage");
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))>); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_1>{});

  struct SharedStorage { 
    array_aligned<Element, size<1>(CtaTileShapeMNK{})> smem;
  };

  // This struct has been modified to have a bool indicating that ptr_row is a 
  // scalar that must be broadcast, instead of containing a scalar that is 
  // valid if ptr_row is null.
  struct Arguments {
    Element const* ptr_row = nullptr;
    bool row_broadcast = true;
    StrideMNL dRow = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  Sm90RowOrScalarBroadcast() { }

  CUTLASS_HOST_DEVICE
  Sm90RowOrScalarBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params)
      , smem(const_cast<Element*>(shared_storage.smem.data())) { }

  Params params;
  Element *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return (!params.row_broadcast && *(params.ptr_row) == Element(0));
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template <class GS_GTensor, class GS_STensor, class GS_CTensor, class Tiled_G2S, class SR_STensor, class SR_RTensor, class CTensor, class ThrResidue, class ThrNum>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
        GS_GTensor tGS_gRow_, GS_STensor tGS_sRow_, 
        GS_CTensor tGS_cRow_, Tiled_G2S tiled_g2s_, 
        SR_STensor tSR_sRow_, SR_RTensor tSR_rRow_,
        CTensor tCcRow_, ThrResidue residue_tCcRow_, ThrNum thr_num_, Params const& params_)
      : tGS_gRow(tGS_gRow_)
      , tGS_sRow(tGS_sRow_)
      , tGS_cRow(tGS_cRow_)
      , tiled_G2S(tiled_g2s_)
      , tSR_sRow(tSR_sRow_)
      , tSR_rRow(tSR_rRow_)
      , tCcRow(tCcRow_)
      , residue_tCcRow(residue_tCcRow_)
      , params(params_) {}

    GS_GTensor tGS_gRow;                                                         // (CPY,CPY_M,CPY_N)
    GS_STensor tGS_sRow;                                                         // (CPY,CPY_M,CPY_N)
    GS_CTensor tGS_cRow;                                                         // (CPY,CPY_M,CPY_N)
    Tiled_G2S tiled_G2S;

    SR_STensor tSR_sRow;                                                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    SR_RTensor tSR_rRow;                                                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N) 
  
    CTensor tCcRow;                                                              // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    ThrResidue residue_tCcRow;                                                   // (m, n)
    ThrNum thr_num;
    Params const& params;

    CUTLASS_DEVICE void
    begin() {
      if (!params.row_broadcast) {
        fill(tSR_rRow, *(params.ptr_row));
        return;
      }

      auto synchronize = [&] () { cutlass::arch::NamedBarrier::sync(thr_num, cutlass::arch::ReservedNamedBarriers::EpilogueBarrier); };
      Tensor tGS_gRow_flt = filter_zeros(tGS_gRow);
      Tensor tGS_sRow_flt = filter_zeros(tGS_sRow);
      Tensor tGS_cRow_flt = make_tensor(tGS_cRow.data(), make_layout(tGS_gRow_flt.shape(), tGS_cRow.stride()));

      for (int i = 0; i < size(tGS_gRow_flt); ++i) {
        if (get<1>(tGS_cRow_flt(i)) >= size<1>(CtaTileShapeMNK{})) {
          continue; // OOB of SMEM, 
        }
        if (elem_less(tGS_cRow_flt(i), make_coord(get<0>(residue_tCcRow), get<1>(residue_tCcRow)))) {
          tGS_sRow_flt(i) = tGS_gRow_flt(i);
        }
        else {
          tGS_sRow_flt(i) = Element(0); // Set to Zero when OOB so LDS could be issue without any preds.
        }
      }
      synchronize();
    }

    CUTLASS_DEVICE void
    begin_loop(int epi_m, int epi_n) {
      if (epi_m == 0) { // Assumes M-major subtile loop
        if (!params.row_broadcast) return; // Do not issue LDS when row is scalar 
        Tensor tSR_sRow_flt = filter_zeros(tSR_sRow(_,_,_,epi_m,epi_n));
        Tensor tSR_rRow_flt = filter_zeros(tSR_rRow);
        copy(tSR_sRow_flt, tSR_rRow_flt);
      }
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<Element, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<Element, FragmentSize> frg_row;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_row[i] = tSR_rRow(epi_v * FragmentSize + i);
      }

      return frg_row;
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    using ThreadCount = decltype(size(args.tiled_copy));

    Tensor mRow = make_tensor(make_gmem_ptr(params.ptr_row), make_shape(M,N,L), params.dRow);
    Tensor gRow = local_tile(mRow(_,_,l), take<0,2>(args.tile_shape_mnk), make_coord(m, n));          // (CTA_M, CTA_N)
    Tensor sRow = make_tensor(make_smem_ptr(smem), 
        make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{})), make_shape(_0{}, _1{}));  // (CTA_M, CTA_N)
    //// G2S: Gmem to Smem
    auto tiled_g2s = make_tiled_copy(Copy_Atom<DefaultCopy, Element>{},
                                     Layout< Shape<_1, ThreadCount>, 
                                            Stride<_0,          _1>>{}, 
                                     Layout<_1>{});   
    auto thr_g2s = tiled_g2s.get_slice(args.thread_idx);
    Tensor tGS_gRow = thr_g2s.partition_S(gRow);
    Tensor tGS_sRow = thr_g2s.partition_D(sRow);

    //// G2S: Coord 
    auto cRow = make_identity_tensor(make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{})));
    Tensor tGS_cRow = thr_g2s.partition_S(cRow);

    //// S2R: Smem to Reg
    Tensor tSR_sRow = sm90_partition_for_epilogue<ReferenceSrc>(sRow, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tSR_rRow = make_tensor_like(take<0,3>(tSR_sRow));                                           // (CPY,CPY_M,CPY_N)

    return ConsumerStoreCallbacks<decltype(tGS_gRow), decltype(tGS_sRow), decltype(tGS_cRow), decltype(tiled_g2s), decltype(tSR_sRow), decltype(tSR_rRow), decltype(args.tCcD), decltype(args.residue_cD), ThreadCount>(
      tGS_gRow, 
      tGS_sRow, 
      tGS_cRow, tiled_g2s, 
      tSR_sRow, 
      tSR_rRow, 
      args.tCcD, 
      args.residue_cD,
      ThreadCount{}, 
      params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// Column vector broadcast
template<
  int Stages,
  class CtaTileShapeMNK,
  class Element,
  class StrideMNL = Stride<_1,_0,_0>,
  int Alignment = 128 / sizeof_bits_v<Element>
>
struct Sm90ColOrScalarBroadcast {
  static_assert(Stages == 0, "Column broadcast doesn't support smem usage yet");
  static_assert(Alignment * sizeof_bits_v<Element> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(
    (cute::is_same_v<StrideMNL, Stride<_1,_0, _0>>) || // col vector broadcast, e.g. per-row alpha/bias
    (cute::is_same_v<StrideMNL, Stride<_1,_0,int>>));  // batched col vector broadcast, e.g. batched per-row bias

  // Accumulator distributes col elements evenly amongst threads so we can just directly load from gmem
  struct SharedStorage { };

  // This struct has been modified to have a bool indicating that ptr_col is a 
  // scalar that must be broadcast, instead of containing a scalar that is 
  // valid if ptr_col is null.
  struct Arguments {
    Element const* ptr_col = nullptr;
    bool col_broadcast = true;
    StrideMNL dCol = {};
  };

  using Params = Arguments;

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return args;
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return (!params.col_broadcast && *(params.ptr_col) == Element(0));
  }

  CUTLASS_HOST_DEVICE
  Sm90ColOrScalarBroadcast() { }

  CUTLASS_HOST_DEVICE
  Sm90ColOrScalarBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params) { }

  Params params;

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(
      GTensor&& tCgCol,
      RTensor&& tCrCol,
      CTensor&& tCcCol,
      ProblemShape problem_shape,
      Params const& params
    ): 
      tCgCol(cute::forward<GTensor>(tCgCol)),
      tCrCol(cute::forward<RTensor>(tCrCol)),
      tCcCol(cute::forward<CTensor>(tCcCol)),
      m(get<0>(problem_shape)),
      params(params) {}

    GTensor tCgCol;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    RTensor tCrCol;
    CTensor tCcCol;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    Params const& params;
    int m;

    CUTLASS_DEVICE void
    begin() {
      Tensor pred = make_tensor<bool>(shape(tCgCol));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(pred); ++i) {
        pred(i) = get<0>(tCcCol(i)) < m;
      }

      if (!params.col_broadcast) {
        fill(tCrCol, *(params.ptr_col));
        return;
      }

      // Filter so we don't issue redundant copies over stride-0 modes
      // (only works if 0-strides are in same location, which is by construction)
      copy_if(pred, filter(tCgCol), filter(tCrCol));
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<Element, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<Element, FragmentSize> frg_col;
      Tensor tCrCol_mn = tCrCol(_,_,_,epi_m,epi_n);

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_col[i] = tCrCol_mn(epi_v * FragmentSize + i);
      }

      return frg_col;
    }

  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    Tensor mCol = make_tensor(make_gmem_ptr(params.ptr_col), make_shape(M,N,L), params.dCol);
    Tensor tCgCol = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      mCol, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrCol = make_tensor_like(tCgCol);                                          // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)

    // Generate an identity tensor matching the shape of the global tensor and 
    //  partition the same way, this will be used to generate the predicate
    //  tensor for loading
    Tensor cCol = make_identity_tensor(mCol.shape());
    Tensor tCcCol = sm90_partition_for_epilogue<ReferenceSrc>(                         // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
      cCol, args.tile_shape_mnk, args.tile_coord_mnkl, args.epi_tile, args.tiled_copy, args.thread_idx);

    return ConsumerStoreCallbacks(
      cute::move(tCgCol), 
      cute::move(tCrCol), 
      cute::move(tCcCol), 
      args.problem_shape_mnkl, 
      params
    );
  }
};

}
