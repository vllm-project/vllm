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
  // Row bcast reuses the mbarriers from the epilogue subtile load pipeline, so this must be at least
  // ceil_div(StagesC, epi tiles per CTA tile) + 1 to ensure no data races
  int Stages,
  class CtaTileShapeMNK,
  class Element,
  class StrideMNL = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<Element>
>
struct Sm90RowOrScalarBroadcast {
  static_assert(Alignment * sizeof_bits_v<Element> % 128 == 0, "sub-16B alignment not supported yet");
  static_assert(
    (cute::is_same_v<StrideMNL, Stride<_0,_1, _0>>) || // row vector broadcast, e.g. per-col alpha/bias
    (cute::is_same_v<StrideMNL, Stride<_0,_1,int>>));  // batched row vector broadcast

  // Accumulator doesn't distribute row elements evenly amongst threads so we must buffer in smem
  struct SharedStorage {
    alignas(16) array_aligned<Element, size<1>(CtaTileShapeMNK{}) * Stages> smem_row;
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
      : params(params),
        smem_row(const_cast<Element*>(shared_storage.smem_row.data())) { }

  Params params;
  Element* smem_row;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return true;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return (!params.row_broadcast && *(params.ptr_row) == Element(0));
  }

  template <int EpiTiles, class GTensor, class STensor>
  struct ProducerLoadCallbacks : EmptyProducerLoadCallbacks {
    CUTLASS_DEVICE
    ProducerLoadCallbacks(GTensor&& gRow, STensor&& sRow, Params const& params)
      : gRow(cute::forward<GTensor>(gRow)),
        sRow(cute::forward<STensor>(sRow)),
        params(params) {}

    GTensor gRow;                                                                                 // (CTA_M,CTA_N)
    STensor sRow;                                                                                 // (CTA_M,CTA_N,PIPE)
    Params const& params;

    CUTLASS_DEVICE void
    begin(uint64_t* full_mbarrier_ptr, int load_iteration, bool issue_tma_load) {
      if (params.ptr_row == nullptr) {
        return;
      }

      if (issue_tma_load) {
        // Increment the expect-tx count of the first subtile's mbarrier by the row vector's byte-size
        constexpr uint32_t copy_bytes = size<1>(CtaTileShapeMNK{}) * sizeof_bits_v<Element> / 8;
        cutlass::arch::ClusterTransactionBarrier::expect_transaction(full_mbarrier_ptr, copy_bytes);
        // Issue the TMA bulk copy
        auto bulk_copy = Copy_Atom<SM90_BULK_COPY_AUTO, Element>{}.with(*full_mbarrier_ptr);
        // Filter so we don't issue redundant copies over stride-0 modes
        int bcast_pipe_index = (load_iteration / EpiTiles) % Stages;
        copy(bulk_copy, filter(gRow), filter(sRow(_,_,bcast_pipe_index)));
      }
    }
  };

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {

    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;
    Tensor mRow = make_tensor(make_gmem_ptr(params.ptr_row), make_shape(M,N,L), params.dRow);
    Tensor gRow = local_tile(mRow, take<0,2>(args.tile_shape_mnk), make_coord(m,n,l));            // (CTA_M,CTA_N)
    Tensor sRow = make_tensor(make_smem_ptr(smem_row),                                            // (CTA_M,CTA_N,PIPE)
                    make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{}), Stages),
                    make_stride(_0{},_1{},size<1>(CtaTileShapeMNK{})));

    constexpr int EpiTiles = decltype(size<1>(zipped_divide(make_layout(take<0,2>(args.tile_shape_mnk)), args.epi_tile)))::value;
    return ProducerLoadCallbacks<EpiTiles, decltype(gRow), decltype(sRow)>(
      cute::move(gRow), cute::move(sRow), params);
  }

  template <int EpiTiles, class RTensor, class STensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(RTensor&& tCrRow, STensor&& tCsRow, Params const& params)
      : tCrRow(cute::forward<RTensor>(tCrRow)),
        tCsRow(cute::forward<STensor>(tCsRow)),
        params(params) {}

    RTensor tCrRow;                                                               // (CPY,CPY_M,CPY_N)
    STensor tCsRow;                                                               // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,PIPE)
    Params const& params;

    CUTLASS_DEVICE void
    previsit(int epi_m, int epi_n, int load_iteration, bool is_producer_load_needed) {
      if (!params.row_broadcast) {
        fill(tCrRow, *(params.ptr_row));
        return;
      }

      if (epi_m == 0) { // Assumes M-major subtile loop
        // Filter so we don't issue redundant copies over stride-0 modes
        // (only works if 0-strides are in same location, which is by construction)
        int bcast_pipe_index = (load_iteration / EpiTiles) % Stages;
        copy_aligned(filter(tCsRow(_,_,_,epi_m,epi_n,bcast_pipe_index)), filter(tCrRow));
      }
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<Element, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<Element, FragmentSize> frg_row;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        frg_row[i] = tCrRow(epi_v * FragmentSize + i);
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

    Tensor sRow = make_tensor(make_smem_ptr(smem_row),                                            // (CTA_M,CTA_N,PIPE)
                    make_shape(size<0>(CtaTileShapeMNK{}), size<1>(CtaTileShapeMNK{}), Stages),
                    make_stride(_0{},_1{},size<1>(CtaTileShapeMNK{})));
    Tensor tCsRow = sm90_partition_for_epilogue<ReferenceSrc>(                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N,PIPE)
                      sRow, args.epi_tile, args.tiled_copy, args.thread_idx);
    Tensor tCrRow = make_tensor_like(take<0,3>(tCsRow));                                           // (CPY,CPY_M,CPY_N)

    constexpr int EpiTiles = decltype(size<1>(zipped_divide(make_layout(take<0,2>(args.tile_shape_mnk)), args.epi_tile)))::value;
    return ConsumerStoreCallbacks<EpiTiles, decltype(tCrRow), decltype(tCsRow)>(
      cute::move(tCrRow), cute::move(tCsRow), params);
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

  template<class GTensor, class RTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(GTensor&& tCgCol, RTensor&& tCrCol, Params const& params)
      : tCgCol(cute::forward<GTensor>(tCgCol)),
        tCrCol(cute::forward<RTensor>(tCrCol)),
        params(params) {}

    GTensor tCgCol;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    RTensor tCrCol;                                                                    // (CPY,CPY_M,CPY_N,EPI_M,EPI_N)
    Params const& params;

    CUTLASS_DEVICE void
    begin() {
      if (!params.col_broadcast) {
        fill(tCrCol, *(params.ptr_col));
        return;
      }

      // Filter so we don't issue redundant copies over stride-0 modes
      // (only works if 0-strides are in same location, which is by construction)
      copy_aligned(filter(tCgCol), filter(tCrCol));
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

    return ConsumerStoreCallbacks<decltype(tCgCol), decltype(tCrCol)>(
      cute::move(tCgCol), cute::move(tCrCol), params);
  }
};

}
