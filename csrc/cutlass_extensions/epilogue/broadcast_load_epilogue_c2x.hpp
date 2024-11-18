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
// include/cutlass/epilogue/fusion/visitor_load.hpp from
// https://github.com/NVIDIA/cutlass v3.5.0
// It has been modified to support either
// row/column or scalar broadcasting where the tensor being loaded from is
// always passed in via a device pointer. This lets one compiled kernel handle
// all cases of per-tensor or per-channel/per-token quantization.
//
// This interface also allows the scales to be passed in as tensors that
// consistently reside on the device, which avoids an issue with a previous
// implementation where scalars needed to be on the CPU since they
// were passed in via float values. This created a potential performance hazard
// if scales were initially on the device, and caused torch.compile graph
// breaks when moving scales to the CPU.
//
#pragma once

// Turn off clang-format for the entire file to keep it close to upstream
// clang-format off

#include "cutlass/epilogue/threadblock/fusion/visitor_2x.hpp"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cute/tensor.hpp"

namespace cutlass::epilogue::threadblock {

using namespace cute;
using namespace detail;

template<
  class ThreadMap,
  class Element,
  class StrideMNL
>
struct VisitorRowOrScalarBroadcast {

  // This struct has been modified to have a bool indicating that ptr_row is a 
  // scalar that must be broadcast.
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

  struct SharedStorage {};

  // Global load type
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorRowOrScalarBroadcast() { }

  CUTLASS_HOST_DEVICE
  VisitorRowOrScalarBroadcast(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gRow,
      RTensor&& tC_rRow,
      CTensor&& tC_cRow,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gRow(cute::forward<GTensor>(tC_gRow)),
      tC_rRow(cute::forward<RTensor>(tC_rRow)),
      tC_cRow(cute::forward<CTensor>(tC_cRow)),
      n(get<1>(problem_shape)),
      params_ptr(params_ptr) { }

    GTensor tC_gRow;
    RTensor tC_rRow;
    CTensor tC_cRow;
    Params const* params_ptr;
    int n;

    // This function is modified from VisitorRowBroadcast
    CUTLASS_DEVICE void
    begin_epilogue() {
      clear(tC_rRow);
      auto src_v = filter(tC_gRow);
      auto coord_v = filter(tC_cRow);
      auto dst_v = filter(tC_rRow);

      if (params_ptr->row_broadcast) {
        // In this case we are loading from a row vector and broadcasting
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          bool guard = get<1>(coord_v(i)) < n;
          cutlass::arch::global_load<VecType, sizeof(VecType)>(
              dst_v(i), (void const*)&src_v(i), guard);
        }
      } else {
        // In this case we are loading from a scalar and broadcasting
        VecType filled_vec;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < VecLength; i++) {
          reinterpret_cast<Element*>(&filled_vec)[i] = *(params_ptr->ptr_row);
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          if (get<1>(coord_v(i)) < n) {
            dst_v(i) = filled_vec;
          }
        }
      }
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Tensor rRow_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rRow));
      return rRow_frg(column_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mRow = make_tensor(
      make_gmem_ptr(params_ptr->ptr_row),
      problem_shape,
      params_ptr->dRow);

    // VECTOR, FRAGMENT_COLUMN
    Tensor tC_gRow = recast<VecType>(
      ThreadMap::partition(mRow, thread_idx, threadblock_tile_offset)
    )(_,_,_0{},_0{},_0{},_0{});
    Tensor tC_rRow = make_tensor_like(tC_gRow);

    // Generate the pred tensor
    Tensor cRow = make_identity_tensor(mRow.shape());
    Tensor tC_cRow = outer_partition(
      ThreadMap::partition(cRow, thread_idx, threadblock_tile_offset)(_,_,_0{},_0{},_0{},_0{}),
      Shape<Int<VecLength>>{},
      (_0{})
    );

    return Callbacks<
      decltype(tC_gRow), decltype(tC_rRow),
      decltype(tC_cRow), ProblemShape>(
      cute::move(tC_gRow),
      cute::move(tC_rRow),
      cute::move(tC_cRow),
      problem_shape,
      params_ptr
    );
  }

};

/////////////////////////////////////////////////////////////////////////////////////////////////

// This is a modified RowBroadcast that will broadcast 0 if ptr_row is null
template<
  class ThreadMap,
  class Element,
  class StrideMNL
>
struct VisitorRowOrZeroBroadcast {

  // This struct has been modified to remove null_default (because it's always 0)
  struct Arguments {
    Element const* ptr_row = nullptr;
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

  struct SharedStorage {};

  // Global load type
  static int constexpr vec_bits = ThreadMap::kElementsPerAccess * sizeof_bits<Element>::value;
  using VecType = uint_bit_t<cute::min(128, vec_bits)>;
  static int constexpr VecLength = sizeof(VecType) / sizeof(Element);

  CUTLASS_HOST_DEVICE
  VisitorRowOrZeroBroadcast() { }

  CUTLASS_HOST_DEVICE
  VisitorRowOrZeroBroadcast(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gRow,
      RTensor&& tC_rRow,
      CTensor&& tC_cRow,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gRow(cute::forward<GTensor>(tC_gRow)),
      tC_rRow(cute::forward<RTensor>(tC_rRow)),
      tC_cRow(cute::forward<CTensor>(tC_cRow)),
      n(get<1>(problem_shape)),
      params_ptr(params_ptr) { }

    GTensor tC_gRow;
    RTensor tC_rRow;
    CTensor tC_cRow;
    Params const* params_ptr;
    int n;

    // This function is modified from VisitorRowBroadcast
    CUTLASS_DEVICE void
    begin_epilogue() {
      clear(tC_rRow);
      auto src_v = filter(tC_gRow);
      auto coord_v = filter(tC_cRow);
      auto dst_v = filter(tC_rRow);

      if (params_ptr->ptr_row != nullptr) {
        // In this case we are loading from a row vector and broadcasting
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          bool guard = get<1>(coord_v(i)) < n;
          cutlass::arch::global_load<VecType, sizeof(VecType)>(
              dst_v(i), (void const*)&src_v(i), guard);
        }
      } else {
        // In this case we are broadcasting 0
        VecType filled_vec;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < VecLength; i++) {
          reinterpret_cast<Element*>(&filled_vec)[i] = Element{0};
        }

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(src_v); ++i) {
          if (get<1>(coord_v(i)) < n) {
            dst_v(i) = filled_vec;
          }
        }
      }
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Tensor rRow_frg = recast<Array<Element, FragmentSize>>(coalesce(tC_rRow));
      return rRow_frg(column_idx);
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mRow = make_tensor(
      make_gmem_ptr(params_ptr->ptr_row),
      problem_shape,
      params_ptr->dRow);

    // VECTOR, FRAGMENT_COLUMN
    Tensor tC_gRow = recast<VecType>(
      ThreadMap::partition(mRow, thread_idx, threadblock_tile_offset)
    )(_,_,_0{},_0{},_0{},_0{});
    Tensor tC_rRow = make_tensor_like(tC_gRow);

    // Generate the pred tensor
    Tensor cRow = make_identity_tensor(mRow.shape());
    Tensor tC_cRow = outer_partition(
      ThreadMap::partition(cRow, thread_idx, threadblock_tile_offset)(_,_,_0{},_0{},_0{},_0{}),
      Shape<Int<VecLength>>{},
      (_0{})
    );

    return Callbacks<
      decltype(tC_gRow), decltype(tC_rRow),
      decltype(tC_cRow), ProblemShape>(
      cute::move(tC_gRow),
      cute::move(tC_rRow),
      cute::move(tC_cRow),
      problem_shape,
      params_ptr
    );
  }

};


/////////////////////////////////////////////////////////////////////////////////////////////////

// Column vector broadcast
template<
  class ThreadMap,
  class Element,
  class StrideMNL = Stride<_1,_0,_0>
>
struct VisitorColOrScalarBroadcast {

  // This struct has been modified to have a bool indicating that ptr_col is a
  // scalar that must be broadcast.
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

  struct SharedStorage { };

  CUTLASS_HOST_DEVICE
  VisitorColOrScalarBroadcast() { }

  CUTLASS_HOST_DEVICE
  VisitorColOrScalarBroadcast(Params const& params, SharedStorage const& shared_storage)
    : params_ptr(&params) { }

  Params const* params_ptr;

  template <class GTensor, class RTensor, class CTensor, class ProblemShape>
  struct Callbacks : EmptyCallbacks {
    CUTLASS_DEVICE
    Callbacks(
      GTensor&& tC_gCol,
      RTensor&& tC_rCol,
      CTensor&& tC_cCol,
      ProblemShape problem_shape,
      Params const* params_ptr
    ):
      tC_gCol(cute::forward<GTensor>(tC_gCol)),
      tC_rCol(cute::forward<RTensor>(tC_rCol)),
      tC_cCol(cute::forward<CTensor>(tC_cCol)),
      m(get<0>(problem_shape)),
      params_ptr(params_ptr) { }

    GTensor tC_gCol;
    RTensor tC_rCol;
    CTensor tC_cCol;
    Params const* params_ptr;
    int m;

    // This function is modified from VisitorColBroadcast
    CUTLASS_DEVICE void 
    begin_epilogue() {
      clear(tC_rCol);

      Tensor pred = make_tensor<bool>(shape(tC_gCol));
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(pred); ++i) {
        pred(i) = get<0>(tC_cCol(i)) < m;
      }

      if (params_ptr->col_broadcast) {
        // In this case we are loading from a column vector and broadcasting
        copy_if(pred, tC_gCol, tC_rCol);
      } else {
        // In this case we are loading from a scalar and broadcasting
        auto dst_v = filter(tC_rCol);

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < size(dst_v); ++i) {
          if (pred(i)) {
            dst_v(i) = *(params_ptr->ptr_col);
          }
        }
      }
    }

    template <class ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE auto // returns an Array
    visit(int iter_idx, int row_idx, int column_idx, int frg_idx,
          Array<ElementAccumulator, FragmentSize> const& frg_acc) {
      Array<Element, FragmentSize> frg_col;
      frg_col.fill(tC_rCol(row_idx,iter_idx));
      return frg_col;
    }
  };

  template <class ProblemShape>
  CUTLASS_DEVICE auto
  get_callbacks(
    gemm::GemmCoord threadblock_tile_offset,
    int thread_idx,
    ProblemShape problem_shape
  ) {
    Tensor mCol = make_tensor(
      make_gmem_ptr(params_ptr->ptr_col),
      problem_shape,
      params_ptr->dCol);

    // VECTOR, FRAGMENT_COLUMN, FRAGMENT_ROW, ITERATION_ROW, ITERATION_GROUP, ITERATION_CLUSTER
    Tensor tC_gCol = group_modes<1,4>(
      ThreadMap::partition(mCol, thread_idx, threadblock_tile_offset)(_0{},_0{},_,_,_,_));
    Tensor tC_rCol = make_tensor_like(tC_gCol);

    // Generate the pred tensor
    Tensor cCol = make_identity_tensor(mCol.shape());
    Tensor tC_cCol = group_modes<1,4>(
      ThreadMap::partition(cCol, thread_idx, threadblock_tile_offset)(_0{},_0{},_,_,_,_));

    return Callbacks<
      decltype(tC_gCol), decltype(tC_rCol),
      decltype(tC_cCol), ProblemShape>(
      cute::move(tC_gCol),
      cute::move(tC_rCol),
      cute::move(tC_cCol),
      problem_shape,
      params_ptr
    );
  }
};

}
