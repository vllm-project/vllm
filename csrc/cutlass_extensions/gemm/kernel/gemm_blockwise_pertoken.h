/***************************************************************************************************
 * Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights
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

/*
 * Adapted from CUTLASS v4.7.1
 *   include/cutlass/gemm/kernel/gemm_universal_blockwise.h
 *   include/cutlass/gemm/kernel/gemm_blockwise.h (Sm89 specialization)
 *
 * Change: scales are passed as BlockwiseScaleArgs (pointer + strides), and
 * problem_size.m() is passed to the per-token mainloop. Split-K is not
 * supported (the K-block index used for scaling assumes one K split).
 */

#pragma once

#include "cutlass/arch/arch.h"
#include "cutlass/cutlass.h"
#include "cutlass/epilogue/threadblock/default_epilogue_tensor_op.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"

#include "../threadblock/mma_multistage_blockwise_pertoken.h"

namespace cutlass {
namespace gemm {
namespace kernel {

template <typename Mma_, typename Epilogue_, typename ThreadblockSwizzle_>
struct GemmUniversalBlockwisePerToken {
  using Mma = Mma_;
  using Epilogue = Epilogue_;
  using OutputOp = typename Epilogue::OutputOp;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  using ScaleArgs = cutlass::gemm::threadblock::BlockwiseScaleArgs;

  using WarpCount = typename Mma::WarpCount;
  static int const kThreadCount = 32 * WarpCount::kCount;

  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;
    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorA::TensorRef ref_A;
    typename Mma::IteratorB::Params params_B;
    typename Mma::IteratorB::TensorRef ref_B;
    typename Epilogue::OutputTileIterator::Params params_C;
    typename Epilogue::OutputTileIterator::TensorRef ref_C;
    typename Epilogue::OutputTileIterator::Params params_D;
    typename Epilogue::OutputTileIterator::TensorRef ref_D;
    typename OutputOp::Params output_op;
    ScaleArgs scales;

    CUTLASS_HOST_DEVICE
    Params() : swizzle_log_tile(0) {}

    CUTLASS_HOST_DEVICE
    Params(cutlass::gemm::GemmCoord const& problem_size,
           cutlass::gemm::GemmCoord const& grid_tiled_shape,
           typename Mma::IteratorA::TensorRef ref_A,
           typename Mma::IteratorB::TensorRef ref_B,
           typename Epilogue::OutputTileIterator::TensorRef ref_C,
           typename Epilogue::OutputTileIterator::TensorRef ref_D,
           ScaleArgs const& scales,
           typename OutputOp::Params output_op = typename OutputOp::Params())
        : problem_size(problem_size),
          grid_tiled_shape(grid_tiled_shape),
          swizzle_log_tile(ThreadblockSwizzle().get_log_tile(grid_tiled_shape)),
          params_A(ref_A.layout()),
          ref_A(ref_A),
          params_B(ref_B.layout()),
          ref_B(ref_B),
          params_C(ref_C.layout()),
          ref_C(ref_C),
          params_D(ref_D.layout()),
          ref_D(ref_D),
          output_op(output_op),
          scales(scales) {}
  };

  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    typename Epilogue::SharedStorage epilogue;
  };

  CUTLASS_HOST_DEVICE
  GemmUniversalBlockwisePerToken() {}

  CUTLASS_DEVICE
  void operator()(Params const& params, SharedStorage& shared_storage) {
    ThreadblockSwizzle threadblock_swizzle;

    cutlass::gemm::GemmCoord threadblock_tile_offset =
        threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    if (params.grid_tiled_shape.m() <= threadblock_tile_offset.m() ||
        params.grid_tiled_shape.n() <= threadblock_tile_offset.n()) {
      return;
    }

    cutlass::MatrixCoord tb_offset_A{
        threadblock_tile_offset.m() * Mma::Shape::kM, 0};
    cutlass::MatrixCoord tb_offset_B{
        0, threadblock_tile_offset.n() * Mma::Shape::kN};

    int const problem_size_k = params.problem_size.k();
    int gemm_k_iterations =
        (problem_size_k + Mma::Shape::kK - 1) / Mma::Shape::kK;

    int thread_idx = threadIdx.x;

    typename Mma::IteratorA iterator_A(
        params.params_A, params.ref_A.data(),
        {params.problem_size.m(), problem_size_k}, thread_idx, tb_offset_A);

    typename Mma::IteratorB iterator_B(
        params.params_B, params.ref_B.data(),
        {problem_size_k, params.problem_size.n()}, thread_idx, tb_offset_B);

    int warp_idx = canonical_warp_idx_sync();
    int lane_idx = threadIdx.x % 32;

    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);

    typename Mma::FragmentC accumulators;
    accumulators.clear();

    mma(gemm_k_iterations, accumulators, iterator_A, iterator_B, accumulators,
        params.scales, threadblock_tile_offset.m(), threadblock_tile_offset.n(),
        params.problem_size.m());

    OutputOp output_op(params.output_op);

    cutlass::MatrixCoord threadblock_offset(
        threadblock_tile_offset.m() * Mma::Shape::kM,
        threadblock_tile_offset.n() * Mma::Shape::kN);

    typename Epilogue::OutputTileIterator iterator_C(
        params.params_C, params.ref_C.data(), params.problem_size.mn(),
        thread_idx, threadblock_offset);

    typename Epilogue::OutputTileIterator iterator_D(
        params.params_D, params.ref_D.data(), params.problem_size.mn(),
        thread_idx, threadblock_offset);

    Epilogue epilogue(shared_storage.epilogue, thread_idx, warp_idx, lane_idx);

    epilogue(output_op, iterator_D, accumulators, iterator_C);
  }
};

/// Assembles mainloop + epilogue + kernel for SM89 FP8 with per-token A
/// scales and 128x128 B scales. Output is row-major.
template <typename ElementA, typename LayoutA, int kAlignmentA,
          typename ElementB, typename LayoutB, int kAlignmentB,
          typename ElementC, typename ElementAccumulator,
          typename ThreadblockShape, typename WarpShape,
          typename InstructionShape, typename EpilogueOutputOp,
          typename ThreadblockSwizzle, int Stages, typename Operator>
struct GemmBlockwisePerTokenSm89 {
  using Mma = typename cutlass::gemm::threadblock::DefaultMmaBlockwisePerToken<
      ElementA, LayoutA, kAlignmentA, ElementB, LayoutB, kAlignmentB,
      ElementAccumulator, layout::RowMajor, ThreadblockShape, WarpShape,
      InstructionShape, Stages, Operator>::ThreadblockMma;

  static_assert(ThreadblockShape::kK % WarpShape::kK == 0,
                "ThreadblockShape::kK must be divisible by WarpShape::kK.");
  static int const kPartitionsK = ThreadblockShape::kK / WarpShape::kK;

  using Epilogue =
      typename cutlass::epilogue::threadblock::DefaultEpilogueTensorOp<
          ThreadblockShape, typename Mma::Operator, kPartitionsK,
          EpilogueOutputOp, EpilogueOutputOp::kCount>::Epilogue;

  using GemmKernel =
      GemmUniversalBlockwisePerToken<Mma, Epilogue, ThreadblockSwizzle>;
};

}  // namespace kernel
}  // namespace gemm
}  // namespace cutlass
