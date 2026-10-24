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
 *   include/cutlass/gemm/threadblock/mma_multistage_blockwise.h
 *   include/cutlass/gemm/threadblock/default_mma_multistage_blockwise.h
 *
 * Change: CUTLASS's Ada blockwise mainloop applies one A scale per 128x128
 * block of A. vLLM's block-FP8 activations are quantized per token group
 * (1x128), so A has one scale per row per 128-wide K block. This mainloop
 * applies scale_A[row, kb] * scale_B[n_blk, kb] per accumulator row.
 *
 * The scales are constant within a 128-wide K block, so each block's MMAs
 * accumulate unscaled into a block fragment, which is scaled once and added
 * to the main accumulator. This keeps two accumulator fragments live (not
 * three) and does one scale per element per K block instead of one per
 * 32-wide k-group, which avoids register spills on 4-warp tiles.
 * The N-side scale block is looked up per warp, so N tiles wider than 128
 * stay correct as long as a warp's N range sits inside one scale block.
 *
 * Scales are fp32 and passed as pointer + strides (any layout):
 *   scale_A[m, kb]     at ptr_a[m * a_stride_m + kb * a_stride_k]   (1x128
 * groups) scale_B[nb, kb]    at ptr_b[nb * b_stride_n + kb * b_stride_k]
 * (128x128 blocks) vLLM passes activation scales column-major (a_stride_m = 1).
 */

#pragma once

#include "cutlass/aligned_buffer.h"
#include "cutlass/arch/memory.h"
#include "cutlass/array.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/threadblock/default_mma_core_sm80.h"
#include "cutlass/gemm/threadblock/mma_base.h"
#include "cutlass/gemm/threadblock/mma_multistage.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_shape.h"
#include "cutlass/numeric_types.h"
#include "cutlass/transform/threadblock/predicated_tile_access_iterator.h"

namespace cutlass {
namespace gemm {
namespace threadblock {

/// Where the block scales live. Strides are in elements.
struct BlockwiseScaleArgs {
  float const* ptr_a = nullptr;
  int64_t a_stride_m = 0;
  int64_t a_stride_k = 0;
  float const* ptr_b = nullptr;
  int64_t b_stride_n = 0;
  int64_t b_stride_k = 0;
  int num_k_blocks = 0;  // ceil(K / 128)
};

template <typename Shape_, typename IteratorA_, typename SmemIteratorA_,
          cutlass::arch::CacheOperation::Kind CacheOpA, typename IteratorB_,
          typename SmemIteratorB_, cutlass::arch::CacheOperation::Kind CacheOpB,
          typename ElementC_, typename LayoutC_, typename Policy_, int Stages,
          SharedMemoryClearOption SharedMemoryClear =
              SharedMemoryClearOption::kNone,
          typename Enable = bool>
class MmaMultistageBlockwisePerToken
    : public MmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA,
                           IteratorB_, SmemIteratorB_, CacheOpB, ElementC_,
                           LayoutC_, Policy_, Stages, SharedMemoryClear,
                           Enable> {
 public:
  using Base =
      MmaMultistage<Shape_, IteratorA_, SmemIteratorA_, CacheOpA, IteratorB_,
                    SmemIteratorB_, CacheOpB, ElementC_, LayoutC_, Policy_,
                    Stages, SharedMemoryClear, Enable>;
  using Shape = Shape_;
  using IteratorA = IteratorA_;
  using IteratorB = IteratorB_;
  using ElementC = ElementC_;
  using LayoutC = LayoutC_;
  using Policy = Policy_;
  using SmemIteratorA = SmemIteratorA_;
  using SmemIteratorB = SmemIteratorB_;

  static cutlass::arch::CacheOperation::Kind const kCacheOpA = CacheOpA;
  static cutlass::arch::CacheOperation::Kind const kCacheOpB = CacheOpB;

  using FragmentC = typename Policy::Operator::FragmentC;
  using Operator = typename Policy::Operator;
  using ArchTag = arch::Sm80;
  static ComplexTransform const kTransformA = Operator::kTransformA;
  static ComplexTransform const kTransformB = Operator::kTransformB;

  using Detail = typename Base::Detail;

  using Base::advance_smem_read_stage;
  using Base::advance_smem_write_stage;
  using Base::copy_tiles_and_advance;
  using Base::gmem_wait;
  using Base::prologue;
  using Base::wind_down;

  // ---- per-row scaling geometry ------------------------------------------
  static constexpr int kScaleBlock = 128;

  using WarpShape = typename Operator::Shape;
  using InstructionShape = typename Operator::InstructionShape;

  // Number of 16x8 MMA tiles a warp covers along M and N.
  static constexpr int kMmaRows = WarpShape::kM / InstructionShape::kM;
  static constexpr int kMmaCols = WarpShape::kN / InstructionShape::kN;

  // Each thread owns 2 rows (lane/4 and lane/4 + 8) per 16-row MMA tile.
  static constexpr int kRowsPerThread = 2 * kMmaRows;

  static_assert(InstructionShape::kM == 16 && InstructionShape::kN == 8,
                "per-row scaling assumes the 16x8xK tensor op accumulator "
                "layout");
  static_assert(FragmentC::kElements == kMmaRows * kMmaCols * 4,
                "unexpected accumulator fragment size");
  static_assert(Shape::kK == kScaleBlock,
                "one mainloop iteration must cover exactly one 128-wide K "
                "scale block");
  static_assert(Shape::kN % kScaleBlock == 0 || kScaleBlock % Shape::kN == 0,
                "threadblock N tile must not straddle a 128-wide N scale "
                "block");
  // The B scale is looked up per warp, so a warp's N range must sit inside
  // one 128-wide N scale block (tiles wider than 128 span several blocks).
  static_assert(kScaleBlock % WarpShape::kN == 0,
                "warp N tile must divide the 128-wide N scale block");

 private:
  using PipeState = typename Base::PublicPipeState;

  Operator warp_mma_;
  int lane_idx_;
  int warp_idx_m_;
  int warp_idx_n_;

 public:
  CUTLASS_DEVICE
  MmaMultistageBlockwisePerToken(typename Base::SharedStorage& shared_storage,
                                 int thread_idx, int warp_idx, int lane_idx)
      : Base(shared_storage, thread_idx, warp_idx, lane_idx),
        lane_idx_(lane_idx) {
    // Same warp -> (m, n) mapping as MmaMultistage's constructor.
    int warp_idx_mn = warp_idx % (Base::WarpCount::kM * Base::WarpCount::kN);
    warp_idx_m_ = warp_idx_mn % Base::WarpCount::kM;
    warp_idx_n_ = warp_idx_mn / Base::WarpCount::kM;
  }

  /// scale_A[row, kb] * scale_B[n block, kb] for the rows this thread owns
  /// (0 for rows past M).
  CUTLASS_DEVICE
  void load_row_scales(float (&dst)[kRowsPerThread], int k_block_idx,
                       BlockwiseScaleArgs const& scales, int warp_row0,
                       int problem_m, int block_n_idx) const {
    float const scale_b = scales.ptr_b[block_n_idx * scales.b_stride_n +
                                       k_block_idx * scales.b_stride_k];
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < kMmaRows; ++i) {
      CUTLASS_PRAGMA_UNROLL
      for (int h = 0; h < 2; ++h) {
        int const row = warp_row0 + i * InstructionShape::kM + h * 8;
        dst[2 * i + h] = row < problem_m
                             ? scales.ptr_a[row * scales.a_stride_m +
                                            k_block_idx * scales.a_stride_k] *
                                   scale_b
                             : 0.f;
      }
    }
  }

  CUTLASS_DEVICE
  void mac_loop_iter(PipeState& pipe_state, FragmentC& accum,
                     IteratorA& iterator_A, IteratorB& iterator_B,
                     int& gemm_k_iterations, BlockwiseScaleArgs const& scales,
                     int k_iter_idx, int warp_row0, int problem_m,
                     int block_n_idx) {
    // K block for this iteration. The pipeline runs Stages-1 extra drain
    // iterations, so clamp to the last valid K block (same as CUTLASS).
    int k_block_idx = k_iter_idx;
    if (k_block_idx >= scales.num_k_blocks) {
      k_block_idx = scales.num_k_blocks - 1;
    }

    float row_scale[kRowsPerThread];
    load_row_scales(row_scale, k_block_idx, scales, warp_row0, problem_m,
                    block_n_idx);

    FragmentC block_accum;
    block_accum.clear();

    CUTLASS_PRAGMA_UNROLL
    for (int warp_mma_k = 0; warp_mma_k < Base::kWarpGemmIterations;
         ++warp_mma_k) {
      this->warp_tile_iterator_A_.set_kgroup_index((warp_mma_k + 1) %
                                                   Base::kWarpGemmIterations);
      this->warp_tile_iterator_A_.load(
          pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_A_;

      this->warp_tile_iterator_B_.set_kgroup_index((warp_mma_k + 1) %
                                                   Base::kWarpGemmIterations);
      this->warp_tile_iterator_B_.load(
          pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      ++this->warp_tile_iterator_B_;

      if (warp_mma_k > 0) {
        warp_mma_.transform(pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
                            pipe_state.warp_transformed_frag_B_[warp_mma_k % 2],
                            pipe_state.warp_loaded_frag_A_[warp_mma_k % 2],
                            pipe_state.warp_loaded_frag_B_[warp_mma_k % 2]);
      }

      warp_mma_(
          block_accum, pipe_state.warp_transformed_frag_A_[warp_mma_k % 2],
          pipe_state.warp_transformed_frag_B_[warp_mma_k % 2], block_accum);

      if (warp_mma_k < Base::kWarpGemmIterations - 1) {
        int group_start_iteration_A = warp_mma_k * Detail::kAccessesPerGroupA;
        int group_start_iteration_B = warp_mma_k * Detail::kAccessesPerGroupB;
        copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                               group_start_iteration_B);
      }

      if (warp_mma_k + 2 == Base::kWarpGemmIterations) {
        int group_start_iteration_A =
            (warp_mma_k + 1) * Detail::kAccessesPerGroupA;
        int group_start_iteration_B =
            (warp_mma_k + 1) * Detail::kAccessesPerGroupB;
        copy_tiles_and_advance(iterator_A, iterator_B, group_start_iteration_A,
                               group_start_iteration_B);

        cutlass::arch::cp_async_fence();
        gmem_wait();

        advance_smem_write_stage(iterator_A, iterator_B);
        advance_smem_read_stage();

        --gemm_k_iterations;
        iterator_A.clear_mask(gemm_k_iterations == 0);
        iterator_B.clear_mask(gemm_k_iterations == 0);
      }

      if (warp_mma_k + 1 == Base::kWarpGemmIterations) {
        warp_mma_.transform(
            pipe_state.warp_transformed_frag_A_[(warp_mma_k + 1) % 2],
            pipe_state.warp_transformed_frag_B_[(warp_mma_k + 1) % 2],
            pipe_state.warp_loaded_frag_A_[(warp_mma_k + 1) % 2],
            pipe_state.warp_loaded_frag_B_[(warp_mma_k + 1) % 2]);
      }
    }

    // Per-row dequantization, once per 128-wide K block. Accumulators are
    // kMmaRows x kMmaCols MMA tiles (tile = m + n * kMmaRows), 4 floats per
    // tile: elements 0,1 -> row lane/4, elements 2,3 -> row lane/4 + 8.
    CUTLASS_PRAGMA_UNROLL
    for (int el = 0; el < FragmentC::kElements; ++el) {
      accum[el] +=
          block_accum[el] * row_scale[2 * ((el / 4) % kMmaRows) + (el % 4) / 2];
    }
  }

  CUTLASS_DEVICE
  void gemm_iters(int gemm_k_iterations, FragmentC& accum,
                  IteratorA& iterator_A, IteratorB& iterator_B,
                  BlockwiseScaleArgs const& scales, int warp_row0,
                  int problem_m, int block_n_idx) {
    PipeState pipe_state;

    iterator_A.clear_mask(gemm_k_iterations == 0);
    iterator_B.clear_mask(gemm_k_iterations == 0);

    this->warp_tile_iterator_A_.set_kgroup_index(0);
    this->warp_tile_iterator_A_.load(pipe_state.warp_loaded_frag_A_[0]);
    ++this->warp_tile_iterator_A_;

    this->warp_tile_iterator_B_.set_kgroup_index(0);
    this->warp_tile_iterator_B_.load(pipe_state.warp_loaded_frag_B_[0]);
    ++this->warp_tile_iterator_B_;

    warp_mma_.transform(pipe_state.warp_transformed_frag_A_[0],
                        pipe_state.warp_transformed_frag_B_[0],
                        pipe_state.warp_loaded_frag_A_[0],
                        pipe_state.warp_loaded_frag_B_[0]);

    int k_iter_idx = 0;

    CUTLASS_GEMM_LOOP
    for (; gemm_k_iterations > (-Base::kStages + 1); ++k_iter_idx) {
      mac_loop_iter(pipe_state, accum, iterator_A, iterator_B,
                    gemm_k_iterations, scales, k_iter_idx, warp_row0, problem_m,
                    block_n_idx);
    }

    cutlass::arch::cp_async_fence();
    cutlass::arch::cp_async_wait<0>();
    __syncthreads();
  }

  /// Threadblock-scoped GEMM. Differs from CUTLASS's version by taking the
  /// scales as BlockwiseScaleArgs and the problem's M (to bound row reads).
  CUTLASS_DEVICE
  void operator()(int gemm_k_iterations, FragmentC& accum, IteratorA iterator_A,
                  IteratorB iterator_B, FragmentC const& src_accum,
                  BlockwiseScaleArgs const& scales, int threadblock_tile_m,
                  int threadblock_tile_n, int problem_m) {
    // First output row owned by this thread (for MMA tile 0, half 0).
    int const warp_row0 = threadblock_tile_m * Shape::kM +
                          warp_idx_m_ * WarpShape::kM + (lane_idx_ >> 2);

    // N scale block of this warp's columns (per warp: a 256-wide tile spans
    // two 128-wide scale blocks).
    int const block_n_idx =
        (threadblock_tile_n * Shape::kN + warp_idx_n_ * WarpShape::kN) /
        kScaleBlock;

    prologue(iterator_A, iterator_B, gemm_k_iterations);
    gmem_wait();

    accum = src_accum;

    gemm_iters(gemm_k_iterations, accum, iterator_A, iterator_B, scales,
               warp_row0, problem_m, block_n_idx);
  }
};

/// Builds the threadblock mainloop (same iterators as CUTLASS's
/// DefaultMmaBlockwise, tensor-op path only).
template <
    typename ElementA, typename LayoutA, int kAlignmentA, typename ElementB,
    typename LayoutB, int kAlignmentB, typename ElementAccumulator,
    typename LayoutC, typename ThreadblockShape, typename WarpShape,
    typename InstructionShape, int Stages, typename Operator,
    SharedMemoryClearOption SharedMemoryClear = SharedMemoryClearOption::kNone>
struct DefaultMmaBlockwisePerToken {
  static cutlass::arch::CacheOperation::Kind const CacheOpA =
      ((sizeof_bits<ElementA>::value * kAlignmentA) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  static cutlass::arch::CacheOperation::Kind const CacheOpB =
      ((sizeof_bits<ElementB>::value * kAlignmentB) == 128)
          ? cutlass::arch::CacheOperation::Global
          : cutlass::arch::CacheOperation::Always;

  using MmaCore = typename cutlass::gemm::threadblock::DefaultMmaCore<
      ThreadblockShape, WarpShape, InstructionShape, ElementA, LayoutA,
      ElementB, LayoutB, ElementAccumulator, LayoutC, arch::OpClassTensorOp,
      Stages, Operator, false, CacheOpA, CacheOpB>;

  using ThreadMapA = typename MmaCore::IteratorThreadMapA;
  using AccessTypeA = cutlass::Array<ElementA, kAlignmentA>;
  using IteratorA =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kM, ThreadblockShape::kK>,
          ElementA, LayoutA, 1, ThreadMapA, AccessTypeA>;

  using ThreadMapB = typename MmaCore::IteratorThreadMapB;
  using AccessTypeB = cutlass::Array<ElementB, kAlignmentB>;
  using IteratorB =
      cutlass::transform::threadblock::PredicatedTileAccessIterator<
          cutlass::MatrixShape<ThreadblockShape::kK, ThreadblockShape::kN>,
          ElementB, LayoutB, 0, ThreadMapB, AccessTypeB>;

  using ThreadblockMma = MmaMultistageBlockwisePerToken<
      typename MmaCore::Shape, IteratorA, typename MmaCore::SmemIteratorA,
      MmaCore::kCacheOpA, IteratorB, typename MmaCore::SmemIteratorB,
      MmaCore::kCacheOpB, ElementAccumulator, LayoutC,
      typename MmaCore::MmaPolicy, Stages, SharedMemoryClear>;
};

}  // namespace threadblock
}  // namespace gemm
}  // namespace cutlass
