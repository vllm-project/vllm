/***************************************************************************************************
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 *modification, are not permit- ted.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT,
 *INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 *DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 *OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 *NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 *EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#pragma once

#include "kernel_params.h"
#include "dtype.h"
// #include "gen_ctx.h"

namespace gemm {

struct GemmOptions {
  // The all-reduce algorithm.
  AllReduceAlgo mAllReduceAlgo{AllReduceAlgo::None};
  // Relative error tolerance.
  float mAtol{1e-4f};
  // Whether to verify the correctness. 0: No check, 1: full.
  int mChecksResults{1};
  // Cluster size in X dim.
  int mClusterX{1};
  // Cluster size in Y dim.
  int mClusterY{1};
  // Data type of the accumulators.
  Dtype mDtypeAcc{Dtype::Fp32};
  // Data type of the inputs.
  Dtype mDtypeElt{Dtype::Fp16};
  // Data type of the outputs.
  Dtype mDtypeC{Dtype::Void};
  // Tile size for the epilogue in M dimension.
  int mEpilogueTileM{128};
  // Tile size for the epilogue in N dimension.
  int mEpilogueTileN{32};
  // The K dimension of GEMM.
  int mK{16 * 8};
  // The M dimension of GEMM.
  int mM{128 * 2};
  // Size of the MMA instruction in the K dimension.
  int mMmaK{16};
  // Size of the MMA instruction in the M dimension.
  int mMmaM{64};
  // Size of the MMA instruction in the N dimension.
  int mMmaN{16};
  // The N dimension of GEMM.
  int mN{64 * 4};
  // Benchmark steps.
  int mNumBenchmarkSteps{1};
  // The depth of the mainloop pipeline.
  int mNumStages{2};
  // Warmup steps.
  int mNumWarmUpSteps{0};
  // Whether to output debug tensors.
  bool mOutputDebugTensors{false};
  // Relative error tolerance.
  float mRtol{1e-4f};
  // Scale of the output before quantization to fp8. Ignored for the other IO
  // datatypes. For FP8, the default value remaps from [-448.f, 448.f] to
  // [-1.f, 1.f].
  float mScaleC{1.f / 448.f};
  // Number of partitions along K dimension. When mNumSlicesForSplitK > 1,
  // the problem is distributed across several SMs, where each CTA works on its
  // local K slice. Partial results are accumulated afterwards.
  int mNumSlicesForSplitK{1};
  // Reorder rows/cols in the A matrix for the better memory accesses in the
  // M-major epilogue.
  bool mUseShuffledMatrixA{false};
  // Whether to skip kernel generation (for debug purpose).
  bool mSkipsKernelGen{false};
  // Save output of MMA in M-major format.
  bool mTransposeMmaOutput{false};
  // M tile dimension of GEMM.
  int mTileM{128};
  // N tile dimension of GEMM.
  int mTileN{32};
  // K tile dimension of GEMM.
  int mTileK{16};
  // Units in the last place tolerance for e4m3 verification.
  int mUlpTol{0};
  // Use TMA to store the result.
  bool mUseTmaStore{true};
  // Use two different warps for A and B matrix load.
  bool mUseTwoTmaLoadWarps{false};
  // Level of verbose information.
  int mVerbosity{1};
};

}  // namespace gemm
