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

// #include "tma_descriptor.h"
#include <cuda.h>
#include "dtype.h"
namespace gemm {

enum class AllReduceAlgo : uint32_t {
  // Does not apply all-reduce.
  None = 0,
  // Reduction occurs at L2 cache; pulls N-1 partial outputs from peer devices.
  // Result is
  // non-deterministic. Potentially lower latency at cost of higher memory
  // traffic.
  OneShot,
  // Reduction occurs at switch; pulls 1/Nth of the output from switch
  // (reduce-scatter phase) and
  // store to multicast mem (all-gather phase). Result is deterministic. Lower
  // memory traffic at
  // cost of potentially higher latency.
  TwoShot,
};

inline CUtensorMap buildNdTmaDescriptor(Dtype dtype,
                                        std::vector<uint64_t> const& shapes,
                                        std::vector<uint64_t> const& strides,
                                        int32_t tileSizeMn, int32_t tileSizeK,
                                        void* gmemAddr) {
  CUtensorMap desc{};
  // The data type.
  CUtensorMapDataType tmaDataFormat;
  if (dtype == Dtype::E4m3) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  } else if (dtype == Dtype::Fp16) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if (dtype == Dtype::Bfloat16) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  }
  // else if (dtype == Dtype::E2m1) {
  //   tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
  // }
  else if (dtype == Dtype::Fp32) {
    tmaDataFormat = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    std::cerr << "Unexepected dtype " << static_cast<int32_t>(dtype)
              << std::endl;
    assert(false);
  }

  // The swizzle type.
  CUtensorMapSwizzle swizzleType;
  int32_t tileKSizeInBytes =
      (tileSizeK * dtypeGetNumBits(dtype)) / /* bits */ 8;
  if ((tileKSizeInBytes % 128) == 0) {
    swizzleType = CU_TENSOR_MAP_SWIZZLE_128B;
  } else if ((tileKSizeInBytes % 64) == 0) {
    swizzleType = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if ((tileKSizeInBytes % 32) == 0) {
    swizzleType = CU_TENSOR_MAP_SWIZZLE_32B;
  } else {
    std::cerr << "Unexepected tileKSizeInBytes " << tileKSizeInBytes
              << std::endl;
    assert(false);
  }

  // Check gmem address must be 16B-aligned
  assert((reinterpret_cast<uint64_t>(gmemAddr) & 0b1111) == 0);  //

  // Check shape must be in range [1, 2^32]
  int32_t dim = shapes.size();
  // Expect 2 dimensions.
  assert(dim == 2 || dim == 3);
  // Check shape range.
  for (int32_t ii = 0; ii < dim; ++ii) {
    assert(shapes[ii] >= (uint64_t(1)));        // Size must be min 1
    assert(shapes[ii] <= (uint64_t(1) << 32));  // Size must be max 2^32
  }

  // TMA descriptor does not store the zeroth stride and assumes it is 1.
  assert(static_cast<int32_t>(strides.size()) == dim);
  assert(strides[0] == 1);

  // Build strides in bytes.
  // cuTensorMapEncodeTiled ignores the stride of the first dimention
  // (implicitly 1).
  std::vector<uint64_t> stridesInBytes(dim - 1);
  for (int32_t ii = 0; ii < dim - 1; ++ii) {
    stridesInBytes[ii] =
        (strides[ii + 1] * dtypeGetNumBits(dtype)) / /* bits */ 8;
  }

  // Set the number of elements in the packed uint32_t element.
  auto const numEltsPerUInt32 = 4 * /* bits */ 8 / dtypeGetNumBits(dtype);
  // The number of elements in 128B.
  auto const numEltsIn128B = numEltsPerUInt32 /*4B*/ * 32;
  // The number of tile K hidden size (per token) in each block of shared
  // memory.
  auto const numEltsInClampedTileKSize = std::min(numEltsIn128B, tileSizeK);

  // Build tile shapes.
  std::vector<uint32_t> tileShapes(dim, 1);
  tileShapes[0] = numEltsInClampedTileKSize;  // tileSizeK
  tileShapes[1] = tileSizeMn;                 // tileSizeMn

  // Set tile strides to 1;
  std::vector<uint32_t> tileStrides(dim, 1);

  // Build the descriptor.
  CUresult result = cuTensorMapEncodeTiled(
      &desc, tmaDataFormat,
      /*tensorRank=*/dim, gmemAddr, shapes.data(), stridesInBytes.data(),
      tileShapes.data(), tileStrides.data(),
      /*interleave=*/CU_TENSOR_MAP_INTERLEAVE_NONE, swizzleType,
      /*l2Promotion=*/CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
      /*oobFill=*/CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  if (result != CUDA_SUCCESS) {
    std::cerr << "Error: Failed to initialize the TMA descriptor " << result
              << std::endl;

    std::cerr << "tmaFormat: " << static_cast<int>(tmaDataFormat)
              << " dim: " << dim << " gmem: " << gmemAddr << std::endl;

    std::cerr << "Shape: ";
    for (int ii = 0; ii < dim; ++ii) {
      std::cerr << shapes[ii] << " ";
    }
    std::cerr << std::endl;

    std::cerr << "Stride: ";
    for (int ii = 0; ii < dim - 1; ++ii) {
      std::cerr << stridesInBytes[ii] << " ";
    }
    std::cerr << std::endl;

    std::cerr << "tileShapes: ";
    for (int ii = 0; ii < dim; ++ii) {
      std::cerr << tileShapes[ii] << " ";
    }
    std::cerr << std::endl;

    std::cerr << "tileStrides: ";
    for (int ii = 0; ii < dim; ++ii) {
      std::cerr << tileStrides[ii] << " ";
    }
    std::cerr << std::endl;
    std::cerr << "swizzleType: " << int(swizzleType) << std::endl;
    assert(false);
  }

  return desc;
}

struct KernelParams {
  //
  // Gemm parameters.
  //

  // The output matrix C. The shape is m x n. Layout is row-major (contiguous in
  // the n dimension). (when useTmaStore is false)
  void* ptrC;

  // TMA descriptor for A.
  CUtensorMap tmaA;
  // TMA descriptor for B.
  CUtensorMap tmaB;
  // TMA descriptor for C, (when useTmaStore is true)
  CUtensorMap tmaC;

  // The scaling factors to dequantize A. It is used when the DeepSeek Fp8
  // recipe is enabled.
  float const* ptrDqSfsA;
  // The scaling factors to dequantize B. It is used when the DeepSeek Fp8
  // recipe is enabled.
  float const* ptrDqSfsB;

  // The device output scale for FP8 quantization. It can either be a static
  // value passed to the kernel or it can be computed by the kernel.
  // TensorRT-LLM fp8 kernels expect a single scaling factor on the device.
  //
  // When DeepSeek FP8 recipe is used, the array is filled with dequantization
  // factors to later dequantize the C values.
  float* ptrScaleC;

  // The M dimension. It is the total number of tokens if A is the activation
  // matrix.
  int32_t m;
  // The N dimension. It is the number of output channels if B is the weight
  // matrix.
  int32_t n;
  // The K dimension. It is the hidden dimension of the input matrices.
  int32_t k;

  //
  // All-reduce parameters.
  //

  // The rank id.
  int rank;
  // The number of peer devices in tensor-parallel group.
  int tpGrpSize;
  // Pointer for output with multicast mapping. It is used by the "reduce" op
  // (LDGMC.ADD) of the two-shot reduce-scatter phase.
  void* multimemC;
  // Pointer for partial sums for split-k computation.
  void* ptrPartialSumsForSplitK;
  // Pointer for partial sums for split-k data with multicast mapping.
  // It is used by the "reduce" op (LDGMC.ADD)
  // of the two-shot reduce-scatter phase with numSlicesForSplitK > 1.
  void* multimemPartialSumsForSplitK;

  // The barriers in global memory.
  //
  // The kernel arrives on (with release ordering) the multicast mapping of the
  // barrier to broadcast amongst peer devices. It then waits (with acquire
  // ordering) on the unicast mapping of the barrier.
  //
  // Flags in global memory that sync on "entrance" of reduce-scatter phase in
  // two-shot all-reduce.
  void* ptrTileBars;
  void* multimemTileBars;

  // Flags in global memory that sync on "exit" after the all-reduce finishes.
  void* ptrCompletionBars;
  void* multimemCompletionBars;

  // The barriers in global memory for split k reduction.
  // The kernel arrives on the barrier and CtaIdx.z == 0 waits
  // on the barrier to flip to perform a reduction.
  void* ptrSplitKCompletionBars;

  //
  // Methods.
  //

  enum class MatrixType { MatrixA = 0, MatrixB };

  // Create the TMA shape/stride for A/B.
  template <class GemmOptions>
  static auto makeTmaShapeStrideAb(GemmOptions const& options,
                                   MatrixType matrixType) {
    // The outer dimension.
    auto numTokens =
        (matrixType == MatrixType::MatrixA) ? options.mM : options.mN;
    // The inner dimension.
    auto hiddenSize = options.mK;
    // The cute tensor shape for A/B: (numTokens, hiddenSize).
    // Note that TMA descriptor expects the first dimension's stride to be
    // 1, so swap the first two dimension so that the hiddenSize dimension comes
    // first.
    auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSize),
                                       static_cast<uint64_t>(numTokens)};

    // Assemble the stride (strideTokens, 1).
    // Swap the first two dimension as mentioned before.
    auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};

    return std::make_tuple(shape, stride);
  }

  // Create the TMA shape/stride for C.
  template <class GemmOptions>
  static auto makeTmaShapeStrideC(GemmOptions const& options) {
    // The number of tokens.
    auto numTokens = options.mTransposeMmaOutput ? options.mN : options.mM;
    // The hidden dimension.
    auto hiddenSize = options.mTransposeMmaOutput ? options.mM : options.mN;
    // Note that TMA descriptor expects the first dimension's stride to be
    // 1, so swap the first two dimension so that the hiddenSize dimension comes
    // first.
    auto shape = std::vector<uint64_t>{static_cast<uint64_t>(hiddenSize),
                                       static_cast<uint64_t>(numTokens)};

    // Assemble the stride (strideTensor, strideTokens, 1).
    // Swap the first two dimension as mentioned before.
    auto stride = std::vector<uint64_t>{1, static_cast<uint64_t>(hiddenSize)};
    if (options.mNumSlicesForSplitK > 1) {
      shape.push_back(static_cast<uint64_t>(options.mNumSlicesForSplitK));
      stride.push_back(static_cast<uint64_t>(numTokens * hiddenSize));
    }

    return std::make_tuple(shape, stride);
  }

  // Setup the kernel parameters.
  template <class GemmOptions_>
  static KernelParams setKernelParams(
      GemmOptions_ const& options, void const* ptrA, float const* ptrDqSfsA,
      void const* ptrB, float const* ptrDqSfsB, void* ptrC, void* multimemC,
      float* ptrScaleC, void* ptrPartialSumsForSplitK,
      void* multimemPartialSumsForSplitK, void* ptrTileBars,
      void* multimemTileBars, void* ptrCompletionBars,
      void* multimemCompletionBars, void* ptrSplitKCompletionBars, int rank,
      int tpGrpSize) {
    // Is one-shot all-reduce?
    bool const oneShotAr{options.mAllReduceAlgo == AllReduceAlgo::OneShot};
    // Is two-shot all-reduce?
    bool const twoShotAr{options.mAllReduceAlgo == AllReduceAlgo::TwoShot};
    // Are there peer devices?
    bool const multiDevice{tpGrpSize > 1};

    // Create the return struct.
    KernelParams params;

    // Shape/stride for gmem tensor A.
    auto [shapeA, strideA] = makeTmaShapeStrideAb(options, MatrixType::MatrixA);
    // Build tma descriptor for A.
    params.tmaA = gemm::buildNdTmaDescriptor(options.mDtypeElt, shapeA, strideA,
                                             options.mTileM, options.mTileK,
                                             const_cast<void*>(ptrA));

    // Shape/stride for gmem tensor B.
    auto [shapeB, strideB] = makeTmaShapeStrideAb(options, MatrixType::MatrixB);
    // Build tma descriptor for B.
    params.tmaB = gemm::buildNdTmaDescriptor(options.mDtypeElt, shapeB, strideB,
                                             options.mTileN, options.mTileK,
                                             const_cast<void*>(ptrB));

    if (options.mUseTmaStore) {
      // Shape/stride for gmem tensor C.
      auto [shapeC, strideC] = makeTmaShapeStrideC(options);

      // Swap M and N tiles for the M-major epilogue.
      auto outputTileM = options.mTransposeMmaOutput ? options.mEpilogueTileN
                                                     : options.mEpilogueTileM;
      auto outputTileN = options.mTransposeMmaOutput ? options.mEpilogueTileM
                                                     : options.mEpilogueTileN;

      // One-shot performs TMA reduction on multicast mapping of the output
      // buffer directly. Two-shot performs TMA store on unicast mapping of the
      // output buffer. The reduction happens in the next phase.
      void* ptrTmaC{oneShotAr && multiDevice ? multimemC : ptrC};
      auto dtypeC{options.mDtypeC};
      if (options.mNumSlicesForSplitK > 1) {
        ptrTmaC = oneShotAr && multiDevice ? multimemPartialSumsForSplitK
                                           : ptrPartialSumsForSplitK;
      }
      // Regarless of output dtype, both two-shot all-reduce and split-K
      // reduction store partial accumulation results to global memory in
      // float32 precision.
      if ((twoShotAr && multiDevice) || options.mNumSlicesForSplitK > 1) {
        dtypeC = options.mDtypeAcc;
      }

      // Build tma descriptor for C.
      params.tmaC =
          gemm::buildNdTmaDescriptor(dtypeC, shapeC, strideC, outputTileM,
                                     outputTileN, const_cast<void*>(ptrTmaC));
    }

    // Set the dequantization factors for A and B when DeepSeek FP8 recipe is
    // used.
    params.ptrDqSfsA = ptrDqSfsA;
    params.ptrDqSfsB = ptrDqSfsB;

    // Also set ptrC (it may be used by the NCCL reduction code in
    // "layers/Llama").
    params.ptrC = ptrC;
    params.ptrScaleC = ptrScaleC;

    params.m = options.mM;
    params.n = options.mN;
    params.k = options.mK;

    params.rank = rank;
    params.tpGrpSize = tpGrpSize;

    params.multimemC = multimemC;
    params.ptrPartialSumsForSplitK = ptrPartialSumsForSplitK;
    params.multimemPartialSumsForSplitK = multimemPartialSumsForSplitK;
    params.ptrTileBars = ptrTileBars;
    params.multimemTileBars = multimemTileBars;
    params.ptrCompletionBars = ptrCompletionBars;
    params.multimemCompletionBars = multimemCompletionBars;

    params.ptrSplitKCompletionBars = ptrSplitKCompletionBars;

    return params;
  }

  // Setup the kernel parameters.
  template <class GemmOptions_>
  static KernelParams setKernelParams(
      GemmOptions_ const& options, void const* ptrA, void const* ptrB,
      void* ptrC, void* multimemC, float const* ptrScaleC, void* ptrTileBars,
      void* multimemTileBars, void* ptrCompletionBars,
      void* multimemCompletionBars, int rank, int tpGrpSize) {
    return setKernelParams(options, ptrA, nullptr, ptrB, nullptr, ptrC,
                           multimemC, ptrScaleC, ptrTileBars, multimemTileBars,
                           ptrCompletionBars, multimemCompletionBars, rank,
                           tpGrpSize);
  }
};

}  // namespace gemm
