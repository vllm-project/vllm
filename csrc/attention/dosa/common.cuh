#pragma once

#include "torch_utils.h"
#include <torch/csrc/stable/library.h>
#include <cuda.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>
#include <cutlass/float8.h>
#include <deep_gemm/common/cute_tie.cuh>
#include <deep_gemm/common/ring_pipeline.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/mma/sm100.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tcgen05.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace DOSA_NAMESPACE {
using Tensor = torch::stable::Tensor;
using DeviceGuard = torch::stable::accelerator::DeviceGuard;
}  // namespace DOSA_NAMESPACE

#define DOSA_OP(name) DOSA_OP_PREFIX name
#define DOSA_CUDA_CHECK(expression)                       \
  do {                                                    \
    cudaError_t dosa_cuda_error = (expression);           \
    STD_TORCH_CHECK(dosa_cuda_error == cudaSuccess,       \
                    cudaGetErrorString(dosa_cuda_error)); \
  } while (false)
