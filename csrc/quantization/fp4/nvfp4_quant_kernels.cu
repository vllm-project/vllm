/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <torch/all.h>

#include <cuda_runtime_api.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#include <cuda_fp8.h>
#include "dispatch_utils.h"

#include "cuda_utils.h"
#include "launch_bounds_utils.h"
#include "nvfp4_utils.cuh"

namespace vllm {

template <typename Int>
__host__ __device__ inline Int round_up(Int x, Int y) {
  static_assert(std::is_integral_v<Int>,
                "round_up argument must be integral type");
  return ((x + y - 1) / y) * y;
}

// Compute effective rows for grid configuration with swizzled SF layouts.
inline int computeEffectiveRows(int m) {
  constexpr int ROW_TILE = 128;
  return round_up(m, ROW_TILE);
}

// Use UE4M3 by default.
template <class Type, bool UE8M0_SF = false>
__global__ void __launch_bounds__(512, VLLM_BLOCKS_PER_SM(512))
    cvt_fp16_to_fp4(int32_t numRows, int32_t numCols, Type const* in,
                    float const* SFScale, uint32_t* out, uint32_t* SFout) {
  using PackedVec = PackedVec<Type>;
  static constexpr int CVT_FP4_NUM_THREADS_PER_SF =
      (CVT_FP4_SF_VEC_SIZE / CVT_FP4_ELTS_PER_THREAD);
  static_assert(sizeof(PackedVec) == sizeof(Type) * CVT_FP4_ELTS_PER_THREAD,
                "Vec size is not matched.");

  // Precompute SF layout parameter (constant for entire kernel).
  int32_t const numKTiles = (numCols + 63) / 64;

  int sf_m = round_up<int>(numRows, 128);
  int sf_n_unpadded = numCols / CVT_FP4_SF_VEC_SIZE;
  int sf_n_int = round_up<int>(sf_n_unpadded, 4) / 4;
  int num_padded_cols = sf_n_int * 4 * CVT_FP4_SF_VEC_SIZE;

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const global_scale = SFScale == nullptr ? 1.0f : SFScale[0];

  // Iterate over all rows and cols including padded ones -
  //  ensures we visit every single scale factor address to initialize it.
  for (int rowIdx = blockIdx.x; rowIdx < sf_m; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x;
         colIdx < num_padded_cols / CVT_FP4_ELTS_PER_THREAD;
         colIdx += blockDim.x) {
      int elem_idx = colIdx * CVT_FP4_ELTS_PER_THREAD;

      PackedVec in_vec;
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;

      // If we are outside valid rows OR outside valid columns -> Use Zeros
      if (rowIdx >= numRows || elem_idx >= numCols) {
        memset(&in_vec, 0, sizeof(PackedVec));

      } else {
        // Valid Region: Load actual data
        in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      }

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numKTiles, SFout);

      auto out_val =
          cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, global_scale, sf_out);

      // We do NOT write output for padding because the 'out' tensor is not
      // padded.
      if (rowIdx < numRows && elem_idx < numCols) {
        // Same as inOffset because 8 elements are packed into one uint32_t.
        out[inOffset] = out_val;
      }
    }
  }
}

}  // namespace vllm

void scaled_fp4_quant_sm1xxa(torch::Tensor const& output,
                             torch::Tensor const& input,
                             torch::Tensor const& output_sf,
                             torch::Tensor const& input_sf) {
  int32_t m = input.size(0);
  int32_t n = input.size(1);

  TORCH_CHECK(n % 16 == 0, "The N dimension must be multiple of 16.");
  TORCH_CHECK(input.scalar_type() == at::ScalarType::Half ||
                  input.scalar_type() == at::ScalarType::BFloat16,
              "Unsupported input data type for quantize_to_fp4.");

  int multiProcessorCount =
      get_device_attribute(cudaDevAttrMultiProcessorCount, -1);

  auto input_sf_ptr = static_cast<float const*>(input_sf.data_ptr());
  auto sf_out = static_cast<int32_t*>(output_sf.data_ptr());
  auto output_ptr = static_cast<int64_t*>(output.data_ptr());
  const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
  auto stream = at::cuda::getCurrentCUDAStream(input.get_device());

  // Grid, Block size. Each thread converts 8 values.
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  int effectiveRows = vllm::computeEffectiveRows(m);
  dim3 grid(std::min(effectiveRows, multiProcessorCount * numBlocksPerSM));

  VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "nvfp4_quant_kernel", [&] {
    using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
    auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
    // NOTE: We don't support e8m0 scales at this moment.
    vllm::cvt_fp16_to_fp4<cuda_type, false><<<grid, block, 0, stream>>>(
        m, n, input_ptr, input_sf_ptr, reinterpret_cast<uint32_t*>(output_ptr),
        reinterpret_cast<uint32_t*>(sf_out));
  });
}