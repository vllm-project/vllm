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
  return (x + y - 1) / y * y;
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

  int sf_m = round_up<int>(numRows, 128);
  int sf_n_unpadded = numCols / CVT_FP4_SF_VEC_SIZE; // 464 / 16 = 29 valid SFs
  int sf_n_int = round_up<int>(sf_n_unpadded, 4) / 4; // round_up(29,4)/4 = 32/4 = 8 blocks
  
  // Debug: Print dimensions (only once from block 0, thread 0)
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("[FP4_QUANT_DEBUG] numRows=%d, numCols=%d, sf_m=%d, sf_n_unpadded=%d, sf_n_int=%d\n",
           numRows, numCols, sf_m, sf_n_unpadded, sf_n_int);
    printf("[FP4_QUANT_DEBUG] Total SFs needed: %d, Blocks allocated: %d (holds %d SFs)\n",
           sf_n_unpadded, sf_n_int, sf_n_int * 4);
  }
  
  // Calculate which uint32 block contains the boundary (padding starts)
  // sf_n_unpadded = 29 valid SFs.
  // They occupy blocks 0-7. Block 7 contains SFs 28-31, where SFs 29-31 are EXTRA COLUMNS (padding).
  int last_valid_block = (sf_n_unpadded - 1) / 4; // e.g., 29-1=28, 28/4=7
  int valid_sf_in_last_block = sf_n_unpadded % 4; // e.g., 29 % 4 = 1 (only SF 28 is valid)

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    printf("[FP4_QUANT_DEBUG] Boundary block: %d, Valid SFs in boundary: %d (padding SFs: %d-%d)\n",
           last_valid_block, valid_sf_in_last_block, sf_n_unpadded, sf_n_int * 4 - 1);
  }

  // Unified loop: Zero out padding for ALL rows (both valid and extra)
  // - Extra ROWS (numRows to sf_m-1): Zero ALL blocks (0 to sf_n_int-1)
  // - Valid ROWS (0 to numRows-1): Zero only the boundary block containing tail padding (extra columns)
  for (int row = blockIdx.x; row < sf_m; row += gridDim.x) {
    if (row >= numRows) { 
      // Extra row (padding row): Zero ALL blocks
      for (int col = threadIdx.x; col < sf_n_int; col += blockDim.x) {
        SFout[row * sf_n_int + col] = 0x00;
        // Debug: Print for first extra row only
        if (row == numRows && col == 0) {
          printf("[FP4_QUANT_DEBUG] Zeroing EXTRA ROW %d, ALL blocks (0-%d)\n", row, sf_n_int - 1);
        }
      }
    } else if (valid_sf_in_last_block != 0) {
      // Valid row: Zero only the boundary block containing tail padding (EXTRA COLUMNS)
      // The data loop below will overwrite the valid SF (e.g., SF 28)
      if (threadIdx.x == 0) {
        SFout[row * sf_n_int + last_valid_block] = 0x00;
        // Debug: Print for first valid row only
        if (row == 0) {
          printf("[FP4_QUANT_DEBUG] Zeroing VALID ROW %d, boundary block %d (contains SFs %d-%d, padding: %d-%d)\n",
                 row, last_valid_block, last_valid_block * 4, last_valid_block * 4 + 3,
                 sf_n_unpadded, last_valid_block * 4 + 3);
        }
      }
    }
  }

  // Get the global scaling factor, which will be applied to the SF.
  // Note SFScale is the same as next GEMM's alpha, which is
  // (448.f / (Alpha_A / 6.f)).
  float const global_scale = SFScale == nullptr ? 1.0f : SFScale[0];

  // Input tensor row/col loops - This writes REAL DATA
  for (int rowIdx = blockIdx.x; rowIdx < numRows; rowIdx += gridDim.x) {
    for (int colIdx = threadIdx.x; colIdx < numCols / CVT_FP4_ELTS_PER_THREAD;
         colIdx += blockDim.x) {
      int64_t inOffset = rowIdx * (numCols / CVT_FP4_ELTS_PER_THREAD) + colIdx;
      PackedVec in_vec = reinterpret_cast<PackedVec const*>(in)[inOffset];
      // Get the output tensor offset.
      // Same as inOffset because 8 elements are packed into one uint32_t.
      int64_t outOffset = inOffset;
      auto& out_pos = out[outOffset];

      auto sf_out =
          cvt_quant_to_fp4_get_sf_out_offset<uint32_t,
                                             CVT_FP4_NUM_THREADS_PER_SF>(
              rowIdx, colIdx, numCols, SFout);

      // This function calculates the scaling factor and writes it to *sf_out
      // For boundary block (e.g., block 7), this OVERWRITES the zero we wrote earlier
      out_pos =
          cvt_warp_fp16_to_fp4<Type, UE8M0_SF>(in_vec, global_scale, sf_out);
      
      // Debug: Print when we write to the boundary region (for row 0 only, to avoid spam)
      // Each colIdx processes CVT_FP4_ELTS_PER_THREAD=8 elements, which creates 1 SF
      // For numCols=464, we have 464/8=58 colIdx iterations per row
      // SF index = colIdx / (CVT_FP4_ELTS_PER_THREAD / CVT_FP4_SF_VEC_SIZE) = colIdx / (8/16) = colIdx * 2
      // Actually, cvt_warp_fp16_to_fp4 is called with groups, let me check the mapping...
      // For simplicity, print when processing the last few column indices
      if (rowIdx == 0 && colIdx >= (numCols / CVT_FP4_ELTS_PER_THREAD) - 2) {
        printf("[FP4_QUANT_DEBUG] Writing REAL DATA for row %d, colIdx %d (processes elements %d-%d)\n",
               rowIdx, colIdx, colIdx * CVT_FP4_ELTS_PER_THREAD, 
               (colIdx + 1) * CVT_FP4_ELTS_PER_THREAD - 1);
      }
    }
  }
}

template <typename T>
void invokeFP4Quantization(int m, int n, T const* input, float const* SFScale,
                           int64_t* output, int32_t* SFOuput, bool useUE8M0,
                           int multiProcessorCount, cudaStream_t stream) {
  // Grid, Block size.
  // Each thread converts 8 values.
  dim3 block(std::min(int(n / ELTS_PER_THREAD), 512));
  // Get number of blocks per SM
  int const numBlocksPerSM =
      vllm_runtime_blocks_per_sm(static_cast<int>(block.x));
  dim3 grid(std::min(int(m), multiProcessorCount * numBlocksPerSM));

  // Launch the cvt kernel.
  if (useUE8M0) {
    cvt_fp16_to_fp4<T, true><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOuput));
  } else {
    cvt_fp16_to_fp4<T, false><<<grid, block, 0, stream>>>(
        m, n, input, SFScale, reinterpret_cast<uint32_t*>(output),
        reinterpret_cast<uint32_t*>(SFOuput));
  }
}

// Instantiate the function.
template void invokeFP4Quantization(int m, int n, half const* input,
                                    float const* SFScale, int64_t* output,
                                    int32_t* SFOuput, bool useUE8M0,
                                    int multiProcessorCount,
                                    cudaStream_t stream);

template void invokeFP4Quantization(int m, int n, __nv_bfloat16 const* input,
                                    float const* SFScale, int64_t* output,
                                    int32_t* SFOuput, bool useUE8M0,
                                    int multiProcessorCount,
                                    cudaStream_t stream);

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

  // We don't support e8m0 scales at this moment.
  bool useUE8M0 = false;

  VLLM_DISPATCH_HALF_TYPES(input.scalar_type(), "nvfp4_quant_kernel", [&] {
    using cuda_type = vllm::CUDATypeConverter<scalar_t>::Type;
    auto input_ptr = static_cast<cuda_type const*>(input.data_ptr());
    vllm::invokeFP4Quantization(m, n, input_ptr, input_sf_ptr, output_ptr,
                                sf_out, useUE8M0, multiProcessorCount, stream);
  });
}
